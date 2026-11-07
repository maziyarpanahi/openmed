/**
 * Crash-safe, resumable, integrity-checked generation cache for signed
 * portable bundles in the browser (contract: docs/models/bundle-cache.md).
 *
 * Storage goes through a small key/value adapter: the Origin Private File
 * System when available, IndexedDB otherwise, or memory for tests. Payload
 * bytes are stored as one entry per verified chunk, so a crash can never leave
 * a torn chunk and resume never rewrites earlier bytes. Every record write is a
 * single atomic replace (an OPFS writable stream commits on close; an
 * IndexedDB put is one transaction). Cross-tab exclusion uses the Web Locks
 * API. The cache never opens a network connection and stores only opaque ids,
 * digests, sizes, and reason codes.
 */

import {
  canonicalJson,
  verifyPortableBundle,
  type PortableBundleContext,
  type PortableBundleReason,
  type PortableBundleTrustPolicy,
} from "./portable-bundle";

export const BUNDLE_CACHE_CONTRACT_VERSION = 1;
export const BUNDLE_CACHE_SNAPSHOT_SCHEMA = "openmed.bundle_cache.snapshot";

export const BUNDLE_GENERATION_STATES = [
  "absent",
  "staging",
  "resumable",
  "verifying",
  "ready",
  "active",
  "superseded",
  "evictable",
  "quarantined",
  "evicted",
] as const;
export type BundleGenerationState = (typeof BUNDLE_GENERATION_STATES)[number];

export const BUNDLE_PROTECTIONS = ["active", "last_green", "rollback", "pinned", "leased"] as const;
export type BundleProtection = (typeof BUNDLE_PROTECTIONS)[number];

export const BUNDLE_CACHE_REASONS = [
  "bundle_rejected",
  "lock_timeout",
  "quota_exceeded",
  "storage_pressure",
  "transfer_interrupted",
  "cancelled",
  "range_mismatch",
  "verification_failed",
  "storage_io",
  "generation_not_ready",
  "generation_corrupt",
  "no_active_generation",
  "rollback_unavailable",
  "lease_unknown",
] as const;
export type BundleCacheReason = (typeof BUNDLE_CACHE_REASONS)[number];

export const BUNDLE_RESUME_REASONS = [
  "source_changed",
  "trust_policy_changed",
  "chunk_size_changed",
  "partial_digest_mismatch",
  "range_not_honored",
] as const;
export type BundleResumeReason = (typeof BUNDLE_RESUME_REASONS)[number];

export const BUNDLE_RECOVERY_ACTIONS = [
  "stale_lock_cleared",
  "trash_purged",
  "quarantine_purged",
  "staging_orphaned",
  "staging_expired",
  "staging_resumable",
  "generation_quarantined",
  "pointer_repaired",
  "lease_expired",
] as const;
export type BundleRecoveryAction = (typeof BUNDLE_RECOVERY_ACTIONS)[number];

/** Error carrying only controlled reason codes. */
export class BundleCacheError extends Error {
  readonly reason: BundleCacheReason;
  readonly bundleReason: PortableBundleReason | null;

  constructor(reason: BundleCacheReason, bundleReason: PortableBundleReason | null = null) {
    super(reason);
    this.name = "BundleCacheError";
    this.reason = reason;
    this.bundleReason = bundleReason;
  }
}

/** Thrown only by test checkpoints to simulate a tab or process dying. */
export class BundleCacheCrash extends Error {
  readonly checkpoint: string;

  constructor(checkpoint: string) {
    super(checkpoint);
    this.name = "BundleCacheCrash";
    this.checkpoint = checkpoint;
  }
}

// ---------------------------------------------------------------------------
// Storage adapters.

/** Minimal key/value storage the cache needs. Keys are `/`-separated names. */
export interface BundleCacheStorage {
  readonly kind: "opfs" | "indexeddb" | "memory";
  get(key: string): Promise<Uint8Array | null>;
  /** Atomically replaces the value of `key`. */
  put(key: string, value: Uint8Array): Promise<void>;
  delete(key: string): Promise<void>;
  /** Every key below `prefix/`, sorted. */
  list(prefix: string): Promise<string[]>;
  /** Removes every key below `prefix/`. */
  deletePrefix(prefix: string): Promise<void>;
  size(key: string): Promise<number | null>;
  /** Optional zero-copy access for loaders (OPFS returns a `File`). */
  blob?(key: string): Promise<Blob | null>;
}

function keyParts(key: string): string[] {
  const parts = key.split("/");
  if (parts.some((part) => part.length === 0 || part === "." || part === "..")) {
    throw new BundleCacheError("storage_io");
  }
  return parts;
}

/** In-memory storage for tests and ephemeral pages. */
export class MemoryBundleCacheStorage implements BundleCacheStorage {
  readonly kind = "memory" as const;
  private readonly entries = new Map<string, Uint8Array>();

  async get(key: string): Promise<Uint8Array | null> {
    const value = this.entries.get(key);
    return value ? value.slice() : null;
  }

  async put(key: string, value: Uint8Array): Promise<void> {
    keyParts(key);
    this.entries.set(key, value.slice());
  }

  async delete(key: string): Promise<void> {
    this.entries.delete(key);
  }

  async list(prefix: string): Promise<string[]> {
    const start = `${prefix}/`;
    return [...this.entries.keys()].filter((key) => key.startsWith(start)).sort(compareText);
  }

  async deletePrefix(prefix: string): Promise<void> {
    for (const key of await this.list(prefix)) this.entries.delete(key);
  }

  async size(key: string): Promise<number | null> {
    return this.entries.get(key)?.length ?? null;
  }
}

/** Subset of `FileSystemDirectoryHandle` the OPFS adapter uses. */
export interface OpfsDirectoryLike {
  readonly kind: "directory";
  getDirectoryHandle(name: string, options?: { create?: boolean }): Promise<OpfsDirectoryLike>;
  getFileHandle(name: string, options?: { create?: boolean }): Promise<OpfsFileLike>;
  removeEntry(name: string, options?: { recursive?: boolean }): Promise<void>;
  entries(): AsyncIterable<[string, OpfsDirectoryLike | OpfsFileLike]>;
}

/** Subset of `FileSystemFileHandle` the OPFS adapter uses. */
export interface OpfsFileLike {
  readonly kind: "file";
  getFile(): Promise<Blob>;
  createWritable(): Promise<{ write(data: Uint8Array): Promise<void>; close(): Promise<void>; abort?(): Promise<void> }>;
}

function isNotFound(error: unknown): boolean {
  const name = (error as { name?: unknown } | null)?.name;
  return name === "NotFoundError" || name === "TypeMismatchError";
}

function isQuotaError(error: unknown): boolean {
  return (error as { name?: unknown } | null)?.name === "QuotaExceededError";
}

/** Origin Private File System storage; each value is one file. */
export class OpfsBundleCacheStorage implements BundleCacheStorage {
  readonly kind = "opfs" as const;

  constructor(private readonly root: OpfsDirectoryLike) {}

  private async directory(parts: string[], create: boolean): Promise<OpfsDirectoryLike | null> {
    let current = this.root;
    for (const part of parts) {
      try {
        current = await current.getDirectoryHandle(part, { create });
      } catch (error) {
        if (isNotFound(error)) return null;
        throw error;
      }
    }
    return current;
  }

  private async file(key: string): Promise<OpfsFileLike | null> {
    const parts = keyParts(key);
    const name = parts.pop() as string;
    const parent = await this.directory(parts, false);
    if (!parent) return null;
    try {
      return await parent.getFileHandle(name);
    } catch (error) {
      if (isNotFound(error)) return null;
      throw error;
    }
  }

  async get(key: string): Promise<Uint8Array | null> {
    const blob = await this.blob(key);
    return blob ? new Uint8Array(await blob.arrayBuffer()) : null;
  }

  async blob(key: string): Promise<Blob | null> {
    const handle = await this.file(key);
    return handle ? handle.getFile() : null;
  }

  async put(key: string, value: Uint8Array): Promise<void> {
    const parts = keyParts(key);
    const name = parts.pop() as string;
    const parent = (await this.directory(parts, true)) as OpfsDirectoryLike;
    const handle = await parent.getFileHandle(name, { create: true });
    const writable = await handle.createWritable();
    try {
      await writable.write(value);
      await writable.close();
    } catch (error) {
      await writable.abort?.().catch(() => undefined);
      throw error;
    }
  }

  async delete(key: string): Promise<void> {
    const parts = keyParts(key);
    const name = parts.pop() as string;
    const parent = await this.directory(parts, false);
    if (!parent) return;
    try {
      await parent.removeEntry(name, { recursive: true });
    } catch (error) {
      if (!isNotFound(error)) throw error;
    }
  }

  async list(prefix: string): Promise<string[]> {
    const start = await this.directory(keyParts(prefix), false);
    if (!start) return [];
    const keys: string[] = [];
    const walk = async (directory: OpfsDirectoryLike, path: string): Promise<void> => {
      for await (const [name, handle] of directory.entries()) {
        const key = `${path}/${name}`;
        if (handle.kind === "directory") await walk(handle, key);
        else keys.push(key);
      }
    };
    await walk(start, prefix);
    return keys.sort(compareText);
  }

  async deletePrefix(prefix: string): Promise<void> {
    await this.delete(prefix);
  }

  async size(key: string): Promise<number | null> {
    const blob = await this.blob(key);
    return blob ? blob.size : null;
  }
}

/** Subset of `IDBFactory` the IndexedDB adapter uses. */
export interface IdbFactoryLike {
  open(name: string, version?: number): IdbOpenRequestLike;
}
interface IdbRequestLike<T> {
  result: T;
  error: unknown;
  onsuccess: ((event: unknown) => void) | null;
  onerror: ((event: unknown) => void) | null;
}
interface IdbOpenRequestLike extends IdbRequestLike<IdbDatabaseLike> {
  onupgradeneeded: ((event: unknown) => void) | null;
}
interface IdbDatabaseLike {
  objectStoreNames: { contains(name: string): boolean };
  createObjectStore(name: string): unknown;
  transaction(store: string, mode: "readonly" | "readwrite"): IdbTransactionLike;
}
interface IdbTransactionLike {
  objectStore(name: string): IdbStoreLike;
  oncomplete: ((event: unknown) => void) | null;
  onerror: ((event: unknown) => void) | null;
  onabort: ((event: unknown) => void) | null;
  error: unknown;
}
interface IdbStoreLike {
  get(key: string): IdbRequestLike<unknown>;
  put(value: unknown, key: string): IdbRequestLike<unknown>;
  delete(key: unknown): IdbRequestLike<unknown>;
  getAllKeys(range: unknown): IdbRequestLike<unknown[]>;
}
/** Subset of `IDBKeyRange` the IndexedDB adapter uses. */
export interface IdbKeyRangeLike {
  bound(lower: string, upper: string): unknown;
}

const IDB_STORE = "entries";

/** IndexedDB storage: one object store, one value per key, one transaction per write. */
export class IndexedDbBundleCacheStorage implements BundleCacheStorage {
  readonly kind = "indexeddb" as const;

  private constructor(
    private readonly db: IdbDatabaseLike,
    private readonly keyRange: IdbKeyRangeLike,
  ) {}

  /** Opens (or creates) the cache database. */
  static async open(
    options: { name?: string; factory?: IdbFactoryLike; keyRange?: IdbKeyRangeLike } = {},
  ): Promise<IndexedDbBundleCacheStorage> {
    const scope = globalThis as unknown as { indexedDB?: IdbFactoryLike; IDBKeyRange?: IdbKeyRangeLike };
    const factory = options.factory ?? scope.indexedDB;
    const keyRange = options.keyRange ?? scope.IDBKeyRange;
    if (!factory || !keyRange) throw new BundleCacheError("storage_io");
    const request = factory.open(options.name ?? "openmed-bundle-cache", 1);
    const db = await new Promise<IdbDatabaseLike>((resolve, reject) => {
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains(IDB_STORE)) request.result.createObjectStore(IDB_STORE);
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    return new IndexedDbBundleCacheStorage(db, keyRange);
  }

  private run<T>(mode: "readonly" | "readwrite", body: (store: IdbStoreLike) => IdbRequestLike<T>): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const transaction = this.db.transaction(IDB_STORE, mode);
      const request = body(transaction.objectStore(IDB_STORE));
      let result: T;
      request.onsuccess = () => {
        result = request.result;
      };
      request.onerror = () => reject(request.error);
      transaction.oncomplete = () => resolve(result);
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
  }

  private range(prefix: string): unknown {
    return this.keyRange.bound(`${prefix}/`, `${prefix}/￿`);
  }

  async get(key: string): Promise<Uint8Array | null> {
    const value = await this.run("readonly", (store) => store.get(key));
    if (value instanceof Uint8Array) return value.slice();
    if (value instanceof ArrayBuffer) return new Uint8Array(value.slice(0));
    return null;
  }

  async put(key: string, value: Uint8Array): Promise<void> {
    keyParts(key);
    await this.run("readwrite", (store) => store.put(value.slice(), key));
  }

  async delete(key: string): Promise<void> {
    await this.run("readwrite", (store) => store.delete(key));
  }

  async list(prefix: string): Promise<string[]> {
    const keys = await this.run("readonly", (store) => store.getAllKeys(this.range(prefix)));
    return keys.filter((key): key is string => typeof key === "string").sort(compareText);
  }

  async deletePrefix(prefix: string): Promise<void> {
    await this.run("readwrite", (store) => store.delete(this.range(prefix)));
  }

  async size(key: string): Promise<number | null> {
    return (await this.get(key))?.length ?? null;
  }
}

/**
 * Opens the best available persistent storage: OPFS, then IndexedDB.
 * Fails with `storage_io` when neither exists. It never requests persistent
 * storage; call `navigator.storage.persist()` from a user gesture if needed.
 */
export async function openBundleCacheStorage(
  options: { directoryName?: string; databaseName?: string } = {},
): Promise<BundleCacheStorage> {
  const scope = globalThis as unknown as {
    navigator?: { storage?: { getDirectory?: () => Promise<OpfsDirectoryLike> } };
  };
  const getDirectory = scope.navigator?.storage?.getDirectory;
  if (getDirectory) {
    try {
      const root = await getDirectory.call(scope.navigator?.storage);
      const directory = await root.getDirectoryHandle(options.directoryName ?? "openmed-bundle-cache", {
        create: true,
      });
      return new OpfsBundleCacheStorage(directory);
    } catch {
      // Fall back to IndexedDB (for example in private browsing modes without OPFS).
    }
  }
  return IndexedDbBundleCacheStorage.open(options.databaseName ? { name: options.databaseName } : {});
}

// ---------------------------------------------------------------------------
// Range sources.

/** A byte range returned by a range source. */
export interface BundleRangeResponse {
  /** Offset of the first byte `read` returns; `0` when the source ignored the range. */
  readonly offset: number;
  /** Returns up to `maxBytes` bytes, or `null` at end of file. */
  read(maxBytes: number): Promise<Uint8Array | null>;
  close?(): Promise<void> | void;
}

/** Supplies bundle files by byte range. Implementations perform any network access. */
export interface BundleRangeSource {
  /** Opaque identity (for example a URL plus an ETag). Only its digest is stored. */
  readonly identity: string;
  open(path: string, offset: number): Promise<BundleRangeResponse>;
}

/**
 * Offline range source over files the page already holds (for example from a
 * file picker or an extracted archive). Performs no network access.
 */
export function createBlobRangeSource(
  files: ReadonlyMap<string, Blob | Uint8Array>,
  identity = "local-files",
): BundleRangeSource {
  return {
    identity,
    async open(path: string, offset: number): Promise<BundleRangeResponse> {
      const value = files.get(path);
      if (value === undefined) throw new BundleCacheError("transfer_interrupted");
      const blob = value instanceof Uint8Array ? new Blob([value as BlobPart]) : value;
      let position = Math.min(offset, blob.size);
      return {
        offset: position,
        async read(maxBytes: number): Promise<Uint8Array | null> {
          if (position >= blob.size) return null;
          const end = Math.min(blob.size, position + maxBytes);
          const bytes = new Uint8Array(await blob.slice(position, end).arrayBuffer());
          position = end;
          return bytes;
        },
      };
    },
  };
}

// ---------------------------------------------------------------------------
// Public types.

export interface BundleCacheOptions {
  readonly storage: BundleCacheStorage;
  /** Byte budget over the ready generations of every bundle. */
  readonly quotaBytes?: number;
  /** Resume checkpoint size; every committed chunk records its SHA-256. */
  readonly chunkBytes?: number;
  readonly lockTimeoutMs?: number;
  /** How long a lease protects its generation once its owner is gone. */
  readonly leaseTtlMs?: number;
  /** Staging older than this since its last checkpoint is removed by recovery. */
  readonly stagingMaxAgeMs?: number;
  /** Clock in epoch milliseconds. */
  readonly now?: () => number;
  /** Free-storage probe; defaults to `navigator.storage.estimate()`. */
  readonly availableBytes?: () => Promise<number>;
  /** Opaque id of this cache instance in lock and lease records. */
  readonly ownerId?: string;
  /** Web Locks manager; `null` forces the in-page mutex. Defaults to `navigator.locks`. */
  readonly locks?: BundleLockManager | null;
}

/** Subset of the Web Locks `LockManager`. */
export interface BundleLockManager {
  request<T>(
    name: string,
    options: { signal?: AbortSignal; ifAvailable?: boolean },
    callback: (lock: unknown) => Promise<T>,
  ): Promise<T>;
}

export interface BundleAcquireRequest {
  readonly manifest: Uint8Array;
  readonly envelope: Uint8Array;
  readonly source: BundleRangeSource;
  readonly trustPolicy: PortableBundleTrustPolicy;
  readonly context: PortableBundleContext;
  readonly activate?: boolean;
  readonly confirm?: boolean;
  /** Checked after every committed chunk; staged bytes stay resumable. */
  readonly signal?: AbortSignal;
}

export interface BundleAcquireResult {
  readonly ok: boolean;
  readonly reason: BundleCacheReason | null;
  readonly bundleReason: PortableBundleReason | null;
  readonly bundleKey: string | null;
  readonly generation: string | null;
  readonly state: BundleGenerationState | null;
  readonly fromCache: boolean;
  readonly resumed: boolean;
  readonly resumeReason: BundleResumeReason | null;
  readonly bytesTransferred: number;
  readonly verifiedBytes: number;
  readonly filesVerified: number;
  readonly evicted: readonly string[];
}

export interface BundleGenerationSnapshot {
  readonly generation: string;
  readonly state: BundleGenerationState;
  readonly totalBytes: number;
  readonly verifiedBytes: number;
  readonly protections: readonly BundleProtection[];
}

export interface BundleCacheSnapshot {
  readonly schema: typeof BUNDLE_CACHE_SNAPSHOT_SCHEMA;
  readonly schemaVersion: 1;
  readonly bundleKey: string;
  readonly active: string | null;
  readonly lastGreen: string | null;
  readonly rollback: string | null;
  readonly generations: readonly BundleGenerationSnapshot[];
  readonly quarantined: readonly { readonly generation: string; readonly reason: string }[];
  readonly usedBytes: number;
}

export interface BundleQuotaDecision {
  readonly quotaBytes: number;
  readonly usedBytes: number;
  readonly additionalBytes: number;
  readonly evicted: readonly string[];
  readonly satisfied: boolean;
}

export interface BundleRecoveryStep {
  readonly action: BundleRecoveryAction;
  readonly bundleKey: string;
  readonly generation: string | null;
}

export interface BundleRecoveryReport {
  readonly actions: readonly BundleRecoveryStep[];
}

export type BundleOpenVerification = "metadata" | "full";

/** A lease on a verified generation; it is never evicted while held. */
export interface BundleLease {
  readonly id: string;
  readonly bundleKey: string;
  readonly generation: string;
  readonly filesChecked: number;
  /** Reads one declared file of the leased generation. */
  readFile(path: string): Promise<Uint8Array>;
  /** Returns one declared file as a `Blob` without copying when the storage allows it. */
  readBlob(path: string): Promise<Blob>;
  release(): Promise<void>;
}

// ---------------------------------------------------------------------------
// SHA-256 (incremental; WebCrypto has no streaming digest).

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
  0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
  0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
  0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
  0xc67178f2,
]);

/** Incremental SHA-256 over byte chunks. */
export class Sha256 {
  private readonly state = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  private readonly block = new Uint8Array(64);
  private readonly words = new Uint32Array(64);
  private blockLength = 0;
  private totalLength = 0;

  update(data: Uint8Array): this {
    let index = 0;
    this.totalLength += data.length;
    while (index < data.length) {
      const take = Math.min(64 - this.blockLength, data.length - index);
      this.block.set(data.subarray(index, index + take), this.blockLength);
      this.blockLength += take;
      index += take;
      if (this.blockLength === 64) {
        this.compress();
        this.blockLength = 0;
      }
    }
    return this;
  }

  /** Returns `sha256:<hex>`. */
  digest(): string {
    const bits = this.totalLength * 8;
    this.block[this.blockLength++] = 0x80;
    if (this.blockLength > 56) {
      this.block.fill(0, this.blockLength);
      this.compress();
      this.blockLength = 0;
    }
    this.block.fill(0, this.blockLength, 56);
    const high = Math.floor(bits / 0x100000000);
    const view = new DataView(this.block.buffer);
    view.setUint32(56, high >>> 0);
    view.setUint32(60, bits >>> 0);
    this.compress();
    let hex = "";
    for (const word of this.state) hex += word.toString(16).padStart(8, "0");
    return `sha256:${hex}`;
  }

  private compress(): void {
    const w = this.words;
    const block = this.block;
    for (let i = 0; i < 16; i += 1) {
      w[i] = ((block[i * 4]! << 24) | (block[i * 4 + 1]! << 16) | (block[i * 4 + 2]! << 8) | block[i * 4 + 3]!) >>> 0;
    }
    for (let i = 16; i < 64; i += 1) {
      const a = w[i - 15]!;
      const b = w[i - 2]!;
      const s0 = ((a >>> 7) | (a << 25)) ^ ((a >>> 18) | (a << 14)) ^ (a >>> 3);
      const s1 = ((b >>> 17) | (b << 15)) ^ ((b >>> 19) | (b << 13)) ^ (b >>> 10);
      w[i] = (w[i - 16]! + s0 + w[i - 7]! + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = this.state as unknown as number[];
    for (let i = 0; i < 64; i += 1) {
      const s1 = ((e! >>> 6) | (e! << 26)) ^ ((e! >>> 11) | (e! << 21)) ^ ((e! >>> 25) | (e! << 7));
      const choice = (e! & f!) ^ (~e! & g!);
      const t1 = (h! + s1 + choice + K[i]! + w[i]!) >>> 0;
      const s0 = ((a! >>> 2) | (a! << 30)) ^ ((a! >>> 13) | (a! << 19)) ^ ((a! >>> 22) | (a! << 10));
      const majority = (a! & b!) ^ (a! & c!) ^ (b! & c!);
      const t2 = (s0 + majority) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d! + t1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (t1 + t2) >>> 0;
    }
    const state = this.state;
    state[0] = (state[0]! + a!) >>> 0;
    state[1] = (state[1]! + b!) >>> 0;
    state[2] = (state[2]! + c!) >>> 0;
    state[3] = (state[3]! + d!) >>> 0;
    state[4] = (state[4]! + e!) >>> 0;
    state[5] = (state[5]! + f!) >>> 0;
    state[6] = (state[6]! + g!) >>> 0;
    state[7] = (state[7]! + h!) >>> 0;
  }
}

function sha256Text(data: Uint8Array): string {
  return new Sha256().update(data).digest();
}

// ---------------------------------------------------------------------------
// Identities.

const encoder = new TextEncoder();
const decoder = new TextDecoder();
const GENERATION = /^g-[0-9a-f]{32}$/;
const BUNDLE = /^b-[0-9a-f]{32}$/;

function compareText(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

/** Opaque storage key of a bundle id. */
export function bundleCacheKey(bundleId: string): string {
  return `b-${sha256Text(encoder.encode(bundleId)).slice(7, 39)}`;
}

/** Opaque generation id of a manifest content identity (`sha256:<hex>`). */
export function bundleGenerationId(contentId: string): string {
  return `g-${contentId.replace(/^sha256:/, "").slice(0, 32)}`;
}

function base64Url(bytes: Uint8Array): string {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
  let output = "";
  for (let index = 0; index < bytes.length; index += 3) {
    const a = bytes[index]!;
    const b = bytes[index + 1] ?? 0;
    const c = bytes[index + 2] ?? 0;
    const triple = (a << 16) | (b << 8) | c;
    output += alphabet[(triple >> 18) & 63]! + alphabet[(triple >> 12) & 63]!;
    if (index + 1 < bytes.length) output += alphabet[(triple >> 6) & 63]!;
    if (index + 2 < bytes.length) output += alphabet[triple & 63]!;
  }
  return output;
}

/** Binding digest of a trust policy, recorded with resumable bytes. */
export function bundleTrustDigest(policy: PortableBundleTrustPolicy): string {
  const keys = [...policy.keys]
    .sort((left, right) => compareText(left.kid, right.kid))
    .map((key) => ({
      kid: key.kid,
      not_after: key.notAfter,
      not_before: key.notBefore,
      revoked: key.revoked,
      x: base64Url(key.publicKey),
    }));
  const value = {
    keys,
    revoked_content_ids: [...policy.revokedContentIds].sort(compareText),
    threshold: policy.threshold,
  };
  return sha256Text(encoder.encode(canonicalJson(value)));
}

// ---------------------------------------------------------------------------
// Locks.

class LocalMutex {
  private readonly tails = new Map<string, Promise<void>>();

  async acquire(name: string, timeoutMs: number): Promise<(() => void) | null> {
    const previous = this.tails.get(name) ?? Promise.resolve();
    let release!: () => void;
    const current = new Promise<void>((resolve) => {
      release = resolve;
    });
    const tail = previous.then(() => current);
    this.tails.set(name, tail);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const acquired = await Promise.race([
      previous.then(() => true),
      new Promise<boolean>((resolve) => {
        timer = setTimeout(() => resolve(false), timeoutMs);
      }),
    ]);
    if (timer !== undefined) clearTimeout(timer);
    const done = () => {
      release();
      if (this.tails.get(name) === tail) this.tails.delete(name);
    };
    if (!acquired) {
      void previous.then(done);
      return null;
    }
    return done;
  }
}

const localMutexes = new WeakMap<BundleCacheStorage, LocalMutex>();

function localMutex(storage: BundleCacheStorage): LocalMutex {
  let mutex = localMutexes.get(storage);
  if (!mutex) {
    mutex = new LocalMutex();
    localMutexes.set(storage, mutex);
  }
  return mutex;
}

// ---------------------------------------------------------------------------
// Records.

interface Pointer {
  active: string | null;
  lastGreen: string | null;
  rollback: string | null;
}

interface GenerationRecord {
  contentId: string;
  sequence: number;
  totalBytes: number;
  fileCount: number;
  chunkBytes: number;
  activated: boolean;
  pinned: boolean;
  verifiedAt: number;
  lastAccessed: number;
}

interface StagingRecord {
  contentId: string;
  sourceDigest: string;
  trustDigest: string;
  chunkBytes: number;
  phase: "staging" | "verifying";
  createdAt: number;
  updatedAt: number;
  files: string[][];
}

interface DeclaredFile {
  path: string;
  size: number;
  sha256: string;
}

interface ManifestHeader {
  bundleId: string;
  sequence: number;
  files: DeclaredFile[];
}

function manifestHeader(manifest: Uint8Array): ManifestHeader | null {
  try {
    const value = JSON.parse(decoder.decode(manifest)) as {
      bundle: { id: string; sequence: number };
      files: { path: string; size_bytes: number; sha256: string }[];
    };
    return {
      bundleId: value.bundle.id,
      sequence: value.bundle.sequence,
      files: value.files.map((file) => ({ path: file.path, size: file.size_bytes, sha256: file.sha256 })),
    };
  } catch {
    return null;
  }
}

function ref(value: unknown): string | null {
  return typeof value === "string" && GENERATION.test(value) ? value : null;
}

// ---------------------------------------------------------------------------
// The cache.

/**
 * Crash-safe generation cache for signed portable bundles. All operations are
 * asynchronous and serialized per bundle by a Web Lock (or an in-page mutex).
 */
export class BundleGenerationCache {
  readonly contractVersion = BUNDLE_CACHE_CONTRACT_VERSION;
  /** Test seam: called at each named crash checkpoint. */
  checkpointHook: ((name: string) => void) | null = null;

  private readonly storage: BundleCacheStorage;
  private readonly quotaBytes: number;
  private readonly chunkBytes: number;
  private readonly lockTimeoutMs: number;
  private readonly leaseTtlMs: number;
  private readonly stagingMaxAgeMs: number;
  private readonly clock: () => number;
  private readonly availableBytes: () => Promise<number>;
  private readonly ownerId: string;
  private readonly locks: BundleLockManager | null;
  private readonly heldLeases = new Set<string>();

  constructor(options: BundleCacheOptions) {
    this.storage = options.storage;
    this.quotaBytes = Math.max(0, options.quotaBytes ?? 2 * 1024 * 1024 * 1024);
    this.chunkBytes = Math.max(1, Math.floor(options.chunkBytes ?? 4 * 1024 * 1024));
    this.lockTimeoutMs = Math.max(0, options.lockTimeoutMs ?? 30_000);
    this.leaseTtlMs = Math.max(1, options.leaseTtlMs ?? 600_000);
    this.stagingMaxAgeMs = Math.max(1, options.stagingMaxAgeMs ?? 604_800_000);
    this.clock = options.now ?? (() => Date.now());
    this.availableBytes = options.availableBytes ?? defaultAvailableBytes;
    this.ownerId = options.ownerId ?? randomId();
    const scope = globalThis as unknown as { navigator?: { locks?: BundleLockManager } };
    this.locks = options.locks === undefined ? (scope.navigator?.locks ?? null) : options.locks;
  }

  toString(): string {
    return `BundleGenerationCache(${this.storage.kind})`;
  }

  // -- public API ----------------------------------------------------------

  /** Verifies, stages, commits, and optionally activates one signed bundle. */
  async acquire(request: BundleAcquireRequest): Promise<BundleAcquireResult> {
    const verification = await verifyPortableBundle({
      manifest: request.manifest,
      envelope: request.envelope,
      trustPolicy: request.trustPolicy,
      context: request.context,
    });
    const contentId = verification.contentId;
    if (!verification.ok || contentId === null) {
      return failure("bundle_rejected", verification.reason, null, null);
    }
    const header = manifestHeader(request.manifest);
    if (!header) return failure("bundle_rejected", "manifest_malformed", null, null);
    const key = bundleCacheKey(header.bundleId);
    const generation = bundleGenerationId(contentId);
    try {
      return await this.locked(key, () => this.acquireLocked(request, header, key, generation, contentId));
    } catch (error) {
      if (error instanceof BundleCacheCrash) throw error;
      if (error instanceof BundleCacheError) return failure(error.reason, error.bundleReason, key, generation);
      return failure(isQuotaError(error) ? "storage_pressure" : "storage_io", null, key, generation);
    }
  }

  /** Activates a ready generation; with `confirm` it also becomes last green. */
  async activate(bundleId: string, generation: string, options: { confirm?: boolean } = {}): Promise<void> {
    const key = bundleCacheKey(bundleId);
    await this.locked(key, () => this.activateLocked(key, generation, options.confirm ?? true));
  }

  /** Marks the active generation as last green after a probation activation. */
  async confirm(bundleId: string): Promise<void> {
    const key = bundleCacheKey(bundleId);
    await this.locked(key, async () => {
      const pointer = await this.readPointer(key);
      if (!pointer.active) throw new BundleCacheError("no_active_generation");
      await this.writePointer(key, { ...pointer, lastGreen: pointer.active });
    });
  }

  /** Reactivates the retained rollback generation without any transfer. */
  async rollback(bundleId: string, options: { quarantine?: boolean } = {}): Promise<void> {
    const key = bundleCacheKey(bundleId);
    await this.locked(key, async () => {
      const pointer = await this.readPointer(key);
      const target = pointer.rollback;
      if (!target || (await this.metadataCheck(key, target)) !== null) {
        throw new BundleCacheError("rollback_unavailable");
      }
      const abandoned = pointer.active;
      await this.touch(key, target, true);
      const lastGreen = pointer.lastGreen === abandoned ? target : pointer.lastGreen;
      await this.writePointer(key, { active: target, lastGreen, rollback: null });
      if (options.quarantine && abandoned && abandoned !== target) {
        await this.quarantineReady(key, abandoned, "rolled_back");
      }
    });
  }

  /** Pins a ready generation so eviction never removes it. */
  async pin(bundleId: string, generation: string): Promise<void> {
    await this.setPinned(bundleId, generation, true);
  }

  /** Removes a pin. */
  async unpin(bundleId: string, generation: string): Promise<void> {
    await this.setPinned(bundleId, generation, false);
  }

  /** Leases a ready generation (the active one by default) after a metadata check. */
  async lease(bundleId: string, generation: string | null = null): Promise<BundleLease> {
    const key = bundleCacheKey(bundleId);
    return this.locked(key, async () => {
      const target = generation ?? (await this.readPointer(key)).active;
      if (!target) throw new BundleCacheError("no_active_generation");
      return this.createLease(key, target, "metadata");
    });
  }

  /**
   * Opens the active generation and returns a lease on it. A mismatch
   * quarantines the generation, falls the pointer back, and fails with
   * `generation_corrupt`.
   */
  async open(bundleId: string, options: { verification?: BundleOpenVerification } = {}): Promise<BundleLease> {
    const key = bundleCacheKey(bundleId);
    return this.locked(key, async () => {
      const active = (await this.readPointer(key)).active;
      if (!active) throw new BundleCacheError("no_active_generation");
      return this.createLease(key, active, options.verification ?? "metadata");
    });
  }

  /** Releases a lease. */
  async release(lease: BundleLease): Promise<void> {
    await this.locked(lease.bundleKey, async () => {
      const known = this.heldLeases.delete(lease.id);
      const recordKey = `${lease.bundleKey}/lease/${lease.generation}/${lease.id}`;
      const existed = (await this.storage.get(recordKey)) !== null;
      await this.storage.delete(recordKey);
      if (!existed && !known) throw new BundleCacheError("lease_unknown");
    });
  }

  /** Evicts least recently used unprotected generations until `additionalBytes` fit. */
  async evict(additionalBytes = 0): Promise<BundleQuotaDecision> {
    return this.applyQuota(Math.max(0, additionalBytes), null, null);
  }

  /** Idempotent, bounded crash recovery over every bundle; call at startup. */
  async recover(): Promise<BundleRecoveryReport> {
    const actions: BundleRecoveryStep[] = [];
    for (const key of await this.bundleKeys()) {
      try {
        await this.locked(key, (stale) => this.recoverLocked(key, stale, actions));
      } catch (error) {
        if (error instanceof BundleCacheCrash) throw error;
        if (!(error instanceof BundleCacheError && error.reason === "lock_timeout")) throw error;
      }
    }
    return { actions };
  }

  /** Value-free snapshot of one bundle. */
  async snapshot(bundleId: string): Promise<BundleCacheSnapshot> {
    const key = bundleCacheKey(bundleId);
    return this.locked(key, () => this.snapshotLocked(key));
  }

  // -- acquisition ---------------------------------------------------------

  private async acquireLocked(
    request: BundleAcquireRequest,
    header: ManifestHeader,
    key: string,
    generation: string,
    contentId: string,
  ): Promise<BundleAcquireResult> {
    const files = header.files;
    const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
    const activate = request.activate ?? true;
    const confirm = request.confirm ?? true;
    if ((await this.metadataCheck(key, generation)) === null) {
      await this.touch(key, generation, null);
      if (activate) await this.activateLocked(key, generation, confirm);
      return this.success(key, generation, totalBytes, files.length, { fromCache: true });
    }
    if (await this.readGeneration(key, generation)) {
      // A ready record whose files changed is unusable: quarantine it first.
      await this.quarantineReady(key, generation, "payload_mismatch");
      await this.repairPointer(key);
    }

    for (const other of await this.recordNames(key, "staging")) {
      if (other !== generation) await this.discardStaging(key, other);
    }
    if ((await this.storage.get(`${key}/quarantine/${generation}`)) !== null) {
      await this.purgeQuarantine(key, generation);
    }

    const sourceDigest = sha256Text(encoder.encode(request.source.identity));
    const trust = bundleTrustDigest(request.trustPolicy);
    const chunk = this.chunkBytes;
    let resumeReason: BundleResumeReason | null = null;
    let chunks: string[][] = files.map(() => []);
    const record = await this.readStaging(key, generation);
    let fresh = true;
    if (record && record.contentId === contentId && record.files.length === files.length) {
      if (record.sourceDigest !== sourceDigest) resumeReason = "source_changed";
      else if (record.trustDigest !== trust) resumeReason = "trust_policy_changed";
      else if (record.chunkBytes !== chunk) resumeReason = "chunk_size_changed";
      if (resumeReason === null) {
        fresh = false;
        chunks = [];
        for (const [index, file] of files.entries()) {
          const kept = await this.keptChunks(key, generation, index, file.size, record.files[index] ?? []);
          if (kept.mismatch) resumeReason = "partial_digest_mismatch";
          chunks.push(kept.digests);
        }
      }
    }
    if (fresh) await this.discardStaging(key, generation);
    const keptBytes = files.reduce((sum, file, index) => sum + this.verified(chunks[index]!.length, file.size), 0);
    const resumed = keptBytes > 0;
    const createdAt = (!fresh && record ? record.createdAt : null) ?? this.now();
    const checkpointRecord = (phase: "staging" | "verifying") =>
      this.writeStaging(key, generation, {
        contentId,
        sourceDigest,
        trustDigest: trust,
        chunkBytes: chunk,
        phase,
        createdAt,
        updatedAt: this.now(),
        files: chunks,
      });

    if ((await this.availableBytes()) < totalBytes - keptBytes) {
      if (!fresh) await checkpointRecord("staging");
      throw new BundleCacheError("storage_pressure");
    }
    const plan = await this.planQuota(totalBytes, null);
    if (!plan.satisfied) {
      if (!fresh) await checkpointRecord("staging");
      throw new BundleCacheError("quota_exceeded");
    }
    const evicted = await this.applyPlan(plan.candidates, key);

    await checkpointRecord("staging");
    if (fresh) {
      await this.storage.put(`${key}/payload/${generation}/m`, request.manifest);
      await this.storage.put(`${key}/payload/${generation}/e`, request.envelope);
      this.checkpoint("staging_record_written");
    }

    let transferred = 0;
    const cancelled = () => request.signal?.aborted === true;
    for (const [index, file] of files.entries()) {
      if (cancelled()) throw new BundleCacheError("cancelled");
      let have = this.verified(chunks[index]!.length, file.size);
      if (have === file.size) continue;
      let response: BundleRangeResponse;
      try {
        response = await request.source.open(file.path, have);
      } catch {
        throw new BundleCacheError("transfer_interrupted");
      }
      try {
        if (response.offset !== have) {
          if (response.offset !== 0) throw new BundleCacheError("range_mismatch");
          await this.storage.deletePrefix(`${key}/payload/${generation}/${index}`);
          chunks[index] = [];
          have = 0;
          resumeReason ??= "range_not_honored";
        }
        while (have < file.size) {
          const want = Math.min(chunk, file.size - have);
          const data = await readFully(response, want);
          transferred += data.length;
          if (data.length < want) throw new BundleCacheError("transfer_interrupted");
          await this.storage.put(`${key}/payload/${generation}/${index}/${chunks[index]!.length}`, data);
          chunks[index]!.push(sha256Text(data));
          have += want;
          await checkpointRecord("staging");
          this.checkpoint("chunk_committed");
          if (cancelled()) throw new BundleCacheError("cancelled");
        }
        let extra: Uint8Array | null = null;
        try {
          extra = await response.read(1);
        } catch {
          extra = null;
        }
        if (extra && extra.length > 0) {
          await this.quarantineStaging(key, generation, "file_size_mismatch");
          throw new BundleCacheError("verification_failed", "file_size_mismatch");
        }
      } finally {
        await response.close?.();
      }
    }

    await checkpointRecord("verifying");
    this.checkpoint("verifying_record_written");
    const failed = await this.verifyPayload(key, generation, files, chunk, true);
    if (failed) {
      await this.quarantineStaging(key, generation, failed);
      throw new BundleCacheError("verification_failed", failed);
    }

    const stamp = this.now();
    await this.writeGeneration(key, generation, {
      contentId,
      sequence: header.sequence,
      totalBytes,
      fileCount: files.length,
      chunkBytes: chunk,
      activated: false,
      pinned: false,
      verifiedAt: stamp,
      lastAccessed: stamp,
    });
    this.checkpoint("generation_committed");
    await this.storage.delete(`${key}/staging/${generation}`);

    if (activate) await this.activateLocked(key, generation, confirm);
    const trimmed = await this.applyQuota(0, generation, key);
    return this.success(key, generation, totalBytes, files.length, {
      fromCache: false,
      resumed,
      resumeReason,
      transferred,
      evicted: [...evicted, ...trimmed.evicted],
    });
  }

  private async keptChunks(
    key: string,
    generation: string,
    index: number,
    size: number,
    recorded: readonly string[],
  ): Promise<{ digests: string[]; mismatch: boolean }> {
    const kept: string[] = [];
    let mismatch = false;
    for (const [chunkIndex, digest] of recorded.entries()) {
      const start = chunkIndex * this.chunkBytes;
      const want = Math.min(this.chunkBytes, size - start);
      if (want <= 0) break;
      const data = await this.storage.get(`${key}/payload/${generation}/${index}/${chunkIndex}`);
      if (!data || data.length !== want) break;
      if (sha256Text(data) !== digest) {
        mismatch = true;
        break;
      }
      kept.push(digest);
    }
    const prefix = `${key}/payload/${generation}/${index}`;
    for (const entry of await this.storage.list(prefix)) {
      const chunkIndex = Number(entry.slice(prefix.length + 1));
      if (!(chunkIndex < kept.length)) await this.storage.delete(entry);
    }
    return { digests: kept, mismatch };
  }

  private verified(chunkCount: number, size: number): number {
    return Math.min(chunkCount * this.chunkBytes, size);
  }

  /** First verifier reason the payload fails, or `null`. */
  private async verifyPayload(
    key: string,
    generation: string,
    files: readonly DeclaredFile[],
    chunk: number,
    full: boolean,
  ): Promise<PortableBundleReason | null> {
    for (const [index, file] of files.entries()) {
      const prefix = `${key}/payload/${generation}/${index}`;
      const expected = Math.ceil(file.size / chunk);
      const entries = await this.storage.list(prefix);
      if (entries.length === 0 && file.size > 0) return "file_missing";
      let size = 0;
      const hasher = new Sha256();
      for (let chunkIndex = 0; chunkIndex < Math.max(expected, entries.length); chunkIndex += 1) {
        const chunkKey = `${prefix}/${chunkIndex}`;
        if (full) {
          const data = await this.storage.get(chunkKey);
          if (!data) return size === 0 ? "file_missing" : "file_size_mismatch";
          size += data.length;
          hasher.update(data);
        } else {
          const length = await this.storage.size(chunkKey);
          if (length === null) return "file_size_mismatch";
          size += length;
        }
      }
      if (size !== file.size || entries.length !== expected) return "file_size_mismatch";
      if (full && hasher.digest() !== file.sha256) return "file_digest_mismatch";
    }
    return null;
  }

  // -- activation and leases ------------------------------------------------

  private async activateLocked(key: string, generation: string, confirm: boolean): Promise<void> {
    if (!(await this.readGeneration(key, generation))) throw new BundleCacheError("generation_not_ready");
    const reason = await this.metadataCheck(key, generation);
    if (reason) {
      await this.quarantineReady(key, generation, reason);
      await this.repairPointer(key);
      throw new BundleCacheError("generation_corrupt", reason);
    }
    const pointer = await this.readPointer(key);
    let next: Pointer;
    if (pointer.active === generation) {
      next = { ...pointer, lastGreen: confirm ? generation : pointer.lastGreen };
    } else {
      const previous = pointer.active ?? pointer.rollback;
      next = {
        active: generation,
        lastGreen: confirm ? generation : pointer.lastGreen,
        rollback: previous === generation ? null : previous,
      };
    }
    await this.touch(key, generation, true);
    await this.writePointer(key, next);
    this.checkpoint("pointer_committed");
  }

  private async setPinned(bundleId: string, generation: string, pinned: boolean): Promise<void> {
    const key = bundleCacheKey(bundleId);
    await this.locked(key, async () => {
      const record = await this.readGeneration(key, generation);
      if (!record) throw new BundleCacheError("generation_not_ready");
      await this.writeGeneration(key, generation, { ...record, pinned });
    });
  }

  private async createLease(
    key: string,
    generation: string,
    verification: BundleOpenVerification,
  ): Promise<BundleLease> {
    const record = await this.readGeneration(key, generation);
    if (!record) throw new BundleCacheError("generation_not_ready");
    let reason = await this.metadataCheck(key, generation);
    const files = (await this.declaredFiles(key, generation)) ?? [];
    if (!reason && verification === "full") {
      reason = await this.verifyPayload(key, generation, files, record.chunkBytes, true);
    }
    if (reason) {
      await this.quarantineReady(key, generation, reason);
      await this.repairPointer(key);
      throw new BundleCacheError("generation_corrupt", reason);
    }
    const id = randomId();
    await this.writeRecord(`${key}/lease/${generation}/${id}`, {
      expires_at_ms: this.now() + this.leaseTtlMs,
      generation,
      lease: id,
      owner: this.ownerId,
      schema: "openmed.bundle_cache.lease",
      schema_version: 1,
    });
    this.heldLeases.add(id);
    await this.touch(key, generation, null);
    const storage = this.storage;
    const cache = this;
    const chunkKeys = async (path: string): Promise<string[]> => {
      const index = files.findIndex((file) => file.path === path);
      if (index < 0) throw new BundleCacheError("generation_not_ready");
      const count = Math.ceil(files[index]!.size / record.chunkBytes);
      return Array.from({ length: count }, (_, chunkIndex) => `${key}/payload/${generation}/${index}/${chunkIndex}`);
    };
    return {
      id,
      bundleKey: key,
      generation,
      filesChecked: record.fileCount,
      async readFile(path: string): Promise<Uint8Array> {
        const parts: Uint8Array[] = [];
        for (const chunkKey of await chunkKeys(path)) {
          const data = await storage.get(chunkKey);
          if (!data) throw new BundleCacheError("generation_corrupt", "file_missing");
          parts.push(data);
        }
        const output = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
        let offset = 0;
        for (const part of parts) {
          output.set(part, offset);
          offset += part.length;
        }
        return output;
      },
      async readBlob(path: string): Promise<Blob> {
        const parts: BlobPart[] = [];
        for (const chunkKey of await chunkKeys(path)) {
          const blob = storage.blob ? await storage.blob(chunkKey) : null;
          const data = blob ?? (await storage.get(chunkKey));
          if (!data) throw new BundleCacheError("generation_corrupt", "file_missing");
          parts.push(data as BlobPart);
        }
        return new Blob(parts);
      },
      async release(): Promise<void> {
        await cache.release(this).catch(() => undefined);
      },
      toString(): string {
        return `BundleLease(${generation})`;
      },
    } as BundleLease;
  }

  // -- quota and eviction ---------------------------------------------------

  private async planQuota(
    additionalBytes: number,
    exclude: string | null,
  ): Promise<{ candidates: { key: string; generation: string; size: number }[]; satisfied: boolean }> {
    let used = 0;
    const candidates: { key: string; generation: string; size: number; lastAccessed: number }[] = [];
    for (const key of await this.bundleKeys()) {
      const pointer = await this.readPointer(key);
      for (const generation of await this.recordNames(key, "generation")) {
        const record = await this.readGeneration(key, generation);
        if (!record) continue;
        used += record.totalBytes;
        if (generation !== exclude && (await this.protections(key, generation, pointer, record)).length === 0) {
          candidates.push({ key, generation, size: record.totalBytes, lastAccessed: record.lastAccessed });
        }
      }
    }
    candidates.sort((left, right) =>
      left.lastAccessed !== right.lastAccessed
        ? left.lastAccessed - right.lastAccessed
        : compareText(left.generation, right.generation),
    );
    const chosen: { key: string; generation: string; size: number }[] = [];
    for (const candidate of candidates) {
      if (used + additionalBytes <= this.quotaBytes) break;
      chosen.push(candidate);
      used -= candidate.size;
    }
    return { candidates: chosen, satisfied: used + additionalBytes <= this.quotaBytes };
  }

  private async applyQuota(
    additionalBytes: number,
    exclude: string | null,
    heldKey: string | null,
  ): Promise<BundleQuotaDecision> {
    const plan = await this.planQuota(additionalBytes, exclude);
    const evicted = await this.applyPlan(plan.candidates, heldKey);
    const used = await this.usedBytes();
    return {
      quotaBytes: this.quotaBytes,
      usedBytes: used,
      additionalBytes,
      evicted,
      satisfied: used + additionalBytes <= this.quotaBytes,
    };
  }

  private async applyPlan(
    candidates: readonly { key: string; generation: string }[],
    heldKey: string | null,
  ): Promise<string[]> {
    const evicted: string[] = [];
    for (const candidate of candidates) {
      const evict = async () => {
        const record = await this.readGeneration(candidate.key, candidate.generation);
        const pointer = await this.readPointer(candidate.key);
        if (!record || (await this.protections(candidate.key, candidate.generation, pointer, record)).length > 0) {
          return;
        }
        await this.storage.delete(`${candidate.key}/generation/${candidate.generation}`);
        this.checkpoint("eviction_committed");
        await this.storage.deletePrefix(`${candidate.key}/payload/${candidate.generation}`);
        await this.storage.deletePrefix(`${candidate.key}/lease/${candidate.generation}`);
        evicted.push(candidate.generation);
      };
      if (candidate.key === heldKey) {
        await evict();
      } else {
        // Wait only when no bundle lock is held, so two acquisitions evicting
        // each other's generations cannot deadlock.
        try {
          await this.locked(candidate.key, evict, heldKey === null ? this.lockTimeoutMs : 0);
        } catch (error) {
          if (!(error instanceof BundleCacheError && error.reason === "lock_timeout")) throw error;
        }
      }
    }
    return evicted;
  }

  private async usedBytes(): Promise<number> {
    let used = 0;
    for (const key of await this.bundleKeys()) {
      for (const generation of await this.recordNames(key, "generation")) {
        used += (await this.readGeneration(key, generation))?.totalBytes ?? 0;
      }
    }
    return used;
  }

  private async protections(
    key: string,
    generation: string,
    pointer: Pointer,
    record: GenerationRecord,
  ): Promise<BundleProtection[]> {
    const result: BundleProtection[] = [];
    if (pointer.active === generation) result.push("active");
    if (pointer.lastGreen === generation) result.push("last_green");
    if (pointer.rollback === generation) result.push("rollback");
    if (record.pinned) result.push("pinned");
    if (await this.isLeased(key, generation)) result.push("leased");
    return result;
  }

  private async isLeased(key: string, generation: string): Promise<boolean> {
    const now = this.now();
    for (const entry of await this.storage.list(`${key}/lease/${generation}`)) {
      const lease = await this.readJson(entry);
      if (!lease) continue;
      const mine = lease.owner === this.ownerId && this.heldLeases.has(String(lease.lease));
      if (mine || Number(lease.expires_at_ms) > now) return true;
    }
    return false;
  }

  private async stateOf(
    key: string,
    generation: string,
    pointer: Pointer,
    record: GenerationRecord,
  ): Promise<BundleGenerationState> {
    const protections = await this.protections(key, generation, pointer, record);
    if (protections.includes("active")) return "active";
    if (protections.length === 0) return "evictable";
    return record.activated ? "superseded" : "ready";
  }

  // -- recovery and snapshots ------------------------------------------------

  private async recoverLocked(key: string, stale: boolean, actions: BundleRecoveryStep[]): Promise<void> {
    const report = (action: BundleRecoveryAction, generation: string | null) =>
      actions.push({ action, bundleKey: key, generation });
    if (stale) report("stale_lock_cleared", null);
    const staging = await this.recordNames(key, "staging");
    const ready = await this.recordNames(key, "generation");
    const quarantined = await this.recordNames(key, "quarantine");
    const owned = new Set([...staging, ...ready, ...quarantined]);
    for (const generation of await this.payloadNames(key)) {
      if (owned.has(generation)) continue;
      await this.storage.deletePrefix(`${key}/payload/${generation}`);
      report("trash_purged", ref(generation));
    }
    for (const generation of quarantined) {
      await this.purgeQuarantine(key, generation);
      report("quarantine_purged", ref(generation));
    }
    const readySet = new Set(ready);
    for (const generation of staging) {
      const record = await this.readStaging(key, generation);
      if (!ref(generation) || !record) {
        await this.discardStaging(key, generation);
        report("staging_orphaned", ref(generation));
      } else if (readySet.has(generation)) {
        await this.storage.delete(`${key}/staging/${generation}`);
      } else if (this.now() - record.updatedAt > this.stagingMaxAgeMs) {
        await this.discardStaging(key, generation);
        report("staging_expired", generation);
      } else {
        if (record.phase !== "staging") await this.writeStaging(key, generation, { ...record, phase: "staging" });
        report("staging_resumable", generation);
      }
    }
    for (const generation of ready) {
      if ((await this.metadataCheck(key, generation)) !== null) {
        await this.quarantineReady(key, generation, "payload_mismatch");
        report("generation_quarantined", ref(generation));
      }
    }
    if (await this.repairPointer(key)) report("pointer_repaired", null);
    const now = this.now();
    for (const entry of await this.storage.list(`${key}/lease`)) {
      const lease = await this.readJson(entry);
      const mine = lease !== null && lease.owner === this.ownerId && this.heldLeases.has(String(lease.lease));
      if (lease === null || (!mine && Number(lease.expires_at_ms) <= now)) {
        await this.storage.delete(entry);
        report("lease_expired", ref(entry.split("/")[2]));
      }
    }
  }

  private async snapshotLocked(key: string): Promise<BundleCacheSnapshot> {
    const pointer = await this.readPointer(key);
    const generations: BundleGenerationSnapshot[] = [];
    let used = 0;
    for (const generation of await this.recordNames(key, "generation")) {
      const record = await this.readGeneration(key, generation);
      if (!record) continue;
      used += record.totalBytes;
      generations.push({
        generation,
        state: await this.stateOf(key, generation, pointer, record),
        totalBytes: record.totalBytes,
        verifiedBytes: record.totalBytes,
        protections: await this.protections(key, generation, pointer, record),
      });
    }
    for (const generation of await this.recordNames(key, "staging")) {
      if (!ref(generation) || generations.some((entry) => entry.generation === generation)) continue;
      const record = await this.readStaging(key, generation);
      const files = await this.declaredFiles(key, generation);
      if (!record || !files) continue;
      const verified = files.reduce(
        (sum, file, index) => sum + this.verified(record.files[index]?.length ?? 0, file.size),
        0,
      );
      generations.push({
        generation,
        state: "resumable",
        totalBytes: files.reduce((sum, file) => sum + file.size, 0),
        verifiedBytes: verified,
        protections: [],
      });
    }
    const quarantined = [];
    for (const generation of await this.recordNames(key, "quarantine")) {
      if (!ref(generation)) continue;
      const record = await this.readJson(`${key}/quarantine/${generation}`);
      quarantined.push({ generation, reason: typeof record?.reason === "string" ? record.reason : "unknown" });
    }
    generations.sort((left, right) => compareText(left.generation, right.generation));
    return {
      schema: BUNDLE_CACHE_SNAPSHOT_SCHEMA,
      schemaVersion: 1,
      bundleKey: key,
      active: pointer.active,
      lastGreen: pointer.lastGreen,
      rollback: pointer.rollback,
      generations,
      quarantined,
      usedBytes: used,
    };
  }

  /** Falls an invalid active generation back to last green, then rollback. */
  private async repairPointer(key: string): Promise<boolean> {
    const pointer = await this.readPointer(key);
    const valid = async (generation: string | null) =>
      generation !== null && (await this.readGeneration(key, generation)) !== null;
    const lastGreen = (await valid(pointer.lastGreen)) ? pointer.lastGreen : null;
    let rollback = (await valid(pointer.rollback)) ? pointer.rollback : null;
    const active = (await valid(pointer.active)) ? pointer.active : (lastGreen ?? rollback);
    if (rollback === active) rollback = null;
    if (active === pointer.active && lastGreen === pointer.lastGreen && rollback === pointer.rollback) return false;
    await this.writePointer(key, { active, lastGreen, rollback });
    return true;
  }

  // -- records and storage ---------------------------------------------------

  private async bundleKeys(): Promise<string[]> {
    const names = new Set<string>();
    for (const entry of await this.storage.list("bundles")) {
      const name = entry.split("/")[1];
      if (name && BUNDLE.test(name)) names.add(name);
    }
    return [...names].sort(compareText);
  }

  private async recordNames(key: string, kind: "staging" | "generation" | "quarantine"): Promise<string[]> {
    const prefix = `${key}/${kind}`;
    return (await this.storage.list(prefix))
      .map((entry) => entry.slice(prefix.length + 1))
      .filter((name) => !name.includes("/"))
      .sort(compareText);
  }

  private async payloadNames(key: string): Promise<string[]> {
    const names = new Set<string>();
    for (const entry of await this.storage.list(`${key}/payload`)) {
      const name = entry.split("/")[2];
      if (name) names.add(name);
    }
    return [...names].sort(compareText);
  }

  private async readJson(key: string): Promise<Record<string, unknown> | null> {
    const data = await this.storage.get(key);
    if (!data) return null;
    try {
      const value: unknown = JSON.parse(decoder.decode(data));
      return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
    } catch {
      return null;
    }
  }

  private async writeRecord(key: string, value: Record<string, unknown>): Promise<void> {
    await this.storage.put(key, encoder.encode(canonicalJson(value)));
  }

  private async readPointer(key: string): Promise<Pointer> {
    const value = await this.readJson(`${key}/pointer`);
    return { active: ref(value?.active), lastGreen: ref(value?.last_green), rollback: ref(value?.rollback) };
  }

  private async writePointer(key: string, pointer: Pointer): Promise<void> {
    await this.writeRecord(`${key}/pointer`, {
      active: pointer.active,
      last_green: pointer.lastGreen,
      rollback: pointer.rollback,
      schema: "openmed.bundle_cache.pointer",
      schema_version: 1,
      updated_at_ms: this.now(),
    });
  }

  private async readGeneration(key: string, generation: string): Promise<GenerationRecord | null> {
    if (!GENERATION.test(generation)) return null;
    const value = await this.readJson(`${key}/generation/${generation}`);
    if (!value || value.schema !== "openmed.bundle_cache.generation" || value.state !== "ready") return null;
    const numbers = ["sequence", "total_bytes", "file_count", "chunk_bytes", "verified_at_ms", "last_accessed_ms"];
    if (typeof value.content_id !== "string" || numbers.some((name) => typeof value[name] !== "number")) return null;
    if (typeof value.activated !== "boolean" || typeof value.pinned !== "boolean") return null;
    if (bundleGenerationId(value.content_id) !== generation) return null;
    return {
      contentId: value.content_id,
      sequence: value.sequence as number,
      totalBytes: value.total_bytes as number,
      fileCount: value.file_count as number,
      chunkBytes: value.chunk_bytes as number,
      activated: value.activated,
      pinned: value.pinned,
      verifiedAt: value.verified_at_ms as number,
      lastAccessed: value.last_accessed_ms as number,
    };
  }

  private async writeGeneration(key: string, generation: string, record: GenerationRecord): Promise<void> {
    await this.writeRecord(`${key}/generation/${generation}`, {
      activated: record.activated,
      chunk_bytes: record.chunkBytes,
      content_id: record.contentId,
      file_count: record.fileCount,
      generation,
      last_accessed_ms: record.lastAccessed,
      pinned: record.pinned,
      schema: "openmed.bundle_cache.generation",
      schema_version: 1,
      sequence: record.sequence,
      state: "ready",
      total_bytes: record.totalBytes,
      verified_at_ms: record.verifiedAt,
    });
  }

  private async touch(key: string, generation: string, activated: boolean | null): Promise<void> {
    const record = await this.readGeneration(key, generation);
    if (!record) return;
    await this.writeGeneration(key, generation, {
      ...record,
      activated: activated ?? record.activated,
      lastAccessed: this.now(),
    });
  }

  private async readStaging(key: string, generation: string): Promise<StagingRecord | null> {
    const value = await this.readJson(`${key}/staging/${generation}`);
    if (!value || value.schema !== "openmed.bundle_cache.staging") return null;
    const files = value.files;
    if (
      typeof value.content_id !== "string" ||
      typeof value.source_digest !== "string" ||
      typeof value.trust_digest !== "string" ||
      typeof value.chunk_bytes !== "number" ||
      (value.phase !== "staging" && value.phase !== "verifying") ||
      typeof value.created_at_ms !== "number" ||
      typeof value.updated_at_ms !== "number" ||
      !Array.isArray(files) ||
      !files.every((file) => Array.isArray(file) && file.every((digest) => typeof digest === "string"))
    ) {
      return null;
    }
    return {
      contentId: value.content_id,
      sourceDigest: value.source_digest,
      trustDigest: value.trust_digest,
      chunkBytes: value.chunk_bytes,
      phase: value.phase,
      createdAt: value.created_at_ms,
      updatedAt: value.updated_at_ms,
      files: files as string[][],
    };
  }

  private async writeStaging(key: string, generation: string, record: StagingRecord): Promise<void> {
    await this.writeRecord(`${key}/staging/${generation}`, {
      chunk_bytes: record.chunkBytes,
      content_id: record.contentId,
      created_at_ms: record.createdAt,
      files: record.files,
      phase: record.phase,
      schema: "openmed.bundle_cache.staging",
      schema_version: 1,
      source_digest: record.sourceDigest,
      trust_digest: record.trustDigest,
      updated_at_ms: record.updatedAt,
    });
  }

  private async declaredFiles(key: string, generation: string): Promise<DeclaredFile[] | null> {
    const manifest = await this.storage.get(`${key}/payload/${generation}/m`);
    return manifest ? (manifestHeader(manifest)?.files ?? null) : null;
  }

  /** Cheap check: the record, the manifest identity, the declared files, and their sizes. */
  private async metadataCheck(key: string, generation: string): Promise<PortableBundleReason | null> {
    const record = await this.readGeneration(key, generation);
    if (!record) return "file_missing";
    const manifest = await this.storage.get(`${key}/payload/${generation}/m`);
    if (!manifest) return "file_missing";
    if (sha256Text(manifest) !== record.contentId) return "file_digest_mismatch";
    const files = manifestHeader(manifest)?.files;
    if (!files) return "manifest_malformed";
    if (files.length !== record.fileCount || files.reduce((sum, file) => sum + file.size, 0) !== record.totalBytes) {
      return "file_size_mismatch";
    }
    return this.verifyPayload(key, generation, files, record.chunkBytes, false);
  }

  private async discardStaging(key: string, generation: string): Promise<void> {
    await this.storage.delete(`${key}/staging/${generation}`);
    await this.storage.deletePrefix(`${key}/payload/${generation}`);
  }

  private async purgeQuarantine(key: string, generation: string): Promise<void> {
    await this.storage.delete(`${key}/quarantine/${generation}`);
    await this.storage.deletePrefix(`${key}/payload/${generation}`);
  }

  private async writeQuarantine(key: string, generation: string, reason: string): Promise<void> {
    await this.writeRecord(`${key}/quarantine/${generation}`, {
      generation,
      quarantined_at_ms: this.now(),
      reason,
      schema: "openmed.bundle_cache.quarantine",
      schema_version: 1,
    });
  }

  private async quarantineStaging(key: string, generation: string, reason: string): Promise<void> {
    await this.writeQuarantine(key, generation, reason);
    await this.storage.delete(`${key}/staging/${generation}`);
  }

  private async quarantineReady(key: string, generation: string, reason: string): Promise<void> {
    await this.writeQuarantine(key, generation, reason);
    await this.storage.delete(`${key}/generation/${generation}`);
  }

  private now(): number {
    return this.clock();
  }

  private checkpoint(name: string): void {
    this.checkpointHook?.(name);
  }

  // -- locks -----------------------------------------------------------------

  private async locked<T>(
    key: string,
    body: (stale: boolean) => Promise<T>,
    timeoutMs: number = this.lockTimeoutMs,
  ): Promise<T> {
    const name = `openmed-bundle-cache:${key}`;
    const run = async (): Promise<T> => {
      const recordKey = `${key}/lock`;
      const stale = (await this.storage.get(recordKey)) !== null;
      await this.writeRecord(recordKey, {
        acquired_at_ms: this.now(),
        owner: this.ownerId,
        schema: "openmed.bundle_cache.lock",
        schema_version: 1,
      });
      // Register the bundle so recovery can enumerate it.
      if ((await this.storage.size(`bundles/${key}`)) === null) {
        await this.storage.put(`bundles/${key}`, new Uint8Array(0));
      }
      try {
        const result = await body(stale);
        await this.storage.delete(recordKey);
        return result;
      } catch (error) {
        if (!(error instanceof BundleCacheCrash)) await this.storage.delete(recordKey);
        throw error;
      }
    };
    if (this.locks) {
      if (timeoutMs === 0) {
        return this.locks.request(name, { ifAvailable: true }, async (lock) => {
          if (lock === null) throw new BundleCacheError("lock_timeout");
          return run();
        });
      }
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      let granted = false;
      try {
        return await this.locks.request(name, { signal: controller.signal }, async () => {
          granted = true;
          clearTimeout(timer);
          return run();
        });
      } catch (error) {
        if (!granted && (error as { name?: unknown } | null)?.name === "AbortError") {
          throw new BundleCacheError("lock_timeout");
        }
        throw error;
      } finally {
        clearTimeout(timer);
      }
    }
    const release = await localMutex(this.storage).acquire(name, timeoutMs);
    if (!release) throw new BundleCacheError("lock_timeout");
    try {
      return await run();
    } finally {
      release();
    }
  }

  // -- results ---------------------------------------------------------------

  private async success(
    key: string,
    generation: string,
    totalBytes: number,
    fileCount: number,
    extra: {
      fromCache: boolean;
      resumed?: boolean;
      resumeReason?: BundleResumeReason | null;
      transferred?: number;
      evicted?: readonly string[];
    },
  ): Promise<BundleAcquireResult> {
    const pointer = await this.readPointer(key);
    const record = await this.readGeneration(key, generation);
    return {
      ok: true,
      reason: null,
      bundleReason: null,
      bundleKey: key,
      generation,
      state: record ? await this.stateOf(key, generation, pointer, record) : null,
      fromCache: extra.fromCache,
      resumed: extra.resumed ?? false,
      resumeReason: extra.resumeReason ?? null,
      bytesTransferred: extra.transferred ?? 0,
      verifiedBytes: totalBytes,
      filesVerified: fileCount,
      evicted: extra.evicted ?? [],
    };
  }
}

function failure(
  reason: BundleCacheReason,
  bundleReason: PortableBundleReason | null,
  key: string | null,
  generation: string | null,
): BundleAcquireResult {
  return {
    ok: false,
    reason,
    bundleReason,
    bundleKey: key,
    generation,
    state: null,
    fromCache: false,
    resumed: false,
    resumeReason: null,
    bytesTransferred: 0,
    verifiedBytes: 0,
    filesVerified: 0,
    evicted: [],
  };
}

async function readFully(response: BundleRangeResponse, want: number): Promise<Uint8Array> {
  const output = new Uint8Array(want);
  let filled = 0;
  try {
    while (filled < want) {
      const next = await response.read(want - filled);
      if (!next || next.length === 0) break;
      const take = Math.min(next.length, want - filled);
      output.set(next.subarray(0, take), filled);
      filled += take;
    }
  } catch {
    // A failing source is reported as an interrupted transfer by the caller.
  }
  return output.subarray(0, filled);
}

async function defaultAvailableBytes(): Promise<number> {
  const scope = globalThis as unknown as {
    navigator?: { storage?: { estimate?: () => Promise<{ quota?: number; usage?: number }> } };
  };
  const estimate = await scope.navigator?.storage?.estimate?.().catch(() => null);
  if (!estimate || typeof estimate.quota !== "number") return Number.MAX_SAFE_INTEGER;
  return Math.max(0, estimate.quota - (estimate.usage ?? 0));
}

function randomId(): string {
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}
