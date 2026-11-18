import assert from "node:assert/strict";
import { createHash, randomBytes } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  BUNDLE_CACHE_REASONS,
  BUNDLE_GENERATION_STATES,
  BUNDLE_RECOVERY_ACTIONS,
  BUNDLE_RESUME_REASONS,
  BundleCacheError,
  BundleGenerationCache,
  IndexedDbBundleCacheStorage,
  MemoryBundleCacheStorage,
  OpfsBundleCacheStorage,
  Sha256,
  bundleCacheKey,
  bundleGenerationId,
  bundleTrustDigest,
  createBlobRangeSource,
  parsePortableBundleTrustPolicy,
  portableBundleContentId,
  type BundleCacheStorage,
  type BundleLease,
  type BundleLockManager,
  type BundleRangeResponse,
  type BundleRangeSource,
  type IdbFactoryLike,
  type OpfsDirectoryLike,
  type OpfsFileLike,
} from "../../js/openmedkit-web/src/index";
import { BundleCacheCrash } from "../../js/openmedkit-web/src/bundle-cache";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const vectorsPath = join(rootDir, "tests", "fixtures", "bundle_cache", "vectors.json");

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
type JsonObject = { [key: string]: Json };
type Bundle = {
  bundle_id: string;
  content_id: string;
  envelope: string;
  files: { path: string; text: string }[];
  generation: string;
  manifest: string;
  total_bytes: number;
};
type Vectors = {
  bundles: Record<string, Bundle>;
  checkpoints: string[];
  clock: { start_ms: number; step_ms: number };
  context: { now: string };
  identities: { bundle_id: string; bundle_key: string; trust_digests: Record<string, string> };
  reason_codes: string[];
  recovery_actions: string[];
  resume_reasons: string[];
  scenarios: { id: string; options: Record<string, number>; steps: JsonObject[] }[];
  states: string[];
  synthetic_only: boolean;
  trust_policies: Record<string, unknown>;
};

const vectors = JSON.parse(readFileSync(vectorsPath, "utf8")) as Vectors;
const encoder = new TextEncoder();
const bundleId = vectors.identities.bundle_id;

function bundle(alias: string): Bundle {
  const value = vectors.bundles[alias];
  assert.ok(value, alias);
  return value;
}

function files(alias: string): Map<string, Uint8Array> {
  return new Map(bundle(alias).files.map((entry) => [entry.path, encoder.encode(entry.text)]));
}

function policy(name: string) {
  return parsePortableBundleTrustPolicy(vectors.trust_policies[name]);
}

// ---------------------------------------------------------------------------
// Storage fakes: a minimal OPFS directory tree and a minimal IndexedDB.

class FakeFileHandle implements OpfsFileLike {
  readonly kind = "file" as const;
  data = new Uint8Array(0);

  async getFile(): Promise<Blob> {
    return new Blob([this.data as BlobPart]);
  }

  async createWritable() {
    let pending = new Uint8Array(0);
    return {
      write: async (chunk: Uint8Array) => {
        const next = new Uint8Array(pending.length + chunk.length);
        next.set(pending, 0);
        next.set(chunk, pending.length);
        pending = next;
      },
      close: async () => {
        this.data = pending;
      },
      abort: async () => undefined,
    };
  }
}

class FakeDirectoryHandle implements OpfsDirectoryLike {
  readonly kind = "directory" as const;
  readonly children = new Map<string, FakeDirectoryHandle | FakeFileHandle>();

  async getDirectoryHandle(name: string, options: { create?: boolean } = {}): Promise<OpfsDirectoryLike> {
    let child = this.children.get(name);
    if (!child) {
      if (!options.create) throw new DOMException("missing", "NotFoundError");
      child = new FakeDirectoryHandle();
      this.children.set(name, child);
    }
    if (child.kind !== "directory") throw new DOMException("type", "TypeMismatchError");
    return child;
  }

  async getFileHandle(name: string, options: { create?: boolean } = {}): Promise<OpfsFileLike> {
    let child = this.children.get(name);
    if (!child) {
      if (!options.create) throw new DOMException("missing", "NotFoundError");
      child = new FakeFileHandle();
      this.children.set(name, child);
    }
    if (child.kind !== "file") throw new DOMException("type", "TypeMismatchError");
    return child;
  }

  async removeEntry(name: string): Promise<void> {
    if (!this.children.delete(name)) throw new DOMException("missing", "NotFoundError");
  }

  async *entries(): AsyncIterable<[string, OpfsDirectoryLike | OpfsFileLike]> {
    for (const entry of [...this.children.entries()]) yield entry;
  }
}

function fakeIndexedDb(): { factory: IdbFactoryLike; keyRange: { bound(lower: string, upper: string): unknown } } {
  const store = new Map<string, unknown>();
  let created = false;
  const inRange = (key: string, range: { lower: string; upper: string }) => key >= range.lower && key <= range.upper;
  const request = <T>(operation: () => T, complete: () => void) => {
    const value: { result: T; error: unknown; onsuccess: ((event: unknown) => void) | null; onerror: null } = {
      result: undefined as T,
      error: null,
      onsuccess: null,
      onerror: null,
    };
    queueMicrotask(() => {
      value.result = operation();
      value.onsuccess?.({});
      queueMicrotask(complete);
    });
    return value;
  };
  const db = {
    objectStoreNames: { contains: () => created },
    createObjectStore: () => {
      created = true;
    },
    transaction: () => {
      const transaction = {
        oncomplete: null as ((event: unknown) => void) | null,
        onerror: null,
        onabort: null,
        error: null,
        objectStore: () => api,
      };
      const done = () => transaction.oncomplete?.({});
      const api = {
        get: (key: string) => request(() => store.get(key), done),
        put: (value: unknown, key: string) =>
          request(() => {
            store.set(key, value);
          }, done),
        delete: (key: unknown) =>
          request(() => {
            if (typeof key === "string") store.delete(key);
            else for (const name of [...store.keys()]) if (inRange(name, key as never)) store.delete(name);
          }, done),
        getAllKeys: (range: unknown) =>
          request(() => [...store.keys()].filter((name) => inRange(name, range as never)).sort(), done),
      };
      return transaction;
    },
  };
  const factory = {
    open: () => {
      const value = {
        result: db,
        error: null,
        onsuccess: null as ((event: unknown) => void) | null,
        onerror: null,
        onupgradeneeded: null as ((event: unknown) => void) | null,
      };
      queueMicrotask(() => {
        if (!created) value.onupgradeneeded?.({});
        value.onsuccess?.({});
      });
      return value;
    },
  };
  return { factory: factory as unknown as IdbFactoryLike, keyRange: { bound: (lower, upper) => ({ lower, upper }) } };
}

const backends: Record<string, () => Promise<BundleCacheStorage>> = {
  memory: async () => new MemoryBundleCacheStorage(),
  opfs: async () => new OpfsBundleCacheStorage(new FakeDirectoryHandle()),
  indexeddb: async () => IndexedDbBundleCacheStorage.open(fakeIndexedDb()),
};

// ---------------------------------------------------------------------------
// Fake range source with scripted faults.

class FakeSource implements BundleRangeSource {
  delivered = 0;

  constructor(
    private readonly bytes: Map<string, Uint8Array>,
    readonly identity: string,
    private readonly options: {
      failAfter?: number | null;
      cancelAfter?: number | null;
      corrupt?: { path: string; offset: number } | null;
      extra?: { path: string; count: number } | null;
      ignoreRange?: boolean;
      onRead?: () => Promise<void> | void;
      controller?: AbortController;
    } = {},
  ) {}

  async open(path: string, offset: number): Promise<BundleRangeResponse> {
    let data = this.bytes.get(path);
    if (!data) throw new Error("missing");
    const { corrupt, extra } = this.options;
    if (corrupt && corrupt.path === path) {
      data = data.slice();
      data[corrupt.offset] = data[corrupt.offset]! ^ 0xff;
    }
    if (extra && extra.path === path) {
      const longer = new Uint8Array(data.length + extra.count).fill(0x58);
      longer.set(data, 0);
      data = longer;
    }
    const start = this.options.ignoreRange ? 0 : offset;
    let position = start;
    const bytes = data;
    return {
      offset: start,
      read: async (maxBytes: number) => {
        await this.options.onRead?.();
        if (position >= bytes.length) return null;
        let allowed = maxBytes;
        if (this.options.failAfter != null) {
          const left = this.options.failAfter - this.delivered;
          if (left <= 0) throw new Error("interrupted");
          allowed = Math.min(allowed, left);
        }
        const chunk = bytes.subarray(position, Math.min(bytes.length, position + allowed));
        position += chunk.length;
        this.delivered += chunk.length;
        if (this.options.cancelAfter != null && this.delivered >= this.options.cancelAfter) {
          this.options.controller?.abort();
        }
        return chunk.slice();
      },
    };
  }
}

function request(
  alias: string,
  source: BundleRangeSource,
  extra: { policy?: string; activate?: boolean; confirm?: boolean; signal?: AbortSignal } = {},
) {
  return {
    manifest: encoder.encode(bundle(alias).manifest),
    envelope: encoder.encode(bundle(alias).envelope),
    source,
    trustPolicy: policy(extra.policy ?? "default"),
    context: { now: vectors.context.now },
    activate: extra.activate ?? true,
    confirm: extra.confirm ?? true,
    ...(extra.signal ? { signal: extra.signal } : {}),
  };
}

// ---------------------------------------------------------------------------
// Vector replay.

const ALIAS_KEYS = new Set(["generation", "active", "last_green", "rollback", "evicted"]);

function resolve(value: Json, key: string | null = null): Json {
  if (Array.isArray(value)) return value.map((item) => resolve(item, key));
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([name, item]) => [name, resolve(item, name)]));
  }
  if (typeof value === "string" && key && ALIAS_KEYS.has(key)) return bundle(value).generation;
  return value;
}

function canonical(value: unknown): string {
  const sort = (item: unknown): unknown => {
    if (Array.isArray(item)) return item.map(sort);
    if (item && typeof item === "object") {
      return Object.fromEntries(
        Object.keys(item)
          .sort()
          .map((name) => [name, sort((item as Record<string, unknown>)[name])]),
      );
    }
    return item;
  };
  return JSON.stringify(sort(value));
}

function failureOf(error: unknown): JsonObject {
  if (!(error instanceof BundleCacheError)) throw error;
  return { ok: false, reason: error.reason, bundle_reason: error.bundleReason };
}

async function attempt(block: () => Promise<JsonObject>): Promise<JsonObject> {
  try {
    return await block();
  } catch (error) {
    return failureOf(error);
  }
}

async function replay(scenario: Vectors["scenarios"][number], storage: BundleCacheStorage, label: string) {
  let now = vectors.clock.start_ms;
  let owner = 1;
  let available = Number.MAX_SAFE_INTEGER;
  const leases = new Map<string, BundleLease>();
  const options = scenario.options;
  const newCache = () =>
    new BundleGenerationCache({
      storage,
      quotaBytes: options.quota_bytes!,
      chunkBytes: options.chunk_bytes!,
      lockTimeoutMs: 2_000,
      leaseTtlMs: options.lease_ttl_ms!,
      stagingMaxAgeMs: options.staging_max_age_ms!,
      now: () => now,
      availableBytes: async () => available,
      ownerId: `owner-${owner++}`,
      locks: null,
    });
  let cache = newCache();
  const installCrash = (raw: Json | undefined) => {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
      cache.checkpointHook = null;
      return;
    }
    let seen = 0;
    const name = raw.checkpoint as string;
    const occurrence = raw.occurrence as number;
    cache.checkpointHook = (checkpoint) => {
      if (checkpoint !== name) return;
      seen += 1;
      if (seen === occurrence) throw new BundleCacheCrash(checkpoint);
    };
  };
  const crashed = async (block: () => Promise<unknown>): Promise<JsonObject> => {
    await assert.rejects(block, BundleCacheCrash);
    return { crashed: true };
  };

  for (const [index, step] of scenario.steps.entries()) {
    now += vectors.clock.step_ms;
    const expect = step.expect as JsonObject | undefined;
    let actual: JsonObject | null = null;
    switch (step.op) {
      case "acquire": {
        const described = step.source as JsonObject;
        const controller = new AbortController();
        const source = new FakeSource(files(described.serve as string), described.identity as string, {
          failAfter: described.fail_after_bytes as number | null,
          cancelAfter: described.cancel_after_bytes as number | null,
          corrupt: described.corrupt as { path: string; offset: number } | null,
          extra: described.extra_bytes as { path: string; count: number } | null,
          ignoreRange: described.ignore_range as boolean,
          controller,
        });
        installCrash(step.crash_at);
        if (typeof step.storage_available_bytes === "number") available = step.storage_available_bytes;
        const acquireRequest = request(step.bundle as string, source, {
          policy: step.policy as string,
          activate: step.activate as boolean,
          confirm: step.confirm as boolean,
          signal: controller.signal,
        });
        try {
          if (expect?.crashed) {
            actual = await crashed(() => cache.acquire(acquireRequest));
          } else {
            const result = await cache.acquire(acquireRequest);
            actual = result.ok
              ? {
                  ok: true,
                  generation: result.generation,
                  state: result.state,
                  from_cache: result.fromCache,
                  resumed: result.resumed,
                  resume_reason: result.resumeReason,
                  bytes_transferred: result.bytesTransferred,
                  verified_bytes: result.verifiedBytes,
                  files_verified: result.filesVerified,
                  evicted: [...result.evicted],
                }
              : { ok: false, reason: result.reason, bundle_reason: result.bundleReason };
          }
        } finally {
          available = Number.MAX_SAFE_INTEGER;
          cache.checkpointHook = null;
        }
        break;
      }
      case "restart":
        now += step.advance_ms as number;
        leases.clear();
        cache = newCache();
        break;
      case "recover":
        actual = {
          actions: (await cache.recover()).actions.map((item) => ({
            action: item.action,
            generation: item.generation,
          })),
        };
        break;
      case "activate":
        actual = await attempt(async () => {
          await cache.activate(bundleId, bundle(step.generation as string).generation, {
            confirm: step.confirm as boolean,
          });
          return { ok: true };
        });
        break;
      case "confirm":
        actual = await attempt(async () => {
          await cache.confirm(bundleId);
          return { ok: true };
        });
        break;
      case "rollback":
        actual = await attempt(async () => {
          await cache.rollback(bundleId, { quarantine: step.quarantine as boolean });
          return { ok: true };
        });
        break;
      case "pin":
      case "unpin":
        actual = await attempt(async () => {
          const target = bundle(step.generation as string).generation;
          if (step.op === "pin") await cache.pin(bundleId, target);
          else await cache.unpin(bundleId, target);
          return { ok: true };
        });
        break;
      case "lease":
        actual = await attempt(async () => {
          const target = step.generation ? bundle(step.generation as string).generation : null;
          const lease = await cache.lease(bundleId, target);
          leases.set(step.lease as string, lease);
          return { ok: true, generation: lease.generation };
        });
        break;
      case "release":
        actual = await attempt(async () => {
          await cache.release(leases.get(step.lease as string)!);
          return { ok: true };
        });
        break;
      case "open":
        actual = await attempt(async () => {
          const lease = await cache.open(bundleId, {
            verification: step.verification === "full" ? "full" : "metadata",
          });
          leases.set(step.lease as string, lease);
          return { ok: true, generation: lease.generation, files_checked: lease.filesChecked };
        });
        break;
      case "evict":
        installCrash(step.crash_at);
        try {
          if (expect?.crashed) {
            actual = await crashed(() => cache.evict(step.additional_bytes as number));
          } else {
            const decision = await cache.evict(step.additional_bytes as number);
            actual = { evicted: [...decision.evicted], satisfied: decision.satisfied, used_bytes: decision.usedBytes };
          }
        } finally {
          cache.checkpointHook = null;
        }
        break;
      case "tamper":
        await tamper(storage, step, options.chunk_bytes!);
        break;
      case "advance":
        now += step.ms as number;
        break;
      case "snapshot": {
        const snapshot = await cache.snapshot(bundleId);
        actual = {
          pointer: { active: snapshot.active, last_green: snapshot.lastGreen, rollback: snapshot.rollback },
          generations: snapshot.generations.map((entry) => ({
            generation: entry.generation,
            state: entry.state,
            total_bytes: entry.totalBytes,
            verified_bytes: entry.verifiedBytes,
            protections: [...entry.protections],
          })),
          quarantined: snapshot.quarantined.map((entry) => ({ generation: entry.generation, reason: entry.reason })),
          used_bytes: snapshot.usedBytes,
        };
        break;
      }
      default:
        assert.fail(`unknown op ${String(step.op)}`);
    }
    if (actual !== null) {
      assert.equal(canonical(actual), canonical(resolve(expect ?? null)), `${label} ${scenario.id} step ${index} ${String(step.op)}`);
    }
  }
}

/** Tampers with stored payload chunks the way the vectors describe. */
async function tamper(storage: BundleCacheStorage, step: JsonObject, chunkBytes: number): Promise<void> {
  const alias = step.generation as string;
  const fileIndex = bundle(alias).files.findIndex((entry) => entry.path === step.path);
  const prefix = `${bundleCacheKey(bundleId)}/payload/${bundle(alias).generation}/${fileIndex}`;
  const offset = step.offset as number;
  const chunkIndex = Math.floor(offset / chunkBytes);
  const chunkKey = `${prefix}/${chunkIndex}`;
  const data = await storage.get(chunkKey);
  assert.ok(data, "tamper target exists");
  if (step.mode === "truncate") {
    for (const key of await storage.list(prefix)) {
      if (Number(key.slice(prefix.length + 1)) > chunkIndex) await storage.delete(key);
    }
    await storage.put(chunkKey, data.subarray(0, offset % chunkBytes));
  } else {
    data[offset % chunkBytes] = data[offset % chunkBytes]! ^ 0xff;
    await storage.put(chunkKey, data);
  }
}

for (const [name, open] of Object.entries(backends)) {
  test(`every shared bundle-cache scenario replays exactly on ${name} storage`, async () => {
    assert.ok(vectors.scenarios.length >= 30);
    for (const scenario of vectors.scenarios) await replay(scenario, await open(), name);
  });
}

test("identities and vocabularies match the shared vectors", async () => {
  assert.equal(vectors.synthetic_only, true);
  assert.equal(bundleCacheKey(bundleId), vectors.identities.bundle_key);
  for (const [name, digest] of Object.entries(vectors.identities.trust_digests)) {
    assert.equal(bundleTrustDigest(policy(name)), digest, name);
  }
  for (const alias of ["v1", "v2", "v3", "v4"]) {
    const contentId = await portableBundleContentId(encoder.encode(bundle(alias).manifest));
    assert.equal(contentId, bundle(alias).content_id);
    assert.equal(bundleGenerationId(contentId), bundle(alias).generation);
  }
  assert.deepEqual(vectors.reason_codes, [...BUNDLE_CACHE_REASONS]);
  assert.deepEqual(vectors.states, [...BUNDLE_GENERATION_STATES]);
  assert.deepEqual(vectors.recovery_actions, [...BUNDLE_RECOVERY_ACTIONS]);
  assert.deepEqual(vectors.resume_reasons, [...BUNDLE_RESUME_REASONS]);
});

test("incremental SHA-256 matches node:crypto across block boundaries", () => {
  for (const length of [0, 1, 55, 56, 63, 64, 65, 119, 120, 1000, 4097]) {
    const data = new Uint8Array(randomBytes(length));
    const expected = `sha256:${createHash("sha256").update(data).digest("hex")}`;
    const hasher = new Sha256();
    for (let offset = 0; offset < data.length; offset += 7) hasher.update(data.subarray(offset, offset + 7));
    assert.equal(hasher.digest(), expected, String(length));
  }
});

function cacheFor(storage: BundleCacheStorage, ownerId: string, extra: Partial<{ lockTimeoutMs: number; locks: BundleLockManager | null }> = {}) {
  return new BundleGenerationCache({
    storage,
    chunkBytes: 16,
    ownerId,
    lockTimeoutMs: extra.lockTimeoutMs ?? 5_000,
    locks: extra.locks === undefined ? null : extra.locks,
  });
}

/** A Web Locks stand-in with exclusive, queued, abortable requests. */
function fakeLockManager(): BundleLockManager {
  const queues = new Map<string, Promise<void>>();
  return {
    async request(name, options, callback) {
      const previous = queues.get(name) ?? Promise.resolve();
      let release!: () => void;
      const mine = new Promise<void>((resolve) => {
        release = resolve;
      });
      const tail = previous.then(() => mine);
      queues.set(name, tail);
      if (options.ifAvailable) {
        const idle = await Promise.race([previous.then(() => true), Promise.resolve(false)]);
        if (!idle) {
          void previous.then(release);
          return callback(null);
        }
      }
      const aborted = new Promise<never>((_, reject) => {
        options.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      });
      try {
        await Promise.race([previous, aborted]);
      } catch (error) {
        void previous.then(release);
        throw error;
      }
      try {
        return await callback({ name });
      } finally {
        release();
        if (queues.get(name) === tail) queues.delete(name);
      }
    },
  };
}

for (const locks of [null, "web-locks"] as const) {
  test(`concurrent callers converge on one verified generation (${locks ?? "in-page mutex"})`, async () => {
    const storage = new MemoryBundleCacheStorage();
    const manager = locks ? fakeLockManager() : null;
    const caches = ["tab-a", "tab-b", "tab-c"].map((owner) => cacheFor(storage, owner, { locks: manager }));
    const results = await Promise.all(
      caches.map((cache) =>
        cache.acquire(request("v1", new FakeSource(files("v1"), "source-a", { onRead: () => new Promise(setImmediate) }))),
      ),
    );
    assert.ok(results.every((result) => result.ok), JSON.stringify(results));
    assert.equal(results.filter((result) => !result.fromCache).length, 1);
    assert.deepEqual(new Set(results.map((result) => result.generation)), new Set([bundle("v1").generation]));
    const snapshot = await caches[0]!.snapshot(bundleId);
    assert.deepEqual(snapshot.generations.map((entry) => [entry.generation, entry.state]), [[bundle("v1").generation, "active"]]);
    const lease = await caches[1]!.open(bundleId, { verification: "full" });
    for (const [path, data] of files("v1")) assert.deepEqual(await lease.readFile(path), data);
    assert.equal((await lease.readBlob("config.json")).size, files("v1").get("config.json")!.length);
    await lease.release();
  });

  test(`lock waits are bounded and typed (${locks ?? "in-page mutex"})`, async () => {
    const storage = new MemoryBundleCacheStorage();
    const manager = locks ? fakeLockManager() : null;
    const holder = cacheFor(storage, "holder", { locks: manager });
    const waiter = cacheFor(storage, "waiter", { locks: manager, lockTimeoutMs: 50 });
    let enter!: () => void;
    const entered = new Promise<void>((resolve) => {
      enter = resolve;
    });
    let proceed!: () => void;
    const gate = new Promise<void>((resolve) => {
      proceed = resolve;
    });
    let first = true;
    const held = holder.acquire(
      request(
        "v1",
        new FakeSource(files("v1"), "source-a", {
          onRead: async () => {
            if (first) {
              first = false;
              enter();
              await gate;
            }
          },
        }),
      ),
    );
    await entered;
    const refused = await waiter.acquire(request("v1", new FakeSource(files("v1"), "source-a")));
    assert.equal(refused.reason, "lock_timeout");
    await assert.rejects(() => waiter.snapshot(bundleId), (error: unknown) => {
      return error instanceof BundleCacheError && error.reason === "lock_timeout";
    });
    proceed();
    assert.equal((await held).ok, true);
    assert.equal((await waiter.acquire(request("v1", new FakeSource(files("v1"), "source-a")))).fromCache, true);
  });
}

test("a prepared bundle imports and opens with zero network access", async () => {
  const original = globalThis.fetch;
  let attempts = 0;
  globalThis.fetch = (async () => {
    attempts += 1;
    throw new Error("network denied");
  }) as typeof fetch;
  try {
    const storage = await backends.opfs!();
    const prepared = new Map<string, Blob>([...files("v1")].map(([path, data]) => [path, new Blob([data as BlobPart])]));
    const imported = await cacheFor(storage, "importer").acquire(
      request("v1", createBlobRangeSource(prepared), { activate: false }),
    );
    assert.equal(imported.ok, true);
    assert.equal(imported.state, "evictable");
    const restarted = cacheFor(storage, "restarted");
    assert.deepEqual((await restarted.recover()).actions, []);
    await restarted.activate(bundleId, bundle("v1").generation);
    const lease = await restarted.open(bundleId, { verification: "full" });
    assert.equal(lease.filesChecked, files("v1").size);
    assert.deepEqual(await lease.readFile("onnx/model.onnx"), files("v1").get("onnx/model.onnx"));
    await lease.release();
  } finally {
    globalThis.fetch = original;
  }
  assert.equal(attempts, 0);
});

test("browser storage pressure mid-transfer keeps resumable bytes and the active generation", async () => {
  const inner = new MemoryBundleCacheStorage();
  let chunkWrites = 0;
  let full = false;
  const storage: BundleCacheStorage = {
    kind: "memory",
    get: (key) => inner.get(key),
    put: async (key, value) => {
      if (/\/payload\/[^/]+\/\d+\/\d+$/.test(key) && full && ++chunkWrites > 3) {
        throw new DOMException("quota", "QuotaExceededError");
      }
      await inner.put(key, value);
    },
    delete: (key) => inner.delete(key),
    list: (prefix) => inner.list(prefix),
    deletePrefix: (prefix) => inner.deletePrefix(prefix),
    size: (key) => inner.size(key),
  };
  const cache = cacheFor(storage, "owner");
  assert.equal((await cache.acquire(request("v1", new FakeSource(files("v1"), "source-a")))).ok, true);
  full = true;
  const pressured = await cache.acquire(request("v2", new FakeSource(files("v2"), "source-a")));
  assert.equal(pressured.reason, "storage_pressure");
  const snapshot = await cache.snapshot(bundleId);
  assert.equal(snapshot.active, bundle("v1").generation);
  const staged = snapshot.generations.find((entry) => entry.generation === bundle("v2").generation);
  assert.equal(staged?.state, "resumable");
  assert.equal(staged?.verifiedBytes, 48);
  full = false;
  const resumed = await cache.acquire(request("v2", new FakeSource(files("v2"), "source-a")));
  assert.equal(resumed.ok, true);
  assert.equal(resumed.resumed, true);
  assert.equal(resumed.bytesTransferred, bundle("v2").total_bytes - 48);
});

test("records, results, and errors carry no source identities or credentials", async () => {
  const storage = new MemoryBundleCacheStorage();
  const identity = "https://models.invalid/private?token=sentinel-credential";
  const cache = cacheFor(storage, "owner");
  const interrupted = await cache.acquire(request("v1", new FakeSource(files("v1"), identity, { failAfter: 100 })));
  assert.equal(interrupted.reason, "transfer_interrupted");
  const result = await cache.acquire(request("v1", new FakeSource(files("v1"), identity)));
  assert.equal(result.ok, true);
  const lease = await cache.open(bundleId);
  const error = await cache.rollback(bundleId).catch((caught: unknown) => caught);
  assert.ok(error instanceof BundleCacheError);
  const surfaces = [
    JSON.stringify(result),
    JSON.stringify(interrupted),
    String(lease),
    String(cache),
    JSON.stringify(await cache.snapshot(bundleId)),
    JSON.stringify(await cache.recover()),
    error.message,
    String(error),
  ];
  const sentinels = ["sentinel-credential", "models.invalid", "token=", "https://"];
  for (const surface of surfaces) {
    for (const sentinel of sentinels) assert.ok(!surface.includes(sentinel), `leak of ${sentinel} in ${surface}`);
  }
  const decoder = new TextDecoder();
  const keys = await storage.list(bundleCacheKey(bundleId));
  const records = keys.filter((key) => !key.includes("/payload/"));
  assert.ok(records.length >= 2);
  for (const key of records) {
    const text = decoder.decode((await storage.get(key))!);
    for (const sentinel of sentinels) assert.ok(!text.includes(sentinel), `leak of ${sentinel} in a record`);
  }
  await lease.release();
});
