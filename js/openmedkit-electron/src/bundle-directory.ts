/**
 * Verify a signed OpenMed portable bundle directory from Node or an Electron
 * utility process.
 *
 * This is a thin directory reader around the `openmed` TypeScript verifier
 * (`verifyPortableBundle`). It never follows symbolic links, reads only
 * regular files, opens each declared file once with `O_NOFOLLOW` where the
 * platform supports it, and keeps the verified bytes so a loader can use
 * exactly what was checked. Results carry only reason codes, digests, counts
 * and indexes.
 */
import {
  closeSync,
  constants as fsConstants,
  fstatSync,
  openSync,
  readSync,
} from "node:fs";
import { lstat, open, readdir } from "node:fs/promises";
import { join } from "node:path";

import {
  PORTABLE_BUNDLE_DEFAULT_LIMITS,
  PORTABLE_BUNDLE_ENVELOPE_FILENAME,
  PORTABLE_BUNDLE_MANIFEST_FILENAME,
  PortableBundleError,
  parsePortableBundleManifest,
  verifyPortableBundle,
  type PortableBundleContext,
  type PortableBundleReason,
  type PortableBundleTrustPolicy,
  type PortableBundleVerification,
} from "openmed";

/** Result of {@link verifyPortableBundleDirectory}. */
export interface PortableBundleDirectoryResult {
  readonly verification: PortableBundleVerification;
  /** Parsed manifest; set only when the bundle verified with its file tree. */
  readonly manifest: Readonly<Record<string, unknown>> | null;
  /** Verified bytes by relative path; set only when the bundle verified. */
  readonly files: ReadonlyMap<string, Uint8Array> | null;
}

const NO_FOLLOW = (fsConstants as { O_NOFOLLOW?: number }).O_NOFOLLOW ?? 0;
const MAX_WALK_ENTRIES_FLOOR = 4_096;

/**
 * Verify a bundle directory and return its verified bytes.
 *
 * Checks run in the portable contract's order: the manifest and envelope are
 * verified first, and the directory tree is read only when they pass. A
 * symbolic link or special file anywhere in the tree is `path_unsafe`; a
 * missing or unreadable root, manifest or envelope is `file_missing`.
 *
 * Args:
 *   root: Absolute bundle directory chosen by the host.
 *   input: Caller trust policy and verification context.
 *
 * Returns:
 *   The verification result, plus the manifest and verified file bytes when
 *   the bundle is eligible for activation.
 */
export async function verifyPortableBundleDirectory(
  root: string,
  input: {
    readonly trustPolicy: PortableBundleTrustPolicy;
    readonly context: PortableBundleContext;
  },
): Promise<PortableBundleDirectoryResult> {
  const { trustPolicy, context } = input;
  const limits = {
    maxManifestBytes:
      context.maxManifestBytes ?? PORTABLE_BUNDLE_DEFAULT_LIMITS.maxManifestBytes,
    maxEnvelopeBytes:
      context.maxEnvelopeBytes ?? PORTABLE_BUNDLE_DEFAULT_LIMITS.maxEnvelopeBytes,
    maxFiles: context.maxFiles ?? PORTABLE_BUNDLE_DEFAULT_LIMITS.maxFiles,
  };

  // Each document is read up to one byte past its limit, so the verifier
  // reports `manifest_too_large`/`envelope_too_large` in contract order.
  let manifest: Uint8Array;
  let envelope: Uint8Array;
  try {
    manifest = await readBounded(
      join(root, PORTABLE_BUNDLE_MANIFEST_FILENAME),
      limits.maxManifestBytes,
    );
    envelope = await readBounded(
      join(root, PORTABLE_BUNDLE_ENVELOPE_FILENAME),
      limits.maxEnvelopeBytes,
    );
  } catch (error) {
    return rejected(reasonOf(error), null);
  }

  const first = await verifyPortableBundle({ manifest, envelope, trustPolicy, context });
  if (!first.ok) {
    return { verification: first, manifest: null, files: null };
  }

  const parsed = parsePortableBundleManifest(
    manifest,
    context.understoodExtensions ?? [],
  );
  const declared = new Map<string, number>();
  for (const item of parsed.files as Array<Record<string, unknown>>) {
    declared.set(item.path as string, item.size_bytes as number);
  }

  let paths: string[];
  try {
    paths = await walk(root, Math.max(limits.maxFiles * 2, MAX_WALK_ENTRIES_FLOOR));
  } catch (error) {
    return rejected(reasonOf(error), first);
  }

  const tree = new VerifiedBundleTree(root, paths, declared);
  let verification: PortableBundleVerification;
  try {
    verification = await verifyPortableBundle({
      manifest,
      envelope,
      trustPolicy,
      context,
      files: tree,
    });
  } catch (error) {
    if (error instanceof PortableBundleError) {
      throw error;
    }
    return rejected(reasonOf(error), first);
  }
  if (!verification.ok || !verification.filesVerified) {
    return { verification, manifest: null, files: null };
  }
  return { verification, manifest: parsed, files: tree.verifiedBytes() };
}

/** Lazily reads declared files; undeclared files are listed but never read. */
class VerifiedBundleTree implements ReadonlyMap<string, Uint8Array> {
  private readonly paths: readonly string[];
  private readonly cache = new Map<string, Uint8Array>();
  private static readonly EMPTY = new Uint8Array(0);

  constructor(
    private readonly root: string,
    paths: readonly string[],
    private readonly declared: ReadonlyMap<string, number>,
  ) {
    this.paths = paths;
  }

  get size(): number {
    return this.paths.length;
  }

  has(path: string): boolean {
    return this.paths.includes(path);
  }

  get(path: string): Uint8Array | undefined {
    if (!this.paths.includes(path)) {
      return undefined;
    }
    const expected = this.declared.get(path);
    if (expected === undefined) {
      return VerifiedBundleTree.EMPTY;
    }
    const cached = this.cache.get(path);
    if (cached !== undefined) {
      return cached;
    }
    const bytes = readDeclaredSync(join(this.root, ...path.split("/")), expected);
    if (bytes.length === expected) {
      this.cache.set(path, bytes);
    }
    return bytes;
  }

  verifiedBytes(): ReadonlyMap<string, Uint8Array> {
    return new Map(this.cache);
  }

  forEach(
    callback: (value: Uint8Array, key: string, map: ReadonlyMap<string, Uint8Array>) => void,
  ): void {
    for (const path of this.paths) {
      callback(this.get(path) as Uint8Array, path, this);
    }
  }

  *entries(): MapIterator<[string, Uint8Array]> {
    for (const path of this.paths) {
      yield [path, this.get(path) as Uint8Array];
    }
  }

  *keys(): MapIterator<string> {
    yield* this.paths;
  }

  *values(): MapIterator<Uint8Array> {
    for (const path of this.paths) {
      yield this.get(path) as Uint8Array;
    }
  }

  [Symbol.iterator](): MapIterator<[string, Uint8Array]> {
    return this.entries();
  }
}

class Refusal extends Error {
  constructor(readonly reason: PortableBundleReason) {
    super(reason);
  }
}

function reasonOf(error: unknown): PortableBundleReason {
  return error instanceof Refusal ? error.reason : "file_missing";
}

function rejected(
  reason: PortableBundleReason,
  previous: PortableBundleVerification | null,
): PortableBundleDirectoryResult {
  return {
    verification: {
      ok: false,
      reason,
      contentId: previous?.contentId ?? null,
      validSignatures: previous?.validSignatures ?? 0,
      eligibleRuntimes: previous?.eligibleRuntimes ?? [],
      filesChecked: 0,
      filesVerified: false,
    },
    manifest: null,
    files: null,
  };
}

async function readBounded(path: string, limit: number): Promise<Uint8Array> {
  const handle = await open(path, fsConstants.O_RDONLY | NO_FOLLOW).catch(() => {
    throw new Refusal("file_missing");
  });
  try {
    const info = await handle.stat();
    if (!info.isFile()) {
      throw new Refusal("file_missing");
    }
    const capacity = Math.min(info.size, limit) + 1;
    const buffer = new Uint8Array(capacity);
    let length = 0;
    while (length < capacity) {
      const { bytesRead } = await handle.read(buffer, length, capacity - length, null);
      if (bytesRead === 0) {
        break;
      }
      length += bytesRead;
    }
    return buffer.slice(0, length);
  } finally {
    await handle.close();
  }
}

async function walk(root: string, maxEntries: number): Promise<string[]> {
  const rootInfo = await lstat(root).catch(() => {
    throw new Refusal("file_missing");
  });
  if (rootInfo.isSymbolicLink() || !rootInfo.isDirectory()) {
    throw new Refusal("file_missing");
  }
  const paths: string[] = [];
  let seen = 0;
  const visit = async (directory: string, prefix: string): Promise<void> => {
    const entries = await readdir(directory, { withFileTypes: true }).catch(() => {
      throw new Refusal("file_missing");
    });
    entries.sort((left, right) => (left.name < right.name ? -1 : 1));
    for (const entry of entries) {
      seen += 1;
      if (seen > maxEntries) {
        throw new Refusal("size_limit_exceeded");
      }
      const relative = prefix === "" ? entry.name : `${prefix}/${entry.name}`;
      const absolute = join(directory, entry.name);
      const info = await lstat(absolute).catch(() => {
        throw new Refusal("file_missing");
      });
      if (info.isSymbolicLink()) {
        throw new Refusal("path_unsafe");
      }
      if (info.isDirectory()) {
        await visit(absolute, relative);
      } else if (info.isFile()) {
        paths.push(relative);
      } else {
        throw new Refusal("path_unsafe");
      }
    }
  };
  await visit(root, "");
  return paths;
}

function readDeclaredSync(path: string, expected: number): Uint8Array {
  // Synchronous because the verifier's map lookup is synchronous. The size is
  // checked on the open descriptor before reading, so a file whose size does
  // not match is never read into memory.
  const descriptor = openSync(path, fsConstants.O_RDONLY | NO_FOLLOW);
  try {
    const info = fstatSync(descriptor);
    if (!info.isFile()) {
      throw new Refusal("path_unsafe");
    }
    if (info.size !== expected) {
      return new Uint8Array(expected === 0 ? 1 : 0);
    }
    const buffer = new Uint8Array(expected);
    let length = 0;
    while (length < expected) {
      const bytesRead = readSync(descriptor, buffer, length, expected - length, null);
      if (bytesRead === 0) {
        break;
      }
      length += bytesRead;
    }
    return length === expected ? buffer : buffer.slice(0, length);
  } finally {
    closeSync(descriptor);
  }
}
