import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  PORTABLE_BUNDLE_PAYLOAD_TYPE,
  PORTABLE_BUNDLE_REASONS,
  PortableBundleError,
  canonicalJson,
  parsePortableBundleManifest,
  parsePortableBundleTrustPolicy,
  portableBundleContentId,
  portableBundleSigningInput,
  verifyPortableBundle,
  type PortableBundleContext,
  type PortableBundleTarget,
} from "../../js/openmedkit-web/src/index";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const vectorsPath = join(rootDir, "tests", "fixtures", "portable_bundle", "vectors.json");

type Vectors = {
  synthetic_only: boolean;
  test_only_keys: boolean;
  payload_type: string;
  reason_codes: string[];
  golden: { content_id: string; signing_input_sha256: string };
  manifests: Record<string, { text?: string; base?: string; at?: number; delete?: number; insert?: string }>;
  envelopes: Record<string, string>;
  contexts: Record<string, RawContext>;
  trust_policies: Record<string, unknown>;
  trees: Record<string, { path: string; text: string }[]>;
  cases: RawCase[];
  canonicalization: { id: string; input: string; canonical: string }[];
  canonicalization_rejects: { id: string; input: string }[];
  ed25519_rfc8032: { name: string; public_hex: string; message_hex: string; signature_hex: string }[];
};

type RawTarget = {
  platform: string;
  sdk_name: string;
  sdk_version: string;
  runtimes: string[];
  os_version: string | null;
  policy_version: number | null;
  memory_bytes: number | null;
  storage_bytes: number | null;
};

type RawContext = {
  now: string;
  target: RawTarget | null;
  sequence_floors: Record<string, number>;
  understood_extensions: string[];
  max_manifest_bytes: number;
  max_envelope_bytes: number;
  max_total_bytes: number;
  max_files: number;
};

type RawCase = {
  id: string;
  manifest: string;
  envelope: string;
  context: string;
  trust_policy: string;
  tree: string | null;
  tree_remove: string[];
  tree_set: { path: string; text: string }[];
  expected: {
    ok: boolean;
    reason: string | null;
    content_id: string | null;
    valid_signatures: number;
    eligible_runtimes: number[];
    files_checked: number;
    files_verified: boolean;
  };
};

const encoder = new TextEncoder();
const vectorsPromise = readFile(vectorsPath, "utf8").then((text) => JSON.parse(text) as Vectors);

function manifestText(vectors: Vectors, key: string): string {
  const entry = vectors.manifests[key];
  assert.ok(entry, key);
  if (entry.text !== undefined) return entry.text;
  const base = manifestText(vectors, entry.base as string);
  const at = entry.at as number;
  return base.slice(0, at) + (entry.insert as string) + base.slice(at + (entry.delete as number));
}

function context(raw: RawContext): PortableBundleContext {
  const target: PortableBundleTarget | null =
    raw.target === null
      ? null
      : {
          platform: raw.target.platform,
          sdkName: raw.target.sdk_name,
          sdkVersion: raw.target.sdk_version,
          runtimes: raw.target.runtimes,
          osVersion: raw.target.os_version,
          policyVersion: raw.target.policy_version,
          memoryBytes: raw.target.memory_bytes,
          storageBytes: raw.target.storage_bytes,
        };
  return {
    now: raw.now,
    target,
    sequenceFloors: raw.sequence_floors,
    understoodExtensions: raw.understood_extensions,
    maxManifestBytes: raw.max_manifest_bytes,
    maxEnvelopeBytes: raw.max_envelope_bytes,
    maxTotalBytes: raw.max_total_bytes,
    maxFiles: raw.max_files,
  };
}

function tree(vectors: Vectors, item: RawCase): Map<string, Uint8Array> | null {
  if (item.tree === null) return null;
  const files = new Map<string, Uint8Array>();
  for (const entry of vectors.trees[item.tree] ?? []) files.set(entry.path, encoder.encode(entry.text));
  for (const path of item.tree_remove) files.delete(path);
  for (const entry of item.tree_set) files.set(entry.path, encoder.encode(entry.text));
  return files;
}

async function run(vectors: Vectors, item: RawCase) {
  const result = await verifyPortableBundle({
    manifest: encoder.encode(manifestText(vectors, item.manifest)),
    envelope: encoder.encode(vectors.envelopes[item.envelope] as string),
    trustPolicy: parsePortableBundleTrustPolicy(vectors.trust_policies[item.trust_policy]),
    context: context(vectors.contexts[item.context] as RawContext),
    files: tree(vectors, item),
  });
  return {
    ok: result.ok,
    reason: result.reason,
    content_id: result.contentId,
    valid_signatures: result.validSignatures,
    eligible_runtimes: [...result.eligibleRuntimes],
    files_checked: result.filesChecked,
    files_verified: result.filesVerified,
  };
}

function hexBytes(text: string): Uint8Array {
  return Uint8Array.from(text.match(/../g) ?? [], (pair) => Number.parseInt(pair, 16));
}

test("portable bundle vectors are synthetic and share the reason vocabulary", async () => {
  const vectors = await vectorsPromise;
  assert.equal(vectors.synthetic_only, true);
  assert.equal(vectors.test_only_keys, true);
  assert.equal(vectors.payload_type, PORTABLE_BUNDLE_PAYLOAD_TYPE);
  assert.deepEqual(vectors.reason_codes, [...PORTABLE_BUNDLE_REASONS]);
});

test("every golden portable bundle case matches the Python reference", async () => {
  const vectors = await vectorsPromise;
  assert.ok(vectors.cases.length > 50);
  for (const item of vectors.cases) {
    assert.deepEqual(await run(vectors, item), item.expected, item.id);
  }
});

test("golden content identity and signing input match", async () => {
  const vectors = await vectorsPromise;
  const golden = encoder.encode(manifestText(vectors, "golden"));
  assert.equal(await portableBundleContentId(golden), vectors.golden.content_id);
  const digest = await globalThis.crypto.subtle.digest(
    "SHA-256",
    portableBundleSigningInput(golden),
  );
  const hex = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0"));
  assert.equal(hex.join(""), vectors.golden.signing_input_sha256);
});

test("canonicalization vectors match RFC 8785 output", async () => {
  const vectors = await vectorsPromise;
  for (const item of vectors.canonicalization) {
    assert.equal(canonicalJson(JSON.parse(item.input)), item.canonical, item.id);
  }
  for (const item of vectors.canonicalization_rejects) {
    assert.throws(
      () => canonicalJson(JSON.parse(item.input)),
      (error: unknown) =>
        error instanceof PortableBundleError && error.reason === "manifest_not_canonical",
      item.id,
    );
  }
});

test("unknown non-critical extensions round-trip byte for byte", async () => {
  const vectors = await vectorsPromise;
  const text = manifestText(vectors, "golden");
  const value = parsePortableBundleManifest(encoder.encode(text));
  assert.deepEqual(value.extensions, { "org.openmed.test.note": { kept: true, revision: 1 } });
  assert.equal(canonicalJson(value), text);
  assert.throws(
    () => parsePortableBundleManifest(encoder.encode(manifestText(vectors, "valid_critical_extension_understood"))),
    (error: unknown) =>
      error instanceof PortableBundleError && error.reason === "unknown_critical_extension",
  );
});

test("RFC 8032 vectors verify through WebCrypto Ed25519", async () => {
  const vectors = await vectorsPromise;
  for (const item of vectors.ed25519_rfc8032) {
    const key = await globalThis.crypto.subtle.importKey(
      "raw",
      hexBytes(item.public_hex),
      { name: "Ed25519" },
      false,
      ["verify"],
    );
    assert.equal(
      await globalThis.crypto.subtle.verify(
        { name: "Ed25519" },
        key,
        hexBytes(item.signature_hex),
        hexBytes(item.message_hex),
      ),
      true,
      item.name,
    );
  }
});

test("portable bundle verification never touches the network", async () => {
  const vectors = await vectorsPromise;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (() => {
    throw new Error("network access attempted");
  }) as typeof fetch;
  try {
    for (const item of vectors.cases) {
      assert.deepEqual(await run(vectors, item), item.expected, item.id);
    }
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("caller input errors carry only reason codes", async () => {
  const vectors = await vectorsPromise;
  assert.throws(
    () => parsePortableBundleTrustPolicy({ schema: "other" }),
    (error: unknown) => error instanceof PortableBundleError && error.message === "trust_policy_invalid",
  );
  const golden = encoder.encode(manifestText(vectors, "golden"));
  await assert.rejects(
    verifyPortableBundle({
      manifest: golden,
      envelope: encoder.encode(vectors.envelopes.golden as string),
      trustPolicy: parsePortableBundleTrustPolicy(vectors.trust_policies.default),
      context: { now: "yesterday" },
    }),
    (error: unknown) => error instanceof PortableBundleError && error.message === "context_invalid",
  );
});
