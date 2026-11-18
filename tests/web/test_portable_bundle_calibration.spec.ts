import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  DECISION_CALIBRATION_EXTENSION,
  DECISION_CALIBRATION_EXTENSION_VERSION,
  DECISION_CALIBRATION_REASONS,
  DecisionCalibrationError,
  bundleCalibrationProfileJson,
  decisionCalibrationRuntimeDigest,
  parsePortableBundleTrustPolicy,
  portableBundleContentId,
  readDecisionCalibration,
  resolveDecisionCalibration,
  verifyPortableBundle,
  type DecisionCalibrationRequest,
  type PortableBundleReason,
  type PortableBundleVerification,
} from "../../js/openmedkit-web/src/index";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const vectorsPath = join(rootDir, "tests", "fixtures", "portable_bundle_calibration", "vectors.json");

type RawVerification = {
  ok: boolean;
  reason: string | null;
  content_id: string | null;
  valid_signatures: number;
  eligible_runtimes: number[];
  files_checked: number;
  files_verified: boolean;
};

type Vectors = {
  synthetic_only: boolean;
  test_only_keys: boolean;
  extension: string;
  extension_version: number;
  reason_codes: string[];
  now: string;
  target: {
    platform: string;
    sdk_name: string;
    sdk_version: string;
    runtimes: string[];
    os_version: string | null;
    policy_version: number | null;
    memory_bytes: number | null;
    storage_bytes: number | null;
  };
  trust_policy: unknown;
  tree: { path: string; text: string }[];
  golden: { content_id: string; profile_digests: string[]; runtime_digests: Record<string, string> };
  manifests: Record<string, { text?: string; base?: string; at?: number; delete?: number; insert?: string }>;
  verification_cases: {
    id: string;
    manifest: string;
    envelope: string;
    understood_extensions: string[];
    tree_set: { path: string; text: string }[];
    expected: RawVerification;
  }[];
  read_cases: {
    id: string;
    manifest: string;
    verification: RawVerification;
    expected: { ok: boolean; reason: string | null; content_id: string; profiles: string[] };
  }[];
  resolve_cases: {
    id: string;
    manifest: string;
    verification: RawVerification;
    request: Record<string, string | null>;
    expected:
      | { error: string }
      | { ok: boolean; reason: string | null; content_id: string; profile_digest: string | null };
  }[];
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

function manifest(vectors: Vectors, key: string): Uint8Array {
  return encoder.encode(manifestText(vectors, key));
}

function verification(raw: RawVerification): PortableBundleVerification {
  return {
    ok: raw.ok,
    reason: raw.reason as PortableBundleReason | null,
    contentId: raw.content_id,
    validSignatures: raw.valid_signatures,
    eligibleRuntimes: raw.eligible_runtimes,
    filesChecked: raw.files_checked,
    filesVerified: raw.files_verified,
  };
}

function request(raw: Record<string, string | null>): DecisionCalibrationRequest {
  return {
    calibrationId: raw.calibration_id as string,
    runtimeId: raw.runtime_id as string,
    decisionSchemaVersion: raw.decision_schema_version as string,
    modelId: raw.model_id,
    modelRevision: raw.model_revision,
    runtimeDigest: raw.runtime_digest,
    scopeKind: raw.scope_kind,
    scopeDigest: raw.scope_digest,
    profileDigest: raw.profile_digest,
  };
}

test("calibration vectors share the reason vocabulary", async () => {
  const vectors = await vectorsPromise;
  assert.equal(vectors.synthetic_only, true);
  assert.equal(vectors.test_only_keys, true);
  assert.equal(vectors.extension, DECISION_CALIBRATION_EXTENSION);
  assert.equal(vectors.extension_version, DECISION_CALIBRATION_EXTENSION_VERSION);
  assert.deepEqual(vectors.reason_codes, [...DECISION_CALIBRATION_REASONS]);
});

test("verification cases match the Python reference", async () => {
  const vectors = await vectorsPromise;
  const trustPolicy = parsePortableBundleTrustPolicy(vectors.trust_policy);
  assert.ok(vectors.verification_cases.length > 10);
  for (const item of vectors.verification_cases) {
    const files = new Map<string, Uint8Array>();
    for (const entry of [...vectors.tree, ...item.tree_set]) {
      files.set(entry.path, encoder.encode(entry.text));
    }
    const result = await verifyPortableBundle({
      manifest: manifest(vectors, item.manifest),
      envelope: encoder.encode(item.envelope),
      trustPolicy,
      context: {
        now: vectors.now,
        target: {
          platform: vectors.target.platform,
          sdkName: vectors.target.sdk_name,
          sdkVersion: vectors.target.sdk_version,
          runtimes: vectors.target.runtimes,
          osVersion: vectors.target.os_version,
          policyVersion: vectors.target.policy_version,
          memoryBytes: vectors.target.memory_bytes,
          storageBytes: vectors.target.storage_bytes,
        },
        understoodExtensions: item.understood_extensions,
      },
      files,
    });
    assert.deepEqual(
      {
        ok: result.ok,
        reason: result.reason,
        content_id: result.contentId,
        valid_signatures: result.validSignatures,
        eligible_runtimes: result.eligibleRuntimes,
        files_checked: result.filesChecked,
        files_verified: result.filesVerified,
      },
      item.expected,
      item.id,
    );
  }
});

test("read cases match the Python reference", async () => {
  const vectors = await vectorsPromise;
  assert.ok(vectors.read_cases.length > 40);
  for (const item of vectors.read_cases) {
    const result = await readDecisionCalibration(
      manifest(vectors, item.manifest),
      verification(item.verification),
    );
    assert.deepEqual(
      {
        ok: result.ok,
        reason: result.reason,
        content_id: result.contentId,
        profiles: result.profiles.map(bundleCalibrationProfileJson),
      },
      item.expected,
      item.id,
    );
  }
});

test("resolve cases match the Python reference", async () => {
  const vectors = await vectorsPromise;
  assert.ok(vectors.resolve_cases.length > 20);
  for (const item of vectors.resolve_cases) {
    const call = resolveDecisionCalibration(
      manifest(vectors, item.manifest),
      verification(item.verification),
      request(item.request),
    );
    if ("error" in item.expected) {
      const reason = item.expected.error;
      await assert.rejects(
        call,
        (error: unknown) => error instanceof DecisionCalibrationError && error.reason === reason,
        item.id,
      );
      continue;
    }
    const selection = await call;
    assert.deepEqual(
      {
        ok: selection.ok,
        reason: selection.reason,
        content_id: selection.contentId,
        profile_digest: selection.profile?.profileDigest ?? null,
      },
      item.expected,
      item.id,
    );
  }
});

test("golden digests and typed profiles", async () => {
  const vectors = await vectorsPromise;
  const data = manifest(vectors, "golden");
  assert.equal(await portableBundleContentId(data), vectors.golden.content_id);
  for (const [runtimeId, digest] of Object.entries(vectors.golden.runtime_digests)) {
    assert.equal(await decisionCalibrationRuntimeDigest(data, runtimeId), digest);
  }
  assert.equal(await decisionCalibrationRuntimeDigest(data, "tflite-int8"), null);
  const read = await readDecisionCalibration(data, {
    ok: true,
    reason: null,
    contentId: vectors.golden.content_id,
    validSignatures: 1,
    eligibleRuntimes: [1, 2],
    filesChecked: 10,
    filesVerified: true,
  });
  assert.deepEqual(
    read.profiles.map((item) => item.profileDigest),
    vectors.golden.profile_digests,
  );
  const [triage, quantized, screening, labels] = read.profiles;
  assert.deepEqual(triage?.method, { kind: "deterministic_margin" });
  assert.equal(triage?.minimumConfidencePpm, 700000);
  assert.deepEqual(quantized?.method, { kind: "temperature_scaling", temperaturePpm: 1250000 });
  assert.deepEqual(screening?.method, {
    kind: "platt_scaling",
    slopePpm: -1800000,
    interceptPpm: 250000,
  });
  assert.equal(labels?.scopeKind, "labels");
  assert.equal(labels?.evaluationPath, null);
});

test("calibration errors carry only reason codes", async () => {
  const vectors = await vectorsPromise;
  await assert.rejects(
    resolveDecisionCalibration(manifest(vectors, "golden"), verification(vectors.read_cases[0]!.verification), {
      calibrationId: "openmed.synthetic.triage.v1",
      runtimeId: "onnx-fp32",
      decisionSchemaVersion: "1.1.0",
      modelRevision: "SENTINEL-REVISION",
    }),
    (error: unknown) =>
      error instanceof DecisionCalibrationError &&
      error.message === "calibration_request_invalid" &&
      !String(error).includes("SENTINEL"),
  );
});
