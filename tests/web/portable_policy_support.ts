// Shared helpers for tests/fixtures/parity/portable_policy_vectors.json: the
// fixture's types and the calls that replay a decision, error or load case
// through the web portable policy engine. Used by test_portable_policy.spec.ts
// and the contract-gate reporter.
import assert from "node:assert/strict";

import {
  PORTABLE_POLICY_RESOURCES,
  decidePortablePolicy,
  decodePortablePolicyProfile,
  loadPortablePolicy,
  type PortablePolicyDetection,
} from "../../js/openmedkit-web/src/index";

export interface DecisionCase {
  id: string;
  profile: string;
  text: string;
  detections: PortablePolicyDetection[];
  overrides?: Record<string, string>;
  surrogate_key?: string;
  expected: {
    decisions: unknown[];
    deidentified_text: string;
    policy_digest: string;
    result_json: string;
    result_sha256: string;
  };
}

export interface ErrorCase extends Omit<DecisionCase, "expected"> {
  expected: { code: string; field_path: string };
}

export interface LoadCase {
  id: string;
  profile: string;
  expected_digest?: string;
  manifest_patch?: Record<string, unknown>;
  profile_patch?: Record<string, unknown>;
  expected: {
    code?: string;
    field_path?: string;
    digest?: string;
    name?: string;
    required_capabilities?: string[];
  };
}

export interface Vectors {
  schema: string;
  version: number;
  manifest_sha256: string;
  profiles: Record<string, string>;
  excluded_profiles: Record<string, string>;
  supported_capabilities: string[];
  surrogate_test_keys: Record<string, string>;
  consumers: { surface: string; test: string; gate: string }[];
  decision_cases: DecisionCase[];
  error_cases: ErrorCase[];
  load_cases: LoadCase[];
}

export function keyFor(fixture: Vectors, name: string | undefined): Uint8Array | undefined {
  if (name === undefined) return undefined;
  const hex = fixture.surrogate_test_keys[name];
  assert.ok(hex !== undefined, `missing synthetic key ${name}`);
  return new Uint8Array(Buffer.from(hex, "hex"));
}

export function decide(fixture: Vectors, item: DecisionCase | ErrorCase) {
  const policy = loadPortablePolicy(item.profile);
  const surrogateKey = keyFor(fixture, item.surrogate_key);
  return decidePortablePolicy(policy, item.text, item.detections, {
    ...(item.overrides ? { overrides: item.overrides } : {}),
    ...(surrogateKey ? { surrogateKey } : {}),
  });
}

export function patchedManifest(patch: Record<string, unknown>): Record<string, string> {
  const manifest = JSON.parse(
    PORTABLE_POLICY_RESOURCES["portable_policy_manifest.json"] as string,
  ) as Record<string, unknown>;
  for (const [field, value] of Object.entries(patch)) {
    const current = manifest[field];
    manifest[field] =
      typeof value === "object" && value !== null && typeof current === "object" && current !== null
        ? { ...(current as object), ...(value as object) }
        : value;
  }
  return { ...PORTABLE_POLICY_RESOURCES, "portable_policy_manifest.json": JSON.stringify(manifest) };
}

export function runLoadCase(item: LoadCase) {
  if (item.profile_patch === undefined) {
    return loadPortablePolicy(item.profile, {
      ...(item.expected_digest ? { expectedDigest: item.expected_digest } : {}),
      ...(item.manifest_patch ? { resources: patchedManifest(item.manifest_patch) } : {}),
    });
  }
  const policy = loadPortablePolicy(item.profile);
  const document = JSON.parse(PORTABLE_POLICY_RESOURCES[`${policy.name}.json`] as string) as Record<
    string,
    unknown
  >;
  return decodePortablePolicyProfile(
    { ...document, ...item.profile_patch },
    { name: policy.name, digest: policy.digest, taxonomy: policy.taxonomy },
  );
}
