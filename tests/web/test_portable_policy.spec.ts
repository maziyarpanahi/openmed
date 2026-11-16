import assert from "node:assert/strict";
import { createHash, createHmac } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  KIT_PARITY_CANONICAL_LABELS,
  PORTABLE_POLICY_RESOURCES,
  PORTABLE_POLICY_SUPPORTED_CAPABILITIES,
  PORTABLE_TAXONOMY_LABELS,
  PortablePolicyError,
  decidePortablePolicy,
  decodePortablePolicyProfile,
  loadPortablePolicy,
  portablePolicyNames,
  portableSurrogateToken,
  serializeKitParityResult,
  sha256Digest,
  spansToRedactedText,
  type OpenMedSpan,
  type PortablePolicyDetection,
} from "../../js/openmedkit-web/src/index";

// Shared portable policy decision vectors, derived from the Python reference
// engine by scripts/parity/generate_portable_policies.py.
const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const VECTORS_PATH = join(rootDir, "tests", "fixtures", "parity", "portable_policy_vectors.json");
const KIT_DIRS = [
  join(rootDir, "swift", "OpenMedKit", "Sources", "OpenMedKit", "Resources", "policies"),
  join(rootDir, "android", "openmedkit", "src", "main", "assets", "policies"),
];

interface DecisionCase {
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

interface ErrorCase extends Omit<DecisionCase, "expected"> {
  expected: { code: string; field_path: string };
}

interface LoadCase {
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

interface Vectors {
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

let cached: Vectors | undefined;

async function vectors(): Promise<Vectors> {
  cached ??= JSON.parse(await readFile(VECTORS_PATH, "utf8")) as Vectors;
  return cached;
}

function sha256Hex(text: string): string {
  return createHash("sha256").update(Buffer.from(text, "utf8")).digest("hex");
}

function keyFor(fixture: Vectors, name: string | undefined): Uint8Array | undefined {
  if (name === undefined) return undefined;
  const hex = fixture.surrogate_test_keys[name];
  assert.ok(hex !== undefined, `missing synthetic key ${name}`);
  return new Uint8Array(Buffer.from(hex, "hex"));
}

function decide(fixture: Vectors, item: DecisionCase | ErrorCase) {
  const policy = loadPortablePolicy(item.profile);
  const surrogateKey = keyFor(fixture, item.surrogate_key);
  return decidePortablePolicy(policy, item.text, item.detections, {
    ...(item.overrides ? { overrides: item.overrides } : {}),
    ...(surrogateKey ? { surrogateKey } : {}),
  });
}

function patchedManifest(patch: Record<string, unknown>): Record<string, string> {
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

function runLoadCase(item: LoadCase) {
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

test("bundled resources are byte-identical to the Swift and Android copies", async () => {
  const fixture = await vectors();
  const manifest = JSON.parse(PORTABLE_POLICY_RESOURCES["portable_policy_manifest.json"] as string) as {
    profiles: Record<string, string>;
    taxonomy_digest: string;
  };
  assert.equal(fixture.schema, "openmed.portable_policy_vectors");
  assert.equal(fixture.version, 1);
  assert.equal(
    sha256Hex(PORTABLE_POLICY_RESOURCES["portable_policy_manifest.json"] as string),
    fixture.manifest_sha256,
  );
  assert.deepEqual(manifest.profiles, fixture.profiles);
  assert.deepEqual(portablePolicyNames(), Object.keys(fixture.profiles).sort());
  assert.deepEqual([...PORTABLE_POLICY_SUPPORTED_CAPABILITIES], fixture.supported_capabilities);
  for (const [name, text] of Object.entries(PORTABLE_POLICY_RESOURCES)) {
    for (const directory of KIT_DIRS) {
      assert.equal(await readFile(join(directory, name), "utf8"), text, name);
    }
  }
  for (const [name, digest] of Object.entries(manifest.profiles)) {
    assert.equal(sha256Digest(PORTABLE_POLICY_RESOURCES[`${name}.json`] as string), digest, name);
  }
  assert.equal(
    sha256Digest(PORTABLE_POLICY_RESOURCES["portable_policy_taxonomy.json"] as string),
    manifest.taxonomy_digest,
  );
  for (const excluded of Object.keys(fixture.excluded_profiles)) {
    assert.equal(PORTABLE_POLICY_RESOURCES[`${excluded}.json`], undefined, excluded);
  }
  assert.ok(
    fixture.consumers.some(
      (item) => item.surface === "web" && item.test === "tests/web/test_portable_policy.spec.ts",
    ),
  );
});

test("kit-parity vocabulary is the portable taxonomy", () => {
  assert.deepEqual([...KIT_PARITY_CANONICAL_LABELS], [...PORTABLE_TAXONOMY_LABELS]);
  assert.ok(PORTABLE_TAXONOMY_LABELS.includes("DIET_TYPE"));
  assert.ok(PORTABLE_TAXONOMY_LABELS.includes("NUTRITION_TARGET"));
});

test("decision vectors match the Python reference byte for byte", async () => {
  const fixture = await vectors();
  assert.ok(fixture.decision_cases.length >= 25);
  for (const item of fixture.decision_cases) {
    const outcome = decide(fixture, item);
    const actual = serializeKitParityResult(outcome.result);
    assert.equal(actual, item.expected.result_json, `case=${item.id} result bytes differ`);
    assert.equal(sha256Hex(actual), item.expected.result_sha256, item.id);
    assert.equal(
      Buffer.compare(
        Buffer.from(outcome.result.deidentified_text, "utf8"),
        Buffer.from(item.expected.deidentified_text, "utf8"),
      ),
      0,
      `case=${item.id} de-identified UTF-8 bytes differ`,
    );
    assert.deepEqual(outcome.decisions, item.expected.decisions, `case=${item.id} decisions differ`);
    assert.equal(outcome.policyDigest, item.expected.policy_digest, item.id);
  }
});

test("invalid detections, overrides, and keys fail with value-free reasons", async () => {
  const fixture = await vectors();
  for (const item of fixture.error_cases) {
    assert.throws(
      () => decide(fixture, item),
      (error: unknown) => {
        assert.ok(error instanceof PortablePolicyError, item.id);
        assert.equal(error.code, item.expected.code, item.id);
        assert.equal(error.fieldPath, item.expected.field_path, item.id);
        assert.ok(!error.message.includes(item.text) || item.text === "", item.id);
        return true;
      },
    );
  }
});

test("load vectors pin digests, aliases, and fail-closed refusals", async () => {
  const fixture = await vectors();
  for (const item of fixture.load_cases) {
    if (item.expected.code !== undefined) {
      assert.throws(
        () => runLoadCase(item),
        (error: unknown) => {
          assert.ok(error instanceof PortablePolicyError, item.id);
          assert.equal(error.code, item.expected.code, item.id);
          assert.equal(error.fieldPath, item.expected.field_path, item.id);
          return true;
        },
      );
      continue;
    }
    const policy = runLoadCase(item);
    assert.equal(policy.name, item.expected.name, item.id);
    assert.equal(policy.digest, item.expected.digest, item.id);
    assert.deepEqual([...policy.requiredCapabilities], item.expected.required_capabilities, item.id);
  }
});

test("hashing and keyed surrogates agree with node:crypto", async () => {
  const fixture = await vectors();
  for (const text of ["", "abc", "x".repeat(55), "y".repeat(56), "z".repeat(1000), "Léa \u{1F600}"]) {
    assert.equal(sha256Digest(text), `sha256:${sha256Hex(text)}`);
  }
  for (const name of ["synthetic_test_key_a", "synthetic_test_key_long"]) {
    const key = keyFor(fixture, name) as Uint8Array;
    const message = Buffer.concat([
      Buffer.from("openmed.portable_surrogate.v1"),
      Buffer.of(0),
      Buffer.from("PERSON"),
      Buffer.of(0),
      Buffer.from("Ada Quill", "utf8"),
    ]);
    const expected = createHmac("sha256", key).update(message).digest("hex").slice(0, 16);
    assert.equal(portableSurrogateToken(key, "PERSON", "Ada Quill"), expected);
  }
});

test("spansToRedactedText leaves keep spans in place", () => {
  const span = (start: number, end: number, action: OpenMedSpan["action"], replacement: string | null) =>
    ({
      start,
      end,
      action,
      replacement,
      canonical_label: "DATE",
    }) as unknown as OpenMedSpan;
  const text = "on 2026-03-02 and 2026-04-09";
  assert.equal(
    spansToRedactedText(text, [span(3, 13, "keep", null), span(18, 28, "redact", "[DATE]")]),
    "on 2026-03-02 and [DATE]",
  );
});

test("decisions run offline and leak no source text or key material", async () => {
  const fixture = await vectors();
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (() => {
    throw new Error("network disabled in portable policy tests");
  }) as typeof fetch;
  try {
    for (const item of fixture.decision_cases) {
      const outcome = decide(fixture, item);
      const decisions = JSON.stringify(outcome.decisions);
      for (const detection of item.detections) {
        const surface = [...item.text].slice(detection.start, detection.end).join("");
        if (surface.length >= 3) {
          assert.ok(!decisions.includes(surface), `case=${item.id} decision echoes source text`);
        }
      }
      for (const hex of Object.values(fixture.surrogate_test_keys)) {
        assert.ok(!JSON.stringify(outcome).includes(hex), `case=${item.id} echoes key material`);
      }
    }
  } finally {
    globalThis.fetch = originalFetch;
  }
});
