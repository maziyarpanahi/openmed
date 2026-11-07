import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  DEFAULT_PROVIDERS,
  DeviceCapabilityError,
  EXECUTION_EXTENSION,
  EXECUTION_PLAN_REASONS,
  EXECUTION_RESOLVER_VERSION,
  ExecutionConstraintsError,
  FORMAT_PROVIDERS,
  MODEL_TIERS,
  NETWORK_STATES,
  PLATFORM_PROVIDERS,
  PROVIDER_ACCELERATORS,
  budgetBucket,
  canonicalJson,
  coarseOsVersion,
  deviceCapabilityProfileDigest,
  executionConstraintsDigest,
  executionPlanInvalidation,
  parseDeviceCapabilityProfile,
  parseExecutionConstraints,
  probeWebDeviceCapabilities,
  resolveExecutionPlan,
  webDeviceCapabilityProfile,
  type DeviceCapabilityProfile,
  type ExecutionPlan,
  type PortableBundleReason,
  type PortableBundleVerification,
  type ResolveExecutionPlanInput,
} from "../../js/openmedkit-web/src/index";
import {
  BUDGET_LADDER,
  CPU_ARCHITECTURES,
  DEVICE_FEATURES,
} from "../../js/openmedkit-web/src/device-capabilities";
import {
  EXECUTION_PLAN_ASSUMPTIONS,
  EXECUTION_STEP_ROLES,
  PLAN_INVALIDATION_COMPONENTS,
} from "../../js/openmedkit-web/src/execution-plan";

const rootDir = fileURLToPath(new URL("../..", import.meta.url));
const vectorsPath = join(rootDir, "tests", "fixtures", "device_capabilities", "vectors.json");

type RawVerification = {
  ok: boolean;
  reason: string | null;
  content_id: string | null;
  valid_signatures: number;
  eligible_runtimes: number[];
  files_checked: number;
  files_verified: boolean;
};

type RawCase = {
  id: string;
  manifest: string;
  verification: RawVerification;
  profile: string;
  constraints: string;
  expected: ExecutionPlan;
};

type Vectors = {
  identity_sentinels: string[];
  manifests: Record<string, { content_id: string; text: string }>;
  profiles: Record<string, { digest: string; document: DeviceCapabilityProfile }>;
  constraints: Record<string, { digest: string; document: unknown }>;
  normalization: { id: string; kit: string; probe: Record<string, unknown>; expected: unknown }[];
  cases: RawCase[];
  invalidations: { plan: string; inputs: string; expected: string[] }[];
  vocabulary: Record<string, unknown>;
};

const vectors = JSON.parse(readFileSync(vectorsPath, "utf8")) as Vectors;
const casesById = new Map(vectors.cases.map((item) => [item.id, item]));
const encoder = new TextEncoder();

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

function inputs(item: RawCase): ResolveExecutionPlanInput {
  const manifest = vectors.manifests[item.manifest];
  const profile = vectors.profiles[item.profile];
  const constraints = vectors.constraints[item.constraints];
  assert.ok(manifest && profile && constraints, item.id);
  return {
    manifest: encoder.encode(manifest.text),
    verification: verification(item.verification),
    profile: parseDeviceCapabilityProfile(profile.document),
    constraints: parseExecutionConstraints(constraints.document),
  };
}

function webOptions(probe: Record<string, unknown>) {
  return {
    sdkVersion: probe.sdk_version as string,
    webgpuShaderF16: (probe.webgpu_shader_f16 as boolean | undefined) ?? false,
    runtimes: (probe.runtimes as string[] | undefined) ?? ["onnx"],
    memoryBudgetBytes: (probe.memory_budget_bytes as number | undefined) ?? null,
    storageFreeBytes: (probe.storage_free_bytes as number | undefined) ?? null,
    maxSequenceLength: (probe.max_sequence_length as number | undefined) ?? null,
    operatorSets: (probe.operator_sets as Record<string, number> | undefined) ?? {},
    network: (probe.network as "offline" | "online" | "unknown" | undefined) ?? "unknown",
  };
}

function ortProfile(probe: Record<string, unknown>) {
  return {
    webgpu: (probe.webgpu as boolean | undefined) ?? false,
    wasm: (probe.wasm as boolean | undefined) ?? false,
    wasmSimd: (probe.wasm_simd as boolean | undefined) ?? false,
    sharedArrayBuffer: (probe.shared_array_buffer as boolean | undefined) ?? false,
    crossOriginIsolated: (probe.cross_origin_isolated as boolean | undefined) ?? false,
    hardwareConcurrency: (probe.hardware_concurrency as number | undefined) ?? 1,
  };
}

function assertNoSentinels(value: unknown, label: string): void {
  const text = JSON.stringify(value);
  for (const sentinel of vectors.identity_sentinels) {
    assert.ok(!text.includes(sentinel), `${label} leaked an identity value`);
  }
}

test("web adapter normalizes every shared web probe to the golden profile", () => {
  const web = vectors.normalization.filter((item) => item.kit === "web");
  assert.ok(web.length >= 5);
  for (const item of web) {
    const profile = webDeviceCapabilityProfile(ortProfile(item.probe), webOptions(item.probe));
    assert.equal(canonicalJson(profile), canonicalJson(item.expected), item.id);
    assertNoSentinels(profile, item.id);
  }
});

test("browser probe ignores user agent and GPU adapter strings", async () => {
  const sentinel = vectors.identity_sentinels;
  const adapter = {
    features: new Set(["shader-f16"]),
    info: { vendor: sentinel[2], architecture: sentinel[4], description: sentinel[4] },
    requestDevice: () => ({}),
  };
  const profile = await probeWebDeviceCapabilities({
    globalScope: {
      WebAssembly: { validate: () => true },
      SharedArrayBuffer: ArrayBuffer as unknown as typeof SharedArrayBuffer,
      crossOriginIsolated: true,
      navigator: {
        gpu: { requestAdapter: async () => adapter },
        hardwareConcurrency: 8,
        userAgent: sentinel[3],
        onLine: true,
      } as never,
    },
    sdkVersion: "4.0.0",
    memoryBudgetBytes: 3_000_000_000,
    storageFreeBytes: 10_000_000_000,
    operatorSets: { onnx: 21 },
  });
  const expected = vectors.profiles.web_full;
  assert.ok(expected);
  assert.equal(canonicalJson(profile), canonicalJson(expected.document));
  assertNoSentinels(profile, "probe");
});

test("profile documents round trip to identical bytes and digests", async () => {
  for (const [name, entry] of Object.entries(vectors.profiles)) {
    const profile = parseDeviceCapabilityProfile(entry.document);
    assert.equal(canonicalJson(profile), canonicalJson(entry.document), name);
    assert.equal(await deviceCapabilityProfileDigest(profile), entry.digest, name);
    assertNoSentinels(profile, name);
  }
});

test("constraint documents round trip to identical bytes and digests", async () => {
  for (const [name, entry] of Object.entries(vectors.constraints)) {
    const constraints = parseExecutionConstraints(entry.document);
    assert.equal(canonicalJson(constraints), canonicalJson(entry.document), name);
    assert.equal(await executionConstraintsDigest(constraints), entry.digest, name);
  }
});

test("resolver reproduces every golden plan byte for byte", async () => {
  for (const item of vectors.cases) {
    const plan = await resolveExecutionPlan(inputs(item));
    assert.equal(canonicalJson(plan), canonicalJson(item.expected), item.id);
    assertNoSentinels(plan, item.id);
  }
});

test("plan invalidation reports the same changed components", async () => {
  for (const item of vectors.invalidations) {
    const planCase = casesById.get(item.plan);
    const inputsCase = casesById.get(item.inputs);
    assert.ok(planCase && inputsCase);
    const plan = await resolveExecutionPlan(inputs(planCase));
    const changed = await executionPlanInvalidation({ plan, ...inputs(inputsCase) });
    assert.deepEqual(changed, item.expected, `${item.plan} -> ${item.inputs}`);
  }
});

test("resolution never touches the network", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = (() => {
    throw new Error("network access attempted");
  }) as typeof fetch;
  try {
    for (const item of vectors.cases) {
      const plan = await resolveExecutionPlan(inputs(item));
      assert.equal(plan.plan_key, item.expected.plan_key, item.id);
    }
  } finally {
    globalThis.fetch = original;
  }
});

test("vocabulary matches the shared vectors", () => {
  const vocabulary = vectors.vocabulary;
  const sorted = (value: Readonly<Record<string, readonly string[]>>) =>
    Object.fromEntries(Object.entries(value).map(([key, items]) => [key, [...items].sort()]));
  assert.deepEqual(vocabulary.reason_codes, [...EXECUTION_PLAN_REASONS]);
  assert.deepEqual(vocabulary.provider_accelerators, { ...PROVIDER_ACCELERATORS });
  assert.deepEqual(vocabulary.platform_providers, sorted(PLATFORM_PROVIDERS));
  assert.deepEqual(vocabulary.format_providers, sorted(FORMAT_PROVIDERS));
  assert.deepEqual(vocabulary.default_providers, { ...DEFAULT_PROVIDERS });
  assert.deepEqual(vocabulary.features, [...DEVICE_FEATURES]);
  assert.deepEqual(vocabulary.cpu_architectures, [...CPU_ARCHITECTURES]);
  assert.deepEqual(vocabulary.model_tiers, [...MODEL_TIERS]);
  assert.deepEqual(vocabulary.network_states, [...NETWORK_STATES]);
  assert.deepEqual(vocabulary.step_roles, [...EXECUTION_STEP_ROLES]);
  assert.deepEqual(vocabulary.assumptions, [...EXECUTION_PLAN_ASSUMPTIONS]);
  assert.deepEqual(vocabulary.invalidation_components, [...PLAN_INVALIDATION_COMPONENTS]);
  assert.equal(vocabulary.execution_extension, EXECUTION_EXTENSION);
  assert.equal(vocabulary.resolver_version, EXECUTION_RESOLVER_VERSION);
  assert.deepEqual(vocabulary.budget_ladder, {
    max_bytes: BUDGET_LADDER.maxBytes,
    min_bytes: BUDGET_LADDER.minBytes,
  });
});

test("budget ladder and OS versions stay coarse", () => {
  assert.equal(budgetBucket(null), null);
  assert.equal(budgetBucket(BUDGET_LADDER.minBytes - 1), 0);
  assert.equal(budgetBucket(150 * 1024 * 1024), 128 * 1024 * 1024);
  assert.equal(budgetBucket(3_000_000_000), 2 ** 31);
  assert.equal(budgetBucket(2 ** 45), 2 ** 40);
  assert.throws(() => budgetBucket(-1), DeviceCapabilityError);
  assert.equal(coarseOsVersion("17.4.1"), "17.4");
  assert.throws(() => coarseOsVersion("17.4-beta"), DeviceCapabilityError);
});

test("closed documents reject identity-bearing members", () => {
  const base = vectors.profiles.web_full;
  assert.ok(base);
  const mutations: Record<string, unknown>[] = [
    { device_id: vectors.identity_sentinels[1] },
    { providers: ["cpu"] },
    { sdk_name: vectors.identity_sentinels[0] },
    { memory_budget_bytes: 3_000_000_000 },
    { os_version: "17.4.1" },
    { accelerators: ["npu"] },
    { network: vectors.identity_sentinels[3] },
  ];
  for (const mutation of mutations) {
    assert.throws(
      () => parseDeviceCapabilityProfile({ ...base.document, ...mutation }),
      (error: unknown) => error instanceof DeviceCapabilityError && error.message === "profile_invalid",
    );
  }
  const defaults = vectors.constraints.default;
  assert.ok(defaults);
  for (const mutation of [
    { remote_endpoint: "https://example.invalid" },
    { local_only: "yes" },
    { max_steps: 0 },
    { disallowed_providers: ["cloud"] },
    { revoked_artifacts: ["md5:abc"] },
  ]) {
    assert.throws(
      () => parseExecutionConstraints({ ...(defaults.document as object), ...mutation }),
      (error: unknown) => error instanceof ExecutionConstraintsError && error.message === "constraints_invalid",
    );
  }
});
