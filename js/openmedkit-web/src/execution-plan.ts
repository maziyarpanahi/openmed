/**
 * Deterministic execution plans for signed portable model bundles.
 *
 * Mirrors `openmed/core/execution_plan.py`: the same checks in the same order,
 * producing byte-identical plan documents for the shared vectors in
 * `tests/fixtures/device_capabilities/`. A plan lists only `(runtime,
 * provider)` steps declared by the signed bundle; when nothing qualifies the
 * result is an explicit `unsupported` document. There is never a remote,
 * cloud, or network step. Resolution performs no network access and loads no
 * model.
 */

import {
  MAX_OPERATOR_SET,
  MAX_SEQUENCE_LENGTH,
  PLATFORM_PROVIDERS,
  PROVIDER_ACCELERATORS,
  deviceCapabilityProfileDigest,
  modelTierRank,
  type DeviceCapabilityProfile,
  type ExecutionProvider,
} from "./device-capabilities";
import {
  canonicalJson,
  parsePortableBundleManifest,
  portableBundleContentId,
  type PortableBundleVerification,
} from "./portable-bundle";

export const EXECUTION_PLAN_SCHEMA = "openmed.execution_plan";
export const EXECUTION_PLAN_SCHEMA_VERSION = 1;
export const EXECUTION_CONSTRAINTS_SCHEMA = "openmed.execution_constraints";
export const EXECUTION_CONSTRAINTS_SCHEMA_VERSION = 1;
export const EXECUTION_RESOLVER_VERSION = 1;
/** Manifest extension that declares providers and qualifications per runtime. */
export const EXECUTION_EXTENSION = "org.openmed.execution";
export const EXECUTION_EXTENSION_VERSION = 1;

/** Providers that can execute each bundle runtime format. */
export const FORMAT_PROVIDERS: Readonly<Record<string, readonly ExecutionProvider[]>> = {
  coreml: ["coreml", "cpu"],
  mlx: ["cpu", "metal"],
  onnx: ["coreml", "cpu", "nnapi", "qnn", "wasm", "webgpu"],
  ort: ["coreml", "cpu", "nnapi", "qnn"],
  pytorch: ["cpu"],
  transformersjs: ["wasm", "webgpu"],
};

/** Baseline providers for a runtime the execution extension does not describe. */
export const DEFAULT_PROVIDERS: Readonly<Record<string, readonly ExecutionProvider[]>> = {
  coreml: ["coreml"],
  mlx: ["metal"],
  onnx: ["cpu", "wasm"],
  ort: ["cpu"],
  pytorch: ["cpu"],
  transformersjs: ["wasm"],
};

/** Stable, value-free reason codes, in evaluation order. */
export const EXECUTION_PLAN_REASONS = [
  "remote_execution_unsupported",
  "bundle_revoked",
  "bundle_unverified",
  "bundle_files_unverified",
  "sdk_incompatible",
  "execution_metadata_invalid",
  "no_eligible_runtime",
  "runtime_not_eligible",
  "artifact_revoked",
  "format_unsupported",
  "platform_unsupported",
  "os_version_unknown",
  "os_version_unsupported",
  "precision_unsupported",
  "precision_disallowed",
  "memory_insufficient",
  "storage_insufficient",
  "model_tier_exceeded",
  "sequence_limit_exceeded",
  "sequence_limit_undeclared",
  "operator_set_unknown",
  "operator_set_unsupported",
  "feature_unavailable",
  "evidence_missing",
  "provider_unavailable",
  "provider_disallowed",
  "energy_policy_excluded",
  "not_declared_fallback",
  "fallback_bound_reached",
] as const;

export type ExecutionPlanReason = (typeof EXECUTION_PLAN_REASONS)[number];
export const EXECUTION_STEP_ROLES = ["selected", "provider_fallback", "artifact_fallback"] as const;
export const EXECUTION_PLAN_ASSUMPTIONS = [
  "device_sequence_limit_unknown",
  "memory_budget_unknown",
  "model_tier_limit_unknown",
  "storage_budget_unknown",
] as const;
export const PLAN_INVALIDATION_COMPONENTS = [
  "bundle",
  "constraints",
  "profile",
  "resolver",
  "verification",
] as const;
export const ENERGY_POSTURES = ["balanced", "low_power"] as const;
export const DEFAULT_MAX_STEPS = 4;
export const MAX_STEPS_LIMIT = 16;
export const MAX_DEADLINE_MS = 86_400_000;

/** Caller constraints document (`execution_constraints`). */
export interface ExecutionConstraints {
  readonly schema: typeof EXECUTION_CONSTRAINTS_SCHEMA;
  readonly schema_version: typeof EXECUTION_CONSTRAINTS_SCHEMA_VERSION;
  /** Must be true; false resolves to `remote_execution_unsupported`. */
  readonly local_only: boolean;
  readonly allowed_precisions: readonly string[] | null;
  readonly disallowed_providers: readonly string[];
  readonly max_memory_bytes: number | null;
  readonly min_sequence_length: number | null;
  readonly energy: (typeof ENERGY_POSTURES)[number];
  readonly required_evidence: readonly string[];
  readonly revoked_artifacts: readonly string[];
  readonly max_steps: number;
  readonly deadline_ms: number | null;
  readonly require_verified_files: boolean;
}

export type ExecutionConstraintsInit = Partial<
  Omit<ExecutionConstraints, "schema" | "schema_version">
>;

export interface ExecutionPlanStep {
  readonly accelerator: string;
  readonly artifact_sha256: string;
  readonly evidence: readonly string[];
  readonly format: string;
  readonly max_sequence_length: number | null;
  readonly min_memory_bytes: number;
  readonly precision: string;
  readonly provider: string;
  readonly role: (typeof EXECUTION_STEP_ROLES)[number];
  readonly runtime_id: string;
  readonly runtime_index: number;
  readonly tier: string | null;
}

export interface ExecutionPlanSkip {
  readonly provider: string | null;
  readonly reason: ExecutionPlanReason;
  readonly runtime_id: string;
}

/** Value-free resolver result (`execution_plan`). */
export interface ExecutionPlan {
  readonly assumptions: readonly string[];
  readonly bounds: {
    readonly deadline_ms: number | null;
    readonly max_memory_bytes: number | null;
    readonly max_steps: number;
  };
  readonly bundle_content_id: string;
  readonly constraints_digest: string;
  readonly local_only: true;
  readonly plan_key: string;
  readonly profile_digest: string;
  readonly reason: ExecutionPlanReason | null;
  readonly resolver_version: number;
  readonly root_runtime: string | null;
  readonly schema: typeof EXECUTION_PLAN_SCHEMA;
  readonly schema_version: typeof EXECUTION_PLAN_SCHEMA_VERSION;
  readonly skipped: readonly ExecutionPlanSkip[];
  readonly status: "planned" | "unsupported";
  readonly steps: readonly ExecutionPlanStep[];
  readonly verification_digest: string;
}

/** Error raised for malformed caller constraints; the message is a code only. */
export class ExecutionConstraintsError extends Error {
  readonly reason = "constraints_invalid" as const;

  constructor() {
    super("constraints_invalid");
    this.name = "ExecutionConstraintsError";
  }
}

const PRECISIONS = ["bf16", "fp16", "fp32", "int2", "int4", "int8"];
const PROVIDERS = Object.keys(PROVIDER_ACCELERATORS);
const TOKEN = /^[a-z0-9][a-z0-9._-]{0,63}$/;
const SHA256 = /^sha256:[0-9a-f]{64}$/;
const CONSTRAINT_KEYS = [
  "allowed_precisions",
  "deadline_ms",
  "disallowed_providers",
  "energy",
  "local_only",
  "max_memory_bytes",
  "max_steps",
  "min_sequence_length",
  "require_verified_files",
  "required_evidence",
  "revoked_artifacts",
  "schema",
  "schema_version",
];
const META_KEYS = [
  "evidence",
  "features",
  "max_sequence_length",
  "min_operator_set",
  "providers",
  "tier",
];

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: JsonRecord, keys: readonly string[]): boolean {
  const own = Object.keys(value);
  return own.length === keys.length && keys.every((key) => own.includes(key));
}

function isUint(value: unknown, maximum = Number.MAX_SAFE_INTEGER): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 && value <= maximum;
}

function constraintSet(value: unknown, valid: (item: string) => boolean): string[] {
  if (!Array.isArray(value)) throw new ExecutionConstraintsError();
  for (const item of value) {
    if (typeof item !== "string" || !valid(item)) throw new ExecutionConstraintsError();
  }
  return [...new Set(value as string[])].sort();
}

/** Validate members (defaults applied) and return the constraints document. */
export function createExecutionConstraints(init: ExecutionConstraintsInit = {}): ExecutionConstraints {
  const value = {
    allowed_precisions: init.allowed_precisions ?? null,
    deadline_ms: init.deadline_ms ?? null,
    disallowed_providers: init.disallowed_providers ?? [],
    energy: init.energy ?? "balanced",
    local_only: init.local_only ?? true,
    max_memory_bytes: init.max_memory_bytes ?? null,
    max_steps: init.max_steps ?? DEFAULT_MAX_STEPS,
    min_sequence_length: init.min_sequence_length ?? null,
    require_verified_files: init.require_verified_files ?? true,
    required_evidence: init.required_evidence ?? [],
    revoked_artifacts: init.revoked_artifacts ?? [],
  };
  if (
    typeof value.local_only !== "boolean" ||
    typeof value.require_verified_files !== "boolean" ||
    !(ENERGY_POSTURES as readonly string[]).includes(value.energy) ||
    (value.max_memory_bytes !== null && !isUint(value.max_memory_bytes)) ||
    (value.min_sequence_length !== null &&
      !(isUint(value.min_sequence_length, MAX_SEQUENCE_LENGTH) && value.min_sequence_length > 0)) ||
    !isUint(value.max_steps, MAX_STEPS_LIMIT) ||
    value.max_steps < 1 ||
    (value.deadline_ms !== null && !(isUint(value.deadline_ms, MAX_DEADLINE_MS) && value.deadline_ms > 0))
  ) {
    throw new ExecutionConstraintsError();
  }
  return {
    allowed_precisions:
      value.allowed_precisions === null
        ? null
        : constraintSet(value.allowed_precisions, (item) => PRECISIONS.includes(item)),
    deadline_ms: value.deadline_ms,
    disallowed_providers: constraintSet(value.disallowed_providers, (item) => PROVIDERS.includes(item)),
    energy: value.energy,
    local_only: value.local_only,
    max_memory_bytes: value.max_memory_bytes,
    max_steps: value.max_steps,
    min_sequence_length: value.min_sequence_length,
    require_verified_files: value.require_verified_files,
    required_evidence: constraintSet(value.required_evidence, (item) => TOKEN.test(item)),
    revoked_artifacts: constraintSet(value.revoked_artifacts, (item) => SHA256.test(item)),
    schema: EXECUTION_CONSTRAINTS_SCHEMA,
    schema_version: EXECUTION_CONSTRAINTS_SCHEMA_VERSION,
  };
}

/** Parse a constraints document strictly; unknown or missing members fail. */
export function parseExecutionConstraints(value: unknown): ExecutionConstraints {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, CONSTRAINT_KEYS) ||
    value.schema !== EXECUTION_CONSTRAINTS_SCHEMA ||
    value.schema_version !== EXECUTION_CONSTRAINTS_SCHEMA_VERSION
  ) {
    throw new ExecutionConstraintsError();
  }
  const members = { ...value };
  delete members.schema;
  delete members.schema_version;
  for (const key of ["allowed_precisions", "deadline_ms", "max_memory_bytes", "min_sequence_length"]) {
    if (members[key] === undefined) throw new ExecutionConstraintsError();
  }
  return createExecutionConstraints(members as unknown as ExecutionConstraintsInit);
}

async function digest(value: unknown): Promise<string> {
  return portableBundleContentId(new TextEncoder().encode(canonicalJson(value)));
}

/** Return the `sha256:` digest of a constraints document. */
export async function executionConstraintsDigest(constraints: ExecutionConstraints): Promise<string> {
  return digest(constraints);
}

/** Return the digest of the verification facts a plan depends on. */
export async function verificationDigest(verification: PortableBundleVerification): Promise<string> {
  return digest({
    content_id: verification.contentId,
    eligible_runtimes: [...verification.eligibleRuntimes],
    files_verified: verification.filesVerified,
    ok: verification.ok,
    reason: verification.reason,
  });
}

/** Return the plan invalidation key for a set of input digests. */
export async function computePlanKey(
  bundleContentId: string,
  profileDigest: string,
  constraintsDigest: string,
  verificationDigestValue: string,
  resolverVersion: number = EXECUTION_RESOLVER_VERSION,
): Promise<string> {
  return digest({
    bundle_content_id: bundleContentId,
    constraints_digest: constraintsDigest,
    profile_digest: profileDigest,
    resolver_version: resolverVersion,
    verification_digest: verificationDigestValue,
  });
}

interface RuntimeMeta {
  readonly providers: readonly string[];
  readonly features: readonly string[];
  readonly evidence: readonly string[];
  readonly maxSequenceLength: number | null;
  readonly minOperatorSet: number | null;
  readonly tier: string | null;
  readonly declared: boolean;
}

class MetadataInvalid extends Error {}

function tokenList(value: unknown, maximum: number): string[] {
  if (!Array.isArray(value) || value.length > maximum) throw new MetadataInvalid();
  for (const item of value) {
    if (typeof item !== "string" || !TOKEN.test(item)) throw new MetadataInvalid();
  }
  if (new Set(value).size !== value.length) throw new MetadataInvalid();
  return value as string[];
}

function optionalBounded(value: unknown, maximum: number): number | null {
  if (value === null) return null;
  if (!isUint(value, maximum) || value === 0) throw new MetadataInvalid();
  return value;
}

function parseExtension(manifest: JsonRecord): Map<string, RuntimeMeta> {
  const runtimes = manifest.runtimes as JsonRecord[];
  const evidenceKinds = new Set((manifest.evidence as JsonRecord[]).map((item) => item.kind as string));
  const ids = runtimes.map((item) => item.id as string);
  const raw = (manifest.extensions as JsonRecord)[EXECUTION_EXTENSION];
  let declared: JsonRecord = {};
  if (raw !== undefined) {
    if (
      !isRecord(raw) ||
      !hasExactKeys(raw, ["runtimes", "schema_version"]) ||
      raw.schema_version !== EXECUTION_EXTENSION_VERSION ||
      !isRecord(raw.runtimes) ||
      !Object.keys(raw.runtimes).every((key) => ids.includes(key))
    ) {
      throw new MetadataInvalid();
    }
    declared = raw.runtimes;
  }
  const metas = new Map<string, RuntimeMeta>();
  for (const runtime of runtimes) {
    const id = runtime.id as string;
    const format = runtime.format as string;
    const entry = Object.prototype.hasOwnProperty.call(declared, id) ? declared[id] : undefined;
    if (entry === undefined) {
      metas.set(id, {
        providers: DEFAULT_PROVIDERS[format] ?? [],
        features: [],
        evidence: [],
        maxSequenceLength: null,
        minOperatorSet: null,
        tier: null,
        declared: false,
      });
      continue;
    }
    if (!isRecord(entry) || !hasExactKeys(entry, META_KEYS)) throw new MetadataInvalid();
    const providers = tokenList(entry.providers, 8);
    const allowed = FORMAT_PROVIDERS[format] ?? [];
    if (providers.length === 0 || !providers.every((item) => (allowed as readonly string[]).includes(item))) {
      throw new MetadataInvalid();
    }
    const features = tokenList(entry.features, 16);
    const evidence = tokenList(entry.evidence, 16);
    if (!evidence.every((item) => evidenceKinds.has(item))) throw new MetadataInvalid();
    const tier = entry.tier;
    if (tier !== null && (typeof tier !== "string" || modelTierRank(tier) < 0)) {
      throw new MetadataInvalid();
    }
    metas.set(id, {
      providers,
      features,
      evidence: [...evidence].sort(),
      maxSequenceLength: optionalBounded(entry.max_sequence_length, MAX_SEQUENCE_LENGTH),
      minOperatorSet: optionalBounded(entry.min_operator_set, MAX_OPERATOR_SET),
      tier: tier as string | null,
      declared: true,
    });
  }
  return metas;
}

function providersOn(meta: RuntimeMeta, platform: string): readonly string[] {
  if (meta.declared) return meta.providers;
  const valid = PLATFORM_PROVIDERS[platform] ?? [];
  return meta.providers.filter((item) => (valid as readonly string[]).includes(item));
}

function versionKey(value: string): number[] {
  const parts = value.split(".").map((part) => Number(part));
  while (parts.length < 4) parts.push(0);
  return parts;
}

function compareVersions(left: string, right: string): number {
  const a = versionKey(left);
  const b = versionKey(right);
  for (let index = 0; index < 4; index += 1) {
    const difference = (a[index] ?? 0) - (b[index] ?? 0);
    if (difference !== 0) return difference < 0 ? -1 : 1;
  }
  return 0;
}

function sdkCompatible(manifest: JsonRecord, profile: DeviceCapabilityProfile): boolean {
  const sdks = (manifest.compatibility as JsonRecord).sdks as JsonRecord[];
  const sdk = sdks.find((item) => item.name === profile.sdk_name);
  if (sdk === undefined) return false;
  const max = sdk.max_exclusive as string | null;
  return (
    compareVersions(profile.sdk_version, sdk.min as string) >= 0 &&
    (max === null || compareVersions(profile.sdk_version, max) < 0)
  );
}

function runtimeReason(
  index: number,
  runtime: JsonRecord,
  meta: RuntimeMeta,
  digests: ReadonlyMap<string, string>,
  verification: PortableBundleVerification,
  profile: DeviceCapabilityProfile,
  constraints: ExecutionConstraints,
  memoryLimit: number | null,
): ExecutionPlanReason | null {
  if (!verification.eligibleRuntimes.includes(index)) return "runtime_not_eligible";
  const paths = [runtime.entrypoint as string, ...(runtime.files as string[])];
  if (paths.some((path) => constraints.revoked_artifacts.includes(digests.get(path) ?? ""))) {
    return "artifact_revoked";
  }
  const format = runtime.format as string;
  if (!profile.runtimes.includes(format)) return "format_unsupported";
  const platform = (runtime.platforms as JsonRecord[]).find((item) => item.platform === profile.platform);
  if (platform === undefined) return "platform_unsupported";
  const low = platform.min_os as string | null;
  const high = platform.max_os_exclusive as string | null;
  if (low !== null || high !== null) {
    if (profile.os_version === null) return "os_version_unknown";
    if (
      (low !== null && compareVersions(profile.os_version, low) < 0) ||
      (high !== null && compareVersions(profile.os_version, high) >= 0)
    ) {
      return "os_version_unsupported";
    }
  }
  const precision = runtime.precision as string;
  if (!profile.precisions.includes(precision)) return "precision_unsupported";
  if (constraints.allowed_precisions !== null && !constraints.allowed_precisions.includes(precision)) {
    return "precision_disallowed";
  }
  if (memoryLimit !== null && (runtime.min_memory_bytes as number) > memoryLimit) {
    return "memory_insufficient";
  }
  if (
    profile.storage_budget_bytes !== null &&
    (runtime.min_storage_bytes as number) > profile.storage_budget_bytes
  ) {
    return "storage_insufficient";
  }
  if (
    meta.tier !== null &&
    profile.max_model_tier !== null &&
    modelTierRank(meta.tier) > modelTierRank(profile.max_model_tier)
  ) {
    return "model_tier_exceeded";
  }
  const required = constraints.min_sequence_length;
  if (required !== null) {
    if (profile.max_sequence_length !== null && profile.max_sequence_length < required) {
      return "sequence_limit_exceeded";
    }
    if (meta.maxSequenceLength === null) return "sequence_limit_undeclared";
    if (meta.maxSequenceLength < required) return "sequence_limit_exceeded";
  }
  if (meta.minOperatorSet !== null) {
    const supported = profile.operator_sets[format];
    if (supported === undefined) return "operator_set_unknown";
    if (supported < meta.minOperatorSet) return "operator_set_unsupported";
  }
  if (!meta.features.every((item) => profile.features.includes(item))) return "feature_unavailable";
  if (!constraints.required_evidence.every((item) => meta.evidence.includes(item))) {
    return "evidence_missing";
  }
  return null;
}

function providerReason(
  provider: string,
  profile: DeviceCapabilityProfile,
  constraints: ExecutionConstraints,
): ExecutionPlanReason | null {
  if (!profile.providers.includes(provider)) return "provider_unavailable";
  if (constraints.disallowed_providers.includes(provider)) return "provider_disallowed";
  if (
    constraints.energy === "low_power" &&
    PROVIDER_ACCELERATORS[provider as ExecutionProvider] === "gpu"
  ) {
    return "energy_policy_excluded";
  }
  return null;
}

function parseManifest(manifest: Uint8Array): JsonRecord | null {
  try {
    const raw = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(manifest)) as unknown;
    const critical = isRecord(raw) && Array.isArray(raw.critical_extensions) ? raw.critical_extensions : [];
    const understood = critical.filter((item): item is string => typeof item === "string");
    return parsePortableBundleManifest(manifest, understood) as JsonRecord;
  } catch {
    return null;
  }
}

export interface ResolveExecutionPlanInput {
  /** Exact bytes of `openmed-bundle.json`. */
  readonly manifest: Uint8Array;
  /** The verifier result for those bytes. */
  readonly verification: PortableBundleVerification;
  readonly profile: DeviceCapabilityProfile;
  readonly constraints?: ExecutionConstraints;
}

/**
 * Resolve an ordered, value-free execution plan or an explicit refusal. The
 * plan follows exactly one declared fallback chain: the first root runtime
 * (in manifest order) whose chain contains a qualified step.
 */
export async function resolveExecutionPlan(input: ResolveExecutionPlanInput): Promise<ExecutionPlan> {
  const { manifest: manifestBytes, verification, profile } = input;
  const constraints = input.constraints ?? createExecutionConstraints();
  const bundleContentId = await portableBundleContentId(manifestBytes);
  const profileDigest = await deviceCapabilityProfileDigest(profile);
  const constraintsDigest = await executionConstraintsDigest(constraints);
  const verificationValue = await verificationDigest(verification);
  const planKey = await computePlanKey(bundleContentId, profileDigest, constraintsDigest, verificationValue);
  const limits = [profile.memory_budget_bytes, constraints.max_memory_bytes].filter(
    (item): item is number => item !== null,
  );
  const memoryLimit = limits.length === 0 ? null : Math.min(...limits);

  const result = (
    reason: ExecutionPlanReason | null,
    root: string | null = null,
    steps: ExecutionPlanStep[] = [],
    skipped: ExecutionPlanSkip[] = [],
    assumptions: Set<string> = new Set(),
  ): ExecutionPlan => ({
    assumptions: [...assumptions].sort(),
    bounds: {
      deadline_ms: constraints.deadline_ms,
      max_memory_bytes: memoryLimit,
      max_steps: constraints.max_steps,
    },
    bundle_content_id: bundleContentId,
    constraints_digest: constraintsDigest,
    local_only: true,
    plan_key: planKey,
    profile_digest: profileDigest,
    reason,
    resolver_version: EXECUTION_RESOLVER_VERSION,
    root_runtime: root,
    schema: EXECUTION_PLAN_SCHEMA,
    schema_version: EXECUTION_PLAN_SCHEMA_VERSION,
    skipped,
    status: reason === null ? "planned" : "unsupported",
    steps,
    verification_digest: verificationValue,
  });

  if (!constraints.local_only) return result("remote_execution_unsupported");
  if (!verification.ok) {
    return result(verification.reason === "bundle_revoked" ? "bundle_revoked" : "bundle_unverified");
  }
  const manifest = parseManifest(manifestBytes);
  if (verification.contentId !== bundleContentId || manifest === null) return result("bundle_unverified");
  if (constraints.require_verified_files && !verification.filesVerified) {
    return result("bundle_files_unverified");
  }
  if (!sdkCompatible(manifest, profile)) return result("sdk_incompatible");
  let metas: Map<string, RuntimeMeta>;
  try {
    metas = parseExtension(manifest);
  } catch (error) {
    if (error instanceof MetadataInvalid) return result("execution_metadata_invalid");
    throw error;
  }

  const runtimes = manifest.runtimes as JsonRecord[];
  const digests = new Map(
    (manifest.files as JsonRecord[]).map((item) => [item.path as string, item.sha256 as string]),
  );
  const byId = new Map(runtimes.map((item) => [item.id as string, item]));
  const exclusions = new Map<string, ExecutionPlanReason>();
  const pairReasons = new Map<string, ExecutionPlanReason>();
  const qualified = new Map<string, string[]>();
  const pairKey = (runtimeId: string, provider: string): string => `${runtimeId}\u0000${provider}`;

  runtimes.forEach((runtime, index) => {
    const id = runtime.id as string;
    const meta = metas.get(id) as RuntimeMeta;
    const reason = runtimeReason(index, runtime, meta, digests, verification, profile, constraints, memoryLimit);
    if (reason !== null) {
      exclusions.set(id, reason);
      return;
    }
    const providers: string[] = [];
    for (const provider of providersOn(meta, profile.platform)) {
      const providerFailure = providerReason(provider, profile, constraints);
      if (providerFailure === null) providers.push(provider);
      else pairReasons.set(pairKey(id, provider), providerFailure);
    }
    qualified.set(id, providers);
  });

  const fallbackTargets = new Set(
    runtimes.map((item) => item.fallback).filter((item): item is string => typeof item === "string"),
  );
  let root: string | null = null;
  let candidates: [string, string][] = [];
  for (const runtime of runtimes) {
    const id = runtime.id as string;
    if (fallbackTargets.has(id)) continue;
    const chain = [id];
    for (;;) {
      const following = (byId.get(chain[chain.length - 1] as string) as JsonRecord).fallback;
      if (typeof following !== "string") break;
      chain.push(following);
    }
    const pairs: [string, string][] = chain.flatMap((item) =>
      (qualified.get(item) ?? []).map((provider): [string, string] => [item, provider]),
    );
    if (pairs.length > 0) {
      root = id;
      candidates = pairs;
      break;
    }
  }

  const used = candidates.slice(0, constraints.max_steps);
  const usedKeys = new Set(used.map(([id, provider]) => pairKey(id, provider)));
  for (const [id, provider] of candidates.slice(constraints.max_steps)) {
    pairReasons.set(pairKey(id, provider), "fallback_bound_reached");
  }
  for (const [id, providers] of qualified) {
    for (const provider of providers) {
      const key = pairKey(id, provider);
      if (!usedKeys.has(key) && !pairReasons.has(key)) pairReasons.set(key, "not_declared_fallback");
    }
  }

  const skipped: ExecutionPlanSkip[] = [];
  for (const runtime of runtimes) {
    const id = runtime.id as string;
    const exclusion = exclusions.get(id);
    if (exclusion !== undefined) {
      skipped.push({ provider: null, reason: exclusion, runtime_id: id });
      continue;
    }
    for (const provider of providersOn(metas.get(id) as RuntimeMeta, profile.platform)) {
      const reason = pairReasons.get(pairKey(id, provider));
      if (reason !== undefined) skipped.push({ provider, reason, runtime_id: id });
    }
  }
  if (used.length === 0) return result("no_eligible_runtime", null, [], skipped);

  const steps: ExecutionPlanStep[] = [];
  const assumptions = new Set<string>();
  used.forEach(([id, provider], position) => {
    const runtime = byId.get(id) as JsonRecord;
    const meta = metas.get(id) as RuntimeMeta;
    const previous = position > 0 ? used[position - 1] : undefined;
    const role =
      previous === undefined ? "selected" : previous[0] === id ? "provider_fallback" : "artifact_fallback";
    const minMemory = runtime.min_memory_bytes as number;
    if (memoryLimit === null && minMemory > 0) assumptions.add("memory_budget_unknown");
    if (profile.storage_budget_bytes === null && (runtime.min_storage_bytes as number) > 0) {
      assumptions.add("storage_budget_unknown");
    }
    if (meta.tier !== null && profile.max_model_tier === null) assumptions.add("model_tier_limit_unknown");
    if (constraints.min_sequence_length !== null && profile.max_sequence_length === null) {
      assumptions.add("device_sequence_limit_unknown");
    }
    steps.push({
      accelerator: PROVIDER_ACCELERATORS[provider as ExecutionProvider],
      artifact_sha256: digests.get(runtime.entrypoint as string) as string,
      evidence: meta.evidence,
      format: runtime.format as string,
      max_sequence_length: meta.maxSequenceLength,
      min_memory_bytes: minMemory,
      precision: runtime.precision as string,
      provider,
      role,
      runtime_id: id,
      runtime_index: runtimes.indexOf(runtime),
      tier: meta.tier,
    });
  });
  return result(null, root, steps, skipped, assumptions);
}

export interface PlanInvalidationInput extends ResolveExecutionPlanInput {
  readonly plan: ExecutionPlan;
}

/**
 * Return which inputs changed since `plan` was resolved (sorted component
 * names). A cached plan may be reused only when the result is empty.
 */
export async function executionPlanInvalidation(input: PlanInvalidationInput): Promise<string[]> {
  const { plan } = input;
  const constraints = input.constraints ?? createExecutionConstraints();
  const changed: string[] = [];
  if (plan.bundle_content_id !== (await portableBundleContentId(input.manifest))) changed.push("bundle");
  if (plan.constraints_digest !== (await executionConstraintsDigest(constraints))) changed.push("constraints");
  if (plan.profile_digest !== (await deviceCapabilityProfileDigest(input.profile))) changed.push("profile");
  const expectedKey = await computePlanKey(
    plan.bundle_content_id,
    plan.profile_digest,
    plan.constraints_digest,
    plan.verification_digest,
  );
  if (plan.plan_key !== expectedKey) changed.push("resolver");
  if (plan.verification_digest !== (await verificationDigest(input.verification))) changed.push("verification");
  return changed;
}

