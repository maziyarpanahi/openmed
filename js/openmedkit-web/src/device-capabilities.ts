/**
 * Normalized device-capability profiles for the web kit.
 *
 * Mirrors `openmed/core/device_capabilities.py`: the same closed vocabulary,
 * budget ladder, and canonical document, so a profile built here encodes to
 * the same bytes and digest as the Python reference for the shared vectors in
 * `tests/fixtures/device_capabilities/`. The web adapter maps the existing
 * ONNX Runtime Web probe (`runtime/capability.ts`) into the profile. Nothing in
 * a profile can carry a user agent, GPU adapter string, or exact memory size.
 */

import { canonicalJson, portableBundleContentId } from "./portable-bundle";
import {
  probeOrtWebCapabilities,
  type OrtCapabilityGlobalScope,
  type OrtWebCapabilityProbeOptions,
  type OrtWebCapabilityProfile,
} from "./runtime/capability";

export const DEVICE_CAPABILITIES_SCHEMA = "openmed.device_capabilities";
export const DEVICE_CAPABILITIES_SCHEMA_VERSION = 1;

/** Accelerator kind behind each normalized execution provider. */
export const PROVIDER_ACCELERATORS = {
  coreml: "npu",
  cpu: "cpu",
  metal: "gpu",
  nnapi: "npu",
  qnn: "npu",
  wasm: "cpu",
  webgpu: "gpu",
} as const;

export type ExecutionProvider = keyof typeof PROVIDER_ACCELERATORS;
export type AcceleratorKind = (typeof PROVIDER_ACCELERATORS)[ExecutionProvider];

/** Execution providers a profile may report on each platform. */
export const PLATFORM_PROVIDERS: Readonly<Record<string, readonly ExecutionProvider[]>> = {
  android: ["cpu", "nnapi", "qnn"],
  ios: ["coreml", "cpu", "metal"],
  linux: ["cpu"],
  macos: ["coreml", "cpu", "metal"],
  visionos: ["coreml", "cpu"],
  watchos: ["coreml", "cpu"],
  web: ["wasm", "webgpu"],
  windows: ["cpu"],
};

export const CPU_ARCHITECTURES = ["arm64", "armv7", "unknown", "wasm32", "x86_64"] as const;
export const DEVICE_FEATURES = [
  "cpu_avx2",
  "cpu_fp16",
  "cpu_neon",
  "wasm_simd",
  "wasm_threads",
  "webgpu_shader_f16",
] as const;
/** Model tiers in increasing capacity order. */
export const MODEL_TIERS = ["nano", "tiny", "base"] as const;
export const NETWORK_STATES = ["offline", "online", "unknown"] as const;

export type ModelTier = (typeof MODEL_TIERS)[number];
export type NetworkState = (typeof NETWORK_STATES)[number];

const MIB = 1 << 20;
/** Budgets are `0` or a power of two from `minBytes` to `maxBytes`. */
export const BUDGET_LADDER = { minBytes: 16 * MIB, maxBytes: 2 ** 40 } as const;
export const MAX_SEQUENCE_LENGTH = 1 << 20;
export const MAX_OPERATOR_SET = 1000;
export const WEB_SDK_NAME = "openmedkit-web";

const PLATFORMS = ["android", "ios", "linux", "macos", "visionos", "watchos", "web", "windows"];
const PRECISIONS = ["bf16", "fp16", "fp32", "int2", "int4", "int8"];
/** Platforms each portable-bundle runtime format may declare. */
export const FORMAT_PLATFORMS: Readonly<Record<string, readonly string[]>> = {
  coreml: ["ios", "macos", "visionos", "watchos"],
  mlx: ["ios", "macos"],
  onnx: ["android", "ios", "linux", "macos", "web", "windows"],
  ort: ["android", "ios"],
  pytorch: ["linux", "macos", "windows"],
  transformersjs: ["web"],
};
const ACCELERATORS = ["cpu", "gpu", "npu"];
const TOKEN = /^[a-z0-9][a-z0-9._-]{0,63}$/;
const VERSION = /^(?:0|[1-9][0-9]{0,8})(?:\.(?:0|[1-9][0-9]{0,8})){0,3}$/;
const COARSE_OS = /^(?:0|[1-9][0-9]{0,8})(?:\.(?:0|[1-9][0-9]{0,8}))?$/;
const MIN_THREADED_WASM_CONCURRENCY = 2;
const PROFILE_KEYS = [
  "accelerators",
  "cpu_architecture",
  "features",
  "max_model_tier",
  "max_sequence_length",
  "memory_budget_bytes",
  "network",
  "operator_sets",
  "os_version",
  "platform",
  "precisions",
  "providers",
  "runtimes",
  "schema",
  "schema_version",
  "sdk_name",
  "sdk_version",
  "storage_budget_bytes",
];

/** Canonical profile document (`device_capability_profile`). */
export interface DeviceCapabilityProfile {
  readonly schema: typeof DEVICE_CAPABILITIES_SCHEMA;
  readonly schema_version: typeof DEVICE_CAPABILITIES_SCHEMA_VERSION;
  readonly platform: string;
  readonly sdk_name: string;
  readonly sdk_version: string;
  readonly os_version: string | null;
  readonly cpu_architecture: string;
  readonly providers: readonly string[];
  readonly accelerators: readonly string[];
  readonly runtimes: readonly string[];
  readonly precisions: readonly string[];
  readonly features: readonly string[];
  readonly operator_sets: Readonly<Record<string, number>>;
  readonly memory_budget_bytes: number | null;
  readonly storage_budget_bytes: number | null;
  readonly max_sequence_length: number | null;
  readonly max_model_tier: string | null;
  readonly network: string;
}

/** Profile members; `schema`, `schema_version`, and `accelerators` are derived. */
export type DeviceCapabilityProfileInit = Omit<
  DeviceCapabilityProfile,
  "schema" | "schema_version" | "accelerators"
>;

export type DeviceCapabilityReason = "profile_invalid" | "probe_invalid";

/** Error carrying only a controlled reason code. */
export class DeviceCapabilityError extends Error {
  readonly reason: DeviceCapabilityReason;

  constructor(reason: DeviceCapabilityReason) {
    super(reason);
    this.name = "DeviceCapabilityError";
    this.reason = reason;
  }
}

function isUint(value: unknown, maximum = Number.MAX_SAFE_INTEGER): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 && value <= maximum;
}

/**
 * Round a measured byte budget down to the coarse ladder: `0`, then every
 * power of two from 16 MiB to 1 TiB.
 */
export function budgetBucket(value: number | null | undefined): number | null {
  if (value === null || value === undefined) return null;
  if (!isUint(value)) throw new DeviceCapabilityError("probe_invalid");
  if (value < BUDGET_LADDER.minBytes) return 0;
  let bucket: number = BUDGET_LADDER.minBytes;
  while (bucket < BUDGET_LADDER.maxBytes && bucket * 2 <= value) bucket *= 2;
  return bucket;
}

function isBudget(value: unknown): boolean {
  return value === null || (isUint(value) && budgetBucket(value) === value);
}

/** Keep only the major and minor components of a dotted OS version. */
export function coarseOsVersion(value: string | null | undefined): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== "string" || !VERSION.test(value)) {
    throw new DeviceCapabilityError("probe_invalid");
  }
  return value.split(".").slice(0, 2).join(".");
}

/** Return the capacity rank of a model tier (`nano` is 0). */
export function modelTierRank(tier: string): number {
  return (MODEL_TIERS as readonly string[]).indexOf(tier);
}

function stringSet(value: unknown, allowed: readonly string[]): string[] {
  if (!Array.isArray(value)) throw new DeviceCapabilityError("profile_invalid");
  for (const item of value) {
    if (typeof item !== "string" || !allowed.includes(item)) {
      throw new DeviceCapabilityError("profile_invalid");
    }
  }
  return [...new Set(value as string[])].sort();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Validate members and return the canonical profile document. */
export function createDeviceCapabilityProfile(
  init: DeviceCapabilityProfileInit,
): DeviceCapabilityProfile {
  const invalid = (): never => {
    throw new DeviceCapabilityError("profile_invalid");
  };
  if (
    !PLATFORMS.includes(init.platform) ||
    typeof init.sdk_name !== "string" ||
    !TOKEN.test(init.sdk_name) ||
    typeof init.sdk_version !== "string" ||
    !VERSION.test(init.sdk_version) ||
    (init.os_version !== null &&
      (typeof init.os_version !== "string" || !COARSE_OS.test(init.os_version))) ||
    !(CPU_ARCHITECTURES as readonly string[]).includes(init.cpu_architecture) ||
    !isBudget(init.memory_budget_bytes) ||
    !isBudget(init.storage_budget_bytes) ||
    (init.max_sequence_length !== null &&
      !(isUint(init.max_sequence_length, MAX_SEQUENCE_LENGTH) && init.max_sequence_length > 0)) ||
    (init.max_model_tier !== null && modelTierRank(init.max_model_tier) < 0) ||
    !(NETWORK_STATES as readonly string[]).includes(init.network) ||
    !isRecord(init.operator_sets)
  ) {
    invalid();
  }
  const providers = stringSet(init.providers, PLATFORM_PROVIDERS[init.platform] ?? []);
  const runtimes = stringSet(
    init.runtimes,
    Object.keys(FORMAT_PLATFORMS).filter((key) =>
      (FORMAT_PLATFORMS[key] ?? []).includes(init.platform),
    ),
  );
  const precisions = stringSet(init.precisions, PRECISIONS);
  const features = stringSet(init.features, DEVICE_FEATURES);
  const operatorSets: Record<string, number> = {};
  for (const key of Object.keys(init.operator_sets).sort()) {
    const value = init.operator_sets[key];
    if (!(key in FORMAT_PLATFORMS) || !isUint(value, MAX_OPERATOR_SET) || value === 0) invalid();
    operatorSets[key] = value as number;
  }
  const accelerators = [
    ...new Set(providers.map((item) => PROVIDER_ACCELERATORS[item as ExecutionProvider])),
  ].sort();
  return {
    accelerators,
    cpu_architecture: init.cpu_architecture,
    features,
    max_model_tier: init.max_model_tier,
    max_sequence_length: init.max_sequence_length,
    memory_budget_bytes: init.memory_budget_bytes,
    network: init.network,
    operator_sets: operatorSets,
    os_version: init.os_version,
    platform: init.platform,
    precisions,
    providers,
    runtimes,
    schema: DEVICE_CAPABILITIES_SCHEMA,
    schema_version: DEVICE_CAPABILITIES_SCHEMA_VERSION,
    sdk_name: init.sdk_name,
    sdk_version: init.sdk_version,
    storage_budget_bytes: init.storage_budget_bytes,
  };
}

/** Parse a profile document strictly; unknown or missing members fail. */
export function parseDeviceCapabilityProfile(value: unknown): DeviceCapabilityProfile {
  if (
    !isRecord(value) ||
    Object.keys(value).length !== PROFILE_KEYS.length ||
    !PROFILE_KEYS.every((key) => Object.prototype.hasOwnProperty.call(value, key)) ||
    value.schema !== DEVICE_CAPABILITIES_SCHEMA ||
    value.schema_version !== DEVICE_CAPABILITIES_SCHEMA_VERSION
  ) {
    throw new DeviceCapabilityError("profile_invalid");
  }
  const profile = createDeviceCapabilityProfile(value as unknown as DeviceCapabilityProfileInit);
  const accelerators = stringSet(value.accelerators, ACCELERATORS);
  if (accelerators.join(",") !== profile.accelerators.join(",")) {
    throw new DeviceCapabilityError("profile_invalid");
  }
  return profile;
}

/** Return the `sha256:` digest of every resolution-relevant member (not `network`). */
export async function deviceCapabilityProfileDigest(
  profile: DeviceCapabilityProfile,
): Promise<string> {
  const { network: _network, ...relevant } = profile;
  return portableBundleContentId(new TextEncoder().encode(canonicalJson(relevant)));
}

/** Caller-supplied facts the ONNX Runtime Web probe cannot observe. */
export interface WebCapabilityProfileOptions {
  /** SDK version matched against bundle `compatibility`. */
  readonly sdkVersion: string;
  /** Whether the WebGPU adapter exposes `shader-f16`. */
  readonly webgpuShaderF16?: boolean;
  /** Bundle formats this app can load; defaults to `["onnx"]`. */
  readonly runtimes?: readonly string[];
  /** Memory the app lets the runtime use, in bytes; rounded down. */
  readonly memoryBudgetBytes?: number | null;
  /** Storage the model cache may use, in bytes; rounded down. */
  readonly storageFreeBytes?: number | null;
  readonly maxSequenceLength?: number | null;
  /** Highest ONNX opset the loaded ONNX Runtime Web build supports. */
  readonly operatorSets?: Readonly<Record<string, number>>;
  readonly network?: NetworkState;
}

/**
 * Map an `OrtWebCapabilityProfile` into the normalized profile. Mirrors
 * `profile_from_web_probe` in the Python reference.
 */
export function webDeviceCapabilityProfile(
  probe: OrtWebCapabilityProfile,
  options: WebCapabilityProfileOptions,
): DeviceCapabilityProfile {
  if (
    typeof probe.webgpu !== "boolean" ||
    typeof probe.wasm !== "boolean" ||
    typeof probe.wasmSimd !== "boolean" ||
    typeof probe.sharedArrayBuffer !== "boolean" ||
    typeof probe.crossOriginIsolated !== "boolean" ||
    !isUint(probe.hardwareConcurrency) ||
    (options.webgpuShaderF16 !== undefined && typeof options.webgpuShaderF16 !== "boolean")
  ) {
    throw new DeviceCapabilityError("probe_invalid");
  }
  const simd = probe.wasm && probe.wasmSimd;
  const threads =
    simd &&
    probe.sharedArrayBuffer &&
    probe.crossOriginIsolated &&
    probe.hardwareConcurrency >= MIN_THREADED_WASM_CONCURRENCY;
  const providers: string[] = [];
  if (probe.wasm) providers.push("wasm");
  if (probe.webgpu) providers.push("webgpu");
  const precisions =
    providers.length === 0 ? [] : ["fp32", "int4", "int8", ...(probe.webgpu ? ["fp16"] : [])];
  const features: string[] = [];
  if (simd) features.push("wasm_simd");
  if (threads) features.push("wasm_threads");
  if (probe.webgpu && options.webgpuShaderF16 === true) features.push("webgpu_shader_f16");
  const network = options.network ?? "unknown";
  if (!(NETWORK_STATES as readonly string[]).includes(network) || !isRecord(options.operatorSets ?? {})) {
    throw new DeviceCapabilityError("probe_invalid");
  }
  try {
    return createDeviceCapabilityProfile({
      cpu_architecture: "wasm32",
      features,
      max_model_tier: null,
      max_sequence_length: options.maxSequenceLength ?? null,
      memory_budget_bytes: budgetBucket(options.memoryBudgetBytes),
      network,
      operator_sets: { ...(options.operatorSets ?? {}) },
      os_version: null,
      platform: "web",
      precisions,
      providers,
      runtimes: [...(options.runtimes ?? ["onnx"])],
      sdk_name: WEB_SDK_NAME,
      sdk_version: options.sdkVersion,
      storage_budget_bytes: budgetBucket(options.storageFreeBytes),
    });
  } catch (error) {
    if (error instanceof DeviceCapabilityError) throw new DeviceCapabilityError("probe_invalid");
    throw error;
  }
}

/** Options for {@link probeWebDeviceCapabilities}. */
export type WebDeviceCapabilityProbeOptions = OrtWebCapabilityProbeOptions &
  Omit<WebCapabilityProfileOptions, "webgpuShaderF16" | "network"> & {
    readonly webgpuShaderF16?: boolean;
    readonly network?: NetworkState;
  };

interface ShaderF16Navigator {
  gpu?: {
    requestAdapter?: (options?: unknown) => Promise<unknown> | unknown;
  } | null;
  onLine?: boolean;
}

/**
 * Probe the current browser with the existing ONNX Runtime Web probe and
 * normalize the result. Only the adapter's `shader-f16` feature bit is read;
 * adapter vendor, architecture, and description strings are never touched.
 * No network request is made.
 */
export async function probeWebDeviceCapabilities(
  options: WebDeviceCapabilityProbeOptions,
): Promise<DeviceCapabilityProfile> {
  const result = await probeOrtWebCapabilities(options);
  const scope = (options.globalScope ?? globalThis) as OrtCapabilityGlobalScope;
  const navigatorLike = scope.navigator as ShaderF16Navigator | undefined;
  let shaderF16 = options.webgpuShaderF16;
  if (shaderF16 === undefined) {
    shaderF16 = false;
    if (result.profile.webgpu && typeof navigatorLike?.gpu?.requestAdapter === "function") {
      try {
        const adapter = (await navigatorLike.gpu.requestAdapter({
          powerPreference: options.powerPreference ?? "high-performance",
        })) as { features?: { has?: (name: string) => boolean } } | null;
        shaderF16 = adapter?.features?.has?.("shader-f16") === true;
      } catch {
        shaderF16 = false;
      }
    }
  }
  let network: NetworkState = options.network ?? "unknown";
  if (options.network === undefined && typeof navigatorLike?.onLine === "boolean") {
    network = navigatorLike.onLine ? "online" : "offline";
  }
  const profileOptions: WebCapabilityProfileOptions = {
    sdkVersion: options.sdkVersion,
    webgpuShaderF16: shaderF16,
    network,
    ...(options.runtimes !== undefined ? { runtimes: options.runtimes } : {}),
    ...(options.memoryBudgetBytes !== undefined
      ? { memoryBudgetBytes: options.memoryBudgetBytes }
      : {}),
    ...(options.storageFreeBytes !== undefined ? { storageFreeBytes: options.storageFreeBytes } : {}),
    ...(options.maxSequenceLength !== undefined
      ? { maxSequenceLength: options.maxSequenceLength }
      : {}),
    ...(options.operatorSets !== undefined ? { operatorSets: options.operatorSets } : {}),
  };
  return webDeviceCapabilityProfile(result.profile, profileOptions);
}
