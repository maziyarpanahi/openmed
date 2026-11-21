import { decodeBioTokenSpans } from "./decoder";
import {
  assertLocalModelReference,
  loadOnnxModel,
  loadTokenClassificationPipeline,
} from "./model-loader";
import { resolveOfflineOnly } from "./offline";
import {
  OPENMED_SPAN_SCHEMA_VERSION,
  type DeidentifyOptions,
  type ExtractPiiOptions,
  type LoadModelOptions,
  type ModelLoader,
  type OpenMedDeidentifyResult,
  type OpenMedSpan,
  type SpanAction,
  type TokenClassificationPipeline,
} from "./types";

export {
  CANONICAL_LABELS,
  OPENMED_SPAN_SCHEMA_VERSION,
  POLICY_LABELS,
  SPAN_ACTIONS,
} from "./types";
export type {
  CanonicalLabel,
  DecodedEntitySpan,
  DeidentifyOptions,
  ExtractPiiOptions,
  LoadModelOptions,
  LoadOnnxModelOptions,
  ModelLoader,
  OnnxWasmPaths,
  OpenMedOnnxVariant,
  OpenMedDeidentifyResult,
  OpenMedSpan,
  PolicyLabel,
  RawTokenClassificationEntity,
  RawTokenClassificationOutput,
  RawTokenClassificationPipeline,
  RawTransformersRuntime,
  SpanAction,
  TextHash,
  TokenClassificationCallOptions,
  TokenClassificationEntity,
  TokenClassificationOutput,
  TokenClassificationPipeline,
  TransformersRuntime,
} from "./types";
export {
  decodeBioTokenSpans,
  flattenTokenClassificationOutput,
  normalizeLabel,
  policyLabelFor,
  refinePrivacyFilterSpan,
  trimSpanWhitespace,
} from "./decoder";
export { alignTokenOffsets } from "./offsets";
export {
  KIT_PARITY_ACTIONS,
  KIT_PARITY_CANONICAL_LABELS,
  KIT_PARITY_CONFIDENCE_SCALE,
  KIT_PARITY_DECODER_CONTRACT_VERSIONS,
  KIT_PARITY_OFFSET_CONTRACT_VERSIONS,
  KIT_PARITY_OFFSET_UNIT,
  KIT_PARITY_PROFILE,
  KIT_PARITY_PROFILE_VERSION,
  KitParityError,
  acceptKitParityResult,
  applyKitParitySpans,
  checkKitParityCompatibility,
  confidenceToPpm,
  parseKitParityResult,
  scalarLength,
  scalarToUtf16Offset,
  scalarToUtf8Offset,
  serializeKitParityResult,
  toKitParityResult,
  utf16ToScalarOffset,
  utf8ToScalarOffset,
  validateKitParityResult,
  verifyKitParityResult,
} from "./kit-parity";
export {
  PORTABLE_POLICY_DEFAULT_PROFILE,
  PORTABLE_POLICY_IR_SCHEMA,
  PORTABLE_POLICY_IR_VERSION,
  PORTABLE_POLICY_MANIFEST_FILE,
  PORTABLE_POLICY_SUPPORTED_CAPABILITIES,
  PORTABLE_POLICY_SURROGATE_KEY_MAX_BYTES,
  PORTABLE_POLICY_SURROGATE_KEY_MIN_BYTES,
  PORTABLE_POLICY_TAXONOMY_FILE,
  PortablePolicyError,
  decidePortablePolicy,
  decodePortablePolicyManifest,
  decodePortablePolicyProfile,
  decodePortablePolicyTaxonomy,
  loadPortablePolicy,
  normalizePortableLabel,
  portableLabelKey,
  portablePolicyNames,
  portablePolicyReplacement,
  portableSurrogateToken,
  resolvePortablePolicyName,
  sha256Digest,
} from "./portable-policy";
export type {
  LoadPortablePolicyOptions,
  PortablePolicy,
  PortablePolicyDecideOptions,
  PortablePolicyDecision,
  PortablePolicyDecisionResult,
  PortablePolicyDetection,
  PortablePolicyManifest,
  PortablePolicyTaxonomy,
  PortablePolicyThreshold,
} from "./portable-policy";
export {
  PORTABLE_POLICY_RESOURCES,
  PORTABLE_TAXONOMY_LABELS,
} from "./portable-policy-resources";
export type {
  KitParityAction,
  KitParityBundleContract,
  KitParityResult,
  KitParitySpan,
} from "./kit-parity";
export {
  isLocalModelReference,
  loadOnnxModel,
  loadTokenClassificationPipeline,
} from "./model-loader";
export { OpenMedOfflineError, isOfflineOnly, setOfflineOnly } from "./offline";
export type { OpenMedOfflineErrorCode } from "./offline";
export {
  DECISION_ADVISORY,
  DECISION_COMPATIBILITY_POLICY,
  DECISION_POLICY_VERSION,
  DECISION_SCHEMA_VERSION,
  DEFAULT_DECISION_ACCESS_POLICY,
  DEFAULT_DECISION_CALIBRATION_ID,
  DEFAULT_DECISION_CALIBRATION_PROFILE,
  DEFAULT_DECISION_QUESTION_ID,
  DecisionEncoderError,
  DecisionRequest,
  DecisionRuntimeError,
  EncoderDecisionRuntime,
  GLICLASS_DEFAULT_MAX_LENGTH,
  GLICLASS_DEFAULT_SCORE_LEVELS,
  SUPPORTED_DECISION_SCHEMA_VERSIONS,
  createOrtDecisionEncoder,
  formatGLiClassInput,
  loadOrtWebDecisionEncoder,
  parseDecisionRequest,
  validateDecisionCalibrationProfile,
} from "./decision";
export type {
  DecisionAccess,
  DecisionAccessPolicy,
  DecisionBackendIdentity,
  DecisionBackendIdentityInput,
  DecisionBackendKind,
  DecisionCalibrationProfile,
  DecisionEncoder,
  DecisionEncoderErrorReason,
  DecisionErrorKind,
  DecisionMode,
  DecisionOptionScore,
  DecisionQuestionResult,
  DecisionResult,
  DecisionReview,
  DecisionState,
  DecisionTokenizer,
  EncoderDecisionRuntimeOptions,
  GLiClassDecisionConfiguration,
  GLiClassExample,
  OrtDecisionEncoderOptions,
} from "./decision";
export {
  detectOrtWebCapabilities,
  probeOrtWebCapabilities,
  selectOrtWebBackend,
} from "./runtime/capability";
export type {
  CapabilityDetectionOptions,
  OrtCapabilityGlobalScope,
  OrtWebBackend,
  OrtWebBackendChoice,
  OrtWebCapabilityProbeOptions,
  OrtWebCapabilityProbeResult,
  OrtWebCapabilityProfile,
  WebGpuAdapterProbe,
  WebGpuNavigatorProbe,
} from "./runtime/capability";
export {
  assertOfflineAssetPath,
  clearOrtWebSessionCache,
  configureOrtWebRuntime,
  loadOrtWebSession,
  loadOrtWebTokenClassificationPipeline,
  normalizeOrtAssetPath,
} from "./runtime/ort-web-loader";
export type {
  OrtExecutionProvider,
  OrtFeeds,
  OrtInferenceSession,
  OrtResults,
  OrtSessionCreateOptions,
  OrtTensorConstructor,
  OrtTensorLike,
  OrtTokenClassificationDecodeContext,
  OrtWebExternalDataOptions,
  OrtWebExternalDataShard,
  OrtWebLoadedSession,
  OrtWebLoaderOptions,
  OrtWebRuntime,
  OrtWebRuntimeProvider,
  OrtWebSessionCache,
  OrtWebTokenClassificationPipelineOptions,
} from "./runtime/ort-web-loader";
export {
  CLASSIFY_WGSL_SOURCE,
  DEFAULT_WEBGPU_LOGIT_TOLERANCE,
  DEFAULT_WEBGPU_MAX_RECALL_DELTA,
  WEBGPU_BENCHMARK_SUITE,
  WebGpuClassificationHead,
  WebGpuTokenClassificationSession,
  WebGpuVerificationError,
  certifyWebGpuReference,
  createWebGpuClassificationHead,
  decodeWebGpuTokenSpans,
  evaluateWebGpuRecallGate,
  loadWebGpuTokenClassificationSession,
} from "./runtime/webgpu-session";
export type {
  TokenClassificationLogits,
  TokenIdData,
  WebGpuBenchmarkOptions,
  WebGpuBenchmarkReport,
  WebGpuClassificationHeadConfig,
  WebGpuDeviceBenchmarkMetrics,
  WebGpuInputNames,
  WebGpuModelPaths,
  WebGpuRecallGate,
  WebGpuRecallGateOptions,
  WebGpuReferenceCertification,
  WebGpuReferenceCertificationOptions,
  WebGpuTokenBatch,
  WebGpuTokenClassificationSessionOptions,
  WebGpuTokenSpan,
} from "./runtime/webgpu-session";
export {
  INFERENCE_WORKER_ERROR_CODES,
  INFERENCE_WORKER_PROTOCOL_VERSION,
  InferenceWorkerError,
} from "./worker/protocol";
export type {
  InferenceWorkerErrorCode,
  InferenceWorkerLoadReason,
  InferenceWorkerLoadSpec,
  InferenceWorkerPlan,
  NormalizedInferenceWorkerLoadSpec,
} from "./worker/protocol";
export {
  InferenceWorkerClient,
  createInferenceWorkerClient,
} from "./worker/client";
export type {
  InferenceWorkerClientOptions,
  InferenceWorkerEndpoint,
  InferenceWorkerEventListener,
  InferenceWorkerLoadOptions,
  InferenceWorkerPipelineOptions,
  InferenceWorkerRecoverOptions,
  InferenceWorkerRunOptions,
  InferenceWorkerState,
  InferenceWorkerStats,
} from "./worker/client";
export { serveInferenceWorker } from "./worker/server";
export type {
  InferenceWorkerLoadContext,
  InferenceWorkerPipeline,
  InferenceWorkerRunContext,
  InferenceWorkerRuntime,
  InferenceWorkerScope,
  InferenceWorkerServer,
  InferenceWorkerServerStats,
  ServeInferenceWorkerOptions,
} from "./worker/server";

export {
  PORTABLE_BUNDLE_DEFAULT_LIMITS,
  PORTABLE_BUNDLE_ENVELOPE_FILENAME,
  PORTABLE_BUNDLE_ENVELOPE_SCHEMA,
  PORTABLE_BUNDLE_MANIFEST_FILENAME,
  PORTABLE_BUNDLE_PAYLOAD_TYPE,
  PORTABLE_BUNDLE_REASONS,
  PORTABLE_BUNDLE_SCHEMA,
  PORTABLE_BUNDLE_SCHEMA_VERSION,
  PORTABLE_BUNDLE_TRUST_POLICY_SCHEMA,
  PORTABLE_BUNDLE_VERIFICATION_SCHEMA,
  PortableBundleError,
  canonicalJson,
  parsePortableBundleManifest,
  parsePortableBundleTrustPolicy,
  portableBundleContentId,
  portableBundleSigningInput,
  verifyPortableBundle,
} from "./portable-bundle";
export type {
  PortableBundleContext,
  PortableBundleReason,
  PortableBundleTarget,
  PortableBundleTrustedKey,
  PortableBundleTrustPolicy,
  PortableBundleVerification,
  VerifyPortableBundleInput,
} from "./portable-bundle";
export {
  BUNDLE_CACHE_CONTRACT_VERSION,
  BUNDLE_CACHE_REASONS,
  BUNDLE_CACHE_SNAPSHOT_SCHEMA,
  BUNDLE_GENERATION_STATES,
  BUNDLE_PROTECTIONS,
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
  openBundleCacheStorage,
} from "./bundle-cache";
export type {
  BundleAcquireRequest,
  BundleAcquireResult,
  BundleCacheOptions,
  BundleCacheReason,
  BundleCacheSnapshot,
  BundleCacheStorage,
  BundleGenerationSnapshot,
  BundleGenerationState,
  BundleLease,
  BundleLockManager,
  BundleOpenVerification,
  BundleProtection,
  BundleQuotaDecision,
  BundleRangeResponse,
  BundleRangeSource,
  BundleRecoveryAction,
  BundleRecoveryReport,
  BundleRecoveryStep,
  BundleResumeReason,
  IdbFactoryLike,
  IdbKeyRangeLike,
  OpfsDirectoryLike,
  OpfsFileLike,
} from "./bundle-cache";

export {
  BUDGET_LADDER,
  CPU_ARCHITECTURES,
  DEVICE_CAPABILITIES_SCHEMA,
  DEVICE_CAPABILITIES_SCHEMA_VERSION,
  DEVICE_FEATURES,
  DeviceCapabilityError,
  MODEL_TIERS,
  NETWORK_STATES,
  PLATFORM_PROVIDERS,
  PROVIDER_ACCELERATORS,
  budgetBucket,
  coarseOsVersion,
  createDeviceCapabilityProfile,
  deviceCapabilityProfileDigest,
  parseDeviceCapabilityProfile,
  probeWebDeviceCapabilities,
  webDeviceCapabilityProfile,
} from "./device-capabilities";
export type {
  AcceleratorKind,
  DeviceCapabilityProfile,
  DeviceCapabilityProfileInit,
  DeviceCapabilityReason,
  ExecutionProvider,
  ModelTier,
  NetworkState,
  WebCapabilityProfileOptions,
  WebDeviceCapabilityProbeOptions,
} from "./device-capabilities";

export {
  DEFAULT_PROVIDERS,
  EXECUTION_CONSTRAINTS_SCHEMA,
  EXECUTION_EXTENSION,
  EXECUTION_PLAN_REASONS,
  EXECUTION_PLAN_SCHEMA,
  EXECUTION_RESOLVER_VERSION,
  ExecutionConstraintsError,
  FORMAT_PROVIDERS,
  computePlanKey,
  createExecutionConstraints,
  executionConstraintsDigest,
  executionPlanInvalidation,
  parseExecutionConstraints,
  resolveExecutionPlan,
  verificationDigest,
} from "./execution-plan";
export type {
  ExecutionConstraints,
  ExecutionConstraintsInit,
  ExecutionPlan,
  ExecutionPlanReason,
  ExecutionPlanSkip,
  ExecutionPlanStep,
  PlanInvalidationInput,
  ResolveExecutionPlanInput,
} from "./execution-plan";

export {
  DECISION_CALIBRATION_EXTENSION,
  DECISION_CALIBRATION_EXTENSION_VERSION,
  DECISION_CALIBRATION_PPM_SCALE,
  DECISION_CALIBRATION_REASONS,
  DecisionCalibrationError,
  bundleCalibrationProfileJson,
  decisionCalibrationRuntimeDigest,
  readDecisionCalibration,
  resolveDecisionCalibration,
  validateDecisionCalibrationRequest,
} from "./portable-bundle-calibration";
export type {
  BundleCalibrationProfile,
  DecisionCalibrationMethod,
  DecisionCalibrationRead,
  DecisionCalibrationReason,
  DecisionCalibrationRequest,
  DecisionCalibrationRuntimeBinding,
  DecisionCalibrationSelection,
} from "./portable-bundle-calibration";

export const DEFAULT_MODEL_ID =
  "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android";
/** Hub commit of `DEFAULT_MODEL_ID` that the default path requests. */
export const DEFAULT_MODEL_REVISION = "79f7db205869b1be4be23ac4f42aa95bdedc5aee";
const ONNX_ANDROID_REPO_PATTERN = /-onnx-android$/i;

interface OwnedPipeline {
  ready: Promise<TokenClassificationPipeline>;
  active: number;
  drained?: () => void;
  disposal?: Promise<void>;
}
const ownedPipelines = new Map<string, OwnedPipeline>();
const optionIdentities = new WeakMap<object, number>();
let nextOptionIdentity = 0;
function identity(value: object): number {
  let id = optionIdentities.get(value);
  if (id === undefined) {
    id = ++nextOptionIdentity;
    optionIdentities.set(value, id);
  }
  return id;
}
function optionKey(value: unknown, seen = new Set<object>()): string {
  if ((typeof value === "object" && value !== null) || typeof value === "function") {
    const object = value as object;
    if (typeof value === "function" || seen.has(object) ||
        (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype)) {
      return `identity:${identity(object)}`;
    }
    if (Object.getOwnPropertySymbols(object).length > 0) {
      throw new Error("OpenMed loader options do not support symbol keys");
    }
    seen.add(object);
    const key = Array.isArray(value)
      ? `array:[${value.map(item => optionKey(item, seen)).join(",")}]`
      : `object:{${Object.keys(value).sort().map(key =>
          `${JSON.stringify(key)}:${optionKey((value as Record<string, unknown>)[key], seen)}`
        ).join(",")}}`;
    seen.delete(object);
    return key;
  }
  if (typeof value === "symbol") {
    throw new Error("OpenMed loader options do not support symbol values");
  }
  return JSON.stringify([typeof value, Object.is(value, -0) ? "-0" : String(value)]);
}
async function disposeOwned(entry: OwnedPipeline): Promise<void> {
  return entry.disposal ??= (async () => {
    let pipeline;
    try { pipeline = await entry.ready; } catch { return; }
    if (entry.active > 0) {
      await new Promise<void>(resolve => { entry.drained = resolve; });
    }
    try { await pipeline.dispose?.(); } catch {
      throw new Error("OpenMed pipeline disposal failed");
    }
  })();
}
/** Drain and release package-created pipelines; caller pipelines remain owned by callers. */
export async function disposeDefaultPipelines(): Promise<void> {
  const entries = [...ownedPipelines.values()];
  ownedPipelines.clear();
  const results = await Promise.allSettled(entries.map(disposeOwned));
  if (results.some(result => result.status === "rejected")) {
    throw new Error("OpenMed pipeline disposal failed");
  }
}
async function acquireOwned(model: string, options: ExtractPiiOptions): Promise<OwnedPipeline> {
  const loader = options.modelLoader ?? defaultModelLoaderFor(model);
  const { runtime, ...loaderOptions } = options.loaderOptions ?? {};
  const key = `${JSON.stringify(model)}:${identity(loader)}:${runtime === undefined ? "default" : identity(runtime)}:${optionKey(loaderOptions)}`;
  let entry = ownedPipelines.get(key);
  if (entry === undefined) {
    entry = { ready: Promise.resolve().then(() => loader(model, options.loaderOptions)), active: 0 };
    ownedPipelines.set(key, entry);
    const current = entry;
    void entry.ready.catch(() => { if (ownedPipelines.get(key) === current) ownedPipelines.delete(key); });
  }
  entry.active++;
  return entry;
}
function releaseOwned(entry: OwnedPipeline): void {
  entry.active--;
  if (entry.active === 0) entry.drained?.();
}

export async function extractPii(
  text: string,
  options: ExtractPiiOptions = {},
): Promise<OpenMedSpan[]> {
  options = withAcquisitionPolicy(options);
  const hashSecret =
    options.hashSecret == null
      ? await ephemeralHashSecret()
      : new Uint8Array(toBytes(options.hashSecret));
  if (hashSecret.byteLength === 0) {
    throw new Error("hashSecret must not be empty");
  }
  const model = options.model ?? DEFAULT_MODEL_ID;
  const owned = options.pipeline === undefined ? await acquireOwned(model, options) : undefined;
  let released = false;
  try {
    const pipeline = options.pipeline ?? await owned!.ready;
    // Keep "O" tokens so offset alignment sees the full token sequence.
    const rawOutput = await pipeline(text, {
      aggregation_strategy: "none",
      ignore_labels: [],
      ...(options.pipelineOptions ?? {}),
    });
    const decoded = decodeBioTokenSpans(text, rawOutput, {
      threshold: options.threshold ?? 0,
    });

    const spans: OpenMedSpan[] = [];
    for (const entity of decoded) {
      const surface = text.slice(entity.start, entity.end);
      spans.push({
        schema_version: OPENMED_SPAN_SCHEMA_VERSION,
        doc_id: options.docId ?? "document",
        start: entity.start,
        end: entity.end,
        text_hash: await hmacTextHash(surface, hashSecret),
        entity_type: entity.entity_type,
        canonical_label: entity.canonical_label,
        policy_label: entity.policy_label,
        regulatory_tags: [...(options.regulatoryTags ?? [])],
        score: entity.score,
        detector: options.detector ?? "transformersjs",
        evidence: {
          token_count: entity.token_count,
        },
        action: "keep",
        replacement: null,
        reversible_id: null,
        section: options.section ?? null,
        metadata: {
          ...(options.model || !options.pipeline ? { model } : {}),
          ...(options.metadata ?? {}),
        },
      });
    }
    return spans;
  } catch (error) {
    if (owned !== undefined) {
      for (const [key, entry] of ownedPipelines) {
        if (entry === owned) ownedPipelines.delete(key);
      }
      // Release this operation before waiting for other active calls to drain.
      releaseOwned(owned);
      released = true;
      await disposeOwned(owned);
      throw error;
    }
    throw error;
  } finally {
    if (owned !== undefined && !released) releaseOwned(owned);
  }
}

function defaultModelLoaderFor(model: string): ModelLoader {
  return ONNX_ANDROID_REPO_PATTERN.test(model)
    ? loadOnnxModel
    : loadTokenClassificationPipeline;
}

/** Pin the default model and refuse remote references before any loader runs. */
function withAcquisitionPolicy(options: ExtractPiiOptions): ExtractPiiOptions {
  if (options.pipeline !== undefined) {
    return options;
  }
  const model = options.model ?? DEFAULT_MODEL_ID;
  const loaderOptions: LoadModelOptions = { ...options.loaderOptions };
  if (
    resolveOfflineOnly(options.offlineOnly) ||
    loaderOptions.offlineOnly === true
  ) {
    assertLocalModelReference(model);
    loaderOptions.offlineOnly = true;
  }
  if (model === DEFAULT_MODEL_ID && loaderOptions.revision === undefined) {
    loaderOptions.revision = DEFAULT_MODEL_REVISION;
  }
  return { ...options, loaderOptions };
}

export async function deidentify(
  text: string,
  options: DeidentifyOptions = {},
): Promise<OpenMedDeidentifyResult> {
  const spans = await extractPii(text, options);
  const redactionSpans = spans.map((span) => {
    const replacement = options.replacement?.(span) ?? `[${span.canonical_label}]`;
    return {
      ...span,
      action: "redact" as SpanAction,
      replacement,
    };
  });
  return {
    text,
    deidentifiedText: spansToRedactedText(text, redactionSpans),
    spans: redactionSpans,
  };
}

/**
 * Apply span replacements right to left at UTF-16 offsets.
 *
 * Spans whose action is `keep` are left in the text: a kept span is an
 * explicit policy decision, never a redaction.
 */
export function spansToRedactedText(text: string, spans: OpenMedSpan[]): string {
  let redacted = text;
  const applied = spans.filter((span) => span.action !== "keep");
  for (const span of applied.sort((left, right) => right.start - left.start)) {
    redacted =
      redacted.slice(0, span.start) +
      (span.replacement ?? `[${span.canonical_label}]`) +
      redacted.slice(span.end);
  }
  return redacted;
}

export async function hmacTextHash(
  surface: string | Uint8Array,
  secret: string | Uint8Array,
): Promise<`hmac-sha256:${string}`> {
  const payload = toBytes(surface);
  const key = new Uint8Array(toBytes(secret));
  if (key.byteLength === 0) {
    throw new Error("hashSecret must not be empty");
  }
  const subtle = globalThis.crypto?.subtle;
  if (subtle) {
    const cryptoKey = await subtle.importKey(
      "raw",
      toArrayBuffer(key),
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["sign"],
    );
    const signature = await subtle.sign("HMAC", cryptoKey, toArrayBuffer(payload));
    return `hmac-sha256:${toHex(new Uint8Array(signature))}`;
  }

  const { createHmac } = await import("node:crypto");
  const digest = createHmac("sha256", key).update(payload).digest("hex");
  return `hmac-sha256:${digest}`;
}

async function ephemeralHashSecret(): Promise<Uint8Array> {
  if (globalThis.crypto?.getRandomValues) {
    return globalThis.crypto.getRandomValues(new Uint8Array(32));
  }
  const { randomBytes } = await import("node:crypto");
  return randomBytes(32);
}

function toBytes(value: string | Uint8Array): Uint8Array {
  return typeof value === "string" ? new TextEncoder().encode(value) : value;
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function toHex(bytes: Uint8Array): string {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join(
    "",
  );
}

export { CrisisRouting, type CrisisNotice, type CrisisResource } from "./crisis-routing";
