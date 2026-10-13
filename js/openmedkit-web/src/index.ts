import { decodeBioTokenSpans } from "./decoder";
import { loadOnnxModel, loadTokenClassificationPipeline } from "./model-loader";
import {
  OPENMED_SPAN_SCHEMA_VERSION,
  type DeidentifyOptions,
  type ExtractPiiOptions,
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
  isLocalModelReference,
  loadOnnxModel,
  loadTokenClassificationPipeline,
} from "./model-loader";
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

export const DEFAULT_MODEL_ID =
  "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android";
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

export function spansToRedactedText(text: string, spans: OpenMedSpan[]): string {
  let redacted = text;
  for (const span of [...spans].sort((left, right) => right.start - left.start)) {
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
