export const OPENMED_REACT_NATIVE_MODULE_NAME = "OpenMedKitRN" as const;
export const OPENMED_SPAN_SCHEMA_VERSION = 1 as const;

export type OpenMedPlatform = "ios" | "android";
export type OpenMedBackendKind = "mlx" | "coreml" | "onnx";
export type SpanAction =
  | "keep"
  | "redact"
  | "replace"
  | "mask"
  | "remove"
  | "hash";
export type PolicyLabel =
  | "DIRECT_IDENTIFIER"
  | "QUASI_IDENTIFIER"
  | "CLINICAL_CONCEPT";
export type TextHash = `hmac-sha256:${string}` | `sha256:${string}`;

export interface OpenMedModelSource {
  modelPath: string;
  backend?: OpenMedBackendKind;
  id2LabelPath?: string;
  tokenizerName?: string;
  tokenizerFolderPath?: string;
  cacheKey?: string;
}

export interface OpenMedModelLoadResult {
  cacheKey: string;
  modelPath: string;
  backend: OpenMedBackendKind;
  platform: OpenMedPlatform;
  loaded: boolean;
}

export interface OpenMedBridgeOptions {
  confidenceThreshold?: number;
  useSmartMerging?: boolean;
  docId?: string;
  hashSecret?: string;
  detector?: string | null;
  metadata?: Record<string, unknown>;
}

export interface OpenMedDeidentifyOptions extends OpenMedBridgeOptions {
  policy?: string;
}

export interface OpenMedSpan {
  schema_version: typeof OPENMED_SPAN_SCHEMA_VERSION;
  doc_id: string;
  start: number;
  end: number;
  text_hash: TextHash;
  entity_type: string;
  canonical_label: string;
  policy_label: PolicyLabel;
  regulatory_tags: string[];
  score: number | null;
  detector: string | null;
  evidence: Record<string, unknown>;
  action: SpanAction;
  replacement: string | null;
  reversible_id: string | null;
  section: string | null;
  metadata: Record<string, unknown>;
}

export interface OpenMedDeidentifyResult {
  text: string;
  deidentifiedText: string;
  spans: OpenMedSpan[];
}

export interface OpenMedKitNativeModule {
  loadModel(source: NativeModelSource): Promise<OpenMedModelLoadResult>;
  analyzeText(
    text: string,
    options?: NativeBridgeOptions,
  ): Promise<NativeSpanResponse>;
  extractPii(
    text: string,
    options?: NativeBridgeOptions,
  ): Promise<NativeSpanResponse>;
  deidentify(
    text: string,
    options?: NativeDeidentifyOptions,
  ): Promise<NativeDeidentifyResponse>;
}

interface NativeModelSource {
  modelPath: string;
  backend: OpenMedBackendKind;
  cacheKey: string;
  id2LabelPath?: string;
  tokenizerName?: string;
  tokenizerFolderPath?: string;
}

interface NativeBridgeOptions {
  confidenceThreshold?: number;
  useSmartMerging?: boolean;
  docId?: string;
  hashSecret?: string;
  detector?: string | null;
  metadata?: Record<string, unknown>;
}

interface NativeDeidentifyOptions extends NativeBridgeOptions {
  policy: string;
}

type NativeSpan = Partial<OpenMedSpan> & Record<string, unknown>;
type NativeSpanResponse = NativeSpan[] | { spans?: NativeSpan[] };
type NativeDeidentifyResponse =
  | Partial<OpenMedDeidentifyResult>
  | {
      text?: string;
      originalText?: string;
      deidentifiedText?: string;
      redactedText?: string;
      spans?: NativeSpan[];
    };

interface ReactNativeRuntime {
  NativeModules?: Record<string, unknown>;
  TurboModuleRegistry?: {
    get<T = unknown>(name: string): T | null | undefined;
  };
  Platform?: {
    OS?: string;
  };
}

declare const require: undefined | ((moduleName: string) => unknown);

let injectedNativeModule: OpenMedKitNativeModule | null = null;

export function setOpenMedKitNativeModuleForTests(
  module: OpenMedKitNativeModule | null,
): void {
  injectedNativeModule = module;
}

export async function loadModel(
  source: OpenMedModelSource,
): Promise<OpenMedModelLoadResult> {
  const nativeSource = normalizeModelSource(source);
  const result = await nativeModule().loadModel(nativeSource);
  const resultRecord = asRecord(result);
  return {
    cacheKey: readString(resultRecord, "cacheKey", nativeSource.cacheKey),
    modelPath: readString(resultRecord, "modelPath", nativeSource.modelPath),
    backend: readBackend(result.backend, nativeSource.backend),
    platform: readPlatform(result.platform, currentPlatform()),
    loaded: result.loaded !== false,
  };
}

export async function analyzeText(
  text: string,
  options: OpenMedBridgeOptions = {},
): Promise<OpenMedSpan[]> {
  const response = await nativeModule().analyzeText(text, normalizeOptions(options));
  return normalizeSpans(response, options);
}

export async function extractPii(
  text: string,
  options: OpenMedBridgeOptions = {},
): Promise<OpenMedSpan[]> {
  const response = await nativeModule().extractPii(text, normalizeOptions(options));
  return normalizeSpans(response, options);
}

export async function deidentify(
  text: string,
  options: OpenMedDeidentifyOptions = {},
): Promise<OpenMedDeidentifyResult> {
  const nativeOptions: NativeDeidentifyOptions = {
    ...normalizeOptions(options),
    policy: options.policy ?? "hipaa_safe_harbor",
  };
  const response = await nativeModule().deidentify(text, nativeOptions);
  const responseRecord = asRecord(response);
  const spans = normalizeSpans(
    Array.isArray(responseRecord.spans)
      ? (responseRecord.spans as NativeSpan[])
      : [],
    options,
  );
  return {
    text: readString(responseRecord, "text", text),
    deidentifiedText:
      stringValue(responseRecord.deidentifiedText) ??
      stringValue(responseRecord.redactedText) ??
      text,
    spans,
  };
}

function nativeModule(): OpenMedKitNativeModule {
  if (injectedNativeModule) {
    return injectedNativeModule;
  }

  const runtime = loadReactNative();
  const turboModule =
    runtime?.TurboModuleRegistry?.get<OpenMedKitNativeModule>(
      OPENMED_REACT_NATIVE_MODULE_NAME,
    ) ?? null;
  const bridgeModule = runtime?.NativeModules?.[OPENMED_REACT_NATIVE_MODULE_NAME];
  const resolved = turboModule ?? bridgeModule;

  if (isNativeModule(resolved)) {
    return resolved;
  }

  throw new Error(
    `${OPENMED_REACT_NATIVE_MODULE_NAME} is not installed in the React Native runtime`,
  );
}

function loadReactNative(): ReactNativeRuntime | null {
  if (typeof require !== "function") {
    return null;
  }
  try {
    return require("react-native") as ReactNativeRuntime;
  } catch {
    return null;
  }
}

function isNativeModule(value: unknown): value is OpenMedKitNativeModule {
  const candidate = asRecord(value);
  return (
    typeof candidate.loadModel === "function" &&
    typeof candidate.analyzeText === "function" &&
    typeof candidate.extractPii === "function" &&
    typeof candidate.deidentify === "function"
  );
}

function normalizeModelSource(source: OpenMedModelSource): NativeModelSource {
  const modelPath = source.modelPath.trim();
  if (!modelPath) {
    throw new Error("OpenMedKit modelPath must not be blank");
  }

  const backend = source.backend ?? defaultBackendForPlatform();
  const nativeSource: NativeModelSource = {
    modelPath,
    backend,
    cacheKey: source.cacheKey ?? `${backend}:${modelPath}`,
  };
  if (source.id2LabelPath !== undefined) {
    nativeSource.id2LabelPath = source.id2LabelPath;
  }
  if (source.tokenizerName !== undefined) {
    nativeSource.tokenizerName = source.tokenizerName;
  }
  if (source.tokenizerFolderPath !== undefined) {
    nativeSource.tokenizerFolderPath = source.tokenizerFolderPath;
  }
  return nativeSource;
}

function normalizeOptions(options: OpenMedBridgeOptions): NativeBridgeOptions {
  const normalized: NativeBridgeOptions = {};
  if (options.confidenceThreshold !== undefined) {
    normalized.confidenceThreshold = options.confidenceThreshold;
  }
  if (options.useSmartMerging !== undefined) {
    normalized.useSmartMerging = options.useSmartMerging;
  }
  if (options.docId !== undefined) {
    normalized.docId = options.docId;
  }
  if (options.hashSecret !== undefined) {
    normalized.hashSecret = options.hashSecret;
  }
  if (options.detector !== undefined) {
    normalized.detector = options.detector;
  }
  if (options.metadata !== undefined) {
    normalized.metadata = options.metadata;
  }
  return normalized;
}

function normalizeSpans(
  response: NativeSpanResponse,
  options: OpenMedBridgeOptions,
): OpenMedSpan[] {
  const rawSpans = Array.isArray(response)
    ? response
    : Array.isArray(response.spans)
      ? response.spans
      : [];

  return rawSpans
    .map((span) => normalizeSpan(span, options))
    .sort((left, right) => left.start - right.start || left.end - right.end);
}

function normalizeSpan(
  span: NativeSpan,
  options: OpenMedBridgeOptions,
): OpenMedSpan {
  const record = asRecord(span);
  const canonicalLabel =
    stringValue(record.canonical_label) ??
    stringValue(record.canonicalLabel) ??
    stringValue(record.label) ??
    stringValue(record.entity_type) ??
    "OTHER";
  const entityType =
    stringValue(record.entity_type) ??
    stringValue(record.entityType) ??
    stringValue(record.label) ??
    canonicalLabel;

  return {
    schema_version: OPENMED_SPAN_SCHEMA_VERSION,
    doc_id: stringValue(record.doc_id) ?? stringValue(record.docId) ?? "document",
    start: numberValue(record.start) ?? 0,
    end: numberValue(record.end) ?? 0,
    text_hash:
      textHashValue(record.text_hash) ??
      textHashValue(record.textHash) ??
      "sha256:missing-native-text-hash",
    entity_type: entityType,
    canonical_label: canonicalLabel,
    policy_label:
      policyLabelValue(record.policy_label) ??
      policyLabelValue(record.policyLabel) ??
      policyLabelFor(canonicalLabel),
    regulatory_tags: stringArrayValue(record.regulatory_tags) ?? [],
    score: numberValue(record.score) ?? numberValue(record.confidence),
    detector:
      stringValue(record.detector) ??
      options.detector ??
      `${OPENMED_REACT_NATIVE_MODULE_NAME}:${currentPlatform()}`,
    evidence: recordValue(record.evidence) ?? {},
    action: spanActionValue(record.action) ?? "keep",
    replacement: stringValue(record.replacement) ?? null,
    reversible_id:
      stringValue(record.reversible_id) ?? stringValue(record.reversibleId) ?? null,
    section: stringValue(record.section) ?? null,
    metadata: {
      ...recordValue(record.metadata),
      ...options.metadata,
    },
  };
}

function currentPlatform(): OpenMedPlatform {
  const runtimePlatform = loadReactNative()?.Platform?.OS;
  return runtimePlatform === "android" ? "android" : "ios";
}

function defaultBackendForPlatform(): OpenMedBackendKind {
  return currentPlatform() === "android" ? "onnx" : "mlx";
}

function readString(
  record: Record<string, unknown>,
  key: string,
  fallback: string,
): string {
  return stringValue(record[key]) ?? fallback;
}

function readBackend(
  value: unknown,
  fallback: OpenMedBackendKind,
): OpenMedBackendKind {
  return value === "mlx" || value === "coreml" || value === "onnx"
    ? value
    : fallback;
}

function readPlatform(value: unknown, fallback: OpenMedPlatform): OpenMedPlatform {
  return value === "android" || value === "ios" ? value : fallback;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function recordValue(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringArrayValue(value: unknown): string[] | undefined {
  return Array.isArray(value) && value.every((item) => typeof item === "string")
    ? value
    : undefined;
}

function textHashValue(value: unknown): TextHash | undefined {
  return typeof value === "string" &&
    (value.startsWith("hmac-sha256:") || value.startsWith("sha256:"))
    ? (value as TextHash)
    : undefined;
}

function spanActionValue(value: unknown): SpanAction | undefined {
  return value === "keep" ||
    value === "redact" ||
    value === "replace" ||
    value === "mask" ||
    value === "remove" ||
    value === "hash"
    ? value
    : undefined;
}

function policyLabelValue(value: unknown): PolicyLabel | undefined {
  return value === "DIRECT_IDENTIFIER" ||
    value === "QUASI_IDENTIFIER" ||
    value === "CLINICAL_CONCEPT"
    ? value
    : undefined;
}

function policyLabelFor(canonicalLabel: string): PolicyLabel {
  if (CLINICAL_CONCEPT_LABELS.has(canonicalLabel)) {
    return "CLINICAL_CONCEPT";
  }
  if (QUASI_IDENTIFIER_LABELS.has(canonicalLabel)) {
    return "QUASI_IDENTIFIER";
  }
  return "DIRECT_IDENTIFIER";
}

const CLINICAL_CONCEPT_LABELS = new Set([
  "MICROORGANISM",
  "ANTIBIOTIC",
  "SUSCEPTIBILITY",
  "CONDITION",
  "MEDICATION",
  "LAB_TEST",
  "PROCEDURE",
  "BODY_SITE",
  "DIET_TYPE",
  "NUTRITION_TARGET",
  "FEEDING_ROUTE",
  "NUTRITIONAL_STATUS",
  "OTHER",
]);

const QUASI_IDENTIFIER_LABELS = new Set([
  "LOCATION",
  "ZIPCODE",
  "ORDINAL_DIRECTION",
  "DATE",
  "TIME",
  "AGE",
  "CREDIT_CARD_ISSUER",
  "AMOUNT",
  "CURRENCY",
  "GENDER",
  "EYE_COLOR",
  "HEIGHT",
  "ORGANIZATION",
  "JOB_TITLE",
  "JOB_DEPARTMENT",
  "OCCUPATION",
]);
