import {
  CANONICAL_LABELS,
  POLICY_LABELS,
  SPAN_ACTIONS,
  type CanonicalLabel,
  type PolicyLabel,
  type SpanAction,
} from "./generated/contract-types";

// Label, policy and action lists are generated from span.schema.json by
// scripts/codegen/emit_ts_types.py; edit the schema and regenerate instead.
export { CANONICAL_LABELS, POLICY_LABELS, SPAN_ACTIONS };
export type { CanonicalLabel, PolicyLabel, SpanAction };

export const OPENMED_SPAN_SCHEMA_VERSION = 1 as const;

export type TextHash = `hmac-sha256:${string}`;

export interface OpenMedSpan {
  schema_version: typeof OPENMED_SPAN_SCHEMA_VERSION;
  doc_id: string;
  start: number;
  end: number;
  text_hash: TextHash;
  entity_type: string;
  canonical_label: CanonicalLabel;
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

export interface TokenClassificationEntity {
  entity?: string;
  entity_group?: string;
  score?: number;
  word?: string;
  start: number;
  end: number;
  index?: number;
}

/** Unaligned runtime input; public aligned entities retain numeric offsets. */
export type RawTokenClassificationEntity =
  Omit<TokenClassificationEntity, "start" | "end"> & {
    start?: number;
    end?: number;
  };

export type TokenClassificationOutput =
  | TokenClassificationEntity[]
  | TokenClassificationEntity[][];

export type RawTokenClassificationOutput =
  | RawTokenClassificationEntity[]
  | RawTokenClassificationEntity[][];

export interface TokenClassificationCallOptions {
  aggregation_strategy?: "none" | "simple" | "first" | "average" | "max";
  ignore_labels?: string[];
  [key: string]: unknown;
}

export type TokenClassificationPipeline = ((
  text: string,
  options?: TokenClassificationCallOptions,
) => TokenClassificationOutput | Promise<TokenClassificationOutput>) & {
  dispose?: () => Promise<void> | void;
};

export type RawTokenClassificationPipeline = (
  text: string,
  options?: TokenClassificationCallOptions,
) => RawTokenClassificationOutput | Promise<RawTokenClassificationOutput>;

export interface DecodedEntitySpan {
  entity_type: string;
  canonical_label: CanonicalLabel;
  policy_label: PolicyLabel;
  start: number;
  end: number;
  score: number | null;
  token_count: number;
}

export interface ExtractPiiOptions {
  pipeline?: RawTokenClassificationPipeline;
  model?: string;
  modelLoader?: ModelLoader;
  loaderOptions?: LoadModelOptions;
  threshold?: number;
  docId?: string;
  /** Non-empty private key for stable hashes; omitted keys are random per call. */
  hashSecret?: string | Uint8Array;
  detector?: string | null;
  lang?: string;
  section?: string | null;
  regulatoryTags?: string[];
  metadata?: Record<string, unknown>;
  pipelineOptions?: TokenClassificationCallOptions;
  /** Refuse Hub IDs and remote assets before any loader runs. */
  offlineOnly?: boolean;
}

export interface DeidentifyOptions extends ExtractPiiOptions {
  replacement?: (span: OpenMedSpan) => string;
}

export interface OpenMedDeidentifyResult {
  text: string;
  deidentifiedText: string;
  spans: OpenMedSpan[];
}

export interface LoadModelOptions {
  localFilesOnly?: boolean;
  allowRemoteModels?: boolean;
  revision?: string;
  quantized?: boolean;
  dtype?: string;
  device?: string;
  runtime?: RawTransformersRuntime | (() => Promise<RawTransformersRuntime>);
  pipelineOptions?: Record<string, unknown>;
  /** Load from local files only; Hub IDs and remote assets are refused. */
  offlineOnly?: boolean;
  /** Local ONNX Runtime WebAssembly location; replaces the CDN default. */
  wasmPaths?: OnnxWasmPaths;
}

/** Directory prefix, or `{ mjs, wasm }` file locations, for ONNX Runtime Web. */
export type OnnxWasmPaths = string | { mjs?: string; wasm?: string };

export type OpenMedOnnxVariant = "int8" | "fp32" | "fp16";

export interface LoadOnnxModelOptions extends LoadModelOptions {
  variant?: OpenMedOnnxVariant;
}

export interface TransformersRuntime {
  pipeline: (
    task: "token-classification",
    model: string,
    options?: Record<string, unknown>,
  ) => Promise<TokenClassificationPipeline> | TokenClassificationPipeline;
  env?: {
    allowRemoteModels?: boolean;
    allowLocalModels?: boolean;
    [key: string]: unknown;
  };
}

/** Runtime input that may omit offsets; model loaders align their output. */
export interface RawTransformersRuntime extends Omit<TransformersRuntime, "pipeline"> {
  pipeline: (
    task: "token-classification",
    model: string,
    options?: Record<string, unknown>,
  ) => Promise<RawTokenClassificationPipeline> | RawTokenClassificationPipeline;
}

export type ModelLoader = (
  model: string,
  options?: LoadModelOptions,
) => Promise<TokenClassificationPipeline>;
