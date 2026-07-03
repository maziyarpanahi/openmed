export const OPENMED_SPAN_SCHEMA_VERSION = 1 as const;

export const CANONICAL_LABELS = [
  "ACCOUNT_NUMBER",
  "AGE",
  "AMOUNT",
  "API_KEY",
  "BIC",
  "BITCOIN_ADDRESS",
  "BUILDING_NUMBER",
  "CREDIT_CARD",
  "CREDIT_CARD_ISSUER",
  "CURRENCY",
  "CVV",
  "DATE",
  "DATE_OF_BIRTH",
  "EMAIL",
  "ETHEREUM_ADDRESS",
  "EYE_COLOR",
  "FIRST_NAME",
  "GENDER",
  "GPS_COORDINATES",
  "HEIGHT",
  "IBAN",
  "ID_NUM",
  "IMEI",
  "IP_ADDRESS",
  "JOB_DEPARTMENT",
  "JOB_TITLE",
  "LAST_NAME",
  "LITECOIN_ADDRESS",
  "LOCATION",
  "MAC_ADDRESS",
  "MASKED_NUMBER",
  "MIDDLE_NAME",
  "OCCUPATION",
  "ORDINAL_DIRECTION",
  "ORGANIZATION",
  "OTHER",
  "PASSWORD",
  "PERSON",
  "PHONE",
  "PIN",
  "PREFIX",
  "SSN",
  "STREET_ADDRESS",
  "TIME",
  "URL",
  "USER_AGENT",
  "USERNAME",
  "VEHICLE_REGISTRATION",
  "VIN",
  "ZIPCODE"
] as const;

export type CanonicalLabel = (typeof CANONICAL_LABELS)[number];

export const POLICY_LABELS = [
  "DIRECT_IDENTIFIER",
  "QUASI_IDENTIFIER",
  "CLINICAL_CONCEPT"
] as const;

export type PolicyLabel = (typeof POLICY_LABELS)[number];

export const SPAN_ACTIONS = [
  "keep",
  "redact",
  "replace",
  "mask",
  "hash",
  "format_preserve"
] as const;

export type SpanAction = (typeof SPAN_ACTIONS)[number];

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

export type TokenClassificationOutput =
  | TokenClassificationEntity[]
  | TokenClassificationEntity[][];

export interface TokenClassificationCallOptions {
  aggregation_strategy?: "none" | "simple" | "first" | "average" | "max";
  ignore_labels?: string[];
  [key: string]: unknown;
}

export type TokenClassificationPipeline = (
  text: string,
  options?: TokenClassificationCallOptions,
) => TokenClassificationOutput | Promise<TokenClassificationOutput>;

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
  pipeline?: TokenClassificationPipeline;
  model?: string;
  modelLoader?: ModelLoader;
  loaderOptions?: LoadModelOptions;
  threshold?: number;
  docId?: string;
  hashSecret?: string | Uint8Array;
  detector?: string | null;
  lang?: string;
  section?: string | null;
  regulatoryTags?: string[];
  metadata?: Record<string, unknown>;
  pipelineOptions?: TokenClassificationCallOptions;
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
  runtime?: TransformersRuntime | (() => Promise<TransformersRuntime>);
  pipelineOptions?: Record<string, unknown>;
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

export type ModelLoader = (
  model: string,
  options?: LoadModelOptions,
) => Promise<TokenClassificationPipeline>;
