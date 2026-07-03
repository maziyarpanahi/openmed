export type JsonObject = Record<string, unknown>;

export type FetchLike = (
  input: string,
  init?: RequestInit,
) => Promise<Response>;

export type KeepAliveValue = number | string;

export type AggregationStrategy = "simple" | "first" | "average" | "max";

export type PIILanguage =
  | "en"
  | "fr"
  | "de"
  | "it"
  | "es"
  | "nl"
  | "hi"
  | "te"
  | "pt"
  | "ar"
  | "ja"
  | "tr";

export type DeidentificationMethod =
  | "mask"
  | "remove"
  | "replace"
  | "hash"
  | "shift_dates";

export interface OpenMedClientOptions {
  baseUrl: string;
  fetch?: FetchLike;
}

export interface AnalyzeRequest {
  text: string;
  model_name?: string;
  confidence_threshold?: number | null;
  group_entities?: boolean;
  aggregation_strategy?: AggregationStrategy | null;
  sentence_detection?: boolean;
  sentence_language?: string;
  sentence_clean?: boolean;
  use_fast_tokenizer?: boolean;
  keep_alive?: KeepAliveValue | null;
}

export interface PIIExtractRequest {
  text: string;
  model_name?: string;
  confidence_threshold?: number;
  use_smart_merging?: boolean;
  lang?: PIILanguage;
  normalize_accents?: boolean | null;
  keep_alive?: KeepAliveValue | null;
}

export interface PIIExtractStreamRequest extends PIIExtractRequest {
  chunk_size?: number;
  window_chars?: number;
  tokenizer_context_chars?: number;
  max_entity_chars?: number;
  include_text?: boolean;
}

export interface PIIDeidentifyRequest {
  text: string;
  method?: DeidentificationMethod;
  model_name?: string;
  confidence_threshold?: number;
  keep_year?: boolean;
  shift_dates?: boolean | null;
  date_shift_days?: number | null;
  keep_mapping?: boolean;
  policy?: string | null;
  use_smart_merging?: boolean;
  use_safety_sweep?: boolean;
  lang?: PIILanguage;
  normalize_accents?: boolean | null;
  keep_alive?: KeepAliveValue | null;
}

export interface ModelUnloadRequest {
  model_name?: string | null;
  all?: boolean;
}

export interface EntityPrediction {
  text: string;
  label: string;
  confidence: number;
  start: number | null;
  end: number | null;
  metadata: JsonObject;
}

export interface PredictionResult {
  text: string;
  entities: EntityPrediction[];
  model_name: string;
  timestamp: string;
  processing_time: number | null;
  metadata?: JsonObject | null;
}

export interface PIIDeidentifiedEntity extends EntityPrediction {
  entity_type: string;
  redacted_text: string | null;
  canonical_label: string | null;
  sources: string[];
  evidence: JsonObject;
  threshold: number | null;
  action: string | null;
  surrogate: string | null;
  reversible_id?: string;
}

export interface PIIDeidentifyResponse {
  original_text: string;
  deidentified_text: string;
  pii_entities: PIIDeidentifiedEntity[];
  method: string;
  timestamp: string;
  num_entities_redacted: number;
  metadata: JsonObject;
  audit_report: JsonObject | null;
  mapping?: Record<string, string>;
}

export type AnalyzeResponse = PredictionResult;

export type PIIExtractResponse = PredictionResult;

export type PIIExtractStreamResponse = string;

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  profile: string;
  [key: string]: unknown;
}

export interface ProbeResponse {
  status: string;
  service: string;
  [key: string]: unknown;
}

export type ReadyzResponse = ProbeResponse;

export interface LoadedModelStats {
  models?: number;
  tokenizers?: number;
  pipelines?: number;
  active_requests?: number;
  keep_alive_seconds_remaining?: number | null;
  resident?: boolean;
  [key: string]: unknown;
}

export interface LoadedModelsResponse {
  default_keep_alive_seconds?: number;
  max_resident_models?: number | null;
  warm_models?: string[];
  models?: Record<string, LoadedModelStats>;
  [key: string]: unknown;
}

export interface ModelUnloadResponse {
  unloaded?: boolean;
  model_name?: string;
  released?: Record<string, number>;
  active_requests?: number;
  [key: string]: unknown;
}

export interface OpenMedValidationDetail {
  field: string;
  message: string;
  type: string;
}

export interface OpenMedErrorEnvelope {
  error: {
    code: string;
    message: string;
    details: unknown;
  };
}

export class OpenMedApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: unknown;
  readonly envelope: OpenMedErrorEnvelope;
  readonly error: OpenMedErrorEnvelope["error"];

  constructor(status: number, envelope: OpenMedErrorEnvelope) {
    super(envelope.error.message);
    Object.setPrototypeOf(this, new.target.prototype);
    this.name = "OpenMedApiError";
    this.status = status;
    this.code = envelope.error.code;
    this.details = envelope.error.details;
    this.envelope = envelope;
    this.error = envelope.error;
  }
}

export class OpenMedClient {
  private readonly baseUrl: string;
  private readonly fetchImpl: FetchLike;

  constructor(options: OpenMedClientOptions) {
    const baseUrl = options.baseUrl.trim().replace(/\/+$/, "");
    if (!baseUrl) {
      throw new Error("OpenMedClient requires a non-empty baseUrl.");
    }

    const fetchImpl = options.fetch ?? globalThis.fetch;
    if (typeof fetchImpl !== "function") {
      throw new Error("OpenMedClient requires a fetch implementation.");
    }

    this.baseUrl = baseUrl;
    this.fetchImpl =
      options.fetch ?? (fetchImpl.bind(globalThis) as FetchLike);
  }

  async analyze(request: AnalyzeRequest): Promise<AnalyzeResponse> {
    return this.post("/analyze", request);
  }

  async extractPii(request: PIIExtractRequest): Promise<PIIExtractResponse> {
    return this.post("/pii/extract", request);
  }

  async extractPiiStream(
    request: PIIExtractStreamRequest,
  ): Promise<PIIExtractStreamResponse> {
    return this.post("/pii/extract/stream", request);
  }

  async deidentify(
    request: PIIDeidentifyRequest,
  ): Promise<PIIDeidentifyResponse> {
    return this.post("/pii/deidentify", request);
  }

  async health(): Promise<HealthResponse> {
    return this.get("/health");
  }

  async livez(): Promise<ProbeResponse> {
    return this.get("/livez");
  }

  async readyz(): Promise<ReadyzResponse> {
    return this.get("/readyz");
  }

  async loadedModels(): Promise<LoadedModelsResponse> {
    return this.get("/models/loaded");
  }

  async unloadModels(
    request: ModelUnloadRequest,
  ): Promise<ModelUnloadResponse> {
    return this.post("/models/unload", request);
  }

  private async get<T>(path: string): Promise<T> {
    return this.request<T>(path, { method: "GET" });
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  private async request<T>(path: string, init: RequestInit): Promise<T> {
    const response = await this.fetchImpl(this.url(path), {
      ...init,
      headers: {
        accept: "application/json",
        ...headersToRecord(init.headers),
      },
    });
    const payload = await readPayload(response);

    if (!response.ok) {
      throw new OpenMedApiError(
        response.status,
        toOpenMedErrorEnvelope(payload, response.status),
      );
    }

    return payload as T;
  }

  private url(path: string): string {
    return `${this.baseUrl}${path.startsWith("/") ? path : `/${path}`}`;
  }
}

function headersToRecord(headers?: HeadersInit): Record<string, string> {
  if (!headers) {
    return {};
  }

  if (typeof Headers !== "undefined" && headers instanceof Headers) {
    const result: Record<string, string> = {};
    headers.forEach((value, key) => {
      result[key] = value;
    });
    return result;
  }

  if (Array.isArray(headers)) {
    return Object.fromEntries(headers);
  }

  return { ...(headers as Record<string, string>) };
}

async function readPayload(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

function toOpenMedErrorEnvelope(
  payload: unknown,
  status: number,
): OpenMedErrorEnvelope {
  if (isRecord(payload) && isRecord(payload.error)) {
    const code = payload.error.code;
    const message = payload.error.message;

    if (typeof code === "string" && typeof message === "string") {
      return {
        error: {
          code,
          message,
          details:
            "details" in payload.error ? payload.error.details : null,
        },
      };
    }
  }

  return {
    error: {
      code: "http_error",
      message: `Request failed with status ${status}`,
      details: payload ?? null,
    },
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
