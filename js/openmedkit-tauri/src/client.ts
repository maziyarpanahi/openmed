import { invoke } from "@tauri-apps/api/core";

export const OPENMED_SPAN_SCHEMA_VERSION = 1 as const;

export type SpanAction =
  | "keep"
  | "redact"
  | "replace"
  | "mask"
  | "remove"
  | "hash"
  | "format_preserve";
export type PolicyLabel =
  | "DIRECT_IDENTIFIER"
  | "QUASI_IDENTIFIER"
  | "CLINICAL_CONCEPT";
export type TextHash = `hmac-sha256:${string}`;

export interface OpenMedSpan {
  schema_version: typeof OPENMED_SPAN_SCHEMA_VERSION;
  doc_id: string;
  start: number;
  end: number;
  text_hash: TextHash;
  entity_type: string;
  canonical_label: string;
  policy_label: PolicyLabel | null;
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

export interface SidecarDeidentifyOptions {
  modelName?: string;
  policy?: string;
  method?: "mask" | "remove" | "replace" | "hash" | "format_preserve";
  confidenceThreshold?: number;
  lang?: string;
  docId?: string;
  useSmartMerging?: boolean;
  useSafetySweep?: boolean;
  deterministicOnly?: boolean;
}

export interface SidecarDeidentifyResult {
  deidentifiedText: string;
  spans: OpenMedSpan[];
}

export interface SidecarPingResult {
  offline: true;
  protocolVersion: typeof OPENMED_SPAN_SCHEMA_VERSION;
}

export type SidecarErrorCode =
  | "INVALID_REQUEST"
  | "PROCESSING_FAILED"
  | "SIDECAR_IO"
  | "SIDECAR_NOT_RUNNING"
  | "SIDECAR_PROTOCOL"
  | "SIDECAR_SPAWN_FAILED"
  | "SIDECAR_TERMINATED";

export class OpenMedSidecarError extends Error {
  readonly code: SidecarErrorCode;

  constructor(code: SidecarErrorCode, message: string) {
    super(message);
    this.name = "OpenMedSidecarError";
    this.code = code;
  }
}

export type TauriInvoke = <T>(
  command: string,
  args?: Record<string, unknown>,
) => Promise<T>;

export class OpenMedTauriClient {
  readonly #invoke: TauriInvoke;

  constructor(invokeFunction: TauriInvoke = invoke) {
    this.#invoke = invokeFunction;
  }

  async ping(): Promise<SidecarPingResult> {
    return this.#call<SidecarPingResult>("openmed_sidecar_ping");
  }

  async deidentify(
    text: string,
    options: SidecarDeidentifyOptions = {},
  ): Promise<SidecarDeidentifyResult> {
    if (typeof text !== "string") {
      throw new OpenMedSidecarError(
        "INVALID_REQUEST",
        "text must be a string",
      );
    }
    const result = await this.#call<SidecarDeidentifyResult>(
      "openmed_sidecar_deidentify",
      { request: { text, options } },
    );
    if (
      typeof result.deidentifiedText !== "string" ||
      !Array.isArray(result.spans)
    ) {
      throw new OpenMedSidecarError(
        "SIDECAR_PROTOCOL",
        "The sidecar returned an invalid response.",
      );
    }
    return result;
  }

  async shutdown(): Promise<void> {
    await this.#call<{ shutdown: boolean }>("openmed_sidecar_shutdown");
  }

  async #call<T>(
    command: string,
    args?: Record<string, unknown>,
  ): Promise<T> {
    try {
      return await this.#invoke<T>(command, args);
    } catch (error: unknown) {
      if (error instanceof OpenMedSidecarError) {
        throw error;
      }
      const payload = asRecord(error);
      const code = readErrorCode(payload.code);
      const message =
        typeof payload.message === "string"
          ? payload.message
          : "The OpenMed sidecar command failed.";
      throw new OpenMedSidecarError(code, message);
    }
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : {};
}

function readErrorCode(value: unknown): SidecarErrorCode {
  switch (value) {
    case "INVALID_REQUEST":
    case "PROCESSING_FAILED":
    case "SIDECAR_IO":
    case "SIDECAR_NOT_RUNNING":
    case "SIDECAR_PROTOCOL":
    case "SIDECAR_SPAWN_FAILED":
    case "SIDECAR_TERMINATED":
      return value;
    default:
      return "SIDECAR_IO";
  }
}

const defaultClient = new OpenMedTauriClient();

export function pingSidecar(): Promise<SidecarPingResult> {
  return defaultClient.ping();
}

export function deidentify(
  text: string,
  options: SidecarDeidentifyOptions = {},
): Promise<SidecarDeidentifyResult> {
  return defaultClient.deidentify(text, options);
}

export function shutdownSidecar(): Promise<void> {
  return defaultClient.shutdown();
}
