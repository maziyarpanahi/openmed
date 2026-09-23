import { invoke } from "@tauri-apps/api/core";

export const OPENMED_SPAN_SCHEMA_VERSION = 1 as const;
export const OPENMED_TAURI_MAX_TEXT_CHARS = 1_000_000;
export const OPENMED_TAURI_MAX_TEXT_BYTES = 4_000_000;
export const OPENMED_TAURI_MAX_SPANS = 65_536;

const MAX_DEIDENTIFIED_TEXT_CHARS = 8_000_000;
const MAX_DEIDENTIFIED_TEXT_BYTES = 32_000_000;
const MAX_DOC_ID_CHARS = 256;
const MAX_SHORT_FIELD_CHARS = 128;
const TEXT_HASH_PATTERN = /^hmac-sha256:[0-9a-f]{64}$/;
const TOKEN_PATTERN = /^[A-Za-z0-9_.:-]{1,128}$/;
const CANONICAL_LABEL_PATTERN = /^[A-Z0-9_]{1,128}$/;
const POLICY_LABELS = new Set([
  "DIRECT_IDENTIFIER",
  "QUASI_IDENTIFIER",
  "CLINICAL_CONCEPT",
]);
const SPAN_ACTIONS = new Set<SpanAction>([
  "keep",
  "redact",
  "replace",
  "mask",
  "hash",
  "format_preserve",
]);
const OPTION_KEYS = new Set([
  "policy",
  "method",
  "confidenceThreshold",
  "lang",
  "docId",
  "useSmartMerging",
  "useSafetySweep",
  "deterministicOnly",
]);

export type SpanAction =
  | "keep"
  | "redact"
  | "replace"
  | "mask"
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
  | "SIDECAR_BUSY"
  | "SIDECAR_CONFIGURATION"
  | "SIDECAR_NOT_RUNNING"
  | "SIDECAR_PROTOCOL"
  | "SIDECAR_SPAWN_FAILED"
  | "SIDECAR_TERMINATED"
  | "SIDECAR_TIMEOUT";

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
    const result = await this.#call<SidecarPingResult>("openmed_sidecar_ping");
    if (
      !isRecord(result) ||
      result.offline !== true ||
      result.protocolVersion !== OPENMED_SPAN_SCHEMA_VERSION
    ) {
      throw protocolError();
    }
    return { offline: true, protocolVersion: OPENMED_SPAN_SCHEMA_VERSION };
  }

  async deidentify(
    text: string,
    options: SidecarDeidentifyOptions = {},
  ): Promise<SidecarDeidentifyResult> {
    assertDeidentifyRequest(text, options);
    const result = await this.#call<SidecarDeidentifyResult>(
      "openmed_sidecar_deidentify",
      { request: { text, options } },
    );
    assertDeidentifyResult(result, codePointLength(text));
    return {
      deidentifiedText: result.deidentifiedText,
      spans: result.spans.map(copySpan),
    };
  }

  async shutdown(): Promise<void> {
    const result = await this.#call<{ shutdown: boolean }>(
      "openmed_sidecar_shutdown",
    );
    if (!isRecord(result) || result.shutdown !== true) {
      throw protocolError();
    }
  }

  async #call<T>(
    command: string,
    args?: Record<string, unknown>,
  ): Promise<T> {
    try {
      return await this.#invoke<T>(command, args);
    } catch (error: unknown) {
      const payload = asRecord(error);
      const code = readErrorCode(payload.code);
      throw new OpenMedSidecarError(code, safeErrorMessage(code));
    }
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readErrorCode(value: unknown): SidecarErrorCode {
  switch (value) {
    case "INVALID_REQUEST":
    case "PROCESSING_FAILED":
    case "SIDECAR_BUSY":
    case "SIDECAR_CONFIGURATION":
    case "SIDECAR_IO":
    case "SIDECAR_NOT_RUNNING":
    case "SIDECAR_PROTOCOL":
    case "SIDECAR_SPAWN_FAILED":
    case "SIDECAR_TERMINATED":
    case "SIDECAR_TIMEOUT":
      return value;
    default:
      return "SIDECAR_IO";
  }
}

function safeErrorMessage(code: SidecarErrorCode): string {
  switch (code) {
    case "INVALID_REQUEST":
      return "The OpenMed sidecar request is invalid.";
    case "PROCESSING_FAILED":
      return "OpenMed de-identification failed; verify the local model bundle.";
    case "SIDECAR_BUSY":
      return "The OpenMed sidecar is already processing a request.";
    case "SIDECAR_CONFIGURATION":
      return "The OpenMed sidecar configuration is invalid.";
    case "SIDECAR_NOT_RUNNING":
    case "SIDECAR_TERMINATED":
      return "The OpenMed sidecar terminated before responding.";
    case "SIDECAR_PROTOCOL":
      return "The OpenMed sidecar returned an invalid response.";
    case "SIDECAR_SPAWN_FAILED":
      return "The OpenMed sidecar process could not be started.";
    case "SIDECAR_TIMEOUT":
      return "The OpenMed sidecar did not respond before the deadline.";
    case "SIDECAR_IO":
      return "The OpenMed sidecar command failed.";
  }
}

function assertDeidentifyRequest(
  text: unknown,
  options: unknown,
): asserts text is string {
  if (
    typeof text !== "string" ||
    text.length === 0 ||
    text.length > OPENMED_TAURI_MAX_TEXT_CHARS * 2 ||
    hasUnpairedSurrogate(text) ||
    codePointLength(text) > OPENMED_TAURI_MAX_TEXT_CHARS ||
    utf8ByteLength(text) > OPENMED_TAURI_MAX_TEXT_BYTES ||
    !isRecord(options)
  ) {
    throw invalidRequestError();
  }
  if (Object.keys(options).some((key) => !OPTION_KEYS.has(key))) {
    throw invalidRequestError();
  }
  if (
    !validOptionalString(options.policy, 128) ||
    !validOptionalString(options.method, 32) ||
    !validOptionalString(options.lang, 16) ||
    !validOptionalString(options.docId, MAX_DOC_ID_CHARS) ||
    !validOptionalBoolean(options.useSmartMerging) ||
    !validOptionalBoolean(options.useSafetySweep) ||
    !validOptionalBoolean(options.deterministicOnly) ||
    (options.confidenceThreshold !== undefined &&
      (typeof options.confidenceThreshold !== "number" ||
        !Number.isFinite(options.confidenceThreshold) ||
        options.confidenceThreshold < 0 ||
        options.confidenceThreshold > 1)) ||
    (options.method !== undefined &&
      options.method !== "mask" &&
      options.method !== "remove" &&
      options.method !== "replace" &&
      options.method !== "hash" &&
      options.method !== "format_preserve")
  ) {
    throw invalidRequestError();
  }
}

function assertDeidentifyResult(
  value: unknown,
  sourceCharacters: number,
): asserts value is SidecarDeidentifyResult {
  if (
    !isRecord(value) ||
    typeof value.deidentifiedText !== "string" ||
    value.deidentifiedText.length > MAX_DEIDENTIFIED_TEXT_CHARS * 2 ||
    hasUnpairedSurrogate(value.deidentifiedText) ||
    codePointLength(value.deidentifiedText) > MAX_DEIDENTIFIED_TEXT_CHARS ||
    utf8ByteLength(value.deidentifiedText) > MAX_DEIDENTIFIED_TEXT_BYTES ||
    !Array.isArray(value.spans) ||
    value.spans.length > OPENMED_TAURI_MAX_SPANS ||
    value.spans.length > sourceCharacters ||
    !value.spans.every((span) => isOpenMedSpan(span, sourceCharacters))
  ) {
    throw protocolError();
  }
  const ordered = [...value.spans].sort(
    (left, right) => left.start - right.start || left.end - right.end,
  );
  let previousEnd = 0;
  for (const span of ordered) {
    if (span.start < previousEnd) {
      throw protocolError();
    }
    previousEnd = span.end;
  }
}

function isOpenMedSpan(
  value: unknown,
  sourceCharacters: number,
): value is OpenMedSpan {
  if (!isRecord(value)) {
    return false;
  }
  return (
    value.schema_version === OPENMED_SPAN_SCHEMA_VERSION &&
    typeof value.doc_id === "string" &&
    value.doc_id.length > 0 &&
    codePointLength(value.doc_id) <= MAX_DOC_ID_CHARS &&
    Number.isSafeInteger(value.start) &&
    Number.isSafeInteger(value.end) &&
    (value.start as number) >= 0 &&
    (value.end as number) > (value.start as number) &&
    (value.end as number) <= sourceCharacters &&
    typeof value.text_hash === "string" &&
    TEXT_HASH_PATTERN.test(value.text_hash) &&
    typeof value.entity_type === "string" &&
    TOKEN_PATTERN.test(value.entity_type) &&
    typeof value.canonical_label === "string" &&
    CANONICAL_LABEL_PATTERN.test(value.canonical_label) &&
    (value.policy_label === null ||
      (typeof value.policy_label === "string" &&
        POLICY_LABELS.has(value.policy_label))) &&
    Array.isArray(value.regulatory_tags) &&
    value.regulatory_tags.length <= 64 &&
    value.regulatory_tags.every(
      (tag) =>
        typeof tag === "string" &&
        tag.length > 0 &&
        codePointLength(tag) <= MAX_SHORT_FIELD_CHARS,
    ) &&
    (value.score === null ||
      (typeof value.score === "number" &&
        Number.isFinite(value.score) &&
        value.score >= 0 &&
        value.score <= 1)) &&
    validNullableString(value.detector, MAX_SHORT_FIELD_CHARS, false) &&
    isRecord(value.evidence) &&
    typeof value.action === "string" &&
    SPAN_ACTIONS.has(value.action as SpanAction) &&
    validNullableString(value.replacement, 4_096, true) &&
    validNullableString(value.reversible_id, 512, false) &&
    validNullableString(value.section, 256, false) &&
    isRecord(value.metadata)
  );
}

function copySpan(span: OpenMedSpan): OpenMedSpan {
  return {
    schema_version: span.schema_version,
    doc_id: span.doc_id,
    start: span.start,
    end: span.end,
    text_hash: span.text_hash,
    entity_type: span.entity_type,
    canonical_label: span.canonical_label,
    policy_label: span.policy_label,
    regulatory_tags: [...span.regulatory_tags],
    score: span.score,
    detector: span.detector,
    evidence: { ...span.evidence },
    action: span.action,
    replacement: span.replacement,
    reversible_id: span.reversible_id,
    section: span.section,
    metadata: { ...span.metadata },
  };
}

function validOptionalString(value: unknown, maximum: number): boolean {
  return (
    value === undefined ||
    (typeof value === "string" &&
      value.length > 0 &&
      !hasUnpairedSurrogate(value) &&
      codePointLength(value) <= maximum)
  );
}

function validNullableString(
  value: unknown,
  maximum: number,
  allowEmpty: boolean,
): boolean {
  return (
    value === null ||
    (typeof value === "string" &&
      (allowEmpty || value.length > 0) &&
      !hasUnpairedSurrogate(value) &&
      codePointLength(value) <= maximum)
  );
}

function validOptionalBoolean(value: unknown): boolean {
  return value === undefined || typeof value === "boolean";
}

function codePointLength(value: string): number {
  let length = 0;
  for (const _character of value) {
    length += 1;
  }
  return length;
}

function utf8ByteLength(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}

function hasUnpairedSurrogate(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (next < 0xdc00 || next > 0xdfff) {
        return true;
      }
      index += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      return true;
    }
  }
  return false;
}

function invalidRequestError(): OpenMedSidecarError {
  return new OpenMedSidecarError(
    "INVALID_REQUEST",
    safeErrorMessage("INVALID_REQUEST"),
  );
}

function protocolError(): OpenMedSidecarError {
  return new OpenMedSidecarError(
    "SIDECAR_PROTOCOL",
    safeErrorMessage("SIDECAR_PROTOCOL"),
  );
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
