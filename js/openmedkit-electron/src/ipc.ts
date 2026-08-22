import type { OpenMedSpan } from "openmed";

export const OPENMED_DEIDENTIFY_CHANNEL = "openmed:deidentify" as const;
export const OPENMED_ELECTRON_SCHEMA_VERSION = 1 as const;
export const OPENMED_ELECTRON_MAX_TEXT_LENGTH = 1_000_000;
export const OPENMED_ELECTRON_MAX_TEXT_BYTES = 3_000_000;
export const OPENMED_ELECTRON_MAX_SPANS = 65_536;
export const OPENMED_UTILITY_MODEL_PATH_ARGUMENT = "--openmed-model-path";

const MAX_REQUEST_ID_LENGTH = 80;
const MAX_ENTITY_TYPE_LENGTH = 128;
const REQUEST_ID_PATTERN = /^[A-Za-z0-9_-]+$/;
const ENTITY_TYPE_PATTERN = /^[A-Za-z][A-Za-z0-9_.:-]*$/;
const CANONICAL_LABEL_PATTERN = /^[A-Z][A-Z0-9_]*$/;
const POLICY_LABEL_SET = new Set([
  "DIRECT_IDENTIFIER",
  "QUASI_IDENTIFIER",
  "CLINICAL_CONCEPT",
]);
let rendererRequestSequence = 0;

export type RendererOpenMedSpan = Pick<
  OpenMedSpan,
  | "schema_version"
  | "start"
  | "end"
  | "entity_type"
  | "canonical_label"
  | "policy_label"
  | "score"
>;

export interface ElectronDeidentifyRequest {
  schemaVersion: typeof OPENMED_ELECTRON_SCHEMA_VERSION;
  requestId: string;
  text: string;
}

export interface ElectronDeidentifyResponse {
  schemaVersion: typeof OPENMED_ELECTRON_SCHEMA_VERSION;
  requestId: string;
  spans: RendererOpenMedSpan[];
}

export interface UtilityDeidentifyRequest extends ElectronDeidentifyRequest {
  type: "deidentify";
}

export interface UtilityDeidentifySuccess {
  type: "deidentify-result";
  schemaVersion: typeof OPENMED_ELECTRON_SCHEMA_VERSION;
  requestId: string;
  ok: true;
  spans: RendererOpenMedSpan[];
}

export interface UtilityDeidentifyFailure {
  type: "deidentify-result";
  schemaVersion: typeof OPENMED_ELECTRON_SCHEMA_VERSION;
  requestId: string;
  ok: false;
  errorCode: "INVALID_REQUEST" | "INFERENCE_FAILED";
}

export type UtilityDeidentifyResponse =
  | UtilityDeidentifySuccess
  | UtilityDeidentifyFailure;

export interface ElectronIpcInvokeEventLike {
  sender: { id: number };
  senderFrame?: unknown;
}

export interface RegisterElectronDeidentifyIpcOptions {
  authorizeSender(event: ElectronIpcInvokeEventLike): boolean;
}

export interface IpcMainLike {
  handle(
    channel: string,
    listener: (event: unknown, request: unknown) => Promise<unknown> | unknown,
  ): void;
  removeHandler(channel: string): void;
}

export interface IpcRendererLike {
  invoke(channel: string, request: unknown): Promise<unknown>;
}

export interface ElectronDeidentifyServiceLike {
  deidentify(request: ElectronDeidentifyRequest): Promise<ElectronDeidentifyResponse>;
}

export function createElectronDeidentifyClient(ipcRenderer: IpcRendererLike): {
  deidentify(text: string): Promise<ElectronDeidentifyResponse>;
} {
  return {
    async deidentify(text: string): Promise<ElectronDeidentifyResponse> {
      const request: ElectronDeidentifyRequest = {
        schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
        requestId: nextRendererRequestId(),
        text,
      };
      assertElectronDeidentifyRequest(request);
      const response = await ipcRenderer.invoke(OPENMED_DEIDENTIFY_CHANNEL, request);
      assertElectronDeidentifyResponse(response, request.requestId, text.length);
      return {
        schemaVersion: response.schemaVersion,
        requestId: response.requestId,
        spans: response.spans.map(toRendererOpenMedSpan),
      };
    },
  };
}

export function registerElectronDeidentifyIpc(
  ipcMain: IpcMainLike,
  service: ElectronDeidentifyServiceLike,
  options: RegisterElectronDeidentifyIpcOptions,
): () => void {
  if (typeof options?.authorizeSender !== "function") {
    throw new TypeError("An OpenMed Electron sender authorizer is required.");
  }
  ipcMain.handle(OPENMED_DEIDENTIFY_CHANNEL, async (event, request) => {
    if (!isAuthorizedEvent(event, options.authorizeSender)) {
      throw new Error("Unauthorized OpenMed Electron IPC sender.");
    }
    assertElectronDeidentifyRequest(request);
    return service.deidentify(request);
  });
  return () => ipcMain.removeHandler(OPENMED_DEIDENTIFY_CHANNEL);
}

export function toRendererOpenMedSpan(
  span: RendererOpenMedSpan,
): RendererOpenMedSpan {
  return {
    schema_version: span.schema_version,
    start: span.start,
    end: span.end,
    entity_type: span.entity_type,
    canonical_label: span.canonical_label,
    policy_label: span.policy_label,
    score: span.score,
  };
}

export function redactTextWithSpans(
  text: string,
  spans: readonly RendererOpenMedSpan[],
): string {
  assertRendererOpenMedSpans(spans, text.length);
  let redacted = text;
  for (const span of [...spans].sort((left, right) => right.start - left.start)) {
    redacted =
      redacted.slice(0, span.start) +
      `[${span.canonical_label}]` +
      redacted.slice(span.end);
  }
  return redacted;
}

export function assertElectronDeidentifyRequest(
  request: unknown,
  maxTextLength = OPENMED_ELECTRON_MAX_TEXT_LENGTH,
): asserts request is ElectronDeidentifyRequest {
  if (
    !Number.isSafeInteger(maxTextLength) ||
    maxTextLength <= 0 ||
    maxTextLength > OPENMED_ELECTRON_MAX_TEXT_LENGTH
  ) {
    throw new TypeError("Invalid OpenMed Electron request text limit.");
  }
  if (!isRecord(request)) {
    throw new TypeError("Invalid OpenMed Electron request.");
  }
  if (request.schemaVersion !== OPENMED_ELECTRON_SCHEMA_VERSION) {
    throw new TypeError("Unsupported OpenMed Electron schema version.");
  }
  if (!isRequestId(request.requestId)) {
    throw new TypeError("Invalid OpenMed Electron request identifier.");
  }
  if (
    typeof request.text !== "string" ||
    request.text.length === 0 ||
    request.text.length > maxTextLength ||
    utf8ByteLength(request.text) >
      Math.min(OPENMED_ELECTRON_MAX_TEXT_BYTES, maxTextLength * 3)
  ) {
    throw new TypeError("Invalid OpenMed Electron request text.");
  }
}

export function assertRendererOpenMedSpans(
  spans: unknown,
  textLength: number,
): asserts spans is RendererOpenMedSpan[] {
  if (
    !Number.isSafeInteger(textLength) ||
    textLength < 0 ||
    textLength > OPENMED_ELECTRON_MAX_TEXT_LENGTH ||
    !Array.isArray(spans) ||
    spans.length > OPENMED_ELECTRON_MAX_SPANS ||
    spans.length > textLength ||
    !spans.every(isRendererOpenMedSpan)
  ) {
    throw new TypeError("Invalid OpenMed Electron renderer spans.");
  }

  const ordered = [...spans].sort((left, right) => left.start - right.start);
  let previousEnd = 0;
  for (const span of ordered) {
    if (span.end > textLength || span.start < previousEnd) {
      throw new TypeError("Invalid OpenMed Electron renderer spans.");
    }
    previousEnd = span.end;
  }
}

export function isUtilityDeidentifyResponse(
  response: unknown,
): response is UtilityDeidentifyResponse {
  if (
    !isRecord(response) ||
    response.type !== "deidentify-result" ||
    response.schemaVersion !== OPENMED_ELECTRON_SCHEMA_VERSION ||
    !isRequestId(response.requestId) ||
    typeof response.ok !== "boolean"
  ) {
    return false;
  }
  return response.ok
    ? Array.isArray(response.spans) &&
        response.spans.length <= OPENMED_ELECTRON_MAX_SPANS &&
        response.spans.every(isRendererOpenMedSpan)
    : response.errorCode === "INVALID_REQUEST" ||
        response.errorCode === "INFERENCE_FAILED";
}

function assertElectronDeidentifyResponse(
  response: unknown,
  requestId: string,
  textLength: number,
): asserts response is ElectronDeidentifyResponse {
  if (
    !isRecord(response) ||
    response.schemaVersion !== OPENMED_ELECTRON_SCHEMA_VERSION ||
    response.requestId !== requestId
  ) {
    throw new TypeError("Invalid OpenMed Electron response.");
  }
  assertRendererOpenMedSpans(response.spans, textLength);
}

function isRendererOpenMedSpan(value: unknown): value is RendererOpenMedSpan {
  if (!isRecord(value)) {
    return false;
  }
  return (
    value.schema_version === 1 &&
    Number.isSafeInteger(value.start) &&
    Number.isSafeInteger(value.end) &&
    (value.start as number) >= 0 &&
    (value.end as number) > (value.start as number) &&
    typeof value.entity_type === "string" &&
    value.entity_type.length > 0 &&
    value.entity_type.length <= MAX_ENTITY_TYPE_LENGTH &&
    ENTITY_TYPE_PATTERN.test(value.entity_type) &&
    typeof value.canonical_label === "string" &&
    value.canonical_label.length <= MAX_ENTITY_TYPE_LENGTH &&
    CANONICAL_LABEL_PATTERN.test(value.canonical_label) &&
    typeof value.policy_label === "string" &&
    POLICY_LABEL_SET.has(value.policy_label) &&
    (value.score === null ||
      (typeof value.score === "number" &&
        Number.isFinite(value.score) &&
        value.score >= 0 &&
        value.score <= 1))
  );
}

function isAuthorizedEvent(
  event: unknown,
  authorizeSender: (event: ElectronIpcInvokeEventLike) => boolean,
): boolean {
  if (
    !isRecord(event) ||
    !isRecord(event.sender) ||
    !Number.isSafeInteger(event.sender.id) ||
    (event.sender.id as number) < 0
  ) {
    return false;
  }
  try {
    return authorizeSender(event as unknown as ElectronIpcInvokeEventLike) === true;
  } catch {
    return false;
  }
}

function isRequestId(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value.length <= MAX_REQUEST_ID_LENGTH &&
    REQUEST_ID_PATTERN.test(value)
  );
}

function nextRendererRequestId(): string {
  const randomUuid = globalThis.crypto?.randomUUID?.();
  if (randomUuid) {
    return `renderer-${randomUuid}`;
  }
  rendererRequestSequence =
    rendererRequestSequence >= Number.MAX_SAFE_INTEGER
      ? 1
      : rendererRequestSequence + 1;
  return `renderer-${Date.now()}-${rendererRequestSequence}`;
}

function utf8ByteLength(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
