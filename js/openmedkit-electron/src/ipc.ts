import {
  CANONICAL_LABELS,
  POLICY_LABELS,
  type OpenMedSpan,
} from "openmed";

export const OPENMED_DEIDENTIFY_CHANNEL = "openmed:deidentify" as const;
export const OPENMED_ELECTRON_SCHEMA_VERSION = 1 as const;

const MAX_REQUEST_ID_LENGTH = 80;
const REQUEST_ID_PATTERN = /^[A-Za-z0-9_-]+$/;
const MAX_ENTITY_TYPE_LENGTH = 80;
const ENTITY_TYPE_PATTERN = /^[A-Z][A-Z0-9_:-]*$/;
const CANONICAL_LABEL_SET = new Set<string>(CANONICAL_LABELS);
const POLICY_LABEL_SET = new Set<string>(POLICY_LABELS);
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
  modelPath: string;
}

export interface UtilityDeidentifySuccess {
  type: "deidentify-result";
  requestId: string;
  ok: true;
  spans: RendererOpenMedSpan[];
}

export interface UtilityDeidentifyFailure {
  type: "deidentify-result";
  requestId: string;
  ok: false;
  errorCode: "INVALID_REQUEST" | "INFERENCE_FAILED";
}

export type UtilityDeidentifyResponse =
  | UtilityDeidentifySuccess
  | UtilityDeidentifyFailure;

export interface IpcMainLike<Event = unknown> {
  handle(
    channel: string,
    listener: (event: Event, request: unknown) => Promise<unknown> | unknown,
  ): void;
  removeHandler(channel: string): void;
}

export interface IpcRendererLike {
  invoke(channel: string, request: unknown): Promise<unknown>;
}

export interface ElectronDeidentifyServiceLike {
  deidentify(request: ElectronDeidentifyRequest): Promise<ElectronDeidentifyResponse>;
}

export interface ElectronDeidentifyIpcOptions<Event> {
  isTrustedSender(event: Event): boolean;
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

export function registerElectronDeidentifyIpc<Event>(
  ipcMain: IpcMainLike<Event>,
  service: ElectronDeidentifyServiceLike,
  options: ElectronDeidentifyIpcOptions<Event>,
): () => void {
  ipcMain.handle(OPENMED_DEIDENTIFY_CHANNEL, async (event, request) => {
    if (!isTrustedSender(options, event)) {
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
  assertRendererSpans(spans, text.length);
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
  maxTextLength = 1_000_000,
): asserts request is ElectronDeidentifyRequest {
  if (!isRecord(request)) {
    throw new TypeError("Invalid OpenMed Electron request.");
  }
  if (request.schemaVersion !== OPENMED_ELECTRON_SCHEMA_VERSION) {
    throw new TypeError("Unsupported OpenMed Electron schema version.");
  }
  if (
    typeof request.requestId !== "string" ||
    request.requestId.length === 0 ||
    request.requestId.length > MAX_REQUEST_ID_LENGTH ||
    !REQUEST_ID_PATTERN.test(request.requestId)
  ) {
    throw new TypeError("Invalid OpenMed Electron request identifier.");
  }
  if (
    typeof request.text !== "string" ||
    request.text.length === 0 ||
    request.text.length > maxTextLength
  ) {
    throw new TypeError("Invalid OpenMed Electron request text.");
  }
}

export function isUtilityDeidentifyResponse(
  response: unknown,
): response is UtilityDeidentifyResponse {
  if (
    !isRecord(response) ||
    response.type !== "deidentify-result" ||
    !isValidRequestId(response.requestId) ||
    typeof response.ok !== "boolean"
  ) {
    return false;
  }
  return response.ok
    ? Array.isArray(response.spans) && response.spans.every(isRendererOpenMedSpan)
    : response.errorCode === "INVALID_REQUEST" ||
        response.errorCode === "INFERENCE_FAILED";
}

export function assertRendererSpans(
  spans: readonly RendererOpenMedSpan[],
  textLength: number,
): void {
  if (!Number.isSafeInteger(textLength) || textLength < 0) {
    throw new TypeError("Invalid OpenMed Electron source length.");
  }
  if (spans.length > textLength) {
    throw new TypeError("Invalid OpenMed Electron response spans.");
  }

  let previousEnd = 0;
  for (const span of spans) {
    if (
      !isRendererOpenMedSpan(span) ||
      span.start < previousEnd ||
      span.end > textLength
    ) {
      throw new TypeError("Invalid OpenMed Electron response spans.");
    }
    previousEnd = span.end;
  }
}

function assertElectronDeidentifyResponse(
  response: unknown,
  requestId: string,
  textLength: number,
): asserts response is ElectronDeidentifyResponse {
  if (
    !isRecord(response) ||
    response.schemaVersion !== OPENMED_ELECTRON_SCHEMA_VERSION ||
    response.requestId !== requestId ||
    !Array.isArray(response.spans)
  ) {
    throw new TypeError("Invalid OpenMed Electron response.");
  }
  assertRendererSpans(response.spans, textLength);
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
    CANONICAL_LABEL_SET.has(value.canonical_label as string) &&
    POLICY_LABEL_SET.has(value.policy_label as string) &&
    (value.score === null ||
      (typeof value.score === "number" &&
        Number.isFinite(value.score) &&
        value.score >= 0 &&
        value.score <= 1))
  );
}

function isValidRequestId(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value.length <= MAX_REQUEST_ID_LENGTH &&
    REQUEST_ID_PATTERN.test(value)
  );
}

function isTrustedSender<Event>(
  options: ElectronDeidentifyIpcOptions<Event>,
  event: Event,
): boolean {
  try {
    return options.isTrustedSender(event) === true;
  } catch {
    return false;
  }
}

function nextRendererRequestId(): string {
  const randomUuid = globalThis.crypto?.randomUUID?.();
  if (randomUuid) {
    return `renderer-${randomUuid}`;
  }
  rendererRequestSequence += 1;
  return `renderer-${Date.now()}-${rendererRequestSequence}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
