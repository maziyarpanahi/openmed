import type { OpenMedSpan } from "openmed";

export const OPENMED_DEIDENTIFY_CHANNEL = "openmed:deidentify" as const;
export const OPENMED_ELECTRON_SCHEMA_VERSION = 1 as const;

const MAX_REQUEST_ID_LENGTH = 80;
const REQUEST_ID_PATTERN = /^[A-Za-z0-9_-]+$/;
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
      const response = await ipcRenderer.invoke(OPENMED_DEIDENTIFY_CHANNEL, request);
      assertElectronDeidentifyResponse(response, request.requestId);
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
): () => void {
  ipcMain.handle(OPENMED_DEIDENTIFY_CHANNEL, async (_event, request) => {
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
  let redacted = text;
  for (const span of [...spans].sort((left, right) => right.start - left.start)) {
    if (span.start < 0 || span.end < span.start || span.end > text.length) {
      throw new Error("OpenMed returned an invalid renderer span.");
    }
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
    typeof response.requestId !== "string" ||
    typeof response.ok !== "boolean"
  ) {
    return false;
  }
  return response.ok
    ? Array.isArray(response.spans) && response.spans.every(isRendererOpenMedSpan)
    : response.errorCode === "INVALID_REQUEST" ||
        response.errorCode === "INFERENCE_FAILED";
}

function assertElectronDeidentifyResponse(
  response: unknown,
  requestId: string,
): asserts response is ElectronDeidentifyResponse {
  if (
    !isRecord(response) ||
    response.schemaVersion !== OPENMED_ELECTRON_SCHEMA_VERSION ||
    response.requestId !== requestId ||
    !Array.isArray(response.spans) ||
    !response.spans.every(isRendererOpenMedSpan)
  ) {
    throw new TypeError("Invalid OpenMed Electron response.");
  }
}

function isRendererOpenMedSpan(value: unknown): value is RendererOpenMedSpan {
  if (!isRecord(value)) {
    return false;
  }
  return (
    value.schema_version === 1 &&
    Number.isInteger(value.start) &&
    Number.isInteger(value.end) &&
    typeof value.entity_type === "string" &&
    typeof value.canonical_label === "string" &&
    typeof value.policy_label === "string" &&
    (value.score === null ||
      (typeof value.score === "number" && Number.isFinite(value.score)))
  );
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
