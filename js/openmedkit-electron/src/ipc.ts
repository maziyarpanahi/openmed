import {
  CANONICAL_LABELS,
  POLICY_LABELS,
  type OpenMedSpan,
} from "openmed";

import { findWebPreferenceRelaxations } from "./window-hardening";

export const OPENMED_DEIDENTIFY_CHANNEL = "openmed:deidentify" as const;
export const OPENMED_CANCEL_CHANNEL = "openmed:deidentify:cancel" as const;
export const OPENMED_ELECTRON_SCHEMA_VERSION = 1 as const;

const MAX_REQUEST_ID_LENGTH = 80;
const REQUEST_ID_PATTERN = /^[A-Za-z0-9_-]+$/;
const MAX_ENTITY_TYPE_LENGTH = 80;
const ENTITY_TYPE_PATTERN = /^[A-Z][A-Z0-9_:-]*$/;
const CONTENT_ID_PATTERN = /^sha256:[0-9a-f]{64}$/;
const MAX_OWNER_LENGTH = 128;
const REQUEST_KEYS = ["requestId", "schemaVersion", "text"] as const;
const CANCEL_KEYS = ["requestId", "schemaVersion"] as const;
const CANONICAL_LABEL_SET = new Set<string>(CANONICAL_LABELS);
const POLICY_LABEL_SET = new Set<string>(POLICY_LABELS);
let rendererRequestSequence = 0;

/**
 * Stable, value-free error codes shared by the main process and renderers.
 *
 * Every failure that crosses the IPC boundary is reported as one of these
 * codes. Messages are fixed strings and never include request text, paths,
 * model output or exception detail.
 */
export const ELECTRON_DEIDENTIFY_ERROR_CODES = [
  "INVALID_REQUEST",
  "UNTRUSTED_SENDER",
  "DUPLICATE_REQUEST",
  "QUEUE_FULL",
  "CANCELLED",
  "REQUEST_TIMEOUT",
  "SERVICE_DISPOSED",
  "BUNDLE_UNAVAILABLE",
  "BUNDLE_REJECTED",
  "BUNDLE_ACTIVATION_TIMEOUT",
  "UTILITY_PROCESS_START_FAILED",
  "UTILITY_PROCESS_SEND_FAILED",
  "UTILITY_PROCESS_EXITED",
  "INVALID_UTILITY_RESPONSE",
  "INFERENCE_FAILED",
  "INVALID_RESPONSE",
  "INTERNAL_ERROR",
] as const;

export type ElectronDeidentifyErrorCode =
  (typeof ELECTRON_DEIDENTIFY_ERROR_CODES)[number];

const ERROR_CODE_SET = new Set<string>(ELECTRON_DEIDENTIFY_ERROR_CODES);

const ERROR_MESSAGES: Record<ElectronDeidentifyErrorCode, string> = {
  INVALID_REQUEST: "Invalid OpenMed Electron request.",
  UNTRUSTED_SENDER: "Unauthorized OpenMed Electron IPC sender.",
  DUPLICATE_REQUEST: "Duplicate OpenMed Electron request identifier.",
  QUEUE_FULL: "The OpenMed de-identification queue is full.",
  CANCELLED: "The OpenMed de-identification request was cancelled.",
  REQUEST_TIMEOUT: "OpenMed de-identification timed out.",
  SERVICE_DISPOSED: "The OpenMed de-identification service is disposed.",
  BUNDLE_UNAVAILABLE: "No verified OpenMed model bundle is active.",
  BUNDLE_REJECTED: "The OpenMed model bundle failed verification.",
  BUNDLE_ACTIVATION_TIMEOUT: "OpenMed model bundle activation timed out.",
  UTILITY_PROCESS_START_FAILED: "OpenMed utility process could not be started.",
  UTILITY_PROCESS_SEND_FAILED: "OpenMed utility process communication failed.",
  UTILITY_PROCESS_EXITED: "OpenMed utility process exited.",
  INVALID_UTILITY_RESPONSE:
    "OpenMed utility process returned an invalid response.",
  INFERENCE_FAILED: "OpenMed de-identification failed.",
  INVALID_RESPONSE: "Invalid OpenMed Electron response.",
  INTERNAL_ERROR: "OpenMed de-identification failed unexpectedly.",
};

/** Error carrying only a controlled {@link ElectronDeidentifyErrorCode}. */
export class ElectronDeidentifyError extends Error {
  readonly code: ElectronDeidentifyErrorCode;

  constructor(code: ElectronDeidentifyErrorCode) {
    super(ERROR_MESSAGES[code]);
    this.name = "ElectronDeidentifyError";
    this.code = code;
  }
}

/** Return whether `value` is a known OpenMed Electron error code. */
export function isElectronDeidentifyErrorCode(
  value: unknown,
): value is ElectronDeidentifyErrorCode {
  return typeof value === "string" && ERROR_CODE_SET.has(value);
}

/**
 * Map any thrown value to a controlled error code.
 *
 * Only {@link ElectronDeidentifyError} instances keep their code; anything
 * else becomes `fallback`, so exception text never reaches a renderer.
 */
export function electronErrorCodeOf(
  error: unknown,
  fallback: ElectronDeidentifyErrorCode = "INTERNAL_ERROR",
): ElectronDeidentifyErrorCode {
  return error instanceof ElectronDeidentifyError ? error.code : fallback;
}

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

/** The verified bundle generation that produced a result. */
export interface ElectronRendererBundleRef {
  generation: number;
  contentId: string;
}

export interface ElectronDeidentifyRequest {
  schemaVersion: typeof OPENMED_ELECTRON_SCHEMA_VERSION;
  requestId: string;
  text: string;
}

export interface ElectronDeidentifyResponse {
  schemaVersion: typeof OPENMED_ELECTRON_SCHEMA_VERSION;
  requestId: string;
  ok: true;
  spans: RendererOpenMedSpan[];
  /** `null` when the service runs an unverified legacy model directory. */
  bundle: ElectronRendererBundleRef | null;
}

export interface ElectronDeidentifyFailureResponse {
  schemaVersion: typeof OPENMED_ELECTRON_SCHEMA_VERSION;
  requestId: string;
  ok: false;
  errorCode: ElectronDeidentifyErrorCode;
}

export type ElectronDeidentifyIpcResponse =
  | ElectronDeidentifyResponse
  | ElectronDeidentifyFailureResponse;

export interface ElectronCancelRequest {
  schemaVersion: typeof OPENMED_ELECTRON_SCHEMA_VERSION;
  requestId: string;
}

export interface ElectronCancelResponse {
  schemaVersion: typeof OPENMED_ELECTRON_SCHEMA_VERSION;
  requestId: string;
  cancelled: boolean;
}

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

/** Per-call options the IPC layer passes to the service. */
export interface ElectronDeidentifyCallOptions {
  /** Opaque key of the renderer that owns the request. */
  owner?: string;
}

export interface ElectronDeidentifyServiceLike {
  deidentify(
    request: ElectronDeidentifyRequest,
    options?: ElectronDeidentifyCallOptions,
  ): Promise<ElectronDeidentifyResponse>;
  cancel?(requestId: string, owner?: string): boolean;
  cancelOwner?(owner: string): number;
}

/** The subset of an IPC event's `sender` used for ownership and hardening. */
export interface ElectronIpcSenderLike {
  id?: unknown;
  getLastWebPreferences?(): unknown;
  once?(event: "destroyed", listener: () => void): unknown;
  on?(event: "render-process-gone", listener: () => void): unknown;
}

export interface ElectronDeidentifyIpcOptions<Event> {
  /** App-specific authorization (sending window and main frame). */
  isTrustedSender(event: Event): boolean;
  /**
   * Opaque owner key for a sender. Defaults to `event.sender.id`. Requests are
   * scoped by owner, and an owner's work is cancelled when its renderer is
   * destroyed or crashes.
   */
  ownerOf?(event: Event): string;
}

export interface ElectronDeidentifyTask {
  readonly requestId: string;
  /** Resolves with spans or rejects with an {@link ElectronDeidentifyError}. */
  readonly result: Promise<ElectronDeidentifyResponse>;
  /**
   * Never rejects: resolves with the success response or the typed failure
   * response. Use it from pages behind `contextBridge`, which copies errors
   * without their `code` property.
   */
  readonly outcome: Promise<ElectronDeidentifyIpcResponse>;
  /** Cancel the request; resolves `true` when the main process dropped it. */
  cancel(): Promise<boolean>;
}

export interface ElectronDeidentifyClientOptions {
  signal?: AbortSignal;
}

export interface ElectronDeidentifyClient {
  deidentify(
    text: string,
    options?: ElectronDeidentifyClientOptions,
  ): Promise<ElectronDeidentifyResponse>;
  submit(text: string): ElectronDeidentifyTask;
}

/**
 * Create the renderer client for the preload bridge.
 *
 * The client exposes only `deidentify` and `submit`. Callers never choose
 * model paths, bundles, runtimes or policy: requests carry the schema
 * version, an opaque request id and the text. Every failure rejects with an
 * {@link ElectronDeidentifyError} carrying a controlled code.
 *
 * Args:
 *   ipcRenderer: Electron's `ipcRenderer` (or a compatible object).
 *
 * Returns:
 *   A client whose methods can be exposed through `contextBridge`.
 */
export function createElectronDeidentifyClient(
  ipcRenderer: IpcRendererLike,
): ElectronDeidentifyClient {
  const submit = (text: string): ElectronDeidentifyTask => {
    const requestId = nextRendererRequestId();
    const request: ElectronDeidentifyRequest = {
      schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
      requestId,
      text,
    };
    let settled = false;
    let rejectCancelled: (error: Error) => void = () => undefined;
    const cancelled = new Promise<never>((_resolve, reject) => {
      rejectCancelled = reject;
    });
    const invocation = (async () => {
      try {
        assertElectronDeidentifyRequest(request);
      } catch {
        throw new ElectronDeidentifyError("INVALID_REQUEST");
      }
      let response: unknown;
      try {
        response = await ipcRenderer.invoke(OPENMED_DEIDENTIFY_CHANNEL, request);
      } catch {
        throw new ElectronDeidentifyError("INTERNAL_ERROR");
      }
      return parseElectronDeidentifyResponse(response, requestId, text.length);
    })();
    const result = Promise.race([invocation, cancelled]).finally(() => {
      settled = true;
    });
    // The race observes both promises; a dropped invocation never surfaces.
    invocation.catch(() => undefined);
    cancelled.catch(() => undefined);
    const outcome = result.then(
      (response): ElectronDeidentifyIpcResponse => response,
      (error: unknown) => failureResponse(requestId, electronErrorCodeOf(error)),
    );
    return {
      requestId,
      result,
      outcome,
      async cancel(): Promise<boolean> {
        if (settled) {
          return false;
        }
        rejectCancelled(new ElectronDeidentifyError("CANCELLED"));
        const cancelRequest: ElectronCancelRequest = {
          schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
          requestId,
        };
        try {
          const ack = await ipcRenderer.invoke(OPENMED_CANCEL_CHANNEL, cancelRequest);
          return (
            isRecord(ack) &&
            ack.schemaVersion === OPENMED_ELECTRON_SCHEMA_VERSION &&
            ack.requestId === requestId &&
            ack.cancelled === true
          );
        } catch {
          return false;
        }
      },
    };
  };

  return {
    submit,
    async deidentify(
      text: string,
      options: ElectronDeidentifyClientOptions = {},
    ): Promise<ElectronDeidentifyResponse> {
      const signal = options.signal;
      if (signal?.aborted) {
        throw new ElectronDeidentifyError("CANCELLED");
      }
      const task = submit(text);
      const onAbort = () => {
        void task.cancel();
      };
      signal?.addEventListener("abort", onAbort, { once: true });
      try {
        return await task.result;
      } finally {
        signal?.removeEventListener("abort", onAbort);
      }
    },
  };
}

/**
 * Register the OpenMed de-identification and cancel channels on `ipcMain`.
 *
 * A sender must pass `isTrustedSender` and its resolved web preferences must
 * be hardened (see `findWebPreferenceRelaxations`); otherwise the request is
 * refused with `UNTRUSTED_SENDER` before any inference. Requests with
 * unknown members (for example a model path, bundle, runtime or policy) are
 * refused with `INVALID_REQUEST`. Failures are returned as typed responses
 * carrying a controlled code. A renderer's pending work is cancelled when it
 * is destroyed or its process is gone.
 *
 * Args:
 *   ipcMain: Electron's `ipcMain` (or a compatible object).
 *   service: The de-identification service.
 *   options: Authorization and ownership hooks.
 *
 * Returns:
 *   A function that removes both channel handlers.
 */
export function registerElectronDeidentifyIpc<Event>(
  ipcMain: IpcMainLike<Event>,
  service: ElectronDeidentifyServiceLike,
  options: ElectronDeidentifyIpcOptions<Event>,
): () => void {
  let active = true;
  const watched = new Set<string>();

  const authorize = (event: Event): string | null => {
    if (!isTrustedSender(options, event) || !hasHardenedSender(event)) {
      return null;
    }
    const owner = ownerOf(options, event);
    if (owner !== null) {
      watchOwner(event, owner);
    }
    return owner;
  };

  const watchOwner = (event: Event, owner: string): void => {
    if (watched.has(owner)) {
      return;
    }
    const sender = senderOf(event);
    const release = (forget: boolean) => () => {
      if (forget) {
        watched.delete(owner);
      }
      if (active) {
        try {
          service.cancelOwner?.(owner);
        } catch {
          // Teardown is best-effort; the service bounds every request anyway.
        }
      }
    };
    try {
      sender?.once?.("destroyed", release(true));
      sender?.on?.("render-process-gone", release(false));
      watched.add(owner);
    } catch {
      // A sender without lifecycle events still has request timeouts.
    }
  };

  ipcMain.handle(OPENMED_DEIDENTIFY_CHANNEL, async (event, request) => {
    const requestId = safeRequestId(request);
    const owner = authorize(event);
    if (owner === null) {
      return failureResponse(requestId, "UNTRUSTED_SENDER");
    }
    try {
      assertElectronDeidentifyRequest(request);
    } catch {
      return failureResponse(requestId, "INVALID_REQUEST");
    }
    try {
      const response = await service.deidentify(
        {
          schemaVersion: request.schemaVersion,
          requestId: request.requestId,
          text: request.text,
        },
        { owner },
      );
      return {
        schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
        requestId: response.requestId,
        ok: true,
        spans: response.spans.map(toRendererOpenMedSpan),
        bundle: response.bundle
          ? {
              generation: response.bundle.generation,
              contentId: response.bundle.contentId,
            }
          : null,
      } satisfies ElectronDeidentifyResponse;
    } catch (error) {
      return failureResponse(requestId, electronErrorCodeOf(error));
    }
  });

  ipcMain.handle(OPENMED_CANCEL_CHANNEL, async (event, request) => {
    const requestId = safeRequestId(request);
    const owner = authorize(event);
    if (owner === null) {
      return failureResponse(requestId, "UNTRUSTED_SENDER");
    }
    if (!isExactRecord(request, CANCEL_KEYS) || !isValidCancelRequest(request)) {
      return failureResponse(requestId, "INVALID_REQUEST");
    }
    let cancelled = false;
    try {
      cancelled = service.cancel?.(request.requestId, owner) === true;
    } catch {
      cancelled = false;
    }
    return {
      schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
      requestId: request.requestId,
      cancelled,
    } satisfies ElectronCancelResponse;
  });

  return () => {
    active = false;
    ipcMain.removeHandler(OPENMED_DEIDENTIFY_CHANNEL);
    ipcMain.removeHandler(OPENMED_CANCEL_CHANNEL);
  };
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

/**
 * Validate a renderer request.
 *
 * The request must be a plain object with exactly `schemaVersion`,
 * `requestId` and `text`. Any other member is refused, so a renderer cannot
 * smuggle a model path, bundle, runtime or policy choice to the main process.
 */
export function assertElectronDeidentifyRequest(
  request: unknown,
  maxTextLength = 1_000_000,
): asserts request is ElectronDeidentifyRequest {
  if (!isExactRecord(request, REQUEST_KEYS)) {
    throw new TypeError("Invalid OpenMed Electron request.");
  }
  if (request.schemaVersion !== OPENMED_ELECTRON_SCHEMA_VERSION) {
    throw new TypeError("Unsupported OpenMed Electron schema version.");
  }
  if (!isValidRequestId(request.requestId)) {
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

export function assertRendererSpans(
  spans: readonly RendererOpenMedSpan[],
  textLength: number,
): void {
  if (!Number.isSafeInteger(textLength) || textLength < 0) {
    throw new TypeError("Invalid OpenMed Electron source length.");
  }
  if (!Array.isArray(spans) || spans.length > textLength) {
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

export function isRendererOpenMedSpan(value: unknown): value is RendererOpenMedSpan {
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

export function isValidRequestId(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value.length <= MAX_REQUEST_ID_LENGTH &&
    REQUEST_ID_PATTERN.test(value)
  );
}

export function isBundleContentId(value: unknown): value is string {
  return typeof value === "string" && CONTENT_ID_PATTERN.test(value);
}

function parseElectronDeidentifyResponse(
  response: unknown,
  requestId: string,
  textLength: number,
): ElectronDeidentifyResponse {
  if (
    !isRecord(response) ||
    response.schemaVersion !== OPENMED_ELECTRON_SCHEMA_VERSION ||
    response.requestId !== requestId
  ) {
    throw new ElectronDeidentifyError("INVALID_RESPONSE");
  }
  if (response.ok === false) {
    throw new ElectronDeidentifyError(
      isElectronDeidentifyErrorCode(response.errorCode)
        ? response.errorCode
        : "INVALID_RESPONSE",
    );
  }
  if (response.ok !== true || !Array.isArray(response.spans)) {
    throw new ElectronDeidentifyError("INVALID_RESPONSE");
  }
  const bundle = response.bundle;
  if (
    bundle !== null &&
    !(
      isRecord(bundle) &&
      Number.isSafeInteger(bundle.generation) &&
      (bundle.generation as number) >= 1 &&
      isBundleContentId(bundle.contentId)
    )
  ) {
    throw new ElectronDeidentifyError("INVALID_RESPONSE");
  }
  const spans = response.spans as RendererOpenMedSpan[];
  try {
    assertRendererSpans(spans, textLength);
  } catch {
    throw new ElectronDeidentifyError("INVALID_RESPONSE");
  }
  return {
    schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
    requestId,
    ok: true,
    spans: spans.map(toRendererOpenMedSpan),
    bundle:
      bundle === null
        ? null
        : {
            generation: bundle.generation as number,
            contentId: bundle.contentId as string,
          },
  };
}

function failureResponse(
  requestId: string,
  errorCode: ElectronDeidentifyErrorCode,
): ElectronDeidentifyFailureResponse {
  return {
    schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
    requestId,
    ok: false,
    errorCode,
  };
}

function isValidCancelRequest(
  request: Record<string, unknown>,
): request is Record<string, unknown> & ElectronCancelRequest {
  return (
    request.schemaVersion === OPENMED_ELECTRON_SCHEMA_VERSION &&
    isValidRequestId(request.requestId)
  );
}

function safeRequestId(request: unknown): string {
  return isRecord(request) && isValidRequestId(request.requestId)
    ? request.requestId
    : "invalid-request";
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

function hasHardenedSender(event: unknown): boolean {
  const sender = senderOf(event);
  if (typeof sender?.getLastWebPreferences !== "function") {
    return false;
  }
  try {
    return findWebPreferenceRelaxations(sender.getLastWebPreferences()).length === 0;
  } catch {
    return false;
  }
}

function ownerOf<Event>(
  options: ElectronDeidentifyIpcOptions<Event>,
  event: Event,
): string | null {
  let owner: unknown;
  try {
    owner = options.ownerOf ? options.ownerOf(event) : senderOf(event)?.id;
  } catch {
    return null;
  }
  if (typeof owner === "number" && Number.isSafeInteger(owner)) {
    return `webcontents-${owner}`;
  }
  return typeof owner === "string" &&
    owner.length > 0 &&
    owner.length <= MAX_OWNER_LENGTH
    ? owner
    : null;
}

function senderOf(event: unknown): ElectronIpcSenderLike | undefined {
  if (!isRecord(event)) {
    return undefined;
  }
  const sender = event.sender;
  return isRecord(sender) ? (sender as ElectronIpcSenderLike) : undefined;
}

function nextRendererRequestId(): string {
  const randomUuid = globalThis.crypto?.randomUUID?.();
  if (randomUuid) {
    return `renderer-${randomUuid}`;
  }
  rendererRequestSequence += 1;
  return `renderer-${Date.now()}-${rendererRequestSequence}`;
}

function isExactRecord(
  value: unknown,
  keys: readonly string[],
): value is Record<string, unknown> {
  if (!isRecord(value) || Array.isArray(value)) {
    return false;
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    return false;
  }
  const present = Object.keys(value);
  return (
    present.length === keys.length && present.every((key) => keys.includes(key))
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
