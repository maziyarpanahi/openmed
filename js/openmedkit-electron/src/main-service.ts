import { isAbsolute, normalize } from "node:path";

import {
  OPENMED_ELECTRON_SCHEMA_VERSION,
  assertElectronDeidentifyRequest,
  assertRendererSpans,
  isUtilityDeidentifyResponse,
  toRendererOpenMedSpan,
  type ElectronDeidentifyRequest,
  type ElectronDeidentifyResponse,
  type UtilityDeidentifyFailure,
  type UtilityDeidentifyRequest,
  type UtilityDeidentifyResponse,
} from "./ipc";

export interface UtilityProcessLike {
  postMessage(message: unknown): void;
  on(event: "message", listener: (message: unknown) => void): this;
  on(event: "exit", listener: (code: number) => void): this;
  kill(): boolean;
}

export interface UtilityProcessModuleLike<Session = ElectronOfflineSessionLike> {
  fork(
    modulePath: string,
    args?: string[],
    options?: {
      env?: Record<string, string>;
      session?: Session;
      serviceName?: string;
      stdio?: "ignore";
    },
  ): UtilityProcessLike;
}

export interface ElectronOfflineSessionLike {
  webRequest: {
    onBeforeRequest(
      listener:
        | ((
            details: unknown,
            callback: (response: { cancel: boolean }) => void,
          ) => void)
        | null,
    ): void;
  };
}

export type ElectronDeidentifyServiceEvent =
  | { event: "request_started" }
  | { event: "request_completed"; spanCount: number }
  | { event: "request_failed"; errorCode: string };

export interface ElectronDeidentifyServiceOptions<
  Session extends ElectronOfflineSessionLike = ElectronOfflineSessionLike,
> {
  utilityProcess: UtilityProcessModuleLike<Session>;
  offlineSession: Session;
  workerPath: string;
  modelPath: string;
  requestTimeoutMs?: number;
  maxTextLength?: number;
  maxPendingRequests?: number;
  logger?: (event: ElectronDeidentifyServiceEvent) => void;
}

interface PendingRequest {
  resolve(response: ElectronDeidentifyResponse): void;
  reject(error: Error): void;
  timeout: ReturnType<typeof setTimeout>;
  textLength: number;
}

const DEFAULT_REQUEST_TIMEOUT_MS = 120_000;
const DEFAULT_MAX_TEXT_LENGTH = 1_000_000;
const DEFAULT_MAX_PENDING_REQUESTS = 16;
const MAX_REQUEST_TIMEOUT_MS = 300_000;
const MAX_PENDING_REQUESTS = 64;
const MAX_PATH_LENGTH = 4_096;

export class ElectronDeidentifyService<
  Session extends ElectronOfflineSessionLike = ElectronOfflineSessionLike,
> {
  private child: UtilityProcessLike | undefined;
  private readonly pending = new Map<string, PendingRequest>();
  private readonly utilityProcess: UtilityProcessModuleLike<Session>;
  private readonly offlineSession: Session;
  private readonly workerPath: string;
  private readonly modelPath: string;
  private readonly requestTimeoutMs: number;
  private readonly maxTextLength: number;
  private readonly maxPendingRequests: number;
  private readonly logger:
    | ((event: ElectronDeidentifyServiceEvent) => void)
    | undefined;
  private disposed = false;

  constructor(options: ElectronDeidentifyServiceOptions<Session>) {
    this.workerPath = validateAbsolutePath(
      options.workerPath,
      "utility-process",
    );
    this.modelPath = validateAbsolutePath(options.modelPath, "model cache");
    this.requestTimeoutMs = validateBoundedInteger(
      options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
      "request timeout",
      MAX_REQUEST_TIMEOUT_MS,
    );
    this.maxTextLength = validateBoundedInteger(
      options.maxTextLength ?? DEFAULT_MAX_TEXT_LENGTH,
      "maximum text length",
      DEFAULT_MAX_TEXT_LENGTH,
    );
    this.maxPendingRequests = validateBoundedInteger(
      options.maxPendingRequests ?? DEFAULT_MAX_PENDING_REQUESTS,
      "maximum pending request count",
      MAX_PENDING_REQUESTS,
    );
    if (
      typeof options.offlineSession !== "object" ||
      options.offlineSession === null ||
      typeof options.offlineSession.webRequest?.onBeforeRequest !== "function"
    ) {
      throw new TypeError("A dedicated offline Electron session is required.");
    }
    if (typeof options.utilityProcess?.fork !== "function") {
      throw new TypeError("The Electron utility-process module is required.");
    }
    if (options.logger !== undefined && typeof options.logger !== "function") {
      throw new TypeError("The OpenMed Electron logger must be a function.");
    }

    this.utilityProcess = options.utilityProcess;
    this.offlineSession = options.offlineSession;
    this.logger = options.logger;
    this.offlineSession.webRequest.onBeforeRequest((_details, callback) => {
      callback({ cancel: true });
    });
  }

  async deidentify(
    request: ElectronDeidentifyRequest,
  ): Promise<ElectronDeidentifyResponse> {
    assertElectronDeidentifyRequest(
      request,
      this.maxTextLength,
    );
    if (this.disposed) {
      throw new Error("The OpenMed de-identification service is disposed.");
    }
    if (this.pending.has(request.requestId)) {
      throw new TypeError("Duplicate OpenMed Electron request identifier.");
    }
    if (this.pending.size >= this.maxPendingRequests) {
      throw new Error("The OpenMed de-identification queue is full.");
    }

    let child: UtilityProcessLike;
    try {
      child = this.getOrCreateChild();
    } catch {
      this.emit({
        event: "request_failed",
        errorCode: "UTILITY_PROCESS_START_FAILED",
      });
      throw new Error("OpenMed utility process could not be started.");
    }
    this.emit({ event: "request_started" });
    return new Promise<ElectronDeidentifyResponse>((resolve, reject) => {
      const timeout = setTimeout(() => {
        if (this.pending.has(request.requestId)) {
          this.failChild(
            child,
            "REQUEST_TIMEOUT",
            "OpenMed de-identification timed out.",
          );
        }
      }, this.requestTimeoutMs);
      this.pending.set(request.requestId, {
        resolve,
        reject,
        timeout,
        textLength: request.text.length,
      });

      const utilityRequest: UtilityDeidentifyRequest = {
        ...request,
        type: "deidentify",
        modelPath: this.modelPath,
      };
      try {
        child.postMessage(utilityRequest);
      } catch {
        this.failChild(
          child,
          "UTILITY_PROCESS_SEND_FAILED",
          "OpenMed utility process communication failed.",
        );
      }
    });
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.rejectPending("SERVICE_DISPOSED", "OpenMed service was disposed.");
    try {
      this.child?.kill();
    } catch {
      // The service is already disposed; no diagnostic may contain process data.
    }
    this.child = undefined;
  }

  private getOrCreateChild(): UtilityProcessLike {
    if (this.child) {
      return this.child;
    }
    const child = this.utilityProcess.fork(this.workerPath, [], {
      env: {
        HF_HUB_OFFLINE: "1",
        TRANSFORMERS_OFFLINE: "1",
      },
      session: this.offlineSession,
      serviceName: "OpenMed de-identification",
      stdio: "ignore",
    });
    child.on("message", (message) => this.handleMessage(child, message));
    child.on("exit", () => {
      if (this.child !== child) {
        return;
      }
      this.child = undefined;
      this.rejectPending(
        "UTILITY_PROCESS_EXITED",
        "OpenMed utility process exited.",
      );
    });
    this.child = child;
    return child;
  }

  private handleMessage(child: UtilityProcessLike, message: unknown): void {
    if (this.child !== child) {
      return;
    }
    if (!isUtilityDeidentifyResponse(message)) {
      this.failChild(
        child,
        "INVALID_UTILITY_RESPONSE",
        "OpenMed utility process returned an invalid response.",
      );
      return;
    }
    const pending = this.pending.get(message.requestId);
    if (!pending) {
      this.failChild(
        child,
        "INVALID_UTILITY_RESPONSE",
        "OpenMed utility process returned an invalid response.",
      );
      return;
    }
    if (message.ok) {
      try {
        assertRendererSpans(message.spans, pending.textLength);
      } catch {
        this.failChild(
          child,
          "INVALID_UTILITY_RESPONSE",
          "OpenMed utility process returned an invalid response.",
        );
        return;
      }
      clearTimeout(pending.timeout);
      this.pending.delete(message.requestId);
      this.emit({
        event: "request_completed",
        spanCount: message.spans.length,
      });
      pending.resolve({
        schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
        requestId: message.requestId,
        spans: message.spans.map(toRendererOpenMedSpan),
      });
      return;
    }
    clearTimeout(pending.timeout);
    this.pending.delete(message.requestId);
    this.emit({
      event: "request_failed",
      errorCode: message.errorCode,
    });
    pending.reject(new Error("OpenMed de-identification failed."));
  }

  private rejectPending(errorCode: string, message: string): void {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeout);
      pending.reject(new Error(message));
    }
    if (this.pending.size > 0) {
      this.emit({ event: "request_failed", errorCode });
    }
    this.pending.clear();
  }

  private failChild(
    child: UtilityProcessLike,
    errorCode: string,
    message: string,
  ): void {
    if (this.child !== child) {
      return;
    }
    this.child = undefined;
    this.rejectPending(errorCode, message);
    try {
      child.kill();
    } catch {
      // Failure is already surfaced through the bounded service error above.
    }
  }

  private emit(event: ElectronDeidentifyServiceEvent): void {
    try {
      this.logger?.(event);
    } catch {
      // Logging is observational and must never alter inference behavior.
    }
  }
}

export function isUtilityDeidentifyMessage(
  message: unknown,
): message is UtilityDeidentifyRequest {
  if (typeof message !== "object" || message === null) {
    return false;
  }
  const candidate = message as Record<string, unknown>;
  if (candidate.type !== "deidentify" || typeof candidate.modelPath !== "string") {
    return false;
  }
  try {
    assertElectronDeidentifyRequest(candidate);
  } catch {
    return false;
  }
  return isSafeAbsolutePath(candidate.modelPath);
}

function validateAbsolutePath(value: unknown, label: string): string {
  if (!isSafeAbsolutePath(value)) {
    throw new TypeError(`The OpenMed ${label} path must be an absolute path.`);
  }
  return normalize(value);
}

function isSafeAbsolutePath(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value.length <= MAX_PATH_LENGTH &&
    !value.includes("\0") &&
    isAbsolute(value)
  );
}

function validateBoundedInteger(
  value: unknown,
  label: string,
  maximum: number,
): number {
  if (
    !Number.isSafeInteger(value) ||
    (value as number) < 1 ||
    (value as number) > maximum
  ) {
    throw new TypeError(`Invalid OpenMed Electron ${label}.`);
  }
  return value as number;
}

export function inferenceFailure(
  requestId: string,
  errorCode: UtilityDeidentifyFailure["errorCode"],
): UtilityDeidentifyResponse {
  return {
    type: "deidentify-result",
    requestId,
    ok: false,
    errorCode,
  };
}
