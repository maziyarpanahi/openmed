import { isAbsolute } from "node:path";

import {
  OPENMED_ELECTRON_MAX_TEXT_LENGTH,
  OPENMED_ELECTRON_SCHEMA_VERSION,
  OPENMED_UTILITY_MODEL_PATH_ARGUMENT,
  assertElectronDeidentifyRequest,
  assertRendererOpenMedSpans,
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

export interface UtilityProcessModuleLike {
  fork(
    modulePath: string,
    args?: string[],
    options?: {
      env?: Record<string, string>;
      serviceName?: string;
      stdio?: "ignore";
    },
  ): UtilityProcessLike;
}

export type ElectronDeidentifyServiceEvent =
  | { event: "request_started" }
  | { event: "request_completed"; spanCount: number }
  | { event: "request_failed"; errorCode: string };

export interface ElectronDeidentifyServiceOptions {
  utilityProcess: UtilityProcessModuleLike;
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
const MAX_REQUEST_TIMEOUT_MS = 600_000;
const DEFAULT_MAX_PENDING_REQUESTS = 8;
const MAX_PENDING_REQUESTS = 64;
const REQUEST_ID_PATTERN = /^[A-Za-z0-9_-]{1,80}$/;

export class ElectronDeidentifyService {
  private child: UtilityProcessLike | undefined;
  private readonly pending = new Map<string, PendingRequest>();
  private readonly requestTimeoutMs: number;
  private readonly maxTextLength: number;
  private readonly maxPendingRequests: number;
  private disposed = false;

  constructor(private readonly options: ElectronDeidentifyServiceOptions) {
    if (!isAbsolute(options.workerPath)) {
      throw new TypeError("The OpenMed utility-process path must be absolute.");
    }
    if (!isAbsolute(options.modelPath)) {
      throw new TypeError("The OpenMed model cache path must be absolute.");
    }
    this.requestTimeoutMs = boundedPositiveInteger(
      options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
      MAX_REQUEST_TIMEOUT_MS,
      "request timeout",
    );
    this.maxTextLength = boundedPositiveInteger(
      options.maxTextLength ?? OPENMED_ELECTRON_MAX_TEXT_LENGTH,
      OPENMED_ELECTRON_MAX_TEXT_LENGTH,
      "text length",
    );
    this.maxPendingRequests = boundedPositiveInteger(
      options.maxPendingRequests ?? DEFAULT_MAX_PENDING_REQUESTS,
      MAX_PENDING_REQUESTS,
      "pending-request count",
    );
  }

  async deidentify(
    request: ElectronDeidentifyRequest,
  ): Promise<ElectronDeidentifyResponse> {
    if (this.disposed) {
      throw new Error("The OpenMed de-identification service is disposed.");
    }
    assertElectronDeidentifyRequest(request, this.maxTextLength);
    if (this.pending.has(request.requestId)) {
      throw new TypeError("Duplicate OpenMed Electron request identifier.");
    }
    if (this.pending.size >= this.maxPendingRequests) {
      this.options.logger?.({
        event: "request_failed",
        errorCode: "TOO_MANY_REQUESTS",
      });
      throw new Error("The OpenMed de-identification service is busy.");
    }

    let child: UtilityProcessLike;
    try {
      child = this.getOrCreateChild();
    } catch {
      this.options.logger?.({
        event: "request_failed",
        errorCode: "UTILITY_PROCESS_START_FAILED",
      });
      throw new Error("OpenMed utility process could not be started.");
    }

    this.options.logger?.({ event: "request_started" });
    return new Promise<ElectronDeidentifyResponse>((resolve, reject) => {
      const timeout = setTimeout(() => {
        if (this.child !== child || !this.pending.has(request.requestId)) {
          return;
        }
        this.terminateChild(child);
        this.rejectPending(
          "REQUEST_TIMEOUT",
          "OpenMed de-identification timed out.",
        );
      }, this.requestTimeoutMs);
      this.pending.set(request.requestId, {
        resolve,
        reject,
        timeout,
        textLength: request.text.length,
      });

      const utilityRequest: UtilityDeidentifyRequest = {
        type: "deidentify",
        schemaVersion: request.schemaVersion,
        requestId: request.requestId,
        text: request.text,
      };
      try {
        child.postMessage(utilityRequest);
      } catch {
        this.terminateChild(child);
        this.rejectPending(
          "UTILITY_MESSAGE_FAILED",
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
    if (this.child) {
      this.terminateChild(this.child);
    }
  }

  private getOrCreateChild(): UtilityProcessLike {
    if (this.child) {
      return this.child;
    }
    const child = this.options.utilityProcess.fork(
      this.options.workerPath,
      [OPENMED_UTILITY_MODEL_PATH_ARGUMENT, this.options.modelPath],
      {
        env: {
          HF_HUB_OFFLINE: "1",
          TRANSFORMERS_OFFLINE: "1",
        },
        serviceName: "OpenMed de-identification",
        stdio: "ignore",
      },
    );
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
      this.terminateChild(child);
      this.rejectPending(
        "INVALID_UTILITY_RESPONSE",
        "OpenMed utility process returned an invalid response.",
      );
      return;
    }
    const pending = this.pending.get(message.requestId);
    if (!pending) {
      this.terminateChild(child);
      this.rejectPending(
        "UNEXPECTED_UTILITY_RESPONSE",
        "OpenMed utility process returned an unexpected response.",
      );
      return;
    }
    if (message.ok) {
      try {
        assertRendererOpenMedSpans(message.spans, pending.textLength);
      } catch {
        this.terminateChild(child);
        this.rejectPending(
          "INVALID_UTILITY_RESPONSE",
          "OpenMed utility process returned an invalid response.",
        );
        return;
      }
    }

    clearTimeout(pending.timeout);
    this.pending.delete(message.requestId);
    if (message.ok) {
      this.options.logger?.({
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
    this.options.logger?.({
      event: "request_failed",
      errorCode: message.errorCode,
    });
    pending.reject(new Error("OpenMed de-identification failed."));
  }

  private terminateChild(child: UtilityProcessLike): void {
    if (this.child === child) {
      this.child = undefined;
    }
    try {
      child.kill();
    } catch {
      // The child is already detached from future requests and messages.
    }
  }

  private rejectPending(errorCode: string, message: string): void {
    const hadPending = this.pending.size > 0;
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeout);
      pending.reject(new Error(message));
    }
    this.pending.clear();
    if (hadPending) {
      this.options.logger?.({ event: "request_failed", errorCode });
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
  if (candidate.type !== "deidentify") {
    return false;
  }
  try {
    assertElectronDeidentifyRequest(candidate);
  } catch {
    return false;
  }
  return true;
}

export function inferenceFailure(
  requestId: unknown,
  errorCode: UtilityDeidentifyFailure["errorCode"],
): UtilityDeidentifyResponse {
  return {
    type: "deidentify-result",
    schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
    requestId:
      typeof requestId === "string" && REQUEST_ID_PATTERN.test(requestId)
        ? requestId
        : "invalid-request",
    ok: false,
    errorCode,
  };
}

function boundedPositiveInteger(
  value: number,
  maximum: number,
  name: string,
): number {
  if (!Number.isSafeInteger(value) || value <= 0 || value > maximum) {
    throw new TypeError(`Invalid OpenMed Electron ${name}.`);
  }
  return value;
}
