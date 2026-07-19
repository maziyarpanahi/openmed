import { isAbsolute } from "node:path";

import {
  OPENMED_ELECTRON_SCHEMA_VERSION,
  assertElectronDeidentifyRequest,
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
  logger?: (event: ElectronDeidentifyServiceEvent) => void;
}

interface PendingRequest {
  resolve(response: ElectronDeidentifyResponse): void;
  reject(error: Error): void;
  timeout: ReturnType<typeof setTimeout>;
}

const DEFAULT_REQUEST_TIMEOUT_MS = 120_000;
const DEFAULT_MAX_TEXT_LENGTH = 1_000_000;

export class ElectronDeidentifyService {
  private child: UtilityProcessLike | undefined;
  private readonly pending = new Map<string, PendingRequest>();
  private disposed = false;

  constructor(private readonly options: ElectronDeidentifyServiceOptions) {
    if (!isAbsolute(options.workerPath)) {
      throw new TypeError("The OpenMed utility-process path must be absolute.");
    }
    if (!isAbsolute(options.modelPath)) {
      throw new TypeError("The OpenMed model cache path must be absolute.");
    }
  }

  async deidentify(
    request: ElectronDeidentifyRequest,
  ): Promise<ElectronDeidentifyResponse> {
    assertElectronDeidentifyRequest(
      request,
      this.options.maxTextLength ?? DEFAULT_MAX_TEXT_LENGTH,
    );
    if (this.disposed) {
      throw new Error("The OpenMed de-identification service is disposed.");
    }
    if (this.pending.has(request.requestId)) {
      throw new TypeError("Duplicate OpenMed Electron request identifier.");
    }

    const child = this.getOrCreateChild();
    this.options.logger?.({ event: "request_started" });
    return new Promise<ElectronDeidentifyResponse>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(request.requestId);
        this.options.logger?.({
          event: "request_failed",
          errorCode: "REQUEST_TIMEOUT",
        });
        reject(new Error("OpenMed de-identification timed out."));
      }, this.options.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS);
      this.pending.set(request.requestId, { resolve, reject, timeout });

      const utilityRequest: UtilityDeidentifyRequest = {
        ...request,
        type: "deidentify",
        modelPath: this.options.modelPath,
      };
      child.postMessage(utilityRequest);
    });
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.rejectPending("SERVICE_DISPOSED", "OpenMed service was disposed.");
    this.child?.kill();
    this.child = undefined;
  }

  private getOrCreateChild(): UtilityProcessLike {
    if (this.child) {
      return this.child;
    }
    const child = this.options.utilityProcess.fork(this.options.workerPath, [], {
      env: {
        HF_HUB_OFFLINE: "1",
        TRANSFORMERS_OFFLINE: "1",
      },
      serviceName: "OpenMed de-identification",
      stdio: "ignore",
    });
    child.on("message", (message) => this.handleMessage(message));
    child.on("exit", () => {
      if (this.child === child) {
        this.child = undefined;
      }
      this.rejectPending(
        "UTILITY_PROCESS_EXITED",
        "OpenMed utility process exited.",
      );
    });
    this.child = child;
    return child;
  }

  private handleMessage(message: unknown): void {
    if (!isUtilityDeidentifyResponse(message)) {
      return;
    }
    const pending = this.pending.get(message.requestId);
    if (!pending) {
      return;
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

  private rejectPending(errorCode: string, message: string): void {
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeout);
      pending.reject(new Error(message));
    }
    if (this.pending.size > 0) {
      this.options.logger?.({ event: "request_failed", errorCode });
    }
    this.pending.clear();
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
  return isAbsolute(candidate.modelPath);
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
