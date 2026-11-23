import { normalize } from "node:path";

import type { PortableBundleTarget } from "openmed";

import {
  ElectronDeidentifyError,
  OPENMED_ELECTRON_SCHEMA_VERSION,
  assertElectronDeidentifyRequest,
  assertRendererSpans,
  isBundleContentId,
  toRendererOpenMedSpan,
  type ElectronDeidentifyCallOptions,
  type ElectronDeidentifyErrorCode,
  type ElectronDeidentifyRequest,
  type ElectronDeidentifyResponse,
} from "./ipc";
import {
  isSafeAbsolutePath,
  isUtilityBundleReadyMessage,
  isUtilityBundleRejectedMessage,
  isUtilityDeidentifyResponse,
  utilityFailure,
  type ElectronBundleReason,
  type UtilityBundleReadyMessage,
  type UtilityBundleSource,
  type UtilityBundleVerification,
  type UtilityDeidentifyRequest,
  type UtilityDeidentifyResponse,
  type UtilityErrorCode,
  type UtilityLoadBundleMessage,
} from "./utility-protocol";

export type { ElectronBundleReason } from "./utility-protocol";

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

/**
 * A host-pinned, signed portable model bundle.
 *
 * Every field comes from the host process. Renderers can never supply or
 * change any of them.
 */
export interface ElectronBundleSource {
  /** Absolute path of the bundle directory (app-owned and read-only). */
  root: string;
  /** Caller trust policy document (`openmed.portable_bundle.trust_policy`). */
  trustPolicy: unknown;
  /**
   * Verification target used for SDK, policy and runtime selection. Pass
   * `null` only to accept every runtime the bundle declares.
   */
  target: PortableBundleTarget | null;
  /** Highest accepted `sequence` per bundle id (anti-rollback). */
  sequenceFloors?: Readonly<Record<string, number>>;
  /** Critical extensions the host implements. */
  understoodExtensions?: readonly string[];
  /** Optional exact content identity to pin (`sha256:<hex>`). */
  expectedContentId?: string;
  /** Optional verification limits. */
  limits?: {
    maxManifestBytes?: number;
    maxEnvelopeBytes?: number;
    maxTotalBytes?: number;
    maxFiles?: number;
  };
}

/** Value-free summary of a bundle verification. */
export type ElectronBundleVerification = UtilityBundleVerification;

/** A bundle generation that verified inside the current utility process. */
export interface ElectronBundleHandle {
  readonly generation: number;
  readonly contentId: string;
  readonly runtimeId: string;
  readonly verification: ElectronBundleVerification;
}

/** Raised when a bundle generation fails verification or loading. */
export class ElectronBundleActivationError extends ElectronDeidentifyError {
  readonly reason: ElectronBundleReason;
  readonly generation: number;
  readonly verification: ElectronBundleVerification | null;

  constructor(
    generation: number,
    reason: ElectronBundleReason,
    verification: ElectronBundleVerification | null,
  ) {
    super("BUNDLE_REJECTED");
    this.name = "ElectronBundleActivationError";
    this.reason = reason;
    this.generation = generation;
    this.verification = verification;
  }
}

export type ElectronDeidentifyServiceEvent =
  | { event: "request_started" }
  | { event: "request_completed"; spanCount: number }
  | { event: "request_failed"; errorCode: string }
  | { event: "request_cancelled" }
  | { event: "utility_process_started" }
  | { event: "utility_process_exited" }
  | { event: "bundle_verified"; generation: number; contentId: string }
  | { event: "bundle_rejected"; generation: number; reason: ElectronBundleReason };

/** Value-free counters for health checks and tests. */
export interface ElectronDeidentifyServiceDiagnostics {
  readonly pendingRequests: number;
  readonly cancellingRequests: number;
  readonly utilityProcessStarts: number;
  readonly utilityProcessRunning: boolean;
  readonly generation: number;
  readonly activeBundle: {
    readonly generation: number;
    readonly contentId: string;
    readonly runtimeId: string;
  } | null;
  readonly disposed: boolean;
}

export interface ElectronDeidentifyServiceOptions<
  Session extends ElectronOfflineSessionLike = ElectronOfflineSessionLike,
> {
  utilityProcess: UtilityProcessModuleLike<Session>;
  offlineSession: Session;
  workerPath: string;
  /** Legacy, unverified local model directory. Exactly one of `modelPath`/`bundle`. */
  modelPath?: string;
  /** Verified portable bundle. Exactly one of `modelPath`/`bundle`. */
  bundle?: ElectronBundleSource;
  /** Verification clock; defaults to the system clock. */
  now?: () => Date;
  requestTimeoutMs?: number;
  maxTextLength?: number;
  maxPendingRequests?: number;
  logger?: (event: ElectronDeidentifyServiceEvent) => void;
}

interface Job {
  jobId: string;
  requestId: string;
  ownerKey: string;
  owner: string;
  generation: number | null;
  child: UtilityProcessLike;
  textLength: number;
  timeout: ReturnType<typeof setTimeout>;
  resolve(response: ElectronDeidentifyResponse): void;
  reject(error: Error): void;
}

interface Waiter {
  resolve(handle: ElectronBundleHandle): void;
  reject(error: Error): void;
  timeout: ReturnType<typeof setTimeout>;
}

const DEFAULT_REQUEST_TIMEOUT_MS = 120_000;
const DEFAULT_MAX_TEXT_LENGTH = 1_000_000;
const DEFAULT_MAX_PENDING_REQUESTS = 16;
const MAX_REQUEST_TIMEOUT_MS = 300_000;
const MAX_PENDING_REQUESTS = 64;
const MAX_OWNER_LENGTH = 128;
const MAIN_OWNER = "main";

/**
 * Owns the OpenMed utility process for an Electron app.
 *
 * In bundle mode the utility process verifies a signed portable bundle with
 * the `openmed` verifier before it serves any request, and again after every
 * restart. Each request is pinned to the bundle generation active when it was
 * submitted; its result carries that generation and content identity, and a
 * result from any other generation is refused. Requests are owned by their
 * renderer and settle exactly once: with spans, or with an
 * {@link ElectronDeidentifyError} carrying a controlled code.
 */
export class ElectronDeidentifyService<
  Session extends ElectronOfflineSessionLike = ElectronOfflineSessionLike,
> {
  private child: UtilityProcessLike | undefined;
  private readonly jobs = new Map<string, Job>();
  private readonly owners = new Map<string, string>();
  private readonly cancelling = new Map<
    string,
    { child: UtilityProcessLike; timeout: ReturnType<typeof setTimeout> }
  >();
  private readonly ready = new Map<number, ElectronBundleHandle>();
  private readonly waiters = new Map<number, Waiter[]>();
  private readonly utilityProcess: UtilityProcessModuleLike<Session>;
  private readonly offlineSession: Session;
  private readonly workerPath: string;
  private readonly modelPath: string | undefined;
  private source: UtilityBundleSource | undefined;
  private generation = 0;
  private jobSequence = 0;
  private utilityProcessStarts = 0;
  private readonly now: () => Date;
  private readonly requestTimeoutMs: number;
  private readonly maxTextLength: number;
  private readonly maxPendingRequests: number;
  private readonly logger:
    | ((event: ElectronDeidentifyServiceEvent) => void)
    | undefined;
  private lastRejection: ElectronBundleActivationError | undefined;
  private disposed = false;

  constructor(options: ElectronDeidentifyServiceOptions<Session>) {
    this.workerPath = validateAbsolutePath(
      options.workerPath,
      "utility-process",
    );
    if ((options.modelPath === undefined) === (options.bundle === undefined)) {
      throw new TypeError(
        "Configure exactly one of an OpenMed model bundle or a model path.",
      );
    }
    this.modelPath =
      options.modelPath === undefined
        ? undefined
        : validateAbsolutePath(options.modelPath, "model cache");
    if (options.bundle !== undefined) {
      this.source = normalizeBundleSource(options.bundle);
      this.generation = 1;
    }
    if (options.now !== undefined && typeof options.now !== "function") {
      throw new TypeError("The OpenMed verification clock must be a function.");
    }
    this.now = options.now ?? (() => new Date());
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

  /**
   * De-identify one request in the utility process.
   *
   * Args:
   *   request: A validated renderer request.
   *   options: The owning renderer key (defaults to the main process).
   *
   * Returns:
   *   Renderer-safe spans plus the bundle generation that produced them.
   *
   * Raises:
   *   ElectronDeidentifyError: With a controlled code on every failure.
   */
  async deidentify(
    request: ElectronDeidentifyRequest,
    options: ElectronDeidentifyCallOptions = {},
  ): Promise<ElectronDeidentifyResponse> {
    try {
      assertElectronDeidentifyRequest(request, this.maxTextLength);
    } catch {
      throw new ElectronDeidentifyError("INVALID_REQUEST");
    }
    const owner = normalizeOwner(options.owner);
    if (this.disposed) {
      throw new ElectronDeidentifyError("SERVICE_DISPOSED");
    }
    const ownerKey = `${owner}\u0000${request.requestId}`;
    if (this.owners.has(ownerKey)) {
      throw new ElectronDeidentifyError("DUPLICATE_REQUEST");
    }
    if (this.jobs.size >= this.maxPendingRequests) {
      throw new ElectronDeidentifyError("QUEUE_FULL");
    }

    let child: UtilityProcessLike;
    try {
      child = this.getOrCreateChild();
    } catch {
      this.emit({
        event: "request_failed",
        errorCode: "UTILITY_PROCESS_START_FAILED",
      });
      throw new ElectronDeidentifyError("UTILITY_PROCESS_START_FAILED");
    }
    this.emit({ event: "request_started" });
    this.jobSequence += 1;
    const jobId = `job-${this.jobSequence}`;
    const generation = this.source === undefined ? null : this.generation;
    return new Promise<ElectronDeidentifyResponse>((resolve, reject) => {
      const timeout = setTimeout(() => {
        if (this.jobs.has(jobId)) {
          this.failChild(child, "REQUEST_TIMEOUT");
        }
      }, this.requestTimeoutMs);
      this.jobs.set(jobId, {
        jobId,
        requestId: request.requestId,
        ownerKey,
        owner,
        generation,
        child,
        textLength: request.text.length,
        timeout,
        resolve,
        reject,
      });
      this.owners.set(ownerKey, jobId);

      const message: UtilityDeidentifyRequest = {
        type: "deidentify",
        schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
        requestId: jobId,
        text: request.text,
        ...(generation === null
          ? { modelPath: this.modelPath as string }
          : { generation }),
      };
      this.post(child, message);
    });
  }

  /**
   * Cancel one pending request.
   *
   * The request rejects with `CANCELLED` at once; the utility process skips
   * it if it is still queued and discards its output if it is running.
   *
   * Args:
   *   requestId: The renderer request id.
   *   owner: The owning renderer key (defaults to the main process).
   *
   * Returns:
   *   True when a pending request was cancelled.
   */
  cancel(requestId: string, owner?: string): boolean {
    let ownerKey: string;
    try {
      ownerKey = `${normalizeOwner(owner)}\u0000${requestId}`;
    } catch {
      return false;
    }
    const jobId = this.owners.get(ownerKey);
    const job = jobId === undefined ? undefined : this.jobs.get(jobId);
    if (job === undefined) {
      return false;
    }
    this.cancelJob(job);
    return true;
  }

  /**
   * Cancel every pending request of one renderer, for example on teardown.
   *
   * Returns:
   *   The number of requests cancelled.
   */
  cancelOwner(owner: string): number {
    let normalized: string;
    try {
      normalized = normalizeOwner(owner);
    } catch {
      return 0;
    }
    let count = 0;
    for (const job of [...this.jobs.values()]) {
      if (job.owner === normalized) {
        this.cancelJob(job);
        count += 1;
      }
    }
    return count;
  }

  /**
   * Verify and activate a new bundle generation.
   *
   * Requests submitted before the call keep running on the generation they
   * were submitted for; requests submitted after it run only on the new
   * generation, and fail with `BUNDLE_UNAVAILABLE` if it is rejected.
   *
   * Args:
   *   source: The new host-pinned bundle.
   *
   * Returns:
   *   The verified handle of the new generation.
   *
   * Raises:
   *   ElectronBundleActivationError: When verification or loading fails.
   *   ElectronDeidentifyError: When the utility process fails or exits.
   */
  async activateBundle(source: ElectronBundleSource): Promise<ElectronBundleHandle> {
    if (this.disposed) {
      throw new ElectronDeidentifyError("SERVICE_DISPOSED");
    }
    if (this.source === undefined) {
      throw new TypeError("This OpenMed service was created with a model path.");
    }
    const normalized = normalizeBundleSource(source);
    this.source = normalized;
    this.generation += 1;
    const generation = this.generation;
    const waiting = this.waitFor(generation);
    const existing = this.child;
    if (existing === undefined) {
      this.startForBundle();
    } else {
      this.post(existing, this.loadMessage());
    }
    return waiting;
  }

  /**
   * Resolve the current generation's handle, starting the utility process
   * and verifying the bundle if needed.
   */
  async whenReady(): Promise<ElectronBundleHandle> {
    if (this.disposed) {
      throw new ElectronDeidentifyError("SERVICE_DISPOSED");
    }
    if (this.source === undefined) {
      throw new TypeError("This OpenMed service was created with a model path.");
    }
    const handle = this.ready.get(this.generation);
    if (handle !== undefined && this.child !== undefined) {
      return handle;
    }
    if (
      this.child !== undefined &&
      this.lastRejection !== undefined &&
      this.lastRejection.generation === this.generation
    ) {
      throw this.lastRejection;
    }
    const waiting = this.waitFor(this.generation);
    if (this.child === undefined) {
      this.startForBundle();
    }
    return waiting;
  }

  /** Return value-free counters. */
  diagnostics(): ElectronDeidentifyServiceDiagnostics {
    const handle = this.ready.get(this.generation);
    return {
      pendingRequests: this.jobs.size,
      cancellingRequests: this.cancelling.size,
      utilityProcessStarts: this.utilityProcessStarts,
      utilityProcessRunning: this.child !== undefined,
      generation: this.generation,
      activeBundle:
        handle === undefined || this.child === undefined
          ? null
          : {
              generation: handle.generation,
              contentId: handle.contentId,
              runtimeId: handle.runtimeId,
            },
      disposed: this.disposed,
    };
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    const child = this.child;
    this.child = undefined;
    this.retire("SERVICE_DISPOSED");
    try {
      child?.kill();
    } catch {
      // The service is already disposed; no diagnostic may contain process data.
    }
  }

  private startForBundle(): void {
    try {
      this.getOrCreateChild();
    } catch {
      this.rejectWaiters("UTILITY_PROCESS_START_FAILED");
    }
  }

  private waitFor(generation: number): Promise<ElectronBundleHandle> {
    return new Promise<ElectronBundleHandle>((resolve, reject) => {
      const waiter: Waiter = {
        resolve,
        reject,
        timeout: setTimeout(() => {
          const list = this.waiters.get(generation);
          const index = list?.indexOf(waiter) ?? -1;
          if (list && index >= 0) {
            list.splice(index, 1);
            reject(new ElectronDeidentifyError("BUNDLE_ACTIVATION_TIMEOUT"));
          }
        }, this.requestTimeoutMs),
      };
      const list = this.waiters.get(generation) ?? [];
      list.push(waiter);
      this.waiters.set(generation, list);
    });
  }

  private settleWaiters(
    generation: number,
    outcome: { handle: ElectronBundleHandle } | { error: Error },
  ): void {
    const list = this.waiters.get(generation) ?? [];
    this.waiters.delete(generation);
    for (const waiter of list) {
      clearTimeout(waiter.timeout);
      if ("handle" in outcome) {
        waiter.resolve(outcome.handle);
      } else {
        waiter.reject(outcome.error);
      }
    }
  }

  private rejectWaiters(code: ElectronDeidentifyErrorCode): void {
    for (const generation of [...this.waiters.keys()]) {
      this.settleWaiters(generation, { error: new ElectronDeidentifyError(code) });
    }
  }

  private loadMessage(): UtilityLoadBundleMessage {
    return {
      type: "load-bundle",
      generation: this.generation,
      now: formatInstant(this.now()),
      source: this.source as UtilityBundleSource,
    };
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
    this.utilityProcessStarts += 1;
    child.on("message", (message) => this.handleMessage(child, message));
    child.on("exit", () => {
      if (this.child !== child) {
        return;
      }
      this.child = undefined;
      this.emit({ event: "utility_process_exited" });
      this.retire("UTILITY_PROCESS_EXITED");
    });
    if (this.source !== undefined) {
      try {
        child.postMessage(this.loadMessage());
      } catch {
        try {
          child.kill();
        } catch {
          // The fork is discarded; startup failure is reported by the caller.
        }
        throw new ElectronDeidentifyError("UTILITY_PROCESS_START_FAILED");
      }
    }
    this.child = child;
    this.emit({ event: "utility_process_started" });
    return child;
  }

  private post(child: UtilityProcessLike, message: unknown): void {
    try {
      child.postMessage(message);
    } catch {
      this.failChild(child, "UTILITY_PROCESS_SEND_FAILED");
    }
  }

  private handleMessage(child: UtilityProcessLike, message: unknown): void {
    if (this.child !== child) {
      return;
    }
    if (isUtilityBundleReadyMessage(message)) {
      this.handleBundleReady(child, message);
      return;
    }
    if (isUtilityBundleRejectedMessage(message)) {
      if (this.source === undefined || message.generation > this.generation) {
        this.failChild(child, "INVALID_UTILITY_RESPONSE");
        return;
      }
      this.ready.delete(message.generation);
      this.emit({
        event: "bundle_rejected",
        generation: message.generation,
        reason: message.reason,
      });
      const rejection = new ElectronBundleActivationError(
        message.generation,
        message.reason,
        message.verification,
      );
      this.lastRejection = rejection;
      this.settleWaiters(message.generation, { error: rejection });
      return;
    }
    if (!isUtilityDeidentifyResponse(message)) {
      this.failChild(child, "INVALID_UTILITY_RESPONSE");
      return;
    }
    const cancelled = this.cancelling.get(message.requestId);
    if (cancelled !== undefined && cancelled.child === child) {
      clearTimeout(cancelled.timeout);
      this.cancelling.delete(message.requestId);
      return;
    }
    const job = this.jobs.get(message.requestId);
    if (job === undefined || message.generation !== job.generation) {
      this.failChild(child, "INVALID_UTILITY_RESPONSE");
      return;
    }
    if (!message.ok) {
      if (message.errorCode === "CANCELLED") {
        this.failChild(child, "INVALID_UTILITY_RESPONSE");
        return;
      }
      this.settleJob(job);
      this.emit({ event: "request_failed", errorCode: message.errorCode });
      job.reject(new ElectronDeidentifyError(serviceCodeFor(message.errorCode)));
      return;
    }
    const handle = job.generation === null ? undefined : this.ready.get(job.generation);
    try {
      if (job.generation !== null && handle === undefined) {
        throw new TypeError("result before verification");
      }
      assertRendererSpans(message.spans, job.textLength);
    } catch {
      this.failChild(child, "INVALID_UTILITY_RESPONSE");
      return;
    }
    this.settleJob(job);
    this.emit({ event: "request_completed", spanCount: message.spans.length });
    job.resolve({
      schemaVersion: OPENMED_ELECTRON_SCHEMA_VERSION,
      requestId: job.requestId,
      ok: true,
      spans: message.spans.map(toRendererOpenMedSpan),
      bundle:
        handle === undefined
          ? null
          : { generation: handle.generation, contentId: handle.contentId },
    });
    this.pruneReady();
  }

  private handleBundleReady(
    child: UtilityProcessLike,
    message: UtilityBundleReadyMessage,
  ): void {
    const source = this.source;
    if (
      source === undefined ||
      message.generation > this.generation ||
      (message.generation === this.generation &&
        source.expectedContentId !== null &&
        source.expectedContentId !== message.contentId)
    ) {
      this.failChild(child, "INVALID_UTILITY_RESPONSE");
      return;
    }
    const handle: ElectronBundleHandle = Object.freeze({
      generation: message.generation,
      contentId: message.contentId,
      runtimeId: message.runtimeId,
      verification: Object.freeze({
        ...message.verification,
        eligibleRuntimes: [...message.verification.eligibleRuntimes],
      }),
    });
    this.ready.set(message.generation, handle);
    this.emit({
      event: "bundle_verified",
      generation: message.generation,
      contentId: message.contentId,
    });
    this.settleWaiters(message.generation, { handle });
    this.pruneReady();
  }

  private pruneReady(): void {
    const referenced = new Set<number | null>(
      [...this.jobs.values()].map((job) => job.generation),
    );
    for (const generation of [...this.ready.keys()]) {
      if (generation !== this.generation && !referenced.has(generation)) {
        this.ready.delete(generation);
      }
    }
  }

  private cancelJob(job: Job): void {
    this.settleJob(job);
    const timeout = setTimeout(() => {
      if (this.cancelling.has(job.jobId)) {
        this.failChild(job.child, "REQUEST_TIMEOUT");
      }
    }, this.requestTimeoutMs);
    this.cancelling.set(job.jobId, { child: job.child, timeout });
    this.emit({ event: "request_cancelled" });
    job.reject(new ElectronDeidentifyError("CANCELLED"));
    this.post(job.child, { type: "cancel", requestId: job.jobId });
  }

  private settleJob(job: Job): void {
    clearTimeout(job.timeout);
    this.jobs.delete(job.jobId);
    this.owners.delete(job.ownerKey);
  }

  /** Reject all work of the retired child exactly once. */
  private retire(code: ElectronDeidentifyErrorCode): void {
    const jobs = [...this.jobs.values()];
    for (const job of jobs) {
      this.settleJob(job);
      job.reject(new ElectronDeidentifyError(code));
    }
    if (jobs.length > 0) {
      this.emit({ event: "request_failed", errorCode: code });
    }
    for (const entry of this.cancelling.values()) {
      clearTimeout(entry.timeout);
    }
    this.cancelling.clear();
    this.ready.clear();
    this.lastRejection = undefined;
    this.rejectWaiters(code);
  }

  private failChild(child: UtilityProcessLike, code: ElectronDeidentifyErrorCode): void {
    if (this.child !== child) {
      return;
    }
    this.child = undefined;
    this.retire(code);
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

export { isUtilityDeidentifyMessage } from "./utility-protocol";

/** Build a value-free utility failure (kept for custom utility entrypoints). */
export function inferenceFailure(
  requestId: string,
  errorCode: UtilityErrorCode,
): UtilityDeidentifyResponse {
  return utilityFailure(requestId, errorCode);
}

/** Format a verification instant as `YYYY-MM-DDTHH:MM:SSZ`. */
export function formatVerificationInstant(value: Date): string {
  return formatInstant(value);
}

function formatInstant(value: Date): string {
  if (!(value instanceof Date) || Number.isNaN(value.getTime())) {
    throw new TypeError("Invalid OpenMed verification clock value.");
  }
  return value.toISOString().replace(/\.\d{3}Z$/, "Z");
}

function serviceCodeFor(code: UtilityErrorCode): ElectronDeidentifyErrorCode {
  switch (code) {
    case "INVALID_REQUEST":
      return "INVALID_REQUEST";
    case "BUNDLE_UNAVAILABLE":
      return "BUNDLE_UNAVAILABLE";
    default:
      return "INFERENCE_FAILED";
  }
}

function normalizeOwner(owner: unknown): string {
  if (owner === undefined) {
    return MAIN_OWNER;
  }
  if (typeof owner !== "string" || owner.length === 0 || owner.length > MAX_OWNER_LENGTH) {
    throw new ElectronDeidentifyError("INVALID_REQUEST");
  }
  return owner;
}

function normalizeBundleSource(source: ElectronBundleSource): UtilityBundleSource {
  if (typeof source !== "object" || source === null) {
    throw new TypeError("Invalid OpenMed model bundle source.");
  }
  const root = validateAbsolutePath(source.root, "model bundle");
  const target = source.target;
  if (target === undefined) {
    throw new TypeError("An OpenMed bundle verification target is required.");
  }
  const floors = source.sequenceFloors ?? {};
  const extensions = source.understoodExtensions ?? [];
  const limits = source.limits ?? {};
  if (
    typeof floors !== "object" ||
    floors === null ||
    !Object.values(floors).every((floor) => Number.isSafeInteger(floor) && floor >= 0) ||
    !Array.isArray(extensions) ||
    !extensions.every((item) => typeof item === "string") ||
    (source.expectedContentId !== undefined && !isBundleContentId(source.expectedContentId)) ||
    typeof limits !== "object" ||
    limits === null
  ) {
    throw new TypeError("Invalid OpenMed model bundle source.");
  }
  const cleanLimits: UtilityBundleSource["limits"] = {};
  for (const key of [
    "maxManifestBytes",
    "maxEnvelopeBytes",
    "maxTotalBytes",
    "maxFiles",
  ] as const) {
    const value = limits[key];
    if (value !== undefined) {
      if (!Number.isSafeInteger(value) || value < 0) {
        throw new TypeError("Invalid OpenMed model bundle source.");
      }
      cleanLimits[key] = value;
    }
  }
  return {
    root,
    trustPolicy: structuredClone(source.trustPolicy),
    target:
      target === null
        ? null
        : {
            platform: target.platform,
            sdkName: target.sdkName,
            sdkVersion: target.sdkVersion,
            runtimes: [...target.runtimes],
            osVersion: target.osVersion ?? null,
            policyVersion: target.policyVersion ?? null,
            memoryBytes: target.memoryBytes ?? null,
            storageBytes: target.storageBytes ?? null,
          },
    sequenceFloors: { ...floors },
    understoodExtensions: [...extensions],
    expectedContentId: source.expectedContentId ?? null,
    limits: cleanLimits,
  };
}

function validateAbsolutePath(value: unknown, label: string): string {
  if (!isSafeAbsolutePath(value)) {
    throw new TypeError(`The OpenMed ${label} path must be an absolute path.`);
  }
  return normalize(value);
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
