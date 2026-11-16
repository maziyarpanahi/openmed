import type {
  TokenClassificationCallOptions,
  TokenClassificationEntity,
  TokenClassificationPipeline,
} from "../types";
import {
  DEFAULT_INFERENCE_WORKER_MAX_IN_FLIGHT,
  DEFAULT_INFERENCE_WORKER_MAX_INPUT_CHARS,
  DEFAULT_INFERENCE_WORKER_MAX_QUEUE_SIZE,
  INFERENCE_WORKER_PROTOCOL_VERSION,
  InferenceWorkerError,
  decodeInferenceTokens,
  encodeTextUnits,
  isInferenceWorkerErrorCode,
  isOfflineModelRef,
  isPlainData,
  isPositiveId,
  isRecord,
  resolveBound,
  type HostToWorkerMessage,
  type InferenceWorkerErrorCode,
  type InferenceWorkerLoadReason,
  type InferenceWorkerLoadSpec,
  type InferenceWorkerPlan,
  type NormalizedInferenceWorkerLoadSpec,
} from "./protocol";

/** Listener shape shared by `Worker`, `MessagePort`, and test doubles. */
export type InferenceWorkerEventListener = (event: {
  data?: unknown;
  preventDefault?: () => void;
}) => void;

/** Minimal host-side view of a dedicated worker (a DOM `Worker` fits). */
export interface InferenceWorkerEndpoint {
  postMessage(message: unknown, transfer: Transferable[]): void;
  addEventListener(
    type: "message" | "messageerror" | "error",
    listener: InferenceWorkerEventListener,
  ): void;
  removeEventListener(
    type: "message" | "messageerror" | "error",
    listener: InferenceWorkerEventListener,
  ): void;
  terminate?(): void;
}

/** Lifecycle state of an {@link InferenceWorkerClient}. */
export type InferenceWorkerState =
  | "idle"
  | "loading"
  | "ready"
  | "invalidated"
  | "failed"
  | "disposed";

/** Options for {@link createInferenceWorkerClient}. */
export interface InferenceWorkerClientOptions {
  /** Creates a fresh worker; called lazily and again on restart. */
  createWorker: () => InferenceWorkerEndpoint;
  /** Requests that may wait on the host before `queue_full`. Defaults to 16. */
  maxQueueSize?: number;
  /** Requests outstanding in the worker, cancelled ones included. Defaults to 1. */
  maxInFlight?: number;
  /** Per-request input bound in UTF-16 code units. Defaults to 1,000,000. */
  maxInputChars?: number;
  /**
   * Extra plan check run after the built-in verification.
   *
   * `previous` is the last verified plan during `recover()`, otherwise null.
   * Returning false or throwing rejects the plan with `plan_incompatible`.
   */
  acceptPlan?: (
    plan: InferenceWorkerPlan,
    previous: InferenceWorkerPlan | null,
  ) => boolean;
}

/** Options for {@link InferenceWorkerClient.run}. */
export interface InferenceWorkerRunOptions {
  /** Cancels the request; the worker is told and late results are dropped. */
  signal?: AbortSignal;
  /** Plain-data pipeline options forwarded to the worker pipeline. */
  callOptions?: TokenClassificationCallOptions;
  /** Model generation that must still own the worker. Defaults to the current one. */
  generation?: number;
}

/** Options for {@link InferenceWorkerClient.load}. */
export interface InferenceWorkerLoadOptions {
  signal?: AbortSignal;
  /** Replace the worker first, for example when a runtime ignores aborts. */
  restart?: boolean;
}

/** Options for {@link InferenceWorkerClient.recover}. */
export interface InferenceWorkerRecoverOptions {
  signal?: AbortSignal;
  /** Replace the worker even if it is still alive. */
  restart?: boolean;
}

/** Options for {@link InferenceWorkerClient.pipeline}. */
export interface InferenceWorkerPipelineOptions {
  /** Signal applied to every call made through the returned pipeline. */
  signal?: AbortSignal;
}

/** Value-free counters describing the client. */
export interface InferenceWorkerStats {
  state: InferenceWorkerState;
  generation: number;
  epoch: number;
  queued: number;
  inFlight: number;
  draining: number;
  maxQueueSize: number;
  maxInFlight: number;
  workerStarts: number;
  completed: number;
  cancelled: number;
  rejectedQueueFull: number;
  suppressedLateResults: number;
}

interface PendingRequest {
  id: number;
  generation: number;
  epoch: number;
  text: string | null;
  callOptions: Record<string, unknown>;
  phase: "queued" | "dispatched" | "draining";
  settled: boolean;
  resolve: (tokens: TokenClassificationEntity[]) => void;
  reject: (error: InferenceWorkerError) => void;
  cleanup: () => void;
}

interface PendingLoad {
  id: number;
  epoch: number;
  generation: number;
  spec: NormalizedInferenceWorkerLoadSpec;
  previous: InferenceWorkerPlan | null;
  resolve: (plan: InferenceWorkerPlan) => void;
  reject: (error: InferenceWorkerError) => void;
  cleanup: () => void;
}

type ListenerSet = Record<
  "message" | "messageerror" | "error",
  InferenceWorkerEventListener
>;

/**
 * Host-side facade that routes token classification through a dedicated worker.
 *
 * Requests are admitted into a bounded queue, carry request ids and the model
 * generation that owns them, and can be cancelled with an `AbortSignal`.
 * Cancelled, superseded, and invalidated requests never receive results. Input
 * text is transferred as a UTF-16 buffer and dropped once dispatched; errors are
 * {@link InferenceWorkerError} values with controlled codes only.
 */
export class InferenceWorkerClient {
  private readonly createWorker: () => InferenceWorkerEndpoint;
  private readonly maxQueueSize: number;
  private readonly maxInFlight: number;
  private readonly maxInputChars: number;
  private readonly acceptPlan: InferenceWorkerClientOptions["acceptPlan"];
  private endpoint: InferenceWorkerEndpoint | null = null;
  private listeners: ListenerSet | null = null;
  private stateValue: InferenceWorkerState = "idle";
  private generationValue = 0;
  private epochValue = 0;
  private nextId = 1;
  private queue: PendingRequest[] = [];
  private readonly inFlight = new Map<number, PendingRequest>();
  private pendingLoad: PendingLoad | null = null;
  private verifiedPlan: InferenceWorkerPlan | null = null;
  private verifiedSpec: NormalizedInferenceWorkerLoadSpec | null = null;
  private workerStarts = 0;
  private completed = 0;
  private cancelled = 0;
  private rejectedQueueFull = 0;
  private suppressedLateResults = 0;

  /**
   * Creates a client; the worker itself starts on the first `load()`.
   *
   * Args:
   *   options: Worker factory and queue bounds.
   *
   * Raises:
   *   TypeError: If `createWorker` is not a function.
   *   RangeError: If a bound is not a positive integer.
   */
  constructor(options: InferenceWorkerClientOptions) {
    if (typeof options?.createWorker !== "function") {
      throw new TypeError("Inference worker client requires createWorker.");
    }
    this.createWorker = options.createWorker;
    this.maxQueueSize = resolveBound(
      options.maxQueueSize,
      DEFAULT_INFERENCE_WORKER_MAX_QUEUE_SIZE,
    );
    this.maxInFlight = resolveBound(
      options.maxInFlight,
      DEFAULT_INFERENCE_WORKER_MAX_IN_FLIGHT,
    );
    this.maxInputChars = resolveBound(
      options.maxInputChars,
      DEFAULT_INFERENCE_WORKER_MAX_INPUT_CHARS,
    );
    this.acceptPlan = options.acceptPlan;
  }

  /** Current lifecycle state. */
  get state(): InferenceWorkerState {
    return this.stateValue;
  }

  /** Current model generation; increments on every `load()`. */
  get generation(): number {
    return this.generationValue;
  }

  /** Last verified plan while the client is ready, otherwise null. */
  get plan(): InferenceWorkerPlan | null {
    return this.stateValue === "ready" && this.verifiedPlan !== null
      ? { ...this.verifiedPlan }
      : null;
  }

  /**
   * Loads a model into the worker and starts a new model generation.
   *
   * Outstanding requests from earlier generations are rejected with
   * `superseded`, and pipelines created for them reject with `stale_generation`.
   *
   * Args:
   *   spec: Structured-clone-safe model description; `localOnly` defaults to true.
   *   options: Optional abort signal and `restart` to force a fresh worker.
   *
   * Returns:
   *   The verified execution plan reported by the worker.
   *
   * Raises:
   *   InferenceWorkerError: `invalid_request`, `load_failed`,
   *     `plan_incompatible`, `cancelled`, `superseded`, `worker_failed`, or
   *     `disposed`.
   */
  load(
    spec: InferenceWorkerLoadSpec,
    options: InferenceWorkerLoadOptions = {},
  ): Promise<InferenceWorkerPlan> {
    if (this.stateValue === "disposed") {
      return rejected("disposed");
    }
    const normalized = normalizeLoadSpec(spec);
    if (normalized === null) {
      return rejected("invalid_request");
    }
    if (options.signal?.aborted === true) {
      return rejected("cancelled");
    }
    this.generationValue += 1;
    this.verifiedPlan = null;
    this.verifiedSpec = null;
    return this.beginLoad(
      normalized,
      "load",
      null,
      options.signal,
      options.restart === true,
    );
  }

  /**
   * Reloads the last verified model after device loss, invalidation, or failure.
   *
   * Recovery never widens network access: the reload is forced to
   * `localOnly: true`, and the new plan must match the previous model reference
   * and digest before the generation is reused. A failed worker is replaced.
   *
   * Args:
   *   options: Optional abort signal and `restart` to force a fresh worker.
   *
   * Returns:
   *   The newly verified plan; the model generation is unchanged.
   *
   * Raises:
   *   InferenceWorkerError: `not_loaded` when no plan was ever verified, or
   *     the same codes as `load()`.
   */
  recover(options: InferenceWorkerRecoverOptions = {}): Promise<InferenceWorkerPlan> {
    if (this.stateValue === "disposed") {
      return rejected("disposed");
    }
    const previous = this.verifiedPlan;
    const spec = this.verifiedSpec;
    if (previous === null || spec === null) {
      return rejected("not_loaded");
    }
    if (this.stateValue === "loading") {
      return rejected("invalid_request");
    }
    if (this.stateValue === "ready" && options.restart !== true) {
      return Promise.resolve({ ...previous });
    }
    if (options.signal?.aborted === true) {
      return rejected("cancelled");
    }
    const restart =
      options.restart === true ||
      this.stateValue === "failed" ||
      this.endpoint === null;
    return this.beginLoad(
      {
        ...spec,
        localOnly: true,
        modelDigest: spec.modelDigest ?? previous.modelDigest,
      },
      "recover",
      previous,
      options.signal,
      restart,
    );
  }

  /**
   * Runs token classification in the worker.
   *
   * Args:
   *   text: Source text; it is transferred to the worker and not retained.
   *   options: Abort signal, plain-data call options, and owning generation.
   *
   * Returns:
   *   Aligned token entities with labels, scores, indices and offsets only.
   *
   * Raises:
   *   InferenceWorkerError: `queue_full`, `cancelled`, `stale_generation`,
   *     `superseded`, `not_loaded`, `plan_invalidated`, `device_lost`,
   *     `input_too_large`, `invalid_request`, `inference_failed`,
   *     `alignment_failed`, `worker_failed`, or `disposed`.
   */
  run(
    text: string,
    options: InferenceWorkerRunOptions = {},
  ): Promise<TokenClassificationEntity[]> {
    if (this.stateValue === "disposed") {
      return rejected("disposed");
    }
    if (typeof text !== "string") {
      return rejected("invalid_request");
    }
    if (text.length > this.maxInputChars) {
      return rejected("input_too_large");
    }
    const callOptions = options.callOptions ?? {};
    if (!isRecord(callOptions) || !isPlainData(callOptions)) {
      return rejected("invalid_request");
    }
    const signal = options.signal;
    if (signal?.aborted === true) {
      this.cancelled += 1;
      return rejected("cancelled");
    }
    const generation = options.generation ?? this.generationValue;
    const admission = this.admissionError(generation);
    if (admission !== null) {
      return rejected(admission);
    }
    if (this.queue.length >= this.maxQueueSize) {
      this.rejectedQueueFull += 1;
      return rejected("queue_full");
    }
    return new Promise<TokenClassificationEntity[]>((resolve, reject) => {
      const request: PendingRequest = {
        id: this.nextId++,
        generation,
        epoch: 0,
        text,
        callOptions: { ...callOptions },
        phase: "queued",
        settled: false,
        resolve,
        reject,
        cleanup: noop,
      };
      if (signal !== undefined) {
        const onAbort = () => this.cancelRequest(request);
        signal.addEventListener("abort", onAbort, { once: true });
        request.cleanup = () => signal.removeEventListener("abort", onAbort);
      }
      this.queue.push(request);
      this.pump();
    });
  }

  /**
   * Returns a token-classification pipeline owned by the current generation.
   *
   * Pass it to `extractPii()` or `deidentify()`; hashing and redaction stay on
   * the host. Calls after a later `load()` reject with `stale_generation`.
   *
   * Args:
   *   options: Optional abort signal applied to every call.
   *
   * Returns:
   *   A pipeline function routed through this client.
   */
  pipeline(options: InferenceWorkerPipelineOptions = {}): TokenClassificationPipeline {
    const generation = this.generationValue;
    const signal = options.signal;
    return (text: string, callOptions?: TokenClassificationCallOptions) => {
      const runOptions: InferenceWorkerRunOptions = { generation };
      if (signal !== undefined) {
        runOptions.signal = signal;
      }
      if (callOptions !== undefined) {
        runOptions.callOptions = callOptions;
      }
      return this.run(text, runOptions);
    };
  }

  /**
   * Invalidates the current execution plan, for example after the host sees
   * GPU device loss or replaces the runtime. Outstanding requests reject with
   * `plan_invalidated`; call `recover()` before reuse.
   */
  invalidate(): void {
    if (this.stateValue === "disposed" || this.stateValue === "failed") {
      return;
    }
    this.epochValue += 1;
    const pending = this.pendingLoad;
    if (pending !== null) {
      this.postCancel(pending.id);
      this.settleLoad(pending, "plan_invalidated");
    }
    this.rejectAll("plan_invalidated");
    this.stateValue = this.verifiedPlan === null ? "idle" : "invalidated";
  }

  /**
   * Rejects outstanding work with `disposed`, asks the worker to release its
   * runtime, and terminates it. Later calls reject with `disposed`.
   */
  dispose(): Promise<void> {
    if (this.stateValue === "disposed") {
      return Promise.resolve();
    }
    this.stateValue = "disposed";
    if (this.pendingLoad !== null) {
      this.settleLoad(this.pendingLoad, "disposed");
    }
    this.rejectAll("disposed");
    this.post({ protocol: INFERENCE_WORKER_PROTOCOL_VERSION, type: "dispose" });
    this.releaseEndpoint();
    this.verifiedPlan = null;
    this.verifiedSpec = null;
    return Promise.resolve();
  }

  /** Returns value-free counters; never includes text or model references. */
  stats(): InferenceWorkerStats {
    let draining = 0;
    for (const request of this.inFlight.values()) {
      if (request.phase === "draining") {
        draining += 1;
      }
    }
    return {
      state: this.stateValue,
      generation: this.generationValue,
      epoch: this.epochValue,
      queued: this.queue.length,
      inFlight: this.inFlight.size,
      draining,
      maxQueueSize: this.maxQueueSize,
      maxInFlight: this.maxInFlight,
      workerStarts: this.workerStarts,
      completed: this.completed,
      cancelled: this.cancelled,
      rejectedQueueFull: this.rejectedQueueFull,
      suppressedLateResults: this.suppressedLateResults,
    };
  }

  private beginLoad(
    spec: NormalizedInferenceWorkerLoadSpec,
    reason: InferenceWorkerLoadReason,
    previous: InferenceWorkerPlan | null,
    signal: AbortSignal | undefined,
    restart: boolean,
  ): Promise<InferenceWorkerPlan> {
    this.epochValue += 1;
    const epoch = this.epochValue;
    const generation = this.generationValue;
    if (this.pendingLoad !== null) {
      this.settleLoad(this.pendingLoad, "superseded");
    }
    this.rejectAll("superseded");
    if (!this.ensureEndpoint(restart)) {
      this.stateValue = "failed";
      return rejected("worker_failed");
    }
    this.stateValue = "loading";
    return new Promise<InferenceWorkerPlan>((resolve, reject) => {
      const pending: PendingLoad = {
        id: this.nextId++,
        epoch,
        generation,
        spec,
        previous,
        resolve,
        reject,
        cleanup: noop,
      };
      if (signal !== undefined) {
        const onAbort = () => this.cancelLoad(pending);
        signal.addEventListener("abort", onAbort, { once: true });
        pending.cleanup = () => signal.removeEventListener("abort", onAbort);
      }
      this.pendingLoad = pending;
      const posted = this.post({
        protocol: INFERENCE_WORKER_PROTOCOL_VERSION,
        type: "load",
        id: pending.id,
        generation,
        epoch,
        reason,
        spec,
      });
      if (!posted) {
        this.failWorker("worker_failed");
      }
    });
  }

  private admissionError(generation: number): InferenceWorkerErrorCode | null {
    if (this.generationValue === 0 || generation === 0) {
      return "not_loaded";
    }
    if (generation !== this.generationValue) {
      return "stale_generation";
    }
    switch (this.stateValue) {
      case "loading":
      case "ready":
        return null;
      case "invalidated":
        return "plan_invalidated";
      case "failed":
        return "worker_failed";
      case "disposed":
        return "disposed";
      case "idle":
        return "not_loaded";
    }
  }

  private pump(): void {
    if (this.stateValue !== "ready" || this.endpoint === null) {
      return;
    }
    while (this.inFlight.size < this.maxInFlight && this.queue.length > 0) {
      const request = this.queue.shift();
      if (request !== undefined) {
        this.dispatch(request);
      }
    }
  }

  private dispatch(request: PendingRequest): void {
    const input = encodeTextUnits(request.text ?? "");
    request.text = null;
    request.phase = "dispatched";
    request.epoch = this.epochValue;
    this.inFlight.set(request.id, request);
    const posted = this.post(
      {
        protocol: INFERENCE_WORKER_PROTOCOL_VERSION,
        type: "infer",
        id: request.id,
        epoch: request.epoch,
        input,
        callOptions: request.callOptions,
      },
      [input],
    );
    if (!posted) {
      this.inFlight.delete(request.id);
      this.settleRequest(request, "invalid_request");
    }
  }

  private cancelRequest(request: PendingRequest): void {
    if (request.settled) {
      return;
    }
    this.cancelled += 1;
    if (request.phase === "queued") {
      const position = this.queue.indexOf(request);
      if (position !== -1) {
        this.queue.splice(position, 1);
      }
      request.text = null;
      this.settleRequest(request, "cancelled");
      return;
    }
    request.phase = "draining";
    this.postCancel(request.id);
    this.settleRequest(request, "cancelled");
  }

  private cancelLoad(pending: PendingLoad): void {
    if (this.pendingLoad !== pending) {
      return;
    }
    this.postCancel(pending.id);
    this.settleLoad(pending, "cancelled");
    this.stateValue = pending.previous === null ? "idle" : "invalidated";
    this.rejectQueued(pending.previous === null ? "not_loaded" : "plan_invalidated");
  }

  private onMessage(data: unknown): void {
    if (!isRecord(data) || data.protocol !== INFERENCE_WORKER_PROTOCOL_VERSION) {
      this.failWorker("protocol_error");
      return;
    }
    switch (data.type) {
      case "ready":
        this.onReady(data);
        return;
      case "result":
        this.onResult(data);
        return;
      case "error":
        this.onError(data);
        return;
      case "cancelled":
        this.onCancelled(data);
        return;
      case "device_lost":
        this.onDeviceLost(data);
        return;
      case "disposed":
        return;
      default:
        this.failWorker("protocol_error");
    }
  }

  private onReady(data: Record<string, unknown>): void {
    const pending = this.pendingLoad;
    if (pending === null || data.id !== pending.id) {
      this.suppressedLateResults += 1;
      return;
    }
    const plan = parsePlan(data.plan);
    if (plan === null || data.epoch !== pending.epoch) {
      this.failWorker("protocol_error");
      return;
    }
    if (!this.verifyPlan(plan, pending)) {
      this.settleLoad(pending, "plan_incompatible");
      this.stateValue = pending.previous === null ? "idle" : "invalidated";
      this.rejectQueued(pending.previous === null ? "not_loaded" : "plan_invalidated");
      return;
    }
    this.pendingLoad = null;
    pending.cleanup();
    this.verifiedPlan = plan;
    this.verifiedSpec = pending.spec;
    this.stateValue = "ready";
    pending.resolve({ ...plan });
    this.pump();
  }

  private onResult(data: Record<string, unknown>): void {
    const request = this.takeInFlight(data);
    if (request === null) {
      return;
    }
    const tokens = decodeInferenceTokens(data.tokens);
    if (tokens === null) {
      this.settleRequest(request, "protocol_error");
      this.failWorker("protocol_error");
      return;
    }
    if (!request.settled) {
      this.completed += 1;
      request.settled = true;
      request.cleanup();
      request.resolve(tokens);
    }
    this.pump();
  }

  private onError(data: Record<string, unknown>): void {
    const code = isInferenceWorkerErrorCode(data.code) ? data.code : null;
    if (code === null) {
      this.failWorker("protocol_error");
      return;
    }
    const pending = this.pendingLoad;
    if (pending !== null && data.id === pending.id) {
      this.settleLoad(pending, code);
      this.stateValue = pending.previous === null ? "idle" : "invalidated";
      this.rejectQueued(pending.previous === null ? "not_loaded" : "plan_invalidated");
      return;
    }
    const request = this.takeInFlight(data);
    if (request === null) {
      return;
    }
    this.settleRequest(request, code);
    this.pump();
  }

  private onCancelled(data: Record<string, unknown>): void {
    const request = isPositiveId(data.id) ? this.inFlight.get(data.id) : undefined;
    if (request === undefined) {
      return;
    }
    if (data.epoch !== request.epoch) {
      this.failWorker("protocol_error");
      return;
    }
    this.inFlight.delete(request.id);
    this.settleRequest(request, "cancelled");
    this.pump();
  }

  private onDeviceLost(data: Record<string, unknown>): void {
    if (data.epoch !== this.epochValue || this.stateValue !== "ready") {
      return;
    }
    this.stateValue = "invalidated";
    this.rejectAll("device_lost");
  }

  /** Removes the addressed in-flight request; late or stale replies return null. */
  private takeInFlight(data: Record<string, unknown>): PendingRequest | null {
    const request = isPositiveId(data.id) ? this.inFlight.get(data.id) : undefined;
    if (request === undefined) {
      this.suppressedLateResults += 1;
      return null;
    }
    if (data.epoch !== request.epoch) {
      this.failWorker("protocol_error");
      return null;
    }
    this.inFlight.delete(request.id);
    if (request.phase === "draining") {
      this.suppressedLateResults += 1;
      this.pump();
      return null;
    }
    return request;
  }

  private verifyPlan(plan: InferenceWorkerPlan, pending: PendingLoad): boolean {
    const spec = pending.spec;
    const previous = pending.previous;
    const builtIn =
      plan.generation === pending.generation &&
      plan.epoch === pending.epoch &&
      plan.modelRef === spec.modelRef &&
      (!spec.localOnly || plan.localOnly) &&
      (spec.allowedBackends === null || spec.allowedBackends.includes(plan.backend)) &&
      (spec.modelDigest === null || plan.modelDigest === spec.modelDigest) &&
      (previous === null ||
        (plan.localOnly &&
          plan.modelRef === previous.modelRef &&
          (previous.modelDigest === null ||
            plan.modelDigest === previous.modelDigest)));
    if (!builtIn) {
      return false;
    }
    if (this.acceptPlan === undefined) {
      return true;
    }
    try {
      return (
        this.acceptPlan(
          { ...plan },
          previous === null ? null : { ...previous },
        ) === true
      );
    } catch {
      return false;
    }
  }

  private ensureEndpoint(restart: boolean): boolean {
    if (this.endpoint !== null && !restart) {
      return true;
    }
    this.releaseEndpoint();
    let endpoint: InferenceWorkerEndpoint;
    try {
      endpoint = this.createWorker();
    } catch {
      return false;
    }
    const listeners: ListenerSet = {
      message: (event) => this.onMessage(event?.data),
      messageerror: () => this.failWorker("protocol_error"),
      error: (event) => {
        // Cancel default reporting so a worker error message never reaches
        // the console; the client only surfaces a controlled code.
        try {
          event?.preventDefault?.();
        } catch {
          // Ignore hosts whose error events cannot be cancelled.
        }
        this.failWorker("worker_failed");
      },
    };
    try {
      endpoint.addEventListener("message", listeners.message);
      endpoint.addEventListener("messageerror", listeners.messageerror);
      endpoint.addEventListener("error", listeners.error);
    } catch {
      terminate(endpoint);
      return false;
    }
    this.endpoint = endpoint;
    this.listeners = listeners;
    this.workerStarts += 1;
    return true;
  }

  private releaseEndpoint(): void {
    const endpoint = this.endpoint;
    const listeners = this.listeners;
    this.endpoint = null;
    this.listeners = null;
    if (endpoint === null) {
      return;
    }
    if (listeners !== null) {
      for (const type of ["message", "messageerror", "error"] as const) {
        try {
          endpoint.removeEventListener(type, listeners[type]);
        } catch {
          // Listener removal is best effort during teardown.
        }
      }
    }
    terminate(endpoint);
  }

  private failWorker(code: InferenceWorkerErrorCode): void {
    if (this.stateValue === "disposed") {
      return;
    }
    this.releaseEndpoint();
    this.epochValue += 1;
    this.stateValue = "failed";
    if (this.pendingLoad !== null) {
      this.settleLoad(this.pendingLoad, code);
    }
    this.rejectAll(code);
  }

  private rejectAll(code: InferenceWorkerErrorCode): void {
    const inFlight = [...this.inFlight.values()];
    this.inFlight.clear();
    for (const request of inFlight) {
      this.settleRequest(request, code);
    }
    this.rejectQueued(code);
  }

  private rejectQueued(code: InferenceWorkerErrorCode): void {
    const queued = this.queue;
    this.queue = [];
    for (const request of queued) {
      request.text = null;
      this.settleRequest(request, code);
    }
  }

  private settleRequest(request: PendingRequest, code: InferenceWorkerErrorCode): void {
    if (request.settled) {
      return;
    }
    request.settled = true;
    request.cleanup();
    request.reject(new InferenceWorkerError(code));
  }

  private settleLoad(pending: PendingLoad, code: InferenceWorkerErrorCode): void {
    if (this.pendingLoad === pending) {
      this.pendingLoad = null;
    }
    pending.cleanup();
    pending.reject(new InferenceWorkerError(code));
  }

  private postCancel(id: number): void {
    this.post({ protocol: INFERENCE_WORKER_PROTOCOL_VERSION, type: "cancel", id });
  }

  private post(message: HostToWorkerMessage, transfer: ArrayBuffer[] = []): boolean {
    if (this.endpoint === null) {
      return false;
    }
    try {
      this.endpoint.postMessage(message, transfer);
      return true;
    } catch {
      return false;
    }
  }
}

/**
 * Creates an {@link InferenceWorkerClient}.
 *
 * Args:
 *   options: Worker factory and queue bounds.
 *
 * Returns:
 *   A client in the `idle` state; call `load()` before running requests.
 */
export function createInferenceWorkerClient(
  options: InferenceWorkerClientOptions,
): InferenceWorkerClient {
  return new InferenceWorkerClient(options);
}

function normalizeLoadSpec(
  spec: InferenceWorkerLoadSpec,
): NormalizedInferenceWorkerLoadSpec | null {
  if (!isRecord(spec)) {
    return null;
  }
  const { modelRef, allowedBackends, modelDigest } = spec;
  const localOnly = spec.localOnly ?? true;
  const options = spec.options ?? {};
  if (
    typeof modelRef !== "string" ||
    modelRef.trim().length === 0 ||
    typeof localOnly !== "boolean" ||
    !(
      allowedBackends === undefined ||
      (Array.isArray(allowedBackends) &&
        allowedBackends.length > 0 &&
        allowedBackends.every((backend) => typeof backend === "string"))
    ) ||
    !(modelDigest === undefined || typeof modelDigest === "string") ||
    !isRecord(options) ||
    !isPlainData(options)
  ) {
    return null;
  }
  if (localOnly && !isOfflineModelRef(modelRef)) {
    return null;
  }
  return {
    modelRef,
    localOnly,
    allowedBackends: allowedBackends === undefined ? null : [...allowedBackends],
    modelDigest: modelDigest ?? null,
    options: { ...options },
  };
}

function parsePlan(value: unknown): InferenceWorkerPlan | null {
  if (!isRecord(value)) {
    return null;
  }
  const { protocol, generation, epoch, modelRef, backend, localOnly, modelDigest } =
    value;
  if (
    protocol !== INFERENCE_WORKER_PROTOCOL_VERSION ||
    !isPositiveId(generation) ||
    !isPositiveId(epoch) ||
    typeof modelRef !== "string" ||
    typeof backend !== "string" ||
    typeof localOnly !== "boolean" ||
    !(modelDigest === null || typeof modelDigest === "string")
  ) {
    return null;
  }
  return {
    protocol,
    generation,
    epoch,
    modelRef,
    backend,
    localOnly,
    modelDigest,
  };
}

function rejected<T>(code: InferenceWorkerErrorCode): Promise<T> {
  return Promise.reject(new InferenceWorkerError(code));
}

function terminate(endpoint: InferenceWorkerEndpoint): void {
  try {
    endpoint.terminate?.();
  } catch {
    // Termination is best effort; the endpoint is no longer referenced.
  }
}

function noop(): void {}
