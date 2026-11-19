import { flattenTokenClassificationOutput } from "../decoder";
import { alignTokenOffsets } from "../offsets";
import type {
  RawTokenClassificationOutput,
  TokenClassificationCallOptions,
  TokenClassificationEntity,
} from "../types";
import {
  DEFAULT_INFERENCE_WORKER_MAX_INPUT_CHARS,
  DEFAULT_INFERENCE_WORKER_MAX_QUEUE_SIZE,
  INFERENCE_WORKER_PROTOCOL_VERSION,
  InferenceWorkerError,
  decodeTextUnits,
  encodeInferenceTokens,
  isOfflineModelRef,
  isPlainData,
  isPositiveId,
  isRecord,
  resolveBound,
  type HostToWorkerMessage,
  type InferenceWorkerErrorCode,
  type InferenceWorkerLoadReason,
  type InferenceWorkerPlan,
  type NormalizedInferenceWorkerLoadSpec,
  type WorkerToHostMessage,
} from "./protocol";

/** Per-request context handed to a worker pipeline. */
export interface InferenceWorkerRunContext {
  /** Aborted when the host cancels the request or the plan is invalidated. */
  signal: AbortSignal;
}

/**
 * Token-classification pipeline that runs inside the worker.
 *
 * Any `RawTokenClassificationPipeline` fits; pipelines that can stop early may
 * also observe the third `context.signal` argument.
 */
export type InferenceWorkerPipeline = (
  text: string,
  options?: TokenClassificationCallOptions,
  context?: InferenceWorkerRunContext,
) => RawTokenClassificationOutput | Promise<RawTokenClassificationOutput>;

/** Runtime returned by the worker's `load` handler. */
export interface InferenceWorkerRuntime {
  /** Pipeline that tokenizes and classifies text inside the worker. */
  pipeline: InferenceWorkerPipeline;
  /** Selected backend, for example `"webgpu"` or `"wasm-basic"`. */
  backend: string;
  /** True only when the runtime makes no remote model or asset requests. */
  localOnly: boolean;
  /** Optional digest of the loaded model artifact for plan verification. */
  modelDigest?: string | null;
  /** Settles when the GPU device is lost, for example `GPUDevice.lost`. */
  deviceLost?: PromiseLike<unknown>;
  /** Releases the runtime; defaults to `pipeline.dispose()` when present. */
  dispose?: () => void | Promise<void>;
}

/** Context passed to the worker's `load` handler. */
export interface InferenceWorkerLoadContext {
  /** Model generation owning this runtime. */
  generation: number;
  /** Plan epoch assigned by the host for this load. */
  epoch: number;
  /** `"recover"` after device loss, invalidation, or a worker restart. */
  reason: InferenceWorkerLoadReason;
  /** Aborted when the load is cancelled or superseded. */
  signal: AbortSignal;
}

/** Minimal dedicated-worker scope used by the server. */
export interface InferenceWorkerScope {
  postMessage(message: unknown, transfer: Transferable[]): void;
  addEventListener(
    type: "message",
    listener: (event: { data?: unknown }) => void,
  ): void;
  removeEventListener(
    type: "message",
    listener: (event: { data?: unknown }) => void,
  ): void;
  close?(): void;
}

/** Options for {@link serveInferenceWorker}. */
export interface ServeInferenceWorkerOptions {
  /** Loads a runtime for a verified, normalised load spec. */
  load: (
    spec: NormalizedInferenceWorkerLoadSpec,
    context: InferenceWorkerLoadContext,
  ) => InferenceWorkerRuntime | Promise<InferenceWorkerRuntime>;
  /** Worker scope; defaults to `globalThis` inside a dedicated worker. */
  scope?: InferenceWorkerScope;
  /** Requests the worker holds before replying `queue_full`. Defaults to 16. */
  maxQueueSize?: number;
  /** Per-request input bound in UTF-16 code units. Defaults to 1,000,000. */
  maxInputChars?: number;
  /** Close the scope after a host `dispose` message. Defaults to true. */
  closeOnDispose?: boolean;
}

/** Value-free counters describing the worker server. */
export interface InferenceWorkerServerStats {
  loaded: boolean;
  epoch: number;
  queued: number;
  running: boolean;
  completed: number;
  cancelled: number;
  failed: number;
}

/** Handle returned by {@link serveInferenceWorker}. */
export interface InferenceWorkerServer {
  /** Returns value-free counters; never includes input text or model refs. */
  stats(): InferenceWorkerServerStats;
  /** Stops listening, cancels queued work and releases the runtime. */
  close(): Promise<void>;
}

interface InferJob {
  id: number;
  epoch: number;
  input: ArrayBuffer | null;
  callOptions: Record<string, unknown>;
  controller: AbortController;
  cancelled: boolean;
  supersededBy: InferenceWorkerErrorCode | null;
}

interface LoadJob {
  controller: AbortController;
  cancelled: boolean;
}

type ParsedHostMessage =
  | { ok: true; message: HostToWorkerMessage }
  | { ok: false; id: number; epoch: number; code: InferenceWorkerErrorCode };

const HANDLER_ERROR_CODES: ReadonlySet<InferenceWorkerErrorCode> = new Set([
  "alignment_failed",
  "cancelled",
  "device_lost",
  "inference_failed",
  "input_too_large",
  "invalid_request",
  "load_failed",
  "plan_incompatible",
]);

const MAX_MODEL_REF_CHARS = 4096;

/**
 * Serves OpenMed token-classification requests from inside a dedicated worker.
 *
 * The server runs one request at a time from a bounded queue, aborts work the
 * host cancels, never posts results for cancelled or superseded requests,
 * invalidates its runtime on GPU device loss, and reports failures only as
 * controlled codes. It never logs, stores, or echoes input text.
 *
 * Args:
 *   options: The runtime loader, optional scope, and queue bounds.
 *
 * Returns:
 *   A handle exposing value-free stats and an explicit `close()`.
 *
 * Raises:
 *   TypeError: If no worker scope is available.
 *   RangeError: If a bound is not a positive integer.
 */
export function serveInferenceWorker(
  options: ServeInferenceWorkerOptions,
): InferenceWorkerServer {
  if (typeof options?.load !== "function") {
    throw new TypeError("serveInferenceWorker requires a load handler.");
  }
  const scope = options.scope ?? defaultScope();
  const server = new InferenceWorkerServerImpl(scope, options);
  return {
    stats: () => server.stats(),
    close: () => server.close(),
  };
}

class InferenceWorkerServerImpl {
  private readonly scope: InferenceWorkerScope;
  private readonly loadRuntime: ServeInferenceWorkerOptions["load"];
  private readonly maxQueueSize: number;
  private readonly maxInputChars: number;
  private readonly closeOnDispose: boolean;
  private readonly listener: (event: { data?: unknown }) => void;
  private runtime: InferenceWorkerRuntime | null = null;
  private epoch = 0;
  private lostEpoch = 0;
  private latestLoadEpoch = 0;
  private readonly loads = new Map<number, LoadJob>();
  private lane: Promise<void> = Promise.resolve();
  private queue: InferJob[] = [];
  private running: InferJob | null = null;
  private runningDone: Promise<void> = Promise.resolve();
  private closed = false;
  private completed = 0;
  private cancelledCount = 0;
  private failed = 0;

  constructor(scope: InferenceWorkerScope, options: ServeInferenceWorkerOptions) {
    this.scope = scope;
    this.loadRuntime = options.load;
    this.maxQueueSize = resolveBound(
      options.maxQueueSize,
      DEFAULT_INFERENCE_WORKER_MAX_QUEUE_SIZE,
    );
    this.maxInputChars = resolveBound(
      options.maxInputChars,
      DEFAULT_INFERENCE_WORKER_MAX_INPUT_CHARS,
    );
    this.closeOnDispose = options.closeOnDispose ?? true;
    this.listener = (event) => this.onMessage(event?.data);
    scope.addEventListener("message", this.listener);
  }

  stats(): InferenceWorkerServerStats {
    return {
      loaded: this.runtime !== null,
      epoch: this.runtime === null ? 0 : this.epoch,
      queued: this.queue.length,
      running: this.running !== null,
      completed: this.completed,
      cancelled: this.cancelledCount,
      failed: this.failed,
    };
  }

  close(): Promise<void> {
    if (!this.closed) {
      this.closed = true;
      this.scope.removeEventListener("message", this.listener);
      this.shutdown(false);
    }
    return this.lane;
  }

  private onMessage(data: unknown): void {
    if (this.closed) {
      return;
    }
    const parsed = this.parse(data);
    if (!parsed.ok) {
      this.post({
        protocol: INFERENCE_WORKER_PROTOCOL_VERSION,
        type: "error",
        id: parsed.id,
        epoch: parsed.epoch,
        code: parsed.code,
      });
      return;
    }
    const message = parsed.message;
    switch (message.type) {
      case "load":
        this.onLoad(message);
        return;
      case "infer":
        this.onInfer(message);
        return;
      case "cancel":
        this.onCancel(message.id);
        return;
      case "dispose":
        this.closed = true;
        this.shutdown(true);
        return;
    }
  }

  private onLoad(message: Extract<HostToWorkerMessage, { type: "load" }>): void {
    this.latestLoadEpoch = Math.max(this.latestLoadEpoch, message.epoch);
    for (const load of this.loads.values()) {
      load.controller.abort();
    }
    this.supersedeWork("superseded");
    const job: LoadJob = { controller: new AbortController(), cancelled: false };
    this.loads.set(message.id, job);
    this.enqueueLane(async () => {
      try {
        await this.performLoad(message, job);
      } finally {
        this.loads.delete(message.id);
      }
    });
  }

  private async performLoad(
    message: Extract<HostToWorkerMessage, { type: "load" }>,
    job: LoadJob,
  ): Promise<void> {
    await this.runningDone;
    await this.releaseRuntime();
    const interrupted = (): InferenceWorkerErrorCode | null => {
      if (job.cancelled) {
        return "cancelled";
      }
      if (job.controller.signal.aborted || message.epoch !== this.latestLoadEpoch) {
        return "superseded";
      }
      return this.closed ? "disposed" : null;
    };
    const early = interrupted();
    if (early !== null) {
      this.postError(message.id, message.epoch, early);
      return;
    }
    let runtime: InferenceWorkerRuntime;
    try {
      runtime = await this.loadRuntime(message.spec, {
        generation: message.generation,
        epoch: message.epoch,
        reason: message.reason,
        signal: job.controller.signal,
      });
    } catch (error) {
      this.postError(
        message.id,
        message.epoch,
        interrupted() ?? handlerErrorCode(error, "load_failed"),
      );
      return;
    }
    const late = interrupted();
    if (late !== null) {
      await disposeRuntime(runtime);
      this.postError(message.id, message.epoch, late);
      return;
    }
    const plan = verifyRuntime(runtime, message);
    if (typeof plan === "string") {
      await disposeRuntime(runtime);
      this.postError(message.id, message.epoch, plan);
      return;
    }
    this.runtime = runtime;
    this.epoch = message.epoch;
    this.watchDeviceLoss(runtime, message.epoch);
    this.post({
      protocol: INFERENCE_WORKER_PROTOCOL_VERSION,
      type: "ready",
      id: message.id,
      epoch: message.epoch,
      plan,
    });
  }

  private onInfer(message: Extract<HostToWorkerMessage, { type: "infer" }>): void {
    if (this.runtime === null) {
      const code = message.epoch === this.lostEpoch ? "device_lost" : "not_loaded";
      this.postError(message.id, message.epoch, code);
      return;
    }
    if (message.epoch !== this.epoch || this.latestLoadEpoch !== this.epoch) {
      this.postError(message.id, message.epoch, "stale_generation");
      return;
    }
    if (this.queue.length >= this.maxQueueSize) {
      this.postError(message.id, message.epoch, "queue_full");
      return;
    }
    this.queue.push({
      id: message.id,
      epoch: message.epoch,
      input: message.input,
      callOptions: message.callOptions,
      controller: new AbortController(),
      cancelled: false,
      supersededBy: null,
    });
    this.pump();
  }

  private onCancel(id: number): void {
    const load = this.loads.get(id);
    if (load !== undefined) {
      load.cancelled = true;
      load.controller.abort();
      return;
    }
    const queued = this.queue.findIndex((job) => job.id === id);
    if (queued !== -1) {
      const [job] = this.queue.splice(queued, 1);
      if (job !== undefined) {
        job.input = null;
        this.cancelledCount += 1;
        this.post({
          protocol: INFERENCE_WORKER_PROTOCOL_VERSION,
          type: "cancelled",
          id: job.id,
          epoch: job.epoch,
        });
      }
      return;
    }
    if (this.running?.id === id) {
      this.running.cancelled = true;
      this.running.controller.abort();
    }
  }

  private pump(): void {
    const runtime = this.runtime;
    if (this.running !== null || runtime === null) {
      return;
    }
    const job = this.queue.shift();
    if (job === undefined) {
      return;
    }
    this.running = job;
    this.runningDone = this.execute(job, runtime).then(
      () => this.finishJob(job),
      () => this.finishJob(job),
    );
  }

  private finishJob(job: InferJob): void {
    if (this.running === job) {
      this.running = null;
    }
    this.pump();
  }

  private async execute(job: InferJob, runtime: InferenceWorkerRuntime): Promise<void> {
    let aligned: TokenClassificationEntity[] | null = null;
    let code: InferenceWorkerErrorCode | null = null;
    let text: string | null = decodeTextUnits(job.input ?? new ArrayBuffer(0));
    job.input = null;
    try {
      const output = await runtime.pipeline(
        text,
        job.callOptions as TokenClassificationCallOptions,
        { signal: job.controller.signal },
      );
      if (!Array.isArray(output)) {
        code = "inference_failed";
      } else if (!job.controller.signal.aborted) {
        try {
          aligned = alignTokenOffsets(text, flattenTokenClassificationOutput(output));
        } catch {
          code = "alignment_failed";
        }
      }
    } catch (error) {
      code = handlerErrorCode(error, "inference_failed");
    } finally {
      text = null;
    }

    if (job.cancelled) {
      this.cancelledCount += 1;
      this.post({
        protocol: INFERENCE_WORKER_PROTOCOL_VERSION,
        type: "cancelled",
        id: job.id,
        epoch: job.epoch,
      });
      return;
    }
    if (job.supersededBy !== null) {
      this.postError(job.id, job.epoch, job.supersededBy);
      return;
    }
    if (code === "device_lost") {
      this.onDeviceLost(runtime, job.epoch);
    }
    if (code !== null || aligned === null) {
      this.failed += 1;
      this.postError(job.id, job.epoch, code ?? "inference_failed");
      return;
    }
    const encoded = encodeInferenceTokens(aligned);
    const posted = this.post(
      {
        protocol: INFERENCE_WORKER_PROTOCOL_VERSION,
        type: "result",
        id: job.id,
        epoch: job.epoch,
        tokens: encoded.tokens,
      },
      encoded.transfer,
    );
    if (posted) {
      this.completed += 1;
    } else {
      this.failed += 1;
      this.postError(job.id, job.epoch, "inference_failed");
    }
  }

  private watchDeviceLoss(runtime: InferenceWorkerRuntime, epoch: number): void {
    if (runtime.deviceLost === undefined || runtime.deviceLost === null) {
      return;
    }
    const lost = () => this.onDeviceLost(runtime, epoch);
    Promise.resolve(runtime.deviceLost).then(lost, lost);
  }

  private onDeviceLost(runtime: InferenceWorkerRuntime, epoch: number): void {
    if (this.runtime !== runtime || this.epoch !== epoch) {
      return;
    }
    this.runtime = null;
    this.lostEpoch = epoch;
    this.supersedeWork("device_lost");
    this.post({
      protocol: INFERENCE_WORKER_PROTOCOL_VERSION,
      type: "device_lost",
      epoch,
    });
    this.enqueueLane(async () => {
      await this.runningDone;
      await disposeRuntime(runtime);
    });
  }

  private supersedeWork(code: InferenceWorkerErrorCode): void {
    const queued = this.queue;
    this.queue = [];
    for (const job of queued) {
      job.input = null;
      this.postError(job.id, job.epoch, code);
    }
    const running = this.running;
    if (running !== null && running.supersededBy === null) {
      running.supersededBy = code;
      running.controller.abort();
    }
  }

  private shutdown(announce: boolean): void {
    for (const load of this.loads.values()) {
      load.controller.abort();
    }
    this.supersedeWork("disposed");
    this.enqueueLane(async () => {
      await this.runningDone;
      await this.releaseRuntime();
      if (announce) {
        this.post({ protocol: INFERENCE_WORKER_PROTOCOL_VERSION, type: "disposed" });
        this.scope.removeEventListener("message", this.listener);
        if (this.closeOnDispose) {
          try {
            this.scope.close?.();
          } catch {
            // Closing is best effort; the host terminates the worker anyway.
          }
        }
      }
    });
  }

  private async releaseRuntime(): Promise<void> {
    const runtime = this.runtime;
    this.runtime = null;
    if (runtime !== null) {
      await disposeRuntime(runtime);
    }
  }

  private enqueueLane(task: () => Promise<void>): void {
    this.lane = this.lane.then(task).catch(() => undefined);
  }

  private postError(id: number, epoch: number, code: InferenceWorkerErrorCode): void {
    this.post({
      protocol: INFERENCE_WORKER_PROTOCOL_VERSION,
      type: "error",
      id,
      epoch,
      code,
    });
  }

  private post(message: WorkerToHostMessage, transfer: ArrayBuffer[] = []): boolean {
    try {
      this.scope.postMessage(message, transfer);
      return true;
    } catch {
      return false;
    }
  }

  private parse(data: unknown): ParsedHostMessage {
    if (!isRecord(data) || data.protocol !== INFERENCE_WORKER_PROTOCOL_VERSION) {
      return { ok: false, id: 0, epoch: 0, code: "protocol_error" };
    }
    const id = isPositiveId(data.id) ? data.id : 0;
    const epoch = isPositiveId(data.epoch) ? data.epoch : 0;
    switch (data.type) {
      case "dispose":
        return {
          ok: true,
          message: { protocol: INFERENCE_WORKER_PROTOCOL_VERSION, type: "dispose" },
        };
      case "cancel":
        return id === 0
          ? { ok: false, id, epoch, code: "invalid_request" }
          : {
              ok: true,
              message: {
                protocol: INFERENCE_WORKER_PROTOCOL_VERSION,
                type: "cancel",
                id,
              },
            };
      case "infer":
        return this.parseInfer(data, id, epoch);
      case "load":
        return parseLoad(data, id, epoch);
      default:
        return { ok: false, id, epoch, code: "protocol_error" };
    }
  }

  private parseInfer(
    data: Record<string, unknown>,
    id: number,
    epoch: number,
  ): ParsedHostMessage {
    const { input, callOptions } = data;
    if (
      id === 0 ||
      epoch === 0 ||
      !(input instanceof ArrayBuffer) ||
      input.byteLength % 2 !== 0 ||
      !isRecord(callOptions) ||
      !isPlainData(callOptions)
    ) {
      return { ok: false, id, epoch, code: "invalid_request" };
    }
    if (input.byteLength / 2 > this.maxInputChars) {
      return { ok: false, id, epoch, code: "input_too_large" };
    }
    return {
      ok: true,
      message: {
        protocol: INFERENCE_WORKER_PROTOCOL_VERSION,
        type: "infer",
        id,
        epoch,
        input,
        callOptions,
      },
    };
  }
}

function parseLoad(
  data: Record<string, unknown>,
  id: number,
  epoch: number,
): ParsedHostMessage {
  const { generation, reason, spec } = data;
  if (
    id === 0 ||
    epoch === 0 ||
    !isPositiveId(generation) ||
    (reason !== "load" && reason !== "recover") ||
    !isRecord(spec)
  ) {
    return { ok: false, id, epoch, code: "invalid_request" };
  }
  const { modelRef, localOnly, allowedBackends, modelDigest, options } = spec;
  if (
    typeof modelRef !== "string" ||
    modelRef.trim().length === 0 ||
    modelRef.length > MAX_MODEL_REF_CHARS ||
    typeof localOnly !== "boolean" ||
    !(
      allowedBackends === null ||
      (Array.isArray(allowedBackends) &&
        allowedBackends.every((backend) => typeof backend === "string"))
    ) ||
    !(modelDigest === null || typeof modelDigest === "string") ||
    !isRecord(options) ||
    !isPlainData(options)
  ) {
    return { ok: false, id, epoch, code: "invalid_request" };
  }
  if (localOnly && !isOfflineModelRef(modelRef)) {
    return { ok: false, id, epoch, code: "invalid_request" };
  }
  return {
    ok: true,
    message: {
      protocol: INFERENCE_WORKER_PROTOCOL_VERSION,
      type: "load",
      id,
      generation,
      epoch,
      reason,
      spec: {
        modelRef,
        localOnly,
        allowedBackends: allowedBackends as string[] | null,
        modelDigest,
        options,
      },
    },
  };
}

function verifyRuntime(
  runtime: unknown,
  message: Extract<HostToWorkerMessage, { type: "load" }>,
): InferenceWorkerPlan | InferenceWorkerErrorCode {
  if (!isRecord(runtime) || typeof runtime.pipeline !== "function") {
    return "load_failed";
  }
  const { backend, localOnly } = runtime;
  const modelDigest = runtime.modelDigest ?? null;
  const spec = message.spec;
  if (
    typeof backend !== "string" ||
    backend.length === 0 ||
    typeof localOnly !== "boolean" ||
    (spec.localOnly && !localOnly) ||
    (spec.allowedBackends !== null && !spec.allowedBackends.includes(backend)) ||
    !(modelDigest === null || typeof modelDigest === "string") ||
    (spec.modelDigest !== null && modelDigest !== spec.modelDigest)
  ) {
    return "plan_incompatible";
  }
  return {
    protocol: INFERENCE_WORKER_PROTOCOL_VERSION,
    generation: message.generation,
    epoch: message.epoch,
    modelRef: spec.modelRef,
    backend,
    localOnly,
    modelDigest,
  };
}

async function disposeRuntime(runtime: InferenceWorkerRuntime): Promise<void> {
  try {
    if (typeof runtime.dispose === "function") {
      await runtime.dispose();
      return;
    }
    const pipelineDispose = (runtime.pipeline as { dispose?: unknown }).dispose;
    if (typeof pipelineDispose === "function") {
      await (pipelineDispose as () => unknown).call(runtime.pipeline);
    }
  } catch {
    // Release failures are swallowed so no runtime message reaches the host.
  }
}

function handlerErrorCode(
  error: unknown,
  fallback: InferenceWorkerErrorCode,
): InferenceWorkerErrorCode {
  return error instanceof InferenceWorkerError && HANDLER_ERROR_CODES.has(error.code)
    ? error.code
    : fallback;
}

function defaultScope(): InferenceWorkerScope {
  const candidate = globalThis as unknown as Partial<InferenceWorkerScope>;
  if (
    typeof candidate.postMessage !== "function" ||
    typeof candidate.addEventListener !== "function" ||
    typeof candidate.removeEventListener !== "function"
  ) {
    throw new TypeError("serveInferenceWorker requires a dedicated worker scope.");
  }
  return candidate as InferenceWorkerScope;
}
