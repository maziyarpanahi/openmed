/*
 * Local-only LiquidAI LFM2.5 chat adapter for Transformers.js + WebGPU.
 *
 * Operators must mirror the pinned Q4F16 snapshot under this page's origin.
 * The adapter never falls back to Hugging Face, a CDN, or cloud inference.
 */

const ADAPTER_URL = import.meta.url;
const TRANSFORMERS_MODULE_PATH = "./vendor/transformers.web.min.js";
const WORKER_PATH = "./lfm25-chat-worker.mjs";
const NETWORK_POLICY = "same-origin-model-assets-only";
const SOURCE_MODEL = "LiquidAI/LFM2.5-2.6B-ONNX";
const SOURCE_REVISION = "66826372fd4fa166f53be0371c9315745c07cace";
const MODEL_VARIANT = "q4f16";
const MODEL_BYTES = 1_534_153_680;
const REQUIRED_GRAPH_FILES = Object.freeze([
  Object.freeze({
    path: "onnx/model_q4f16.onnx",
    sha256: "871354d43abc0d718a7a089d2fe10ad5d1f83e08acbfb9f45e4d137b41a1e9a4",
    sizeBytes: 222_160,
  }),
  Object.freeze({
    path: "onnx/model_q4f16.onnx_data",
    sha256: "34537bf4a6d70ddf1627bd3709d31c8b7db8d5bcaee2098c45661be59476fbec",
    sizeBytes: 1_063_972_864,
  }),
  Object.freeze({
    path: "onnx/model_q4f16.onnx_data_1",
    sha256: "1b645b44902caccd406098c4dbef5724927c5fb2a2be4a087ae328989b111a7f",
    sizeBytes: 469_958_656,
  }),
]);
const FORBIDDEN_REASONING_MARKERS = Object.freeze(["<think>", "</think>"]);

export async function createOpenMedLfm25ChatRuntime(options = {}) {
  const {
    cache = false,
    contextTokens = 4096,
    device,
    modelUrl,
    networkPolicy,
    onProgress,
    signal,
    transformers: providedTransformers,
  } = options;
  if (networkPolicy !== NETWORK_POLICY) {
    throw new Error(`networkPolicy must be ${NETWORK_POLICY}`);
  }
  if (device !== "webgpu") {
    throw new Error("The LFM2.5 chat adapter requires device=webgpu");
  }
  throwIfAborted(signal);
  await requireFloat16WebGpu(signal);

  const pageUrl = requirePageUrl();
  const modelBaseUrl = resolveSameOriginHttpUrl(modelUrl, "modelUrl", {
    baseUrl: pageUrl,
    directory: true,
  });
  const configUrl = resolveModelAssetUrl(modelBaseUrl, "config.json");
  const transformersUrl = resolveSameOriginHttpUrl(
    new URL(TRANSFORMERS_MODULE_PATH, ADAPTER_URL),
    "Transformers.js module",
    { baseUrl: pageUrl },
  );
  if (providedTransformers == null) {
    return createWorkerRuntime({
      cache,
      contextTokens,
      device,
      modelBaseUrl,
      networkPolicy,
      onProgress,
      pageUrl,
      signal,
    });
  }

  emitProgress(onProgress, {
    detail: "config.json",
    loaded: 0,
    phase: "Validating local LFM2.5 bundle",
    total: MODEL_BYTES,
  });
  const config = validateLfm25Config(
    await loadJson(configUrl, { cache, signal }),
  );
  throwIfAborted(signal);

  const transformers =
    providedTransformers ?? (await importSameOriginModule(transformersUrl, signal));
  configureTransformers(transformers, cache);
  validateTransformers(transformers);

  const generator = await transformers.pipeline(
    "text-generation",
    modelBaseUrl.pathname,
    {
      device: "webgpu",
      dtype: MODEL_VARIANT,
      local_files_only: true,
      progress_callback: (event) => reportPipelineProgress(onProgress, event),
    },
  );
  throwIfAborted(signal);
  if (!generator?.tokenizer || typeof generator !== "function") {
    await releaseResource(generator);
    throw new Error("The local LFM2.5 pipeline did not expose a tokenizer");
  }

  emitProgress(onProgress, {
    detail: "Q4F16 model and tokenizer ready",
    loaded: MODEL_BYTES,
    phase: "LFM2.5 ready",
    total: MODEL_BYTES,
  });
  return new OpenMedLfm25ChatRuntime({
    InterruptableStoppingCriteria: transformers.InterruptableStoppingCriteria,
    TextStreamer: transformers.TextStreamer,
    config,
    contextTokens,
    generator,
    modelBaseUrl,
  });
}

export class OpenMedLfm25ChatRuntime {
  constructor({
    InterruptableStoppingCriteria,
    TextStreamer,
    config,
    contextTokens,
    generator,
    modelBaseUrl,
  }) {
    this.InterruptableStoppingCriteria = InterruptableStoppingCriteria;
    this.TextStreamer = TextStreamer;
    this.config = config;
    this.contextTokens = positiveInteger(contextTokens, "contextTokens");
    this.generator = generator;
    this.modelBaseUrl = modelBaseUrl;
    this.activeGeneration = null;
    this.disposed = false;
  }

  async *generate(messages, generation = {}) {
    if (this.disposed) throw new Error("The LFM2.5 chat runtime is disposed");
    if (this.activeGeneration) {
      throw new Error("Only one LFM2.5 generation may run at a time");
    }
    if (!Array.isArray(messages) || messages.length === 0) {
      throw new TypeError("messages must be a non-empty array");
    }
    if (generation.reasoning !== false) {
      throw new Error("LFM2.5 chat requires direct-generation mode");
    }

    const maxNewTokens = positiveInteger(
      generation.maxNewTokens,
      "maxNewTokens",
    );
    const temperature = finiteNumber(generation.temperature ?? 0, "temperature");
    if (temperature < 0) throw new RangeError("temperature must be non-negative");
    throwIfAborted(generation.signal);

    const rendered = this.generator.tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
      return_dict: false,
      return_tensor: false,
      tokenize: false,
    });
    const prompt = closeTrailingReasoningPrompt(rendered);
    const promptTokenCount = countPromptTokens(this.generator.tokenizer, prompt);
    const availableTokens = Math.max(0, this.contextTokens - promptTokenCount);
    if (availableTokens === 0) {
      throw new Error("The LFM2.5 conversation exceeds the 4,096-token browser cap");
    }
    const queue = new AsyncTextQueue();
    const outputGate = new DirectOutputGate();
    const stoppingCriteria = new this.InterruptableStoppingCriteria();
    const abort = () => stoppingCriteria.interrupt();
    generation.signal?.addEventListener("abort", abort, { once: true });
    let streamedTokenCount = 0;
    const streamer = new this.TextStreamer(this.generator.tokenizer, {
      callback_function: (text) => queue.push(text),
      skip_prompt: true,
      skip_special_tokens: true,
      token_callback_function: (tokens) => {
        streamedTokenCount += Number(tokens?.length ?? 0);
      },
    });
    const pipelineOptions = {
      add_special_tokens: false,
      do_sample: temperature > 0,
      max_new_tokens: Math.min(maxNewTokens, availableTokens),
      repetition_penalty: 1.05,
      return_full_text: false,
      stopping_criteria: stoppingCriteria,
      streamer,
    };
    if (temperature > 0) pipelineOptions.temperature = temperature;

    const work = Promise.resolve()
      .then(() => this.generator(prompt, pipelineOptions))
      .then(
        () => queue.end(),
        (error) => queue.end(error),
      );
    this.activeGeneration = { stoppingCriteria, work };
    let tokenCount = 0;
    try {
      for await (const text of queue) {
        throwIfAborted(generation.signal);
        const delta = outputGate.push(text);
        if (!delta) continue;
        tokenCount = Math.max(tokenCount + 1, streamedTokenCount);
        yield { delta, index: tokenCount - 1, tokenCount };
      }
      await work;
      throwIfAborted(generation.signal);
      const tail = outputGate.finish();
      if (tail) {
        tokenCount += 1;
        yield { delta: tail, index: tokenCount - 1, tokenCount };
      }
    } catch (error) {
      stoppingCriteria.interrupt();
      await work.catch(() => {});
      if (generation.signal?.aborted) {
        throw abortError(generation.signal.reason);
      }
      throw error;
    } finally {
      generation.signal?.removeEventListener("abort", abort);
      if (this.activeGeneration?.work === work) this.activeGeneration = null;
    }
  }

  details() {
    return {
      Architecture: "Lfm2ForCausalLM",
      Model: SOURCE_MODEL,
      Precision: "Q4F16 · 1.53 GB",
      Privacy: "Same-origin assets · no remote fallback",
      Revision: SOURCE_REVISION,
      Runtime: "Transformers.js · ONNX Runtime WebGPU",
    };
  }

  async dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.activeGeneration?.stoppingCriteria.interrupt();
    await this.activeGeneration?.work.catch(() => {});
    this.activeGeneration = null;
    await releaseResource(this.generator);
  }
}

function countPromptTokens(tokenizer, prompt) {
  if (typeof tokenizer !== "function") return 1;
  const encoded = tokenizer(prompt, {
    add_special_tokens: false,
    return_tensor: false,
  });
  const inputIds = encoded?.input_ids;
  const firstSequence = Array.isArray(inputIds?.[0]) ? inputIds[0] : inputIds;
  return Number.isSafeInteger(firstSequence?.length) ? firstSequence.length : 1;
}

export class OpenMedLfm25WorkerRuntime {
  constructor({ details, worker }) {
    this.runtimeDetails = details;
    this.worker = worker;
    this.activeGeneration = null;
    this.disposeResolver = null;
    this.disposed = false;
    this.sequence = 0;
    worker.addEventListener("message", (event) => this.handleMessage(event.data));
    worker.addEventListener("error", (event) => {
      this.activeGeneration?.queue.end(
        new Error(event.message || "The LFM2.5 worker stopped unexpectedly"),
      );
      this.activeGeneration = null;
    });
  }

  async *generate(messages, generation = {}) {
    if (this.disposed) throw new Error("The LFM2.5 chat runtime is disposed");
    if (this.activeGeneration) {
      throw new Error("Only one LFM2.5 generation may run at a time");
    }
    if (!Array.isArray(messages) || messages.length === 0) {
      throw new TypeError("messages must be a non-empty array");
    }
    throwIfAborted(generation.signal);

    const requestId = `generation-${++this.sequence}`;
    const queue = new AsyncValueQueue();
    const cancel = () => this.worker.postMessage({ requestId, type: "cancel" });
    generation.signal?.addEventListener("abort", cancel, { once: true });
    const { signal: _signal, ...serializableGeneration } = generation;
    this.activeGeneration = { queue, requestId };
    this.worker.postMessage({
      generation: serializableGeneration,
      messages,
      requestId,
      type: "generate",
    });
    try {
      for await (const value of queue) {
        throwIfAborted(generation.signal);
        yield value;
      }
      throwIfAborted(generation.signal);
    } catch (error) {
      if (generation.signal?.aborted) throw abortError(generation.signal.reason);
      throw error;
    } finally {
      generation.signal?.removeEventListener("abort", cancel);
      if (this.activeGeneration?.requestId === requestId) this.activeGeneration = null;
    }
  }

  handleMessage(message = {}) {
    if (message.type === "disposed") {
      this.disposeResolver?.();
      return;
    }
    if (message.requestId !== this.activeGeneration?.requestId) return;
    if (message.type === "delta") this.activeGeneration.queue.push(message.value);
    if (message.type === "complete") this.activeGeneration.queue.end();
    if (message.type === "error") {
      const error =
        message.error?.name === "AbortError"
          ? abortError(message.error.message)
          : new Error(message.error?.message || "LFM2.5 generation failed");
      this.activeGeneration.queue.end(error);
    }
  }

  details() {
    return this.runtimeDetails;
  }

  async dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.worker.postMessage({ type: "cancel" });
    const requestId = `dispose-${++this.sequence}`;
    await new Promise((resolve) => {
      const timeout = setTimeout(resolve, 5000);
      this.disposeResolver = () => {
        clearTimeout(timeout);
        resolve();
      };
      this.worker.postMessage({ requestId, type: "dispose" });
    });
    this.disposeResolver = null;
    this.activeGeneration?.queue.end(abortError("Runtime released"));
    this.activeGeneration = null;
    this.worker.terminate();
  }
}

export function validateLfm25Config(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("LFM2.5 config.json must contain an object");
  }
  const architectures = Array.isArray(value.architectures)
    ? value.architectures
    : [];
  if (value.model_type !== "lfm2" || !architectures.includes("Lfm2ForCausalLM")) {
    throw new Error("The local model is not an Lfm2ForCausalLM bundle");
  }
  const expected = {
    eos_token_id: 124_900,
    hidden_size: 2048,
    num_attention_heads: 32,
    num_hidden_layers: 30,
    num_key_value_heads: 8,
    vocab_size: 128_000,
  };
  for (const [key, required] of Object.entries(expected)) {
    if (value[key] !== required) {
      throw new Error(`LFM2.5 config changed for ${key}: ${value[key]}`);
    }
  }
  return value;
}

export function closeTrailingReasoningPrompt(rendered) {
  if (typeof rendered !== "string") {
    throw new TypeError("LFM2.5 tokenizer returned an unsupported prompt");
  }
  const suffix = "<think>";
  if (!rendered.endsWith(suffix)) {
    throw new Error("LFM2.5 chat template is missing its reasoning prompt suffix");
  }
  return `${rendered}</think>\n`;
}

export function resolveSameOriginHttpUrl(value, label, options = {}) {
  const baseUrl = options.baseUrl ?? requirePageUrl();
  const url = new URL(value, baseUrl);
  if (url.origin !== baseUrl.origin) {
    throw new Error(`${label} must use this page's origin`);
  }
  if (!["http:", "https:"].includes(url.protocol)) {
    throw new Error(`${label} must use HTTP or HTTPS`);
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new Error(`${label} must not contain credentials, query data, or a fragment`);
  }
  if (options.directory && !url.pathname.endsWith("/")) {
    throw new Error(`${label} must end with a slash`);
  }
  return url;
}

export function resolveModelAssetUrl(modelBaseUrl, relativePath) {
  if (
    typeof relativePath !== "string" ||
    relativePath.startsWith("/") ||
    relativePath.includes("\\") ||
    relativePath.split("/").some((part) => !part || part === "." || part === "..")
  ) {
    throw new Error("model asset path must be a contained POSIX path");
  }
  const url = new URL(relativePath, modelBaseUrl);
  if (!url.pathname.startsWith(modelBaseUrl.pathname)) {
    throw new Error("model asset escaped the local model directory");
  }
  return url;
}

class DirectOutputGate {
  constructor() {
    this.pending = "";
  }

  push(value) {
    if (typeof value !== "string" || !value) return "";
    this.pending += value;
    if (FORBIDDEN_REASONING_MARKERS.some((marker) => this.pending.includes(marker))) {
      throw new Error("LFM2.5 attempted to emit hidden reasoning");
    }
    const held = longestMarkerPrefixSuffix(this.pending);
    const visibleLength = this.pending.length - held;
    const visible = this.pending.slice(0, visibleLength);
    this.pending = this.pending.slice(visibleLength);
    return visible;
  }

  finish() {
    if (this.pending) {
      throw new Error("LFM2.5 ended with a partial hidden-reasoning marker");
    }
    return "";
  }
}

class AsyncTextQueue {
  constructor() {
    this.values = [];
    this.waiters = [];
    this.closed = false;
    this.error = null;
  }

  push(value) {
    if (this.closed || typeof value !== "string" || !value) return;
    const waiter = this.waiters.shift();
    if (waiter) waiter.resolve({ done: false, value });
    else this.values.push(value);
  }

  end(error = null) {
    if (this.closed) return;
    this.closed = true;
    this.error = error;
    for (const waiter of this.waiters.splice(0)) {
      if (error) waiter.reject(error);
      else waiter.resolve({ done: true, value: undefined });
    }
  }

  next() {
    if (this.values.length > 0) {
      return Promise.resolve({ done: false, value: this.values.shift() });
    }
    if (this.error) return Promise.reject(this.error);
    if (this.closed) return Promise.resolve({ done: true, value: undefined });
    return new Promise((resolve, reject) => this.waiters.push({ reject, resolve }));
  }

  [Symbol.asyncIterator]() {
    return this;
  }
}

class AsyncValueQueue {
  constructor() {
    this.values = [];
    this.waiters = [];
    this.closed = false;
    this.error = null;
  }

  push(value) {
    if (this.closed) return;
    const waiter = this.waiters.shift();
    if (waiter) waiter.resolve({ done: false, value });
    else this.values.push(value);
  }

  end(error = null) {
    if (this.closed) return;
    this.closed = true;
    this.error = error;
    for (const waiter of this.waiters.splice(0)) {
      if (error) waiter.reject(error);
      else waiter.resolve({ done: true, value: undefined });
    }
  }

  next() {
    if (this.values.length > 0) {
      return Promise.resolve({ done: false, value: this.values.shift() });
    }
    if (this.error) return Promise.reject(this.error);
    if (this.closed) return Promise.resolve({ done: true, value: undefined });
    return new Promise((resolve, reject) => this.waiters.push({ reject, resolve }));
  }

  [Symbol.asyncIterator]() {
    return this;
  }
}

function longestMarkerPrefixSuffix(value) {
  let longest = 0;
  for (const marker of FORBIDDEN_REASONING_MARKERS) {
    for (let length = 1; length < marker.length; length += 1) {
      if (value.endsWith(marker.slice(0, length))) longest = Math.max(longest, length);
    }
  }
  return longest;
}

function validateTransformers(transformers) {
  for (const name of ["pipeline", "TextStreamer", "InterruptableStoppingCriteria"]) {
    if (typeof transformers?.[name] !== "function") {
      throw new Error(`The local Transformers.js module must export ${name}`);
    }
  }
}

async function requireFloat16WebGpu(signal) {
  const gpu = globalThis.navigator?.gpu;
  if (!gpu || typeof gpu.requestAdapter !== "function") {
    throw new Error("LFM2.5 Q4F16 requires WebGPU in a secure browser context");
  }
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  throwIfAborted(signal);
  if (!adapter) throw new Error("No WebGPU adapter is available for LFM2.5");
  if (!adapter.features?.has?.("shader-f16")) {
    throw new Error("LFM2.5 Q4F16 requires the WebGPU shader-f16 feature");
  }
}

async function createWorkerRuntime({
  cache,
  contextTokens,
  device,
  modelBaseUrl,
  networkPolicy,
  onProgress,
  pageUrl,
  signal,
}) {
  if (typeof Worker !== "function") {
    throw new Error("This browser does not support module workers");
  }
  const workerUrl = resolveSameOriginHttpUrl(
    new URL(WORKER_PATH, ADAPTER_URL),
    "LFM2.5 worker",
    { baseUrl: pageUrl },
  );
  const worker = new Worker(workerUrl.href, {
    name: "openmed-lfm25-chat",
    type: "module",
  });
  const requestId = "load-1";
  return new Promise((resolve, reject) => {
    const abort = () => {
      worker.terminate();
      reject(abortError(signal?.reason));
    };
    const fail = (error) => {
      signal?.removeEventListener("abort", abort);
      worker.terminate();
      reject(error);
    };
    const onError = (event) => {
      fail(new Error(event.message || "Unable to start the LFM2.5 worker"));
    };
    const onMessage = (event) => {
      const message = event.data ?? {};
      if (message.requestId !== requestId) return;
      if (message.type === "progress") {
        emitProgress(onProgress, message.progress);
        return;
      }
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
      signal?.removeEventListener("abort", abort);
      if (message.type === "loaded") {
        resolve(
          new OpenMedLfm25WorkerRuntime({
            details: message.details,
            worker,
          }),
        );
      } else {
        fail(new Error(message.error?.message || "Unable to load LFM2.5"));
      }
    };
    signal?.addEventListener("abort", abort, { once: true });
    worker.addEventListener("error", onError, { once: true });
    worker.addEventListener("message", onMessage);
    worker.postMessage({
      options: {
        cache,
        contextTokens,
        device,
        modelUrl: modelBaseUrl.href,
        networkPolicy,
      },
      requestId,
      type: "load",
    });
  });
}

function configureTransformers(transformers, cache) {
  const env = transformers?.env;
  if (!env || typeof env !== "object") {
    throw new Error("The local Transformers.js module must export env");
  }
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = "/";
  env.useBrowserCache = Boolean(cache);
  env.useFSCache = false;
  env.useCustomCache = false;
  if ("logLevel" in env) env.logLevel = 5;
}

async function loadJson(url, { cache, signal }) {
  const response = await fetch(url.href, {
    cache: cache ? "default" : "no-store",
    credentials: "omit",
    method: "GET",
    redirect: "error",
    referrerPolicy: "no-referrer",
    signal,
  });
  if (!response.ok) throw new Error(`Unable to load local LFM2.5 config: HTTP ${response.status}`);
  return response.json();
}

async function importSameOriginModule(url, signal) {
  throwIfAborted(signal);
  const module = await import(url.href);
  throwIfAborted(signal);
  return module;
}

function reportPipelineProgress(onProgress, event = {}) {
  const loaded = Math.max(0, Number(event.loaded ?? 0));
  const total = Math.max(0, Number(event.total ?? MODEL_BYTES));
  const file = typeof event.file === "string" ? event.file.split("/").at(-1) : "";
  emitProgress(onProgress, {
    detail: file || "Local Q4F16 assets",
    loaded,
    phase: event.status === "progress" ? "Loading LFM2.5 Q4F16" : "Preparing LFM2.5",
    progress: Number.isFinite(Number(event.progress))
      ? Number(event.progress) / 100
      : undefined,
    total,
  });
}

function emitProgress(callback, value) {
  if (typeof callback === "function") callback(value);
}

function requirePageUrl() {
  const href = globalThis.location?.href;
  if (!href) throw new Error("A browser page URL is required");
  return new URL(href);
}

function positiveInteger(value, label) {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number <= 0) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return number;
}

function finiteNumber(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new TypeError(`${label} must be finite`);
  return number;
}

function throwIfAborted(signal) {
  if (signal?.aborted) throw abortError(signal.reason);
}

function abortError(reason) {
  return new DOMException(String(reason ?? "Operation aborted"), "AbortError");
}

async function releaseResource(resource) {
  if (!resource) return;
  const release = resource.dispose ?? resource.destroy ?? resource.release;
  if (typeof release === "function") await release.call(resource);
}

export {
  MODEL_BYTES,
  MODEL_VARIANT,
  REQUIRED_GRAPH_FILES,
  SOURCE_MODEL,
  SOURCE_REVISION,
};
