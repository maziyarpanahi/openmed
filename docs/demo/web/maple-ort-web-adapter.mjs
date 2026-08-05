/*
 * Reference Maple adapter for a locally supplied ONNX Runtime Web build.
 *
 * This file intentionally contains no runtime binary, tokenizer implementation,
 * model graph, weights, CDN URL, or remote fallback. See README.md for the exact
 * same-origin files that an operator must supply before using it.
 */

const ADAPTER_URL = import.meta.url;
const BUNDLE_MANIFEST = "maple-bundle.json";
const NETWORK_POLICY = "same-origin-model-assets-only";
const ORT_MODULE_PATH = "./vendor/ort.webgpu.min.mjs";
const TOKENIZER_MODULE_PATH = "./vendor/maple-tokenizer.mjs";
const VOCAB_SIZE = 151_936;
const LAYER_COUNT = 24;
const KV_HEADS = 4;
const HEAD_SIZE = 128;
const SOURCE_MODEL = "deepgrove/maple-preview";
const ARCHITECTURE = "MapleForCausalLM";
const QUANTIZATION = "qmoe-4bit-blockwise-128";
const SHA256_PATTERN = /^[0-9a-f]{64}$/u;
const REVISION_PATTERN = /^[0-9a-f]{40,64}$/u;

export async function createOpenMedMapleRuntime(options = {}) {
  const {
    cache = false,
    contextTokens,
    device,
    modelUrl,
    networkPolicy,
    onProgress,
    signal,
  } = options;
  if (networkPolicy !== NETWORK_POLICY) {
    throw new Error(`networkPolicy must be ${NETWORK_POLICY}`);
  }
  if (device !== "webgpu") {
    throw new Error("The reference Maple ONNX adapter requires device=webgpu");
  }
  throwIfAborted(signal);

  const pageUrl = requirePageUrl();
  const modelBaseUrl = resolveSameOriginHttpUrl(modelUrl, "modelUrl", {
    baseUrl: pageUrl,
    directory: true,
  });
  const manifestUrl = resolveBundleAssetUrl(
    modelBaseUrl,
    BUNDLE_MANIFEST,
    "bundle manifest",
  );
  const ortModuleUrl = resolveSameOriginHttpUrl(
    new URL(ORT_MODULE_PATH, ADAPTER_URL),
    "ONNX Runtime Web module",
    { baseUrl: pageUrl },
  );
  const tokenizerModuleUrl = resolveSameOriginHttpUrl(
    new URL(TOKENIZER_MODULE_PATH, ADAPTER_URL),
    "Maple tokenizer module",
    { baseUrl: pageUrl },
  );
  const vendorBaseUrl = resolveSameOriginHttpUrl(
    new URL("./vendor/", ADAPTER_URL),
    "runtime asset directory",
    { baseUrl: pageUrl, directory: true },
  );

  emitProgress(onProgress, {
    phase: "Validating local Maple bundle",
    loaded: 0,
    total: 0,
    detail: BUNDLE_MANIFEST,
  });

  let session;
  let tokenizer;
  try {
    const [ortModule, tokenizerModule, manifest] = await Promise.all([
      importSameOriginModule(ortModuleUrl, signal),
      importSameOriginModule(tokenizerModuleUrl, signal),
      loadBundleManifest(manifestUrl, { cache, signal }),
    ]);
    throwIfAborted(signal);

    const contract = validateOpenMedMapleBundleManifest(manifest, modelBaseUrl);
    const ort = normalizeOrtModule(ortModule);
    configureOrtEnvironment(ort, vendorBaseUrl);
    const createTokenizer = tokenizerModule.createOpenMedMapleTokenizer;
    if (typeof createTokenizer !== "function") {
      throw new Error(
        "The local tokenizer module must export " +
          "createOpenMedMapleTokenizer(options)",
      );
    }
    tokenizer = await createTokenizer({
      modelUrl: modelBaseUrl.href,
      networkPolicy: NETWORK_POLICY,
      signal,
      tokenizerUrl: contract.tokenizerUrl.href,
    });
    validateTokenizer(tokenizer);
    throwIfAborted(signal);

    emitProgress(onProgress, {
      phase: "Creating local ONNX Runtime Web session",
      loaded: 0,
      total: contract.totalBytes,
      detail: contract.graphPath,
    });
    session = await ort.InferenceSession.create(
      contract.graphUrl.href,
      sessionOptions(contract),
    );
    throwIfAborted(signal);
    validateSessionContract(session, contract);

    emitProgress(onProgress, {
      phase: "Maple ONNX session ready",
      loaded: contract.totalBytes,
      total: contract.totalBytes,
      detail: contract.graphPath,
    });
    return new OpenMedMapleOrtWebRuntime({
      cache,
      contextTokens,
      contract,
      ort,
      session,
      tokenizer,
    });
  } catch (error) {
    await releaseResource(session);
    await releaseResource(tokenizer);
    if (signal?.aborted) throw abortError(signal.reason);
    throw error;
  }
}

export class OpenMedMapleOrtWebRuntime {
  constructor({ cache, contextTokens, contract, ort, session, tokenizer }) {
    this.cacheRequested = Boolean(cache);
    this.contract = contract;
    this.contextTokens = positiveInteger(contextTokens, "contextTokens");
    this.effectiveContextTokens = Math.min(
      this.contextTokens,
      contract.maxContextTokens,
    );
    this.ort = ort;
    this.session = session;
    this.tokenizer = tokenizer;
    this.activeGeneration = null;
    this.disposed = false;
    validateSessionContract(session, contract);
  }

  async *generate(messages, generation = {}) {
    if (this.disposed) throw new Error("The Maple ONNX runtime is disposed");
    if (this.activeGeneration) {
      throw new Error("Only one Maple generation may run at a time");
    }
    if (!Array.isArray(messages) || messages.length === 0) {
      throw new TypeError("messages must be a non-empty array");
    }

    const requestedTokens = positiveInteger(
      generation.maxNewTokens,
      "maxNewTokens",
    );
    const temperature = finiteNumber(generation.temperature ?? 0, "temperature");
    const minP = finiteNumber(generation.minP ?? 0, "minP");
    if (temperature < 0) throw new RangeError("temperature must be non-negative");
    if (minP < 0 || minP > 1) throw new RangeError("minP must be between 0 and 1");

    const controller = new AbortController();
    const unlinkCaller = forwardAbort(generation.signal, controller);
    this.activeGeneration = controller;
    let caches = null;
    try {
      throwIfAborted(controller.signal);
      const encoded = await this.tokenizer.encodeMessages(messages, {
        addGenerationPrompt: true,
        signal: controller.signal,
      });
      const promptIds = normalizeTokenIds(encoded, "encoded prompt");
      if (promptIds.length === 0) {
        throw new Error("The Maple tokenizer returned an empty prompt");
      }
      if (promptIds.length > this.contract.maxInputTokens) {
        throw new RangeError(
          `The prompt has ${promptIds.length} tokens; this bundle allows ` +
            `${this.contract.maxInputTokens}`,
        );
      }
      if (promptIds.length >= this.effectiveContextTokens) {
        throw new RangeError("The prompt leaves no room for generated tokens");
      }

      const maxNewTokens = Math.min(
        requestedTokens,
        this.effectiveContextTokens - promptIds.length,
      );
      caches = createEmptyCaches(this.ort, this.contract);
      let currentIds = promptIds;
      let pastLength = 0;
      const generatedIds = [];
      let visibleText = "";

      for (let index = 0; index < maxNewTokens; index += 1) {
        throwIfAborted(controller.signal);
        const step = await runDecoderStep({
          caches,
          contract: this.contract,
          inputIds: currentIds,
          ort: this.ort,
          pastLength,
          session: this.session,
          signal: controller.signal,
        });
        caches = step.caches;
        pastLength += currentIds.length;

        let tokenId;
        try {
          tokenId = await sampleLogitsTensor(step.logits, {
            minP,
            temperature,
          });
        } finally {
          await releaseTensor(step.logits);
        }
        if (this.contract.eosTokenIds.has(tokenId)) break;

        generatedIds.push(tokenId);
        currentIds = [tokenId];
        const decoded = await this.tokenizer.decode(generatedIds, {
          skipSpecialTokens: true,
        });
        if (typeof decoded !== "string") {
          throw new TypeError("The Maple tokenizer decode() result must be a string");
        }
        // Avoid presenting an incomplete UTF-8 byte-token replacement. The next
        // token normally resolves it into a prefix-stable cumulative string.
        if (decoded.endsWith("\ufffd")) continue;
        visibleText = decoded;
        yield {
          index,
          text: visibleText,
          token: tokenId,
          tokenCount: generatedIds.length,
        };
      }
    } catch (error) {
      if (controller.signal.aborted) throw abortError(controller.signal.reason);
      throw error;
    } finally {
      unlinkCaller();
      await releaseCacheMap(caches);
      if (this.activeGeneration === controller) this.activeGeneration = null;
    }
  }

  details() {
    return {
      Runtime: "ONNX Runtime Web reference adapter",
      Validation: "Mock-tested plumbing; real Maple WebGPU inference unvalidated",
      Device: "WebGPU only",
      Graph: "Unified prefill/decode with GPU-resident KV cache",
      Quantization: QUANTIZATION,
      Context: `${this.effectiveContextTokens} tokens`,
      "Persistent cache": this.cacheRequested
        ? "Browser/ORT managed; not cleared by this adapter"
        : "Not requested; ORT behavior is implementation-defined",
    };
  }

  async dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.activeGeneration?.abort("runtime disposed");
    const session = this.session;
    const tokenizer = this.tokenizer;
    this.session = null;
    this.tokenizer = null;
    await releaseResource(session);
    await releaseResource(tokenizer);
  }
}

export function resolveSameOriginHttpUrl(
  value,
  label,
  { baseUrl = requirePageUrl(), directory = false } = {},
) {
  let resolved;
  try {
    resolved = new URL(value, baseUrl);
  } catch {
    throw new Error(`${label} must be a valid URL`);
  }
  const expectedOrigin = new URL(baseUrl).origin;
  if (resolved.origin !== expectedOrigin) {
    throw new Error(`${label} must use exactly this page's origin`);
  }
  if (!["http:", "https:"].includes(resolved.protocol)) {
    throw new Error(`${label} must use HTTP or HTTPS`);
  }
  if (resolved.username || resolved.password || resolved.search || resolved.hash) {
    throw new Error(
      `${label} must not contain credentials, query data, or a fragment`,
    );
  }
  if (directory && !resolved.pathname.endsWith("/")) {
    throw new Error(`${label} must end with a slash`);
  }
  return resolved;
}

export function resolveBundleAssetUrl(modelBaseUrl, path, label = "bundle asset") {
  const normalized = validateRelativeBundlePath(path, label);
  const resolved = resolveSameOriginHttpUrl(normalized, label, {
    baseUrl: modelBaseUrl,
  });
  if (!resolved.pathname.startsWith(modelBaseUrl.pathname)) {
    throw new Error(`${label} escapes modelUrl`);
  }
  return resolved;
}

export function validateOpenMedMapleBundleManifest(manifest, modelBaseUrl) {
  requireExactKeys(
    manifest,
    [
      "schema_version",
      "source_model",
      "source_revision",
      "architecture",
      "quantization",
      "runtime",
      "tokenizer_path",
      "graphs",
      "cache",
      "generation",
      "files",
    ],
    "bundle manifest",
  );
  if (manifest.schema_version !== 1) {
    throw new Error("Unsupported Maple bundle schema_version");
  }
  if (manifest.source_model !== SOURCE_MODEL) {
    throw new Error(`source_model must be ${SOURCE_MODEL}`);
  }
  if (!REVISION_PATTERN.test(manifest.source_revision)) {
    throw new Error("source_revision must be an immutable lowercase commit SHA");
  }
  if (manifest.architecture !== ARCHITECTURE) {
    throw new Error(`architecture must be ${ARCHITECTURE}`);
  }
  if (manifest.quantization !== QUANTIZATION) {
    throw new Error(
      `The reference WebGPU adapter requires ${QUANTIZATION}; ` +
        "it does not run the 2-bit MLX checkpoint",
    );
  }
  if (manifest.runtime !== "onnxruntime-web") {
    throw new Error("runtime must be onnxruntime-web");
  }

  requireExactKeys(
    manifest.graphs,
    [
      "prefill_path",
      "decode_path",
      "input_ids_name",
      "attention_mask_name",
      "position_ids_name",
      "logits_name",
    ],
    "graphs",
  );
  if (manifest.graphs.prefill_path !== manifest.graphs.decode_path) {
    throw new Error("Maple WebGPU requires one unified prefill/decode graph");
  }
  const expectedGraphNames = {
    attention_mask_name: "attention_mask",
    input_ids_name: "input_ids",
    logits_name: "logits",
    position_ids_name: "position_ids",
  };
  for (const [key, expected] of Object.entries(expectedGraphNames)) {
    if (manifest.graphs[key] !== expected) {
      throw new Error(`graphs.${key} must be ${expected}`);
    }
  }

  requireExactKeys(
    manifest.cache,
    ["past_input_prefix", "present_output_prefix"],
    "cache",
  );
  if (manifest.cache.past_input_prefix !== "past_key_values.") {
    throw new Error("cache.past_input_prefix must be past_key_values.");
  }
  if (manifest.cache.present_output_prefix !== "present.") {
    throw new Error("cache.present_output_prefix must be present.");
  }

  requireExactKeys(
    manifest.generation,
    ["eos_token_ids", "max_context_tokens", "max_input_tokens"],
    "generation",
  );
  const eosTokenIds = normalizeTokenIds(
    manifest.generation.eos_token_ids,
    "generation.eos_token_ids",
  );
  if (eosTokenIds.length === 0) {
    throw new Error("generation.eos_token_ids must not be empty");
  }
  const maxContextTokens = positiveInteger(
    manifest.generation.max_context_tokens,
    "generation.max_context_tokens",
  );
  const maxInputTokens = positiveInteger(
    manifest.generation.max_input_tokens,
    "generation.max_input_tokens",
  );
  if (maxInputTokens >= maxContextTokens) {
    throw new Error("max_input_tokens must leave generation context");
  }

  if (!Array.isArray(manifest.files) || manifest.files.length === 0) {
    throw new Error("files must be a non-empty array");
  }
  const paths = new Map();
  let totalBytes = 0;
  for (const [index, file] of manifest.files.entries()) {
    requireExactKeys(file, ["path", "size_bytes", "sha256"], `files[${index}]`);
    const path = validateRelativeBundlePath(file.path, `files[${index}].path`);
    const size = positiveInteger(file.size_bytes, `files[${index}].size_bytes`);
    if (!SHA256_PATTERN.test(file.sha256)) {
      throw new Error(`files[${index}].sha256 must be lowercase SHA-256`);
    }
    if (paths.has(path)) throw new Error(`Duplicate bundle file: ${path}`);
    paths.set(path, file);
    totalBytes += size;
  }

  const graphPath = validateRelativeBundlePath(
    manifest.graphs.prefill_path,
    "graphs.prefill_path",
  );
  const tokenizerPath = validateRelativeBundlePath(
    manifest.tokenizer_path,
    "tokenizer_path",
  );
  if (!paths.has(graphPath)) throw new Error("The graph is not declared in files");
  if (!paths.has(tokenizerPath)) {
    throw new Error("The tokenizer is not declared in files");
  }
  const graphUrl = resolveBundleAssetUrl(modelBaseUrl, graphPath, "Maple graph");
  const tokenizerUrl = resolveBundleAssetUrl(
    modelBaseUrl,
    tokenizerPath,
    "Maple tokenizer data",
  );
  const externalData = [...paths]
    .map(([path]) => path)
    .filter((path) => isGraphExternalData(path, graphPath))
    .map((path) => ({
      data: resolveBundleAssetUrl(modelBaseUrl, path, "ONNX external data").href,
      path,
    }));
  if (externalData.length === 0) {
    throw new Error("The Maple graph must declare external ONNX weight data");
  }

  const pastNames = [];
  const presentNames = [];
  for (let layer = 0; layer < LAYER_COUNT; layer += 1) {
    for (const kind of ["key", "value"]) {
      pastNames.push(`${manifest.cache.past_input_prefix}${layer}.${kind}`);
      presentNames.push(`${manifest.cache.present_output_prefix}${layer}.${kind}`);
    }
  }
  return Object.freeze({
    eosTokenIds: new Set(eosTokenIds),
    externalData,
    graphPath,
    graphUrl,
    logitsName: manifest.graphs.logits_name,
    maxContextTokens,
    maxInputTokens,
    pastNames: Object.freeze(pastNames),
    positionIdsName: manifest.graphs.position_ids_name,
    presentNames: Object.freeze(presentNames),
    tokenizerUrl,
    totalBytes,
  });
}

export function sampleTokenId(
  data,
  {
    offset = 0,
    size = VOCAB_SIZE,
    temperature = 0,
    minP = 0,
    random = Math.random,
  } = {},
) {
  if (!Number.isInteger(offset) || offset < 0 || !Number.isInteger(size) || size < 1) {
    throw new RangeError("Invalid logits slice");
  }
  if (offset + size > data.length) throw new RangeError("Logits data is truncated");
  if (!Number.isFinite(temperature) || temperature < 0) {
    throw new RangeError("temperature must be non-negative");
  }
  if (!Number.isFinite(minP) || minP < 0 || minP > 1) {
    throw new RangeError("minP must be between 0 and 1");
  }

  let bestIndex = 0;
  let bestLogit = -Infinity;
  for (let index = 0; index < size; index += 1) {
    const value = Number(data[offset + index]);
    if (Number.isNaN(value)) continue;
    if (value > bestLogit) {
      bestIndex = index;
      bestLogit = value;
    }
  }
  if (!Number.isFinite(bestLogit)) {
    throw new Error("Maple logits contain no finite value");
  }
  if (temperature === 0) return bestIndex;

  const weights = new Float64Array(size);
  let total = 0;
  for (let index = 0; index < size; index += 1) {
    const logit = Number(data[offset + index]);
    const weight = Number.isFinite(logit)
      ? Math.exp((logit - bestLogit) / temperature)
      : 0;
    if (weight >= minP && weight > 0) {
      weights[index] = weight;
      total += weight;
    }
  }
  if (!(total > 0)) return bestIndex;
  const randomValue = Number(random());
  if (!Number.isFinite(randomValue) || randomValue < 0 || randomValue > 1) {
    throw new Error("The random source must return a number between 0 and 1");
  }
  let target = Math.min(randomValue, 1 - Number.EPSILON) * total;
  for (let index = 0; index < size; index += 1) {
    target -= weights[index];
    if (target < 0) return index;
  }
  return bestIndex;
}

async function runDecoderStep({
  caches,
  contract,
  inputIds,
  ort,
  pastLength,
  session,
  signal,
}) {
  const inputLength = inputIds.length;
  const totalLength = pastLength + inputLength;
  const feeds = {
    input_ids: createInt64Tensor(ort, inputIds, [1, inputLength]),
    attention_mask: createInt64Tensor(
      ort,
      new Array(totalLength).fill(1),
      [1, totalLength],
    ),
    ...caches,
  };
  if (session.inputNames.includes(contract.positionIdsName)) {
    feeds[contract.positionIdsName] = createInt64Tensor(
      ort,
      Array.from({ length: inputLength }, (_unused, index) => pastLength + index),
      [1, inputLength],
    );
  }
  const runOptions = { terminate: false };
  const terminate = () => {
    runOptions.terminate = true;
  };
  signal.addEventListener("abort", terminate, { once: true });
  let outputs;
  try {
    throwIfAborted(signal);
    outputs = await session.run(feeds, session.outputNames, runOptions);
    throwIfAborted(signal);
    validateDecoderOutputs(outputs, contract, inputLength, totalLength);
    const nextCaches = {};
    for (let index = 0; index < contract.pastNames.length; index += 1) {
      nextCaches[contract.pastNames[index]] =
        outputs[contract.presentNames[index]];
    }
    const retained = new Set(Object.values(nextCaches));
    await releaseCacheMap(caches, retained);
    for (const [name, tensor] of Object.entries(outputs)) {
      if (name !== contract.logitsName && !retained.has(tensor)) {
        await releaseTensor(tensor);
      }
    }
    return { caches: nextCaches, logits: outputs[contract.logitsName] };
  } catch (error) {
    if (outputs) await releaseOutputMap(outputs);
    if (signal.aborted) throw abortError(signal.reason);
    throw error;
  } finally {
    signal.removeEventListener("abort", terminate);
    const retainedInputs = new Set(Object.values(caches));
    for (const tensor of Object.values(feeds)) {
      if (!retainedInputs.has(tensor)) await releaseTensor(tensor);
    }
  }
}

async function sampleLogitsTensor(tensor, { minP, temperature }) {
  const dims = Array.from(tensor?.dims ?? []);
  if (
    dims.length !== 3 ||
    dims[0] !== 1 ||
    dims[1] < 1 ||
    dims[2] !== VOCAB_SIZE
  ) {
    throw new Error("logits must have shape [1, sequence, 151936]");
  }
  const raw = await tensorData(tensor);
  const offset = (dims[1] - 1) * VOCAB_SIZE;
  const values =
    tensor.type === "float16"
      ? float16Values(raw.subarray(offset, offset + VOCAB_SIZE))
      : raw;
  return sampleTokenId(values, {
    minP,
    offset: tensor.type === "float16" ? 0 : offset,
    size: VOCAB_SIZE,
    temperature,
  });
}

function validateSessionContract(session, contract) {
  if (!session || typeof session.run !== "function") {
    throw new Error("ONNX Runtime Web did not create an inference session");
  }
  if (!Array.isArray(session.inputNames) || !Array.isArray(session.outputNames)) {
    throw new Error("The ONNX session must expose inputNames and outputNames");
  }
  const requiredInputs = new Set([
    "input_ids",
    "attention_mask",
    ...contract.pastNames,
  ]);
  for (const name of requiredInputs) {
    if (!session.inputNames.includes(name)) {
      throw new Error(`The Maple graph is missing input ${name}`);
    }
  }
  const allowedInputs = new Set([...requiredInputs, contract.positionIdsName]);
  for (const name of session.inputNames) {
    if (!allowedInputs.has(name)) {
      throw new Error(`The Maple graph has unsupported input ${name}`);
    }
  }
  for (const name of [contract.logitsName, ...contract.presentNames]) {
    if (!session.outputNames.includes(name)) {
      throw new Error(`The Maple graph is missing output ${name}`);
    }
  }
}

function validateDecoderOutputs(outputs, contract, inputLength, totalLength) {
  if (!outputs || typeof outputs !== "object") {
    throw new Error("The Maple decoder returned no outputs");
  }
  const logits = outputs[contract.logitsName];
  const logitsDims = Array.from(logits?.dims ?? []);
  if (
    logitsDims.length !== 3 ||
    logitsDims[0] !== 1 ||
    logitsDims[1] !== inputLength ||
    logitsDims[2] !== VOCAB_SIZE
  ) {
    throw new Error("logits must have shape [1, input sequence, 151936]");
  }
  for (const name of contract.presentNames) {
    const dims = Array.from(outputs[name]?.dims ?? []);
    if (
      dims.length !== 4 ||
      dims[0] !== 1 ||
      dims[1] !== KV_HEADS ||
      dims[2] !== totalLength ||
      dims[3] !== HEAD_SIZE
    ) {
      throw new Error(`${name} must have shape [1, 4, ${totalLength}, 128]`);
    }
  }
}

function createEmptyCaches(ort, contract) {
  const caches = {};
  for (const name of contract.pastNames) {
    caches[name] = new ort.Tensor(
      "float16",
      new Uint16Array(0),
      [1, KV_HEADS, 0, HEAD_SIZE],
    );
  }
  return caches;
}

function createInt64Tensor(ort, values, dims) {
  return new ort.Tensor(
    "int64",
    BigInt64Array.from(values, (value) => BigInt(value)),
    dims,
  );
}

function sessionOptions(contract) {
  const preferredOutputLocation = { [contract.logitsName]: "cpu" };
  for (const name of contract.presentNames) {
    preferredOutputLocation[name] = "gpu-buffer";
  }
  return {
    executionProviders: ["webgpu"],
    externalData: contract.externalData,
    preferredOutputLocation,
  };
}

function configureOrtEnvironment(ort, vendorBaseUrl) {
  if (ort.env?.wasm) {
    ort.env.wasm.wasmPaths = vendorBaseUrl.href;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
  }
  if (ort.env) ort.env.logLevel = "fatal";
}

function normalizeOrtModule(module) {
  const candidate = module?.InferenceSession ? module : module?.default;
  if (
    !candidate ||
    typeof candidate.InferenceSession?.create !== "function" ||
    typeof candidate.Tensor !== "function"
  ) {
    throw new Error(
      "The local ONNX Runtime Web module must expose InferenceSession and Tensor",
    );
  }
  return candidate;
}

function validateTokenizer(tokenizer) {
  if (
    !tokenizer ||
    typeof tokenizer.encodeMessages !== "function" ||
    typeof tokenizer.decode !== "function"
  ) {
    throw new Error(
      "createOpenMedMapleTokenizer() must return encodeMessages() and decode()",
    );
  }
}

async function importSameOriginModule(url, signal) {
  throwIfAborted(signal);
  const module = await import(url.href);
  throwIfAborted(signal);
  return module;
}

async function loadBundleManifest(url, { cache, signal }) {
  const response = await fetch(url.href, {
    cache: cache ? "default" : "no-store",
    credentials: "omit",
    method: "GET",
    redirect: "error",
    referrerPolicy: "no-referrer",
    signal,
  });
  if (!response.ok) {
    throw new Error(`Unable to load local ${BUNDLE_MANIFEST}: HTTP ${response.status}`);
  }
  const text = await response.text();
  if (text.length > 1_048_576) {
    throw new Error(`${BUNDLE_MANIFEST} exceeds the 1 MiB safety limit`);
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`${BUNDLE_MANIFEST} must contain valid JSON`);
  }
}

function requirePageUrl() {
  const href = globalThis.location?.href;
  if (typeof href !== "string") {
    throw new Error("The Maple ONNX adapter requires a browser page URL");
  }
  const url = new URL(href);
  if (!["http:", "https:"].includes(url.protocol) || url.origin === "null") {
    throw new Error("Serve the Maple ONNX adapter from localhost or HTTPS");
  }
  return url;
}

function validateRelativeBundlePath(value, label) {
  if (
    typeof value !== "string" ||
    !value ||
    value.startsWith("/") ||
    value.includes("\\") ||
    value.includes("%") ||
    value.includes("?") ||
    value.includes("#")
  ) {
    throw new Error(`${label} must be a normalized relative POSIX path`);
  }
  const parts = value.split("/");
  if (parts.some((part) => !part || part === "." || part === "..")) {
    throw new Error(`${label} must be a normalized relative POSIX path`);
  }
  return value;
}

function requireExactKeys(value, keys, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be a JSON object`);
  }
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (
    actual.length !== expected.length ||
    actual.some((key, index) => key !== expected[index])
  ) {
    throw new Error(`${label} keys do not match the supported contract`);
  }
}

function isGraphExternalData(path, graphPath) {
  return (
    path.startsWith(`${graphPath}.data`) ||
    path === `${graphPath}_data` ||
    path.startsWith(`${graphPath}_data.`)
  );
}

function normalizeTokenIds(value, label) {
  const source = value?.inputIds ?? value?.input_ids ?? value;
  if (!Array.isArray(source) && !ArrayBuffer.isView(source)) {
    throw new TypeError(`${label} must be an array or typed array`);
  }
  return Array.from(source, (item) => {
    const token = typeof item === "bigint" ? Number(item) : item;
    if (!Number.isSafeInteger(token) || token < 0 || token >= VOCAB_SIZE) {
      throw new RangeError(`${label} contains an invalid token id`);
    }
    return token;
  });
}

function positiveInteger(value, label) {
  const result = Number(value);
  if (!Number.isSafeInteger(result) || result <= 0) {
    throw new RangeError(`${label} must be a positive integer`);
  }
  return result;
}

function finiteNumber(value, label) {
  const result = Number(value);
  if (!Number.isFinite(result)) throw new RangeError(`${label} must be finite`);
  return result;
}

function emitProgress(callback, event) {
  if (typeof callback !== "function") return;
  try {
    callback(event);
  } catch {
    // Progress rendering must not retain model resources or clinical input.
  }
}

function forwardAbort(signal, controller) {
  if (!signal) return () => {};
  const forward = () => controller.abort(signal.reason);
  if (signal.aborted) forward();
  else signal.addEventListener("abort", forward, { once: true });
  return () => signal.removeEventListener("abort", forward);
}

function throwIfAborted(signal) {
  if (signal?.aborted) throw abortError(signal.reason);
}

function abortError(reason) {
  if (reason instanceof DOMException && reason.name === "AbortError") return reason;
  return new DOMException("Maple operation cancelled", "AbortError");
}

async function tensorData(tensor) {
  try {
    if (tensor?.data !== undefined) return tensor.data;
  } catch {
    // A GPU-resident tensor can require explicit transfer through getData().
  }
  if (typeof tensor?.getData === "function") return tensor.getData();
  throw new Error("The logits tensor is not readable on the CPU");
}

function float16Values(data) {
  if (!(data instanceof Uint16Array)) return data;
  const values = new Float32Array(data.length);
  for (let index = 0; index < data.length; index += 1) {
    values[index] = float16ToNumber(data[index]);
  }
  return values;
}

function float16ToNumber(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

async function releaseOutputMap(outputs) {
  await Promise.all(Object.values(outputs).map((tensor) => releaseTensor(tensor)));
}

async function releaseCacheMap(caches, retained = new Set()) {
  if (!caches) return;
  await Promise.all(
    Object.values(caches)
      .filter((tensor) => !retained.has(tensor))
      .map((tensor) => releaseTensor(tensor)),
  );
}

async function releaseTensor(tensor) {
  if (!tensor) return;
  const release = tensor.dispose ?? tensor.release;
  if (typeof release === "function") await release.call(tensor);
}

async function releaseResource(resource) {
  if (!resource) return;
  const release = resource.dispose ?? resource.release ?? resource.destroy;
  if (typeof release === "function") await release.call(resource);
}
