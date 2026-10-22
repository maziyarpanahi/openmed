import type {
  TokenClassificationCallOptions,
  TokenClassificationOutput,
  TokenClassificationPipeline,
} from "../types";
import {
  detectOrtWebCapabilities,
  selectOrtWebBackend,
  type CapabilityDetectionOptions,
  type OrtWebBackendChoice,
  type OrtWebCapabilityProfile,
} from "./capability";

const ONNXRUNTIME_WEB_MODULE = "onnxruntime-web";
// Browsers cap a single ArrayBuffer near 2 GiB, so each shard must stay below it.
const DEFAULT_ORT_WEB_MAX_SHARD_BYTES = 2 ** 31 - 1;

export type OrtExecutionProvider = "webgpu" | "wasm";

export type OrtTensorLike = {
  data?: unknown;
  dims?: readonly number[];
  dispose?: () => void;
  type?: string;
  [key: string]: unknown;
};

export type OrtTensorConstructor = new (
  type: string,
  data: unknown,
  dims: readonly number[],
) => OrtTensorLike;

export type OrtFeeds = Record<string, OrtTensorLike | unknown>;
export type OrtResults = Record<string, OrtTensorLike | unknown>;

export interface OrtInferenceSession {
  run(
    feeds: OrtFeeds,
    options?: Record<string, unknown>,
  ): Promise<OrtResults> | OrtResults;
  release?(): Promise<void> | void;
}

export interface OrtSessionCreateOptions {
  executionProviders?: OrtExecutionProvider[];
  graphOptimizationLevel?: "disabled" | "basic" | "extended" | "all" | string;
  [key: string]: unknown;
}

export interface OrtWebRuntime {
  env?: {
    wasm?: {
      wasmPaths?: string;
      simd?: boolean;
      numThreads?: number;
      proxy?: boolean;
      [key: string]: unknown;
    };
    [key: string]: unknown;
  };
  InferenceSession: {
    create(
      modelPath: string,
      options?: OrtSessionCreateOptions,
    ): Promise<OrtInferenceSession> | OrtInferenceSession;
  };
  Tensor?: OrtTensorConstructor;
}

export type OrtWebRuntimeProvider =
  | OrtWebRuntime
  | (() => Promise<OrtWebRuntime> | OrtWebRuntime);

export interface OrtWebExternalDataShard {
  /** File name the ONNX graph references, such as `model.onnx.data.000`. */
  path: string;
  /** Local or same-origin relative location ONNX Runtime Web loads. */
  location: string;
  /** Declared shard size, checked against the limits before loading. */
  byteLength: number;
}

export interface OrtWebExternalDataOptions {
  shards: readonly OrtWebExternalDataShard[];
  /** Defaults to 2 GiB minus one byte. */
  maxShardBytes?: number;
  /** Unbounded by default. */
  maxTotalBytes?: number;
}

export interface OrtWebLoaderOptions {
  modelPath: string;
  assetPath: string;
  runtime?: OrtWebRuntimeProvider;
  capabilities?: Partial<OrtWebCapabilityProfile>;
  globalScope?: CapabilityDetectionOptions["globalScope"];
  sessionOptions?: OrtSessionCreateOptions;
  externalData?: OrtWebExternalDataOptions;
  cache?: OrtWebSessionCache;
}

export interface OrtWebLoadedSession {
  session: OrtInferenceSession;
  runtime?: OrtWebRuntime;
  backend: OrtWebBackendChoice;
  modelPath: string;
  assetPath: string;
  cacheKey: string;
}

export type OrtWebSessionCache = Map<string, Promise<OrtWebLoadedSession>>;

export interface OrtWebTokenClassificationPipelineOptions
  extends OrtWebLoaderOptions {
  tokenize: (text: string) => OrtFeeds | Promise<OrtFeeds>;
  decode: (context: OrtTokenClassificationDecodeContext) =>
    | TokenClassificationOutput
    | Promise<TokenClassificationOutput>;
  runOptions?: Record<string, unknown>;
}

export interface OrtTokenClassificationDecodeContext {
  text: string;
  inputs: OrtFeeds;
  outputs: OrtResults;
  session: OrtInferenceSession;
  backend: OrtWebBackendChoice;
  callOptions?: TokenClassificationCallOptions;
}

export const DEFAULT_ORT_WEB_SESSION_CACHE: OrtWebSessionCache = new Map();

export function assertOfflineAssetPath(path: string, label = "asset path"): string {
  const trimmed = path.trim();
  if (trimmed.length === 0) {
    throw new Error(`ONNX Runtime Web ${label} must not be empty.`);
  }
  // Browsers strip C0 controls (leading NUL..US, tab/CR/LF anywhere) and read "/\"
  // or "\/" as "//", which would turn the path into a remote URL.
  if (/[\u0000-\u001f]/.test(trimmed) || /^[\\/]{2}/.test(trimmed)) {
    throw new Error(`ONNX Runtime Web ${label} must be local/offline, not remote.`);
  }
  if (/^[A-Za-z]:[\\/]/.test(trimmed)) {
    return trimmed;
  }
  if (/^[A-Za-z][A-Za-z0-9+.-]*:/.test(trimmed)) {
    if (!trimmed.startsWith("file://")) {
      throw new Error(
        `ONNX Runtime Web ${label} must be local/offline, not remote.`,
      );
    }
    let fileUrl: URL;
    try {
      fileUrl = new URL(trimmed);
    } catch {
      throw new Error(
        `ONNX Runtime Web ${label} must be local/offline, not remote.`,
      );
    }
    if (
      (fileUrl.hostname !== "" && fileUrl.hostname !== "localhost") ||
      fileUrl.pathname.startsWith("//")
    ) {
      throw new Error(
        `ONNX Runtime Web ${label} must be local/offline, not remote.`,
      );
    }
  }
  return trimmed;
}

export function normalizeOrtAssetPath(path: string): string {
  const localPath = assertOfflineAssetPath(path, "wasm asset path");
  if (localPath.endsWith("/") || localPath.endsWith("\\")) {
    return localPath;
  }
  return `${localPath}/`;
}

export function assertOfflineExternalData(
  options: Pick<OrtWebLoaderOptions, "externalData" | "sessionOptions">,
): { path: string; data: string }[] | undefined {
  if (options.sessionOptions?.externalData !== undefined) {
    throw new Error(
      "ONNX Runtime Web external data must use the typed externalData option.",
    );
  }
  const externalData = options.externalData;
  if (externalData === undefined) {
    return undefined;
  }
  const maxShardBytes = positiveByteCount(
    externalData.maxShardBytes ?? DEFAULT_ORT_WEB_MAX_SHARD_BYTES,
    "external data maxShardBytes",
  );
  const maxTotalBytes =
    externalData.maxTotalBytes === undefined
      ? Number.POSITIVE_INFINITY
      : positiveByteCount(
          externalData.maxTotalBytes,
          "external data maxTotalBytes",
        );
  let totalBytes = 0;
  const files = externalData.shards.map((shard, index) => {
    const label = `external data shard ${index}`;
    const file = {
      path: assertOfflineShardLocation(shard.path, `${label} path`),
      data: assertOfflineShardLocation(shard.location, `${label} location`),
    };
    const byteLength = positiveByteCount(shard.byteLength, `${label} byteLength`);
    if (byteLength > maxShardBytes) {
      throw new Error(
        `ONNX Runtime Web ${label} exceeds the per-shard byte limit.`,
      );
    }
    totalBytes += byteLength;
    return file;
  });
  if (totalBytes > maxTotalBytes) {
    throw new Error("ONNX Runtime Web external data exceeds the total byte limit.");
  }
  return files;
}

export async function loadOrtWebSession(
  options: OrtWebLoaderOptions,
): Promise<OrtWebLoadedSession> {
  const modelPath = assertOfflineAssetPath(options.modelPath, "model path");
  const assetPath = normalizeOrtAssetPath(options.assetPath);
  const externalData = assertOfflineExternalData(options);
  const capabilities = resolveCapabilities(options);
  const backend = selectOrtWebBackend(capabilities);
  const cache = options.cache ?? DEFAULT_ORT_WEB_SESSION_CACHE;
  const sessionOptions: OrtSessionCreateOptions = {
    graphOptimizationLevel: "all",
    ...(options.sessionOptions ?? {}),
    executionProviders: backend.executionProviders,
  };
  if (externalData !== undefined) {
    sessionOptions.externalData = externalData;
  }
  const cacheKey = stableStringify({
    modelPath,
    assetPath,
    backend: {
      name: backend.backend,
      executionProviders: backend.executionProviders,
      wasm: backend.wasm,
    },
    sessionOptions,
  });
  const cached = cache.get(cacheKey);
  if (cached !== undefined) {
    return cached;
  }

  const promise = createOrtWebSession({
    ...options,
    modelPath,
    assetPath,
    backend,
    sessionOptions,
    cacheKey,
  }).catch((error: unknown) => {
    if (cache.get(cacheKey) === promise) {
      cache.delete(cacheKey);
    }
    throw error;
  });
  cache.set(cacheKey, promise);
  return promise;
}

export async function loadOrtWebTokenClassificationPipeline(
  options: OrtWebTokenClassificationPipelineOptions,
): Promise<TokenClassificationPipeline> {
  const loaded = await loadOrtWebSession(options);
  return async (
    text: string,
    callOptions?: TokenClassificationCallOptions,
  ): Promise<TokenClassificationOutput> => {
    const inputs = await options.tokenize(text);
    const outputs = await loaded.session.run(inputs, options.runOptions);
    const context: OrtTokenClassificationDecodeContext = {
      text,
      inputs,
      outputs,
      session: loaded.session,
      backend: loaded.backend,
    };
    if (callOptions !== undefined) {
      context.callOptions = callOptions;
    }
    return options.decode(context);
  };
}

export function configureOrtWebRuntime(
  runtime: OrtWebRuntime,
  backend: OrtWebBackendChoice,
  assetPath: string,
): void {
  const env = (runtime.env ??= {});
  const wasm = (env.wasm ??= {});
  wasm.wasmPaths = normalizeOrtAssetPath(assetPath);
  wasm.simd = backend.wasm.simd;
  wasm.numThreads = backend.wasm.numThreads;
  wasm.proxy = false;
}

export function clearOrtWebSessionCache(cache = DEFAULT_ORT_WEB_SESSION_CACHE) {
  cache.clear();
}

async function createOrtWebSession(
  options: OrtWebLoaderOptions & {
    backend: OrtWebBackendChoice;
    sessionOptions: OrtSessionCreateOptions;
    cacheKey: string;
  },
): Promise<OrtWebLoadedSession> {
  const runtime = await resolveOrtRuntime(options.runtime);
  configureOrtWebRuntime(runtime, options.backend, options.assetPath);
  const session = await runtime.InferenceSession.create(
    options.modelPath,
    options.sessionOptions,
  );
  return {
    session,
    runtime,
    backend: options.backend,
    modelPath: options.modelPath,
    assetPath: options.assetPath,
    cacheKey: options.cacheKey,
  };
}

async function resolveOrtRuntime(
  runtime?: OrtWebRuntimeProvider,
): Promise<OrtWebRuntime> {
  if (typeof runtime === "function") {
    return runtime();
  }
  if (runtime !== undefined) {
    return runtime;
  }
  try {
    return (await import(ONNXRUNTIME_WEB_MODULE)) as OrtWebRuntime;
  } catch (error) {
    throw new Error(
      "Install onnxruntime-web or pass an ONNX Runtime Web runtime object.",
      { cause: error },
    );
  }
}

function resolveCapabilities(
  options: OrtWebLoaderOptions,
): OrtWebCapabilityProfile {
  const detectionOptions: CapabilityDetectionOptions = {};
  if (options.globalScope !== undefined) {
    detectionOptions.globalScope = options.globalScope;
  }
  if (options.capabilities !== undefined) {
    detectionOptions.overrides = options.capabilities;
  }
  return detectOrtWebCapabilities(detectionOptions);
}

function assertOfflineShardLocation(path: string, label: string): string {
  const location = assertOfflineAssetPath(path, label);
  if (location.includes("?")) {
    throw new Error(`ONNX Runtime Web ${label} must not include a query string.`);
  }
  return location;
}

function positiveByteCount(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`ONNX Runtime Web ${label} must be a positive integer.`);
  }
  return value;
}

function stableStringify(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((entry) => stableStringify(entry)).join(",")}]`;
  }
  const entries = Object.entries(value as Record<string, unknown>)
    .filter(([, entry]) => entry !== undefined)
    .sort(([left], [right]) => left.localeCompare(right));
  return `{${entries
    .map(([key, entry]) => `${JSON.stringify(key)}:${stableStringify(entry)}`)
    .join(",")}}`;
}
