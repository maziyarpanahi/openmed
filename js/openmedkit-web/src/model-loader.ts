import {
  failClosedWhenOffline,
  OpenMedOfflineError,
  resolveOfflineOnly,
} from "./offline";
import { alignTokenOffsets } from "./offsets";
import {
  assertOfflineAssetPath,
  normalizeOrtAssetPath,
} from "./runtime/ort-web-loader";
import type {
  LoadModelOptions,
  LoadOnnxModelOptions,
  OnnxWasmPaths,
  RawTokenClassificationEntity,
  RawTransformersRuntime,
  TokenClassificationPipeline,
} from "./types";

const TRANSFORMERS_JS_MODULE = "@huggingface/transformers";
const ONNX_MODEL_FILENAMES = {
  int8: "model_int8",
  fp32: "model",
  fp16: "model_fp16",
} as const;

export async function loadOnnxModel(
  model: string,
  options: LoadOnnxModelOptions = {},
): Promise<TokenClassificationPipeline> {
  const { variant = "int8", pipelineOptions, ...loaderOptions } = options;
  return loadTokenClassificationPipeline(model, {
    ...loaderOptions,
    quantized: false,
    pipelineOptions: {
      subfolder: "",
      model_file_name: ONNX_MODEL_FILENAMES[variant],
      ...pipelineOptions,
    },
  });
}

export async function loadTokenClassificationPipeline(
  model: string,
  options: LoadModelOptions = {},
): Promise<TokenClassificationPipeline> {
  const offlineOnly = resolveOfflineOnly(options.offlineOnly);
  if (offlineOnly || isLocalModelReference(model)) {
    // Explicit local paths and offline-only loads never fall back to the Hub.
    assertLocalModelReference(model);
    options = { ...options, localFilesOnly: true, allowRemoteModels: false };
  }
  const runtime = await resolveRuntime(options.runtime);
  configureLocalRuntimeAssets(runtime, options.wasmPaths, offlineOnly);
  const localReference = isLocalModelReference(model);
  const localFilesOnly = options.localFilesOnly ?? localReference;
  const allowRemoteModels = options.allowRemoteModels ?? !localFilesOnly;
  const previousAllowRemote = runtime.env?.allowRemoteModels;
  const previousAllowLocal = runtime.env?.allowLocalModels;

  if (runtime.env) {
    runtime.env.allowLocalModels = true;
    runtime.env.allowRemoteModels = allowRemoteModels;
  }

  try {
    const pipelineOptions: Record<string, unknown> = {
      ...(options.revision === undefined ? {} : { revision: options.revision }),
      ...(options.quantized === undefined ? {} : { quantized: options.quantized }),
      ...(options.dtype === undefined ? {} : { dtype: options.dtype }),
      ...(options.device === undefined ? {} : { device: options.device }),
      ...(options.pipelineOptions ?? {}),
      local_files_only: localFilesOnly,
      localFilesOnly,
    };
    const pipeline = await failClosedWhenOffline(offlineOnly, () =>
      runtime.pipeline("token-classification", model, pipelineOptions),
    );
    // Preserve runtime properties and resource disposal while normalizing calls.
    return new Proxy(pipeline, {
      async apply(target, _receiver, [text, callOptions]) {
        const output = await target(text, callOptions);
        if (Array.isArray(output[0])) {
          return (output as RawTokenClassificationEntity[][]).map(
            (tokens) => alignTokenOffsets(text, tokens),
          );
        }
        return alignTokenOffsets(text, output as RawTokenClassificationEntity[]);
      },
      get(target, property, receiver) {
        const value = Reflect.get(target, property, receiver);
        return property === "dispose" && typeof value === "function"
          ? value.bind(target)
          : value;
      },
    }) as TokenClassificationPipeline;
  } finally {
    if (runtime.env) {
      if (previousAllowRemote === undefined) {
        delete runtime.env.allowRemoteModels;
      } else {
        runtime.env.allowRemoteModels = previousAllowRemote;
      }
      if (previousAllowLocal === undefined) {
        delete runtime.env.allowLocalModels;
      } else {
        runtime.env.allowLocalModels = previousAllowLocal;
      }
    }
  }
}

export function isLocalModelReference(model: string): boolean {
  if (model.startsWith("file://")) {
    return true;
  }
  if (
    model.startsWith("/") ||
    model.startsWith("./") ||
    model.startsWith("../") ||
    model.startsWith("~")
  ) {
    return true;
  }
  return /^[A-Za-z]:[\\/]/.test(model);
}

async function resolveRuntime(
  runtime?: LoadModelOptions["runtime"],
): Promise<RawTransformersRuntime> {
  if (typeof runtime === "function") {
    return runtime();
  }
  if (runtime) {
    return runtime;
  }
  const moduleName = TRANSFORMERS_JS_MODULE;
  try {
    return (await import(moduleName)) as RawTransformersRuntime;
  } catch (error) {
    throw new Error(
      "Install @huggingface/transformers or pass a token-classification pipeline.",
      { cause: error },
    );
  }
}

/** Throw unless `model` is a canonical local path; the error never names it. */
export function assertLocalModelReference(model: string): void {
  if (!isLocalModelReference(model) || !isLocalLocation(model)) {
    throw new OpenMedOfflineError("remote_model_refused");
  }
}

/**
 * Point ONNX Runtime Web at caller-supplied local WebAssembly assets and, in
 * offline-only mode, refuse runtime settings that would fetch from a CDN or
 * another host. Transformers.js defaults `wasmPaths` to jsDelivr in browsers.
 */
function configureLocalRuntimeAssets(
  runtime: RawTransformersRuntime,
  wasmPaths: OnnxWasmPaths | undefined,
  offlineOnly: boolean,
): void {
  const localWasmPaths =
    wasmPaths === undefined ? undefined : toLocalWasmPaths(wasmPaths);
  const env = runtime.env;
  if (env === undefined) {
    return;
  }
  if (offlineOnly) {
    const { localModelPath } = env;
    const remoteModelPath =
      localModelPath !== undefined &&
      localModelPath !== "" &&
      (typeof localModelPath !== "string" || !isLocalLocation(localModelPath));
    const remoteWasmPaths =
      localWasmPaths === undefined &&
      !isLocalWasmPaths(onnxWasmEnv(env, false)?.wasmPaths);
    if (remoteModelPath || remoteWasmPaths) {
      throw new OpenMedOfflineError("remote_asset_refused");
    }
  }
  if (localWasmPaths !== undefined) {
    // ONNX Runtime reads this once, when its WebAssembly backend starts.
    onnxWasmEnv(env, true)!.wasmPaths = localWasmPaths;
  }
}

function toLocalWasmPaths(paths: OnnxWasmPaths): OnnxWasmPaths {
  if (typeof paths !== "string" && (typeof paths !== "object" || paths === null)) {
    throw new TypeError("wasmPaths must be a string or an object of strings.");
  }
  try {
    if (typeof paths === "string") {
      return normalizeOrtAssetPath(paths);
    }
    const local: { mjs?: string; wasm?: string } = {};
    if (paths.mjs !== undefined) {
      local.mjs = assertOfflineAssetPath(paths.mjs, "wasm asset path");
    }
    if (paths.wasm !== undefined) {
      local.wasm = assertOfflineAssetPath(paths.wasm, "wasm asset path");
    }
    return local;
  } catch {
    throw new OpenMedOfflineError("remote_asset_refused");
  }
}

function isLocalWasmPaths(value: unknown): boolean {
  if (value === undefined || value === "") {
    return true;
  }
  if (typeof value === "string") {
    return isLocalWasmLocation(value);
  }
  if (
    typeof value === "object" &&
    value !== null &&
    Object.getPrototypeOf(value) === Object.prototype
  ) {
    return Object.values(value).every(
      (entry) =>
        entry === undefined ||
        (typeof entry === "string" && isLocalWasmLocation(entry)),
    );
  }
  return false;
}

function isLocalWasmLocation(location: string): boolean {
  // Transformers.js swaps a cached WebAssembly factory in as a blob: URL.
  return location.startsWith("blob:") || isLocalLocation(location);
}

function isLocalLocation(path: string): boolean {
  try {
    assertOfflineAssetPath(path);
    return true;
  } catch {
    return false;
  }
}

function onnxWasmEnv(
  env: NonNullable<RawTransformersRuntime["env"]>,
  create: boolean,
): Record<string, unknown> | undefined {
  let node: Record<string, unknown> = env;
  for (const key of ["backends", "onnx", "wasm"]) {
    const child = node[key];
    if (typeof child === "object" && child !== null) {
      node = child as Record<string, unknown>;
    } else if (create) {
      node = node[key] = {};
    } else {
      return undefined;
    }
  }
  return node;
}
