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

const RUNTIME_ENVIRONMENT_KEYS = ["allowLocalModels", "allowRemoteModels"] as const;

type RuntimeEnvironment = NonNullable<RawTransformersRuntime["env"]>;
type RuntimeEnvironmentKey = (typeof RUNTIME_ENVIRONMENT_KEYS)[number];
type RuntimeEnvironmentValues = Partial<Record<RuntimeEnvironmentKey, boolean>>;

interface EnvironmentSnapshot {
  key: RuntimeEnvironmentKey;
  present: boolean;
  value: boolean | undefined;
}

interface EnvironmentWindow {
  signature: string;
  members: number;
  snapshot: EnvironmentSnapshot[];
}

interface RuntimeEnvironmentCoordinator {
  window: EnvironmentWindow | null;
  waiting: Array<() => void>;
}

const runtimeEnvironmentCoordinators = new WeakMap<
  object,
  RuntimeEnvironmentCoordinator
>();

function environmentSignature(values: RuntimeEnvironmentValues): string {
  return RUNTIME_ENVIRONMENT_KEYS.map((key) => `${key}=${values[key]}`).join("&");
}

function environmentSnapshot(env: RuntimeEnvironment): EnvironmentSnapshot[] {
  return RUNTIME_ENVIRONMENT_KEYS.map((key) => ({
    key,
    present: Object.prototype.hasOwnProperty.call(env, key),
    value: env[key],
  }));
}

function applyEnvironmentValues(
  env: RuntimeEnvironment,
  values: RuntimeEnvironmentValues,
): void {
  for (const key of RUNTIME_ENVIRONMENT_KEYS) {
    const value = values[key];
    if (value !== undefined) {
      env[key] = value;
    }
  }
}

function restoreEnvironment(
  env: RuntimeEnvironment,
  snapshot: EnvironmentSnapshot[],
): void {
  for (const { key, present, value } of snapshot) {
    if (present) {
      Reflect.set(env, key, value);
    } else {
      Reflect.deleteProperty(env, key);
    }
  }
}

function runtimeEnvironmentCoordinator(
  env: RuntimeEnvironment,
): RuntimeEnvironmentCoordinator {
  const existing = runtimeEnvironmentCoordinators.get(env);
  if (existing) {
    return existing;
  }
  const coordinator: RuntimeEnvironmentCoordinator = { window: null, waiting: [] };
  runtimeEnvironmentCoordinators.set(env, coordinator);
  return coordinator;
}

/**
 * Open a runtime environment window, or join the one already open.
 *
 * Transformers.js keeps `allowLocalModels` and `allowRemoteModels` on a module
 * object shared by every caller, so a load that mutates them must not overlap a
 * load that requests different settings. Callers that request the same settings
 * share one window and stay concurrent; callers that request different settings
 * wait for the open window to close. Once the last member finishes, the exact
 * previous state is restored, including keys that were absent.
 */
async function acquireRuntimeEnvironment(
  env: RuntimeEnvironment,
  values: RuntimeEnvironmentValues,
): Promise<() => void> {
  const coordinator = runtimeEnvironmentCoordinator(env);
  const signature = environmentSignature(values);
  while (coordinator.window && coordinator.window.signature !== signature) {
    await new Promise<void>((resolve) => {
      coordinator.waiting.push(resolve);
    });
  }
  if (!coordinator.window) {
    const opened: EnvironmentWindow = {
      signature,
      members: 0,
      snapshot: environmentSnapshot(env),
    };
    coordinator.window = opened;
    applyEnvironmentValues(env, values);
  }
  const window: EnvironmentWindow = coordinator.window;
  window.members += 1;
  return () => releaseRuntimeEnvironment(env, coordinator, window);
}

function releaseRuntimeEnvironment(
  env: RuntimeEnvironment,
  coordinator: RuntimeEnvironmentCoordinator,
  window: EnvironmentWindow,
): void {
  window.members -= 1;
  if (window.members > 0) {
    return;
  }
  restoreEnvironment(env, window.snapshot);
  coordinator.window = null;
  const waiting = coordinator.waiting;
  coordinator.waiting = [];
  for (const resume of waiting) {
    resume();
  }
}

export async function loadOnnxModel(
  model: string,
  options: LoadOnnxModelOptions = {},
): Promise<TokenClassificationPipeline> {
  return loadOnnx(model, options, false);
}

/**
 * Load an OpenMed ONNX model whose runtime assets must stay local, even when
 * the model reference is a Hub id resolved from local files. Package-internal:
 * task packs use it when the caller has not consented to remote models.
 */
export async function loadOnnxModelWithLocalAssets(
  model: string,
  options: LoadOnnxModelOptions = {},
): Promise<TokenClassificationPipeline> {
  return loadOnnx(model, options, true);
}

function loadOnnx(
  model: string,
  options: LoadOnnxModelOptions,
  localAssetsOnly: boolean,
): Promise<TokenClassificationPipeline> {
  const { variant = "int8", pipelineOptions, ...loaderOptions } = options;
  return loadPipeline(
    model,
    {
      ...loaderOptions,
      quantized: false,
      pipelineOptions: {
        subfolder: "",
        model_file_name: ONNX_MODEL_FILENAMES[variant],
        ...pipelineOptions,
      },
    },
    localAssetsOnly,
  );
}

export async function loadTokenClassificationPipeline(
  model: string,
  options: LoadModelOptions = {},
): Promise<TokenClassificationPipeline> {
  return loadPipeline(model, options, false);
}

async function loadPipeline(
  model: string,
  options: LoadModelOptions,
  localAssetsOnly: boolean,
): Promise<TokenClassificationPipeline> {
  const offlineOnly = resolveOfflineOnly(options.offlineOnly);
  if (offlineOnly || isLocalModelReference(model)) {
    // Explicit local paths and offline-only loads never fall back to the Hub.
    assertLocalModelReference(model);
    options = { ...options, localFilesOnly: true, allowRemoteModels: false };
  }
  const runtime = await resolveRuntime(options.runtime);
  configureLocalRuntimeAssets(
    runtime,
    options.wasmPaths,
    offlineOnly || localAssetsOnly,
  );
  const localReference = isLocalModelReference(model);
  const localFilesOnly = options.localFilesOnly ?? localReference;
  const allowRemoteModels = options.allowRemoteModels ?? !localFilesOnly;
  const releaseEnvironment = runtime.env
    ? await acquireRuntimeEnvironment(runtime.env, {
        allowLocalModels: true,
        allowRemoteModels,
      })
    : null;

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
    releaseEnvironment?.();
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
 * offline-only mode or for a local-assets-only load, refuse runtime settings
 * that would fetch from a CDN or another host. Transformers.js defaults
 * `wasmPaths` to jsDelivr in browsers.
 */
function configureLocalRuntimeAssets(
  runtime: RawTransformersRuntime,
  wasmPaths: OnnxWasmPaths | undefined,
  refuseRemoteAssets: boolean,
): void {
  const localWasmPaths =
    wasmPaths === undefined ? undefined : toLocalWasmPaths(wasmPaths);
  const env = runtime.env;
  if (env === undefined) {
    return;
  }
  if (refuseRemoteAssets) {
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
