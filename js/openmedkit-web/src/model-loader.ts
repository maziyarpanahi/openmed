import { alignTokenOffsets } from "./offsets";
import type {
  LoadModelOptions,
  LoadOnnxModelOptions,
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
  const runtime = await resolveRuntime(options.runtime);
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
    const pipeline = await runtime.pipeline(
      "token-classification",
      model,
      pipelineOptions,
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
