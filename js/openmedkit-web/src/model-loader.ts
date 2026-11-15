import { alignTokenOffsets } from "./offsets";
import type {
  LoadModelOptions,
  LoadOnnxModelOptions,
  RawTokenClassificationEntity,
  RawTransformersRuntime,
  TokenClassificationPipeline,
} from "./types";

const TRANSFORMERS_JS_MODULE = "@huggingface/transformers";

// Transformers.js environment settings are shared by every pipeline created
// from the same runtime, so initialization must be serialized per instance.
const runtimeInitializationQueues = new WeakMap<RawTransformersRuntime, Promise<void>>();
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
  const runtime = await resolveRuntime(options.runtime);
  const localReference = isLocalModelReference(model);
  const localFilesOnly = options.localFilesOnly ?? localReference;
  const allowRemoteModels = options.allowRemoteModels ?? !localFilesOnly;

  return withRuntimeEnvironment(runtime, allowRemoteModels, async () => {
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
  });
}

async function withRuntimeEnvironment<T>(
  runtime: RawTransformersRuntime,
  allowRemoteModels: boolean,
  initialize: () => Promise<T>,
): Promise<T> {
  if (!runtime.env) {
    return initialize();
  }

  const previous = runtimeInitializationQueues.get(runtime) ?? Promise.resolve();
  let release!: () => void;
  const current = new Promise<void>((resolve) => {
    release = resolve;
  });
  runtimeInitializationQueues.set(runtime, current);

  await previous;
  const hadAllowRemote = Object.prototype.hasOwnProperty.call(
    runtime.env,
    "allowRemoteModels",
  );
  const previousAllowRemote = runtime.env.allowRemoteModels;
  const hadAllowLocal = Object.prototype.hasOwnProperty.call(
    runtime.env,
    "allowLocalModels",
  );
  const previousAllowLocal = runtime.env.allowLocalModels;

  try {
    runtime.env.allowLocalModels = true;
    runtime.env.allowRemoteModels = allowRemoteModels;
    return await initialize();
  } finally {
    if (hadAllowRemote) {
      runtime.env.allowRemoteModels = previousAllowRemote;
    } else {
      delete runtime.env.allowRemoteModels;
    }
    if (hadAllowLocal) {
      runtime.env.allowLocalModels = previousAllowLocal;
    } else {
      delete runtime.env.allowLocalModels;
    }
    release();
    if (runtimeInitializationQueues.get(runtime) === current) {
      runtimeInitializationQueues.delete(runtime);
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
