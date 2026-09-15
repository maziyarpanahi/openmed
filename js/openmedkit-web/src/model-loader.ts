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
