// Compile-only migration targets from docs/migration/openmedkit-installed-packages.md.
// Type-checking this file against the installed tarball's declarations proves
// the documented replacements resolve for downstream TypeScript projects.
import {
  DEFAULT_MODEL_ID,
  deidentify,
  isLocalModelReference,
  loadOrtWebTokenClassificationPipeline,
  loadTokenClassificationPipeline,
  type DeidentifyOptions,
  type OpenMedDeidentifyResult,
  type OrtWebTokenClassificationPipelineOptions,
  type TokenClassificationPipeline,
} from "openmed";

/** The default model id is a Hub reference, so default calls are not offline. */
export const defaultModelIsRemote: boolean = !isLocalModelReference(DEFAULT_MODEL_ID);

/** Load a pre-staged Transformers.js model directory without remote fallback. */
export async function loadLocalPipeline(
  modelDirectory: string,
): Promise<TokenClassificationPipeline> {
  if (!isLocalModelReference(modelDirectory)) {
    throw new Error("model directory must be a local path");
  }
  return loadTokenClassificationPipeline(modelDirectory, {
    localFilesOnly: true,
    allowRemoteModels: false,
  });
}

/** Load an ONNX Runtime Web session from local model and wasm asset paths. */
export async function loadLocalOrtPipeline(
  options: OrtWebTokenClassificationPipelineOptions,
): Promise<TokenClassificationPipeline> {
  return loadOrtWebTokenClassificationPipeline(options);
}

/** Redact with an explicit pipeline and a caller-held hash key. */
export async function redactOffline(
  text: string,
  pipeline: TokenClassificationPipeline,
  hashSecret: Uint8Array,
): Promise<OpenMedDeidentifyResult> {
  const options: DeidentifyOptions = { pipeline, hashSecret };
  return deidentify(text, options);
}
