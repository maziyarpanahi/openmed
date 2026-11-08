// Compile-only migration target for the React Native bridge tarball.
import {
  deidentify,
  loadModel,
  type OpenMedDeidentifyResult,
  type OpenMedModelSource,
} from "@openmed/openmedkit-react-native";

/** Load a local model through the native module, then redact under a named policy. */
export async function redactOnDevice(
  modelPath: string,
  text: string,
  hashSecret: string,
): Promise<OpenMedDeidentifyResult> {
  const source: OpenMedModelSource = { modelPath, backend: "onnx" };
  await loadModel(source);
  return deidentify(text, { policy: "hipaa_safe_harbor", hashSecret });
}
