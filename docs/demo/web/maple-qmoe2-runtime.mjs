/*
 * Integrity marker around the locally built OpenMed ONNX Runtime Web fork.
 *
 * The underlying WebGPU MatMulNBits shader already supports two-bit weights.
 * OpenMed's audited patch only permits QMoE to route those weights and fixes
 * its packed-weight ratio. Keep the marker outside the generated vendor bundle
 * so operators can rebuild ONNX Runtime reproducibly without editing minified
 * output.
 */

import * as ort from "./vendor/ort.webgpu.min.mjs";

export const openmedMapleRuntimePatch = "openmed-qmoe2-webgpu-v1";
export const env = ort.env;
export const InferenceSession = ort.InferenceSession;
export const Tensor = ort.Tensor;
export default ort;
