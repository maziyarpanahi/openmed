"""ONNX and browser-targeted conversion helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ANDROID_ONNX_FORMAT",
    "ANDROID_ONNX_OPSET",
    "ANDROID_PROFILE_NAME",
    "OPENVINO_FORMAT",
    "OPENVINO_INT8_FORMAT",
    "OPENVINO_PROFILE_NAME",
    "TENSORRT_ENGINE_FORMAT",
    "TENSORRT_FP16_FORMAT",
    "TENSORRT_INT8_FORMAT",
    "TENSORRT_PROFILE_NAME",
    "AndroidProfileValidation",
    "ExportArtifact",
    "INT8_ONNX_FILENAME",
    "ONNX_INT8_FORMAT",
    "OnnxEntity",
    "OnnxModel",
    "OnnxConversionResult",
    "OpenVinoBenchmarkRecord",
    "OpenVinoDeviceSelection",
    "OpenVinoExportResult",
    "OpenVinoExportVerification",
    "OpenVinoQuantizationRejected",
    "OpenVinoQuantizationResult",
    "OpenVinoTokenClassificationSession",
    "OpenVinoVerificationError",
    "TensorRTBenchmarkRecord",
    "TensorRTBuildError",
    "TensorRTBuildResult",
    "TensorRTExportVerification",
    "TensorRTQuantizationRejected",
    "TensorRTReproducibilityError",
    "TensorRTSessionError",
    "TensorRTShapeProfile",
    "TensorRTTokenClassificationSession",
    "TensorRTVerificationError",
    "build_openvino_benchmark_report",
    "build_tensorrt_benchmark_report",
    "build_tensorrt_engine",
    "certify_openvino_reference",
    "certify_tensorrt_reference",
    "apply_int8_recall_certification",
    "OnnxOptimizationConfig",
    "ORT_ANDROID_FORMAT",
    "OrtMobileConversionResult",
    "ShapeBucketConfig",
    "convert",
    "convert_android_onnx_to_ort",
    "export_android_fp16",
    "export_onnx",
    "export_openvino_ir",
    "export_transformersjs_bundle",
    "export_webgpu",
    "measure_openvino_latency",
    "measure_tensorrt_latency",
    "quantize_openvino_int8",
    "resolve_openvino_device",
    "run_onnx_reference_logits",
    "token_spans_from_logits",
    "int8_artifact_metadata",
    "load_onnx_model",
    "optimize_onnx_graph",
    "quantize_android_int8",
    "quantize_dynamic_int8",
    "validate_android_profile",
    "validate_optimized_onnx_export",
    "validate_transformersjs_bundle",
    "validate_transformersjs_contract",
    "write_openvino_benchmark_report",
    "verify_tensorrt_engine_hash",
    "write_tensorrt_benchmark_report",
    "write_int8_recall_delta_report",
    "write_export_manifest",
]


def __getattr__(name: str) -> Any:
    """Load conversion helpers lazily so ``python -m`` stays warning-free."""

    if name in __all__:
        if name in {"OnnxEntity", "OnnxModel", "load_onnx_model"}:
            module = import_module("openmed.onnx.inference")
            return getattr(module, name)
        if name.startswith(("ANDROID_", "Android", "export_android")) or (
            name == "validate_android_profile"
        ):
            module = import_module("openmed.onnx.android_profile")
            return getattr(module, name)
        if name in {
            "OpenVinoDeviceSelection",
            "OpenVinoTokenClassificationSession",
            "resolve_openvino_device",
        }:
            module = import_module("openmed.onnx.openvino_session")
            return getattr(module, name)
        if name.startswith(("OPENVINO_", "OpenVino")) or name in {
            "build_openvino_benchmark_report",
            "certify_openvino_reference",
            "export_openvino_ir",
            "measure_openvino_latency",
            "quantize_openvino_int8",
            "run_onnx_reference_logits",
            "token_spans_from_logits",
            "write_openvino_benchmark_report",
        }:
            module = import_module("openmed.onnx.openvino_export")
            return getattr(module, name)
        if name in {"TensorRTSessionError", "TensorRTTokenClassificationSession"}:
            module = import_module("openmed.onnx.tensorrt_session")
            return getattr(module, name)
        if name.startswith(("TENSORRT_", "TensorRT")) or name in {
            "build_tensorrt_benchmark_report",
            "build_tensorrt_engine",
            "certify_tensorrt_reference",
            "measure_tensorrt_latency",
            "verify_tensorrt_engine_hash",
            "write_tensorrt_benchmark_report",
        }:
            module = import_module("openmed.onnx.tensorrt_export")
            return getattr(module, name)
        if name.startswith(("export_transformersjs", "validate_transformersjs")):
            module = import_module("openmed.onnx.transformersjs")
            return getattr(module, name)
        if name in {
            "INT8_ONNX_FILENAME",
            "ONNX_INT8_FORMAT",
            "apply_int8_recall_certification",
            "int8_artifact_metadata",
            "quantize_android_int8",
            "quantize_dynamic_int8",
            "write_int8_recall_delta_report",
        }:
            module = import_module("openmed.onnx.quantize_int8")
            return getattr(module, name)
        if (
            name.startswith("ORT_")
            or name.startswith("OrtMobile")
            or name == "convert_android_onnx_to_ort"
        ):
            module = import_module("openmed.onnx.ort_mobile")
            return getattr(module, name)
        module = import_module("openmed.onnx.convert")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
