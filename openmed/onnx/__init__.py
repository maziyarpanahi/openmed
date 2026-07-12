"""ONNX and browser-targeted conversion helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from openmed.core.capabilities import is_backend_available as _is_backend_available
from openmed.core.capabilities import require_backend as _require_backend


def is_onnx_available() -> bool:
    """Return True when the ``onnx`` extra is importable, without importing it."""

    return _is_backend_available("onnx")


def ensure_onnx_available() -> None:
    """Raise an actionable error when the ``onnx`` extra is not installed."""

    _require_backend("onnx", feature="ONNX export and inference")


def is_openvino_available() -> bool:
    """Return True when the ``openvino`` extra is importable, without importing."""

    return _is_backend_available("openvino")


def ensure_openvino_available() -> None:
    """Raise an actionable error when the ``openvino`` extra is not installed."""

    _require_backend("openvino", feature="OpenVINO export and inference")


__all__ = [
    "ensure_onnx_available",
    "ensure_openvino_available",
    "is_onnx_available",
    "is_openvino_available",
    "ANDROID_ONNX_FORMAT",
    "ANDROID_ONNX_OPSET",
    "ANDROID_PROFILE_NAME",
    "OPENVINO_FORMAT",
    "OPENVINO_INT8_FORMAT",
    "OPENVINO_PROFILE_NAME",
    "AndroidProfileValidation",
    "ExportArtifact",
    "INT8_ONNX_FILENAME",
    "ONNX_INT8_FORMAT",
    "OnnxConversionResult",
    "OpenVinoBenchmarkRecord",
    "OpenVinoDeviceSelection",
    "OpenVinoExportResult",
    "OpenVinoExportVerification",
    "OpenVinoQuantizationRejected",
    "OpenVinoQuantizationResult",
    "OpenVinoTokenClassificationSession",
    "OpenVinoVerificationError",
    "build_openvino_benchmark_report",
    "certify_openvino_reference",
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
    "quantize_openvino_int8",
    "resolve_openvino_device",
    "run_onnx_reference_logits",
    "token_spans_from_logits",
    "int8_artifact_metadata",
    "optimize_onnx_graph",
    "quantize_android_int8",
    "quantize_dynamic_int8",
    "validate_android_profile",
    "validate_optimized_onnx_export",
    "validate_transformersjs_bundle",
    "validate_transformersjs_contract",
    "write_openvino_benchmark_report",
    "write_int8_recall_delta_report",
    "write_export_manifest",
]


def __getattr__(name: str) -> Any:
    """Load conversion helpers lazily so ``python -m`` stays warning-free."""

    if name in __all__:
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
