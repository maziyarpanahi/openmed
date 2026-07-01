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
    "AndroidProfileValidation",
    "ExportArtifact",
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
    "convert",
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
    "validate_android_profile",
    "validate_transformersjs_bundle",
    "validate_transformersjs_contract",
    "write_openvino_benchmark_report",
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
        module = import_module("openmed.onnx.convert")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
