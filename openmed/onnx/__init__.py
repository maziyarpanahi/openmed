"""ONNX and browser-targeted conversion helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ANDROID_ONNX_FORMAT",
    "ANDROID_ONNX_OPSET",
    "ANDROID_PROFILE_NAME",
    "AndroidProfileValidation",
    "ExportArtifact",
    "OnnxConversionResult",
    "ORT_ANDROID_FORMAT",
    "OrtMobileConversionResult",
    "convert",
    "convert_android_onnx_to_ort",
    "export_android_fp16",
    "export_onnx",
    "export_transformersjs_bundle",
    "export_webgpu",
    "validate_android_profile",
    "validate_transformersjs_bundle",
    "validate_transformersjs_contract",
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
        if name.startswith(("export_transformersjs", "validate_transformersjs")):
            module = import_module("openmed.onnx.transformersjs")
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
