"""ONNX and browser-targeted conversion helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ExportArtifact",
    "OnnxConversionResult",
    "convert",
    "export_onnx",
    "export_webgpu",
    "write_export_manifest",
]


def __getattr__(name: str) -> Any:
    """Load conversion helpers lazily so ``python -m`` stays warning-free."""

    if name in __all__:
        module = import_module("openmed.onnx.convert")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
