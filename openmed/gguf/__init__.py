"""GGUF export helpers for embedding backbones."""

from importlib import import_module

__all__ = [
    "GGUF_FORMAT",
    "MANIFEST_FILENAME",
    "GgufArtifact",
    "GgufConversionResult",
    "GgufExportError",
    "UnsupportedGgufModelError",
    "export_gguf",
]


def __getattr__(name: str):
    """Lazily expose the converter API without preloading its CLI module."""

    if name in __all__:
        module = import_module("openmed.gguf.convert")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
