"""CoreML export tools for OpenMed.

Convert HuggingFace token-classification models to CoreML format
for deployment on iOS and macOS.

Install with: ``pip install openmed[coreml]``
"""

from __future__ import annotations

from openmed.core.capabilities import is_backend_available as _is_backend_available
from openmed.core.capabilities import require_backend as _require_backend

__all__ = ["ensure_coreml_available", "is_coreml_available"]


def is_coreml_available() -> bool:
    """Return True when the ``coreml`` extra is importable, without importing it."""

    return _is_backend_available("coreml")


def ensure_coreml_available() -> None:
    """Raise an actionable error when the ``coreml`` extra is not installed."""

    _require_backend("coreml", feature="CoreML export")
