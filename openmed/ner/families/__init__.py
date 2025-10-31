"""Model family abstractions for zero-shot NER."""

from __future__ import annotations

from .base import ModelFamily
from .gliner import ensure_gliner_available, is_gliner_available

__all__ = [
    "ModelFamily",
    "ensure_gliner_available",
    "is_gliner_available",
]
