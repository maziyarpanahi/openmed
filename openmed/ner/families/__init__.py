"""Model family abstractions for zero-shot NER."""

from __future__ import annotations

from .base import ModelFamily
from .gliner import ensure_gliner_available, is_gliner_available
from .gliner2 import (
    ensure_gliner2_available,
    is_gliner2_available,
    load_gliner2_handle,
    clear_gliner2_cache,
)

__all__ = [
    "ModelFamily",
    "ensure_gliner_available",
    "is_gliner_available",
    "ensure_gliner2_available",
    "is_gliner2_available",
    "load_gliner2_handle",
    "clear_gliner2_cache",
]
