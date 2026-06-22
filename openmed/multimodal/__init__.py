"""Multimodal ingestion and redaction package for section 4.2.

Provides the shared ingest/redact contract (``ExtractedDocument`` and the
``redact_document`` dispatcher) that PDF/DOCX/HTML/image/DICOM ingesters build
on. The per-format parsers and OCR adapters live in sibling modules and are
registered lazily via :func:`register_handler`; this package stays importable
without the ``multimodal`` extra installed.
"""

from __future__ import annotations

from .base import (
    ExtractedDocument,
    SourceSpan,
    ensure_multimodal_available,
    redact_document,
    register_handler,
)
from .exceptions import MissingDependencyError, UnsupportedDocumentError

__all__ = [
    "ExtractedDocument",
    "SourceSpan",
    "redact_document",
    "register_handler",
    "ensure_multimodal_available",
    "MissingDependencyError",
    "UnsupportedDocumentError",
]
