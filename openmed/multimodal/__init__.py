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
from .ocr import OcrResult, available_ocr_engines, run_doctr_ocr
from .ocr import ocr as run_ocr

__all__ = [
    "ExtractedDocument",
    "SourceSpan",
    "OcrResult",
    "redact_document",
    "register_handler",
    "ensure_multimodal_available",
    "available_ocr_engines",
    "run_ocr",
    "run_doctr_ocr",
    "MissingDependencyError",
    "UnsupportedDocumentError",
]
