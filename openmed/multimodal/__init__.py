"""Multimodal ingestion and redaction package for section 4.2.

Provides the shared ingest/redact contract (``ExtractedDocument`` and the
``redact_document`` dispatcher) that PDF/DOCX/HTML/image/DICOM ingesters build
on. The per-format parsers and OCR adapters live in sibling modules and are
registered lazily via :func:`register_handler`; this package stays importable
without the ``multimodal`` extra installed.
"""

from __future__ import annotations

# Importing the CDA adapter registers content-aware XML handling. The adapter is
# stdlib-only, so the public multimodal import path remains free of heavy deps.
from openmed.interop import cda as _cda

from .base import (
    ExtractedDocument,
    SourceSpan,
    ensure_multimodal_available,
    redact_document,
    register_handler,
)
from .exceptions import MissingDependencyError, UnsupportedDocumentError

# Importing the OCR module registers image-format handlers with the dispatcher
# so ``redact_document`` can route scans/images. It stays import-light: OCR
# backends (and Pillow) are only imported when an engine actually runs. The
# ``ocr()`` entry point is left in the submodule (``openmed.multimodal.ocr``)
# to avoid shadowing it with a function.
from .ocr import (
    DocTrEngine,
    FakeOcrEngine,
    OcrEngine,
    OcrResult,
    OcrWord,
    available_ocr_engines,
    register_ocr_engine,
    run_doctr_ocr,
)
from .tabular_csv import (
    ColumnDecision,
    RedactedTable,
    TableView,
    classify_columns,
    read_table,
    redact_table,
)

__all__ = [
    "ExtractedDocument",
    "SourceSpan",
    "redact_document",
    "register_handler",
    "ensure_multimodal_available",
    "MissingDependencyError",
    "UnsupportedDocumentError",
    "OcrResult",
    "OcrWord",
    "OcrEngine",
    "DocTrEngine",
    "FakeOcrEngine",
    "register_ocr_engine",
    "available_ocr_engines",
    "run_doctr_ocr",
    "ColumnDecision",
    "TableView",
    "RedactedTable",
    "read_table",
    "classify_columns",
    "redact_table",
]
