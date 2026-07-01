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

from . import dicom as _dicom

# Importing the Markdown/AsciiDoc adapter registers lightweight text-markup
# handlers. Third-party parser availability is checked only when a handler runs.
from . import documents_markdown as _documents_markdown
from .base import (
    ExtractedDocument,
    SourceSpan,
    ensure_multimodal_available,
    redact_document,
    register_handler,
)
from .chatlog_jsonl import (
    ChatLogRedactionSummary,
    MessagesListAdapter,
    RedactedChatLog,
    TurnRecordAdapter,
    iter_redacted_chatlog_jsonl,
    redact_chatlog_jsonl,
    write_redacted_chatlog_jsonl,
)
from .dicom import (
    DicomHeaderAction,
    DicomHeaderDeidPolicy,
    DicomHeaderDeidResult,
    deidentify_dicom_headers,
)
from .documents_markdown import extract_asciidoc, extract_markdown, redact_source_text
from .documents_pdf import ProjectedRectangle, extract_pdf, project_text_spans
from .exceptions import MissingDependencyError, UnsupportedDocumentError
from .metadata_scrub import (
    MetadataFinding,
    MetadataScrubError,
    MetadataScrubResult,
    ResidualMetadataReport,
    assert_metadata_clean,
    scrub_metadata,
    verify_metadata,
)

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
    "ChatLogRedactionSummary",
    "RedactedChatLog",
    "TurnRecordAdapter",
    "MessagesListAdapter",
    "iter_redacted_chatlog_jsonl",
    "redact_chatlog_jsonl",
    "write_redacted_chatlog_jsonl",
    "DicomHeaderAction",
    "DicomHeaderDeidPolicy",
    "DicomHeaderDeidResult",
    "deidentify_dicom_headers",
    "ProjectedRectangle",
    "extract_pdf",
    "project_text_spans",
    "MetadataFinding",
    "ResidualMetadataReport",
    "MetadataScrubResult",
    "MetadataScrubError",
    "scrub_metadata",
    "verify_metadata",
    "assert_metadata_clean",
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
    "extract_markdown",
    "extract_asciidoc",
    "redact_source_text",
]
