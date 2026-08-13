"""Multimodal ingestion and redaction package for section 4.2.

Intended contents include PDF/DOCX/HTML->text+offsets extraction, OCR, and
image/DICOM redaction. The per-format parsers and OCR adapters use the shared
``ExtractedDocument`` contract and are registered lazily via
:func:`register_handler`, so this package stays importable without the
``multimodal`` extra installed.
"""

from __future__ import annotations

# Importing the CDA adapter registers content-aware XML handling. The adapter is
# stdlib-only, so the public multimodal import path remains free of heavy deps.
from openmed.interop import cda as _cda

from . import contacts_calendar as _contacts_calendar
from . import dicom as _dicom

# Importing the DICOM SR adapter registers content-aware SR flattening for
# ``.dcm`` files. pydicom is imported lazily, so the public multimodal import
# path stays free of the optional imaging extra.
from . import dicom_sr as _dicom_sr

# Importing the Markdown/AsciiDoc adapter registers lightweight text-markup
# handlers. Third-party parser availability is checked only when a handler runs.
from . import documents_docx as _documents_docx
from . import documents_markdown as _documents_markdown
from .base import (
    ExtractedDocument,
    SourceSpan,
    ensure_multimodal_available,
    is_multimodal_available,
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
from .chw_forms import (
    ACTION_GENERALIZE_GEO,
    ChwFieldDecision,
    RedactedChwForm,
    classify_chw_field,
    parse_xform_path,
    redact_chw_form,
)
from .contacts_calendar import redact_contacts_calendar
from .dicom import (
    DicomHeaderAction,
    DicomHeaderDeidPolicy,
    DicomHeaderDeidResult,
    DicomPixelFinding,
    DicomPixelRedactionPolicy,
    DicomPixelRedactionResult,
    DicomResidualTextReport,
    deidentify_dicom_headers,
    redact_dicom_pixels,
)
from .dicom_sr import (
    DICOM_SR_ADVISORY,
    SrContentItem,
    extract_dicom_sr,
    walk_sr_content_tree,
)
from .documents_docx import (
    DocxRedaction,
    DocxRunRange,
    extract_docx,
    map_text_spans_to_docx_runs,
    write_redacted_docx,
)
from .documents_markdown import extract_asciidoc, extract_markdown, redact_source_text
from .documents_pdf import ProjectedRectangle, extract_pdf, project_text_spans
from .epub import extract_epub
from .exceptions import MissingDependencyError, UnsupportedDocumentError
from .image import (
    ImageMetadataReport,
    ImageRedactionVerificationError,
    RedactedImage,
    ResidualPhi,
    ResidualPhiReport,
    assert_no_residual_phi,
    redact_image,
    verify_image_metadata,
    verify_image_redaction,
)
from .layout import (
    FakeLayoutEngine,
    FakeLayoutInput,
    LayoutBlock,
    LayoutColumn,
    LayoutDocument,
    LayoutMapEntry,
    LayoutSpan,
    LayoutWordSpan,
    parse_layout,
)
from .metadata_scrub import (
    MetadataFinding,
    MetadataScrubError,
    MetadataScrubResult,
    ResidualMetadataReport,
    assert_metadata_clean,
    scrub_metadata,
    verify_metadata,
)

# Importing the OCR module registers remaining OCR-only image-format handlers
# (BMP/GIF/WebP). PNG/JPEG/TIFF are registered by ``image`` above because they
# support pixel redaction and metadata stripping. OCR backends (and Pillow) are
# only imported when an engine actually runs. The ``ocr()`` entry point is left
# in the submodule (``openmed.multimodal.ocr``) to avoid shadowing it.
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
from .odt import extract_odt
from .rtf import extract_rtf
from .sms_messages import (
    DEFAULT_SMS_MODEL,
    SHORT_TEXT,
    SHORT_TEXT_PRESET,
    RedactedSMSExport,
    ShortTextPreset,
    SMSRedactionSummary,
    coarsen_timestamp,
    deidentify_short_text,
    iter_redacted_sms_records,
    redact_sms_csv,
    redact_sms_export,
    redact_sms_json,
    resolve_short_text_preset,
)
from .tabular_csv import (
    ColumnDecision,
    RedactedTable,
    TableView,
    classify_columns,
    derive_date_shift_days,
    read_table,
    redact_table,
    shift_quasi_identifier_date,
)
from .verify_pdf import (
    PdfFidelityReport,
    RedactionFidelityError,
    RegionFidelity,
    verify_redacted_pdf,
)

__all__ = [
    "ExtractedDocument",
    "SourceSpan",
    "redact_document",
    "register_handler",
    "ensure_multimodal_available",
    "is_multimodal_available",
    "MissingDependencyError",
    "UnsupportedDocumentError",
    "ChatLogRedactionSummary",
    "RedactedChatLog",
    "TurnRecordAdapter",
    "MessagesListAdapter",
    "iter_redacted_chatlog_jsonl",
    "redact_chatlog_jsonl",
    "write_redacted_chatlog_jsonl",
    "ACTION_GENERALIZE_GEO",
    "ChwFieldDecision",
    "RedactedChwForm",
    "classify_chw_field",
    "parse_xform_path",
    "redact_chw_form",
    "redact_contacts_calendar",
    "DicomHeaderAction",
    "DicomHeaderDeidPolicy",
    "DicomHeaderDeidResult",
    "DicomPixelFinding",
    "DicomPixelRedactionPolicy",
    "DicomPixelRedactionResult",
    "DicomResidualTextReport",
    "deidentify_dicom_headers",
    "redact_dicom_pixels",
    "DICOM_SR_ADVISORY",
    "SrContentItem",
    "extract_dicom_sr",
    "walk_sr_content_tree",
    "ProjectedRectangle",
    "extract_pdf",
    "project_text_spans",
    "DocxRedaction",
    "DocxRunRange",
    "extract_docx",
    "map_text_spans_to_docx_runs",
    "write_redacted_docx",
    "extract_odt",
    "extract_epub",
    "extract_rtf",
    "MetadataFinding",
    "ResidualMetadataReport",
    "MetadataScrubResult",
    "MetadataScrubError",
    "scrub_metadata",
    "verify_metadata",
    "assert_metadata_clean",
    "ImageMetadataReport",
    "ImageRedactionVerificationError",
    "RedactedImage",
    "ResidualPhi",
    "ResidualPhiReport",
    "assert_no_residual_phi",
    "redact_image",
    "verify_image_metadata",
    "verify_image_redaction",
    "OcrResult",
    "OcrWord",
    "OcrEngine",
    "DocTrEngine",
    "FakeOcrEngine",
    "register_ocr_engine",
    "available_ocr_engines",
    "run_doctr_ocr",
    "FakeLayoutEngine",
    "FakeLayoutInput",
    "LayoutBlock",
    "LayoutColumn",
    "LayoutDocument",
    "LayoutMapEntry",
    "LayoutSpan",
    "LayoutWordSpan",
    "parse_layout",
    "DEFAULT_SMS_MODEL",
    "SHORT_TEXT",
    "SHORT_TEXT_PRESET",
    "RedactedSMSExport",
    "SMSRedactionSummary",
    "ShortTextPreset",
    "coarsen_timestamp",
    "deidentify_short_text",
    "iter_redacted_sms_records",
    "redact_sms_csv",
    "redact_sms_export",
    "redact_sms_json",
    "resolve_short_text_preset",
    "ColumnDecision",
    "TableView",
    "RedactedTable",
    "read_table",
    "classify_columns",
    "redact_table",
    "derive_date_shift_days",
    "shift_quasi_identifier_date",
    "extract_markdown",
    "extract_asciidoc",
    "redact_source_text",
    "PdfFidelityReport",
    "RegionFidelity",
    "RedactionFidelityError",
    "verify_redacted_pdf",
]
