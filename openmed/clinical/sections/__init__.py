"""Clinical section detection entry points."""

from .detect import (
    UNSECTIONED_SECTION,
    SectionSpan,
    detect_sections,
    validate_section_spans,
)
from .doctype import (
    DEFAULT_DOCUMENT_TYPE_SIGNATURES_RESOURCE,
    UNKNOWN_DOCUMENT_TYPE,
    DocumentClassification,
    classify_document,
)

__all__ = [
    "DEFAULT_DOCUMENT_TYPE_SIGNATURES_RESOURCE",
    "UNKNOWN_DOCUMENT_TYPE",
    "DocumentClassification",
    "SectionSpan",
    "UNSECTIONED_SECTION",
    "classify_document",
    "detect_sections",
    "validate_section_spans",
]
