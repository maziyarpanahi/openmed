"""Clinical section detection entry points."""

from .detect import UNSECTIONED_SECTION, detect_sections
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
    "UNSECTIONED_SECTION",
    "classify_document",
    "detect_sections",
]
