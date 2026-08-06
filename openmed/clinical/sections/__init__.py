"""Clinical section detection entry points."""

from openmed.clinical.data.doctype_loinc_ontology import (
    DOCUMENT_TYPE_LOINC_MAP,
    DOCUMENT_TYPE_TO_LOINC,
    LOINC_AXIS_NAMES,
    LOINC_DOCUMENT_CODE_MAP,
    LOINC_DOCUMENT_PROVENANCE,
    LOINC_DOCUMENT_SUBSET,
    LOINC_DOCUMENT_SUBSET_MAX_ROWS,
    document_type_loinc_coverage,
    get_document_type_mapping,
)

from .detect import (
    CONTEXT_SECTION_LOINC_CODES,
    LIST_BEARING_SECTION_LABELS,
    LIST_BEARING_SECTION_LOINC_CODES,
    LIST_SECTION_LOINC_CODES,
    SECTION_LOINC_CODES,
    UNSECTIONED_SECTION,
    SectionSpan,
    detect_sections,
    is_list_bearing_section,
    list_section_label,
    parse_section_lists,
    section_label_from_loinc,
    validate_section_spans,
    validate_sections,
)
from .doctype import (
    DEFAULT_DOCUMENT_TYPE_SIGNATURES_RESOURCE,
    UNKNOWN_DOCUMENT_TYPE,
    DocumentClassification,
    classify_document,
)
from .history import segment_history_family

__all__ = [
    "DEFAULT_DOCUMENT_TYPE_SIGNATURES_RESOURCE",
    "DOCUMENT_TYPE_LOINC_MAP",
    "DOCUMENT_TYPE_TO_LOINC",
    "LOINC_AXIS_NAMES",
    "LOINC_DOCUMENT_CODE_MAP",
    "LOINC_DOCUMENT_PROVENANCE",
    "LOINC_DOCUMENT_SUBSET",
    "LOINC_DOCUMENT_SUBSET_MAX_ROWS",
    "UNKNOWN_DOCUMENT_TYPE",
    "DocumentClassification",
    "CONTEXT_SECTION_LOINC_CODES",
    "LIST_BEARING_SECTION_LABELS",
    "LIST_BEARING_SECTION_LOINC_CODES",
    "LIST_SECTION_LOINC_CODES",
    "SECTION_LOINC_CODES",
    "SectionSpan",
    "UNSECTIONED_SECTION",
    "classify_document",
    "document_type_loinc_coverage",
    "detect_sections",
    "get_document_type_mapping",
    "is_list_bearing_section",
    "list_section_label",
    "parse_section_lists",
    "segment_history_family",
    "section_label_from_loinc",
    "validate_sections",
    "validate_section_spans",
]
