"""Small, license-aware LOINC document-ontology mapping.

This module intentionally contains only the code/label rows needed by the
rules-first document-type classifier. It is not a copy of the LOINC release or
an ontology loader. The axis values are the local resolution policy used for
the supported document classes; they are descriptive metadata, not a second
terminology service.

The classifier resolves the document type first, then selects the matching
LOINC document code. A code is never inferred from an axis in isolation. When
the classifier abstains, the type is unmapped, or confidence is below
``LOINC_MIN_CONFIDENCE``, callers receive ``None`` for the LOINC code and axes.
That sentinel keeps an uncertain prediction from becoming an invalid FHIR
Coding. For a multi-axis case such as a CT chest report, the document-level
radiology type remains primary and the subject-matter-domain axis is resolved
to ``imaging``; modality and body site do not replace the document code.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from math import isfinite
from types import MappingProxyType
from typing import Any, TypedDict

LOINC_DOCUMENT_SYSTEM = "http://loinc.org"
LOINC_DOCUMENT_SUBSET_NAME = "openmed-document-ontology-subset"
LOINC_DOCUMENT_SUBSET_LICENSE = "LOINC Terms of Use"
LOINC_DOCUMENT_SUBSET_MAX_ROWS = 32
LOINC_MIN_CONFIDENCE = 0.75

LOINC_AXIS_NAMES: tuple[str, ...] = (
    "type_of_service",
    "subject_matter_domain",
    "role",
    "setting",
)


class LoincDocumentAxes(TypedDict):
    """The four descriptive axes retained for a supported document type."""

    type_of_service: str
    subject_matter_domain: str
    role: str
    setting: str


class LoincDocumentMapping(TypedDict):
    """One code, label, and axis breakdown from the bundled subset."""

    code: str
    label: str
    axes: LoincDocumentAxes


def _entry(
    code: str,
    label: str,
    *,
    type_of_service: str,
    subject_matter_domain: str,
    role: str,
    setting: str,
) -> Mapping[str, Any]:
    """Build one immutable map row without loading an external vocabulary."""

    return MappingProxyType(
        {
            "code": code,
            "label": label,
            "axes": MappingProxyType(
                {
                    "type_of_service": type_of_service,
                    "subject_matter_domain": subject_matter_domain,
                    "role": role,
                    "setting": setting,
                }
            ),
        }
    )


# These are the selected LOINC document-ontology rows used by OpenMed. The
# classifier's canonical labels are deliberately separate from the LOINC
# display labels so existing ``type`` values remain backward compatible.
DOCUMENT_TYPE_TO_LOINC: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "discharge_summary": _entry(
            "18842-5",
            "Discharge summary",
            type_of_service="discharge",
            subject_matter_domain="general medicine",
            role="summary",
            setting="inpatient",
        ),
        "radiology_report": _entry(
            "18748-4",
            "Diagnostic imaging study",
            type_of_service="radiology",
            subject_matter_domain="imaging",
            role="report",
            setting="diagnostic",
        ),
        "pathology_report": _entry(
            "27896-6",
            "Pathology study",
            type_of_service="pathology",
            subject_matter_domain="pathology",
            role="report",
            setting="diagnostic",
        ),
        "progress_note": _entry(
            "11506-3",
            "Progress note",
            type_of_service="clinical",
            subject_matter_domain="general medicine",
            role="progress",
            setting="inpatient",
        ),
        "operative_note": _entry(
            "28570-0",
            "Procedure note",
            type_of_service="surgery",
            subject_matter_domain="procedure",
            role="procedure",
            setting="operating room",
        ),
        "history_and_physical": _entry(
            "34117-2",
            "History and physical note",
            type_of_service="clinical",
            subject_matter_domain="general medicine",
            role="assessment",
            setting="encounter",
        ),
        "consult_note": _entry(
            "11488-4",
            "Consultation note",
            type_of_service="consultation",
            subject_matter_domain="general medicine",
            role="consultation",
            setting="encounter",
        ),
    }
)

DOCUMENT_TYPE_LOINC_MAP = DOCUMENT_TYPE_TO_LOINC

LOINC_DOCUMENT_CODE_MAP: Mapping[str, str] = MappingProxyType(
    {
        document_type: str(mapping["code"])
        for document_type, mapping in DOCUMENT_TYPE_TO_LOINC.items()
    }
)
DOCUMENT_TYPE_TO_LOINC_CODE = LOINC_DOCUMENT_CODE_MAP

LOINC_DOCUMENT_LABEL_MAP: Mapping[str, str] = MappingProxyType(
    {
        document_type: str(mapping["label"])
        for document_type, mapping in DOCUMENT_TYPE_TO_LOINC.items()
    }
)

LOINC_DOCUMENT_SUBSET: tuple[Mapping[str, str], ...] = tuple(
    MappingProxyType({"code": str(mapping["code"]), "label": str(mapping["label"])})
    for mapping in DOCUMENT_TYPE_TO_LOINC.values()
)

LOINC_DOCUMENT_PROVENANCE: Mapping[str, Any] = MappingProxyType(
    {
        "source": "LOINC document ontology selected code/label rows",
        "license": LOINC_DOCUMENT_SUBSET_LICENSE,
        "license_checked": True,
        "restricted_data": False,
        "contains_patient_data": False,
        "full_release_bundled": False,
        "row_count": len(LOINC_DOCUMENT_SUBSET),
    }
)

_DOCUMENT_TYPE_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "discharge summary": "discharge_summary",
        "history and physical": "history_and_physical",
        "history and physical note": "history_and_physical",
        "history physical": "history_and_physical",
        "h and p": "history_and_physical",
        "h&p": "history_and_physical",
        "radiology": "radiology_report",
        "radiology report": "radiology_report",
        "pathology": "pathology_report",
        "pathology report": "pathology_report",
        "progress": "progress_note",
        "progress note": "progress_note",
        "operative": "operative_note",
        "operative note": "operative_note",
        "procedure note": "operative_note",
        "consultation": "consult_note",
        "consultation note": "consult_note",
        "consult note": "consult_note",
    }
)
_DOCUMENT_TYPE_SEPARATOR_RE = re.compile(r"[^a-z0-9]+")


def canonical_document_type(document_type: object) -> str | None:
    """Normalize a classifier label to a supported canonical document type."""

    if not isinstance(document_type, str):
        return None
    normalized = _DOCUMENT_TYPE_SEPARATOR_RE.sub(" ", document_type.casefold()).strip()
    if not normalized:
        return None
    normalized = normalized.replace(" & ", " and ")
    underscored = normalized.replace(" ", "_")
    if underscored in DOCUMENT_TYPE_TO_LOINC:
        return underscored
    return _DOCUMENT_TYPE_ALIASES.get(normalized)


def get_document_type_mapping(
    document_type: object,
    *,
    confidence: float | None = None,
) -> LoincDocumentMapping | None:
    """Return a copy of the selected mapping, or ``None`` as the safe sentinel.

    Args:
        document_type: Canonical classifier type or a supported human-readable
            alias.
        confidence: Optional classifier confidence. Values below
            ``LOINC_MIN_CONFIDENCE`` abstain from code emission.

    Returns:
        A JSON-ready mapping with ``code``, ``label``, and all four axes, or
        ``None`` for an unknown type or low-confidence prediction.
    """

    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return None
        if not isfinite(float(confidence)) or float(confidence) < LOINC_MIN_CONFIDENCE:
            return None

    canonical = canonical_document_type(document_type)
    if canonical is None:
        return None
    mapping = DOCUMENT_TYPE_TO_LOINC.get(canonical)
    if mapping is None:
        return None
    axes = mapping["axes"]
    return {
        "code": str(mapping["code"]),
        "label": str(mapping["label"]),
        "axes": {axis: str(axes[axis]) for axis in LOINC_AXIS_NAMES},
    }


get_loinc_document_mapping = get_document_type_mapping


def document_type_loinc_coverage(
    classifications: Iterable[str | Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize raw-text-free classifier-to-LOINC mapping coverage.

    The public synthetic harness can pass either document-type strings or the
    dictionaries returned by :func:`classify_document`. Only the number and
    names of unmapped types are reported; note text is never retained.
    """

    total = 0
    mapped = 0
    unmapped_types: set[str] = set()
    for item in classifications:
        total += 1
        if isinstance(item, Mapping):
            document_type = item.get("type")
            confidence = item.get("confidence")
        else:
            document_type = item
            confidence = None
        mapping = get_document_type_mapping(
            document_type,
            confidence=confidence if isinstance(confidence, (int, float)) else None,
        )
        if mapping is None:
            mapped_code = item.get("loinc_code") if isinstance(item, Mapping) else None
            if isinstance(mapped_code, str) and mapped_code:
                mapped += 1
            else:
                unmapped_types.add(str(document_type))
        else:
            mapped += 1

    coverage = mapped / total if total else 0.0
    return {
        "total_predictions": total,
        "mapped_predictions": mapped,
        "unmapped_predictions": total - mapped,
        "mapping_coverage": round(coverage, 6),
        "unmapped_types": sorted(unmapped_types),
    }


__all__ = [
    "DOCUMENT_TYPE_LOINC_MAP",
    "DOCUMENT_TYPE_TO_LOINC",
    "DOCUMENT_TYPE_TO_LOINC_CODE",
    "LOINC_AXIS_NAMES",
    "LOINC_DOCUMENT_CODE_MAP",
    "LOINC_DOCUMENT_LABEL_MAP",
    "LOINC_DOCUMENT_PROVENANCE",
    "LOINC_DOCUMENT_SUBSET",
    "LOINC_DOCUMENT_SUBSET_LICENSE",
    "LOINC_DOCUMENT_SUBSET_MAX_ROWS",
    "LOINC_DOCUMENT_SUBSET_NAME",
    "LOINC_DOCUMENT_SYSTEM",
    "LOINC_MIN_CONFIDENCE",
    "LoincDocumentAxes",
    "LoincDocumentMapping",
    "canonical_document_type",
    "document_type_loinc_coverage",
    "get_document_type_mapping",
    "get_loinc_document_mapping",
]
