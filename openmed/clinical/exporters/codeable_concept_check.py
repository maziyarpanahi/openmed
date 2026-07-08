"""FHIR R4 ``CodeableConcept`` assembly and structural consistency checks.

This module checks local shape and text/display consistency only. It does not
validate codes, displays, systems, or bindings against a terminology service,
code system, or value set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

from openmed.clinical.normalization import RankedConcept

from .codeable_concept_simple import codeable_concept, coding

__all__ = [
    "CONCEPT_NORMALIZATION_PROVENANCE_EXTENSION_URL",
    "CodeableConceptFinding",
    "CodeableConceptFindingCode",
    "check_codeable_concept",
    "codeable_concept_from_ranked_candidates",
]

CodeableConceptFindingCode = Literal[
    "missing-text-when-codeless",
    "empty-coding-array",
    "text-display-mismatch",
]


class CodeableConceptFinding(TypedDict):
    """JSON-serializable issue shape emitted by ``check_codeable_concept``."""

    finding_code: CodeableConceptFindingCode
    severity: Literal["error", "warning"]
    code: Literal["required", "structure", "invariant"]
    diagnostics: str
    expression: list[str]


_MISSING_TEXT_WHEN_CODELESS = "missing-text-when-codeless"
_EMPTY_CODING_ARRAY = "empty-coding-array"
_TEXT_DISPLAY_MISMATCH = "text-display-mismatch"

_FINDING_ORDER: dict[str, int] = {
    _MISSING_TEXT_WHEN_CODELESS: 0,
    _EMPTY_CODING_ARRAY: 1,
    _TEXT_DISPLAY_MISMATCH: 2,
}

_MISSING_TEXT_DIAGNOSTICS = "CodeableConcept without usable coding must include text."
_EMPTY_CODING_DIAGNOSTICS = (
    "CodeableConcept.coding is present but empty; omit it or include a Coding."
)
_TEXT_DISPLAY_DIAGNOSTICS = (
    "CodeableConcept.text must match at least one coding.display when both are "
    "available, or a coding display must provide the fallback label."
)
CONCEPT_NORMALIZATION_PROVENANCE_EXTENSION_URL = (
    "https://openmed.ai/fhir/StructureDefinition/concept-normalization-provenance"
)


def check_codeable_concept(
    concept: Any,
    *,
    expression: str = "CodeableConcept",
) -> list[CodeableConceptFinding]:
    """Return deterministic structural findings for a ``CodeableConcept``.

    The checker is intentionally tolerant: malformed inputs are treated as
    missing or unusable fields and never raise. Returned findings expose FHIR
    ``severity``/``code``/``diagnostics``/``expression`` fields so callers can
    pass them directly to the shared ``OperationOutcome`` builder. The custom
    ``finding_code`` field carries the stable OpenMed finding identifier.

    Args:
        concept: Candidate FHIR R4 ``CodeableConcept`` mapping.
        expression: FHIRPath-style base expression for the checked element.

    Returns:
        A deterministic list of JSON-serializable finding dictionaries.
    """

    concept_mapping = concept if isinstance(concept, Mapping) else {}
    text = _clean_string(concept_mapping.get("text"))
    coding_value = concept_mapping.get("coding")
    codings = _usable_codings(coding_value)

    findings: list[CodeableConceptFinding] = []
    if _is_empty_coding_array(coding_value):
        findings.append(
            _finding(
                finding_code=_EMPTY_CODING_ARRAY,
                severity="warning",
                code="structure",
                diagnostics=_EMPTY_CODING_DIAGNOSTICS,
                expression=_field_expression(expression, "coding"),
            )
        )

    if not codings and text is None:
        findings.append(
            _finding(
                finding_code=_MISSING_TEXT_WHEN_CODELESS,
                severity="error",
                code="required",
                diagnostics=_MISSING_TEXT_DIAGNOSTICS,
                expression=_field_expression(expression, "text"),
            )
        )

    if codings and _has_text_display_mismatch(text, codings):
        findings.append(
            _finding(
                finding_code=_TEXT_DISPLAY_MISMATCH,
                severity="warning",
                code="invariant",
                diagnostics=_TEXT_DISPLAY_DIAGNOSTICS,
                expression=_field_expression(expression, "text"),
            )
        )

    return sorted(
        findings,
        key=lambda item: (
            _FINDING_ORDER[item["finding_code"]],
            item["expression"],
        ),
    )


def codeable_concept_from_ranked_candidates(
    candidates: Sequence[RankedConcept],
    *,
    text: str | None = None,
    max_codings: int = 5,
) -> dict[str, Any]:
    """Build a FHIR R4 ``CodeableConcept`` from ranked normalized concepts.

    Each emitted ``Coding`` carries the terminology system/code/display plus a
    provenance extension with the mention offsets, confidence, and backend
    identity used to produce the candidate.
    """

    if not candidates:
        raise ValueError("at least one ranked candidate is required")
    if max_codings <= 0:
        raise ValueError("max_codings must be positive")

    codings = [
        _coding_from_ranked_candidate(candidate)
        for candidate in candidates[:max_codings]
    ]
    return codeable_concept(codings, text=text)


def _coding_from_ranked_candidate(candidate: RankedConcept) -> dict[str, Any]:
    concept = candidate.concept
    result = coding(concept.system_uri, concept.code, concept.display)
    if concept.version is not None:
        result["version"] = concept.version
    result["extension"] = [_normalization_provenance_extension(candidate)]
    return result


def _normalization_provenance_extension(
    candidate: RankedConcept,
) -> dict[str, Any]:
    provenance = candidate.provenance
    extensions: list[dict[str, Any]] = [
        {"url": "confidence", "valueDecimal": candidate.confidence},
        {"url": "backendName", "valueString": provenance.backend_name},
        {"url": "backendVersion", "valueString": provenance.backend_version},
    ]
    if provenance.mention_start is not None:
        extensions.append(
            {"url": "mentionStart", "valueInteger": provenance.mention_start}
        )
    if provenance.mention_end is not None:
        extensions.append({"url": "mentionEnd", "valueInteger": provenance.mention_end})

    return {
        "url": CONCEPT_NORMALIZATION_PROVENANCE_EXTENSION_URL,
        "extension": extensions,
    }


def _finding(
    *,
    finding_code: CodeableConceptFindingCode,
    severity: Literal["error", "warning"],
    code: Literal["required", "structure", "invariant"],
    diagnostics: str,
    expression: str,
) -> CodeableConceptFinding:
    """Build one JSON-serializable finding mapping."""

    return {
        "finding_code": finding_code,
        "severity": severity,
        "code": code,
        "diagnostics": diagnostics,
        "expression": [expression] if expression else [],
    }


def _usable_codings(coding_value: Any) -> list[Mapping[str, Any]]:
    """Return mapping-shaped codings from a tolerant ``coding`` value."""

    if isinstance(coding_value, Mapping):
        return [coding_value]
    if isinstance(coding_value, (str, bytes)) or not isinstance(coding_value, Sequence):
        return []
    return [item for item in coding_value if isinstance(item, Mapping)]


def _has_text_display_mismatch(
    text: str | None,
    codings: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether concept text lacks a case-insensitive display match."""

    displays = [
        display
        for coding in codings
        if (display := _clean_string(coding.get("display"))) is not None
    ]
    if text is None:
        return not displays
    normalized_text = _normalize_for_match(text)
    return all(_normalize_for_match(display) != normalized_text for display in displays)


def _is_empty_coding_array(value: Any) -> bool:
    """Return whether ``coding`` is explicitly present as an empty array."""

    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and not value
    )


def _clean_string(value: Any) -> str | None:
    """Return a stripped non-empty string, else ``None``."""

    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _normalize_for_match(value: str) -> str:
    """Normalize a human label for case-insensitive comparison."""

    return value.casefold()


def _field_expression(base: Any, field: str) -> str:
    """Append ``field`` to a caller-supplied FHIRPath base expression."""

    base_expression = base.strip() if isinstance(base, str) else "CodeableConcept"
    if not base_expression:
        return field
    return f"{base_expression}.{field}"
