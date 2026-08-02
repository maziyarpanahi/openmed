"""Grounding-aware FHIR CodeableConcept core.

Converts a grounded span (its surface text, char offsets, and per-system linker
:class:`~openmed.clinical.grounding.Candidate` codes) into a canonical FHIR R4
``CodeableConcept`` with the correct HL7 system URIs, plus a reverse
``(system, code) -> source offsets`` index for code->span highlighting in review
UIs. This is the shared foundation the per-resource FHIR exporters consume.

Mechanical Coding/CodeableConcept shaping and deterministic ordering are reused
from :mod:`.codeable_concept_simple` (the single source of truth for vocabulary
id -> HL7 system URI); this module maps the grounding linker system tokens
(``RXNORM``/``ICD10CM``/...) onto those URIs and adds UMLS.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from openmed.clinical.grounding.types import GroundedSpan

from .code_provenance import stamp_coding_provenance
from .codeable_concept_simple import codeable_concept as _build_codeable_concept
from .codeable_concept_simple import system_uri as _system_uri

# Grounding linker system token -> canonical HL7 FHIR R4 system URI. The shared
# vocabularies derive from the single-source-of-truth map; UMLS is not in it.
SYSTEM_URI: dict[str, str] = {
    "RXNORM": _system_uri("rxnorm"),
    "ICD10CM": _system_uri("icd-10-cm"),
    "ICD11": _system_uri("icd-11-mms"),
    "LOINC": _system_uri("loinc"),
    "SNOMED": _system_uri("snomed"),
    "HPO": _system_uri("hpo"),
    "MESH": _system_uri("mesh"),
    "UMLS": "http://terminology.hl7.org/CodeSystem/umls",
}

__all__ = ["SYSTEM_URI", "GroundedSpan", "to_codeable_concept", "build_reverse_index"]


def to_codeable_concept(grounded_span: GroundedSpan) -> dict[str, Any]:
    """Build a FHIR R4 ``CodeableConcept`` for a grounded span.

    Each candidate becomes a ``Coding`` with the canonical HL7 system URI, code,
    display, and an internal ``_score`` (the linker score, for downstream
    filtering). Candidates carrying a vocabulary version are stamped through
    the shared code-provenance path. Codings are ordered deterministically by
    the shared system priority; ``.text`` is the source surface. A span with no
    candidates yields a text-only concept.
    """
    if grounded_span.abstained or not grounded_span.candidates:
        result = {"text": grounded_span.text}
        grounding = _grounding_provenance(grounded_span)
        if grounding:
            result["_grounding"] = grounding
        return result

    codings = []
    for candidate in grounded_span.candidates:
        coding = {
            "system": _uri_for(candidate.system),
            "code": candidate.code,
            "display": candidate.display,
            "_score": float(candidate.score),
            **_candidate_calibration_fields(grounded_span),
        }
        if candidate.vocab_version:
            coding = stamp_coding_provenance(
                coding,
                {coding["system"]: candidate.vocab_version},
                source_label="grounding candidate",
            )
        codings.append(coding)
    result = _build_codeable_concept(codings, text=grounded_span.text)
    grounding = _grounding_provenance(grounded_span)
    if grounding:
        result["_grounding"] = grounding
    return result


def build_reverse_index(
    grounded_spans: Iterable[GroundedSpan],
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Map ``(system_uri, code)`` to the source ``(start, end)`` offsets.

    Enables code->span highlighting in review UIs. Offsets accumulate in span
    order so the result is deterministic.
    """
    index: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for span in grounded_spans:
        if span.abstained:
            continue
        for candidate in span.candidates:
            key = (_uri_for(candidate.system), candidate.code)
            index.setdefault(key, []).append((span.start, span.end))
    return index


def _uri_for(system: str) -> str:
    try:
        return SYSTEM_URI[system]
    except KeyError:
        raise ValueError(
            f"Unknown grounding system {system!r}. Known: {sorted(SYSTEM_URI)}."
        ) from None


def _candidate_calibration_fields(grounded_span: GroundedSpan) -> dict[str, Any]:
    has_provenance = bool(_grounding_provenance(grounded_span))
    if (
        grounded_span.calibrated_score is None
        and not grounded_span.abstained
        and not has_provenance
    ):
        return {}
    fields: dict[str, Any] = {"_abstained": bool(grounded_span.abstained)}
    if grounded_span.calibrated_score is not None:
        fields["_calibrated_score"] = float(grounded_span.calibrated_score)
    return fields


def _grounding_provenance(grounded_span: GroundedSpan) -> dict[str, Any]:
    provenance = (
        dict(grounded_span.provenance)
        if isinstance(grounded_span.provenance, Mapping)
        else {}
    )
    calibration = provenance.get("grounding_calibration")
    result: dict[str, Any] = {}
    if isinstance(calibration, Mapping):
        result.update(dict(calibration))
    if grounded_span.calibrated_score is not None:
        result.setdefault("calibrated_score", float(grounded_span.calibrated_score))
    if grounded_span.abstained:
        result["abstained"] = True
    elif result:
        result.setdefault("abstained", False)
    if result:
        result.setdefault("candidate_count", len(grounded_span.candidates))
    return result
