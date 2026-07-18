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

from dataclasses import dataclass
from typing import Any, Iterable

from openmed.clinical.grounding import Candidate

from .codeable_concept_simple import codeable_concept as _build_codeable_concept
from .codeable_concept_simple import system_uri as _system_uri

# Grounding linker system token -> canonical HL7 FHIR R4 system URI. The shared
# vocabularies derive from the single-source-of-truth map; UMLS is not in it.
SYSTEM_URI: dict[str, str] = {
    "RXNORM": _system_uri("rxnorm"),
    "ICD10CM": _system_uri("icd-10-cm"),
    "LOINC": _system_uri("loinc"),
    "SNOMED": _system_uri("snomed"),
    "HPO": _system_uri("hpo"),
    "UMLS": "http://terminology.hl7.org/CodeSystem/umls",
}

__all__ = ["SYSTEM_URI", "GroundedSpan", "to_codeable_concept", "build_reverse_index"]


@dataclass(frozen=True)
class GroundedSpan:
    """A source span with its grounding candidates.

    ``text`` is the source surface (used as ``CodeableConcept.text``),
    ``start``/``end`` are character offsets into the source document, and
    ``candidates`` are the per-system linker results.
    """

    text: str
    start: int
    end: int
    candidates: tuple[Candidate, ...] = ()


def to_codeable_concept(grounded_span: GroundedSpan) -> dict[str, Any]:
    """Build a FHIR R4 ``CodeableConcept`` for a grounded span.

    Each candidate becomes a ``Coding`` with the canonical HL7 system URI, code,
    display, and an extension-free internal ``_score`` (the linker score, for
    downstream filtering). Codings are ordered deterministically by the shared
    system priority; ``.text`` is the source surface. A span with no candidates
    yields a text-only concept.
    """
    if not grounded_span.candidates:
        return {"text": grounded_span.text}

    codings = [
        {
            "system": _uri_for(candidate.system),
            "code": candidate.code,
            "display": candidate.display,
            "_score": float(candidate.score),
        }
        for candidate in grounded_span.candidates
    ]
    return _build_codeable_concept(codings, text=grounded_span.text)


def build_reverse_index(
    grounded_spans: Iterable[GroundedSpan],
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Map ``(system_uri, code)`` to the source ``(start, end)`` offsets.

    Enables code->span highlighting in review UIs. Offsets accumulate in span
    order so the result is deterministic.
    """
    index: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for span in grounded_spans:
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
