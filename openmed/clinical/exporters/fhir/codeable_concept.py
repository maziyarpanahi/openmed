"""FHIR R4 CodeableConcept emission for grounded clinical concepts.

The grounded emitter is deliberately assist-only. Every emitted concept carries
the source evidence offsets and an explicit no-autonomous-decision disclaimer,
while every Coding records how its linker candidate was selected. Validated
SNOMED CT post-coordination remains supported through the existing helpers in
this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ...grounding.ecl import ECLValidator
from ...grounding.postcoordination import (
    SnomedExpression,
    is_postcoordinated_candidate,
)
from ...grounding.types import Candidate, GroundedSpan
from ..codeable_concept_simple import codeable_concept

__all__ = [
    "GROUNDED_CODE_PROVENANCE_EXTENSION_URL",
    "MEDICAL_DEVICE_ASSIST_EXTENSION_URL",
    "MEDICAL_DEVICE_ASSIST_ONLY_DISCLAIMER",
    "POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL",
    "postcoordinated_codeable_concept",
    "stamp_postcoordination_provenance",
    "to_codeable_concept",
]

GROUNDED_CODE_PROVENANCE_EXTENSION_URL = (
    "https://openmed.ai/fhir/StructureDefinition/grounded-code-provenance"
)
MEDICAL_DEVICE_ASSIST_EXTENSION_URL = (
    "https://openmed.ai/fhir/StructureDefinition/medical-device-assist"
)
MEDICAL_DEVICE_ASSIST_ONLY_DISCLAIMER = (
    "Assist-only terminology grounding for human review; not an autonomous "
    "clinical coding, diagnosis, treatment, or billing decision."
)
POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL = (
    "https://openmed.ai/fhir/StructureDefinition/snomed-postcoordination-provenance"
)
_SNOMED_URI = "http://snomed.info/sct"
_UNAVAILABLE_PROVENANCE = "unavailable"


def to_codeable_concept(ranked_concept: GroundedSpan) -> dict[str, Any]:
    """Emit one deterministic FHIR R4 CodeableConcept for a grounded span.

    The input candidates are already ranked by the grounding layer. At most the
    first candidate for each terminology system is emitted. Each resulting
    Coding contains ``system``, ``code``, and ``display`` plus an auditable
    provenance extension with the linker, score, matched alias, vocabulary
    version, and source offsets. The CodeableConcept itself always carries an
    assist-only disclaimer and the same evidence boundary.

    Args:
        ranked_concept: Grounded source span with ranked terminology candidates
            and inclusive/exclusive character offsets.

    Returns:
        A JSON-serializable FHIR R4 CodeableConcept. Abstained or uncoded spans
        produce a text-only concept with the assist-only extension.

    Raises:
        TypeError: If ``ranked_concept`` is not a GroundedSpan.
        ValueError: If the evidence offsets are empty, a candidate score is not
            finite, or a candidate uses an unknown grounding system.
    """

    if not isinstance(ranked_concept, GroundedSpan):
        raise TypeError("ranked_concept must be a GroundedSpan")
    _require_evidence_offsets(ranked_concept)

    candidates = (
        ()
        if ranked_concept.abstained
        else _one_candidate_per_system(ranked_concept.candidates)
    )
    if candidates:
        concept = codeable_concept(
            [
                _coding_with_provenance(candidate, ranked_concept)
                for candidate in candidates
            ],
            text=ranked_concept.text,
        )
    else:
        concept = {"text": ranked_concept.text}
    concept["extension"] = [_assist_only_extension(ranked_concept)]
    return concept


def _one_candidate_per_system(
    candidates: tuple[Candidate, ...],
) -> tuple[Candidate, ...]:
    selected: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        system = candidate.system.strip().casefold()
        if system in seen:
            continue
        seen.add(system)
        selected.append(candidate)
    return tuple(selected)


def _coding_with_provenance(
    candidate: Candidate,
    grounded_span: GroundedSpan,
) -> dict[str, Any]:
    if not math.isfinite(float(candidate.score)):
        raise ValueError("grounding candidate score must be finite")
    coding: dict[str, Any] = {
        "system": _grounding_system_uri(candidate.system),
        "code": candidate.code,
        "display": candidate.display,
        "extension": [_grounding_provenance_extension(candidate, grounded_span)],
    }
    if candidate.vocab_version:
        coding["version"] = candidate.vocab_version
    if is_postcoordinated_candidate(candidate):
        if not candidate.vocab_version:
            raise ValueError(
                "post-coordinated SNOMED candidate requires an edition version"
            )
        coding = stamp_postcoordination_provenance(
            coding,
            edition_uri=candidate.vocab_version,
        )
    return coding


def _grounding_system_uri(system: str) -> str:
    # Import lazily because the shared grounding module delegates validated
    # post-coordinated Coding stamping back to this module.
    from ..codeable_concept import SYSTEM_URI

    try:
        return SYSTEM_URI[system.strip().upper()]
    except KeyError:
        raise ValueError(
            f"Unknown grounding system {system!r}. Known: {sorted(SYSTEM_URI)}."
        ) from None


def _grounding_provenance_extension(
    candidate: Candidate,
    grounded_span: GroundedSpan,
) -> dict[str, Any]:
    return {
        "url": GROUNDED_CODE_PROVENANCE_EXTENSION_URL,
        "extension": [
            {
                "url": "linker",
                "valueString": candidate.source or _UNAVAILABLE_PROVENANCE,
            },
            {"url": "score", "valueDecimal": float(candidate.score)},
            {
                "url": "matched_alias",
                "valueString": candidate.matched_alias or _UNAVAILABLE_PROVENANCE,
            },
            {
                "url": "vocab_version",
                "valueString": candidate.vocab_version or _UNAVAILABLE_PROVENANCE,
            },
            {"url": "evidence_start", "valueUnsignedInt": grounded_span.start},
            {"url": "evidence_end", "valueUnsignedInt": grounded_span.end},
        ],
    }


def _assist_only_extension(grounded_span: GroundedSpan) -> dict[str, Any]:
    return {
        "url": MEDICAL_DEVICE_ASSIST_EXTENSION_URL,
        "extension": [
            {"url": "assist_only", "valueBoolean": True},
            {"url": "autonomous_decision", "valueBoolean": False},
            {"url": "evidence_start", "valueUnsignedInt": grounded_span.start},
            {"url": "evidence_end", "valueUnsignedInt": grounded_span.end},
            {
                "url": "disclaimer",
                "valueString": MEDICAL_DEVICE_ASSIST_ONLY_DISCLAIMER,
            },
        ],
    }


def _require_evidence_offsets(grounded_span: GroundedSpan) -> None:
    if grounded_span.end <= grounded_span.start:
        raise ValueError(
            "FHIR CodeableConcept emission requires non-empty evidence offsets"
        )


def stamp_postcoordination_provenance(
    coding: Mapping[str, Any],
    *,
    edition_uri: str,
) -> dict[str, Any]:
    """Mark a SNOMED ``Coding`` as validated and composed, not looked up."""

    if coding.get("system") != _SNOMED_URI:
        raise ValueError("post-coordination provenance requires a SNOMED Coding")
    if not isinstance(edition_uri, str) or not edition_uri.strip():
        raise ValueError("post-coordinated Coding requires a caller edition URI")
    result = deepcopy(dict(coding))
    result["version"] = edition_uri.strip()
    provenance = {
        "url": POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL,
        "extension": [
            {"url": "origin", "valueCode": "composed"},
            {"url": "eclValidated", "valueBoolean": True},
        ],
    }
    extensions = result.get("extension")
    if extensions is None:
        result["extension"] = [provenance]
        return result
    if not isinstance(extensions, list):
        raise TypeError("Coding.extension must be a list when present")
    result["extension"] = [
        extension
        for extension in extensions
        if not (
            isinstance(extension, Mapping)
            and extension.get("url") == POSTCOORDINATED_CODING_PROVENANCE_EXTENSION_URL
        )
    ]
    result["extension"].append(provenance)
    return result


def postcoordinated_codeable_concept(
    expression: SnomedExpression,
    *,
    validator: ECLValidator,
    text: str | None = None,
    display: str | None = None,
) -> dict[str, Any]:
    """Validate and build a CodeableConcept containing a composed expression."""

    if not isinstance(expression, SnomedExpression):
        raise TypeError("expression must be a SnomedExpression")
    if not isinstance(validator, ECLValidator):
        raise TypeError("validator must be an edition-backed ECLValidator")
    validator.require_valid(expression)
    coding: dict[str, Any] = {
        "system": _SNOMED_URI,
        "code": expression.to_scg(),
        "userSelected": False,
    }
    if display is not None:
        coding["display"] = display
    coding = stamp_postcoordination_provenance(
        coding,
        edition_uri=validator.edition_uri,
    )
    return codeable_concept([coding], text=text)
