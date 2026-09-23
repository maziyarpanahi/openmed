"""Assertion-aware FHIR R4 ``AllergyIntolerance`` export."""

from __future__ import annotations

from typing import Any

from ...grounding.assertion_grounding import (
    GROUNDING_HISTORICAL,
    GROUNDING_HYPOTHETICAL,
    GROUNDING_PRESENT,
    GROUNDING_REFUTED,
    AssertedGroundedSpan,
)
from .codeable_concept import to_codeable_concept

__all__ = [
    "ALLERGY_CLINICAL_STATUS_SYSTEM",
    "ALLERGY_VERIFICATION_STATUS_SYSTEM",
    "to_allergy_intolerance",
]

ALLERGY_CLINICAL_STATUS_SYSTEM = (
    "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical"
)
ALLERGY_VERIFICATION_STATUS_SYSTEM = (
    "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification"
)

_CRITICALITY_CODES = frozenset({"low", "high", "unable-to-assess"})
_TYPE_CODES = frozenset({"allergy", "intolerance"})


def to_allergy_intolerance(
    asserted: AssertedGroundedSpan,
    *,
    patient_reference: str,
    allergy_id: str | None = None,
    encounter_reference: str | None = None,
) -> dict[str, Any] | None:
    """Build a FHIR R4 ``AllergyIntolerance`` from a grounded allergen.

    ``clinicalStatus`` and ``verificationStatus`` are derived from the shared
    assertion status. A refuted span is therefore emitted as ``refuted`` and
    never as an active allergy. Optional ``criticality`` and ``allergy_type``
    metadata are copied only when they contain R4 base codes.

    Args:
        asserted: Assertion-aware grounded allergen span.
        patient_reference: Reference for ``AllergyIntolerance.patient``.
        allergy_id: Optional resource id.
        encounter_reference: Optional Encounter reference. When omitted, the
            span's ``encounter_reference`` metadata is used if present.

    Returns:
        An ``AllergyIntolerance`` mapping, or ``None`` for a non-patient span.

    Raises:
        ValueError: If optional coded metadata is not an R4 base code.
    """

    if not asserted.status.patient_subject:
        return None

    resource: dict[str, Any] = {
        "resourceType": "AllergyIntolerance",
        "verificationStatus": _status_concept(
            ALLERGY_VERIFICATION_STATUS_SYSTEM,
            _verification_status(asserted),
        ),
        "code": to_codeable_concept(asserted.grounded),
        "patient": {"reference": patient_reference},
    }
    if allergy_id is not None:
        resource["id"] = allergy_id

    clinical_status = _clinical_status(asserted)
    if clinical_status is not None:
        resource["clinicalStatus"] = _status_concept(
            ALLERGY_CLINICAL_STATUS_SYSTEM,
            clinical_status,
        )

    criticality = asserted.grounded.metadata.get("criticality")
    if criticality is not None:
        resource["criticality"] = _coded_metadata(
            criticality,
            name="criticality",
            allowed=_CRITICALITY_CODES,
        )

    allergy_type = asserted.grounded.metadata.get("allergy_type")
    if allergy_type is not None:
        resource["type"] = _coded_metadata(
            allergy_type,
            name="allergy_type",
            allowed=_TYPE_CODES,
        )

    encounter = encounter_reference or asserted.grounded.metadata.get(
        "encounter_reference"
    )
    if encounter is not None:
        resource["encounter"] = {"reference": _reference(encounter, "encounter")}
    return resource


def _verification_status(asserted: AssertedGroundedSpan) -> str:
    return {
        GROUNDING_PRESENT: "confirmed",
        GROUNDING_HISTORICAL: "confirmed",
        GROUNDING_REFUTED: "refuted",
        GROUNDING_HYPOTHETICAL: "unconfirmed",
    }.get(asserted.status.status, "unconfirmed")


def _clinical_status(asserted: AssertedGroundedSpan) -> str | None:
    return {
        GROUNDING_PRESENT: "active",
        GROUNDING_HISTORICAL: "inactive",
    }.get(asserted.status.status)


def _status_concept(system: str, code: str) -> dict[str, Any]:
    return {"coding": [{"system": system, "code": code}]}


def _coded_metadata(value: Any, *, name: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {tuple(sorted(allowed))!r}")
    return value


def _reference(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}_reference must be a non-empty FHIR reference")
    return value.strip()
