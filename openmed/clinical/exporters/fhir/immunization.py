"""Assertion-aware FHIR R4 ``Immunization`` export."""

from __future__ import annotations

from typing import Any

from ...grounding.assertion_grounding import (
    GROUNDING_HISTORICAL,
    GROUNDING_PRESENT,
    AssertedGroundedSpan,
)
from .codeable_concept import to_codeable_concept

__all__ = ["to_immunization"]


def to_immunization(
    asserted: AssertedGroundedSpan,
    *,
    patient_reference: str,
    immunization_id: str | None = None,
    encounter_reference: str | None = None,
) -> dict[str, Any] | None:
    """Build a FHIR R4 ``Immunization`` from a grounded vaccine span.

    Affirmed present or historical administrations map to ``completed``;
    other assertion states map conservatively to ``not-done``. The required
    occurrence is taken from ``occurrence_datetime`` or ``occurrence`` span
    metadata. If no occurrence was extracted, ``occurrenceString`` is set to
    ``"unknown"`` rather than inventing a date.

    Args:
        asserted: Assertion-aware grounded vaccine span.
        patient_reference: Reference for ``Immunization.patient``.
        immunization_id: Optional resource id.
        encounter_reference: Optional Encounter reference. When omitted, the
            span's ``encounter_reference`` metadata is used if present.

    Returns:
        An ``Immunization`` mapping, or ``None`` for a non-patient span.

    Raises:
        TypeError: If optional occurrence, lot, or source metadata has an
            unsupported type.
    """

    if not asserted.status.patient_subject:
        return None

    resource: dict[str, Any] = {
        "resourceType": "Immunization",
        "status": _status(asserted),
        "vaccineCode": to_codeable_concept(asserted.grounded),
        "patient": {"reference": patient_reference},
    }
    if immunization_id is not None:
        resource["id"] = immunization_id
    _add_occurrence(resource, asserted.grounded.metadata)

    encounter = encounter_reference or asserted.grounded.metadata.get(
        "encounter_reference"
    )
    if encounter is not None:
        resource["encounter"] = {"reference": _reference(encounter, "encounter")}

    lot_number = asserted.grounded.metadata.get("lot_number")
    if lot_number is not None:
        if not isinstance(lot_number, str):
            raise TypeError("lot_number metadata must be a string")
        resource["lotNumber"] = lot_number

    primary_source = asserted.grounded.metadata.get("primary_source")
    if primary_source is not None:
        if type(primary_source) is not bool:
            raise TypeError("primary_source metadata must be a boolean")
        resource["primarySource"] = primary_source
    return resource


def _status(asserted: AssertedGroundedSpan) -> str:
    if asserted.status.status in {GROUNDING_PRESENT, GROUNDING_HISTORICAL}:
        return "completed"
    return "not-done"


def _add_occurrence(resource: dict[str, Any], metadata: Any) -> None:
    date_time = metadata.get("occurrence_datetime")
    if date_time is not None:
        if not isinstance(date_time, str) or not date_time.strip():
            raise TypeError("occurrence_datetime metadata must be a non-empty string")
        resource["occurrenceDateTime"] = date_time.strip()
        return

    occurrence = metadata.get("occurrence")
    if occurrence is None:
        resource["occurrenceString"] = "unknown"
        return
    if not isinstance(occurrence, str) or not occurrence.strip():
        raise TypeError("occurrence metadata must be a non-empty string")
    resource["occurrenceString"] = occurrence.strip()


def _reference(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}_reference must be a non-empty FHIR reference")
    return value.strip()
