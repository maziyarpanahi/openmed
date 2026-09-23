"""Assertion-aware FHIR R4 ``Observation`` export."""

from __future__ import annotations

from typing import Any

from ...grounding.assertion_grounding import (
    GROUNDING_HYPOTHETICAL,
    GROUNDING_REFUTED,
    AssertedGroundedSpan,
)
from ..codeable_concept import to_codeable_concept

__all__ = ["to_observation"]


def to_observation(
    asserted: AssertedGroundedSpan,
    *,
    subject_reference: str,
    observation_id: str | None = None,
    value: Any = None,
    unit: str | None = None,
) -> dict[str, Any] | None:
    """Build a FHIR R4 ``Observation`` for an asserted grounded span.

    The span's grounded candidates become the Observation ``code`` through
    the shared FHIR CodeableConcept core. Numeric values are represented as a
    FHIR ``valueQuantity`` and preserve an optional UCUM unit; booleans and
    other values use their corresponding simple ``value[x]`` representation.
    A refuted or hypothetical observation is marked ``cancelled`` rather than
    being presented as a final result.

    Args:
        asserted: Assertion-aware grounded span to materialize.
        subject_reference: The Observation subject reference.
        observation_id: Optional resource ``id``.
        value: Optional observation result value.
        unit: Optional UCUM unit display/code for numeric values.

    Returns:
        A ``resourceType="Observation"`` mapping, or ``None`` for a finding
        attributed to a non-patient experiencer.
    """

    if not asserted.status.patient_subject:
        return None

    if value is None:
        value = asserted.grounded.metadata.get("value")
    if unit is None:
        metadata_unit = asserted.grounded.metadata.get("unit")
        if isinstance(metadata_unit, str):
            unit = metadata_unit

    resource: dict[str, Any] = {
        "resourceType": "Observation",
        "status": _observation_status(asserted),
        "code": to_codeable_concept(asserted.grounded),
        "subject": {"reference": subject_reference},
    }
    if observation_id is not None:
        resource["id"] = observation_id
    if value is not None:
        _add_value(resource, value, unit)
    return resource


def _add_value(resource: dict[str, Any], value: Any, unit: str | None) -> None:
    if isinstance(value, bool):
        resource["valueBoolean"] = value
    elif isinstance(value, (int, float)):
        quantity: dict[str, Any] = {"value": value}
        if unit:
            quantity.update(
                {
                    "unit": unit,
                    "system": "http://unitsofmeasure.org",
                    "code": unit,
                }
            )
        resource["valueQuantity"] = quantity
    else:
        resource["valueString"] = str(value)


def _observation_status(asserted: AssertedGroundedSpan) -> str:
    if asserted.status.status in {GROUNDING_REFUTED, GROUNDING_HYPOTHETICAL}:
        return "cancelled"
    return "final"
