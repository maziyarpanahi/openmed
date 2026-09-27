"""Assertion-aware FHIR R4 ``Encounter`` export."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...grounding.assertion_grounding import (
    GROUNDING_HISTORICAL,
    GROUNDING_HYPOTHETICAL,
    GROUNDING_PRESENT,
    GROUNDING_REFUTED,
    AssertedGroundedSpan,
)
from .codeable_concept import to_codeable_concept

__all__ = ["ENCOUNTER_CLASS_SYSTEM", "to_encounter"]

ENCOUNTER_CLASS_SYSTEM = "http://terminology.hl7.org/CodeSystem/v3-ActCode"


def to_encounter(
    asserted: AssertedGroundedSpan,
    *,
    subject_reference: str,
    encounter_id: str | None = None,
) -> dict[str, Any] | None:
    """Build a FHIR R4 ``Encounter`` from grounded visit context.

    R4 requires ``Encounter.class``. Callers provide it through the span's
    ``encounter_class`` metadata as either an ActCode string (for example
    ``"AMB"``) or a ``Coding``-like mapping containing ``system`` and ``code``.
    Optional ``period`` metadata may contain ``start`` and/or ``end`` strings.
    The grounded concept becomes ``Encounter.type`` when the span is coded.

    Args:
        asserted: Assertion-aware grounded encounter span.
        subject_reference: Reference for ``Encounter.subject``.
        encounter_id: Optional resource id used by other resources' references.

    Returns:
        An ``Encounter`` mapping, or ``None`` for a non-patient span.

    Raises:
        ValueError: If required class metadata or optional period metadata is
            missing or malformed.
    """

    if not asserted.status.patient_subject:
        return None

    resource: dict[str, Any] = {
        "resourceType": "Encounter",
        "status": _status(asserted),
        "class": _encounter_class(asserted.grounded.metadata),
        "subject": {"reference": subject_reference},
    }
    if encounter_id is not None:
        resource["id"] = encounter_id
    if asserted.grounded.candidates:
        resource["type"] = [to_codeable_concept(asserted.grounded)]

    period = asserted.grounded.metadata.get("period")
    if period is not None:
        resource["period"] = _period(period)
    return resource


def _status(asserted: AssertedGroundedSpan) -> str:
    return {
        GROUNDING_PRESENT: "in-progress",
        GROUNDING_HISTORICAL: "finished",
        GROUNDING_REFUTED: "cancelled",
        GROUNDING_HYPOTHETICAL: "planned",
    }.get(asserted.status.status, "unknown")


def _encounter_class(metadata: Mapping[str, Any]) -> dict[str, Any]:
    value = metadata.get("encounter_class")
    if isinstance(value, str) and value.strip():
        return {"system": ENCOUNTER_CLASS_SYSTEM, "code": value.strip()}
    if isinstance(value, Mapping):
        system = value.get("system")
        code = value.get("code")
        if not isinstance(system, str) or not system.strip():
            raise ValueError("encounter_class.system must be a non-empty string")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("encounter_class.code must be a non-empty string")
        result = {"system": system.strip(), "code": code.strip()}
        display = value.get("display")
        if display is not None:
            if not isinstance(display, str):
                raise ValueError("encounter_class.display must be a string")
            result["display"] = display
        return result
    raise ValueError(
        "Encounter export requires encounter_class metadata as a code or Coding"
    )


def _period(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("period metadata must be a mapping")
    period: dict[str, str] = {}
    for key in ("start", "end"):
        item = value.get(key)
        if item is not None:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"period.{key} must be a non-empty string")
            period[key] = item.strip()
    if not period:
        raise ValueError("period metadata must contain start or end")
    return period
