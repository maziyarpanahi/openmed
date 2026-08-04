"""Assertion-aware FHIR R4 ``Condition`` export.

Materializes an :class:`~openmed.clinical.grounding.assertion_grounding.AssertedGroundedSpan`
into a FHIR R4 ``Condition`` whose ``verificationStatus`` and ``clinicalStatus``
are driven by the span's derived assertion status, so a refuted, hypothetical,
or historical finding is coded truthfully instead of as an active present
concept.

The safety boundary is enforced structurally: a non-patient (family / other
experiencer) span yields no patient ``Condition`` at all
(:func:`to_condition` returns ``None``), and the emitted ``clinicalStatus`` /
``verificationStatus`` come only from the vetted status map, so an
``active`` + ``confirmed`` Condition can never be produced for a denied or
hypothetical span. The concept ``code`` is built from the span's focus
candidate via the shared grounding-aware ``CodeableConcept`` core.
"""

from __future__ import annotations

from typing import Any

from ...grounding.assertion_grounding import AssertedGroundedSpan
from ..codeable_concept import to_codeable_concept

__all__ = [
    "CONDITION_CLINICAL_SYSTEM",
    "CONDITION_VER_STATUS_SYSTEM",
    "to_condition",
]

#: HL7 terminology system URIs for the two status value sets.
CONDITION_CLINICAL_SYSTEM = "http://terminology.hl7.org/CodeSystem/condition-clinical"
CONDITION_VER_STATUS_SYSTEM = (
    "http://terminology.hl7.org/CodeSystem/condition-ver-status"
)


def to_condition(
    asserted: AssertedGroundedSpan,
    *,
    subject_reference: str,
    condition_id: str | None = None,
) -> dict[str, Any] | None:
    """Build a FHIR R4 ``Condition`` for an assertion-aware grounded span.

    The ``code`` is the span's grounding ``CodeableConcept``;
    ``verificationStatus`` and ``clinicalStatus`` are taken from the span's
    :class:`~openmed.clinical.grounding.assertion_grounding.AssertionGroundingStatus`.
    ``clinicalStatus`` is omitted when the status asserts no active/inactive
    state (a refuted or hypothetical finding), matching the FHIR invariant that
    such a status is not claimed.

    Args:
        asserted: The assertion-aware grounded span to materialize.
        subject_reference: The ``Condition.subject`` reference (e.g.
            ``"Patient/123"``) the finding is attributed to.
        condition_id: Optional resource ``id``.

    Returns:
        A ``resourceType="Condition"`` mapping, or ``None`` when the span
        belongs to a non-patient subject and must be excluded from patient
        resources.
    """

    status = asserted.status
    if not status.patient_subject:
        return None

    # Every coding of the span (the focus concept and any composite /
    # post-coordinated qualifier codes) shares the single Condition-level
    # status, so they all inherit the focus concept's assertion.
    resource: dict[str, Any] = {
        "resourceType": "Condition",
        "verificationStatus": _status_concept(
            CONDITION_VER_STATUS_SYSTEM, status.verification_status
        ),
        "subject": {"reference": subject_reference},
        "code": to_codeable_concept(asserted.grounded),
    }
    if condition_id is not None:
        resource["id"] = condition_id
    if status.clinical_status is not None:
        resource["clinicalStatus"] = _status_concept(
            CONDITION_CLINICAL_SYSTEM, status.clinical_status
        )
    return resource


def _status_concept(system: str, code: str) -> dict[str, Any]:
    return {"coding": [{"system": system, "code": code}]}
