"""Experiencer-aware patient-record span filter (OM-304).

This layer consumes the per-span clinical assertion records produced by the
ConText composition layer and partitions spans into the set eligible for the
patient's record and the set that must not reach it.

The inclusion policy is intentionally hard at the experiencer boundary:

* ``experiencer`` must be ``patient`` (or unset) for a span to be included.
  ``family`` and ``other`` experiencers are excluded from the patient record.
* ``temporality == hypothetical`` spans are excluded as not asserted present.
* ``negation == negated`` patient spans are *kept* and marked ``refuted`` so
  downstream grounding can emit ``verificationStatus=refuted``.

Every input span appears in exactly one output set. Excluded spans carry an
auditable ``exclusion_reason``; included negated spans carry ``record_status``
set to ``refuted``; all other included spans carry ``record_status`` set to
``recorded``.

This filter is a deterministic record-construction aid, not a clinical
determination.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .context import HYPOTHETICAL, NEGATED, PATIENT_EXPERIENCER, ClinicalAssertion

PATIENT_RECORD_FILTER_ADVISORY = (
    "Patient-record span filtering is a deterministic, advisory record-"
    "construction aid. It sharpens which extracted spans are eligible for the "
    "patient's record but is not a clinical decision, diagnosis, or substitute "
    "for qualified review."
)


@dataclass(frozen=True)
class PatientRecordSpan:
    """One span and its assertion after patient-record filtering.

    ``record_status`` is ``recorded`` for an affirmed patient span and
    ``refuted`` for a negated patient span. ``exclusion_reason`` is set only
    for excluded spans, and ``record_status`` is ``None`` in that case.
    """

    span: Any
    assertion: ClinicalAssertion
    record_status: str | None = None
    exclusion_reason: str | None = None


def filter_patient_record(
    spans: Iterable[Any],
    assertions: Iterable[ClinicalAssertion],
) -> tuple[list[PatientRecordSpan], list[PatientRecordSpan]]:
    """Partition ``spans`` into patient-record eligible and excluded lists.

    Each span is paired with the assertion at the same position in
    ``assertions``. The filter applies the documented inclusion policy.

    Args:
        spans: Caller-supplied span objects, in input order.
        assertions: Per-span :class:`~openmed.clinical.context.ClinicalAssertion`
            records aligned 1:1 with ``spans``.

    Returns:
        A tuple ``(included, excluded)`` of :class:`PatientRecordSpan` lists.

    Raises:
        ValueError: If ``spans`` and ``assertions`` have different lengths.
    """

    span_list = list(spans)
    assertion_list = list(assertions)
    if len(span_list) != len(assertion_list):
        raise ValueError(
            f"spans and assertions must have the same length, got "
            f"{len(span_list)} spans and {len(assertion_list)} assertions"
        )

    included: list[PatientRecordSpan] = []
    excluded: list[PatientRecordSpan] = []

    for span, assertion in zip(span_list, assertion_list):
        experiencer = assertion.experiencer or PATIENT_EXPERIENCER

        if experiencer != PATIENT_EXPERIENCER:
            excluded.append(
                PatientRecordSpan(
                    span=span,
                    assertion=assertion,
                    exclusion_reason="non-patient experiencer",
                )
            )
        elif assertion.temporality == HYPOTHETICAL:
            excluded.append(
                PatientRecordSpan(
                    span=span,
                    assertion=assertion,
                    exclusion_reason="hypothetical",
                )
            )
        elif assertion.negation == NEGATED:
            included.append(
                PatientRecordSpan(
                    span=span,
                    assertion=assertion,
                    record_status="refuted",
                )
            )
        else:
            included.append(
                PatientRecordSpan(
                    span=span,
                    assertion=assertion,
                    record_status="recorded",
                )
            )

    return included, excluded


__all__ = [
    "PATIENT_RECORD_FILTER_ADVISORY",
    "PatientRecordSpan",
    "filter_patient_record",
]
