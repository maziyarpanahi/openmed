"""OMOP CDM v5.4 ``observation_period`` export."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from openmed.clinical.grounding.types import GroundedSpan
from openmed.interop.omop import deterministic_omop_id

from ._common import (
    concept_id,
    context_value,
    date_value,
    first_context_value,
    foreign_key,
    iter_spans,
)

__all__ = [
    "OBSERVATION_PERIOD_COLUMNS",
    "to_observation_period",
]

OBSERVATION_PERIOD_COLUMNS: tuple[str, ...] = (
    "observation_period_id",
    "person_id",
    "observation_period_start_date",
    "observation_period_end_date",
    "period_type_concept_id",
)

_CLINICAL_DATE_FIELDS: tuple[str, ...] = (
    "observation_period_start_date",
    "observation_period_end_date",
    "visit_start_date",
    "visit_end_date",
    "condition_start_date",
    "condition_end_date",
    "drug_exposure_start_date",
    "drug_exposure_end_date",
    "measurement_date",
    "procedure_date",
    "procedure_end_date",
    "start_date",
    "end_date",
    "date",
    "note_date",
)


def to_observation_period(
    grounded: GroundedSpan | Iterable[GroundedSpan] | None = None,
    *,
    person_id: int | str | None = None,
    document_id: str = "openmed-document",
    note_date: str | date | datetime | None = None,
    observation_period_start_date: str | date | datetime | None = None,
    observation_period_end_date: str | date | datetime | None = None,
    period_type_concept_id: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Emit a bounding CDM v5.4 observation period when dates are available."""

    spans = () if grounded is None else iter_spans(grounded)
    metadata_person = first_context_value(
        spans, "person_id", "patient_id", "subject_id"
    )
    resolved_person_id = foreign_key(
        person_id if person_id is not None else metadata_person,
        namespace="person",
    )

    available_dates = _available_dates(spans, note_date=note_date)
    normalized_start = date_value(observation_period_start_date)
    normalized_end = date_value(observation_period_end_date)
    if normalized_start is None and available_dates:
        normalized_start = min(available_dates)
    if normalized_end is None and available_dates:
        normalized_end = max(available_dates)
    if normalized_start is None and normalized_end is None:
        return ()
    if normalized_start is None:
        normalized_start = normalized_end
    if normalized_end is None:
        normalized_end = normalized_start
    if normalized_start > normalized_end:
        raise ValueError(
            "observation_period_start_date must not be after "
            "observation_period_end_date"
        )

    explicit_row_id = first_context_value(spans, "observation_period_id")
    row_id = foreign_key(explicit_row_id, namespace="observation_period")
    if row_id is None:
        row_id = deterministic_omop_id(
            "observation_period",
            document_id,
            resolved_person_id,
            normalized_start,
            normalized_end,
        )

    row = {
        "observation_period_id": row_id,
        "person_id": resolved_person_id,
        "observation_period_start_date": normalized_start,
        "observation_period_end_date": normalized_end,
        "period_type_concept_id": concept_id(
            period_type_concept_id
            if period_type_concept_id is not None
            else first_context_value(spans, "period_type_concept_id"),
            name="period_type_concept_id",
            default=0,
        ),
    }
    return ({column: row[column] for column in OBSERVATION_PERIOD_COLUMNS},)


def _available_dates(
    spans: tuple[GroundedSpan, ...],
    *,
    note_date: str | date | datetime | None,
) -> tuple[str, ...]:
    values: list[str] = []
    normalized_note_date = date_value(note_date)
    if normalized_note_date is not None:
        values.append(normalized_note_date)
    for span in spans:
        for name in _CLINICAL_DATE_FIELDS:
            normalized = date_value(context_value(span, name))
            if normalized is not None:
                values.append(normalized)
    return tuple(values)
