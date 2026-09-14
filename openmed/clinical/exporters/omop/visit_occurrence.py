"""OMOP CDM v5.4 ``visit_occurrence`` export."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from openmed.clinical.grounding.types import GroundedSpan
from openmed.interop.omop import deterministic_omop_id

from ._common import (
    concept_id,
    date_value,
    first_context_value,
    foreign_key,
    iter_spans,
)

__all__ = [
    "VISIT_OCCURRENCE_COLUMNS",
    "to_visit_occurrence",
]

VISIT_OCCURRENCE_COLUMNS: tuple[str, ...] = (
    "visit_occurrence_id",
    "person_id",
    "visit_concept_id",
    "visit_start_date",
    "visit_start_datetime",
    "visit_end_date",
    "visit_end_datetime",
    "visit_type_concept_id",
    "provider_id",
    "care_site_id",
    "visit_source_value",
    "visit_source_concept_id",
    "admitted_from_concept_id",
    "admitted_from_source_value",
    "discharged_to_concept_id",
    "discharged_to_source_value",
    "preceding_visit_occurrence_id",
)


def to_visit_occurrence(
    grounded: GroundedSpan | Iterable[GroundedSpan] | None = None,
    *,
    person_id: int | str | None = None,
    visit_occurrence_id: int | str | None = None,
    visit_id: int | str | None = None,
    document_id: str = "openmed-document",
    note_date: str | date | datetime | None = None,
    visit_start_date: str | date | datetime | None = None,
    visit_end_date: str | date | datetime | None = None,
    visit_concept_id: int | None = None,
    visit_type_concept_id: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Emit one document/encounter-scoped CDM v5.4 visit row.

    Caller-provided numeric IDs are preserved. String person and encounter IDs
    receive deterministic local surrogates, and an omitted encounter ID is
    derived from the document, person, and visit dates. Optional fields can be
    supplied through grounded span metadata.
    """

    spans = () if grounded is None else iter_spans(grounded)
    metadata_person = first_context_value(
        spans, "person_id", "patient_id", "subject_id"
    )
    resolved_person_id = foreign_key(
        person_id if person_id is not None else metadata_person,
        namespace="person",
    )

    explicit_visit = (
        visit_occurrence_id
        if visit_occurrence_id is not None
        else visit_id
        if visit_id is not None
        else first_context_value(
            spans,
            "visit_occurrence_id",
            "visit_id",
            "encounter_id",
        )
    )
    resolved_visit_id = foreign_key(explicit_visit, namespace="visit_occurrence")

    started_on = visit_start_date
    if started_on is None:
        started_on = first_context_value(
            spans,
            "visit_start_date",
            "encounter_start_date",
            "start_date",
            "date",
            "note_date",
        )
    if started_on is None:
        started_on = note_date
    ended_on = visit_end_date
    if ended_on is None:
        ended_on = first_context_value(
            spans,
            "visit_end_date",
            "encounter_end_date",
            "end_date",
        )
    if ended_on is None:
        ended_on = started_on
    normalized_start = date_value(started_on)
    normalized_end = date_value(ended_on)
    if normalized_start is not None and normalized_end is not None:
        if normalized_start > normalized_end:
            raise ValueError("visit_start_date must not be after visit_end_date")

    if resolved_visit_id is None:
        resolved_visit_id = deterministic_omop_id(
            "visit_occurrence",
            document_id,
            resolved_person_id,
            normalized_start,
            normalized_end,
        )

    source_value = first_context_value(
        spans, "visit_source_value", "encounter_source_value"
    )
    source_concept_default = 0 if source_value is not None else None
    row = {
        "visit_occurrence_id": resolved_visit_id,
        "person_id": resolved_person_id,
        "visit_concept_id": concept_id(
            visit_concept_id
            if visit_concept_id is not None
            else first_context_value(spans, "visit_concept_id"),
            name="visit_concept_id",
            default=0,
        ),
        "visit_start_date": normalized_start,
        "visit_start_datetime": date_value(
            first_context_value(
                spans,
                "visit_start_datetime",
                "encounter_start_datetime",
                "start_datetime",
            )
        ),
        "visit_end_date": normalized_end,
        "visit_end_datetime": date_value(
            first_context_value(
                spans,
                "visit_end_datetime",
                "encounter_end_datetime",
                "end_datetime",
            )
        ),
        "visit_type_concept_id": concept_id(
            visit_type_concept_id
            if visit_type_concept_id is not None
            else first_context_value(spans, "visit_type_concept_id"),
            name="visit_type_concept_id",
            default=0,
        ),
        "provider_id": foreign_key(
            first_context_value(spans, "provider_id"), namespace="provider"
        ),
        "care_site_id": foreign_key(
            first_context_value(spans, "care_site_id"), namespace="care_site"
        ),
        "visit_source_value": _optional_string(source_value),
        "visit_source_concept_id": concept_id(
            first_context_value(spans, "visit_source_concept_id"),
            name="visit_source_concept_id",
            default=source_concept_default,
        ),
        "admitted_from_concept_id": concept_id(
            first_context_value(spans, "admitted_from_concept_id"),
            name="admitted_from_concept_id",
        ),
        "admitted_from_source_value": _optional_string(
            first_context_value(spans, "admitted_from_source_value")
        ),
        "discharged_to_concept_id": concept_id(
            first_context_value(spans, "discharged_to_concept_id"),
            name="discharged_to_concept_id",
        ),
        "discharged_to_source_value": _optional_string(
            first_context_value(spans, "discharged_to_source_value")
        ),
        "preceding_visit_occurrence_id": foreign_key(
            first_context_value(spans, "preceding_visit_occurrence_id"),
            namespace="visit_occurrence",
        ),
    }
    return ({column: row[column] for column in VISIT_OCCURRENCE_COLUMNS},)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)
