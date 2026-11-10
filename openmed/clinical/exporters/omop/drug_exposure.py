"""OMOP CDM v5.4 ``drug_exposure`` export."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from openmed.clinical.grounding.types import GroundedSpan

from ._common import (
    ConceptResolver,
    concept_id,
    context_value,
    date_value,
    domain_for_span,
    foreign_key,
    iter_spans,
    resolve_concept,
    source_value,
    span_is_exportable,
    table_row_id,
)

__all__ = [
    "DRUG_EXPOSURE_COLUMNS",
    "to_drug_exposure",
]

# These are the CDM v5.4 columns. Dose and administration details remain
# nullable unless the grounded span's metadata supplies them.
DRUG_EXPOSURE_COLUMNS: tuple[str, ...] = (
    "drug_exposure_id",
    "person_id",
    "drug_concept_id",
    "drug_exposure_start_date",
    "drug_exposure_start_datetime",
    "drug_exposure_end_date",
    "drug_exposure_end_datetime",
    "verbatim_end_date",
    "drug_type_concept_id",
    "stop_reason",
    "refills",
    "quantity",
    "days_supply",
    "sig",
    "route_concept_id",
    "lot_number",
    "provider_id",
    "visit_occurrence_id",
    "visit_detail_id",
    "drug_source_value",
    "drug_source_concept_id",
    "route_source_value",
    "dose_unit_source_value",
)


def to_drug_exposure(
    grounded: GroundedSpan | Iterable[GroundedSpan],
    *,
    concept_resolver: ConceptResolver | Any | None = None,
    resolver: ConceptResolver | Any | None = None,
    person_id: int | str | None = None,
    visit_occurrence_id: int | str | None = None,
    visit_id: int | str | None = None,
    document_id: str = "openmed-document",
    note_date: str | date | datetime | None = None,
    drug_exposure_start_date: str | date | datetime | None = None,
    drug_type_concept_id: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Emit CDM v5.4 rows for grounded ``MEDICATION`` spans.

    Concept resolution is injected and conservative: a missing Athena match
    produces ``drug_concept_id=0`` while retaining the source value and source
    concept ID supplied by the resolver. Person and visit foreign keys remain
    nullable when callers do not provide them.
    """

    if concept_resolver is not None and resolver is not None:
        raise ValueError("provide only one of concept_resolver or resolver")
    active_resolver = concept_resolver if concept_resolver is not None else resolver
    rows: list[dict[str, Any]] = []
    for index, span in enumerate(iter_spans(grounded)):
        try:
            domain = domain_for_span(span)
        except ValueError:
            continue
        if domain != "Drug" or not span_is_exportable(span):
            continue

        resolved = resolve_concept(span, active_resolver)
        metadata_person = context_value(span, "person_id", "patient_id", "subject_id")
        metadata_visit = context_value(
            span,
            "visit_occurrence_id",
            "visit_id",
            "encounter_id",
        )
        resolved_person_id = foreign_key(
            person_id if person_id is not None else metadata_person,
            namespace="person",
        )
        visit_input = (
            visit_occurrence_id
            if visit_occurrence_id is not None
            else visit_id
            if visit_id is not None
            else metadata_visit
        )
        resolved_visit_id = foreign_key(visit_input, namespace="visit_occurrence")

        explicit_row_id = context_value(span, "drug_exposure_id")
        row_id = foreign_key(explicit_row_id, namespace="drug_exposure")
        if row_id is None:
            row_id = table_row_id(
                "drug_exposure",
                span,
                index=index,
                document_id=document_id,
                person_id=resolved_person_id,
                visit_occurrence_id=resolved_visit_id,
                concept_id=resolved.standard_concept_id,
            )

        start_date = drug_exposure_start_date
        if start_date is None:
            start_date = context_value(
                span,
                "drug_exposure_start_date",
                "start_date",
                "date",
                "note_date",
            )
        if start_date is None:
            start_date = note_date

        type_id = concept_id(
            drug_type_concept_id
            if drug_type_concept_id is not None
            else context_value(span, "drug_type_concept_id", "type_concept_id"),
            name="drug_type_concept_id",
            default=0,
        )
        row = {
            "drug_exposure_id": row_id,
            "person_id": resolved_person_id,
            "drug_concept_id": resolved.standard_concept_id,
            "drug_exposure_start_date": date_value(start_date),
            "drug_exposure_start_datetime": date_value(
                context_value(span, "drug_exposure_start_datetime", "start_datetime")
            ),
            "drug_exposure_end_date": date_value(
                context_value(span, "drug_exposure_end_date", "end_date")
            ),
            "drug_exposure_end_datetime": date_value(
                context_value(span, "drug_exposure_end_datetime", "end_datetime")
            ),
            "verbatim_end_date": date_value(context_value(span, "verbatim_end_date")),
            "drug_type_concept_id": type_id,
            "stop_reason": context_value(span, "stop_reason"),
            "refills": context_value(span, "refills"),
            "quantity": context_value(span, "quantity"),
            "days_supply": context_value(span, "days_supply"),
            "sig": context_value(span, "sig", "directions"),
            "route_concept_id": concept_id(
                context_value(span, "route_concept_id"),
                name="route_concept_id",
            ),
            "lot_number": context_value(span, "lot_number"),
            "provider_id": foreign_key(
                context_value(span, "provider_id"), namespace="provider"
            ),
            "visit_occurrence_id": resolved_visit_id,
            "visit_detail_id": foreign_key(
                context_value(span, "visit_detail_id"), namespace="visit_detail"
            ),
            "drug_source_value": source_value(
                span,
                explicit=context_value(span, "drug_source_value"),
                fallback_code=resolved.source_code,
            ),
            "drug_source_concept_id": resolved.source_concept_id,
            "route_source_value": context_value(span, "route_source_value"),
            "dose_unit_source_value": context_value(
                span, "dose_unit_source_value", "dose_unit"
            ),
        }
        rows.append({column: row[column] for column in DRUG_EXPOSURE_COLUMNS})
    return tuple(rows)
