"""OMOP CDM v5.4 ``procedure_occurrence`` export."""

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
    "PROCEDURE_OCCURRENCE_COLUMNS",
    "to_procedure_occurrence",
]

PROCEDURE_OCCURRENCE_COLUMNS: tuple[str, ...] = (
    "procedure_occurrence_id",
    "person_id",
    "procedure_concept_id",
    "procedure_date",
    "procedure_datetime",
    "procedure_end_date",
    "procedure_end_datetime",
    "procedure_type_concept_id",
    "modifier_concept_id",
    "quantity",
    "provider_id",
    "visit_occurrence_id",
    "visit_detail_id",
    "procedure_source_value",
    "procedure_source_concept_id",
    "modifier_source_value",
)


def to_procedure_occurrence(
    grounded: GroundedSpan | Iterable[GroundedSpan],
    *,
    concept_resolver: ConceptResolver | Any | None = None,
    resolver: ConceptResolver | Any | None = None,
    person_id: int | str | None = None,
    visit_occurrence_id: int | str | None = None,
    visit_id: int | str | None = None,
    document_id: str = "openmed-document",
    note_date: str | date | datetime | None = None,
    procedure_date: str | date | datetime | None = None,
    procedure_type_concept_id: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Emit CDM v5.4 rows for grounded ``PROCEDURE`` spans.

    Standard and source concept IDs come from the shared Athena-compatible
    resolver. Unmapped procedures use concept ID ``0`` while retaining the
    original source value. Optional CDM attributes are read from span metadata.
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
        if domain != "Procedure" or not span_is_exportable(span):
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

        explicit_row_id = context_value(span, "procedure_occurrence_id")
        row_id = foreign_key(explicit_row_id, namespace="procedure_occurrence")
        if row_id is None:
            row_id = table_row_id(
                "procedure_occurrence",
                span,
                index=index,
                document_id=document_id,
                person_id=resolved_person_id,
                visit_occurrence_id=resolved_visit_id,
                concept_id=resolved.standard_concept_id,
            )

        occurred_on = procedure_date
        if occurred_on is None:
            occurred_on = context_value(
                span,
                "procedure_date",
                "start_date",
                "date",
                "note_date",
            )
        if occurred_on is None:
            occurred_on = note_date

        type_id = concept_id(
            procedure_type_concept_id
            if procedure_type_concept_id is not None
            else context_value(
                span,
                "procedure_type_concept_id",
                "type_concept_id",
            ),
            name="procedure_type_concept_id",
            default=0,
        )
        modifier_source = context_value(span, "modifier_source_value")
        row = {
            "procedure_occurrence_id": row_id,
            "person_id": resolved_person_id,
            "procedure_concept_id": resolved.standard_concept_id,
            "procedure_date": date_value(occurred_on),
            "procedure_datetime": date_value(
                context_value(span, "procedure_datetime", "start_datetime")
            ),
            "procedure_end_date": date_value(
                context_value(span, "procedure_end_date", "end_date")
            ),
            "procedure_end_datetime": date_value(
                context_value(span, "procedure_end_datetime", "end_datetime")
            ),
            "procedure_type_concept_id": type_id,
            "modifier_concept_id": concept_id(
                context_value(span, "modifier_concept_id"),
                name="modifier_concept_id",
            ),
            "quantity": context_value(span, "quantity"),
            "provider_id": foreign_key(
                context_value(span, "provider_id"), namespace="provider"
            ),
            "visit_occurrence_id": resolved_visit_id,
            "visit_detail_id": foreign_key(
                context_value(span, "visit_detail_id"), namespace="visit_detail"
            ),
            "procedure_source_value": source_value(
                span,
                explicit=context_value(span, "procedure_source_value"),
                fallback_code=resolved.source_code,
            ),
            "procedure_source_concept_id": resolved.source_concept_id,
            "modifier_source_value": (
                str(modifier_source) if modifier_source is not None else None
            ),
        }
        rows.append({column: row[column] for column in PROCEDURE_OCCURRENCE_COLUMNS})
    return tuple(rows)
