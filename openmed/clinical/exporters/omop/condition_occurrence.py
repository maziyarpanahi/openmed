"""OMOP CDM v5.4 ``condition_occurrence`` export."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from openmed.clinical.grounding.types import GroundedSpan

from ._common import (
    ConceptResolver,
    assertion_status,
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
    "CONDITION_OCCURRENCE_COLUMNS",
    "to_condition_occurrence",
]

# These are the CDM v5.4 columns. The exporter deliberately returns only the
# table's standard columns; note/NLP provenance belongs to the loader layer.
CONDITION_OCCURRENCE_COLUMNS: tuple[str, ...] = (
    "condition_occurrence_id",
    "person_id",
    "condition_concept_id",
    "condition_start_date",
    "condition_start_datetime",
    "condition_end_date",
    "condition_end_datetime",
    "condition_type_concept_id",
    "condition_status_concept_id",
    "stop_reason",
    "provider_id",
    "visit_occurrence_id",
    "visit_detail_id",
    "condition_source_value",
    "condition_source_concept_id",
    "condition_status_source_value",
)


def to_condition_occurrence(
    grounded: GroundedSpan | Iterable[GroundedSpan],
    *,
    concept_resolver: ConceptResolver | Any | None = None,
    resolver: ConceptResolver | Any | None = None,
    person_id: int | str | None = None,
    visit_occurrence_id: int | str | None = None,
    visit_id: int | str | None = None,
    document_id: str = "openmed-document",
    note_date: str | date | datetime | None = None,
    condition_start_date: str | date | datetime | None = None,
    condition_type_concept_id: int | None = None,
    condition_status_concept_id: int | None = None,
    include_refuted: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Emit CDM v5.4 rows for grounded ``CONDITION`` spans.

    ``concept_resolver`` is an injected Athena-compatible resolver. It may be
    a mapping keyed by ``(system, code)``, a callable, or an object exposing
    ``route_span``/``resolve``. Unresolved candidates use concept ID ``0`` and
    retain their source text/code. Refuted, hypothetical, and non-patient
    assertions are excluded by default; ``include_refuted=True`` is available
    when a caller wants to flag rather than drop refuted rows.

    Foreign keys are nullable when omitted. String foreign keys that are not
    numeric receive deterministic local surrogate IDs, matching the rest of
    OpenMed's local-first OMOP exports.
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
        if domain != "Condition" or not span_is_exportable(
            span, include_refuted=include_refuted
        ):
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

        explicit_row_id = context_value(span, "condition_occurrence_id")
        row_id = foreign_key(explicit_row_id, namespace="condition_occurrence")
        if row_id is None:
            row_id = table_row_id(
                "condition_occurrence",
                span,
                index=index,
                document_id=document_id,
                person_id=resolved_person_id,
                visit_occurrence_id=resolved_visit_id,
                concept_id=resolved.standard_concept_id,
            )

        start_date = condition_start_date
        if start_date is None:
            start_date = context_value(
                span,
                "condition_start_date",
                "start_date",
                "date",
                "note_date",
            )
        if start_date is None:
            start_date = note_date

        type_id = condition_id(
            condition_type_concept_id
            if condition_type_concept_id is not None
            else context_value(
                span,
                "condition_type_concept_id",
                "type_concept_id",
            ),
            name="condition_type_concept_id",
            default=0,
        )
        status_id = condition_id(
            condition_status_concept_id
            if condition_status_concept_id is not None
            else context_value(
                span,
                "condition_status_concept_id",
                "status_concept_id",
            ),
            name="condition_status_concept_id",
        )
        status = assertion_status(span)
        status_source = context_value(
            span,
            "condition_status_source_value",
            "status_source_value",
        )
        if status_source is None and status is not None:
            status_source = status

        row = {
            "condition_occurrence_id": row_id,
            "person_id": resolved_person_id,
            "condition_concept_id": resolved.standard_concept_id,
            "condition_start_date": date_value(start_date),
            "condition_start_datetime": date_value(
                context_value(span, "condition_start_datetime", "start_datetime")
            ),
            "condition_end_date": date_value(
                context_value(span, "condition_end_date", "end_date")
            ),
            "condition_end_datetime": date_value(
                context_value(span, "condition_end_datetime", "end_datetime")
            ),
            "condition_type_concept_id": type_id,
            "condition_status_concept_id": status_id,
            "stop_reason": context_value(span, "stop_reason"),
            "provider_id": foreign_key(
                context_value(span, "provider_id"), namespace="provider"
            ),
            "visit_occurrence_id": resolved_visit_id,
            "visit_detail_id": foreign_key(
                context_value(span, "visit_detail_id"), namespace="visit_detail"
            ),
            "condition_source_value": source_value(
                span,
                explicit=context_value(span, "condition_source_value"),
                fallback_code=resolved.source_code,
            ),
            "condition_source_concept_id": resolved.source_concept_id,
            "condition_status_source_value": (
                str(status_source) if status_source is not None else None
            ),
        }
        rows.append({column: row[column] for column in CONDITION_OCCURRENCE_COLUMNS})
    return tuple(rows)


def condition_id(
    value: Any,
    *,
    name: str,
    default: int | None = None,
) -> int | None:
    """Local alias keeping concept-ID validation readable in row assembly."""

    return concept_id(value, name=name, default=default)
