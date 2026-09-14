"""OMOP CDM v5.4 ``measurement`` export."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from math import isfinite
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
    "MEASUREMENT_COLUMNS",
    "to_measurement",
]

MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "measurement_id",
    "person_id",
    "measurement_concept_id",
    "measurement_date",
    "measurement_datetime",
    "measurement_time",
    "measurement_type_concept_id",
    "operator_concept_id",
    "value_as_number",
    "value_as_concept_id",
    "unit_concept_id",
    "range_low",
    "range_high",
    "provider_id",
    "visit_occurrence_id",
    "visit_detail_id",
    "measurement_source_value",
    "measurement_source_concept_id",
    "unit_source_value",
    "unit_source_concept_id",
    "value_source_value",
    "measurement_event_id",
    "meas_event_field_concept_id",
)


def to_measurement(
    grounded: GroundedSpan | Iterable[GroundedSpan],
    *,
    concept_resolver: ConceptResolver | Any | None = None,
    resolver: ConceptResolver | Any | None = None,
    person_id: int | str | None = None,
    visit_occurrence_id: int | str | None = None,
    visit_id: int | str | None = None,
    document_id: str = "openmed-document",
    note_date: str | date | datetime | None = None,
    measurement_date: str | date | datetime | None = None,
    measurement_type_concept_id: int | None = None,
    value_as_number: int | float | str | None = None,
    unit_concept_id: int | None = None,
    unit_source_value: str | None = None,
    range_low: int | float | str | None = None,
    range_high: int | float | str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Emit CDM v5.4 rows for grounded ``LAB_TEST`` spans.

    The caller-supplied resolver is the same Athena-compatible resolver used
    by the condition and drug exporters. An unresolved test keeps its source
    value and emits ``measurement_concept_id=0``. Numeric results, units, and
    reference ranges can be passed directly or supplied in span metadata.
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
        if domain != "Measurement" or not span_is_exportable(span):
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

        explicit_row_id = context_value(span, "measurement_id")
        row_id = foreign_key(explicit_row_id, namespace="measurement")
        if row_id is None:
            row_id = table_row_id(
                "measurement",
                span,
                index=index,
                document_id=document_id,
                person_id=resolved_person_id,
                visit_occurrence_id=resolved_visit_id,
                concept_id=resolved.standard_concept_id,
            )

        measured_on = measurement_date
        if measured_on is None:
            measured_on = context_value(
                span,
                "measurement_date",
                "result_date",
                "specimen_date",
                "date",
                "note_date",
            )
        if measured_on is None:
            measured_on = note_date

        raw_value = value_as_number
        if raw_value is None:
            raw_value = context_value(
                span,
                "value_as_number",
                "numeric_value",
                "result_value",
                "lab_value",
                "value",
            )
        raw_range = context_value(span, "reference_range", "range")
        resolved_range_low = range_low
        if resolved_range_low is None:
            resolved_range_low = context_value(span, "range_low", "reference_low")
        resolved_range_high = range_high
        if resolved_range_high is None:
            resolved_range_high = context_value(span, "range_high", "reference_high")
        if isinstance(raw_range, Mapping):
            if resolved_range_low is None:
                resolved_range_low = raw_range.get("low")
            if resolved_range_high is None:
                resolved_range_high = raw_range.get("high")

        raw_unit = unit_source_value
        if raw_unit is None:
            raw_unit = context_value(
                span,
                "unit_source_value",
                "unit",
                "units",
                "ucum_unit",
            )
        raw_unit_concept = unit_concept_id
        if raw_unit_concept is None:
            raw_unit_concept = context_value(span, "unit_concept_id")
        unit_default = 0 if raw_unit is not None else None

        type_id = concept_id(
            measurement_type_concept_id
            if measurement_type_concept_id is not None
            else context_value(
                span,
                "measurement_type_concept_id",
                "type_concept_id",
            ),
            name="measurement_type_concept_id",
            default=0,
        )
        value_source = context_value(span, "value_source_value")
        if value_source is None and raw_value is not None:
            value_source = str(raw_value)

        row = {
            "measurement_id": row_id,
            "person_id": resolved_person_id,
            "measurement_concept_id": resolved.standard_concept_id,
            "measurement_date": date_value(measured_on),
            "measurement_datetime": date_value(
                context_value(span, "measurement_datetime", "result_datetime")
            ),
            "measurement_time": _optional_string(
                context_value(span, "measurement_time")
            ),
            "measurement_type_concept_id": type_id,
            "operator_concept_id": concept_id(
                context_value(span, "operator_concept_id"),
                name="operator_concept_id",
            ),
            "value_as_number": _number_value(raw_value, name="value_as_number"),
            "value_as_concept_id": concept_id(
                context_value(span, "value_as_concept_id"),
                name="value_as_concept_id",
            ),
            "unit_concept_id": concept_id(
                raw_unit_concept,
                name="unit_concept_id",
                default=unit_default,
            ),
            "range_low": _number_value(resolved_range_low, name="range_low"),
            "range_high": _number_value(resolved_range_high, name="range_high"),
            "provider_id": foreign_key(
                context_value(span, "provider_id"), namespace="provider"
            ),
            "visit_occurrence_id": resolved_visit_id,
            "visit_detail_id": foreign_key(
                context_value(span, "visit_detail_id"), namespace="visit_detail"
            ),
            "measurement_source_value": source_value(
                span,
                explicit=context_value(span, "measurement_source_value"),
                fallback_code=resolved.source_code,
            ),
            "measurement_source_concept_id": resolved.source_concept_id,
            "unit_source_value": _optional_string(raw_unit),
            "unit_source_concept_id": concept_id(
                context_value(span, "unit_source_concept_id"),
                name="unit_source_concept_id",
                default=unit_default,
            ),
            "value_source_value": _optional_string(value_source),
            "measurement_event_id": foreign_key(
                context_value(span, "measurement_event_id"),
                namespace="measurement_event",
            ),
            "meas_event_field_concept_id": concept_id(
                context_value(span, "meas_event_field_concept_id"),
                name="meas_event_field_concept_id",
            ),
        }
        rows.append({column: row[column] for column in MEASUREMENT_COLUMNS})
    return tuple(rows)


def _number_value(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number or None")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number or None") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number or None")
    return result


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)
