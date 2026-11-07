"""OMOP CDM v5.4 ``note_nlp`` export."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from openmed.clinical.grounding.assertion_grounding import (
    GROUNDING_HYPOTHETICAL,
    GROUNDING_NON_PATIENT,
    GROUNDING_REFUTED,
    assertion_grounding_status,
)
from openmed.clinical.grounding.types import GroundedSpan
from openmed.interop.omop import deterministic_omop_id

from ._common import (
    ConceptResolver,
    concept_id,
    context_value,
    date_value,
    foreign_key,
    iter_spans,
    resolve_concept,
)

__all__ = [
    "NOTE_NLP_COLUMNS",
    "to_note_nlp",
]

NOTE_NLP_COLUMNS: tuple[str, ...] = (
    "note_nlp_id",
    "note_id",
    "section_concept_id",
    "snippet",
    "offset",
    "lexical_variant",
    "note_nlp_concept_id",
    "note_nlp_source_concept_id",
    "nlp_system",
    "nlp_date",
    "nlp_datetime",
    "term_exists",
    "term_temporal",
    "term_modifiers",
)

_NON_EXISTING_STATUSES = {
    GROUNDING_HYPOTHETICAL,
    GROUNDING_NON_PATIENT,
    GROUNDING_REFUTED,
}


def to_note_nlp(
    grounded: GroundedSpan | Iterable[GroundedSpan],
    *,
    concept_resolver: ConceptResolver | Any | None = None,
    resolver: ConceptResolver | Any | None = None,
    note_id: int | str | None = None,
    document_id: str = "openmed-document",
    note_date: str | date | datetime | None = None,
    nlp_date: str | date | datetime | None = None,
    nlp_datetime: str | date | datetime | None = None,
    nlp_system: str = "openmed",
) -> tuple[dict[str, Any], ...]:
    """Emit one CDM v5.4 NOTE_NLP row per grounded span.

    Unlike occurrence-table exporters, this function retains refuted,
    hypothetical, and non-patient mentions. Their assertion axes are encoded
    in ``term_exists``, ``term_temporal``, and deterministic
    ``term_modifiers`` fields so downstream consumers do not mistake them for
    active patient facts.
    """

    if concept_resolver is not None and resolver is not None:
        raise ValueError("provide only one of concept_resolver or resolver")
    if not isinstance(nlp_system, str) or not nlp_system.strip():
        raise ValueError("nlp_system must be a non-empty string")
    active_resolver = concept_resolver if concept_resolver is not None else resolver
    rows: list[dict[str, Any]] = []
    for index, span in enumerate(iter_spans(grounded)):
        resolved = resolve_concept(span, active_resolver)
        explicit_note_id = (
            note_id if note_id is not None else context_value(span, "note_id")
        )
        resolved_note_id = foreign_key(explicit_note_id, namespace="note")
        if resolved_note_id is None:
            resolved_note_id = deterministic_omop_id("note", document_id)

        explicit_row_id = context_value(span, "note_nlp_id")
        row_id = foreign_key(explicit_row_id, namespace="note_nlp")
        if row_id is None:
            row_id = deterministic_omop_id(
                "note_nlp",
                resolved_note_id,
                span.start,
                span.end,
                resolved.standard_concept_id,
                index,
            )

        assertion_fields = _assertion_fields(span)
        processed_on = nlp_date
        if processed_on is None:
            processed_on = context_value(span, "nlp_date")
        if processed_on is None:
            processed_on = note_date
        processed_at = nlp_datetime
        if processed_at is None:
            processed_at = context_value(span, "nlp_datetime")

        snippet = context_value(span, "snippet")
        row = {
            "note_nlp_id": row_id,
            "note_id": resolved_note_id,
            "section_concept_id": concept_id(
                context_value(span, "section_concept_id"),
                name="section_concept_id",
            ),
            "snippet": str(snippet) if snippet is not None else None,
            "offset": span.start,
            "lexical_variant": span.text,
            "note_nlp_concept_id": resolved.standard_concept_id,
            "note_nlp_source_concept_id": resolved.source_concept_id,
            "nlp_system": nlp_system.strip(),
            "nlp_date": date_value(processed_on),
            "nlp_datetime": date_value(processed_at),
            "term_exists": assertion_fields["term_exists"],
            "term_temporal": assertion_fields["term_temporal"],
            "term_modifiers": assertion_fields["term_modifiers"],
        }
        rows.append({column: row[column] for column in NOTE_NLP_COLUMNS})
    return tuple(rows)


def _assertion_fields(span: GroundedSpan) -> dict[str, str | None]:
    assertion = span.assertion
    if assertion is None:
        return {
            "term_exists": "Y",
            "term_temporal": None,
            "term_modifiers": None,
        }
    status = assertion_grounding_status(assertion).status
    return {
        "term_exists": "N" if status in _NON_EXISTING_STATUSES else "Y",
        "term_temporal": assertion.temporality,
        "term_modifiers": json.dumps(
            assertion.to_dict(), sort_keys=True, separators=(",", ":")
        ),
    }
