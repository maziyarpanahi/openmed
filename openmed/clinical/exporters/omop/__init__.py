"""Deterministic OMOP CDM export for grounded clinical spans."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeAlias

from openmed.clinical.grounding.assertion_grounding import (
    GROUNDING_HYPOTHETICAL,
    GROUNDING_NON_PATIENT,
    GROUNDING_REFUTED,
    assertion_grounding_status,
)
from openmed.clinical.grounding.types import Candidate, GroundedSpan
from openmed.interop.omop import (
    OmopCdmTables,
    OmopConstraintViolation,
    load_grounded_notes,
    validate_omop_tables,
)

from ._common import (
    DOMAIN_BY_LABEL,
    DOMAIN_BY_SYSTEM,
    resolve_concept,
)
from .condition_occurrence import (
    CONDITION_OCCURRENCE_COLUMNS,
    to_condition_occurrence,
)
from .drug_exposure import DRUG_EXPOSURE_COLUMNS, to_drug_exposure

__all__ = [
    "CONDITION_OCCURRENCE_COLUMNS",
    "CORE_OMOP_TABLES",
    "ConceptResolver",
    "DRUG_EXPOSURE_COLUMNS",
    "achilles_smoke_check",
    "to_condition_occurrence",
    "to_drug_exposure",
    "to_omop",
]

CORE_OMOP_TABLES: tuple[str, ...] = (
    "condition_occurrence",
    "drug_exposure",
    "measurement",
    "procedure_occurrence",
)

ConceptResolver: TypeAlias = (
    Mapping[tuple[str, str], int]
    | Callable[[GroundedSpan, Candidate | None], int | None]
)

_CONCEPT_COLUMN_BY_TABLE = {
    "condition_occurrence": "condition_concept_id",
    "drug_exposure": "drug_concept_id",
    "measurement": "measurement_concept_id",
    "procedure_occurrence": "procedure_concept_id",
}


def to_omop(
    grounded: GroundedSpan | Iterable[GroundedSpan],
    *,
    table: str | None = None,
    document_text: str | None = None,
    document_id: str = "openmed-document",
    person_id: str | None = None,
    visit_id: str | None = None,
    note_date: str | None = None,
    concept_resolver: ConceptResolver | None = None,
    resolver: ConceptResolver | Any | None = None,
    vocabulary_version: str | None = None,
) -> OmopCdmTables | tuple[dict[str, Any], ...]:
    """Export grounded spans into OMOP CDM v5.4 rows.

    With no ``table`` argument this preserves the existing local-first export:
    an :class:`~openmed.interop.omop.OmopCdmTables` containing the four core
    clinical tables plus supporting concept, person, visit, note, NOTE_NLP, and
    source-to-concept rows. When ``table`` is ``condition_occurrence`` or
    ``drug_exposure``, the function returns only that table's exact CDM v5.4
    row dicts through the table-specific exporters.

    A caller-supplied resolver maps grounded source codes to Athena standard
    concept IDs. Missing mappings remain concept ID ``0`` while source text and
    code provenance are retained. Refuted, hypothetical, and non-patient
    assertions are excluded from the default occurrence-table export.
    """

    if concept_resolver is not None and resolver is not None:
        raise ValueError("provide only one of concept_resolver or resolver")
    active_resolver = concept_resolver if concept_resolver is not None else resolver
    spans = (grounded,) if isinstance(grounded, GroundedSpan) else tuple(grounded)
    if any(not isinstance(span, GroundedSpan) for span in spans):
        raise TypeError("to_omop expects GroundedSpan objects")

    if table is not None:
        if table not in {"condition_occurrence", "drug_exposure"}:
            raise ValueError(
                "table must be 'condition_occurrence', 'drug_exposure', or None"
            )
        if table == "condition_occurrence":
            return to_condition_occurrence(
                spans,
                concept_resolver=active_resolver,
                person_id=person_id,
                visit_id=visit_id,
                document_id=document_id,
                note_date=note_date,
            )
        return to_drug_exposure(
            spans,
            concept_resolver=active_resolver,
            person_id=person_id,
            visit_id=visit_id,
            document_id=document_id,
            note_date=note_date,
        )

    source_text = _document_text(spans, document_text)
    entities: list[dict[str, Any]] = []
    for span in spans:
        if not _exportable_assertion(span):
            continue
        domain = _domain(span)
        candidate = span.candidates[0] if span.candidates else None
        resolved = resolve_concept(span, active_resolver)
        entities.append(
            {
                "text": span.text,
                "start": span.start,
                "end": span.end,
                "domain_id": domain,
                "concept_id": resolved.standard_concept_id,
                "source_concept_id": resolved.source_concept_id,
                "code": candidate.code if candidate else "",
                "vocabulary_id": candidate.system if candidate else "UNMAPPED",
                "concept_name": candidate.display if candidate else span.text,
            }
        )

    note: dict[str, Any] = {
        "document_id": document_id,
        "person_id": person_id or "openmed-subject",
        "note_text": source_text,
        "entities": entities,
    }
    if visit_id is not None:
        note["visit_id"] = visit_id
    if note_date is not None:
        note["note_date"] = note_date
    return load_grounded_notes(
        [note],
        vocabulary_version=vocabulary_version or _vocabulary_version(spans),
    )


def achilles_smoke_check(
    tables: OmopCdmTables,
) -> tuple[OmopConstraintViolation, ...]:
    """Run an offline ACHILLES-style structural preflight over core tables.

    Full OHDSI ACHILLES requires a deployed CDM database and is deliberately
    not bundled. This smoke subset checks expected tables/columns,
    nonnegative standard concept IDs, positive deterministic keys, and all
    concept/note references validated by :func:`validate_omop_tables`.
    """

    violations = list(validate_omop_tables(tables))
    for table in CORE_OMOP_TABLES:
        if table not in tables.tables:
            violations.append(
                OmopConstraintViolation(
                    table=table,
                    column="",
                    reason="missing_core_table",
                )
            )
            continue
        concept_column = _CONCEPT_COLUMN_BY_TABLE[table]
        for row in tables.table(table):
            row_id = _row_id(row)
            if concept_column not in row:
                violations.append(
                    OmopConstraintViolation(
                        table=table,
                        column=concept_column,
                        reason="missing_concept_column",
                        row_id=row_id,
                    )
                )
                continue
            concept = row[concept_column]
            if isinstance(concept, bool) or not isinstance(concept, int) or concept < 0:
                violations.append(
                    OmopConstraintViolation(
                        table=table,
                        column=concept_column,
                        reason="invalid_concept_id",
                        row_id=row_id,
                    )
                )
            if not isinstance(row.get("person_id"), int) or row["person_id"] <= 0:
                violations.append(
                    OmopConstraintViolation(
                        table=table,
                        column="person_id",
                        reason="invalid_person_id",
                        row_id=row_id,
                    )
                )
    return tuple(violations)


def _document_text(spans: tuple[GroundedSpan, ...], document_text: str | None) -> str:
    if document_text is not None:
        if not isinstance(document_text, str):
            raise TypeError("document_text must be a string")
        if spans and max(span.end for span in spans) > len(document_text):
            raise ValueError("grounded span offsets exceed document_text")
        return document_text
    if not spans:
        return ""
    characters = [" "] * max(span.end for span in spans)
    occupied: dict[int, str] = {}
    for span in spans:
        if span.end - span.start != len(span.text):
            raise ValueError(
                "document_text is required when span text length differs from offsets"
            )
        for offset, character in enumerate(span.text, start=span.start):
            previous = occupied.get(offset)
            if previous is not None and previous != character:
                raise ValueError("overlapping grounded spans disagree on source text")
            occupied[offset] = character
            characters[offset] = character
    return "".join(characters)


def _domain(span: GroundedSpan) -> str:
    label = (span.canonical_label or "").upper()
    if label in DOMAIN_BY_LABEL:
        return DOMAIN_BY_LABEL[label]
    if span.candidates:
        system = span.candidates[0].system.upper()
        if system in DOMAIN_BY_SYSTEM:
            return DOMAIN_BY_SYSTEM[system]
    raise ValueError("cannot infer an OMOP domain; provide a supported canonical_label")


def _exportable_assertion(span: GroundedSpan) -> bool:
    if span.assertion is None:
        return True
    status = assertion_grounding_status(span.assertion).status
    return status not in {
        GROUNDING_REFUTED,
        GROUNDING_HYPOTHETICAL,
        GROUNDING_NON_PATIENT,
    }


def _vocabulary_version(spans: tuple[GroundedSpan, ...]) -> str:
    versions = sorted(
        {
            candidate.vocab_version
            for span in spans
            for candidate in span.candidates
            if candidate.vocab_version
        }
    )
    if not versions:
        return ""
    if len(versions) == 1:
        return versions[0]
    digest = hashlib.sha256("\n".join(versions).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _row_id(row: Mapping[str, Any]) -> int | None:
    for key, value in row.items():
        if key.endswith("_id") and isinstance(value, int):
            return value
    return None
