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

__all__ = [
    "CORE_OMOP_TABLES",
    "ConceptResolver",
    "achilles_smoke_check",
    "to_omop",
]

CORE_OMOP_TABLES: tuple[str, ...] = (
    "condition_occurrence",
    "drug_exposure",
    "measurement",
    "procedure_occurrence",
)

_DOMAIN_BY_LABEL = {
    "CONDITION": "Condition",
    "MEDICATION": "Drug",
    "LAB_TEST": "Measurement",
    "PROCEDURE": "Procedure",
}
_DOMAIN_BY_SYSTEM = {
    "HPO": "Condition",
    "ICD10CM": "Condition",
    "ICD11": "Condition",
    "MESH": "Condition",
    "SNOMED": "Condition",
    "UMLS": "Condition",
    "RXNORM": "Drug",
    "LOINC": "Measurement",
}
_CONCEPT_COLUMN_BY_TABLE = {
    "condition_occurrence": "condition_concept_id",
    "drug_exposure": "drug_concept_id",
    "measurement": "measurement_concept_id",
    "procedure_occurrence": "procedure_concept_id",
}

ConceptResolver: TypeAlias = (
    Mapping[tuple[str, str], int]
    | Callable[[GroundedSpan, Candidate | None], int | None]
)


def to_omop(
    grounded: GroundedSpan | Iterable[GroundedSpan],
    *,
    document_text: str | None = None,
    document_id: str = "openmed-document",
    person_id: str = "openmed-subject",
    visit_id: str | None = None,
    note_date: str | None = None,
    concept_resolver: ConceptResolver | None = None,
    vocabulary_version: str | None = None,
) -> OmopCdmTables:
    """Export grounded spans into OMOP CDM v5.4 table rows.

    The returned :class:`~openmed.interop.omop.OmopCdmTables` includes the four
    core clinical tables named by :data:`CORE_OMOP_TABLES` plus the supporting
    ``concept``, ``person``, ``visit_occurrence``, ``note``, ``note_nlp``, and
    ``source_to_concept_map`` tables needed for referential validation.

    A caller-supplied resolver maps ``(system, code)`` pairs to Athena standard
    ``concept_id`` values. Missing mappings remain ``concept_id=0`` while the
    source system/code/value are preserved; the exporter never fabricates an
    OMOP concept identifier. Refuted, hypothetical, and non-patient assertions
    are excluded from occurrence tables.

    Args:
        grounded: One span or an iterable from a single source document.
        document_text: Original source text. When omitted, a deterministic text
            shell is reconstructed from span offsets for NOTE_NLP integrity.
        document_id: Stable source-document identifier.
        person_id: Stable source person identifier (hashed by the CDM loader).
        visit_id: Optional source visit identifier.
        note_date: Optional ISO note date.
        concept_resolver: Mapping or callable returning standard concept IDs.
        vocabulary_version: Optional vocabulary release/version provenance.

    Returns:
        In-memory OMOP CDM v5.4 tables with deterministic identifiers.
    """

    spans = (grounded,) if isinstance(grounded, GroundedSpan) else tuple(grounded)
    if any(not isinstance(span, GroundedSpan) for span in spans):
        raise TypeError("to_omop expects GroundedSpan objects")
    source_text = _document_text(spans, document_text)
    entities: list[dict[str, Any]] = []
    for span in spans:
        if not _exportable_assertion(span):
            continue
        domain = _domain(span)
        candidate = span.candidates[0] if span.candidates else None
        concept_id = _concept_id(span, candidate, concept_resolver)
        entities.append(
            {
                "text": span.text,
                "start": span.start,
                "end": span.end,
                "domain_id": domain,
                "concept_id": concept_id,
                "source_concept_id": 0,
                "code": candidate.code if candidate else "",
                "vocabulary_id": candidate.system if candidate else "UNMAPPED",
                "concept_name": candidate.display if candidate else span.text,
            }
        )

    note: dict[str, Any] = {
        "document_id": document_id,
        "person_id": person_id,
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
    not bundled. This documented smoke subset checks the prerequisites most
    relevant to generated grounding rows: expected tables/columns, nonnegative
    standard concept IDs, positive deterministic keys, and all concept/note
    references validated by :func:`validate_omop_tables`.

    Args:
        tables: In-memory OMOP CDM tables to inspect.

    Returns:
        Deterministically ordered structural violations, or an empty tuple.
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
            concept_id = row[concept_column]
            if (
                isinstance(concept_id, bool)
                or not isinstance(concept_id, int)
                or concept_id < 0
            ):
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
    if label in _DOMAIN_BY_LABEL:
        return _DOMAIN_BY_LABEL[label]
    if span.candidates:
        system = span.candidates[0].system.upper()
        if system in _DOMAIN_BY_SYSTEM:
            return _DOMAIN_BY_SYSTEM[system]
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


def _concept_id(
    span: GroundedSpan,
    candidate: Candidate | None,
    resolver: ConceptResolver | None,
) -> int:
    if resolver is None:
        return 0
    if callable(resolver):
        resolved = resolver(span, candidate)
    elif candidate is None:
        resolved = None
    else:
        keys = (
            (candidate.system, candidate.code),
            (candidate.system.upper(), candidate.code),
            (candidate.system.casefold(), candidate.code),
        )
        resolved = next((resolver[key] for key in keys if key in resolver), None)
    if resolved is None:
        return 0
    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved <= 0:
        raise ValueError("concept_resolver values must be positive integers or None")
    return resolved


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
