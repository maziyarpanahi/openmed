"""OMOP CDM v5.4 loader for grounded clinical note spans.

This module builds the first local-first OMOP loader slice for grounded
OpenMed notes. It deliberately accepts caller-supplied vocabulary/concept
metadata, never bundles restricted vocabulary content, and keeps run summaries
limited to counts, hashes, offsets, and rejection reasons.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

UNMAPPED_CONCEPT_ID = 0
UNMAPPED_CONCEPT_NAME = "No matching concept"
UNMAPPED_VOCABULARY_ID = "UNMAPPED"

LoadMode = Literal["append"]
OmopDomain = Literal["Condition", "Drug", "Measurement", "Procedure", "Observation"]
WriterKind = Literal["duckdb", "sqlite", "parquet"]

_MISSING = object()

_TABLE_ORDER: tuple[str, ...] = (
    "concept",
    "person",
    "visit_occurrence",
    "note",
    "note_nlp",
    "condition_occurrence",
    "drug_exposure",
    "measurement",
    "procedure_occurrence",
    "observation",
    "source_to_concept_map",
)

_PRIMARY_KEYS: Mapping[str, str] = {
    "concept": "concept_id",
    "person": "person_id",
    "visit_occurrence": "visit_occurrence_id",
    "note": "note_id",
    "note_nlp": "note_nlp_id",
    "condition_occurrence": "condition_occurrence_id",
    "drug_exposure": "drug_exposure_id",
    "measurement": "measurement_id",
    "procedure_occurrence": "procedure_occurrence_id",
    "observation": "observation_id",
    "source_to_concept_map": "source_to_concept_map_id",
}

_DOMAIN_ALIASES: Mapping[str, OmopDomain] = {
    "condition": "Condition",
    "condition_occurrence": "Condition",
    "diagnosis": "Condition",
    "disease": "Condition",
    "disorder": "Condition",
    "problem": "Condition",
    "symptom": "Condition",
    "drug": "Drug",
    "drug_exposure": "Drug",
    "medication": "Drug",
    "medicine": "Drug",
    "rx": "Drug",
    "treatment": "Drug",
    "lab": "Measurement",
    "lab_value": "Measurement",
    "laboratory": "Measurement",
    "measurement": "Measurement",
    "vital": "Measurement",
    "vital_sign": "Measurement",
    "procedure": "Procedure",
    "procedure_occurrence": "Procedure",
    "surgery": "Procedure",
    "operation": "Procedure",
    "observation": "Observation",
    "social_history": "Observation",
    "finding": "Observation",
}

_DOMAIN_TABLES: Mapping[OmopDomain, tuple[str, str, str, str]] = {
    "Condition": (
        "condition_occurrence",
        "condition_occurrence_id",
        "condition_concept_id",
        "condition_source_value",
    ),
    "Drug": (
        "drug_exposure",
        "drug_exposure_id",
        "drug_concept_id",
        "drug_source_value",
    ),
    "Measurement": (
        "measurement",
        "measurement_id",
        "measurement_concept_id",
        "measurement_source_value",
    ),
    "Procedure": (
        "procedure_occurrence",
        "procedure_occurrence_id",
        "procedure_concept_id",
        "procedure_source_value",
    ),
    "Observation": (
        "observation",
        "observation_id",
        "observation_concept_id",
        "observation_source_value",
    ),
}

_DOMAIN_SOURCE_CONCEPT_COLUMNS: Mapping[str, str] = {
    "condition_occurrence": "condition_source_concept_id",
    "drug_exposure": "drug_source_concept_id",
    "measurement": "measurement_source_concept_id",
    "procedure_occurrence": "procedure_source_concept_id",
    "observation": "observation_source_concept_id",
}

_SCHEMA_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "concept": (
        "concept_id",
        "concept_name",
        "domain_id",
        "vocabulary_id",
        "concept_class_id",
        "standard_concept",
        "concept_code",
    ),
    "person": ("person_id", "person_source_value"),
    "visit_occurrence": (
        "visit_occurrence_id",
        "person_id",
        "visit_concept_id",
        "visit_start_date",
        "visit_source_value",
        "visit_source_concept_id",
    ),
    "note": (
        "note_id",
        "person_id",
        "visit_occurrence_id",
        "note_date",
        "note_type_concept_id",
        "note_class_concept_id",
        "note_title",
        "note_text",
        "encoding_concept_id",
        "language_concept_id",
        "note_source_value",
        "source_note_hash",
    ),
    "note_nlp": (
        "note_nlp_id",
        "note_id",
        "section_concept_id",
        "snippet",
        "offset",
        "offset_end",
        "lexical_variant",
        "note_nlp_concept_id",
        "note_nlp_source_concept_id",
        "nlp_system",
        "nlp_date",
        "term_exists",
        "term_temporal",
        "term_modifiers",
        "note_nlp_event_id",
        "note_nlp_event_field_concept_id",
    ),
    "condition_occurrence": (
        "condition_occurrence_id",
        "person_id",
        "condition_concept_id",
        "condition_start_date",
        "condition_type_concept_id",
        "visit_occurrence_id",
        "condition_source_value",
        "condition_source_concept_id",
        "note_id",
        "note_nlp_id",
        "source_note_hash",
        "idempotent_key",
    ),
    "drug_exposure": (
        "drug_exposure_id",
        "person_id",
        "drug_concept_id",
        "drug_exposure_start_date",
        "drug_type_concept_id",
        "visit_occurrence_id",
        "drug_source_value",
        "drug_source_concept_id",
        "note_id",
        "note_nlp_id",
        "source_note_hash",
        "idempotent_key",
    ),
    "measurement": (
        "measurement_id",
        "person_id",
        "measurement_concept_id",
        "measurement_date",
        "measurement_type_concept_id",
        "visit_occurrence_id",
        "measurement_source_value",
        "measurement_source_concept_id",
        "note_id",
        "note_nlp_id",
        "source_note_hash",
        "idempotent_key",
    ),
    "procedure_occurrence": (
        "procedure_occurrence_id",
        "person_id",
        "procedure_concept_id",
        "procedure_date",
        "procedure_type_concept_id",
        "visit_occurrence_id",
        "procedure_source_value",
        "procedure_source_concept_id",
        "note_id",
        "note_nlp_id",
        "source_note_hash",
        "idempotent_key",
    ),
    "observation": (
        "observation_id",
        "person_id",
        "observation_concept_id",
        "observation_date",
        "observation_type_concept_id",
        "visit_occurrence_id",
        "observation_source_value",
        "observation_source_concept_id",
        "note_id",
        "note_nlp_id",
        "source_note_hash",
        "idempotent_key",
    ),
    "source_to_concept_map": (
        "source_to_concept_map_id",
        "source_code",
        "source_concept_id",
        "source_vocabulary_id",
        "source_code_description",
        "target_concept_id",
        "target_vocabulary_id",
        "valid_start_date",
        "valid_end_date",
        "invalid_reason",
        "vocabulary_version",
        "source_note_hash",
        "note_nlp_id",
    ),
}

_SQL_DDL: Mapping[str, str] = {
    "concept": """
        CREATE TABLE IF NOT EXISTS concept (
            concept_id BIGINT PRIMARY KEY,
            concept_name TEXT,
            domain_id TEXT,
            vocabulary_id TEXT,
            concept_class_id TEXT,
            standard_concept TEXT,
            concept_code TEXT
        )
    """,
    "person": """
        CREATE TABLE IF NOT EXISTS person (
            person_id BIGINT PRIMARY KEY,
            person_source_value TEXT NOT NULL
        )
    """,
    "visit_occurrence": """
        CREATE TABLE IF NOT EXISTS visit_occurrence (
            visit_occurrence_id BIGINT PRIMARY KEY,
            person_id BIGINT NOT NULL REFERENCES person(person_id),
            visit_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            visit_start_date TEXT,
            visit_source_value TEXT,
            visit_source_concept_id BIGINT NOT NULL REFERENCES concept(concept_id)
        )
    """,
    "note": """
        CREATE TABLE IF NOT EXISTS note (
            note_id BIGINT PRIMARY KEY,
            person_id BIGINT NOT NULL REFERENCES person(person_id),
            visit_occurrence_id BIGINT REFERENCES visit_occurrence(visit_occurrence_id),
            note_date TEXT,
            note_type_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            note_class_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            note_title TEXT,
            note_text TEXT,
            encoding_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            language_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            note_source_value TEXT,
            source_note_hash TEXT NOT NULL
        )
    """,
    "note_nlp": """
        CREATE TABLE IF NOT EXISTS note_nlp (
            note_nlp_id BIGINT PRIMARY KEY,
            note_id BIGINT NOT NULL REFERENCES note(note_id),
            section_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            snippet TEXT,
            "offset" BIGINT NOT NULL,
            offset_end BIGINT NOT NULL,
            lexical_variant TEXT,
            note_nlp_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            note_nlp_source_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            nlp_system TEXT,
            nlp_date TEXT,
            term_exists TEXT,
            term_temporal TEXT,
            term_modifiers TEXT,
            note_nlp_event_id BIGINT,
            note_nlp_event_field_concept_id BIGINT NOT NULL REFERENCES concept(concept_id)
        )
    """,
    "condition_occurrence": """
        CREATE TABLE IF NOT EXISTS condition_occurrence (
            condition_occurrence_id BIGINT PRIMARY KEY,
            person_id BIGINT NOT NULL REFERENCES person(person_id),
            condition_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            condition_start_date TEXT,
            condition_type_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            visit_occurrence_id BIGINT REFERENCES visit_occurrence(visit_occurrence_id),
            condition_source_value TEXT,
            condition_source_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            note_id BIGINT NOT NULL REFERENCES note(note_id),
            note_nlp_id BIGINT NOT NULL REFERENCES note_nlp(note_nlp_id),
            source_note_hash TEXT NOT NULL,
            idempotent_key TEXT NOT NULL
        )
    """,
    "drug_exposure": """
        CREATE TABLE IF NOT EXISTS drug_exposure (
            drug_exposure_id BIGINT PRIMARY KEY,
            person_id BIGINT NOT NULL REFERENCES person(person_id),
            drug_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            drug_exposure_start_date TEXT,
            drug_type_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            visit_occurrence_id BIGINT REFERENCES visit_occurrence(visit_occurrence_id),
            drug_source_value TEXT,
            drug_source_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            note_id BIGINT NOT NULL REFERENCES note(note_id),
            note_nlp_id BIGINT NOT NULL REFERENCES note_nlp(note_nlp_id),
            source_note_hash TEXT NOT NULL,
            idempotent_key TEXT NOT NULL
        )
    """,
    "measurement": """
        CREATE TABLE IF NOT EXISTS measurement (
            measurement_id BIGINT PRIMARY KEY,
            person_id BIGINT NOT NULL REFERENCES person(person_id),
            measurement_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            measurement_date TEXT,
            measurement_type_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            visit_occurrence_id BIGINT REFERENCES visit_occurrence(visit_occurrence_id),
            measurement_source_value TEXT,
            measurement_source_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            note_id BIGINT NOT NULL REFERENCES note(note_id),
            note_nlp_id BIGINT NOT NULL REFERENCES note_nlp(note_nlp_id),
            source_note_hash TEXT NOT NULL,
            idempotent_key TEXT NOT NULL
        )
    """,
    "procedure_occurrence": """
        CREATE TABLE IF NOT EXISTS procedure_occurrence (
            procedure_occurrence_id BIGINT PRIMARY KEY,
            person_id BIGINT NOT NULL REFERENCES person(person_id),
            procedure_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            procedure_date TEXT,
            procedure_type_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            visit_occurrence_id BIGINT REFERENCES visit_occurrence(visit_occurrence_id),
            procedure_source_value TEXT,
            procedure_source_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            note_id BIGINT NOT NULL REFERENCES note(note_id),
            note_nlp_id BIGINT NOT NULL REFERENCES note_nlp(note_nlp_id),
            source_note_hash TEXT NOT NULL,
            idempotent_key TEXT NOT NULL
        )
    """,
    "observation": """
        CREATE TABLE IF NOT EXISTS observation (
            observation_id BIGINT PRIMARY KEY,
            person_id BIGINT NOT NULL REFERENCES person(person_id),
            observation_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            observation_date TEXT,
            observation_type_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            visit_occurrence_id BIGINT REFERENCES visit_occurrence(visit_occurrence_id),
            observation_source_value TEXT,
            observation_source_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            note_id BIGINT NOT NULL REFERENCES note(note_id),
            note_nlp_id BIGINT NOT NULL REFERENCES note_nlp(note_nlp_id),
            source_note_hash TEXT NOT NULL,
            idempotent_key TEXT NOT NULL
        )
    """,
    "source_to_concept_map": """
        CREATE TABLE IF NOT EXISTS source_to_concept_map (
            source_to_concept_map_id BIGINT PRIMARY KEY,
            source_code TEXT,
            source_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            source_vocabulary_id TEXT,
            source_code_description TEXT,
            target_concept_id BIGINT NOT NULL REFERENCES concept(concept_id),
            target_vocabulary_id TEXT,
            valid_start_date TEXT,
            valid_end_date TEXT,
            invalid_reason TEXT,
            vocabulary_version TEXT,
            source_note_hash TEXT NOT NULL,
            note_nlp_id BIGINT NOT NULL REFERENCES note_nlp(note_nlp_id)
        )
    """,
}

_NOTE_ID_FIELDS = ("note_id", "document_id", "doc_id", "source_document_id")
_NOTE_TEXT_FIELDS = ("note_text", "text", "document_text", "source_text")
_NOTE_DATE_FIELDS = ("note_date", "document_date", "date")
_NOTE_HASH_FIELDS = ("source_note_hash", "note_hash", "document_hash")
_PERSON_FIELDS = ("person_id", "patient_id", "person_source_value", "subject_id")
_VISIT_FIELDS = ("visit_id", "encounter_id", "visit_source_value")
_ENTITY_FIELDS = ("entities", "clinical_entities", "spans", "grounded_spans")
_ENTITY_TEXT_FIELDS = (
    "lexical_variant",
    "normalized_text",
    "text",
    "entity_text",
    "word",
    "surface",
    "source_value",
)
_DOMAIN_FIELDS = ("domain_id", "omop_domain", "domain", "entity_label", "label")
_CONCEPT_ID_FIELDS = (
    "concept_id",
    "standard_concept_id",
    "target_concept_id",
    "note_nlp_concept_id",
)
_SOURCE_CONCEPT_ID_FIELDS = ("source_concept_id", "note_nlp_source_concept_id")
_VOCABULARY_FIELDS = ("vocabulary_id", "source_vocabulary_id", "system")
_CODE_FIELDS = ("concept_code", "code", "source_code", "coding_code")
_CONCEPT_NAME_FIELDS = ("concept_name", "display", "code_display")
_METADATA_FIELDS = ("metadata", "meta")
_CODING_FIELDS = ("coding", "codings", "code", "codeable_concept")


@dataclass(frozen=True)
class RejectedSpan:
    """PHI-free rejection detail for an entity that was not loaded."""

    reason: str
    source_note_hash: str
    start: int | None = None
    end: int | None = None
    domain: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable rejection record."""

        return asdict(self)


@dataclass(frozen=True)
class OmopLoadSummary:
    """PHI-free aggregate summary for one OMOP load build."""

    row_counts: Mapping[str, int]
    rejection_counts: Mapping[str, int]
    rejected_spans: tuple[RejectedSpan, ...] = field(default_factory=tuple)
    source_note_hashes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible summary."""

        return {
            "row_counts": dict(self.row_counts),
            "rejection_counts": dict(self.rejection_counts),
            "rejected_spans": [span.to_dict() for span in self.rejected_spans],
            "source_note_hashes": list(self.source_note_hashes),
        }


@dataclass(frozen=True)
class OmopConstraintViolation:
    """Validation violation emitted without raw note text or identifiers."""

    table: str
    column: str
    reason: str
    row_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable validation violation."""

        return asdict(self)


@dataclass(frozen=True)
class OmopCdmTables:
    """In-memory OMOP CDM tables and PHI-free load summary."""

    tables: Mapping[str, tuple[dict[str, Any], ...]]
    summary: OmopLoadSummary

    def table(self, name: str) -> tuple[dict[str, Any], ...]:
        """Return rows for a named CDM table."""

        return self.tables.get(name, ())

    @property
    def row_counts(self) -> Mapping[str, int]:
        """Return per-table row counts."""

        return self.summary.row_counts

    def to_dict(self) -> dict[str, Any]:
        """Return all CDM tables and the aggregate summary."""

        return {
            "tables": {name: list(self.table(name)) for name in _TABLE_ORDER},
            "summary": self.summary.to_dict(),
        }


def load_grounded_notes(
    notes: Iterable[Any],
    *,
    vocabulary_version: str | None = None,
    mode: LoadMode = "append",
    quality_floor: float | None = None,
) -> OmopCdmTables:
    """Build OMOP CDM tables from grounded clinical note records.

    Args:
        notes: Iterable of mappings or objects with note metadata and entity
            spans. Entity spans must include valid ``start``/``end`` offsets to
            produce NOTE_NLP rows.
        vocabulary_version: Optional user-supplied vocabulary version string
            recorded in SOURCE_TO_CONCEPT_MAP rows.
        mode: Load mode. The first implementation supports append mode with
            deterministic upsert keys for idempotent reruns.
        quality_floor: Optional completeness floor. When set, the extracted
            batch is profiled before loading and low-completeness batches are
            rejected without writing rows.

    Returns:
        In-memory CDM rows plus a PHI-free summary.

    Raises:
        ValueError: If an unsupported mode is requested or required note fields
            are missing.
    """

    if mode != "append":
        raise ValueError("OMOP loader currently supports append mode only")

    if quality_floor is not None:
        records = tuple(notes)
        from openmed.structured.quality import (
            assert_profile_gate,
            profile_extracted_batch,
        )

        assert_profile_gate(
            profile_extracted_batch(records, completeness_floor=quality_floor)
        )
        notes = records

    table_rows: dict[str, dict[int, dict[str, Any]]] = {
        table: {} for table in _TABLE_ORDER
    }
    rejections: list[RejectedSpan] = []
    note_hashes: set[str] = set()

    _add_concept_row(table_rows, _unmapped_concept_row())

    for note_index, note in enumerate(notes):
        note_source_value = _required_text(
            _first_value((note,), _NOTE_ID_FIELDS),
            f"note at index {note_index} is missing note_id/document_id",
        )
        person_source_value = _required_text(
            _first_value((note,), _PERSON_FIELDS),
            f"note at index {note_index} is missing person_id/patient_id",
        )
        note_text = _first_text((note,), _NOTE_TEXT_FIELDS)
        source_note_hash = _source_note_hash(note, note_text, note_source_value)
        note_hashes.add(source_note_hash)

        person_id = deterministic_omop_id("person", person_source_value)
        visit_source_value = _first_text((note,), _VISIT_FIELDS)
        if not visit_source_value:
            visit_source_value = f"note:{source_note_hash}"
        visit_occurrence_id = deterministic_omop_id(
            "visit",
            person_source_value,
            visit_source_value,
        )
        note_id = deterministic_omop_id("note", person_source_value, source_note_hash)
        note_date = _first_text((note,), _NOTE_DATE_FIELDS) or None

        _upsert_row(
            table_rows,
            "person",
            {
                "person_id": person_id,
                "person_source_value": person_source_value,
            },
        )
        _upsert_row(
            table_rows,
            "visit_occurrence",
            {
                "visit_occurrence_id": visit_occurrence_id,
                "person_id": person_id,
                "visit_concept_id": UNMAPPED_CONCEPT_ID,
                "visit_start_date": note_date,
                "visit_source_value": visit_source_value,
                "visit_source_concept_id": UNMAPPED_CONCEPT_ID,
            },
        )
        _upsert_row(
            table_rows,
            "note",
            {
                "note_id": note_id,
                "person_id": person_id,
                "visit_occurrence_id": visit_occurrence_id,
                "note_date": note_date,
                "note_type_concept_id": UNMAPPED_CONCEPT_ID,
                "note_class_concept_id": UNMAPPED_CONCEPT_ID,
                "note_title": None,
                "note_text": note_text,
                "encoding_concept_id": UNMAPPED_CONCEPT_ID,
                "language_concept_id": UNMAPPED_CONCEPT_ID,
                "note_source_value": note_source_value,
                "source_note_hash": source_note_hash,
            },
        )

        for entity in _note_entities(note):
            rejection = _load_entity(
                table_rows,
                entity,
                person_id=person_id,
                visit_occurrence_id=visit_occurrence_id,
                note_id=note_id,
                note_date=note_date,
                note_text=note_text,
                source_note_hash=source_note_hash,
                vocabulary_version=vocabulary_version or "",
            )
            if rejection is not None:
                rejections.append(rejection)

    tables = {
        table: tuple(
            _ordered_row(table, row)
            for row in sorted(
                rows.values(), key=lambda item: item[_PRIMARY_KEYS[table]]
            )
        )
        for table, rows in table_rows.items()
    }
    summary = OmopLoadSummary(
        row_counts={table: len(tables[table]) for table in _TABLE_ORDER},
        rejection_counts=dict(Counter(span.reason for span in rejections)),
        rejected_spans=tuple(rejections),
        source_note_hashes=tuple(sorted(note_hashes)),
    )
    return OmopCdmTables(tables=tables, summary=summary)


def load_grounded_jsonl(
    path: str | Path,
    *,
    vocabulary_version: str | None = None,
    mode: LoadMode = "append",
    quality_floor: float | None = None,
) -> OmopCdmTables:
    """Load grounded note records from JSONL into in-memory OMOP tables."""

    records = []
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(payload)
    return load_grounded_notes(
        records,
        vocabulary_version=vocabulary_version,
        mode=mode,
        quality_floor=quality_floor,
    )


def write_omop_duckdb(
    tables: OmopCdmTables,
    target: str | Path | Any = ":memory:",
    *,
    mode: LoadMode = "append",
) -> Any:
    """Create/update an OMOP CDM schema in DuckDB and return the connection."""

    if mode != "append":
        raise ValueError("OMOP DuckDB writer currently supports append mode only")
    duckdb = _load_optional("duckdb", "openmed[duckdb]")
    con = target if hasattr(target, "execute") else duckdb.connect(str(target))
    create_omop_schema(con)
    _upsert_sql_tables(con, tables)
    return con


def write_omop_sqlite(
    tables: OmopCdmTables,
    target: str | Path | sqlite3.Connection = ":memory:",
    *,
    mode: LoadMode = "append",
) -> sqlite3.Connection:
    """Create/update an OMOP CDM schema in SQLite and return the connection."""

    if mode != "append":
        raise ValueError("OMOP SQLite writer currently supports append mode only")
    con = target if isinstance(target, sqlite3.Connection) else sqlite3.connect(target)
    con.execute("PRAGMA foreign_keys = ON")
    create_omop_schema(con)
    _upsert_sql_tables(con, tables)
    con.commit()
    return con


def write_omop_parquet(
    tables: OmopCdmTables,
    directory: str | Path,
    *,
    mode: LoadMode = "append",
) -> Path:
    """Write OMOP CDM tables as one Parquet file per table.

    Existing rows are merged by primary key, so repeated append-mode writes are
    idempotent for the same deterministic loader output.
    """

    if mode != "append":
        raise ValueError("OMOP Parquet writer currently supports append mode only")
    pa = _load_optional("pyarrow", "openmed[columnar]")
    pq = _load_optional("pyarrow.parquet", "openmed[columnar]")
    target_dir = Path(directory).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    for table_name in _TABLE_ORDER:
        path = target_dir / f"{table_name}.parquet"
        merged = _merge_parquet_rows(path, tables.table(table_name), table_name)
        arrow_table = pa.Table.from_pylist(
            [_ordered_row(table_name, row) for row in merged],
            schema=_arrow_schema(pa, table_name),
        )
        pq.write_table(arrow_table, path)

    return target_dir


def create_omop_schema(con: Any) -> Any:
    """Create the minimal OMOP CDM table subset on a DB-API connection."""

    for table in _TABLE_ORDER:
        con.execute(_SQL_DDL[table])
    return con


def validate_omop_tables(
    tables: OmopCdmTables,
) -> tuple[OmopConstraintViolation, ...]:
    """Validate CDM concept references, NOTE_NLP offsets, and row reachability."""

    violations: list[OmopConstraintViolation] = []
    concept_ids = {int(row["concept_id"]) for row in tables.table("concept")}
    note_rows = {int(row["note_id"]): row for row in tables.table("note")}
    note_nlp_rows = {int(row["note_nlp_id"]): row for row in tables.table("note_nlp")}

    for table in (
        "visit_occurrence",
        "note",
        "note_nlp",
        "condition_occurrence",
        "drug_exposure",
        "measurement",
        "procedure_occurrence",
        "observation",
        "source_to_concept_map",
    ):
        for row in tables.table(table):
            row_id = int(row[_PRIMARY_KEYS[table]])
            for column in _concept_reference_columns(table):
                if int(row[column]) not in concept_ids:
                    violations.append(
                        OmopConstraintViolation(
                            table=table,
                            column=column,
                            reason="missing_concept",
                            row_id=row_id,
                        )
                    )

    for row_id, row in note_nlp_rows.items():
        note = note_rows.get(int(row["note_id"]))
        if note is None:
            violations.append(
                OmopConstraintViolation(
                    table="note_nlp",
                    column="note_id",
                    reason="missing_note",
                    row_id=row_id,
                )
            )
            continue
        note_text = str(note.get("note_text") or "")
        start = int(row["offset"])
        end = int(row["offset_end"])
        if start < 0 or end < start or end > len(note_text):
            violations.append(
                OmopConstraintViolation(
                    table="note_nlp",
                    column="offset",
                    reason="invalid_note_offset",
                    row_id=row_id,
                )
            )

    for table in _DOMAIN_SOURCE_CONCEPT_COLUMNS:
        for row in tables.table(table):
            row_id = int(row[_PRIMARY_KEYS[table]])
            if int(row["note_nlp_id"]) not in note_nlp_rows:
                violations.append(
                    OmopConstraintViolation(
                        table=table,
                        column="note_nlp_id",
                        reason="missing_note_nlp",
                        row_id=row_id,
                    )
                )

    return tuple(violations)


def validate_omop_database(con: Any) -> tuple[OmopConstraintViolation, ...]:
    """Validate persisted OMOP tables using the same PHI-free violation shape."""

    tables = {
        table: tuple(_select_all(con, table))
        for table in _TABLE_ORDER
        if _table_exists(con, table)
    }
    return validate_omop_tables(
        OmopCdmTables(
            tables=tables,
            summary=OmopLoadSummary(
                row_counts={
                    table: len(tables.get(table, ())) for table in _TABLE_ORDER
                },
                rejection_counts={},
            ),
        )
    )


def deterministic_omop_id(*parts: Any) -> int:
    """Return a stable positive integer key derived from source identifiers."""

    if not parts:
        raise ValueError("at least one key part is required")
    canonical = "\x1f".join(str(part) for part in parts)
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)


def deterministic_note_hash(note_text: str) -> str:
    """Return a SHA-256 hash for source note text."""

    return hashlib.sha256(note_text.encode("utf-8")).hexdigest()


def _load_entity(
    table_rows: dict[str, dict[int, dict[str, Any]]],
    entity: Any,
    *,
    person_id: int,
    visit_occurrence_id: int,
    note_id: int,
    note_date: str | None,
    note_text: str,
    source_note_hash: str,
    vocabulary_version: str,
) -> RejectedSpan | None:
    domain = _entity_domain(entity)
    start = _optional_int(_first_value(_entity_sources(entity), ("start",)))
    end = _optional_int(_first_value(_entity_sources(entity), ("end",)))

    if domain is None:
        return RejectedSpan(
            reason="unsupported_domain",
            source_note_hash=source_note_hash,
            start=start,
            end=end,
            domain=_first_text(_entity_sources(entity), _DOMAIN_FIELDS) or None,
        )
    if start is None or end is None:
        return RejectedSpan(
            reason="missing_offset",
            source_note_hash=source_note_hash,
            start=start,
            end=end,
            domain=domain,
        )
    if start < 0 or end < start or end > len(note_text):
        return RejectedSpan(
            reason="invalid_offset",
            source_note_hash=source_note_hash,
            start=start,
            end=end,
            domain=domain,
        )

    concept = _entity_concept(entity, domain)
    _add_concept_row(table_rows, concept)

    lexical_variant = _entity_text(entity) or note_text[start:end]
    idempotent_key = _idempotent_key(
        person_id=person_id,
        source_note_hash=source_note_hash,
        start=start,
        end=end,
        concept_id=int(concept["concept_id"]),
    )
    note_nlp_id = deterministic_omop_id("note_nlp", idempotent_key)
    table_name, row_id_column, concept_column, source_value_column = _DOMAIN_TABLES[
        domain
    ]
    domain_row_id = deterministic_omop_id(table_name, idempotent_key)
    source_concept_column = _DOMAIN_SOURCE_CONCEPT_COLUMNS[table_name]

    _upsert_row(
        table_rows,
        "note_nlp",
        {
            "note_nlp_id": note_nlp_id,
            "note_id": note_id,
            "section_concept_id": UNMAPPED_CONCEPT_ID,
            "snippet": None,
            "offset": start,
            "offset_end": end,
            "lexical_variant": lexical_variant,
            "note_nlp_concept_id": int(concept["concept_id"]),
            "note_nlp_source_concept_id": int(concept["source_concept_id"]),
            "nlp_system": "openmed.interop.omop.cdm_loader",
            "nlp_date": None,
            "term_exists": "Y",
            "term_temporal": None,
            "term_modifiers": None,
            "note_nlp_event_id": domain_row_id,
            "note_nlp_event_field_concept_id": UNMAPPED_CONCEPT_ID,
        },
    )
    _upsert_row(
        table_rows,
        table_name,
        {
            row_id_column: domain_row_id,
            "person_id": person_id,
            concept_column: int(concept["concept_id"]),
            _domain_date_column(table_name): note_date,
            _domain_type_column(table_name): UNMAPPED_CONCEPT_ID,
            "visit_occurrence_id": visit_occurrence_id,
            source_value_column: lexical_variant,
            source_concept_column: int(concept["source_concept_id"]),
            "note_id": note_id,
            "note_nlp_id": note_nlp_id,
            "source_note_hash": source_note_hash,
            "idempotent_key": idempotent_key,
        },
    )
    _upsert_row(
        table_rows,
        "source_to_concept_map",
        {
            "source_to_concept_map_id": deterministic_omop_id(
                "source_to_concept_map",
                idempotent_key,
            ),
            "source_code": concept["concept_code"],
            "source_concept_id": int(concept["source_concept_id"]),
            "source_vocabulary_id": concept["vocabulary_id"],
            "source_code_description": lexical_variant,
            "target_concept_id": int(concept["concept_id"]),
            "target_vocabulary_id": concept["vocabulary_id"],
            "valid_start_date": None,
            "valid_end_date": None,
            "invalid_reason": None if int(concept["concept_id"]) else "UNMAPPED",
            "vocabulary_version": vocabulary_version,
            "source_note_hash": source_note_hash,
            "note_nlp_id": note_nlp_id,
        },
    )
    return None


def _entity_concept(entity: Any, domain: OmopDomain) -> dict[str, Any]:
    concept_id = _optional_int(
        _first_value(_entity_sources(entity), _CONCEPT_ID_FIELDS)
    )
    source_concept_id = _optional_int(
        _first_value(_entity_sources(entity), _SOURCE_CONCEPT_ID_FIELDS)
    )
    vocabulary_id = _first_text(_entity_and_coding_sources(entity), _VOCABULARY_FIELDS)
    concept_code = _first_text(_entity_and_coding_sources(entity), _CODE_FIELDS)
    concept_name = _first_text(_entity_and_coding_sources(entity), _CONCEPT_NAME_FIELDS)

    if concept_id is None or concept_id <= 0:
        return {
            "concept_id": UNMAPPED_CONCEPT_ID,
            "source_concept_id": UNMAPPED_CONCEPT_ID,
            "concept_name": UNMAPPED_CONCEPT_NAME,
            "domain_id": domain,
            "vocabulary_id": vocabulary_id or UNMAPPED_VOCABULARY_ID,
            "concept_class_id": "",
            "standard_concept": None,
            "concept_code": concept_code,
        }

    return {
        "concept_id": concept_id,
        "source_concept_id": source_concept_id or concept_id,
        "concept_name": concept_name or _entity_text(entity),
        "domain_id": domain,
        "vocabulary_id": vocabulary_id,
        "concept_class_id": "",
        "standard_concept": "S",
        "concept_code": concept_code,
    }


def _add_concept_row(
    table_rows: dict[str, dict[int, dict[str, Any]]],
    concept: Mapping[str, Any],
) -> None:
    concept_id = int(concept["concept_id"])
    _upsert_row(
        table_rows,
        "concept",
        {
            "concept_id": concept_id,
            "concept_name": concept.get("concept_name") or UNMAPPED_CONCEPT_NAME,
            "domain_id": concept.get("domain_id") or "",
            "vocabulary_id": concept.get("vocabulary_id") or UNMAPPED_VOCABULARY_ID,
            "concept_class_id": concept.get("concept_class_id") or "",
            "standard_concept": concept.get("standard_concept"),
            "concept_code": concept.get("concept_code") or "",
        },
    )
    source_concept_id = int(concept.get("source_concept_id") or concept_id)
    if source_concept_id != concept_id:
        _upsert_row(
            table_rows,
            "concept",
            {
                "concept_id": source_concept_id,
                "concept_name": concept.get("concept_name") or "",
                "domain_id": concept.get("domain_id") or "",
                "vocabulary_id": concept.get("vocabulary_id") or "",
                "concept_class_id": concept.get("concept_class_id") or "",
                "standard_concept": concept.get("standard_concept"),
                "concept_code": concept.get("concept_code") or "",
            },
        )


def _unmapped_concept_row() -> dict[str, Any]:
    return {
        "concept_id": UNMAPPED_CONCEPT_ID,
        "source_concept_id": UNMAPPED_CONCEPT_ID,
        "concept_name": UNMAPPED_CONCEPT_NAME,
        "domain_id": "",
        "vocabulary_id": UNMAPPED_VOCABULARY_ID,
        "concept_class_id": "",
        "standard_concept": None,
        "concept_code": "",
    }


def _source_note_hash(note: Any, note_text: str, note_source_value: str) -> str:
    explicit_hash = _first_text((note,), _NOTE_HASH_FIELDS)
    if explicit_hash:
        return explicit_hash
    if note_text:
        return deterministic_note_hash(note_text)
    return deterministic_note_hash(note_source_value)


def _idempotent_key(
    *,
    person_id: int,
    source_note_hash: str,
    start: int,
    end: int,
    concept_id: int,
) -> str:
    payload = f"{person_id}\x1f{source_note_hash}\x1f{start}:{end}\x1f{concept_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _entity_domain(entity: Any) -> OmopDomain | None:
    for source in _entity_sources(entity):
        value = _first_text((source,), _DOMAIN_FIELDS)
        if not value:
            continue
        normalized = value.strip().replace("-", "_").replace(" ", "_").casefold()
        domain = _DOMAIN_ALIASES.get(normalized)
        if domain is not None:
            return domain
    return None


def _note_entities(note: Any) -> tuple[Any, ...]:
    value = _first_value((note,), _ENTITY_FIELDS)
    if value is _MISSING or value is None:
        return ()
    return tuple(_as_sequence(value))


def _entity_text(entity: Any) -> str:
    return _first_text(_entity_sources(entity), _ENTITY_TEXT_FIELDS)


def _entity_sources(entity: Any) -> tuple[Any, ...]:
    sources: list[Any] = [entity]
    for source in tuple(sources):
        for name in _METADATA_FIELDS:
            metadata = _value(source, name)
            if metadata is not _MISSING and metadata is not None:
                sources.append(metadata)
    return tuple(sources)


def _entity_and_coding_sources(entity: Any) -> tuple[Any, ...]:
    sources = list(_entity_sources(entity))
    for source in tuple(sources):
        for name in _CODING_FIELDS:
            coding = _coerce_coding(_value(source, name))
            if coding is not None:
                sources.insert(0, coding)
    return tuple(sources)


def _coerce_coding(value: Any) -> Mapping[str, Any] | None:
    if value is _MISSING or value is None or isinstance(value, (str, bytes)):
        return None
    if isinstance(value, Mapping):
        nested = _coerce_coding(_value(value, "coding"))
        if nested is not None:
            return nested
        return value
    if isinstance(value, Sequence):
        for item in value:
            coding = _coerce_coding(item)
            if coding is not None:
                return coding
    return None


def _upsert_row(
    table_rows: dict[str, dict[int, dict[str, Any]]],
    table: str,
    row: Mapping[str, Any],
) -> None:
    primary_key = _PRIMARY_KEYS[table]
    table_rows[table][int(row[primary_key])] = _ordered_row(table, row)


def _ordered_row(table: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {column: row.get(column) for column in _SCHEMA_COLUMNS[table]}


def _upsert_sql_tables(con: Any, tables: OmopCdmTables) -> None:
    for table in _TABLE_ORDER:
        rows = tables.table(table)
        if not rows:
            continue
        columns = _SCHEMA_COLUMNS[table]
        column_sql = ", ".join(_quote_identifier(column) for column in columns)
        placeholder_sql = ", ".join("?" for _ in columns)
        statement = (
            f"INSERT OR IGNORE INTO {table} ({column_sql}) VALUES ({placeholder_sql})"
        )
        values = [tuple(row.get(column) for column in columns) for row in rows]
        con.executemany(statement, values)


def _merge_parquet_rows(
    path: Path,
    new_rows: Sequence[Mapping[str, Any]],
    table_name: str,
) -> list[dict[str, Any]]:
    rows_by_key: dict[int, dict[str, Any]] = {}
    primary_key = _PRIMARY_KEYS[table_name]
    if path.exists():
        pq = _load_optional("pyarrow.parquet", "openmed[columnar]")
        for row in pq.read_table(path).to_pylist():
            rows_by_key[int(row[primary_key])] = _ordered_row(table_name, row)
    for row in new_rows:
        rows_by_key[int(row[primary_key])] = _ordered_row(table_name, row)
    return [rows_by_key[key] for key in sorted(rows_by_key)]


def _arrow_schema(pa: Any, table_name: str) -> Any:
    integer_columns = {
        _PRIMARY_KEYS[table_name],
        "person_id",
        "visit_occurrence_id",
        "note_id",
        "note_nlp_id",
        "note_nlp_event_id",
        "offset",
        "offset_end",
        *_concept_reference_columns(table_name),
    }
    fields = [
        pa.field(column, pa.int64() if column in integer_columns else pa.string())
        for column in _SCHEMA_COLUMNS[table_name]
    ]
    return pa.schema(fields)


def _select_all(con: Any, table: str) -> list[dict[str, Any]]:
    columns = _SCHEMA_COLUMNS[table]
    column_sql = ", ".join(_quote_identifier(column) for column in columns)
    rows = con.execute(f"SELECT {column_sql} FROM {table}").fetchall()
    return [dict(zip(columns, row)) for row in rows]


def _table_exists(con: Any, table: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {table} LIMIT 1")
    except Exception:
        return False
    return True


def _concept_reference_columns(table: str) -> tuple[str, ...]:
    return tuple(
        column
        for column in _SCHEMA_COLUMNS[table]
        if column.endswith("_concept_id")
        or column in {"source_concept_id", "target_concept_id"}
    )


def _domain_date_column(table_name: str) -> str:
    if table_name == "drug_exposure":
        return "drug_exposure_start_date"
    if table_name == "condition_occurrence":
        return "condition_start_date"
    if table_name == "procedure_occurrence":
        return "procedure_date"
    return f"{table_name}_date"


def _domain_type_column(table_name: str) -> str:
    if table_name == "drug_exposure":
        return "drug_type_concept_id"
    if table_name == "condition_occurrence":
        return "condition_type_concept_id"
    if table_name == "procedure_occurrence":
        return "procedure_type_concept_id"
    return f"{table_name}_type_concept_id"


def _first_value(sources: Iterable[Any], names: Iterable[str]) -> Any:
    for source in sources:
        for name in names:
            value = _value(source, name)
            if value is not _MISSING and value is not None:
                return value
    return _MISSING


def _first_text(sources: Iterable[Any], names: Iterable[str]) -> str:
    value = _first_value(sources, names)
    if value is _MISSING or value is None:
        return ""
    return str(value).strip()


def _required_text(value: Any, message: str) -> str:
    if value is _MISSING or value is None or not str(value).strip():
        raise ValueError(message)
    return str(value).strip()


def _value(item: Any, name: str) -> Any:
    if isinstance(item, Mapping) and name in item:
        return item[name]
    if hasattr(item, name):
        return getattr(item, name)
    return _MISSING


def _as_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return value
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _optional_int(value: Any) -> int | None:
    if value is _MISSING or value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_optional(module: str, extra: str) -> Any:
    try:
        return import_module(module)
    except ImportError as exc:
        raise ImportError(f"{module} support requires {extra}") from exc


def _quote_identifier(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


__all__ = [
    "LoadMode",
    "OmopCdmTables",
    "OmopConstraintViolation",
    "OmopDomain",
    "OmopLoadSummary",
    "RejectedSpan",
    "UNMAPPED_CONCEPT_ID",
    "UNMAPPED_CONCEPT_NAME",
    "UNMAPPED_VOCABULARY_ID",
    "WriterKind",
    "create_omop_schema",
    "deterministic_note_hash",
    "deterministic_omop_id",
    "load_grounded_jsonl",
    "load_grounded_notes",
    "validate_omop_database",
    "validate_omop_tables",
    "write_omop_duckdb",
    "write_omop_parquet",
    "write_omop_sqlite",
]
