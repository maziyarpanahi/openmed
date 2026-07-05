from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from openmed.interop import adapter_spec, available_adapters, get_adapter
from openmed.interop.omop import (
    UNMAPPED_CONCEPT_ID,
    load_grounded_jsonl,
    load_grounded_notes,
    validate_omop_database,
    validate_omop_tables,
    write_omop_duckdb,
    write_omop_parquet,
    write_omop_sqlite,
)

NOTE_TEXT = (
    "Patient Alice reports diabetes. Aspirin started. A1c 8.2. "
    "Appendectomy completed. Lives alone. Mystery term noted."
)


def _entity(
    surface: str,
    *,
    domain: str,
    concept_id: int | None = None,
    code: str = "",
) -> dict[str, Any]:
    start = NOTE_TEXT.index(surface)
    return {
        "text": surface,
        "domain_id": domain,
        "start": start,
        "end": start + len(surface),
        "concept_id": concept_id,
        "code": code,
        "vocabulary_id": "LOCAL",
        "concept_name": f"Synthetic {surface}",
    }


def _fixture_notes() -> list[dict[str, Any]]:
    return [
        {
            "document_id": "secret-note-456",
            "person_id": "secret-patient-123",
            "visit_id": "visit-1",
            "note_date": "2026-01-02",
            "note_text": NOTE_TEXT,
            "entities": [
                _entity(
                    "diabetes",
                    domain="Condition",
                    concept_id=201826,
                    code="COND-1",
                ),
                _entity("Aspirin", domain="Drug", concept_id=1112807, code="DRUG-1"),
                _entity(
                    "A1c",
                    domain="Measurement",
                    concept_id=3004410,
                    code="MEAS-1",
                ),
                _entity(
                    "Appendectomy",
                    domain="Procedure",
                    concept_id=4017990,
                    code="PROC-1",
                ),
                _entity(
                    "Lives alone",
                    domain="Observation",
                    concept_id=40766527,
                    code="OBS-1",
                ),
                _entity("Mystery term", domain="Condition", code="SRC-UNMAPPED"),
                {
                    "text": "Alice",
                    "domain_id": "Anatomy",
                    "start": NOTE_TEXT.index("Alice"),
                    "end": NOTE_TEXT.index("Alice") + len("Alice"),
                },
            ],
        }
    ]


def _table_counts_from_duckdb(con: Any) -> dict[str, int]:
    return {
        table: con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in _expected_counts()
    }


def _table_counts_from_sqlite(con: sqlite3.Connection) -> dict[str, int]:
    return {
        table: con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in _expected_counts()
    }


def _table_counts_from_parquet(directory: Path) -> dict[str, int]:
    return {
        path.stem: pq.read_table(path).num_rows
        for path in sorted(directory.glob("*.parquet"))
    }


def _expected_counts() -> Mapping[str, int]:
    return {
        "concept": 6,
        "person": 1,
        "visit_occurrence": 1,
        "note": 1,
        "note_nlp": 6,
        "condition_occurrence": 2,
        "drug_exposure": 1,
        "measurement": 1,
        "procedure_occurrence": 1,
        "observation": 1,
        "source_to_concept_map": 6,
    }


def test_load_grounded_notes_builds_valid_duckdb_omop_tables() -> None:
    tables = load_grounded_notes(_fixture_notes(), vocabulary_version="synthetic-v1")

    assert tables.row_counts == _expected_counts()
    assert tables.summary.rejection_counts == {"unsupported_domain": 1}
    assert validate_omop_tables(tables) == ()

    con = write_omop_duckdb(tables)

    assert validate_omop_database(con) == ()
    assert _table_counts_from_duckdb(con) == _expected_counts()
    assert con.execute(
        """
        SELECT condition_concept_id
        FROM condition_occurrence
        WHERE condition_source_value = 'Mystery term'
        """
    ).fetchall() == [(UNMAPPED_CONCEPT_ID,)]
    assert con.execute(
        """
        SELECT target_concept_id, invalid_reason
        FROM source_to_concept_map
        WHERE source_code = 'SRC-UNMAPPED'
        """
    ).fetchall() == [(UNMAPPED_CONCEPT_ID, "UNMAPPED")]


def test_append_mode_is_idempotent_for_duckdb_sqlite_and_parquet(
    tmp_path: Path,
) -> None:
    tables = load_grounded_notes(_fixture_notes(), vocabulary_version="synthetic-v1")

    duckdb_con = write_omop_duckdb(tables)
    write_omop_duckdb(tables, duckdb_con)
    assert _table_counts_from_duckdb(duckdb_con) == _expected_counts()

    sqlite_con = write_omop_sqlite(tables, tmp_path / "omop.sqlite")
    write_omop_sqlite(tables, sqlite_con)
    assert _table_counts_from_sqlite(sqlite_con) == _expected_counts()

    parquet_dir = write_omop_parquet(tables, tmp_path / "parquet")
    write_omop_parquet(tables, parquet_dir)
    assert _table_counts_from_parquet(parquet_dir) == _expected_counts()


def test_every_domain_row_is_reachable_from_note_nlp() -> None:
    tables = load_grounded_notes(_fixture_notes())
    note = tables.table("note")[0]
    note_nlp_rows = {row["note_nlp_id"]: row for row in tables.table("note_nlp")}

    for row in note_nlp_rows.values():
        assert row["note_id"] == note["note_id"]
        assert 0 <= row["offset"] <= row["offset_end"] <= len(NOTE_TEXT)

    for table_name in (
        "condition_occurrence",
        "drug_exposure",
        "measurement",
        "procedure_occurrence",
        "observation",
    ):
        for row in tables.table(table_name):
            note_nlp = note_nlp_rows[row["note_nlp_id"]]
            primary_key = next(key for key in row if key.endswith("_id"))
            assert note_nlp["note_nlp_event_id"] == row[primary_key]


def test_summary_and_rejection_output_are_phi_free(caplog: Any) -> None:
    tables = load_grounded_notes(_fixture_notes())
    summary_payload = json.dumps(tables.summary.to_dict(), sort_keys=True)

    assert caplog.records == []
    assert "Patient Alice" not in summary_payload
    assert NOTE_TEXT not in summary_payload
    assert "secret-note-456" not in summary_payload
    assert "secret-patient-123" not in summary_payload
    assert "Alice" not in summary_payload
    assert "unsupported_domain" in summary_payload


def test_load_grounded_jsonl_matches_in_memory_loader(tmp_path: Path) -> None:
    jsonl = tmp_path / "grounded.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(record) for record in _fixture_notes()) + "\n",
        encoding="utf-8",
    )

    from_jsonl = load_grounded_jsonl(jsonl, vocabulary_version="synthetic-v1")
    from_memory = load_grounded_notes(
        _fixture_notes(), vocabulary_version="synthetic-v1"
    )

    assert from_jsonl.to_dict() == from_memory.to_dict()


def test_omop_loader_quality_floor_blocks_low_completeness_batch() -> None:
    with pytest.raises(ValueError, match="quality profile gate failed"):
        load_grounded_notes(_fixture_notes(), quality_floor=0.99)


def test_omop_loader_is_available_through_interop_registry() -> None:
    adapter = get_adapter("omop")

    assert "omop" in available_adapters()
    assert adapter_spec("omop").module == "openmed.interop.omop"
    assert hasattr(adapter, "load_grounded_notes")
