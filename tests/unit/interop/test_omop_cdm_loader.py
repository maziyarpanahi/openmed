from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from openmed.interop import adapter_spec, available_adapters, get_adapter
from openmed.interop.omop import (
    UNMAPPED_CONCEPT_ID,
    OmopCdmTables,
    OmopLoadSummary,
    VocabularyRouter,
    emit_postgres_ddl,
    load_grounded_jsonl,
    load_grounded_notes,
    validate_omop_database,
    validate_omop_database_report,
    validate_omop_tables,
    validate_omop_tables_report,
    write_omop_duckdb,
    write_omop_parquet,
    write_omop_sqlite,
)

NOTE_TEXT = (
    "Patient Alice reports diabetes. Aspirin started. A1c 8.2. "
    "Appendectomy completed. Lives alone. Mystery term noted."
)
TARGET_NOTE_HASH = "1" * 64
PRESERVED_NOTE_HASH = "2" * 64

_ROUTED_CONCEPTS = {
    "COND-1": (9_101, 201826, "Condition"),
    "DRUG-1": (9_102, 1112807, "Drug"),
    "MEAS-1": (9_103, 3004410, "Measurement"),
    "PROC-1": (9_104, 4017990, "Procedure"),
    "OBS-1": (9_105, 40766527, "Observation"),
}


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


def _vocabulary_router() -> VocabularyRouter:
    target_records = {}
    source_records = {}
    usagi = {}
    for source_code, (source_id, target_id, domain) in _ROUTED_CONCEPTS.items():
        target_code = f"TARGET-{source_code}"
        target_records[target_code] = {
            "concept_id": target_id,
            "concept_name": f"Standard {source_code}",
            "domain_id": domain,
            "vocabulary_id": "SYNTHETIC",
            "concept_class_id": "Standard",
            "standard_concept": "S",
            "concept_code": target_code,
        }
        source_records[source_code] = {
            "concept_id": source_id,
            "concept_name": f"Source {source_code}",
            "domain_id": domain,
            "vocabulary_id": "LOCAL",
            "concept_class_id": "Source",
            "standard_concept": None,
            "concept_code": source_code,
        }
        usagi[f"LOCAL:{source_code}"] = target_id
    return VocabularyRouter(
        {"SYNTHETIC": target_records, "LOCAL": source_records},
        usagi,
        vocabulary_version="SYNTHETIC 2026-01",
    )


def _replacement_fixture_notes(*, include_stale_span: bool) -> list[dict[str, Any]]:
    target_text = "Synthetic alpha and beta findings."
    preserved_text = "Synthetic gamma finding."

    def entity(
        note_text: str,
        surface: str,
        concept_id: int,
    ) -> dict[str, Any]:
        start = note_text.index(surface)
        return {
            "text": surface,
            "domain_id": "Condition",
            "start": start,
            "end": start + len(surface),
            "concept_id": concept_id,
            "code": f"SYN-{concept_id}",
            "vocabulary_id": "LOCAL",
            "concept_name": f"Synthetic {surface}",
        }

    target_entities = [entity(target_text, "alpha", 1001)]
    if include_stale_span:
        target_entities.append(entity(target_text, "beta", 1002))
    return [
        {
            "document_id": "synthetic-target-note",
            "person_id": "synthetic-target-person",
            "source_note_hash": TARGET_NOTE_HASH,
            "note_text": target_text,
            "entities": target_entities,
        },
        {
            "document_id": "synthetic-preserved-note",
            "person_id": "synthetic-preserved-person",
            "source_note_hash": PRESERVED_NOTE_HASH,
            "note_text": preserved_text,
            "entities": [entity(preserved_text, "gamma", 1003)],
        },
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
    assert validate_omop_database_report(con).to_dict() == {
        "count": 0,
        "by_table": {},
        "by_reason": {},
    }
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


def test_vocabulary_router_drives_end_to_end_cdm_load(tmp_path: Path) -> None:
    notes = _fixture_notes()
    invented_ids = set()
    for index, entity in enumerate(notes[0]["entities"][:5], start=1):
        invented_id = 8_000_000 + index
        entity["concept_id"] = invented_id
        invented_ids.add(invented_id)
    jsonl = tmp_path / "routed-grounded.jsonl"
    jsonl.write_text(json.dumps(notes[0]) + "\n", encoding="utf-8")

    tables = load_grounded_jsonl(
        jsonl,
        vocabulary_router=_vocabulary_router(),
    )

    expected_counts = dict(_expected_counts())
    expected_counts["concept"] = 11
    assert tables.row_counts == expected_counts
    assert validate_omop_tables(tables) == ()

    concepts = {row["concept_id"]: row for row in tables.table("concept")}
    assert invented_ids.isdisjoint(concepts)
    assert concepts[UNMAPPED_CONCEPT_ID]["concept_code"] == ""
    assert concepts[UNMAPPED_CONCEPT_ID]["domain_id"] == ""
    for source_code, (source_id, target_id, domain) in _ROUTED_CONCEPTS.items():
        assert concepts[source_id] == {
            "concept_id": source_id,
            "concept_name": f"Source {source_code}",
            "domain_id": domain,
            "vocabulary_id": "LOCAL",
            "concept_class_id": "Source",
            "standard_concept": None,
            "concept_code": source_code,
        }
        assert concepts[target_id]["concept_code"] == f"TARGET-{source_code}"
        assert concepts[target_id]["standard_concept"] == "S"

    provenance = {
        row["source_code"]: row for row in tables.table("source_to_concept_map")
    }
    for source_code, (source_id, target_id, _domain) in _ROUTED_CONCEPTS.items():
        row = provenance[source_code]
        assert row["source_concept_id"] == source_id
        assert row["source_vocabulary_id"] == "LOCAL"
        assert row["target_concept_id"] == target_id
        assert row["target_vocabulary_id"] == "SYNTHETIC"
        assert row["vocabulary_version"] == "SYNTHETIC 2026-01"
    assert provenance["SRC-UNMAPPED"]["target_concept_id"] == UNMAPPED_CONCEPT_ID
    assert provenance["SRC-UNMAPPED"]["target_vocabulary_id"] == "UNMAPPED"
    assert provenance["SRC-UNMAPPED"]["invalid_reason"] == "UNMAPPED"

    con = write_omop_duckdb(tables)
    first_counts = _table_counts_from_duckdb(con)
    write_omop_duckdb(tables, con)
    assert _table_counts_from_duckdb(con) == first_counts == expected_counts
    assert validate_omop_database(con) == ()


def test_vocabulary_router_does_not_mislabel_unknown_target_metadata() -> None:
    note_text = "Synthetic source finding."
    source_code = "SRC-ABSENT"
    router = VocabularyRouter(
        {
            "LOCAL": {
                source_code: {
                    "concept_id": 9_999,
                    "concept_name": "Synthetic source concept",
                    "domain_id": "Condition",
                    "vocabulary_id": "LOCAL",
                    "concept_class_id": "Source",
                    "standard_concept": None,
                    "concept_code": source_code,
                }
            }
        },
        {f"LOCAL:{source_code}": 8_888},
    )

    tables = load_grounded_notes(
        [
            {
                "document_id": "synthetic-note",
                "person_id": "synthetic-person",
                "note_text": note_text,
                "entities": [
                    {
                        "text": "source finding",
                        "start": note_text.index("source finding"),
                        "end": note_text.index("source finding")
                        + len("source finding"),
                        "code": source_code,
                        "vocabulary_id": "LOCAL",
                        "domain_id": "Condition",
                    }
                ],
            }
        ],
        vocabulary_router=router,
    )

    target = next(row for row in tables.table("concept") if row["concept_id"] == 8_888)
    provenance = tables.table("source_to_concept_map")[0]
    assert target["concept_name"] == "Concept 8888"
    assert target["concept_code"] == ""
    assert target["vocabulary_id"] == ""
    assert provenance["target_vocabulary_id"] == ""
    assert validate_omop_tables(tables) == ()


def test_postgres_ddl_is_stable_and_covers_loader_owned_tables() -> None:
    ddl = emit_postgres_ddl()
    con = sqlite3.connect(":memory:")
    con.executescript(ddl)

    assert ddl == emit_postgres_ddl()
    assert re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", ddl) == [
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
    ]
    assert "note_date DATE" in ddl
    assert "valid_end_date DATE" in ddl
    assert "REFERENCES concept(concept_id)" in ddl
    assert "REFERENCES note_nlp(note_nlp_id)" in ddl
    assert ddl.endswith(";\n")
    assert con.execute(
        "SELECT count(*) FROM sqlite_master WHERE type = 'table'"
    ).fetchone() == (11,)
    con.close()


def test_validation_reports_phi_free_concept_and_reachability_counts() -> None:
    valid_tables = load_grounded_notes(_fixture_notes())
    corrupted = {
        name: tuple(dict(row) for row in rows)
        for name, rows in valid_tables.tables.items()
    }
    condition = corrupted["condition_occurrence"][0]
    condition["condition_concept_id"] = 999_999_999
    note_nlp = next(
        row
        for row in corrupted["note_nlp"]
        if row["note_nlp_id"] == condition["note_nlp_id"]
    )
    note_nlp["note_nlp_event_id"] = 888_888_888
    tables = OmopCdmTables(
        tables=corrupted,
        summary=OmopLoadSummary(
            row_counts={name: len(rows) for name, rows in corrupted.items()},
            rejection_counts={},
        ),
    )

    report = validate_omop_tables_report(tables)

    assert report.is_valid is False
    assert report.to_dict() == {
        "count": 3,
        "by_table": {"condition_occurrence": 2, "note_nlp": 1},
        "by_reason": {
            "missing_concept": 1,
            "missing_domain_event": 1,
            "unreachable_from_note_nlp": 1,
        },
    }
    serialized = json.dumps(report.to_dict(), sort_keys=True)
    assert NOTE_TEXT not in serialized
    assert "secret-note-456" not in serialized
    assert "secret-patient-123" not in serialized


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


def test_replace_by_note_removes_stale_rows_and_notifies_consumers(
    tmp_path: Path,
) -> None:
    initial = load_grounded_notes(_replacement_fixture_notes(include_stale_span=True))
    replacement = load_grounded_notes(
        _replacement_fixture_notes(include_stale_span=False)[:1],
        mode="replace_by_note",
    )
    events = []

    duckdb_con = write_omop_duckdb(initial)
    write_omop_duckdb(
        replacement,
        duckdb_con,
        mode="replace_by_note",
        downstream_consumers=(events.append,),
    )
    sqlite_con = write_omop_sqlite(initial, tmp_path / "replace.sqlite")
    write_omop_sqlite(
        replacement,
        sqlite_con,
        downstream_consumers=(events.append,),
    )
    parquet_dir = write_omop_parquet(initial, tmp_path / "replace-parquet")
    write_omop_parquet(
        replacement,
        parquet_dir,
        mode="replace_by_note",
        downstream_consumers=(events.append,),
    )

    for connection in (duckdb_con, sqlite_con):
        assert connection.execute(
            """
            SELECT source_note_hash, count(*)
            FROM condition_occurrence
            GROUP BY source_note_hash
            ORDER BY source_note_hash
            """
        ).fetchall() == [(TARGET_NOTE_HASH, 1), (PRESERVED_NOTE_HASH, 1)]
        assert connection.execute(
            """
            SELECT source_note_hash, count(*)
            FROM source_to_concept_map
            GROUP BY source_note_hash
            ORDER BY source_note_hash
            """
        ).fetchall() == [(TARGET_NOTE_HASH, 1), (PRESERVED_NOTE_HASH, 1)]
        assert connection.execute("SELECT count(*) FROM note").fetchone()[0] == 2

    parquet_conditions = pq.read_table(
        parquet_dir / "condition_occurrence.parquet"
    ).to_pylist()
    assert sorted(row["source_note_hash"] for row in parquet_conditions) == [
        TARGET_NOTE_HASH,
        PRESERVED_NOTE_HASH,
    ]
    assert pq.read_table(parquet_dir / "note.parquet").num_rows == 2
    assert pq.read_table(parquet_dir / "note_nlp.parquet").num_rows == 2
    assert pq.read_table(parquet_dir / "source_to_concept_map.parquet").num_rows == 2

    assert len(events) == 3
    for event in events:
        assert event.mode == "replace_by_note"
        assert event.changed_note_hashes == (TARGET_NOTE_HASH,)
        serialized = json.dumps(event.to_dict(), sort_keys=True)
        assert "Synthetic alpha" not in serialized
        assert "synthetic-target-note" not in serialized
        assert "synthetic-target-person" not in serialized


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


def test_omop_loader_is_available_through_interop_registry() -> None:
    adapter = get_adapter("omop")

    assert "omop" in available_adapters()
    assert adapter_spec("omop").module == "openmed.interop.omop"
    assert hasattr(adapter, "load_grounded_notes")
