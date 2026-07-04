from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.integrations.lakehouse_redact import (
    LakehouseRedactionProgress,
    redact_lakehouse_table,
)
from openmed.processing.batch import BatchItemResult, BatchResult


def test_lakehouse_redact_writes_new_snapshot_and_keeps_source(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = _write_partitioned_table(tmp_path, pyarrow, pq)

    result = redact_lakehouse_table(
        table,
        text_columns=["note"],
        snapshot_id="run-001",
        batch_size=1,
        process_batch_fn=_fake_process_batch,
    )

    snapshot = table / "_openmed_lakehouse" / "snapshots" / "run-001" / "data"
    assert result.snapshot_path == snapshot
    assert (snapshot / "clinic=west" / "part-0.parquet").is_file()
    assert (snapshot / "clinic=east" / "part-0.parquet").is_file()
    assert pq.read_schema(
        snapshot / "clinic=west" / "part-0.parquet"
    ) == pq.read_schema(table / "clinic=west" / "part-0.parquet")
    assert _notes(pq, snapshot) == [
        "Patient [PERSON] called [PHONE]",
        "No direct identifiers",
        "Patient [PERSON] emailed [EMAIL]",
        "[PERSON] returned",
    ]
    assert _notes(pq, table) == [
        "Patient John Doe called 555-0101",
        "No direct identifiers",
        "Patient Jane Roe emailed jane@example.test",
        "John Doe returned",
    ]

    manifest = result.manifest
    assert manifest["dry_run"] is False
    assert manifest["partition_columns"] == ["clinic"]
    assert manifest["partition_count"] == 2
    assert manifest["file_count"] == 2
    assert manifest["total_rows"] == 4
    assert manifest["processed_cells"] == 4
    assert manifest["affected_cells"] == 3
    assert manifest["affected_rows"] == 3
    assert manifest["affected_columns"] == ["note"]
    assert manifest["span_count"] == 5
    assert manifest["per_label_counts"] == {
        "EMAIL": 1,
        "PERSON": 3,
        "PHONE": 1,
    }
    assert json.loads(result.manifest_path.read_text(encoding="utf-8")) == manifest
    assert json.loads(result.checkpoint_path.read_text(encoding="utf-8"))[
        "completed_partitions"
    ] == [partition["partition_id"] for partition in manifest["partitions"]]
    assert result.metadata_path.is_file()
    _assert_no_raw_values(manifest)
    _assert_no_raw_values(
        json.loads(result.checkpoint_path.read_text(encoding="utf-8"))
    )


def test_lakehouse_redact_dry_run_reports_affected_counts_without_writing(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = _write_partitioned_table(tmp_path, pyarrow, pq)

    result = redact_lakehouse_table(
        table,
        text_columns=["note"],
        snapshot_id="dry-run",
        dry_run=True,
        batch_size=2,
        process_batch_fn=_fake_process_batch,
    )

    assert result.dry_run is True
    assert result.snapshot_path is None
    assert result.manifest_path is None
    assert not (table / "_openmed_lakehouse").exists()
    assert result.manifest["dry_run"] is True
    assert result.manifest["total_rows"] == 4
    assert result.manifest["affected_rows"] == 3
    assert result.manifest["affected_cells"] == 3
    assert result.manifest["column_counts"]["note"]["affected_cells"] == 3
    _assert_no_raw_values(result.manifest)


def test_lakehouse_redact_resumes_completed_partitions(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = _write_partitioned_table(tmp_path, pyarrow, pq)
    calls = {"count": 0}

    def interrupting_process_batch(*args, **kwargs) -> BatchResult:
        calls["count"] += 1
        if calls["count"] > 1:
            raise RuntimeError("stop after first partition")
        return _fake_process_batch(*args, **kwargs)

    with pytest.raises(RuntimeError, match="stop after first partition"):
        redact_lakehouse_table(
            table,
            text_columns=["note"],
            snapshot_id="resume-run",
            batch_size=10,
            process_batch_fn=interrupting_process_batch,
        )

    progress: list[LakehouseRedactionProgress] = []
    resumed = redact_lakehouse_table(
        table,
        text_columns=["note"],
        snapshot_id="resume-run",
        resume=True,
        batch_size=10,
        process_batch_fn=_fake_process_batch,
        on_progress=progress.append,
    )

    assert [event.resumed for event in progress] == [True, False]
    assert resumed.manifest["total_rows"] == 4
    assert resumed.manifest["affected_rows"] == 3
    assert _notes(pq, resumed.snapshot_path) == [
        "Patient [PERSON] called [PHONE]",
        "No direct identifiers",
        "Patient [PERSON] emailed [EMAIL]",
        "[PERSON] returned",
    ]


def _write_partitioned_table(tmp_path: Path, pyarrow, pq) -> Path:
    table = tmp_path / "patients"
    west = table / "clinic=west"
    east = table / "clinic=east"
    west.mkdir(parents=True)
    east.mkdir(parents=True)
    schema = pyarrow.schema(
        [
            ("record_id", pyarrow.int64()),
            ("note", pyarrow.string()),
            ("status", pyarrow.string()),
        ]
    )
    pq.write_table(
        pyarrow.Table.from_pylist(
            [
                {
                    "record_id": 1,
                    "note": "Patient John Doe called 555-0101",
                    "status": "open",
                },
                {
                    "record_id": 2,
                    "note": "No direct identifiers",
                    "status": "closed",
                },
            ],
            schema=schema,
        ),
        west / "part-0.parquet",
    )
    pq.write_table(
        pyarrow.Table.from_pylist(
            [
                {
                    "record_id": 3,
                    "note": "Patient Jane Roe emailed jane@example.test",
                    "status": "open",
                },
                {
                    "record_id": 4,
                    "note": "John Doe returned",
                    "status": "closed",
                },
            ],
            schema=schema,
        ),
        east / "part-0.parquet",
    )
    return table


def _notes(pq, root: Path) -> list[str]:
    rows: list[dict] = []
    for path in sorted(root.rglob("*.parquet")):
        if "_openmed_lakehouse" in path.relative_to(root).parts:
            continue
        rows.extend(pq.read_table(path).to_pylist())
    return [row["note"] for row in sorted(rows, key=lambda row: row["record_id"])]


def _fake_process_batch(
    texts: list[str],
    *,
    ids: list[str] | None = None,
    **kwargs,
) -> BatchResult:
    items: list[BatchItemResult] = []
    for index, text in enumerate(texts):
        items.append(
            BatchItemResult(
                id=ids[index] if ids else f"item_{index}",
                result=_fake_deidentify(text, **kwargs),
            )
        )
    return BatchResult(items=items, model_name="test")


def _fake_deidentify(text: str, **kwargs) -> DeidentificationResult:
    replacements = {
        "John Doe": ("[PERSON]", "PERSON"),
        "Jane Roe": ("[PERSON]", "PERSON"),
        "555-0101": ("[PHONE]", "PHONE"),
        "jane@example.test": ("[EMAIL]", "EMAIL"),
    }
    redacted = text
    entities: list[PIIEntity] = []
    for surface, (replacement, label) in replacements.items():
        start = text.find(surface)
        if start == -1:
            continue
        entities.append(
            PIIEntity(
                text=surface,
                label=label,
                confidence=0.99,
                start=start,
                end=start + len(surface),
                entity_type=label,
                original_text=surface,
                redacted_text=replacement,
            )
        )
        redacted = redacted.replace(surface, replacement)
    return DeidentificationResult(
        original_text=text,
        deidentified_text=redacted,
        pii_entities=entities,
        method=kwargs.get("method", "mask"),
        timestamp=datetime.now(),
    )


def _assert_no_raw_values(payload: dict) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    for token in (
        "John Doe",
        "Jane Roe",
        "555-0101",
        "jane@example.test",
        "west",
        "east",
        "open",
        "closed",
        "patients",
        "part-0.parquet",
    ):
        assert token not in rendered
