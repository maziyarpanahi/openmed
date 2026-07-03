from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.processing.batch import redact_dataset


def test_redact_dataset_csv_preserves_non_text_columns_and_summarizes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "patients.csv"
    output_path = tmp_path / "patients.redacted.csv"
    input_path.write_text(
        "id,note,age\n"
        "1,Patient John Doe called 555-0101,42\n"
        "2,Patient Jane Roe emailed jane@example.test,37\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("openmed.core.pii.deidentify", _fake_deidentify)

    result = redact_dataset(
        input_path,
        text_columns=["note"],
        output_path=output_path,
        policy="strict_no_leak",
    )

    rows = list(csv.DictReader(output_path.open(encoding="utf-8", newline="")))
    assert rows == [
        {"id": "1", "note": "Patient [PERSON] called [PHONE]", "age": "42"},
        {"id": "2", "note": "Patient [PERSON] emailed [EMAIL]", "age": "37"},
    ]
    assert result.summary.to_dict() == {
        "input_format": "csv",
        "text_columns": ["note"],
        "total_rows": 2,
        "processed_cells": 2,
        "redacted_cells": 2,
        "total_spans": 4,
        "per_label_counts": {"EMAIL": 1, "PERSON": 2, "PHONE": 1},
        "residual_span_count": 0,
        "residual_leakage_estimate": 0.0,
    }
    _assert_summary_has_no_fixture_phi(result.summary.to_dict())


def test_redact_dataset_jsonl_preserves_non_text_columns_and_summarizes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "notes.jsonl"
    output_path = tmp_path / "notes.redacted.jsonl"
    rows = [
        {
            "id": 1,
            "note": "Patient John Doe called 555-0101",
            "status": "complete",
        },
        {
            "id": 2,
            "note": "Patient Jane Roe emailed jane@example.test",
            "status": "queued",
        },
    ]
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr("openmed.core.pii.deidentify", _fake_deidentify)

    result = redact_dataset(
        input_path,
        text_columns=["note"],
        output_path=output_path,
    )

    redacted_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert redacted_rows == [
        {"id": 1, "note": "Patient [PERSON] called [PHONE]", "status": "complete"},
        {"id": 2, "note": "Patient [PERSON] emailed [EMAIL]", "status": "queued"},
    ]
    assert result.summary.total_rows == 2
    assert result.summary.total_spans == 4
    assert result.summary.per_label_counts == {"PERSON": 2, "PHONE": 1, "EMAIL": 1}
    _assert_summary_has_no_fixture_phi(result.summary.to_dict())


def test_redact_dataset_parquet_preserves_schema_when_pyarrow_is_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    input_path = tmp_path / "patients.parquet"
    output_path = tmp_path / "patients.redacted.parquet"
    table = pyarrow.table(
        {
            "id": [1, 2],
            "note": [
                "Patient John Doe called 555-0101",
                "Patient Jane Roe emailed jane@example.test",
            ],
            "age": [42, 37],
        }
    )
    pq.write_table(table, input_path)
    monkeypatch.setattr("openmed.core.pii.deidentify", _fake_deidentify)

    result = redact_dataset(
        input_path,
        text_columns=["note"],
        output_path=output_path,
        batch_size=1,
    )

    redacted_rows = pq.read_table(output_path).to_pylist()
    assert redacted_rows == [
        {"id": 1, "note": "Patient [PERSON] called [PHONE]", "age": 42},
        {"id": 2, "note": "Patient [PERSON] emailed [EMAIL]", "age": 37},
    ]
    assert result.summary.total_rows == 2
    assert result.summary.total_spans == 4
    assert result.summary.per_label_counts == {"PERSON": 2, "PHONE": 1, "EMAIL": 1}
    _assert_summary_has_no_fixture_phi(result.summary.to_dict())


def test_redact_dataset_cli_emits_phi_free_audit_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "patients.csv"
    output_path = tmp_path / "patients.redacted.csv"
    input_path.write_text(
        "id,note\n1,Patient John Doe called 555-0101\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("openmed.core.pii.deidentify", _fake_deidentify)

    exit_code = main_module.main(
        [
            "redact-dataset",
            str(input_path),
            "--text-columns",
            "note",
            "--output",
            str(output_path),
            "--policy",
            "strict_no_leak",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    summary = json.loads(captured.out)
    assert summary["total_spans"] == 2
    assert summary["per_label_counts"] == {"PERSON": 1, "PHONE": 1}
    assert "John Doe" not in output_path.read_text(encoding="utf-8")
    assert "555-0101" not in output_path.read_text(encoding="utf-8")
    _assert_summary_has_no_fixture_phi(summary)


def test_redact_dataset_cli_can_disable_keep_year(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "patients.csv"
    output_path = tmp_path / "patients.redacted.csv"
    input_path.write_text(
        "id,note\n1,Patient John Doe called 555-0101\n",
        encoding="utf-8",
    )
    calls: list[dict] = []

    def fake_deidentify(text: str, **kwargs) -> DeidentificationResult:
        calls.append(kwargs)
        return _fake_deidentify(text, **kwargs)

    monkeypatch.setattr("openmed.core.pii.deidentify", fake_deidentify)

    exit_code = main_module.main(
        [
            "redact-dataset",
            str(input_path),
            "--text-column",
            "note",
            "--output",
            str(output_path),
            "--no-keep-year",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert calls
    assert calls[0]["keep_year"] is False


def test_redact_dataset_requires_text_columns(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "patients.csv"
    input_path.write_text("id,note\n1,No identifiers\n", encoding="utf-8")

    exit_code = main_module.main(["redact-dataset", str(input_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "At least one --text-column" in captured.err


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


def _assert_summary_has_no_fixture_phi(summary: dict) -> None:
    payload = json.dumps(summary, sort_keys=True)
    for token in ("John Doe", "Jane Roe", "555-0101", "jane@example.test"):
        assert token not in payload
