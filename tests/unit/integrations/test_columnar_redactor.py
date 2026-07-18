from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.integrations.columnar_redactor import (
    ColumnarProgress,
    redact_columnar_dataset,
)
from openmed.processing.batch import BatchItemResult, BatchResult, process_batch


def test_columnar_redactor_streams_parquet_and_profiles_qi(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    input_path = tmp_path / "patients.parquet"
    output_path = tmp_path / "patients.redacted.parquet"
    manifest_path = tmp_path / "manifest.json"
    qi_report_path = tmp_path / "qi-report.json"
    progress_path = tmp_path / "progress.json"
    table = pyarrow.table(
        {
            "record_id": [1, 2, 3, 4],
            "note": [
                "Patient John Doe called 555-0101",
                "Patient Jane Roe emailed jane@example.test",
                "No direct identifiers",
                "Patient John Doe returned",
            ],
            "age": [42, 42, 79, 79],
            "zip": ["02139", "02139", "99501", "99501"],
            "diagnosis": ["flu", "flu", "rare_condition_x", "rare_condition_y"],
        }
    )
    pq.write_table(table, input_path, row_group_size=2)
    progress_events: list[ColumnarProgress] = []

    result = redact_columnar_dataset(
        input_path,
        text_columns=["note"],
        output_path=output_path,
        quasi_identifier_columns=["age", "zip", "diagnosis"],
        manifest_path=manifest_path,
        qi_report_path=qi_report_path,
        progress_path=progress_path,
        batch_size=1,
        low_k_threshold=2,
        process_batch_fn=_fake_process_batch,
        on_progress=progress_events.append,
    )

    redacted_rows = pq.read_table(output_path).to_pylist()
    assert redacted_rows == [
        {
            "record_id": 1,
            "note": "Patient [PERSON] called [PHONE]",
            "age": 42,
            "zip": "02139",
            "diagnosis": "flu",
        },
        {
            "record_id": 2,
            "note": "Patient [PERSON] emailed [EMAIL]",
            "age": 42,
            "zip": "02139",
            "diagnosis": "flu",
        },
        {
            "record_id": 3,
            "note": "No direct identifiers",
            "age": 79,
            "zip": "99501",
            "diagnosis": "rare_condition_x",
        },
        {
            "record_id": 4,
            "note": "Patient [PERSON] returned",
            "age": 79,
            "zip": "99501",
            "diagnosis": "rare_condition_y",
        },
    ]
    assert result.manifest["input_format"] == "parquet"
    assert result.manifest["row_group_count"] == 2
    assert result.manifest["total_rows"] == 4
    assert result.manifest["processed_cells"] == 4
    assert result.manifest["redacted_cells"] == 3
    assert result.manifest["total_spans"] == 5
    assert result.manifest["per_label_counts"] == {
        "EMAIL": 1,
        "PERSON": 3,
        "PHONE": 1,
    }
    assert result.manifest["peak_record_batch_rows"] <= 1
    assert result.qi_report["k_min"] == 1
    assert result.qi_report["low_k_threshold"] == 2
    assert result.qi_report["low_k_class_count"] == 2
    assert result.qi_report["singleton_record_count"] == 2
    assert result.qi_report["risk_flags"] == [
        {
            "type": "low_equivalence_class_size",
            "threshold": 2,
            "class_count": 2,
            "record_count": 2,
            "min_class_size": 1,
        }
    ]
    assert [event.group_index for event in progress_events] == [0, 1]
    assert [event.rows_completed for event in progress_events] == [2, 4]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == result.manifest
    assert json.loads(qi_report_path.read_text(encoding="utf-8")) == result.qi_report
    assert json.loads(progress_path.read_text(encoding="utf-8"))[
        "completed_groups"
    ] == [0, 1]
    _assert_no_raw_values(result.manifest)
    _assert_no_raw_values(result.qi_report)
    _assert_no_raw_values(json.loads(progress_path.read_text(encoding="utf-8")))


def test_columnar_redactor_supports_orc_when_pyarrow_orc_is_available(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    orc = pytest.importorskip("pyarrow.orc")

    input_path = tmp_path / "patients.orc"
    output_path = tmp_path / "patients.redacted.orc"
    table = pyarrow.table(
        {
            "note": [
                "Patient John Doe called 555-0101",
                "Patient Jane Roe emailed jane@example.test",
            ],
            "age": [42, 79],
            "zip": ["02139", "99501"],
        }
    )
    orc.write_table(table, input_path)

    result = redact_columnar_dataset(
        input_path,
        text_columns=["note"],
        output_path=output_path,
        batch_size=1,
        low_k_threshold=3,
        process_batch_fn=_fake_process_batch,
    )

    redacted_rows = orc.ORCFile(output_path).read().to_pylist()
    assert redacted_rows == [
        {"note": "Patient [PERSON] called [PHONE]", "age": 42, "zip": "02139"},
        {
            "note": "Patient [PERSON] emailed [EMAIL]",
            "age": 79,
            "zip": "99501",
        },
    ]
    assert result.manifest["input_format"] == "orc"
    assert result.manifest["group_kind"] == "stripe"
    assert result.qi_report["k_min"] == 1
    assert result.qi_report["low_k_record_count"] == 2
    _assert_no_raw_values(result.manifest)
    _assert_no_raw_values(result.qi_report)


def test_columnar_redactor_resumes_completed_parquet_row_groups(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    input_path = tmp_path / "patients.parquet"
    output_path = tmp_path / "patients.redacted.parquet"
    progress_path = tmp_path / "progress.json"
    table = pyarrow.table(
        {
            "note": [
                "Patient John Doe called 555-0101",
                "Patient Jane Roe emailed jane@example.test",
            ],
            "age": [42, 79],
        }
    )
    pq.write_table(table, input_path, row_group_size=1)

    calls = {"count": 0}

    def interrupting_process_batch(*args, **kwargs) -> BatchResult:
        calls["count"] += 1
        if calls["count"] > 1:
            raise RuntimeError("stop after first row group")
        return _fake_process_batch(*args, **kwargs)

    with pytest.raises(RuntimeError, match="stop after first row group"):
        redact_columnar_dataset(
            input_path,
            text_columns=["note"],
            output_path=output_path,
            progress_path=progress_path,
            batch_size=1,
            process_batch_fn=interrupting_process_batch,
        )

    resumed = redact_columnar_dataset(
        input_path,
        text_columns=["note"],
        output_path=output_path,
        progress_path=progress_path,
        resume=True,
        batch_size=1,
        process_batch_fn=_fake_process_batch,
    )

    assert [row["note"] for row in pq.read_table(output_path).to_pylist()] == [
        "Patient [PERSON] called [PHONE]",
        "Patient [PERSON] emailed [EMAIL]",
    ]
    assert resumed.manifest["total_rows"] == 2
    assert resumed.manifest["processed_cells"] == 2
    assert resumed.qi_report["record_count"] == 2
    _assert_no_raw_values(resumed.manifest)
    _assert_no_raw_values(resumed.qi_report)


def test_process_batch_deidentify_forwards_policy(monkeypatch) -> None:
    from openmed.core import pii

    policies: list[str] = []

    def fake_extract_pii_batch(texts: list[str], **kwargs):
        return [SimpleNamespace(entities=(), text=text) for text in texts]

    def fake_build_deidentification_result(text: str, pii_result, **kwargs):
        policies.append(kwargs["policy"])
        return SimpleNamespace(deidentified_text=text, pii_entities=())

    monkeypatch.setattr(pii, "_extract_pii_batch", fake_extract_pii_batch)
    monkeypatch.setattr(
        pii,
        "_build_deidentification_result",
        fake_build_deidentification_result,
    )

    result = process_batch(
        ["No identifiers here"],
        operation="deidentify",
        policy="strict_no_leak",
        use_safety_sweep=False,
    )

    assert result.successful_items == 1
    assert policies == ["strict_no_leak"]


def _fake_process_batch(
    texts: list[str],
    *,
    ids: list[str] | None = None,
    **kwargs,
) -> BatchResult:
    items: list[BatchItemResult] = []
    for index, text in enumerate(texts):
        result = _fake_deidentify(text, **kwargs)
        items.append(
            BatchItemResult(
                id=ids[index] if ids else f"item_{index}",
                result=result,
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
        "02139",
        "99501",
        "flu",
        "rare_condition_x",
        "rare_condition_y",
    ):
        assert token not in rendered
