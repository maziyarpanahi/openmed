"""Tests for automatic tabular quasi-identifier detection."""

from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path

from openmed.risk import quasi_identifier_key, risk_report
from openmed.structured import ROLE_DIRECT_ID, ROLE_QUASI_ID, scan_table

QI_COLUMNS = {"age", "zip_code", "admission_date", "diagnosis"}

ROWS = [
    {
        "record_id": "peer-1",
        "age": "94",
        "zip_code": "02139",
        "admission_date": "2025-04-01",
        "diagnosis": "routine-follow-up",
        "department": "cardiology",
    },
    {
        "record_id": "target",
        "age": "94",
        "zip_code": "02139",
        "admission_date": "2025-04-09",
        "diagnosis": "rare-alpha-syndrome",
        "department": "cardiology",
    },
    {
        "record_id": "peer-2",
        "age": "94",
        "zip_code": "30301",
        "admission_date": "2025-04-09",
        "diagnosis": "routine-follow-up",
        "department": "cardiology",
    },
    {
        "record_id": "peer-3",
        "age": "72",
        "zip_code": "02139",
        "admission_date": "2025-04-09",
        "diagnosis": "rare-alpha-syndrome",
        "department": "cardiology",
    },
    {
        "record_id": "peer-4",
        "age": "72",
        "zip_code": "30301",
        "admission_date": "2025-04-01",
        "diagnosis": "routine-follow-up",
        "department": "cardiology",
    },
    {
        "record_id": "peer-5",
        "age": "72",
        "zip_code": "30301",
        "admission_date": "2025-04-01",
        "diagnosis": "rare-alpha-syndrome",
        "department": "cardiology",
    },
    {
        "record_id": "peer-6",
        "age": "65",
        "zip_code": "02139",
        "admission_date": "2025-04-01",
        "diagnosis": "routine-follow-up",
        "department": "cardiology",
    },
    {
        "record_id": "peer-7",
        "age": "65",
        "zip_code": "30301",
        "admission_date": "2025-04-09",
        "diagnosis": "routine-follow-up",
        "department": "cardiology",
    },
]


def test_scan_table_surfaces_planted_singleton_qi_set(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "released_style.csv", ROWS)

    manifest = scan_table(csv_path)

    assert manifest["columns"]["record_id"]["role"] == ROLE_DIRECT_ID
    assert manifest["columns"]["age"]["role"] == ROLE_QUASI_ID
    assert manifest["columns"]["zip_code"]["role"] == ROLE_QUASI_ID
    assert manifest["columns"]["admission_date"]["role"] == ROLE_QUASI_ID
    assert manifest["quasi_identifier_sets"]
    assert any(
        QI_COLUMNS <= set(qi_set["columns"])
        and qi_set["singleton_count"] >= 1
        and qi_set["min_equivalence_class_size"] == 1
        for qi_set in manifest["quasi_identifier_sets"]
    )


def test_detected_qi_keys_match_risk_report_singleton_keys(tmp_path: Path) -> None:
    csv_path = _write_csv(tmp_path / "keys.csv", ROWS)
    manifest = scan_table(csv_path)
    qi_set = next(
        item
        for item in manifest["quasi_identifier_sets"]
        if QI_COLUMNS <= set(item["columns"])
    )
    target = next(row for row in ROWS if row["record_id"] == "target")

    detector_key = _canonical_json_bytes(
        quasi_identifier_key(target, fields=qi_set["columns"])
    )
    singleton = next(
        item
        for item in risk_report(ROWS)["singleton_records"]
        if item["record_id"] == "target"
    )
    risk_key = _canonical_json_bytes(singleton["quasi_identifier_key"])

    assert detector_key == risk_key


def test_released_style_fixture_recall_and_phi_safe_manifest(
    tmp_path: Path,
) -> None:
    csv_path = _write_csv(tmp_path / "manifest.csv", ROWS)
    manifest = scan_table(csv_path)
    best_recall = max(
        len(QI_COLUMNS & set(item["columns"])) / len(QI_COLUMNS)
        for item in manifest["quasi_identifier_sets"]
    )

    assert best_recall >= 0.9

    serialized = json.dumps(manifest, sort_keys=True)
    for raw_value in (
        "target",
        "02139",
        "30301",
        "2025-04-09",
        "rare-alpha-syndrome",
        "routine-follow-up",
    ):
        assert raw_value not in serialized


def test_jsonl_scan_uses_same_manifest_contract(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "records.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(row) for row in ROWS) + "\n",
        encoding="utf-8",
    )

    manifest = scan_table(jsonl_path)

    assert manifest["format"] == "jsonl"
    assert manifest["sample"]["sampled_rows"] == len(ROWS)
    assert any(
        QI_COLUMNS <= set(item["columns"]) for item in manifest["quasi_identifier_sets"]
    )


def test_parquet_sampling_stops_at_fixed_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[int] = []

    class FakeSchema:
        names = ["age", "zip_code", "admission_date", "diagnosis"]

    class FakeMetadata:
        num_rows = 5_000_000

    class FakeBatch:
        def __init__(self, rows):
            self._rows = rows

        def to_pylist(self):
            return list(self._rows)

    class FakeParquetFile:
        schema_arrow = FakeSchema()
        metadata = FakeMetadata()

        def __init__(self, path):
            self.path = path

        def iter_batches(self, *, batch_size):
            calls.append(batch_size)
            yield FakeBatch(ROWS[:2])
            calls.append(batch_size)
            yield FakeBatch(ROWS[2:4])
            raise AssertionError("scan_table read beyond the sampling budget")

    fake_pyarrow = types.ModuleType("pyarrow")
    fake_parquet = types.ModuleType("pyarrow.parquet")
    fake_parquet.ParquetFile = FakeParquetFile
    fake_pyarrow.parquet = fake_parquet
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)

    manifest = scan_table(tmp_path / "huge.parquet", max_rows=3, batch_size=2)

    assert calls == [2, 2]
    assert manifest["format"] == "parquet"
    assert manifest["sample"] == {
        "sampled_rows": 3,
        "max_rows": 3,
        "source_rows": 5_000_000,
        "bounded": True,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
