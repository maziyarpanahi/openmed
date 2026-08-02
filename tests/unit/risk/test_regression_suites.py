"""Regression harness for synthetic risk fixture suites."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from openmed.core.pii import deidentify
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.risk import risk_report

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "risk"
NEGATION_TRAPS = FIXTURE_DIR / "negation_traps.jsonl"
UNIQUENESS = FIXTURE_DIR / "quasi_identifier_uniqueness.jsonl"
FORBIDDEN_FIXTURE_MARKERS = (
    "cpt",
    "dua",
    "i2b2",
    "mimic",
    "n2c2",
    "snomed",
    "umls",
)


def test_fixture_files_exist_and_are_synthetic():
    assert NEGATION_TRAPS.exists()
    assert UNIQUENESS.exists()

    for path in (NEGATION_TRAPS, UNIQUENESS):
        meta, rows = _load_jsonl_suite(path)
        assert meta["version"] == 1
        assert rows
        assert all(row.get("synthetic") is True for row in rows)
        text = path.read_text(encoding="utf-8").casefold()
        for marker in FORBIDDEN_FIXTURE_MARKERS:
            assert (
                re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", text)
                is None
            )


def test_negation_trap_suite_has_zero_critical_leakage(monkeypatch):
    meta, rows = _load_jsonl_suite(NEGATION_TRAPS)
    assert meta["suite"] == "negation_traps"

    results = [_run_negation_case(row, monkeypatch) for row in rows]

    assert all(result["leakage_rate"] == 0.0 for result in results)


def test_quasi_identifier_uniqueness_suite_has_no_singletons():
    meta, rows = _load_jsonl_suite(UNIQUENESS)
    assert meta["suite"] == "quasi_identifier_uniqueness"

    report = _assert_uniqueness_expectations(rows, meta["expected"])

    assert report["k_min"] == 2
    assert report["singleton_records"] == []


def test_negation_trap_seeded_regression_fails(monkeypatch):
    _, rows = _load_jsonl_suite(NEGATION_TRAPS)
    bad_case = dict(rows[0])
    bad_case["spans"] = []

    with pytest.raises(AssertionError, match="critical leakage"):
        _run_negation_case(bad_case, monkeypatch)


def test_uniqueness_seeded_regression_fails():
    meta, rows = _load_jsonl_suite(UNIQUENESS)
    seeded_rows = [
        *rows,
        {
            "kind": "record",
            "record_id": "seeded-unique",
            "synthetic": True,
            "age": 94,
            "city": "Hillford",
            "visit_date": "2025-03-15",
            "condition": "longitudinal mobility review",
        },
    ]

    with pytest.raises(AssertionError, match="singleton"):
        _assert_uniqueness_expectations(seeded_rows, meta["expected"])


def _load_jsonl_suite(path: Path) -> tuple[dict, list[dict]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = rows[0]
    assert meta["kind"] == "meta"
    return meta, rows[1:]


def _run_negation_case(row: dict, monkeypatch) -> dict:
    text = row["text"]
    entities = [_entity_from_fixture(text, span) for span in row["spans"]]

    def fake_extract_pii(*args, **kwargs):
        return PredictionResult(
            text=text,
            entities=list(entities),
            model_name="risk-regression-fixture",
            timestamp="2026-06-22T00:00:00",
        )

    monkeypatch.setattr("openmed.core.pii.extract_pii", fake_extract_pii)

    deidentified = deidentify(text, method="mask")
    for phrase in row["expected"]["forbidden_residuals"]:
        assert phrase not in deidentified.deidentified_text, (
            f"critical leakage residual for {row['case_id']}: {phrase}"
        )

    original = {
        "record_id": row["case_id"],
        "text": text,
        "entities": [entity.to_dict() for entity in entities],
    }
    redacted = {
        "record_id": row["case_id"],
        "text": deidentified.deidentified_text,
    }
    report = risk_report(redacted, original=original)
    expected_leakage = row["expected"]["critical_leakage_rate"]
    assert report["leakage_rate"] == expected_leakage, (
        f"critical leakage regression for {row['case_id']}: {report}"
    )
    return report


def _entity_from_fixture(text: str, span: dict) -> EntityPrediction:
    value = span["text"]
    start = text.index(value)
    return EntityPrediction(
        text=value,
        label=span["label"],
        confidence=0.99,
        start=start,
        end=start + len(value),
        metadata={"source": "risk_regression_fixture"},
    )


def _assert_uniqueness_expectations(rows: list[dict], expected: dict) -> dict:
    records = [
        {key: value for key, value in row.items() if key not in {"kind", "synthetic"}}
        for row in rows
    ]
    report = risk_report(records)
    singleton_ids = [record["record_id"] for record in report["singleton_records"]]
    assert report["k_min"] >= expected["min_k"], (
        f"singleton quasi-identifier regression: {report}"
    )
    assert singleton_ids == expected["singleton_record_ids"], (
        f"singleton quasi-identifier regression: {singleton_ids}"
    )
    return report
