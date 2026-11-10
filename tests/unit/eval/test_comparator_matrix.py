"""Tests for comparator benchmark matrix reporting."""

from __future__ import annotations

import json

from openmed.eval import (
    STATUS_NOT_AVAILABLE,
    STATUS_SCORED,
    ComparatorAdapter,
    ComparatorMatrixReport,
    run_comparator_matrix,
)
from openmed.eval.harness import BenchmarkFixture


def test_comparator_matrix_scores_available_systems_and_skips_missing() -> None:
    fixtures = [_fixture()]

    report = run_comparator_matrix(
        fixtures,
        adapters=[
            ComparatorAdapter(name="weak-comparator", runner=_weak_runner),
            ComparatorAdapter(name="missing-extra", runner=_missing_dependency_runner),
        ],
        suite_name="synthetic-comparator",
        openmed_runner=_exact_openmed_runner,
        generated_at="2026-07-01T00:00:00Z",
    )

    rows = {row.system: row for row in report.rows}

    assert isinstance(report, ComparatorMatrixReport)
    assert [row.system for row in report.rows] == [
        "OpenMed",
        "weak-comparator",
        "missing-extra",
    ]
    assert rows["OpenMed"].status == STATUS_SCORED
    assert rows["weak-comparator"].status == STATUS_SCORED
    assert rows["missing-extra"].status == STATUS_NOT_AVAILABLE
    assert rows["missing-extra"].reason == "weak comparator dependency missing"
    assert rows["OpenMed"].leakage_rate == 0.0
    assert rows["weak-comparator"].leakage_rate > rows["OpenMed"].leakage_rate
    assert rows["weak-comparator"].character_recall < rows["OpenMed"].character_recall
    assert len(report.scored_rows) == 2
    assert len(report.skipped_rows) == 1


def test_comparator_matrix_json_and_markdown_are_deterministic() -> None:
    report = run_comparator_matrix(
        [_fixture()],
        adapters=[
            ComparatorAdapter(name="weak-comparator", runner=_weak_runner),
            ComparatorAdapter(name="missing-extra", runner=_missing_dependency_runner),
        ],
        suite_name="synthetic-comparator",
        openmed_runner=_exact_openmed_runner,
        generated_at="2026-07-01T00:00:00Z",
        metadata={"z": 1, "a": {"b": True}},
    )

    first_json = report.to_json()
    second_json = report.to_json()
    markdown = report.to_markdown()

    assert first_json == second_json
    assert markdown == report.to_markdown()
    payload = json.loads(first_json)
    assert payload["rows"][0]["system"] == "OpenMed"
    assert payload["rows"][0]["leakage_rate"] == 0.0
    assert payload["rows"][2]["status"] == STATUS_NOT_AVAILABLE
    assert (
        "| `OpenMed` | scored | 0.00% | 100.00% | 100.00% | 100.00% | 1 | n/a |"
        in markdown
    )
    assert (
        "| `weak-comparator` | scored | 73.33% | 26.67% | 66.67% | 66.67% | 1 | n/a |"
        in markdown
    )
    assert "| `a.b` | true |" in markdown
    assert "| `z` | 1 |" in markdown


def test_comparator_matrix_accepts_fixture_mappings() -> None:
    report = run_comparator_matrix(
        [_fixture_mapping()],
        adapters=[],
        suite_name="mapping-suite",
        openmed_runner=_exact_openmed_runner,
    )

    assert report.fixture_count == 1
    assert report.rows[0].system == "OpenMed"
    assert report.rows[0].leakage_rate == 0.0


def _fixture() -> BenchmarkFixture:
    return BenchmarkFixture.from_mapping(_fixture_mapping())


def _fixture_mapping() -> dict[str, object]:
    return {
        "id": "synthetic-note-1",
        "text": "Patient John has SSN 123-45-6789.",
        "language": "en",
        "gold_spans": [
            {"start": 8, "end": 12, "label": "PERSON"},
            {"start": 21, "end": 32, "label": "SSN"},
        ],
    }


def _exact_openmed_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
) -> tuple[object, ...]:
    assert model_name == "OpenMed"
    assert device == "cpu"
    return fixture.gold_spans


def _weak_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
) -> tuple[dict[str, object], ...]:
    assert model_name == "weak-comparator"
    assert device == "cpu"
    return ({"start": 8, "end": 12, "label": "PERSON"},)


def _missing_dependency_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
) -> tuple[object, ...]:
    del fixture, model_name, device
    raise ImportError("weak comparator dependency missing")
