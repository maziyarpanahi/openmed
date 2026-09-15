"""Tests for the synthetic ConText temporal-consistency suite."""

from __future__ import annotations

import json

import pytest

from openmed.eval.golden import (
    GoldenFixture,
    list_fixture_paths,
    load_golden_fixtures,
)
from openmed.eval.report import BenchmarkReport
from openmed.eval.suites import (
    DEFAULT_SUITES,
    TEMPORAL_CONSISTENCY,
    load_suite_fixtures,
    suite_metadata,
)
from openmed.eval.suites.temporal_consistency import (
    TEMPORAL_CONSISTENCY_AXES,
    TEMPORAL_CONSISTENCY_FIXTURE_PATH,
    TEMPORAL_CONSISTENCY_SCHEMA_VERSION,
    load_temporal_consistency_fixtures,
    run_temporal_consistency_suite,
    temporal_consistency_metadata,
)


def test_committed_fixtures_use_the_shared_golden_schema_and_are_synthetic() -> None:
    fixtures = load_temporal_consistency_fixtures()
    golden = load_golden_fixtures(TEMPORAL_CONSISTENCY_FIXTURE_PATH)

    assert len(fixtures) == len(golden) == 9
    assert all(isinstance(row, GoldenFixture) for row in golden)
    assert all(row.metadata["synthetic"] is True for row in golden)
    assert all(row.metadata["contains_real_phi"] is False for row in golden)
    assert TEMPORAL_CONSISTENCY_FIXTURE_PATH.name not in {
        path.name for path in list_fixture_paths()
    }


def test_hypothetical_fixtures_explicitly_forbid_recent_temporality() -> None:
    fixtures = load_temporal_consistency_fixtures()
    hypothetical = [
        fixture
        for fixture in fixtures
        if fixture.metadata.get("trap") == "hypothetical_not_recent"
    ]

    assert hypothetical
    assert all(
        fixture.expected_axes["temporality"] == "hypothetical"
        for fixture in hypothetical
    )
    assert all(
        fixture.expected_axes["temporality"] != "recent" for fixture in hypothetical
    )


def test_suite_returns_per_axis_benchmark_metrics() -> None:
    report = run_temporal_consistency_suite()

    assert isinstance(report, BenchmarkReport)
    assert report.suite == TEMPORAL_CONSISTENCY
    assert report.fixture_count == 9
    assert report.metrics["per_axis_accuracy"] == {
        "temporality": 1.0,
        "uncertainty": 1.0,
    }
    assert report.metrics["axis_metrics"]["temporality"]["total"] == 9
    assert report.metrics["axis_metrics"]["uncertainty"]["total"] == 9
    assert report.metrics["consistency_score"] == 1.0
    assert report.metrics["consistency"]["consistent_group_count"] == 4


def test_mislabeled_hypothetical_prediction_lowers_consistency() -> None:
    report = run_temporal_consistency_suite(
        predictions_by_fixture={
            "temporal-consistency-hypothetical-if": {
                "temporality": "recent",
                "uncertainty": "uncertain",
            }
        }
    )

    assert report.metrics["temporality_accuracy"] < 1.0
    assert report.metrics["consistency_score"] < 1.0
    assert report.metrics["consistency"]["consistent_group_count"] == 3
    assert report.metrics["consistency"]["mismatches"] == [
        {
            "axis": "temporality",
            "expected": "hypothetical",
            "fixture_id": "temporal-consistency-hypothetical-if",
            "group_id": "pneumonia-hypothetical",
            "predicted": "recent",
            "variant": "conditional_if",
        }
    ]


def test_report_artifact_does_not_include_fixture_text() -> None:
    fixtures = load_temporal_consistency_fixtures()
    serialized = json.dumps(run_temporal_consistency_suite().to_dict())

    for fixture in fixtures:
        assert fixture.text not in serialized


def test_registry_selects_temporal_consistency_suite() -> None:
    assert TEMPORAL_CONSISTENCY in DEFAULT_SUITES
    assert TEMPORAL_CONSISTENCY_AXES == ("temporality", "uncertainty")
    assert load_suite_fixtures(TEMPORAL_CONSISTENCY) == list(
        load_temporal_consistency_fixtures()
    )
    assert suite_metadata(TEMPORAL_CONSISTENCY) == temporal_consistency_metadata()


def test_metadata_declares_schema_and_synthetic_provenance() -> None:
    metadata = temporal_consistency_metadata()

    assert metadata == {
        "contains_real_phi": False,
        "fixture_format": "GoldenFixture",
        "schema_version": TEMPORAL_CONSISTENCY_SCHEMA_VERSION,
        "scored_axes": ["temporality", "uncertainty"],
        "suite": TEMPORAL_CONSISTENCY,
        "synthetic": True,
    }


def test_loader_rejects_a_real_phi_declaration(tmp_path) -> None:
    row = json.loads(
        TEMPORAL_CONSISTENCY_FIXTURE_PATH.read_text(encoding="utf-8").splitlines()[0]
    )
    row["metadata"]["contains_real_phi"] = True
    path = tmp_path / "temporal_consistency.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains_real_phi=false"):
        load_temporal_consistency_fixtures(path)
