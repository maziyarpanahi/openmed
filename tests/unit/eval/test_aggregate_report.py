import json
from dataclasses import replace

import pytest

from openmed.eval.aggregate_report import project_aggregate_report
from openmed.eval.cache import hash_fixture_set
from openmed.eval.harness import BenchmarkFixture, run_benchmark


@pytest.fixture
def benchmark():
    fixtures = [
        BenchmarkFixture.from_mapping(
            {
                "id": f"private-fixture-{i}",
                "text": "secret-surface 99",
                "entities": [{"label": "PERSON", "start": 0, "end": 14}]
                + ([{"label": "AGE", "start": 15, "end": 17}] if i == 0 else []),
            }
        )
        for i in range(5)
    ]
    report = run_benchmark(
        fixtures,
        suite="private-suite",
        model_name="private-model",
        runner=lambda fixture, *_: [] if fixture is fixtures[0] else fixture.gold_spans,
        metadata={"private-note": "private-source-hash"},
    )
    return report, fixtures


def test_credentialed_projection_from_harness(benchmark):
    report, fixtures = benchmark

    def project():
        return project_aggregate_report(report, fixtures, corpus="i2b2", year=2014)

    result = project()
    assert result["dataset_digest"] == hash_fixture_set(fixtures)
    cell = result["by_label"]["PERSON"]
    assert (cell["tp"], cell["fn"], cell["fp"]) == (4, 1, 0)
    assert cell["recall"]["rate"] == 0.8
    assert cell["precision"]["rate"] == 1
    assert cell["leakage"]["rate"] == 0.2
    assert len(cell["recall"]["wilson_95"]) == 2
    assert result["by_label"]["AGE"] == {"status": "suppressed"}
    encoded = json.dumps(result)
    assert encoded == json.dumps(project())
    for private in (
        "secret-surface",
        "private-",
        "start",
        "end",
        "fixture_id",
        "findings",
    ):
        assert private not in encoded


def test_missing_counts_and_undeclared_labels_fail_closed(benchmark):
    report, fixtures = benchmark
    for metrics in (
        {},
        {**report.metrics, "exact_span_f1_by_label": {"private-label": {}}},
    ):
        with pytest.raises(ValueError) as error:
            project_aggregate_report(
                replace(report, metrics=metrics),
                fixtures,
                corpus="synthetic",
                year=2014,
            )
        assert "private-label" not in str(error.value)


def test_floor_is_configurable_and_fixtures_must_match(benchmark):
    report, fixtures = benchmark
    projected = project_aggregate_report(
        report, fixtures, corpus="synthetic", year=2014, minimum_gold_spans=1
    )
    assert projected["by_label"]["AGE"]["precision"]["rate"] is None
    with pytest.raises(ValueError, match="fixture set"):
        project_aggregate_report(report, fixtures[::-1], corpus="synthetic", year=2014)
