"""Unit tests for fairness reporting over demographic surrogate groups."""

from __future__ import annotations

import pytest

from openmed.eval import UNSPECIFIED_GROUP, fairness_report
from openmed.eval.golden import GoldenFixture
from openmed.eval.harness import BenchmarkFixture


def test_fairness_report_detects_group_leakage_disparity() -> None:
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "fairness-leakage",
            "text": "Patient Asha Patel met John Smith.",
            "language": "en",
            "gold_spans": [
                {
                    "start": 8,
                    "end": 18,
                    "label": "PERSON",
                    "group": "south_asian_names",
                },
                {
                    "start": 23,
                    "end": 33,
                    "label": "PERSON",
                    "group": "anglophone_names",
                },
            ],
        }
    )

    def runner(fixture, model_name, device):
        assert model_name == "group-test-model"
        assert device == "cpu"
        return [{"start": 23, "end": 33, "label": "PERSON"}]

    report = fairness_report(
        "group-test-model",
        [fixture],
        runner=runner,
    )

    assert report.per_group["south_asian_names"].leakage_rate == 1.0
    assert report.per_group["south_asian_names"].recall == 0.0
    assert report.per_group["anglophone_names"].leakage_rate == 0.0
    assert report.per_group["anglophone_names"].recall == 1.0
    assert report.leakage_disparity == pytest.approx(1.0)
    assert report.worst_group_leakage == pytest.approx(1.0)
    assert report.worst_group == "south_asian_names"


def test_fairness_report_balanced_groups_have_zero_disparity() -> None:
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "fairness-balanced",
            "text": "Patient Asha Patel met John Smith.",
            "language": "en",
            "gold_spans": [
                {
                    "start": 8,
                    "end": 18,
                    "label": "PERSON",
                    "group": "south_asian_names",
                },
                {
                    "start": 23,
                    "end": 33,
                    "label": "PERSON",
                    "group": "anglophone_names",
                },
            ],
        }
    )

    def runner(fixture, model_name, device):
        return [
            {"start": 8, "end": 18, "label": "PERSON"},
            {"start": 23, "end": 33, "label": "PERSON"},
        ]

    report = fairness_report(runner, [fixture])

    assert report.per_group["south_asian_names"].leakage_rate == 0.0
    assert report.per_group["anglophone_names"].leakage_rate == 0.0
    assert report.leakage_disparity == pytest.approx(0.0)
    assert report.worst_group_leakage == pytest.approx(0.0)


def test_fairness_report_buckets_ungrouped_gold_spans_as_unspecified() -> None:
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "fairness-unspecified",
            "text": "Patient Maria Gomez called.",
            "language": "en",
            "gold_spans": [
                {"start": 8, "end": 19, "label": "PERSON"},
            ],
        }
    )

    def runner(fixture, model_name, device):
        return []

    report = fairness_report("group-test-model", [fixture], runner=runner)

    assert set(report.per_group) == {UNSPECIFIED_GROUP}
    assert report.per_group[UNSPECIFIED_GROUP].leakage_rate == 1.0
    assert report.per_group[UNSPECIFIED_GROUP].recall == 0.0


def test_golden_fixture_schema_preserves_optional_span_group() -> None:
    fixture = GoldenFixture.from_mapping(
        {
            "id": "golden-fairness-group",
            "language": "en",
            "text": "Patient Asha Patel.",
            "gold_spans": [
                {
                    "start": 8,
                    "end": 18,
                    "label": "PERSON",
                    "text": "Asha Patel",
                    "group": "south_asian_names",
                },
            ],
            "metadata": {
                "category": "multilingual",
                "expected_output": {
                    "method": "mask",
                    "text": "Patient [PERSON].",
                },
                "synthetic": True,
            },
        }
    )

    span = fixture.gold_spans[0]
    mapping = fixture.to_mapping()

    assert span.metadata["group"] == "south_asian_names"
    assert mapping["gold_spans"][0]["group"] == "south_asian_names"
    assert GoldenFixture.from_mapping(mapping).to_mapping() == mapping
