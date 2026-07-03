"""Unit tests for fairness reporting over demographic surrogate groups."""

from __future__ import annotations

import json

import pytest

from openmed.core.pii_i18n import SUPPORTED_LANGUAGES
from openmed.eval import (
    UNSPECIFIED_GROUP,
    cross_lingual_transfer_report,
    fairness_report,
    run_cross_lingual_transfer,
)
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


def test_transfer_matrix_covers_supported_languages_and_ranks_weak_target() -> None:
    weak_language = _weak_language()
    languages = tuple(sorted(SUPPORTED_LANGUAGES))

    report = cross_lingual_transfer_report(
        "unit-transfer-model",
        _transfer_fixtures(),
        runner=_transfer_runner(weak_language),
        leakage_floors={language: 0.25 for language in languages},
        ci_resamples=25,
        ci_seed=17,
    )

    assert report.languages == languages
    assert tuple(report.matrix) == languages
    assert all(tuple(report.matrix[source]) == languages for source in languages)
    assert len(report.matrix) == len(SUPPORTED_LANGUAGES)
    assert all(len(row) == len(SUPPORTED_LANGUAGES) for row in report.matrix.values())

    source_language = next(
        language for language in languages if language != weak_language
    )
    assert report.matrix[weak_language][weak_language].zero_shot is False
    assert report.matrix[weak_language][weak_language].leakage_rate == pytest.approx(
        0.0
    )
    assert report.matrix[source_language][weak_language].zero_shot is True
    assert report.matrix[source_language][weak_language].leakage_rate == pytest.approx(
        1.0
    )

    assert len(report.deficiencies) == 1
    assert report.deficiencies[0].rank == 1
    assert report.deficiencies[0].target_language == weak_language
    assert report.deficiencies[0].leakage_rate == pytest.approx(1.0)
    assert report.deficiencies[0].excess == pytest.approx(0.75)


def test_transfer_gap_ci_and_report_rendering_are_byte_stable() -> None:
    weak_language = _weak_language()
    kwargs = {
        "runner": _transfer_runner(weak_language),
        "leakage_floors": {language: 0.25 for language in sorted(SUPPORTED_LANGUAGES)},
        "ci_resamples": 50,
        "ci_seed": 23,
    }

    first = cross_lingual_transfer_report(
        "unit-transfer-model",
        _transfer_fixtures(),
        **kwargs,
    )
    second = cross_lingual_transfer_report(
        "unit-transfer-model",
        _transfer_fixtures(),
        **kwargs,
    )

    assert (
        first.transfer_gaps[weak_language].confidence_interval
        == second.transfer_gaps[weak_language].confidence_interval
    )
    assert first.to_json() == second.to_json()
    assert first.to_markdown() == second.to_markdown()
    assert first.to_json() == first.to_json()
    assert first.to_markdown() == first.to_markdown()

    payload = json.loads(first.to_json())
    assert payload["artifact_type"] == "openmed.cross_lingual_transfer_matrix"
    assert len(payload["languages"]) == len(SUPPORTED_LANGUAGES)
    assert "EN-ID-0" not in first.to_json()
    assert "EN-ID-0" not in first.to_markdown()


def test_harness_wrapper_preserves_requested_transfer_suite_name() -> None:
    report = run_cross_lingual_transfer(
        _transfer_fixtures(),
        suite="cross-lingual-fixtures",
        model_name="unit-transfer-model",
        runner=_transfer_runner(_weak_language()),
        ci_resamples=10,
        ci_seed=3,
    )

    assert report.suite == "cross-lingual-fixtures"
    assert report["suite"] == "cross-lingual-fixtures"


def _transfer_fixtures() -> list[BenchmarkFixture]:
    fixtures: list[BenchmarkFixture] = []
    for language in sorted(SUPPORTED_LANGUAGES):
        for index in range(2):
            text = f"{language.upper()}-ID-{index}"
            fixtures.append(
                BenchmarkFixture.from_mapping(
                    {
                        "id": f"{language}-{index}",
                        "language": language,
                        "text": text,
                        "gold_spans": [
                            {
                                "start": 0,
                                "end": len(text),
                                "label": "ID_NUM",
                                "text": text,
                            }
                        ],
                    }
                )
            )
    return fixtures


def _transfer_runner(weak_language: str):
    def run_fixture(fixture: BenchmarkFixture, model_name: str, device: str):
        assert model_name == "unit-transfer-model"
        assert device == "cpu"
        source_language = fixture.metadata["source_language"]
        target_language = fixture.metadata["target_language"]
        if target_language == weak_language and source_language != target_language:
            return []
        return [
            {
                "start": span.start,
                "end": span.end,
                "label": span.label,
                "text": span.text,
            }
            for span in fixture.gold_spans
        ]

    return run_fixture


def _weak_language() -> str:
    return "th" if "th" in SUPPORTED_LANGUAGES else sorted(SUPPORTED_LANGUAGES)[-1]
