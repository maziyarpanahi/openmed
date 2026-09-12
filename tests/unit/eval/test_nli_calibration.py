"""Tests for deterministic, privacy-safe clinical NLI calibration."""

from __future__ import annotations

import json

import pytest

from openmed.eval import (
    NLICalibrationReport,
    NLIEntailmentFixture,
    build_nli_calibration_report,
    calibrate_nli_thresholds,
)


def _fixtures() -> list[dict[str, object]]:
    return [
        {
            "id": "synthetic-alpha",
            "premise": "Synthetic premise alpha carries marker cobalt.",
            "hypothesis": "The case carries marker cobalt.",
            "gold_label": "entailment",
            "score": 0.90,
        },
        {
            "id": "synthetic-beta",
            "premise": "Synthetic premise beta carries marker ochre.",
            "hypothesis": "The case carries marker cobalt.",
            "gold_label": "contradiction",
            "score": 0.80,
        },
        {
            "id": "synthetic-gamma",
            "premise": "Synthetic premise gamma carries marker jade.",
            "hypothesis": "The case carries a marker.",
            "gold_label": "neutral",
            "score": 0.60,
        },
        {
            "id": "synthetic-delta",
            "premise": "Synthetic premise delta carries marker umber.",
            "hypothesis": "The case carries marker umber.",
            "gold_label": "entailment",
            "score": 0.30,
        },
    ]


def test_threshold_point_reports_abstention_and_confusion_tradeoffs() -> None:
    report = calibrate_nli_thresholds(
        _fixtures(),
        model_id="synthetic-nli-v1",
        thresholds=[0.0, 0.7, 0.85, 1.0],
    )

    points = {point.threshold: point for point in report.threshold_points}
    point = points[0.7]

    assert point.accepted_count == 2
    assert point.abstained_count == 2
    assert point.abstention_rate == pytest.approx(0.5)
    assert point.confusion.true_positives == 1
    assert point.confusion.false_positives == 1
    assert point.confusion.true_negatives == 1
    assert point.confusion.false_negatives == 1
    assert point.precision == pytest.approx(0.5)
    assert point.recall == pytest.approx(0.5)
    assert point.false_positive_rate == pytest.approx(0.5)
    assert point.f1 == pytest.approx(0.5)
    assert point.abstention_by_gold_label == {
        "contradiction": 0.0,
        "entailment": 0.5,
        "neutral": 1.0,
    }


def test_recommendation_respects_precision_floor_and_is_deterministic() -> None:
    first = calibrate_nli_thresholds(
        _fixtures(),
        model_id="synthetic-nli-v1",
        thresholds=[0.0, 0.7, 0.85, 1.0],
        precision_floor=0.8,
    )
    second = calibrate_nli_thresholds(
        list(reversed(_fixtures())),
        model_id="synthetic-nli-v1",
        thresholds=[1.0, 0.85, 0.7, 0.0],
        precision_floor=0.8,
    )

    assert first.recommended_threshold == pytest.approx(0.85)
    assert first.selection == "max_recall_at_precision_floor"
    assert first.recommended_point.precision >= 0.8
    assert first.to_json() == second.to_json()


def test_report_and_fixture_serialization_are_raw_text_free() -> None:
    report = build_nli_calibration_report(
        _fixtures(),
        model_id="synthetic-nli-v1",
        thresholds=[0.7],
    )

    serialized = report.to_json() + report.to_markdown()
    for forbidden in (
        "Synthetic premise alpha",
        "Synthetic premise beta",
        "marker cobalt",
        "marker ochre",
    ):
        assert forbidden not in serialized

    payload = json.loads(report.to_json())
    assert payload["premise_and_hypothesis_included"] is False
    assert payload["fixture_fingerprint"].startswith("sha256:")
    assert payload["model_fingerprint"].startswith("sha256:")
    assert set(payload["recommended_point"]["confusion"]) == {
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    }

    fixture = NLIEntailmentFixture.from_mapping(_fixtures()[0])
    assert "premise" not in fixture.to_dict()
    assert "hypothesis" not in fixture.to_dict()


def test_fingerprints_bind_text_but_do_not_expose_it() -> None:
    first = calibrate_nli_thresholds(_fixtures(), model_id="synthetic-nli-v1")
    changed = _fixtures()
    changed[0]["hypothesis"] = "Synthetic replacement marker vermilion."
    second = calibrate_nli_thresholds(changed, model_id="synthetic-nli-v1")

    assert first.fixture_fingerprint != second.fixture_fingerprint
    assert "vermilion" not in second.to_json()
    assert (
        first.model_fingerprint
        != calibrate_nli_thresholds(
            _fixtures(), model_id="synthetic-nli-v2"
        ).model_fingerprint
    )


def test_default_fixtures_are_offline_and_exported() -> None:
    report = calibrate_nli_thresholds()

    assert isinstance(report, NLICalibrationReport)
    assert report.fixture_count == 4
    assert report.gold_label_counts == {
        "contradiction": 1,
        "entailment": 2,
        "neutral": 1,
    }


def test_invalid_fixture_errors_do_not_echo_raw_text() -> None:
    secret = "synthetic-sensitive-token-should-not-echo"
    with pytest.raises(ValueError, match="unsupported gold label") as raised:
        calibrate_nli_thresholds(
            [
                {
                    "premise": secret,
                    "hypothesis": "Synthetic hypothesis.",
                    "gold_label": secret,
                    "score": 0.5,
                }
            ]
        )

    assert secret not in str(raised.value)
