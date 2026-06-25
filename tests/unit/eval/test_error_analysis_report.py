"""Tests for per-entity eval error-analysis reports."""

from __future__ import annotations

import json

from openmed.eval.error_analysis import MISSED, SPURIOUS, error_report
from openmed.eval.harness import BenchmarkFixture


def test_error_report_counts_confusions_missed_and_spurious() -> None:
    text = "Patient John is 47. Visit planned 2026-01-01 in Room 4."
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "note-1",
            "text": text,
            "language": "en",
            "gold_spans": [
                _span(text, "John", "PERSON"),
                _span(text, "47", "AGE"),
                _span(text, "2026-01-01", "DATE"),
            ],
            "metadata": {
                "predicted_spans": [
                    _span(text, "John", "PERSON"),
                    _span(text, "47", "DATE"),
                    _span(text, "Room 4", "LOCATION"),
                ]
            },
        }
    )

    report = error_report(
        "test-model",
        [fixture],
        suite_name="golden",
        runner=_metadata_runner,
        context_window=4,
    )

    matrix = report.confusion_matrix
    assert matrix["PERSON"]["PERSON"] == 1
    assert matrix["AGE"]["DATE"] == 1
    assert matrix["DATE"][MISSED] == 1
    assert matrix[SPURIOUS]["LOCATION"] == 1

    age_false_negative = report.false_negatives["AGE"][0]
    assert age_false_negative.kind == "label_confusion"
    assert age_false_negative.matched_label == "DATE"
    assert age_false_negative.context_start == _span(text, "47", "AGE")["start"] - 4

    date_false_positive = report.false_positives["DATE"][0]
    assert date_false_positive.kind == "label_confusion"
    assert date_false_positive.matched_label == "AGE"

    spurious_location = report.false_positives["LOCATION"][0]
    assert spurious_location.kind == SPURIOUS
    assert spurious_location.fixture_id == "note-1"
    assert "text" not in spurious_location.to_dict()
    assert "John" not in report.to_json()


def test_error_report_caps_examples_per_label_without_capping_counts() -> None:
    text = "A John B Jane C Room D Hall"
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "note-2",
            "text": text,
            "language": "en",
            "gold_spans": [
                _span(text, "John", "PERSON"),
                _span(text, "Jane", "PERSON"),
            ],
            "metadata": {
                "predicted_spans": [
                    _span(text, "Room", "PHONE"),
                    _span(text, "Hall", "PHONE"),
                ]
            },
        }
    )

    report = error_report(
        "test-model",
        [fixture],
        runner=_metadata_runner,
        example_cap=1,
    )

    assert report.confusion_matrix["PERSON"][MISSED] == 2
    assert report.confusion_matrix[SPURIOUS]["PHONE"] == 2
    assert len(report.false_negatives["PERSON"]) == 1
    assert len(report.false_positives["PHONE"]) == 1


def test_error_analysis_report_serializes_deterministically() -> None:
    text = "Patient Ana is 33."
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "note-3",
            "text": text,
            "language": "en",
            "gold_spans": [_span(text, "33", "AGE")],
            "metadata": {"predicted_spans": [_span(text, "33", "DATE")]},
        }
    )

    report = error_report(
        "test-model",
        [fixture],
        suite_name="golden",
        runner=_metadata_runner,
        generated_at="2026-06-24T00:00:00Z",
        metadata={"z": 1, "a": {"b": True}},
    )

    assert report.to_json() == report.to_json()
    payload = json.loads(report.to_json())
    assert payload["confusion_matrix"]["AGE"]["DATE"] == 1
    assert payload["false_negatives"]["AGE"][0]["matched_label"] == "DATE"

    markdown = report.to_markdown()
    assert markdown == report.to_markdown()
    assert "| `AGE` | `DATE` | 1 |" in markdown
    assert "| `a.b` | true |" in markdown
    assert "| `z` | 1 |" in markdown


def _metadata_runner(fixture, model_name, device):
    assert model_name == "test-model"
    assert device == "cpu"
    return fixture.metadata["predicted_spans"]


def _span(text: str, value: str, label: str, occurrence: int = 0) -> dict[str, object]:
    start = 0
    for _ in range(occurrence + 1):
        index = text.index(value, start)
        start = index + 1
    return {"start": index, "end": index + len(value), "label": label}
