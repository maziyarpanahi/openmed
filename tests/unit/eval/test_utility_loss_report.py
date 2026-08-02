"""Tests for clinical utility-loss reports."""

from __future__ import annotations

import json

import pytest

from openmed.eval import UtilityLossReport, utility_loss_report
from openmed.eval.utility import CLINICAL_CATEGORIES


def test_utility_loss_report_counts_over_redacted_clinical_categories() -> None:
    text = "Patient John takes aspirin for diabetes. Hemoglobin is high."
    john = _span(text, "John", "PERSON")
    aspirin = _span(text, "aspirin", "MEDICATION")
    diabetes = _span(text, "diabetes", "CONDITION")
    hemoglobin = _span(text, "Hemoglobin", "LAB_TEST")
    fixture = {
        "id": "note-1",
        "text": text,
        "language": "en",
        "gold_spans": [john],
        "clinical_spans": [aspirin, diabetes, hemoglobin],
        "metadata": {
            "predicted_spans": [
                john,
                _span(text, "aspirin", "DATE"),
                _span(text, "diabetes", "LOCATION"),
            ]
        },
    }

    report = utility_loss_report(
        "test-model",
        [fixture],
        suite_name="utility",
        runner=_metadata_runner,
        context_window=4,
    )

    assert isinstance(report, UtilityLossReport)
    assert report.overall_over_redaction.numerator == len("aspirin") + len("diabetes")
    assert report.overall_over_redaction.denominator == len(text) - len("John")
    assert report.by_category["MEDICATION"].rate == pytest.approx(1.0)
    assert report.by_category["CONDITION"].rate == pytest.approx(1.0)
    assert report.by_category["LAB_TEST"].rate == pytest.approx(0.0)
    assert report.by_category["MEDICATION"].over_redacted_spans == 1
    assert report.examples["CONDITION"][0].predicted_label == "LOCATION"
    assert report.examples["CONDITION"][0].context_start == diabetes["start"] - 4
    assert "aspirin" not in report.to_json()
    assert "John" not in report.to_markdown()


def test_phi_only_redaction_has_zero_clinical_utility_loss() -> None:
    text = "Patient John takes aspirin."
    john = _span(text, "John", "PERSON")
    fixture = {
        "id": "note-2",
        "text": text,
        "language": "en",
        "gold_spans": [john],
        "clinical_spans": [_span(text, "aspirin", "MEDICATION")],
        "metadata": {"predicted_spans": [john]},
    }

    report = utility_loss_report(
        "test-model",
        [fixture],
        runner=_metadata_runner,
    )

    assert report.overall_over_redaction.rate == 0.0
    assert report.overall_over_redaction.numerator == 0
    assert report.by_category["MEDICATION"].rate == 0.0
    assert report.by_category["MEDICATION"].over_redacted_spans == 0
    assert report.examples["MEDICATION"] == []


def test_examples_are_capped_without_capping_category_counts() -> None:
    text = "Patient takes aspirin and metformin for diabetes."
    aspirin = _span(text, "aspirin", "MEDICATION")
    metformin = _span(text, "metformin", "MEDICATION")
    fixture = {
        "id": "note-3",
        "text": text,
        "language": "en",
        "gold_spans": [],
        "clinical_spans": [
            aspirin,
            metformin,
            _span(text, "diabetes", "CONDITION"),
        ],
        "metadata": {
            "predicted_spans": [
                _span(text, "aspirin", "PERSON"),
                _span(text, "metformin", "PERSON"),
            ]
        },
    }

    report = utility_loss_report(
        "test-model",
        [fixture],
        runner=_metadata_runner,
        example_cap=1,
    )

    assert set(CLINICAL_CATEGORIES).issubset(report.by_category)
    assert report.by_category["MEDICATION"].total_spans == 2
    assert report.by_category["MEDICATION"].over_redacted_spans == 2
    assert report.by_category["MEDICATION"].over_redacted_chars == len("aspirin") + len(
        "metformin"
    )
    assert len(report.examples["MEDICATION"]) == 1
    assert report.by_category["CONDITION"].total_spans == 1
    assert report.by_category["CONDITION"].rate == 0.0


def test_utility_loss_report_serializes_deterministically() -> None:
    text = "Patient Ana takes warfarin."
    fixture = {
        "id": "note-4",
        "text": text,
        "language": "en",
        "gold_spans": [_span(text, "Ana", "PERSON")],
        "clinical_spans": [_span(text, "warfarin", "MEDICATION")],
        "metadata": {"predicted_spans": [_span(text, "warfarin", "PERSON")]},
    }

    report = utility_loss_report(
        "test-model",
        [fixture],
        suite_name="utility",
        runner=_metadata_runner,
        generated_at="2026-06-28T00:00:00Z",
        metadata={"z": 1, "a": {"b": True}},
    )

    assert report.to_json() == report.to_json()
    payload = json.loads(report.to_json())
    assert payload["overall_over_redaction"]["numerator"] == len("warfarin")
    assert payload["by_category"]["MEDICATION"]["rate"] == 1.0
    assert payload["examples"]["MEDICATION"][0]["text_hash"].startswith("sha256:")

    markdown = report.to_markdown()
    assert markdown == report.to_markdown()
    assert "# Utility Loss Report: utility" in markdown
    assert "| `MEDICATION` | 1 | 8 | 8 | 1 | 1 |" in markdown
    assert "| `a.b` | true |" in markdown
    assert "| `z` | 1 |" in markdown
    assert "warfarin" not in markdown


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
