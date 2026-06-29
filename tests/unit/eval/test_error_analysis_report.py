"""Tests for per-entity eval error-analysis reports."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from openmed.eval.error_analysis import (
    MISSED,
    SPURIOUS,
    error_report,
    mine_gate_failure_labeling_queue,
)
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


def test_labeling_queue_from_error_report_is_phi_free_and_tracks_provenance() -> None:
    text = "Patient Jordan Smith has SSN 123-45-6789 in Room 4."
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "note-jordan-smith",
            "text": text,
            "language": "en",
            "gold_spans": [
                _span(text, "Jordan Smith", "PERSON"),
                _span(text, "123-45-6789", "SSN"),
            ],
            "metadata": {"predicted_spans": []},
        }
    )
    report = error_report(
        "test-model",
        [fixture],
        suite_name="synthetic",
        runner=_metadata_runner,
        metadata={"language": "en"},
    )

    artifact = mine_gate_failure_labeling_queue(
        report,
        gate_run_hash="sha256:gate-run",
        report_hash="sha256:error-report",
        label_gate_impacts={"SSN": 4.0, "PERSON": 1.0},
    )
    payload = artifact.to_dict()

    assert [item["label"] for item in payload["items"]] == ["SSN", "PERSON"]
    assert payload == artifact.to_dict()
    assert "artifact_hash" in payload
    assert all(
        item["provenance"]["gate_run_hash"] == "sha256:gate-run"
        and item["provenance"]["report_hash"] == "sha256:error-report"
        for item in payload["items"]
    )
    assert all("fixture_hash" in item["provenance"] for item in payload["items"])
    _assert_no_raw_phi(
        payload,
        ("Jordan", "Smith", "Jordan Smith", "123-45-6789", "note-jordan-smith"),
    )


def test_labeling_queue_candidate_context_is_sanitized() -> None:
    artifact = mine_gate_failure_labeling_queue(
        [
            {
                "label": "PERSON",
                "kind": MISSED,
                "language": "en",
                "span_hash": "sha256:person",
                "surrogate_context": "Patient José Álvarez called 555-111-2222.",
                "text": "José Álvarez",
                "source_text": "Patient José Álvarez called 555-111-2222.",
                "uncertainty": 0.8,
                "gate_impact": 2.0,
            }
        ],
        gate_run_hash="sha256:gate-run",
        report_hash="sha256:candidate-report",
    )

    payload = artifact.to_dict()

    assert payload["items"][0]["surrogate_context"] == (
        "<WORD> <WORD> <WORD> <WORD> <PHONE>."
    )
    _assert_no_raw_phi(
        payload,
        ("José", "Álvarez", "José Álvarez", "555-111-2222"),
    )


def test_labeling_queue_dedupe_reduces_duplicates_and_keeps_modes() -> None:
    names = (
        "Jordan Smith",
        "Casey Brown",
        "Riley Jones",
        "Morgan Davis",
        "Avery Miller",
        "Quinn Wilson",
        "Rowan Moore",
        "Taylor Clark",
        "Jamie Hall",
        "Robin Young",
    )
    candidates: list[dict[str, Any]] = [
        {
            "label": "PERSON",
            "kind": MISSED,
            "language": "en",
            "span_hash": f"sha256:person-{index}",
            "surrogate_context": f"Patient {name} called today.",
            "uncertainty": 0.9,
            "gate_impact": 2.0,
        }
        for index, name in enumerate(names)
    ]
    candidates.extend(
        [
            {
                "label": "DATE",
                "kind": MISSED,
                "language": "en",
                "span_hash": f"sha256:date-{index}",
                "surrogate_context": "Visit date 2026-07-04 was missed.",
                "uncertainty": 0.7,
                "gate_impact": 1.5,
            }
            for index in range(10)
        ]
    )

    artifact = mine_gate_failure_labeling_queue(
        candidates,
        gate_run_hash="sha256:gate-run",
        report_hash="sha256:duplicates",
    )
    labels = {item.label for item in artifact.items}

    assert artifact.raw_candidate_count == 20
    assert artifact.dropped_duplicate_count == 18
    assert artifact.duplicate_reduction_rate >= 0.8
    assert labels == {"DATE", "PERSON"}


def test_labeling_queue_ranking_is_deterministic_and_impact_weighted() -> None:
    candidates = [
        {
            "label": "PERSON",
            "kind": MISSED,
            "language": "en",
            "span_hash": "sha256:person",
            "surrogate_context": "Patient <PERSON> was missed.",
            "uncertainty": 0.9,
            "gate_impact": 1.0,
        },
        {
            "label": "LOCATION",
            "kind": SPURIOUS,
            "language": "en",
            "span_hash": "sha256:location",
            "surrogate_context": "Room <LOCATION> was over-redacted.",
            "uncertainty": 0.4,
            "gate_impact": 5.0,
        },
    ]

    first = mine_gate_failure_labeling_queue(
        candidates,
        gate_run_hash="sha256:gate-run",
        report_hash="sha256:ranking",
    )
    repeat = mine_gate_failure_labeling_queue(
        candidates,
        gate_run_hash="sha256:gate-run",
        report_hash="sha256:ranking",
    )
    reversed_order = mine_gate_failure_labeling_queue(
        list(reversed(candidates)),
        gate_run_hash="sha256:gate-run",
        report_hash="sha256:ranking",
    )
    payload = first.to_dict()

    assert payload == repeat.to_dict()
    assert [item["label"] for item in payload["items"]] == ["LOCATION", "PERSON"]
    assert [item.label for item in reversed_order.items] == ["LOCATION", "PERSON"]
    assert payload["items"][0]["priority"] == 2.0
    assert payload["items"][1]["priority"] == 0.9


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


def _assert_no_raw_phi(payload: Any, raw_values: Iterable[str]) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    strings = tuple(_strings(payload)) + (serialized,)
    for raw_value in raw_values:
        assert all(raw_value not in value for value in strings)


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)
