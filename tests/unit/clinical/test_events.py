"""Tests for deterministic n-ary clinical event extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openmed.clinical import (
    ASSISTIVE_EVENT_DISCLAIMER,
    extract_lab_trend_events,
    extract_medication_change_events,
    link_lab_value_attributes,
    score_event_frame_corpus,
)

_CORPUS_PATH = (
    Path(__file__).parents[2] / "fixtures" / "clinical" / "event_frames.jsonl"
)


def test_medication_change_event_exposes_disclaimer_and_offsets() -> None:
    text = "On Monday, warfarin was increased from 2 mg to 4 mg."

    frames = extract_medication_change_events(
        text,
        [
            {"id": "time", "label": "time", "start": 3, "end": 9},
            {"id": "drug", "label": "drug", "start": 11, "end": 19},
            {"id": "old-dose", "label": "old_dose", "start": 39, "end": 43},
            {"id": "new-dose", "label": "new_dose", "start": 47, "end": 51},
        ],
    )

    assert len(frames) == 1
    frame = frames[0].to_dict()
    assert frame["disclaimer"] == ASSISTIVE_EVENT_DISCLAIMER
    assert "not a clinical decision" in frame["disclaimer"]
    assert frame["roles"]["action"][0]["start"] == 24
    assert frame["roles"]["action"][0]["end"] == 33
    assert frame["roles"]["drug"][0]["value"] == "warfarin"
    assert frame["roles"]["drug"][0]["start"] == 11
    assert frame["roles"]["new_dose"][0]["end"] == 51
    assert frame["provenance"]["role_graph"]["metadata"]["edge_count"] == 4


def test_lab_trend_event_can_consume_lab_value_graph_mentions() -> None:
    text = "Creatinine increased to 2.1 over the past 48 hours."
    lab_graph = link_lab_value_attributes(
        [
            {"id": "creatinine", "label": "lab_name", "start": 0, "end": 10},
            {"id": "creatinine-value", "label": "lab_value", "start": 24, "end": 27},
        ]
    )

    frames = extract_lab_trend_events(text, lab_value_graph=lab_graph)

    assert len(frames) == 1
    frame = frames[0]
    assert frame.role_slots("direction")[0].value == "rising"
    assert frame.role_slots("analyte")[0].value == "Creatinine"
    assert frame.role_slots("magnitude")[0].value == "2.1"
    assert frame.role_slots("time_window")[0].value == "over the past 48 hours"
    assert not frame.cardinality_violations()


def test_conflicting_medication_actions_are_explicit_conflicts() -> None:
    text = "Lasix was increased to 40 mg today. Lasix was held today."
    frames = extract_medication_change_events(
        text,
        [
            {"id": "drug-a", "label": "drug", "start": 0, "end": 5},
            {"id": "new-dose", "label": "new_dose", "start": 23, "end": 28},
            {"id": "time-a", "label": "time", "start": 29, "end": 34},
            {"id": "drug-b", "label": "drug", "start": 36, "end": 41},
            {"id": "time-b", "label": "time", "start": 51, "end": 56},
        ],
    )

    assert [frame.role_slots("action")[0].value for frame in frames] == [
        "increased",
        "held",
    ]
    assert all(frame.conflicts for frame in frames)
    assert {frame.conflicts[0].conflict_type for frame in frames} == {
        "contradictory_medication_actions"
    }


def test_synthetic_gold_event_corpus_meets_ci_gate() -> None:
    predicted_by_case: dict[str, list[Any]] = {}
    gold_by_case: dict[str, list[Any]] = {}
    for case in _load_corpus():
        case_id = str(case["id"])
        predicted_by_case[case_id] = _extract_case(case)
        gold_by_case[case_id] = list(case["gold"])

    score = score_event_frame_corpus(predicted_by_case, gold_by_case)

    assert score.slot_micro_f1 >= 0.80
    assert score.whole_frame_exact_match >= 0.65
    assert score.cardinality_violations == 0


def _load_corpus() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in _CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _extract_case(case: dict[str, Any]) -> list[Any]:
    if case["event_type"] == "medication_change":
        return extract_medication_change_events(case["text"], case["mentions"])
    if case["event_type"] == "lab_trend":
        return extract_lab_trend_events(case["text"], case["mentions"])
    raise AssertionError(f"unexpected event_type {case['event_type']!r}")
