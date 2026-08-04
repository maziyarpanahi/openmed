"""Tests for deterministic n-ary clinical event extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openmed.clinical import (
    ASSISTIVE_EVENT_DISCLAIMER,
    EventFrame,
    RoleSlot,
    attach_coreference_representatives,
    extract_lab_trend_events,
    extract_medication_change_events,
    link_lab_value_attributes,
    resolve_coreference,
    score_event_frame_corpus,
)
from openmed.core.schemas import OpenMedSpan, hmac_text_hash

_CORPUS_PATH = (
    Path(__file__).parents[2] / "fixtures" / "clinical" / "event_frames.jsonl"
)
_COREFERENCE_FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "clinical" / "event_frame_coreference.json"
)
_SYNTHETIC_HASH_SECRET = "synthetic-event-frame-secret"


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


def test_extracted_test_and_treatment_heads_use_coreference_representatives() -> None:
    cases = _load_coreference_cases()
    for case in cases[:2]:
        text = str(case["text"])
        chains = _coreference_chains(case)
        mentions = _event_mentions(case)
        if case["family"] == "treatment":
            frames = extract_medication_change_events(
                text,
                mentions,
                coreference_chains=chains,
            )
            trigger_role = "action"
        else:
            frames = extract_lab_trend_events(
                text,
                mentions,
                coreference_chains=chains,
            )
            trigger_role = "direction"

        frame = next(
            candidate
            for candidate in frames
            if candidate.role_slots(trigger_role)[0].value == case["target_trigger"]
        )
        head = frame.role_slots(str(case["head_role"]))[0]
        representative = chains[0].representative
        source_start, source_end = _offset(
            text,
            str(case["head_surface"]),
            1,
        )

        assert (head.start, head.end) == (
            representative.start,
            representative.end,
        )
        assert head.value == text[representative.start : representative.end]
        assert head.cluster_id == chains[0].chain_id
        assert frame.provenance["coreference"]["cluster_ids"] == [chains[0].chain_id]
        assert len(frame.role_slots(str(case["attribute_role"]))) == 1
        coreference = head.provenance["coreference"]
        assert coreference["representative"] == {
            "start": representative.start,
            "end": representative.end,
            "text_hash": representative.text_hash,
        }
        assert coreference["source_spans"] == [
            {
                "start": source_start,
                "end": source_end,
                "text_hash": chains[0].members[1].text_hash,
            }
        ]
        serialized_provenance = json.dumps(head.to_dict()["provenance"]).casefold()
        assert str(case["head_surface"]).casefold() not in serialized_provenance


def test_problem_mentions_consolidate_once_on_the_cluster_representative() -> None:
    case = _load_coreference_cases()[2]
    text = str(case["text"])
    chains = _coreference_chains(case)
    first_start, first_end = _offset(text, str(case["head_surface"]), 0)
    second_start, second_end = _offset(text, str(case["head_surface"]), 1)
    severity_start, severity_end = _offset(text, str(case["attribute_surface"]), 0)
    frame = EventFrame(
        frame_id="synthetic-problem-frame",
        event_type="lab_trend",
        roles={
            "problem": (
                RoleSlot(
                    role="problem",
                    value=str(case["head_surface"]),
                    start=first_start,
                    end=first_end,
                    label="PROBLEM",
                    source_id="synthetic-problem-1",
                ),
                RoleSlot(
                    role="problem",
                    value=str(case["head_surface"]),
                    start=second_start,
                    end=second_end,
                    label="PROBLEM",
                    source_id="synthetic-problem-2",
                ),
            ),
            "severity": (
                RoleSlot(
                    role="severity",
                    value=str(case["attribute_surface"]),
                    start=severity_start,
                    end=severity_end,
                    label="SEVERITY",
                ),
            ),
        },
    )

    canonical = attach_coreference_representatives(frame, chains, text)

    problem_slots = canonical.role_slots("problem")
    assert len(problem_slots) == 1
    assert (problem_slots[0].start, problem_slots[0].end) == (
        first_start,
        first_end,
    )
    assert problem_slots[0].cluster_id == chains[0].chain_id
    assert problem_slots[0].provenance["coreference"]["source_spans"] == [
        {
            "start": first_start,
            "end": first_end,
            "text_hash": chains[0].members[0].text_hash,
        },
        {
            "start": second_start,
            "end": second_end,
            "text_hash": chains[0].members[1].text_hash,
        },
    ]
    assert len(canonical.role_slots("severity")) == 1
    assert canonical.role_slots("severity")[0] == frame.role_slots("severity")[0]


def _load_corpus() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in _CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_coreference_cases() -> list[dict[str, Any]]:
    return json.loads(_COREFERENCE_FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


def _offset(text: str, surface: str, occurrence: int) -> tuple[int, int]:
    cursor = 0
    for _ in range(occurrence + 1):
        start = text.index(surface, cursor)
        cursor = start + len(surface)
    return start, start + len(surface)


def _coreference_chains(case: dict[str, Any]) -> tuple[Any, ...]:
    text = str(case["text"])
    spans = []
    for occurrence in case["head_occurrences"]:
        start, end = _offset(text, str(case["head_surface"]), int(occurrence))
        spans.append(
            OpenMedSpan(
                doc_id=str(case["id"]),
                start=start,
                end=end,
                text_hash=hmac_text_hash(
                    text[start:end],
                    _SYNTHETIC_HASH_SECRET,
                ),
                entity_type=str(case["entity_type"]),
                canonical_label=str(case["canonical_label"]),
            )
        )
    chains, _index = resolve_coreference(spans, text)
    assert len(chains) == 1
    return chains


def _event_mentions(case: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(case["text"])
    mentions = []
    for item in case["event_mentions"]:
        start, end = _offset(
            text,
            str(item["surface"]),
            int(item["occurrence"]),
        )
        mentions.append(
            {
                "id": item["id"],
                "label": item["label"],
                "start": start,
                "end": end,
                "text_hash": hmac_text_hash(
                    text[start:end],
                    _SYNTHETIC_HASH_SECRET,
                ),
            }
        )
    return mentions


def _extract_case(case: dict[str, Any]) -> list[Any]:
    if case["event_type"] == "medication_change":
        return extract_medication_change_events(case["text"], case["mentions"])
    if case["event_type"] == "lab_trend":
        return extract_lab_trend_events(case["text"], case["mentions"])
    raise AssertionError(f"unexpected event_type {case['event_type']!r}")
