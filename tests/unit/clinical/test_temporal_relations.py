"""Tests for deterministic temporal TLINK candidate extraction."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    TEMPORAL_RELATION_TYPES,
    TemporalRelationCandidate,
    extract_tlink_candidates,
    generate_span_pair_candidates,
)


@pytest.mark.parametrize(
    ("text", "source_text", "target_text", "expected_type"),
    [
        ("Chest pain before dyspnea.", "Chest pain", "dyspnea", "BEFORE"),
        ("Dyspnea after chest pain.", "Dyspnea", "chest pain", "AFTER"),
        ("Cough overlapped with fever.", "Cough", "fever", "OVERLAP"),
        (
            "Hospitalization contained intubation.",
            "Hospitalization",
            "intubation",
            "CONTAINS",
        ),
        ("Symptoms began on 2026-06-01.", "Symptoms", "2026-06-01", "BEGINS_ON"),
        (
            "Antibiotics ended on 2026-06-05.",
            "Antibiotics",
            "2026-06-05",
            "ENDS_ON",
        ),
    ],
)
def test_extract_tlink_candidates_supports_all_relation_types(
    text: str,
    source_text: str,
    target_text: str,
    expected_type: str,
) -> None:
    candidates = extract_tlink_candidates(text, _spans(text, source_text, target_text))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.relation_type == expected_type
    assert _surface(text, candidate.source.start, candidate.source.end) == source_text
    assert _surface(text, candidate.target.start, candidate.target.end) == target_text
    assert candidate.confidence >= 0.5
    assert candidate.cue.start < candidate.cue.end
    assert candidate.source.role in {"EVENT", "TIMEX"}
    assert candidate.target.role in {"EVENT", "TIMEX"}


def test_temporal_relations_reuse_shared_span_pair_generator() -> None:
    text = "Chest pain before dyspnea."
    spans = _spans(text, "Chest pain", "dyspnea")

    pairs = generate_span_pair_candidates(
        text,
        spans,
        include_labels={"EVENT", "TIMEX"},
    )
    candidates = extract_tlink_candidates(text, spans)

    assert len(pairs) == 1
    assert pairs[0].left.start == 0
    assert pairs[0].right.start == text.index("dyspnea")
    assert candidates[0].features["pair_char_distance"] == float(pairs[0].char_distance)


def test_tlink_candidate_output_never_surfaces_raw_note_text() -> None:
    text = "Chest pain before dyspnea."
    candidate = extract_tlink_candidates(text, _spans(text, "Chest pain", "dyspnea"))[0]

    payload = candidate.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert "Chest pain" not in serialized
    assert "dyspnea" not in serialized
    assert "before" not in serialized
    assert '"text"' not in serialized
    assert payload["source"]["text_hash"].startswith("sha256:")
    assert payload["target"]["text_hash"].startswith("sha256:")
    assert payload["cue"]["text_hash"].startswith("sha256:")


def test_tlink_candidates_are_deterministic_for_repeated_extraction() -> None:
    text = "Chest pain before dyspnea. Dyspnea after surgery."
    spans = [
        *_spans(text, "Chest pain", "dyspnea"),
        _span(text, "surgery", "EVENT"),
    ]
    baseline = _payload(extract_tlink_candidates(text, spans))

    for _ in range(50):
        assert (
            _payload(extract_tlink_candidates(text, list(reversed(spans)))) == baseline
        )


def test_temporal_relation_type_registry_lists_expected_schema() -> None:
    assert TEMPORAL_RELATION_TYPES == (
        "BEFORE",
        "AFTER",
        "OVERLAP",
        "CONTAINS",
        "BEGINS_ON",
        "ENDS_ON",
    )


def _spans(text: str, source_text: str, target_text: str) -> list[dict[str, object]]:
    target_label = "TIMEX" if any(char.isdigit() for char in target_text) else "EVENT"
    return [
        _span(text, source_text, "EVENT"),
        _span(text, target_text, target_label),
    ]


def _span(text: str, value: str, label: str) -> dict[str, object]:
    start = text.index(value)
    return {
        "text": value,
        "label": label,
        "start": start,
        "end": start + len(value),
        "score": 0.99,
    }


def _surface(text: str, start: int, end: int) -> str:
    return text[start:end]


def _payload(candidates: tuple[TemporalRelationCandidate, ...]) -> str:
    return json.dumps(
        [candidate.to_dict() for candidate in candidates],
        sort_keys=True,
        separators=(",", ":"),
    )
