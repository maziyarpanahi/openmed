"""Tests for bounded, privacy-safe entity-pair candidate generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical.relations import generate_relation_candidates
from openmed.eval.golden.loader import list_fixture_paths

FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "openmed"
    / "eval"
    / "golden"
    / "fixtures"
    / "relation_candidates.jsonl"
)


def _cases() -> list[dict[str, object]]:
    return [json.loads(line) for line in FIXTURE.read_text().splitlines() if line]


def _signature(candidate: object) -> tuple[int, int, int, int, str]:
    return (
        candidate.head.start,
        candidate.head.end,
        candidate.tail.start,
        candidate.tail.end,
        candidate.relation_type,
    )


def test_synthetic_gold_recall_and_cross_section_pruning() -> None:
    true_positive_count = 0
    gold_count = 0

    for case in _cases():
        candidates = generate_relation_candidates(
            case["text"],
            case["spans"],
            max_window=1,
            allow_adjacent_sections=False,
        )
        predicted = {_signature(candidate) for candidate in candidates}
        gold = {tuple(pair) for pair in case["gold_pairs"]}
        pruned = {tuple(pair) for pair in case["pruned_pairs"]}
        true_positive_count += len(predicted & gold)
        gold_count += len(gold)

        assert not predicted & pruned
        for candidate in candidates:
            assert candidate.head.offset_key() in {
                (span["start"], span["end"]) for span in case["spans"]
            }
            assert candidate.tail.offset_key() in {
                (span["start"], span["end"]) for span in case["spans"]
            }

    assert true_positive_count / gold_count >= 0.95


def test_relation_candidate_fixture_is_not_loaded_as_deidentification_gold() -> None:
    assert FIXTURE not in list_fixture_paths()


def test_output_is_deterministic_and_contains_no_note_text() -> None:
    case = _cases()[0]

    first = generate_relation_candidates(case["text"], reversed(case["spans"]))
    second = generate_relation_candidates(case["text"], case["spans"])

    assert first == second
    assert [_signature(candidate) for candidate in first] == sorted(
        _signature(candidate) for candidate in first
    )
    serialized = json.dumps([candidate.to_dict() for candidate in first])
    assert all(candidate.head.text == "" for candidate in first)
    assert all(candidate.tail.text == "" for candidate in first)
    assert all(candidate.head.text_hash for candidate in first)
    assert all(candidate.tail.text_hash for candidate in first)
    for surface in ("Metformin", "500 mg", "daily", "diabetes", "Diabetes"):
        assert surface not in serialized


def test_sentence_window_and_adjacent_section_rules_are_explicit() -> None:
    text = "DrugA started. Rash followed."
    spans = [
        {"start": 0, "end": 5, "label": "MEDICATION", "section": "plan"},
        {"start": 15, "end": 19, "label": "PROBLEM", "section": "assessment"},
    ]

    assert generate_relation_candidates(text, spans, max_window=0) == ()
    assert (
        generate_relation_candidates(
            text,
            spans,
            max_window=1,
            allow_adjacent_sections=False,
        )
        == ()
    )
    [candidate] = generate_relation_candidates(
        text,
        spans,
        max_window=1,
        allow_adjacent_sections=True,
    )
    assert candidate.features["sentence_distance"] == 1.0
    assert candidate.features["adjacent_section"] == 1.0


def test_max_pairs_bounds_a_500_span_document() -> None:
    surfaces = [f"D{index}" if index % 2 == 0 else f"P{index}" for index in range(500)]
    text = " ".join(surfaces)
    spans = []
    cursor = 0
    for index, surface in enumerate(surfaces):
        spans.append(
            {
                "start": cursor,
                "end": cursor + len(surface),
                "label": "MEDICATION" if index % 2 == 0 else "PROBLEM",
            }
        )
        cursor += len(surface) + 1

    candidates = generate_relation_candidates(text, spans, max_window=0, max_pairs=37)

    assert len(candidates) == 37


@pytest.mark.parametrize(
    ("argument", "value"),
    [("max_window", -1), ("max_pairs", -1)],
)
def test_invalid_generation_guards_raise(argument: str, value: int) -> None:
    with pytest.raises(ValueError):
        generate_relation_candidates("x", (), **{argument: value})
