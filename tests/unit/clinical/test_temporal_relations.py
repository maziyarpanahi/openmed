"""Synthetic tests for deterministic temporal TLINK candidates."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from openmed.clinical import (
    TEMPORAL_GRAPH_SCHEMA_VERSION,
    TEMPORAL_RELATION_TYPES,
    TemporalCueReference,
    TemporalRelationCandidate,
    TemporalSpanReference,
    decode_tlink_candidates,
    extract_tlink_candidates,
)
from openmed.clinical.relations.candidate import build_relation_candidates


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
    assert text[candidate.source.start : candidate.source.end] == source_text
    assert text[candidate.target.start : candidate.target.end] == target_text
    assert candidate.confidence >= 0.5
    assert candidate.cue.start < candidate.cue.end
    assert candidate.source.role == "EVENT"
    assert candidate.target.role in {"EVENT", "TIMEX"}


def test_during_cue_directs_contains_from_timex_to_event() -> None:
    text = "Therapy continued during June 2026."
    candidates = extract_tlink_candidates(
        text,
        [_span(text, "Therapy", "EVENT"), _span(text, "June 2026", "TIMEX")],
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.relation_type == "CONTAINS"
    assert candidate.source.role == "TIMEX"
    assert candidate.target.role == "EVENT"


def test_preceded_by_emits_only_the_directed_after_relation() -> None:
    text = "Dyspnea preceded by chest pain."
    candidates = extract_tlink_candidates(text, _spans(text, "Dyspnea", "chest pain"))

    assert [candidate.relation_type for candidate in candidates] == ["AFTER"]


def test_temporal_relations_reuse_shared_relation_candidate_generator() -> None:
    text = "Chest pain before dyspnea."
    spans = _spans(text, "Chest pain", "dyspnea")

    with patch(
        "openmed.clinical.relations.temporal.build_relation_candidates",
        wraps=build_relation_candidates,
    ) as generator:
        candidates = extract_tlink_candidates(text, spans)

    generator.assert_called_once()
    assert candidates[0].features["pair_char_distance"] == 8.0


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


def test_tlink_candidates_are_stable_for_reordered_input() -> None:
    text = "Chest pain before dyspnea."
    spans = _spans(text, "Chest pain", "dyspnea")
    baseline = _payload(extract_tlink_candidates(text, spans))

    for _ in range(25):
        assert _payload(extract_tlink_candidates(text, reversed(spans))) == baseline


def test_timex_to_timex_candidates_are_not_emitted() -> None:
    text = "Monday before Tuesday."

    assert (
        extract_tlink_candidates(
            text,
            [_span(text, "Monday", "TIMEX"), _span(text, "Tuesday", "TIMEX")],
        )
        == ()
    )


def test_temporal_relation_registry_lists_the_tlink_schema() -> None:
    assert TEMPORAL_RELATION_TYPES == (
        "BEFORE",
        "AFTER",
        "OVERLAP",
        "CONTAINS",
        "BEGINS_ON",
        "ENDS_ON",
    )


def test_decode_tlinks_prunes_weaker_before_after_contradiction() -> None:
    event_a = _temporal_span("event-a", 0)
    event_b = _temporal_span("event-b", 10)
    graph = decode_tlink_candidates(
        [
            _candidate("BEFORE", event_a, event_b, 0.95),
            _candidate("AFTER", event_a, event_b, 0.40),
        ]
    )

    assert graph.edge_keys() == (("BEFORE", "event-a", "event-b"),)
    contradiction = next(
        decision
        for decision in graph.decisions
        if decision.edge.metadata["candidate_relation_type"] == "AFTER"
    )
    assert contradiction.status == "pruned"
    assert contradiction.constraint == "acyclicity:BEFORE"


def test_decode_tlinks_returns_consistent_transitive_reduction() -> None:
    event_a = _temporal_span("event-a", 0)
    event_b = _temporal_span("event-b", 10)
    event_c = _temporal_span("event-c", 20)
    candidates = [
        _candidate("BEFORE", event_a, event_b, 0.95),
        _candidate("BEFORE", event_b, event_c, 0.90),
        _candidate("BEFORE", event_a, event_c, 0.80),
        _candidate("AFTER", event_a, event_c, 0.20),
    ]

    graph = decode_tlink_candidates(reversed(candidates))

    assert graph.edge_keys() == (
        ("BEFORE", "event-a", "event-b"),
        ("BEFORE", "event-b", "event-c"),
    )
    closure = _before_closure(graph.edge_keys())
    assert ("event-a", "event-c") in closure
    assert all((tail, head) not in closure for head, tail in closure)
    pruned = {
        decision.edge.metadata["candidate_relation_type"]: decision.constraint
        for decision in graph.decisions
        if decision.status == "pruned"
    }
    assert pruned["BEFORE"] == "transitive_reduction:BEFORE"
    assert pruned["AFTER"] == "acyclicity:BEFORE"


def test_decode_tlinks_retains_non_ordering_relations_and_safe_provenance() -> None:
    event_a = _temporal_span("event-a", 0)
    event_b = _temporal_span("event-b", 10)
    timex = _temporal_span("timex-a", 20, role="TIMEX")
    graph = decode_tlink_candidates(
        [
            _candidate("OVERLAP", event_b, event_a, 0.80),
            _candidate("CONTAINS", timex, event_a, 0.85),
            _candidate("BEGINS_ON", event_b, timex, 0.90),
            _candidate("ENDS_ON", event_a, timex, 0.75),
        ]
    )

    assert set(graph.edge_keys()) == {
        ("OVERLAP", "event-a", "event-b"),
        ("CONTAINS", "timex-a", "event-a"),
        ("BEGINS_ON", "event-b", "timex-a"),
        ("ENDS_ON", "event-a", "timex-a"),
    }
    payload = json.dumps(graph.to_dict(), sort_keys=True)
    assert "synthetic symptom" not in payload
    assert "synthetic date" not in payload
    assert '"text"' not in payload
    assert all(
        edge.metadata["schema_version"] == TEMPORAL_GRAPH_SCHEMA_VERSION
        for edge in graph.edges
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
        "label": label,
        "start": start,
        "end": start + len(value),
        "score": 0.99,
    }


def _payload(candidates: tuple[TemporalRelationCandidate, ...]) -> str:
    return json.dumps(
        [candidate.to_dict() for candidate in candidates],
        sort_keys=True,
        separators=(",", ":"),
    )


_SAFE_HASH = "sha256:" + "0" * 64


def _temporal_span(
    span_id: str,
    start: int,
    *,
    role: str = "EVENT",
) -> TemporalSpanReference:
    label = "TIMEX" if role == "TIMEX" else "EVENT"
    return TemporalSpanReference(
        span_id=span_id,
        label=label,
        role=role,
        start=start,
        end=start + 5,
        score=0.99,
        text_hash=_SAFE_HASH,
    )


def _candidate(
    relation_type: str,
    source: TemporalSpanReference,
    target: TemporalSpanReference,
    confidence: float,
) -> TemporalRelationCandidate:
    return TemporalRelationCandidate(
        relation_type=relation_type,
        source=source,
        target=target,
        confidence=confidence,
        cue=TemporalCueReference(
            category=relation_type,
            start=100,
            end=101,
            text_hash=_SAFE_HASH,
        ),
        provenance={"fixture": "synthetic-offline"},
    )


def _before_closure(
    edge_keys: tuple[tuple[str, str, str], ...],
) -> set[tuple[str, str]]:
    adjacency: dict[str, set[str]] = {}
    for label, head, tail in edge_keys:
        if label == "BEFORE":
            adjacency.setdefault(head, set()).add(tail)

    closure: set[tuple[str, str]] = set()
    for source in adjacency:
        stack = list(adjacency[source])
        while stack:
            target = stack.pop()
            if (source, target) in closure:
                continue
            closure.add((source, target))
            stack.extend(adjacency.get(target, ()))
    return closure
