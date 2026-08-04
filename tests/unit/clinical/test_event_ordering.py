"""Synthetic tests for the privacy-safe event-ordering timeline API."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import Timeline, order_events
from openmed.eval.metrics import compute_temporal_closure_consistency
from openmed.eval.suites.temporal_tlinks import (
    assert_temporal_tlink_gate,
    load_temporal_tlink_fixtures,
)


def test_order_events_returns_cycle_free_discharge_timeline() -> None:
    text = "Admission before surgery before recovery before discharge."
    timeline = order_events(
        text,
        _event_spans(text, "Admission", "surgery", "recovery", "discharge"),
    )

    assert isinstance(timeline, Timeline)
    assert timeline.is_cycle_free
    assert [event.position for event in timeline.events] == [0, 1, 2, 3]
    assert [text[event.start : event.end] for event in timeline.events] == [
        "Admission",
        "surgery",
        "recovery",
        "discharge",
    ]
    assert all(0.0 <= event.confidence <= 1.0 for event in timeline.events)
    assert len(timeline.reduced_graph) == 3
    assert len(timeline.pruned_edges) == 3
    assert all(
        edge.constraint == "transitive_reduction:TEMPORAL_PRECEDES"
        for edge in timeline.pruned_edges
    )
    assert compute_temporal_closure_consistency(timeline).violations == {}


def test_order_events_reverses_after_for_chronological_positions() -> None:
    text = "Recovery after surgery."
    timeline = order_events(text, _event_spans(text, "Recovery", "surgery"))

    assert [text[event.start : event.end] for event in timeline.events] == [
        "surgery",
        "Recovery",
    ]
    assert timeline.kept_edges[0].relation_type == "AFTER"
    assert timeline.kept_edges[0].source.start == text.index("Recovery")


def test_order_events_explains_kept_and_cycle_pruned_candidates() -> None:
    text = "Admission before surgery after recovery."
    timeline = order_events(
        text,
        _event_spans(text, "Admission", "surgery", "recovery"),
    )

    assert timeline.is_cycle_free
    assert {edge.status for edge in timeline.edge_provenance} == {"kept", "pruned"}
    assert any(
        edge.constraint and edge.constraint.startswith("acyclicity:")
        for edge in timeline.pruned_edges
    )
    assert all(edge.reason for edge in timeline.edge_provenance)


def test_order_events_output_never_surfaces_raw_note_text() -> None:
    text = "Admission before surgery after recovery."
    timeline = order_events(
        text,
        _event_spans(text, "Admission", "surgery", "recovery"),
    )

    payload = timeline.to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    assert '"text":' not in serialized
    assert "Admission" not in serialized
    assert "surgery" not in serialized
    assert "recovery" not in serialized
    assert payload["cycle_free"] is True
    assert all(event["text_hash"].startswith("sha256:") for event in payload["events"])
    assert all(
        edge["source"]["text_hash"].startswith("sha256:")
        and edge["target"]["text_hash"].startswith("sha256:")
        and edge["cue"]["text_hash"].startswith("sha256:")
        for edge in payload["edges"]
    )


def test_order_events_is_stable_and_positions_unlinked_events() -> None:
    text = "Admission and discharge."
    spans = _event_spans(text, "Admission", "discharge")
    baseline = order_events(text, spans)
    reordered = order_events(text, reversed(spans))

    assert baseline.to_dict() == reordered.to_dict()
    assert [event.position for event in baseline.events] == [0, 1]
    assert baseline.edges == ()


def test_order_events_anchors_every_event_in_committed_temporal_gold() -> None:
    anchor_sources: set[str] = set()
    for fixture in load_temporal_tlink_fixtures():
        spans = [
            {
                "id": span.span_id,
                "label": span.label,
                "role": span.role,
                "start": span.start,
                "end": span.end,
                "normalized_value": span.normalized_value,
                "is_dct": span.is_dct,
            }
            for span in fixture.spans
        ]
        timeline = order_events(fixture.text, spans)
        dct = next(span for span in fixture.spans if span.is_dct)
        event_spans = [span for span in fixture.spans if span.role == "EVENT"]

        assert timeline.document_creation_time == dct.normalized_value
        assert timeline.is_cycle_free
        assert len(timeline.events) == len(event_spans)
        assert all(event.anchor is not None for event in timeline.events)
        assert all(event.normalized_value for event in timeline.events)
        assert all(
            event.dct_position in {"before", "overlap"} for event in timeline.events
        )
        anchor_sources.update(
            event.anchor.anchor_source
            for event in timeline.events
            if event.anchor is not None
        )
        assert compute_temporal_closure_consistency(timeline).violations == {}

        serialized = json.dumps(timeline.to_dict(), sort_keys=True)
        assert '"text":' not in serialized
        assert all(
            fixture.text[event.start : event.end] not in serialized
            for event in event_spans
        )

    gate = assert_temporal_tlink_gate()
    assert anchor_sources == {"timex", "dct_fallback"}
    assert gate.gate.awareness.f1 >= 0.75
    assert gate.gate.consistency.violations == {}


def test_order_events_rejects_conflicting_explicit_and_span_dct() -> None:
    text = "Document date: 2026-06-15. Cough worsened."
    spans = [
        {
            "label": "DCT",
            "start": text.index("2026-06-15"),
            "end": text.index("2026-06-15") + len("2026-06-15"),
            "normalized_value": "2026-06-15",
        },
        *_event_spans(text, "Cough"),
    ]

    with pytest.raises(
        ValueError,
        match="explicit document_creation_time conflicts with the DCT span",
    ):
        order_events(text, spans, document_creation_time="2026-06-16")


def _event_spans(text: str, *values: str) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    for index, value in enumerate(values):
        start = text.index(value)
        spans.append(
            {
                "text": value,
                "label": "EVENT",
                "start": start,
                "end": start + len(value),
                "score": 0.99 - index * 0.01,
            }
        )
    return spans
