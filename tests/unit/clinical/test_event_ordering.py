"""Synthetic tests for the privacy-safe event-ordering timeline API."""

from __future__ import annotations

import json

from openmed.clinical import Timeline, order_events


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
    assert timeline.kept_edges
    assert not timeline.pruned_edges


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
