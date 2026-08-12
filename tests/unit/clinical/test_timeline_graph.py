"""Synthetic tests for the privacy-safe evidence-linked timeline graph."""

from __future__ import annotations

import json
from datetime import date

import pytest

from openmed.clinical import (
    ClinicalAssertion,
    TimelineEvidence,
    TimelineGraphCycleError,
    TimelineGraphEvent,
    build_timeline_graph,
)
from openmed.core.audit import hash_text


def test_graph_preserves_typed_events_assertion_context_and_safe_evidence() -> None:
    source = "Synthetic procedure occurred on 2026-06-01. Synthetic finding followed."
    procedure_start = source.index("Synthetic procedure")
    procedure_end = procedure_start + len("Synthetic procedure")
    procedure_date_start = source.index("2026-06-01")
    finding_start = source.index("Synthetic finding")
    finding_end = finding_start + len("Synthetic finding")

    graph = build_timeline_graph(
        [
            {
                "id": "event-procedure",
                "type": "procedure",
                "start": procedure_start,
                "end": procedure_end,
                "text": source[procedure_start:procedure_end],
                "timestamp": date(2026, 6, 1),
                "assertion": {
                    "temporality": "recent",
                    "certainty": "certain",
                    "negation": "affirmed",
                },
                "temporal_evidence": [
                    {
                        "start": procedure_date_start,
                        "end": procedure_date_start + len("2026-06-01"),
                        "value": "2026-06-01",
                        "type": "DATE",
                    }
                ],
            },
            {
                "id": "event-finding",
                "type": "finding",
                "start": finding_start,
                "end": finding_end,
                "text": source[finding_start:finding_end],
                "timestamp": "2026-06-01",
                "assertion": ClinicalAssertion(
                    temporality="recent",
                    certainty="certain",
                    negation="affirmed",
                ),
            },
        ],
        links=[
            {
                "source_id": "event-procedure",
                "target_id": "event-finding",
                "relation": "before",
                "evidence_start": procedure_date_start,
                "evidence_end": procedure_date_start + len("2026-06-01"),
                "evidence_value": "2026-06-01",
            }
        ],
        document_text=source,
    )

    assert graph.ordered_event_ids == ("event-procedure", "event-finding")
    procedure = graph.event("event-procedure")
    assert procedure.event_type == "procedure"
    assert procedure.source_offsets == (procedure_start, procedure_end)
    assert procedure.assertion_context.temporality == "recent"
    assert procedure.temporal_evidence[0].normalized_value == "2026-06-01"
    assert procedure.temporal_evidence[0].text_hash == hash_text("2026-06-01")

    serialized = graph.to_json()
    assert source not in serialized
    assert "Synthetic procedure" not in serialized
    assert "Synthetic finding" not in serialized
    assert graph.to_dict()["cycle_free"] is True


def test_equal_timestamps_have_input_order_independent_tie_breaking() -> None:
    events = [
        TimelineGraphEvent(
            event_id="event-late-offset",
            event_type="observation",
            start=20,
            end=30,
            timestamp="2026-06-01",
        ),
        TimelineGraphEvent(
            event_id="event-early-offset",
            event_type="observation",
            start=2,
            end=12,
            timestamp="2026-06-01",
            temporal_evidence=(
                TimelineEvidence(
                    start=0,
                    end=10,
                    normalized_value="2026-06-01",
                    text_hash=hash_text("2026-06-01"),
                    timex_type="DATE",
                ),
            ),
        ),
    ]

    forward = build_timeline_graph(events)
    reversed_input = build_timeline_graph(reversed(events))

    assert forward.ordered_event_ids == (
        "event-early-offset",
        "event-late-offset",
    )
    assert forward.to_dict() == reversed_input.to_dict()


def test_before_after_cycle_is_rejected_without_echoing_event_values() -> None:
    events = [
        {"id": "event-a", "type": "procedure", "start": 0, "end": 1},
        {"id": "event-b", "type": "finding", "start": 2, "end": 3},
    ]

    with pytest.raises(TimelineGraphCycleError, match="cycle") as error:
        build_timeline_graph(
            events,
            temporal_links=[
                {"source": "event-a", "target": "event-b", "relation": "before"},
                {"source": "event-b", "target": "event-a", "relation": "before"},
            ],
        )

    assert "event-a" not in str(error.value)
    assert "event-b" not in str(error.value)


def test_graph_output_is_json_ready_and_contains_only_explicit_temporal_links() -> None:
    graph = build_timeline_graph(
        [
            {
                "event_id": "event-one",
                "event_type": "event",
                "source_offsets": [4, 9],
                "event_time": "2026-06-01",
            },
            {
                "event_id": "event-two",
                "event_type": "event",
                "source_offsets": [14, 19],
                "event_time": "2026-06-02",
            },
        ],
        temporal_links=[
            {
                "source": "event-one",
                "target": "event-two",
                "relation_type": "AFTER",
            }
        ],
    )

    assert graph.ordered_event_ids == ("event-two", "event-one")
    payload = graph.to_dict()
    assert json.loads(graph.to_json()) == payload
    assert payload["temporal_links"][0]["relation"] == "after"
    assert payload["events"][0]["source_offsets"] == [14, 19]
