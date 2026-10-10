"""Synthetic tests for the privacy-safe evidence-linked timeline graph."""

from __future__ import annotations

import json
from datetime import date

import pytest

from openmed.clinical import (
    TIMELINE_GRAPH_SCHEMA_VERSION,
    ClinicalAssertion,
    TimelineEvidence,
    TimelineGraphCycleError,
    TimelineGraphEvent,
    build_timeline_graph,
)
from openmed.clinical.timeline_graph import (
    TIMELINE_GRAPH_SCHEMA_VERSION as JOURNEY_TIMELINE_GRAPH_SCHEMA_VERSION,
)
from openmed.core.audit import hash_text


@pytest.mark.parametrize(
    "value", ["20260105", "2026-W02-1", "2026-005", "2026-01-05T10:00:00+0530"]
)
def test_timeline_normalized_values_reject_interpreter_dependent_grammars(value):
    with pytest.raises(ValueError) as error:
        TimelineEvidence(0, 1, normalized_value=value)
    assert value not in str(error.value)


@pytest.mark.parametrize("field", ["event_type", "relation"])
def test_freeform_graph_labels_are_opaque(field: str) -> None:
    private = "patient-jane-doe"
    event = TimelineGraphEvent(
        "one",
        private if field == "event_type" else "event",
        0,
        1,
        temporal_evidence=(
            TimelineEvidence(
                0, 1, relation=private if field == "relation" else "temporal_anchor"
            ),
        ),
    )
    serialized = build_timeline_graph([event]).to_json()
    assert private not in serialized
    assert hash_text(private) in serialized


@pytest.mark.parametrize("value", ["sha256:patient-jane-doe", "sha256:" + "a" * 63])
def test_graph_hashes_require_complete_digest(value: str) -> None:
    with pytest.raises(ValueError):
        TimelineEvidence(0, 1, text_hash=value)


def test_graph_experiencer_is_controlled() -> None:
    with pytest.raises(ValueError):
        TimelineGraphEvent(
            "one", "event", 0, 1, assertion={"experiencer": "patient-jane-doe"}
        )


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("where", ["event", "evidence", "link"])
def test_graph_rejects_out_of_document_spans(typed: bool, where: str) -> None:
    from openmed.clinical.clinical_timeline_evidence import TimelineTemporalLink

    evidence = TimelineEvidence(0, 50) if typed else {"start": 0, "end": 50}
    events = (
        [
            TimelineGraphEvent(
                "one",
                "event",
                0,
                50 if where == "event" else 1,
                temporal_evidence=(evidence,) if where == "evidence" else (),
            )
        ]
        if typed
        else [
            {
                "id": "one",
                "start": 0,
                "end": 50 if where == "event" else 1,
                "evidence": [evidence] if where == "evidence" else [],
            }
        ]
    )
    links = (
        [TimelineTemporalLink("one", "one", "overlap", (evidence,))]
        if typed and where == "link"
        else [
            {
                "source": "one",
                "target": "one",
                "relation": "overlap",
                "evidence": [evidence],
            }
        ]
        if where == "link"
        else []
    )
    with pytest.raises(ValueError, match="source document"):
        build_timeline_graph(events, links, document_text="synthetic")


def test_graph_orders_aware_times_by_instant() -> None:
    graph = build_timeline_graph(
        [
            TimelineGraphEvent(
                "later", "event", 0, 1, timestamp="2026-06-01T08:00:00-04:00"
            ),
            TimelineGraphEvent(
                "earlier", "event", 2, 3, timestamp="2026-06-01T10:00:00Z"
            ),
        ]
    )
    assert graph.ordered_event_ids == ("earlier", "later")


@pytest.mark.parametrize("value", ["0000-02", "2026/2027/2028"])
def test_graph_rejects_invalid_partial_dates_and_interval_arity(value: str) -> None:
    with pytest.raises(ValueError):
        TimelineEvidence(0, 1, normalized_value=value)


def test_graph_fixed_envelope_and_sanitized_errors() -> None:
    from openmed.clinical.clinical_timeline_evidence import TimelineGraph

    with pytest.raises(ValueError):
        TimelineGraph((), disclaimer="patient-jane-doe")
    with pytest.raises(ValueError):
        TimelineGraph((), schema_version=True)
    with pytest.raises(TypeError) as error:
        TimelineEvidence(0, 1, confidence="patient-jane-doe")
    assert error.value.__context__ is None
    assert error.value.__cause__ is None


def test_graph_iterator_failures_are_value_free() -> None:
    def records():
        yield {"id": "one", "start": 0, "end": 1}
        raise RuntimeError("patient-jane-doe")

    with pytest.raises(TypeError) as error:
        build_timeline_graph(records())
    assert error.value.__context__ is None
    assert "patient-jane-doe" not in str(error.value)


def test_clinical_and_journey_timeline_graph_contracts_remain_distinct() -> None:
    assert TIMELINE_GRAPH_SCHEMA_VERSION == 1
    assert JOURNEY_TIMELINE_GRAPH_SCHEMA_VERSION == "1.0.0"


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


def test_serialized_graph_hashes_caller_event_identifiers() -> None:
    private_id = "patient-jane-doe"
    graph = build_timeline_graph(
        [
            {"id": private_id, "type": "finding", "start": 0, "end": 4},
            {"id": "event-two", "type": "finding", "start": 5, "end": 9},
        ],
        links=[{"source": private_id, "target": "event-two", "relation": "before"}],
    )

    serialized = graph.to_json()
    assert private_id not in serialized
    assert hash_text(private_id) in serialized
    assert graph.event(private_id).event_id == private_id


def test_temporal_evidence_rejects_source_text_without_echoing_it() -> None:
    private_text = "patient-jane-doe"
    with pytest.raises(ValueError, match="normalized date or duration") as error:
        TimelineEvidence(start=0, end=5, normalized_value=private_text)

    assert private_text not in str(error.value)


@pytest.mark.parametrize(
    "value",
    ["2026-02-30", "2026-13", "2026-06-31T12:00", "P", "PT"],
)
def test_temporal_evidence_rejects_invalid_normalized_values(value: str) -> None:
    with pytest.raises(ValueError, match="normalized date or duration"):
        TimelineEvidence(start=0, end=5, normalized_value=value)


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
