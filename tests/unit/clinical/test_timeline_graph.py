"""Tests for the value-free evidence-linked timeline graph contract."""

from __future__ import annotations

from typing import Any, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from openmed.clinical.journey_contracts import canonical_digest, derived_opaque_id
from openmed.clinical.timeline_graph import (
    EvidenceLinkedTimelineGraph,
    TimelineGraphEdge,
    TimelineGraphNode,
)


def test_graph_rejects_correction_cycle() -> None:
    first = _node("a", 0)
    second = _node("b", 1)
    edges = (
        _edge(first, second, "a-b"),
        _edge(second, first, "b-a"),
    )

    with pytest.raises(ValueError, match="ordering cycle"):
        EvidenceLinkedTimelineGraph(nodes=(first, second), edges=edges)


def test_graph_serialization_is_value_free_and_deterministic() -> None:
    first = _node("a", 0)
    second = _node("b", 1)
    edge = _edge(first, second, "a-b")

    forward = EvidenceLinkedTimelineGraph(nodes=(first, second), edges=(edge,))
    reordered = EvidenceLinkedTimelineGraph(nodes=(second, first), edges=(edge,))

    assert forward.to_dict() == reordered.to_dict()
    assert "value" not in str(forward.to_dict()).casefold()
    assert forward.nodes[0].effective_time == {
        "precision": "day",
        "start": "2026-01-01",
    }


def test_graph_rejects_malformed_nested_values() -> None:
    with pytest.raises(TypeError, match="TimelineGraphNode"):
        EvidenceLinkedTimelineGraph(
            nodes=(cast(Any, "malformed"),),
            edges=(),
        )
    with pytest.raises(TypeError, match="TimelineGraphEdge"):
        EvidenceLinkedTimelineGraph(
            nodes=(_node("a", 0),),
            edges=(cast(Any, "malformed"),),
        )


@given(st.permutations(("a", "b", "c", "d")))
def test_graph_serialization_is_invariant_to_input_order(
    suffixes: list[str],
) -> None:
    positions = {suffix: index for index, suffix in enumerate(("a", "b", "c", "d"))}
    nodes = tuple(_node(suffix, positions[suffix]) for suffix in suffixes)

    graph = EvidenceLinkedTimelineGraph(nodes=nodes, edges=())

    assert [item.fact_id for item in graph.nodes] == [
        derived_opaque_id("fact", suffix) for suffix in ("a", "b", "c", "d")
    ]


def _node(suffix: str, position: int) -> TimelineGraphNode:
    return TimelineGraphNode(
        event_id=derived_opaque_id("journeyevent", suffix),
        fact_id=derived_opaque_id("fact", suffix),
        position=position,
        event_type="condition",
        journey_state="current",
        correction_state="none",
        effective_time={"precision": "day", "start": "2026-01-01"},
        evidence_ids=(derived_opaque_id("evidence", suffix),),
        derivation_hash=canonical_digest({"fixture": suffix}),
        assertion_status="affirmed",
        certainty="certain",
        experiencer="patient",
    )


def _edge(
    source: TimelineGraphNode,
    target: TimelineGraphNode,
    suffix: str,
) -> TimelineGraphEdge:
    return TimelineGraphEdge(
        edge_id=derived_opaque_id("timelineedge", suffix),
        source_event_id=source.event_id,
        target_event_id=target.event_id,
        relation_type="corrects",
        ordering_basis="fact_parent_lineage",
        evidence_ids=tuple(sorted((*source.evidence_ids, *target.evidence_ids))),
        derivation_hash=canonical_digest({"edge": suffix}),
    )
