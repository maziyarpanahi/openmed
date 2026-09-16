"""Tests for joint span-pair entity and relation decoding."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from openmed.clinical import (
    JointHeadConfig,
    JointSpanPairHead,
    SpanPairCandidate,
    decode_joint_span_pairs,
    enumerate_joint_span_candidates,
    sample_negative_span_pairs,
)
from openmed.core.decoding import SpanEdge, SpanNode

FIXTURE_PATH = (
    Path(__file__).parents[3]
    / "openmed"
    / "eval"
    / "golden"
    / "fixtures"
    / "joint_entity_relation.jsonl"
)
CONFIG = JointHeadConfig(
    max_span_width=2,
    max_pair_token_distance=8,
    entity_floor=0.70,
    relation_floor=0.50,
)
HIDDEN_SIZE = 5
SPAN_FEATURE_SIZE = HIDDEN_SIZE * 3 + 1
RELATION_FEATURE_SIZE = SPAN_FEATURE_SIZE * 2 + HIDDEN_SIZE


def _weights(size: int, assignments: dict[int, float]) -> list[float]:
    weights = [0.0] * size
    for index, value in assignments.items():
        weights[index] = value
    return weights


def _semantic_assignments(
    dimension: int,
    *,
    offset: int = 0,
    weight: float = 1.0,
) -> dict[int, float]:
    return {
        offset + dimension: weight,
        offset + HIDDEN_SIZE + dimension: weight,
        offset + HIDDEN_SIZE * 2 + dimension: weight,
    }


def _head() -> JointSpanPairHead:
    entity_weights = {
        "condition": _weights(
            SPAN_FEATURE_SIZE,
            _semantic_assignments(2),
        ),
        "dose": _weights(
            SPAN_FEATURE_SIZE,
            _semantic_assignments(1),
        ),
        "medication": _weights(
            SPAN_FEATURE_SIZE,
            _semantic_assignments(0),
        ),
    }

    has_dose = {
        **_semantic_assignments(0, weight=0.5),
        **_semantic_assignments(1, offset=SPAN_FEATURE_SIZE, weight=0.5),
        **_semantic_assignments(4, offset=SPAN_FEATURE_SIZE, weight=0.5),
    }
    treats = {
        **_semantic_assignments(0, weight=0.5),
        **_semantic_assignments(2, offset=SPAN_FEATURE_SIZE, weight=0.5),
        SPAN_FEATURE_SIZE * 2 + 3: 1.0,
    }
    return JointSpanPairHead(
        entity_weights=entity_weights,
        relation_weights={
            "has_dose": _weights(RELATION_FEATURE_SIZE, has_dose),
            "treats": _weights(RELATION_FEATURE_SIZE, treats),
        },
        entity_bias={label: -2.0 for label in entity_weights},
        relation_bias={"has_dose": -2.0, "treats": -3.0},
    )


def _fixtures() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _decode(row: dict[str, Any]):
    tokens = row["tokens"]
    return decode_joint_span_pairs(
        [token["state"] for token in tokens],
        [(token["start"], token["end"]) for token in tokens],
        head=_head(),
        config=CONFIG,
    )


def _joint_counts(row: dict[str, Any]) -> tuple[int, int, int]:
    output = _decode(row)
    predicted_entities = {
        (node.start, node.end, node.label) for node in output.graph.nodes
    }
    gold_entities = {
        (entity["start"], entity["end"], entity["label"]) for entity in row["entities"]
    }
    nodes_by_id = {node.node_id: node for node in output.graph.nodes}
    predicted_relations = {
        (
            edge.label,
            nodes_by_id[edge.head].start,
            nodes_by_id[edge.head].end,
            nodes_by_id[edge.tail].start,
            nodes_by_id[edge.tail].end,
        )
        for edge in output.graph.edges
    }
    entities_by_id = {entity["id"]: entity for entity in row["entities"]}
    gold_relations = {
        (
            relation["type"],
            entities_by_id[relation["head"]]["start"],
            entities_by_id[relation["head"]]["end"],
            entities_by_id[relation["tail"]]["start"],
            entities_by_id[relation["tail"]]["end"],
        )
        for relation in row["relations"]
    }
    predicted = predicted_entities | predicted_relations
    gold = gold_entities | gold_relations
    return len(predicted & gold), len(predicted - gold), len(gold - predicted)


def test_joint_head_exceeds_committed_joint_micro_f1_floor() -> None:
    rows = _fixtures()
    counts = [_joint_counts(row) for row in rows]
    true_positives = sum(count[0] for count in counts)
    false_positives = sum(count[1] for count in counts)
    false_negatives = sum(count[2] for count in counts)
    micro_f1 = (
        2 * true_positives / (2 * true_positives + false_positives + false_negatives)
    )

    assert micro_f1 >= 0.70
    assert all(
        row["metadata"] == {"synthetic": True, "contains_real_phi": False}
        for row in rows
    )
    boundary_case = next(row for row in rows if "boundary" in row["id"])
    boundary_output = _decode(boundary_case)
    assert not any(
        node.start == 0 and node.end == 13 for node in boundary_output.graph.nodes
    )


def test_low_confidence_over_generated_entity_suppresses_relation() -> None:
    row = next(row for row in _fixtures() if "over-generated" in row["id"])
    output = _decode(row)
    trap = row["traps"][0]
    head = next(
        score
        for score in output.entity_scores
        if score.start == trap["head_start"] and score.end == trap["head_end"]
    )
    tail = next(
        score
        for score in output.entity_scores
        if score.start == trap["start"] and score.end == trap["end"]
    )
    pair = next(
        score
        for score in output.pair_scores
        if score.head == head.node_id
        and score.tail == tail.node_id
        and score.relation_label == trap["relation_type"]
    )

    assert pair.raw_relation_score >= trap["minimum_raw_relation_score"]
    assert pair.entity_consistency_score < CONFIG.entity_floor
    assert pair.suppression_reason == "entity_floor"
    assert pair.emitted is False
    assert output.graph.edges == ()


def test_joint_output_round_trips_span_graph_schema_without_loss() -> None:
    row = next(row for row in _fixtures() if row["id"] == "joint-treatment-relation")
    output = _decode(row)

    assert output.nodes == output.graph.nodes
    assert output.candidate_edges == output.graph.edges
    assert [node.to_dict() for node in output.nodes] == [
        node.to_dict() for node in output.graph.nodes
    ]
    assert [edge.to_dict() for edge in output.candidate_edges] == [
        edge.to_dict() for edge in output.graph.edges
    ]
    assert output.graph.explain().decisions[0].status == "kept"
    assert json.loads(json.dumps(output.to_dict()))["schema_version"] == 1


def test_negative_pair_sampling_and_distractor_false_positive_rate() -> None:
    spans = enumerate_joint_span_candidates(
        ((0, 2), (3, 5), (6, 8), (9, 11)),
        max_span_width=1,
    )
    positive = SpanPairCandidate(spans[0], spans[1], "has_dose")
    negatives = sample_negative_span_pairs(
        spans,
        (positive,),
        max_negatives=5,
        seed=758,
    )

    assert negatives == sample_negative_span_pairs(
        spans,
        (positive,),
        max_negatives=5,
        seed=758,
    )
    assert len(negatives) == 5
    assert all(pair.is_negative for pair in negatives)
    assert positive.stable_key() not in {pair.stable_key() for pair in negatives}

    distractor_rows = [
        row
        for row in _fixtures()
        if any(trap["kind"] == "negative_pair" for trap in row["traps"])
    ]
    false_positive_count = sum(len(_decode(row).graph.edges) for row in distractor_rows)
    assert false_positive_count / len(distractor_rows) == 0.0


def test_deterministic_fallback_matches_learned_output_schema() -> None:
    source_nodes = (
        SpanNode(
            "med",
            0,
            7,
            "medication",
            0.9,
            text_hash="sha256:synthetic-med",
            metadata={"source": "rule"},
        ),
        SpanNode(
            "dose",
            8,
            13,
            "dose",
            0.8,
            text_hash="sha256:synthetic-dose",
            metadata={"source": "rule"},
        ),
    )
    source_edge = SpanEdge(
        "med",
        "dose",
        "has_dose",
        0.75,
        metadata={"source": "rule"},
    )
    first = decode_joint_span_pairs(
        (),
        (),
        fallback_nodes=source_nodes,
        fallback_edges=(source_edge,),
        config=CONFIG,
    )
    second = decode_joint_span_pairs(
        (),
        (),
        fallback_nodes=source_nodes,
        fallback_edges=(source_edge,),
        config=CONFIG,
    )
    learned = _decode(_fixtures()[0])

    assert first == second
    assert first.mode == "fallback"
    assert first.nodes == source_nodes
    assert first.graph.nodes == source_nodes
    assert first.candidate_edges[0].metadata["source"] == "rule"
    assert first.candidate_edges[0].score == pytest.approx(0.60)
    assert set(first.to_dict()) == set(learned.to_dict())
    assert set(first.nodes[0].to_dict()) == set(learned.nodes[0].to_dict())
    assert set(first.candidate_edges[0].to_dict()) == set(
        learned.candidate_edges[0].to_dict()
    )


def test_joint_head_module_has_no_array_framework_dependency() -> None:
    module_path = (
        Path(__file__).parents[3]
        / "openmed"
        / "clinical"
        / "relations"
        / "joint_head.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert imported_roots.isdisjoint({"jax", "mlx", "numpy", "tensorflow", "torch"})
