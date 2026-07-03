"""Tests for pure-Python span-relation graph decoding."""

from __future__ import annotations

import ast
import random
from collections import Counter
from pathlib import Path

from openmed.core.decoding import (
    EdgeCardinality,
    SpanEdge,
    SpanGraph,
    SpanGraphConstraints,
    SpanNode,
    decode_span_graph,
    edge_f1,
)
from openmed.core.explain import explain_span_graph


def _node(node_id: str, start: int, label: str) -> SpanNode:
    return SpanNode(node_id=node_id, start=start, end=start + 1, label=label)


def test_decode_span_graph_finds_global_matching_optimum() -> None:
    nodes = [
        _node("med-a", 0, "medication"),
        _node("med-b", 10, "medication"),
        _node("attr-x", 20, "attribute"),
        _node("attr-y", 30, "attribute"),
        _node("attr-z", 40, "attribute"),
    ]
    edges = [
        SpanEdge("med-a", "attr-x", "links", 10.0),
        SpanEdge("med-a", "attr-y", "links", 9.0),
        SpanEdge("med-b", "attr-x", "links", 9.0),
        SpanEdge("med-b", "attr-z", "links", 1.0),
    ]
    constraints = SpanGraphConstraints(
        type_compatibility={"links": (("medication", "attribute"),)},
        cardinality={"links": EdgeCardinality.one_to_one()},
    )

    graph = decode_span_graph(nodes, edges, constraints=constraints)

    assert graph.score == 18.0
    assert set(graph.edge_keys()) == {
        ("links", "med-a", "attr-y"),
        ("links", "med-b", "attr-x"),
    }


def test_decode_span_graph_is_stable_under_input_reordering() -> None:
    nodes = [
        _node("lab", 0, "lab_name"),
        _node("value-a", 10, "lab_value"),
        _node("value-b", 20, "lab_value"),
        _node("problem-a", 30, "problem"),
        _node("problem-b", 40, "problem"),
    ]
    edges = [
        SpanEdge("lab", "value-a", "has_value", 0.9),
        SpanEdge("lab", "value-b", "has_value", 0.4),
        SpanEdge("problem-a", "problem-b", "contains", 0.8),
    ]
    constraints = SpanGraphConstraints(
        type_compatibility={
            "has_value": (("lab_name", "lab_value"),),
            "contains": (("problem", "problem"),),
        },
        cardinality={"has_value": EdgeCardinality.one_to_one()},
        acyclic_edge_labels=("contains",),
    )
    expected = decode_span_graph(nodes, edges, constraints=constraints).edge_keys()

    for seed in range(50):
        rng = random.Random(seed)
        shuffled_nodes = list(nodes)
        shuffled_edges = list(edges)
        rng.shuffle(shuffled_nodes)
        rng.shuffle(shuffled_edges)

        graph = decode_span_graph(
            shuffled_nodes,
            shuffled_edges,
            constraints=constraints,
        )

        assert graph.edge_keys() == expected


def test_synthetic_graph_gate_prevents_violations_and_meets_edge_f1() -> None:
    constraints = SpanGraphConstraints(
        type_compatibility={
            "has_value": (("lab_name", "lab_value"),),
            "contains": (("problem", "problem"),),
        },
        cardinality={"has_value": EdgeCardinality.one_to_one()},
        acyclic_edge_labels=("contains",),
    )
    predicted_edges: set[tuple[str, str, str]] = set()
    gold_edges: set[tuple[str, str, str]] = set()

    for case_id in range(200):
        graph, gold = _synthetic_graph_case(case_id, constraints)
        _assert_graph_constraints_hold(graph, constraints)
        predicted_edges.update(graph.edge_keys())
        gold_edges.update(gold)

    assert edge_f1(predicted_edges, gold_edges) >= 0.85


def test_explain_span_graph_names_constraints_for_pruned_edges() -> None:
    nodes = [
        _node("lab", 0, "lab_name"),
        _node("value-a", 10, "lab_value"),
        _node("value-b", 20, "lab_value"),
        _node("problem-a", 30, "problem"),
        _node("problem-b", 40, "problem"),
    ]
    edges = [
        SpanEdge("lab", "value-a", "has_value", 1.0),
        SpanEdge("lab", "value-b", "has_value", 0.9),
        SpanEdge("problem-a", "value-a", "has_value", 0.8),
        SpanEdge("problem-a", "problem-b", "contains", 1.0),
        SpanEdge("problem-b", "problem-a", "contains", 0.9),
    ]
    constraints = SpanGraphConstraints(
        type_compatibility={
            "has_value": (("lab_name", "lab_value"),),
            "contains": (("problem", "problem"),),
        },
        cardinality={"has_value": EdgeCardinality.one_to_one()},
        acyclic_edge_labels=("contains",),
    )

    graph = decode_span_graph(nodes, edges, constraints=constraints)
    report = explain_span_graph(graph, fmt="dict")
    pruned_constraints = {
        decision["constraint"]
        for decision in report["edges"]
        if decision["status"] == "pruned"
    }

    assert "type_compatibility:has_value" in pruned_constraints
    assert "cardinality:has_value:max_outgoing_per_head" in pruned_constraints
    assert "acyclicity:contains" in pruned_constraints


def test_graph_module_import_guard_has_no_array_frameworks() -> None:
    graph_path = (
        Path(__file__).parents[3] / "openmed" / "core" / "decoding" / "graph.py"
    )
    tree = ast.parse(graph_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert imported_roots.isdisjoint({"jax", "mlx", "numpy", "tensorflow", "torch"})


def _synthetic_graph_case(
    case_id: int,
    constraints: SpanGraphConstraints,
) -> tuple[SpanGraph, set[tuple[str, str, str]]]:
    prefix = f"case-{case_id}"
    offset = case_id * 100
    nodes = [
        _node(f"{prefix}-lab", offset, "lab_name"),
        _node(f"{prefix}-value-a", offset + 10, "lab_value"),
        _node(f"{prefix}-value-b", offset + 20, "lab_value"),
        _node(f"{prefix}-problem-a", offset + 30, "problem"),
        _node(f"{prefix}-problem-b", offset + 40, "problem"),
        _node(f"{prefix}-problem-c", offset + 50, "problem"),
    ]
    gold = {
        ("has_value", f"{prefix}-lab", f"{prefix}-value-a"),
        ("contains", f"{prefix}-problem-a", f"{prefix}-problem-b"),
        ("contains", f"{prefix}-problem-b", f"{prefix}-problem-c"),
    }
    edges = [
        SpanEdge(f"{prefix}-lab", f"{prefix}-value-a", "has_value", 0.95),
        SpanEdge(f"{prefix}-lab", f"{prefix}-value-b", "has_value", 0.25),
        SpanEdge(f"{prefix}-problem-a", f"{prefix}-value-a", "has_value", 0.99),
        SpanEdge(f"{prefix}-problem-a", f"{prefix}-problem-b", "contains", 0.90),
        SpanEdge(f"{prefix}-problem-b", f"{prefix}-problem-c", "contains", 0.85),
        SpanEdge(f"{prefix}-problem-c", f"{prefix}-problem-a", "contains", 0.10),
    ]
    return decode_span_graph(nodes, edges, constraints=constraints), gold


def _assert_graph_constraints_hold(
    graph: SpanGraph,
    constraints: SpanGraphConstraints,
) -> None:
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    counts_out: Counter[tuple[str, str]] = Counter()
    counts_in: Counter[tuple[str, str]] = Counter()
    adjacency: dict[str, set[str]] = {}

    for edge in graph.edges:
        assert constraints.local_violation(edge, nodes_by_id) is None
        counts_out[(edge.label, edge.head)] += 1
        counts_in[(edge.label, edge.tail)] += 1
        if edge.label in constraints.acyclic_edge_labels:
            adjacency.setdefault(edge.head, set()).add(edge.tail)

    for label, rule in constraints.cardinality.items():
        if rule.max_outgoing_per_head is not None:
            assert all(
                count <= rule.max_outgoing_per_head
                for (edge_label, _), count in counts_out.items()
                if edge_label == label
            )
        if rule.max_incoming_per_tail is not None:
            assert all(
                count <= rule.max_incoming_per_tail
                for (edge_label, _), count in counts_in.items()
                if edge_label == label
            )

    for node_id in adjacency:
        assert not _can_reach(node_id, node_id, adjacency, set())


def _can_reach(
    source: str,
    target: str,
    adjacency: dict[str, set[str]],
    seen: set[str],
) -> bool:
    for next_node in adjacency.get(source, set()):
        if next_node == target:
            return True
        if next_node in seen:
            continue
        seen.add(next_node)
        if _can_reach(next_node, target, adjacency, seen):
            return True
    return False
