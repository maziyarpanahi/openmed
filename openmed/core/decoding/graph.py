"""Pure-Python span-relation graph decoding utilities.

The decoder consumes already-located spans and candidate typed, directed
relations between them. It maximizes retained edge score subject to declarative
constraints, then emits a deterministic graph plus an explanation for kept and
pruned candidates. The module intentionally depends only on the standard
library so it can be shared by MLX, PyTorch, and deterministic clinical
extractors without importing an array framework.

Tie-break order is stable and documented: maximize total retained edge score,
then retain the larger number of edges when scores tie, then choose the
lexicographically smallest canonical edge-key tuple. Canonical keys are built
from edge label, head span key, tail span key, and node ids.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

GRAPH_EXPLAIN_SCHEMA_VERSION = 1

EdgeDecisionStatus = Literal["kept", "pruned"]
ConstraintViolation = tuple[str, str]

_EPSILON = 1e-12


@dataclass(frozen=True)
class SpanNode:
    """One decoded span node available to relation decoding.

    Args:
        node_id: Stable node identifier used by candidate edges.
        start: Inclusive character offset.
        end: Exclusive character offset.
        label: Span type or clinical role.
        score: Optional span confidence used for provenance.
        text_hash: Optional safe hash of the span surface text.
        metadata: Additional JSON-compatible provenance.
    """

    node_id: str
    start: int
    end: int
    label: str
    score: float | None = None
    text_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        if not self.label:
            raise ValueError("label must be non-empty")
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("start and end must be integers")
        if self.start < 0 or self.end < self.start:
            raise ValueError("node offsets must satisfy 0 <= start <= end")
        if self.score is not None and not math.isfinite(float(self.score)):
            raise ValueError("score must be finite when provided")
        object.__setattr__(self, "node_id", str(self.node_id))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "metadata", _plain_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible node representation."""

        return {
            "id": self.node_id,
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "score": self.score,
            "text_hash": self.text_hash,
            "metadata": copy.deepcopy(dict(self.metadata)),
        }


@dataclass(frozen=True)
class SpanEdge:
    """One typed, directed candidate or decoded graph edge."""

    head: str
    tail: str
    label: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.head or not self.tail:
            raise ValueError("edge head and tail must be non-empty node ids")
        if not self.label:
            raise ValueError("edge label must be non-empty")
        if not math.isfinite(float(self.score)):
            raise ValueError("edge score must be finite")
        object.__setattr__(self, "head", str(self.head))
        object.__setattr__(self, "tail", str(self.tail))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "metadata", _plain_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible edge representation."""

        return {
            "head": self.head,
            "tail": self.tail,
            "label": self.label,
            "score": self.score,
            "metadata": copy.deepcopy(dict(self.metadata)),
        }


@dataclass(frozen=True)
class EdgeCardinality:
    """Per-label cardinality limits for decoded graph edges."""

    max_outgoing_per_head: int | None = None
    max_incoming_per_tail: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("max_outgoing_per_head", "max_incoming_per_tail"):
            value = getattr(self, field_name)
            if value is not None and value < 1:
                raise ValueError(f"{field_name} must be positive when provided")

    @classmethod
    def one_to_one(cls) -> "EdgeCardinality":
        """Return a one-to-one cardinality rule for one edge label."""

        return cls(max_outgoing_per_head=1, max_incoming_per_tail=1)

    @classmethod
    def one_to_many(cls) -> "EdgeCardinality":
        """Return a one-to-many rule for one edge label."""

        return cls(max_incoming_per_tail=1)

    @classmethod
    def many_to_one(cls) -> "EdgeCardinality":
        """Return a many-to-one rule for one edge label."""

        return cls(max_outgoing_per_head=1)

    @classmethod
    def many_to_many(cls) -> "EdgeCardinality":
        """Return an unconstrained cardinality rule for one edge label."""

        return cls()


@dataclass(frozen=True)
class SpanGraphConstraints:
    """Declarative edge-admissibility constraints.

    Args:
        allowed_edge_labels: Optional closed set of allowed edge labels.
        type_compatibility: Mapping from edge label to allowed
            ``(head_node_label, tail_node_label)`` pairs.
        cardinality: Mapping from edge label to incoming/outgoing limits.
        acyclic_edge_labels: Edge labels that must remain acyclic when decoded.
        allow_self_edges: Whether an edge may point from a node to itself.
    """

    allowed_edge_labels: Iterable[str] | None = None
    type_compatibility: Mapping[str, Iterable[tuple[str, str]]] = field(
        default_factory=dict
    )
    cardinality: Mapping[str, EdgeCardinality] = field(default_factory=dict)
    acyclic_edge_labels: Iterable[str] = field(default_factory=tuple)
    allow_self_edges: bool = False

    def __post_init__(self) -> None:
        allowed = (
            None
            if self.allowed_edge_labels is None
            else frozenset(str(label) for label in self.allowed_edge_labels)
        )
        compatibility = {
            str(label): frozenset((str(head), str(tail)) for head, tail in pairs)
            for label, pairs in self.type_compatibility.items()
        }
        cardinality = {str(label): rule for label, rule in self.cardinality.items()}
        acyclic = frozenset(str(label) for label in self.acyclic_edge_labels)
        object.__setattr__(self, "allowed_edge_labels", allowed)
        object.__setattr__(self, "type_compatibility", compatibility)
        object.__setattr__(self, "cardinality", cardinality)
        object.__setattr__(self, "acyclic_edge_labels", acyclic)

    def has_dynamic_constraints(self) -> bool:
        """Return whether constraints require subset-level search."""

        return bool(self.cardinality or self.acyclic_edge_labels)

    def local_violation(
        self,
        edge: SpanEdge,
        nodes_by_id: Mapping[str, SpanNode],
    ) -> ConstraintViolation | None:
        """Return the first single-edge constraint violation, if any."""

        if edge.head not in nodes_by_id:
            return (
                "node_exists:head",
                f"head node {edge.head!r} is not present in the graph",
            )
        if edge.tail not in nodes_by_id:
            return (
                "node_exists:tail",
                f"tail node {edge.tail!r} is not present in the graph",
            )
        if not self.allow_self_edges and edge.head == edge.tail:
            return ("self_edge", "self edges are disabled")

        allowed_labels = self.allowed_edge_labels
        if allowed_labels is not None and edge.label not in allowed_labels:
            return (
                f"edge_label:{edge.label}",
                f"edge label {edge.label!r} is not in the allowed label set",
            )

        allowed_pairs = self.type_compatibility.get(edge.label)
        if allowed_pairs is not None:
            pair = (nodes_by_id[edge.head].label, nodes_by_id[edge.tail].label)
            if pair not in allowed_pairs:
                return (
                    f"type_compatibility:{edge.label}",
                    (
                        f"edge label {edge.label!r} does not allow "
                        f"{pair[0]!r} -> {pair[1]!r}"
                    ),
                )
        return None


@dataclass(frozen=True)
class EdgeDecisionTrace:
    """Explanation for one candidate edge decision."""

    edge: SpanEdge
    status: EdgeDecisionStatus
    reason: str
    constraint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible decision representation."""

        return {
            "edge": self.edge.to_dict(),
            "status": self.status,
            "reason": self.reason,
            "constraint": self.constraint,
        }


@dataclass(frozen=True)
class GraphExplainReport:
    """Safe explanation report for a decoded span graph."""

    decisions: tuple[EdgeDecisionTrace, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = GRAPH_EXPLAIN_SCHEMA_VERSION

    def render(self, fmt: Literal["text", "dict"] = "text") -> str | dict[str, Any]:
        """Render this report as compact text or a JSON-compatible dictionary."""

        if fmt == "dict":
            return self.to_dict()
        if fmt != "text":
            raise ValueError("fmt must be 'text' or 'dict'")

        kept = sum(1 for decision in self.decisions if decision.status == "kept")
        pruned = len(self.decisions) - kept
        lines = [f"SpanGraph explain trace (kept={kept}, pruned={pruned})"]
        for decision in self.decisions:
            edge = decision.edge
            constraint = decision.constraint or "none"
            lines.append(
                (
                    f"{decision.status} {edge.label} {edge.head}->{edge.tail} "
                    f"score={edge.score} constraint={constraint} "
                    f"reason={decision.reason}"
                )
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""

        return {
            "schema_version": self.schema_version,
            "metadata": _json_safe(self.metadata),
            "edges": [decision.to_dict() for decision in self.decisions],
        }


@dataclass(frozen=True)
class SpanGraph:
    """Decoded span-relation graph with deterministic node and edge order."""

    nodes: tuple[SpanNode, ...]
    edges: tuple[SpanEdge, ...]
    decisions: tuple[EdgeDecisionTrace, ...] = ()
    score: float = 0.0

    def explain(self) -> GraphExplainReport:
        """Return kept/pruned edge decisions for this graph."""

        return GraphExplainReport(
            decisions=self.decisions,
            metadata={
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "score": self.score,
            },
        )

    def edge_keys(self) -> tuple[tuple[str, str, str], ...]:
        """Return compact edge keys useful for tests and metrics."""

        return tuple((edge.label, edge.head, edge.tail) for edge in self.edges)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible graph representation."""

        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "score": self.score,
            "explain": self.explain().to_dict(),
        }


def decode_span_graph(
    nodes: Sequence[SpanNode],
    candidate_edges: Sequence[SpanEdge],
    *,
    constraints: SpanGraphConstraints | None = None,
    min_edge_score: float | None = None,
) -> SpanGraph:
    """Decode a maximum-score span-relation graph.

    Args:
        nodes: Span nodes available to the graph.
        candidate_edges: Candidate typed, directed edges between nodes.
        constraints: Optional declarative constraints.
        min_edge_score: Optional local score floor. Edges below the floor are
            explained as pruned by ``score_floor`` before global decoding.

    Returns:
        A deterministic :class:`SpanGraph` containing retained edges and
        explanations for every candidate edge.
    """

    resolved_constraints = constraints or SpanGraphConstraints()
    node_tuple = tuple(nodes)
    nodes_by_id = _nodes_by_id(node_tuple)
    ordered_nodes = tuple(sorted(node_tuple, key=_node_sort_key))
    duplicate_decisions, deduped_edges = _dedupe_candidate_edges(
        candidate_edges,
        nodes_by_id,
    )

    local_decisions: list[EdgeDecisionTrace] = list(duplicate_decisions)
    valid_edges: list[SpanEdge] = []
    for edge in deduped_edges:
        violation = resolved_constraints.local_violation(edge, nodes_by_id)
        if violation is None and min_edge_score is not None:
            if edge.score < min_edge_score:
                violation = (
                    "score_floor",
                    (
                        f"edge score {edge.score} is below "
                        f"min_edge_score {min_edge_score}"
                    ),
                )
        if violation is not None:
            constraint, reason = violation
            local_decisions.append(
                EdgeDecisionTrace(
                    edge=edge,
                    status="pruned",
                    reason=reason,
                    constraint=constraint,
                )
            )
            continue
        valid_edges.append(edge)

    if resolved_constraints.has_dynamic_constraints():
        selected_edges = _decode_with_dynamic_constraints(
            valid_edges,
            resolved_constraints,
            nodes_by_id,
        )
    else:
        selected_edges = tuple(edge for edge in valid_edges if edge.score >= 0.0)

    ordered_edges = tuple(
        sorted(selected_edges, key=lambda edge: _edge_canonical_key(edge, nodes_by_id))
    )
    decisions = _build_decisions(
        kept_edges=ordered_edges,
        valid_edges=valid_edges,
        local_decisions=local_decisions,
        constraints=resolved_constraints,
        nodes_by_id=nodes_by_id,
    )
    return SpanGraph(
        nodes=ordered_nodes,
        edges=ordered_edges,
        decisions=decisions,
        score=sum(edge.score for edge in ordered_edges),
    )


def edge_f1(
    predicted_edges: Iterable[tuple[str, str, str]],
    gold_edges: Iterable[tuple[str, str, str]],
) -> float:
    """Return exact-match F1 for compact ``(label, head, tail)`` edge keys."""

    predicted = set(predicted_edges)
    gold = set(gold_edges)
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted)
    recall = true_positive / len(gold)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _nodes_by_id(nodes: Sequence[SpanNode]) -> dict[str, SpanNode]:
    nodes_by_id: dict[str, SpanNode] = {}
    for node in nodes:
        if node.node_id in nodes_by_id:
            raise ValueError(f"duplicate SpanNode id {node.node_id!r}")
        nodes_by_id[node.node_id] = node
    return nodes_by_id


def _dedupe_candidate_edges(
    edges: Sequence[SpanEdge],
    nodes_by_id: Mapping[str, SpanNode],
) -> tuple[tuple[EdgeDecisionTrace, ...], tuple[SpanEdge, ...]]:
    decisions: list[EdgeDecisionTrace] = []
    deduped: dict[tuple[str, str, str], SpanEdge] = {}
    for edge in edges:
        key = (edge.head, edge.tail, edge.label)
        current = deduped.get(key)
        if current is None:
            deduped[key] = edge
            continue
        if _duplicate_edge_is_better(edge, current, nodes_by_id):
            decisions.append(
                EdgeDecisionTrace(
                    edge=current,
                    status="pruned",
                    reason="a higher-priority duplicate candidate was kept",
                    constraint="duplicate_edge",
                )
            )
            deduped[key] = edge
        else:
            decisions.append(
                EdgeDecisionTrace(
                    edge=edge,
                    status="pruned",
                    reason="a higher-priority duplicate candidate was kept",
                    constraint="duplicate_edge",
                )
            )
    return (
        tuple(
            sorted(
                decisions,
                key=lambda decision: _decision_sort_key(decision, nodes_by_id),
            )
        ),
        tuple(
            sorted(
                deduped.values(), key=lambda edge: _edge_search_key(edge, nodes_by_id)
            )
        ),
    )


def _duplicate_edge_is_better(
    candidate: SpanEdge,
    current: SpanEdge,
    nodes_by_id: Mapping[str, SpanNode],
) -> bool:
    if candidate.score != current.score:
        return candidate.score > current.score
    return _metadata_key(candidate) < _metadata_key(current) or (
        _metadata_key(candidate) == _metadata_key(current)
        and _edge_canonical_key(candidate, nodes_by_id)
        < _edge_canonical_key(current, nodes_by_id)
    )


def _decode_with_dynamic_constraints(
    edges: Sequence[SpanEdge],
    constraints: SpanGraphConstraints,
    nodes_by_id: Mapping[str, SpanNode],
) -> tuple[SpanEdge, ...]:
    ordered_edges = tuple(
        sorted(edges, key=lambda edge: _edge_search_key(edge, nodes_by_id))
    )
    positive_suffix = [0.0] * (len(ordered_edges) + 1)
    for index in range(len(ordered_edges) - 1, -1, -1):
        positive_suffix[index] = positive_suffix[index + 1] + max(
            ordered_edges[index].score,
            0.0,
        )

    best_score = -math.inf
    best_edges: tuple[SpanEdge, ...] = ()

    def visit(index: int, selected: list[SpanEdge], score: float) -> None:
        nonlocal best_score, best_edges
        if score + positive_suffix[index] < best_score - _EPSILON:
            return
        if index == len(ordered_edges):
            selected_tuple = tuple(selected)
            if _solution_is_better(
                score,
                selected_tuple,
                best_score,
                best_edges,
                nodes_by_id,
            ):
                best_score = score
                best_edges = selected_tuple
            return

        edge = ordered_edges[index]
        if _dynamic_violation(edge, selected, constraints) is None:
            selected.append(edge)
            visit(index + 1, selected, score + edge.score)
            selected.pop()
        visit(index + 1, selected, score)

    visit(0, [], 0.0)
    return best_edges


def _solution_is_better(
    score: float,
    edges: Sequence[SpanEdge],
    best_score: float,
    best_edges: Sequence[SpanEdge],
    nodes_by_id: Mapping[str, SpanNode],
) -> bool:
    if score > best_score + _EPSILON:
        return True
    if score < best_score - _EPSILON:
        return False
    if len(edges) != len(best_edges):
        return len(edges) > len(best_edges)
    return _solution_key(edges, nodes_by_id) < _solution_key(best_edges, nodes_by_id)


def _solution_key(
    edges: Sequence[SpanEdge],
    nodes_by_id: Mapping[str, SpanNode],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        _edge_canonical_key(edge, nodes_by_id)
        for edge in sorted(
            edges, key=lambda edge: _edge_canonical_key(edge, nodes_by_id)
        )
    )


def _build_decisions(
    *,
    kept_edges: Sequence[SpanEdge],
    valid_edges: Sequence[SpanEdge],
    local_decisions: Sequence[EdgeDecisionTrace],
    constraints: SpanGraphConstraints,
    nodes_by_id: Mapping[str, SpanNode],
) -> tuple[EdgeDecisionTrace, ...]:
    kept_keys = {(edge.head, edge.tail, edge.label) for edge in kept_edges}
    selected = tuple(kept_edges)
    decisions: list[EdgeDecisionTrace] = list(local_decisions)

    for edge in kept_edges:
        decisions.append(
            EdgeDecisionTrace(
                edge=edge,
                status="kept",
                reason="selected by maximum-score graph decode",
            )
        )

    for edge in valid_edges:
        key = (edge.head, edge.tail, edge.label)
        if key in kept_keys:
            continue
        violation = _dynamic_violation(edge, selected, constraints)
        if violation is not None:
            constraint, reason = violation
        elif edge.score < 0.0:
            constraint = "objective:non_negative_edge_score"
            reason = "negative edge score would lower the maximum total score"
        else:
            constraint = "global_decode:tiebreak"
            reason = "edge lost the documented deterministic tie-break"
        decisions.append(
            EdgeDecisionTrace(
                edge=edge,
                status="pruned",
                reason=reason,
                constraint=constraint,
            )
        )

    return tuple(
        sorted(
            decisions, key=lambda decision: _decision_sort_key(decision, nodes_by_id)
        )
    )


def _dynamic_violation(
    edge: SpanEdge,
    selected_edges: Sequence[SpanEdge],
    constraints: SpanGraphConstraints,
) -> ConstraintViolation | None:
    cardinality = constraints.cardinality.get(edge.label)
    if cardinality is not None:
        outgoing = sum(
            1
            for selected in selected_edges
            if selected.label == edge.label and selected.head == edge.head
        )
        if (
            cardinality.max_outgoing_per_head is not None
            and outgoing >= cardinality.max_outgoing_per_head
        ):
            return (
                f"cardinality:{edge.label}:max_outgoing_per_head",
                (
                    f"head {edge.head!r} already has {outgoing} outgoing "
                    f"{edge.label!r} edge(s)"
                ),
            )
        incoming = sum(
            1
            for selected in selected_edges
            if selected.label == edge.label and selected.tail == edge.tail
        )
        if (
            cardinality.max_incoming_per_tail is not None
            and incoming >= cardinality.max_incoming_per_tail
        ):
            return (
                f"cardinality:{edge.label}:max_incoming_per_tail",
                (
                    f"tail {edge.tail!r} already has {incoming} incoming "
                    f"{edge.label!r} edge(s)"
                ),
            )

    acyclic_labels = constraints.acyclic_edge_labels
    if edge.label in acyclic_labels and _creates_cycle(
        edge, selected_edges, acyclic_labels
    ):
        return (
            f"acyclicity:{edge.label}",
            f"adding {edge.head!r} -> {edge.tail!r} would create a cycle",
        )
    return None


def _creates_cycle(
    edge: SpanEdge,
    selected_edges: Sequence[SpanEdge],
    acyclic_labels: frozenset[str],
) -> bool:
    adjacency: dict[str, set[str]] = {}
    for selected in selected_edges:
        if selected.label in acyclic_labels:
            adjacency.setdefault(selected.head, set()).add(selected.tail)
    return _has_path(edge.tail, edge.head, adjacency)


def _has_path(source: str, target: str, adjacency: Mapping[str, set[str]]) -> bool:
    stack = [source]
    seen: set[str] = set()
    while stack:
        node_id = stack.pop()
        if node_id == target:
            return True
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(adjacency.get(node_id, ()))
    return False


def _node_sort_key(node: SpanNode) -> tuple[int, int, str, str]:
    return (node.start, node.end, node.label, node.node_id)


def _edge_search_key(
    edge: SpanEdge,
    nodes_by_id: Mapping[str, SpanNode],
) -> tuple[Any, ...]:
    return (-edge.score, *_edge_canonical_key(edge, nodes_by_id))


def _edge_canonical_key(
    edge: SpanEdge,
    nodes_by_id: Mapping[str, SpanNode],
) -> tuple[Any, ...]:
    head_key = (
        _node_sort_key(nodes_by_id[edge.head]) if edge.head in nodes_by_id else ()
    )
    tail_key = (
        _node_sort_key(nodes_by_id[edge.tail]) if edge.tail in nodes_by_id else ()
    )
    return (edge.label, head_key, tail_key, edge.head, edge.tail)


def _decision_sort_key(
    decision: EdgeDecisionTrace,
    nodes_by_id: Mapping[str, SpanNode],
) -> tuple[Any, ...]:
    status_order = 0 if decision.status == "kept" else 1
    return (
        *_edge_canonical_key(decision.edge, nodes_by_id),
        status_order,
        decision.constraint or "",
    )


def _metadata_key(edge: SpanEdge) -> str:
    return repr(sorted(edge.metadata.items(), key=lambda item: str(item[0])))


def _plain_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in mapping.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)


__all__ = [
    "EdgeCardinality",
    "EdgeDecisionTrace",
    "GraphExplainReport",
    "SpanEdge",
    "SpanGraph",
    "SpanGraphConstraints",
    "SpanNode",
    "decode_span_graph",
    "edge_f1",
]
