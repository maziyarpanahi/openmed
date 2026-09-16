"""Backend-neutral joint entity and relation decoding over encoder states.

The learned path enumerates contiguous spans, builds one shared representation
for entity and relation classification, and down-weights typed relation scores
by the weaker endpoint entity score. The fallback path consumes deterministic
``SpanNode`` and ``SpanEdge`` candidates but returns the identical output
schema, allowing downstream graph consumers to remain head-agnostic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from openmed.core.decoding import (
    SpanEdge,
    SpanGraph,
    SpanGraphConstraints,
    SpanNode,
    decode_span_graph,
)

from .candidate import (
    JointSpanCandidate,
    SpanPairCandidate,
    enumerate_joint_span_candidates,
    enumerate_span_pair_candidates,
)

JOINT_HEAD_SCHEMA_VERSION = 1
JointHeadMode = Literal["learned", "fallback"]


@dataclass(frozen=True)
class JointHeadConfig:
    """Threshold and enumeration configuration for joint decoding.

    Args:
        max_span_width: Maximum encoder-token width of an entity candidate.
        max_pair_token_distance: Optional maximum gap between relation endpoints.
        entity_floor: Minimum top entity probability required to emit a node.
        relation_floor: Minimum consistency-adjusted score required for an edge.
    """

    max_span_width: int = 8
    max_pair_token_distance: int | None = 128
    entity_floor: float = 0.5
    relation_floor: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.max_span_width, bool) or not isinstance(
            self.max_span_width, int
        ):
            raise TypeError("max_span_width must be an integer")
        if self.max_span_width < 1:
            raise ValueError("max_span_width must be positive")
        if self.max_pair_token_distance is not None and (
            isinstance(self.max_pair_token_distance, bool)
            or not isinstance(self.max_pair_token_distance, int)
        ):
            raise TypeError("max_pair_token_distance must be an integer when provided")
        if (
            self.max_pair_token_distance is not None
            and self.max_pair_token_distance < 0
        ):
            raise ValueError(
                "max_pair_token_distance must be non-negative when provided"
            )
        for field_name in ("entity_floor", "relation_floor"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True)
class JointSpanPairHead:
    """Linear multi-label head loadable from any encoder backend.

    Entity features concatenate the first, last, and mean encoder state of a
    span plus normalized token width. Relation features concatenate the two
    shared span representations and the mean state between them. Weights are
    plain Python sequences so callers can load converted parameters without
    importing NumPy, PyTorch, MLX, or another array framework.

    Args:
        entity_weights: Label-keyed linear weights over shared span features.
        relation_weights: Label-keyed linear weights over span-pair features.
        entity_bias: Optional label-keyed entity intercepts.
        relation_bias: Optional label-keyed relation intercepts.
    """

    entity_weights: Mapping[str, Sequence[float]]
    relation_weights: Mapping[str, Sequence[float]]
    entity_bias: Mapping[str, float] = field(default_factory=dict)
    relation_bias: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        entity_weights = _normalize_weights(self.entity_weights, "entity")
        relation_weights = _normalize_weights(self.relation_weights, "relation")
        entity_bias = _normalize_biases(
            self.entity_bias,
            entity_weights,
            "entity",
        )
        relation_bias = _normalize_biases(
            self.relation_bias,
            relation_weights,
            "relation",
        )
        object.__setattr__(self, "entity_weights", entity_weights)
        object.__setattr__(self, "relation_weights", relation_weights)
        object.__setattr__(self, "entity_bias", entity_bias)
        object.__setattr__(self, "relation_bias", relation_bias)

    @property
    def entity_labels(self) -> tuple[str, ...]:
        """Return entity labels in deterministic score order."""

        return tuple(self.entity_weights)

    @property
    def relation_labels(self) -> tuple[str, ...]:
        """Return relation labels in deterministic score order."""

        return tuple(self.relation_weights)

    def score_entities(self, features: Sequence[float]) -> Mapping[str, float]:
        """Return bounded entity probabilities for one shared span vector."""

        return _linear_probabilities(features, self.entity_weights, self.entity_bias)

    def score_relations(self, features: Sequence[float]) -> Mapping[str, float]:
        """Return bounded typed-relation probabilities for one span pair."""

        return _linear_probabilities(
            features,
            self.relation_weights,
            self.relation_bias,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return portable learned-head parameters without backend tensors."""

        return {
            "schema_version": JOINT_HEAD_SCHEMA_VERSION,
            "entity_weights": {
                label: list(weights) for label, weights in self.entity_weights.items()
            },
            "relation_weights": {
                label: list(weights) for label, weights in self.relation_weights.items()
            },
            "entity_bias": dict(self.entity_bias),
            "relation_bias": dict(self.relation_bias),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JointSpanPairHead":
        """Load a linear head from a JSON-compatible parameter mapping."""

        schema_version = payload.get("schema_version", JOINT_HEAD_SCHEMA_VERSION)
        if schema_version != JOINT_HEAD_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported joint-head schema version: {schema_version!r}"
            )
        entity_weights = payload.get("entity_weights")
        relation_weights = payload.get("relation_weights")
        if not isinstance(entity_weights, Mapping) or not isinstance(
            relation_weights,
            Mapping,
        ):
            raise ValueError("joint-head weights must be mappings")
        entity_bias = payload.get("entity_bias", {})
        relation_bias = payload.get("relation_bias", {})
        if not isinstance(entity_bias, Mapping) or not isinstance(
            relation_bias,
            Mapping,
        ):
            raise ValueError("joint-head biases must be mappings")
        return cls(
            entity_weights=entity_weights,
            relation_weights=relation_weights,
            entity_bias=entity_bias,
            relation_bias=relation_bias,
        )


@dataclass(frozen=True)
class JointEntityScore:
    """Entity decision for one enumerated or fallback span.

    Args:
        node_id: Stable identifier shared with graph candidates.
        start: Inclusive source character offset.
        end: Exclusive source character offset.
        label: Highest-scoring entity label.
        score: Probability assigned to ``label``.
        scores: Probability for every configured entity label.
        emitted: Whether the entity cleared the configured floor.
        token_start: Optional inclusive encoder-token index.
        token_end: Optional exclusive encoder-token index.
    """

    node_id: str
    start: int
    end: int
    label: str
    score: float
    scores: Mapping[str, float]
    emitted: bool
    token_start: int | None = None
    token_end: int | None = None

    def __post_init__(self) -> None:
        if not self.node_id or not self.label:
            raise ValueError("joint entity node_id and label must be non-empty")
        if not 0 <= self.start < self.end:
            raise ValueError("joint entity offsets must satisfy 0 <= start < end")
        scores = _score_mapping(self.scores)
        if self.label not in scores:
            raise ValueError("selected entity label must be present in scores")
        score = float(self.score)
        _validate_probability(score, "selected entity score")
        if score != scores[self.label]:
            raise ValueError("selected entity score must match its label score")
        if (self.token_start is None) != (self.token_end is None):
            raise ValueError("token_start and token_end must be provided together")
        if self.token_start is not None and not 0 <= self.token_start < self.token_end:
            raise ValueError("entity token offsets must satisfy 0 <= start < end")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "scores", scores)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible entity decision."""

        return {
            "node_id": self.node_id,
            "start": self.start,
            "end": self.end,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "label": self.label,
            "score": self.score,
            "scores": dict(self.scores),
            "emitted": self.emitted,
        }


@dataclass(frozen=True)
class JointSpanPairScore:
    """Joint endpoint consistency and typed-relation score for one pair.

    Args:
        head: Source node identifier.
        tail: Target node identifier.
        relation_label: Highest-scoring typed relation.
        relation_scores: Probability for every configured relation label.
        raw_relation_score: Probability assigned to ``relation_label``.
        entity_consistency_score: Weaker endpoint entity probability.
        adjusted_relation_score: Raw score multiplied by entity consistency.
        emitted: Whether the pair cleared both confidence floors.
        suppression_reason: Stable reason code when the pair was suppressed.
    """

    head: str
    tail: str
    relation_label: str
    relation_scores: Mapping[str, float]
    raw_relation_score: float
    entity_consistency_score: float
    adjusted_relation_score: float
    emitted: bool
    suppression_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.head or not self.tail or not self.relation_label:
            raise ValueError("joint pair identifiers and label must be non-empty")
        scores = _score_mapping(self.relation_scores)
        if self.relation_label not in scores:
            raise ValueError("selected relation label must be present in scores")
        raw_score = float(self.raw_relation_score)
        consistency = float(self.entity_consistency_score)
        adjusted_score = float(self.adjusted_relation_score)
        _validate_probability(raw_score, "raw relation score")
        _validate_probability(consistency, "entity consistency score")
        _validate_probability(adjusted_score, "adjusted relation score")
        if raw_score != scores[self.relation_label]:
            raise ValueError("raw relation score must match its label score")
        if not math.isclose(
            adjusted_score,
            raw_score * consistency,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("adjusted relation score must couple endpoint confidence")
        object.__setattr__(self, "relation_scores", scores)
        object.__setattr__(self, "raw_relation_score", raw_score)
        object.__setattr__(self, "entity_consistency_score", consistency)
        object.__setattr__(self, "adjusted_relation_score", adjusted_score)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible span-pair decision."""

        return {
            "head": self.head,
            "tail": self.tail,
            "relation_label": self.relation_label,
            "relation_scores": dict(self.relation_scores),
            "raw_relation_score": self.raw_relation_score,
            "entity_consistency_score": self.entity_consistency_score,
            "adjusted_relation_score": self.adjusted_relation_score,
            "emitted": self.emitted,
            "suppression_reason": self.suppression_reason,
        }


@dataclass(frozen=True)
class JointHeadOutput:
    """Head-agnostic entity, edge, and decoded graph result.

    Args:
        mode: Whether learned weights or deterministic candidates were used.
        entity_scores: Decisions for every considered entity span.
        pair_scores: Joint scores for every considered directed span pair.
        nodes: Entity nodes passed to the graph decoder.
        candidate_edges: Consistency-adjusted edges passed to graph decoding.
        graph: Constrained decoded span graph.
        schema_version: Stable portable result schema version.
    """

    mode: JointHeadMode
    entity_scores: tuple[JointEntityScore, ...]
    pair_scores: tuple[JointSpanPairScore, ...]
    nodes: tuple[SpanNode, ...]
    candidate_edges: tuple[SpanEdge, ...]
    graph: SpanGraph
    schema_version: int = JOINT_HEAD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the complete portable joint-decoder result."""

        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "entity_scores": [score.to_dict() for score in self.entity_scores],
            "pair_scores": [score.to_dict() for score in self.pair_scores],
            "nodes": [node.to_dict() for node in self.nodes],
            "candidate_edges": [edge.to_dict() for edge in self.candidate_edges],
            "graph": self.graph.to_dict(),
        }


def decode_joint_span_pairs(
    encoder_states: Sequence[Sequence[float]],
    token_offsets: Sequence[tuple[int, int]],
    *,
    head: JointSpanPairHead | None = None,
    fallback_nodes: Sequence[SpanNode] = (),
    fallback_edges: Sequence[SpanEdge] = (),
    config: JointHeadConfig | None = None,
    constraints: SpanGraphConstraints | None = None,
) -> JointHeadOutput:
    """Decode entities and relations together from one encoder-state pass.

    When ``head`` is absent, deterministic fallback nodes and edges traverse
    the same confidence coupling and constrained graph decoder. This preserves
    the result schema for installations that have not loaded learned weights.

    Args:
        encoder_states: One numeric hidden-state vector per encoder token.
        token_offsets: Matching half-open source character offsets.
        head: Optional learned, backend-neutral linear head.
        fallback_nodes: Deterministic entity candidates used only without a head.
        fallback_edges: Deterministic typed relations used only without a head.
        config: Span enumeration and confidence floors.
        constraints: Optional shared graph-decoder constraints.

    Returns:
        Joint scores, emitted graph candidates, and constrained graph output.
    """

    resolved_config = config or JointHeadConfig()
    if head is None:
        return _decode_fallback(
            encoder_states,
            token_offsets,
            fallback_nodes,
            fallback_edges,
            resolved_config,
            constraints,
        )
    if fallback_nodes or fallback_edges:
        raise ValueError("fallback candidates cannot be supplied with a learned head")
    return _decode_learned(
        encoder_states,
        token_offsets,
        head,
        resolved_config,
        constraints,
    )


def _decode_learned(
    encoder_states: Sequence[Sequence[float]],
    token_offsets: Sequence[tuple[int, int]],
    head: JointSpanPairHead,
    config: JointHeadConfig,
    constraints: SpanGraphConstraints | None,
) -> JointHeadOutput:
    states = _normalize_encoder_states(encoder_states, token_offsets)
    spans = enumerate_joint_span_candidates(
        token_offsets,
        max_span_width=config.max_span_width,
    )
    if not states:
        return _empty_output("learned", constraints)

    representations = {
        span.stable_key(): _span_representation(span, states, config.max_span_width)
        for span in spans
    }
    entity_scores: list[JointEntityScore] = []
    nodes: list[SpanNode] = []
    node_ids: dict[tuple[int, int, int, int], str] = {}
    selected_entity_scores: dict[tuple[int, int, int, int], float] = {}

    for span in spans:
        scores = head.score_entities(representations[span.stable_key()])
        label, score = _top_score(scores)
        node_id = _learned_node_id(span)
        emitted = score >= config.entity_floor
        entity_scores.append(
            JointEntityScore(
                node_id=node_id,
                start=span.start,
                end=span.end,
                token_start=span.token_start,
                token_end=span.token_end,
                label=label,
                score=score,
                scores=scores,
                emitted=emitted,
            )
        )
        selected_entity_scores[span.stable_key()] = score
        if not emitted:
            continue
        node_ids[span.stable_key()] = node_id
        nodes.append(
            SpanNode(
                node_id=node_id,
                start=span.start,
                end=span.end,
                label=label,
                score=score,
                metadata={
                    "entity_scores": dict(scores),
                    "joint_head_schema_version": JOINT_HEAD_SCHEMA_VERSION,
                    "joint_mode": "learned",
                    "token_end": span.token_end,
                    "token_start": span.token_start,
                },
            )
        )

    pair_scores: list[JointSpanPairScore] = []
    candidate_edges: list[SpanEdge] = []
    pairs = enumerate_span_pair_candidates(
        spans,
        max_token_distance=config.max_pair_token_distance,
    )
    for pair in pairs:
        head_key = pair.head.stable_key()
        tail_key = pair.tail.stable_key()
        relation_features = _relation_representation(
            pair,
            representations,
            states,
        )
        relation_scores = head.score_relations(relation_features)
        relation_label, raw_relation_score = _top_score(relation_scores)
        consistency = min(
            selected_entity_scores[head_key],
            selected_entity_scores[tail_key],
        )
        adjusted_score = raw_relation_score * consistency
        suppression_reason = _suppression_reason(
            selected_entity_scores[head_key],
            selected_entity_scores[tail_key],
            adjusted_score,
            config,
        )
        emitted = suppression_reason is None
        head_id = _learned_node_id(pair.head)
        tail_id = _learned_node_id(pair.tail)
        pair_scores.append(
            JointSpanPairScore(
                head=head_id,
                tail=tail_id,
                relation_label=relation_label,
                relation_scores=relation_scores,
                raw_relation_score=raw_relation_score,
                entity_consistency_score=consistency,
                adjusted_relation_score=adjusted_score,
                emitted=emitted,
                suppression_reason=suppression_reason,
            )
        )
        if not emitted:
            continue
        candidate_edges.append(
            SpanEdge(
                head=node_ids[head_key],
                tail=node_ids[tail_key],
                label=relation_label,
                score=adjusted_score,
                metadata=_joint_edge_metadata(
                    mode="learned",
                    head_score=selected_entity_scores[head_key],
                    tail_score=selected_entity_scores[tail_key],
                    raw_relation_score=raw_relation_score,
                    consistency=consistency,
                ),
            )
        )

    return _build_output(
        mode="learned",
        entity_scores=entity_scores,
        pair_scores=pair_scores,
        nodes=nodes,
        candidate_edges=candidate_edges,
        constraints=constraints,
    )


def _decode_fallback(
    encoder_states: Sequence[Sequence[float]],
    token_offsets: Sequence[tuple[int, int]],
    fallback_nodes: Sequence[SpanNode],
    fallback_edges: Sequence[SpanEdge],
    config: JointHeadConfig,
    constraints: SpanGraphConstraints | None,
) -> JointHeadOutput:
    if len(encoder_states) != len(token_offsets):
        raise ValueError("encoder_states and token_offsets must have equal length")
    nodes_by_id: dict[str, SpanNode] = {}
    node_scores: dict[str, float] = {}
    entity_scores: list[JointEntityScore] = []
    emitted_nodes: list[SpanNode] = []
    for node in fallback_nodes:
        if node.node_id in nodes_by_id:
            raise ValueError(f"duplicate fallback SpanNode id {node.node_id!r}")
        score = 1.0 if node.score is None else float(node.score)
        _validate_probability(score, "fallback entity score")
        nodes_by_id[node.node_id] = node
        node_scores[node.node_id] = score
        emitted = score >= config.entity_floor
        entity_scores.append(
            JointEntityScore(
                node_id=node.node_id,
                start=node.start,
                end=node.end,
                label=node.label,
                score=score,
                scores={node.label: score},
                emitted=emitted,
            )
        )
        if emitted:
            emitted_nodes.append(node)

    edge_lookup: dict[tuple[str, str, str], SpanEdge] = {}
    relation_labels: set[str] = set()
    for edge in fallback_edges:
        if edge.head not in nodes_by_id or edge.tail not in nodes_by_id:
            raise ValueError("fallback edges must reference fallback nodes")
        _validate_probability(edge.score, "fallback relation score")
        relation_labels.add(edge.label)
        key = edge.head, edge.tail, edge.label
        current = edge_lookup.get(key)
        if current is None or edge.score > current.score:
            edge_lookup[key] = edge

    pair_scores: list[JointSpanPairScore] = []
    candidate_edges: list[SpanEdge] = []
    ordered_labels = tuple(sorted(relation_labels))
    ordered_nodes = tuple(
        sorted(
            fallback_nodes,
            key=lambda node: (node.start, node.end, node.label, node.node_id),
        )
    )
    if ordered_labels:
        for head_node in ordered_nodes:
            for tail_node in ordered_nodes:
                if head_node.node_id == tail_node.node_id:
                    continue
                scores = {
                    label: (
                        edge_lookup[(head_node.node_id, tail_node.node_id, label)].score
                        if (head_node.node_id, tail_node.node_id, label) in edge_lookup
                        else 0.0
                    )
                    for label in ordered_labels
                }
                relation_label, raw_relation_score = _top_score(scores)
                consistency = min(
                    node_scores[head_node.node_id],
                    node_scores[tail_node.node_id],
                )
                adjusted_score = raw_relation_score * consistency
                relation_key = (
                    head_node.node_id,
                    tail_node.node_id,
                    relation_label,
                )
                suppression_reason = (
                    _suppression_reason(
                        node_scores[head_node.node_id],
                        node_scores[tail_node.node_id],
                        adjusted_score,
                        config,
                    )
                    if relation_key in edge_lookup
                    else "no_relation"
                )
                emitted = suppression_reason is None
                pair_scores.append(
                    JointSpanPairScore(
                        head=head_node.node_id,
                        tail=tail_node.node_id,
                        relation_label=relation_label,
                        relation_scores=scores,
                        raw_relation_score=raw_relation_score,
                        entity_consistency_score=consistency,
                        adjusted_relation_score=adjusted_score,
                        emitted=emitted,
                        suppression_reason=suppression_reason,
                    )
                )
                if not emitted:
                    continue
                source_edge = edge_lookup[relation_key]
                metadata = dict(source_edge.metadata)
                metadata.update(
                    _joint_edge_metadata(
                        mode="fallback",
                        head_score=node_scores[head_node.node_id],
                        tail_score=node_scores[tail_node.node_id],
                        raw_relation_score=raw_relation_score,
                        consistency=consistency,
                    )
                )
                candidate_edges.append(
                    SpanEdge(
                        head=source_edge.head,
                        tail=source_edge.tail,
                        label=source_edge.label,
                        score=adjusted_score,
                        metadata=metadata,
                    )
                )

    return _build_output(
        mode="fallback",
        entity_scores=entity_scores,
        pair_scores=pair_scores,
        nodes=emitted_nodes,
        candidate_edges=candidate_edges,
        constraints=constraints,
    )


def _build_output(
    *,
    mode: JointHeadMode,
    entity_scores: Sequence[JointEntityScore],
    pair_scores: Sequence[JointSpanPairScore],
    nodes: Sequence[SpanNode],
    candidate_edges: Sequence[SpanEdge],
    constraints: SpanGraphConstraints | None,
) -> JointHeadOutput:
    ordered_nodes = tuple(
        sorted(nodes, key=lambda node: (node.start, node.end, node.label, node.node_id))
    )
    ordered_edges = tuple(
        sorted(
            candidate_edges,
            key=lambda edge: (edge.label, edge.head, edge.tail, -edge.score),
        )
    )
    graph = decode_span_graph(
        ordered_nodes,
        ordered_edges,
        constraints=constraints,
    )
    return JointHeadOutput(
        mode=mode,
        entity_scores=tuple(entity_scores),
        pair_scores=tuple(pair_scores),
        nodes=ordered_nodes,
        candidate_edges=ordered_edges,
        graph=graph,
    )


def _empty_output(
    mode: JointHeadMode,
    constraints: SpanGraphConstraints | None,
) -> JointHeadOutput:
    return _build_output(
        mode=mode,
        entity_scores=(),
        pair_scores=(),
        nodes=(),
        candidate_edges=(),
        constraints=constraints,
    )


def _normalize_encoder_states(
    encoder_states: Sequence[Sequence[float]],
    token_offsets: Sequence[tuple[int, int]],
) -> tuple[tuple[float, ...], ...]:
    if len(encoder_states) != len(token_offsets):
        raise ValueError("encoder_states and token_offsets must have equal length")
    if not encoder_states:
        return ()
    normalized: list[tuple[float, ...]] = []
    hidden_size: int | None = None
    for state in encoder_states:
        vector = tuple(float(value) for value in state)
        if not vector:
            raise ValueError("encoder state vectors must be non-empty")
        if hidden_size is None:
            hidden_size = len(vector)
        elif len(vector) != hidden_size:
            raise ValueError("encoder state vectors must have equal length")
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("encoder state values must be finite")
        normalized.append(vector)
    return tuple(normalized)


def _span_representation(
    span: JointSpanCandidate,
    states: Sequence[Sequence[float]],
    max_span_width: int,
) -> tuple[float, ...]:
    covered = states[span.token_start : span.token_end]
    mean = _mean_states(covered, len(states[0]))
    return (
        *states[span.token_start],
        *states[span.token_end - 1],
        *mean,
        span.token_width / max_span_width,
    )


def _relation_representation(
    pair: SpanPairCandidate,
    representations: Mapping[tuple[int, int, int, int], Sequence[float]],
    states: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    head = representations[pair.head.stable_key()]
    tail = representations[pair.tail.stable_key()]
    if pair.head.token_end <= pair.tail.token_start:
        between = states[pair.head.token_end : pair.tail.token_start]
    else:
        between = states[pair.tail.token_end : pair.head.token_start]
    context = _mean_states(between, len(states[0]))
    return (*head, *tail, *context)


def _mean_states(
    states: Sequence[Sequence[float]],
    hidden_size: int,
) -> tuple[float, ...]:
    if not states:
        return (0.0,) * hidden_size
    return tuple(
        sum(state[index] for state in states) / len(states)
        for index in range(hidden_size)
    )


def _normalize_weights(
    weights: Mapping[str, Sequence[float]],
    family: str,
) -> Mapping[str, tuple[float, ...]]:
    if not weights:
        raise ValueError(f"{family}_weights must not be empty")
    normalized: dict[str, tuple[float, ...]] = {}
    width: int | None = None
    for raw_label in sorted(weights):
        label = str(raw_label)
        if not label:
            raise ValueError(f"{family} labels must be non-empty")
        vector = tuple(float(value) for value in weights[raw_label])
        if not vector:
            raise ValueError(f"{family} weight vectors must be non-empty")
        if width is None:
            width = len(vector)
        elif len(vector) != width:
            raise ValueError(f"{family} weight vectors must have equal length")
        if any(not math.isfinite(value) for value in vector):
            raise ValueError(f"{family} weights must be finite")
        normalized[label] = vector
    return MappingProxyType(normalized)


def _normalize_biases(
    biases: Mapping[str, float],
    weights: Mapping[str, Sequence[float]],
    family: str,
) -> Mapping[str, float]:
    unknown = {str(label) for label in biases} - set(weights)
    if unknown:
        raise ValueError(f"{family} biases contain unknown labels: {sorted(unknown)}")
    normalized = {label: float(biases.get(label, 0.0)) for label in weights}
    if any(not math.isfinite(value) for value in normalized.values()):
        raise ValueError(f"{family} biases must be finite")
    return MappingProxyType(normalized)


def _linear_probabilities(
    features: Sequence[float],
    weights: Mapping[str, Sequence[float]],
    biases: Mapping[str, float],
) -> Mapping[str, float]:
    vector = tuple(float(value) for value in features)
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("head feature values must be finite")
    expected_width = len(next(iter(weights.values())))
    if len(vector) != expected_width:
        raise ValueError(
            f"head expected {expected_width} features but received {len(vector)}"
        )
    return _score_mapping(
        {
            label: _sigmoid(
                sum(value * weight for value, weight in zip(vector, label_weights))
                + biases[label]
            )
            for label, label_weights in weights.items()
        }
    )


def _score_mapping(scores: Mapping[str, float]) -> Mapping[str, float]:
    normalized: dict[str, float] = {}
    for raw_label in sorted(scores):
        label = str(raw_label)
        if not label:
            raise ValueError("score labels must be non-empty")
        score = float(scores[raw_label])
        _validate_probability(score, f"score for {label!r}")
        normalized[label] = score
    return MappingProxyType(normalized)


def _top_score(scores: Mapping[str, float]) -> tuple[str, float]:
    if not scores:
        raise ValueError("at least one label score is required")
    label = min(scores, key=lambda candidate: (-scores[candidate], candidate))
    return label, scores[label]


def _suppression_reason(
    head_score: float,
    tail_score: float,
    adjusted_relation_score: float,
    config: JointHeadConfig,
) -> str | None:
    if head_score < config.entity_floor or tail_score < config.entity_floor:
        return "entity_floor"
    if adjusted_relation_score < config.relation_floor:
        return "relation_floor"
    return None


def _joint_edge_metadata(
    *,
    mode: JointHeadMode,
    head_score: float,
    tail_score: float,
    raw_relation_score: float,
    consistency: float,
) -> dict[str, Any]:
    return {
        "adjusted_by_entity_consistency": True,
        "entity_consistency_score": consistency,
        "head_entity_score": head_score,
        "joint_head_schema_version": JOINT_HEAD_SCHEMA_VERSION,
        "joint_mode": mode,
        "raw_relation_score": raw_relation_score,
        "tail_entity_score": tail_score,
    }


def _learned_node_id(span: JointSpanCandidate) -> str:
    return f"joint-span-{span.token_start}-{span.token_end}"


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        decay = math.exp(-value)
        return 1.0 / (1.0 + decay)
    growth = math.exp(value)
    return growth / (1.0 + growth)


def _validate_probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite value between 0 and 1")


__all__ = [
    "JOINT_HEAD_SCHEMA_VERSION",
    "JointEntityScore",
    "JointHeadConfig",
    "JointHeadMode",
    "JointHeadOutput",
    "JointSpanPairHead",
    "JointSpanPairScore",
    "decode_joint_span_pairs",
]
