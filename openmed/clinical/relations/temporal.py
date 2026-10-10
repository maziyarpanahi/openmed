"""Deterministic, privacy-safe temporal TLINK candidate extraction.

The extractor consumes existing EVENT and TIMEX spans and delegates span
pairing and cue scoring to :func:`build_relation_candidates`. Returned
candidates contain only typed labels, offsets, confidence values, and content
hashes. Raw note or cue text is never retained in the public structures.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast

from openmed.core.audit import hash_text
from openmed.core.decoding import (
    SpanEdge,
    SpanGraph,
    SpanGraphConstraints,
    SpanNode,
    decode_span_graph,
)
from openmed.core.labels import normalize_label
from openmed.processing.advanced_ner import EntitySpan

from .candidate import (
    RelationCandidateBatch,
    RelationCandidateRule,
    SpanReference,
    build_relation_candidates,
)

TemporalRelationType = Literal[
    "BEFORE",
    "AFTER",
    "OVERLAP",
    "CONTAINS",
    "BEGINS_ON",
    "ENDS_ON",
]
TemporalSpanRole = Literal["EVENT", "TIMEX"]

TEMPORAL_RELATION_TYPES: tuple[TemporalRelationType, ...] = (
    "BEFORE",
    "AFTER",
    "OVERLAP",
    "CONTAINS",
    "BEGINS_ON",
    "ENDS_ON",
)
TEMPORAL_RELATION_SCHEMA_VERSION = 1
TEMPORAL_GRAPH_SCHEMA_VERSION = 1

_REDUCED_TEMPORAL_RELATION_TYPES = frozenset(
    {"BEFORE", "OVERLAP", "CONTAINS", "BEGINS_ON", "ENDS_ON"}
)
_PARTIAL_ORDER_RELATION_TYPES = frozenset({"BEFORE", "CONTAINS"})

_EVENT_LABEL_ALIASES = frozenset(
    {
        "CONDITION",
        "DIAGNOSIS",
        "EVENT",
        "FINDING",
        "MEDICATION_EVENT",
        "PROBLEM",
        "PROCEDURE",
        "SYMPTOM",
    }
)
_TIMEX_LABEL_ALIASES = frozenset(
    {
        "DATE",
        "DURATION",
        "SET",
        "TEMPORAL_EXPRESSION",
        "TIME",
        "TIMEX",
        "TIMEX3",
    }
)
_EVENT_LABELS = frozenset(normalize_label(label) for label in _EVENT_LABEL_ALIASES)
_TIMEX_LABELS = frozenset(normalize_label(label) for label in _TIMEX_LABEL_ALIASES)
_TEMPORAL_LABELS = _EVENT_LABELS | _TIMEX_LABELS

_BEFORE_DIRECT = "before_direct"
_AFTER_DIRECT = "after_direct"
_OVERLAP_DIRECT = "overlap_direct"
_CONTAINS_DIRECT = "contains_direct"
_CONTAINS_REVERSE = "contains_reverse"
_BEGINS_ON_DIRECT = "begins_on_direct"
_ENDS_ON_DIRECT = "ends_on_direct"

_DIRECT_RULES = frozenset(
    {
        _BEFORE_DIRECT,
        _AFTER_DIRECT,
        _OVERLAP_DIRECT,
        _CONTAINS_DIRECT,
        _BEGINS_ON_DIRECT,
        _ENDS_ON_DIRECT,
    }
)


@dataclass(frozen=True)
class TemporalSpanReference:
    """Privacy-safe reference to one EVENT or TIMEX source span."""

    span_id: str
    label: str
    role: TemporalSpanRole
    start: int
    end: int
    score: float
    text_hash: str

    def __post_init__(self) -> None:
        if not self.span_id or not self.label:
            raise ValueError("temporal span id and label must be non-empty")
        if self.role not in {"EVENT", "TIMEX"}:
            raise ValueError("temporal span role must be EVENT or TIMEX")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("temporal span offsets must satisfy 0 <= start < end")
        if not math.isfinite(float(self.score)):
            raise ValueError("temporal span score must be finite")
        if not self.text_hash.startswith("sha256:"):
            raise ValueError("temporal span text_hash must be a SHA-256 hash")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible reference without raw source text."""

        return {
            "id": self.span_id,
            "label": self.label,
            "role": self.role,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "text_hash": self.text_hash,
        }


@dataclass(frozen=True)
class TemporalCueReference:
    """Privacy-safe offsets and hash for the supporting temporal cue."""

    category: TemporalRelationType
    start: int
    end: int
    text_hash: str

    def __post_init__(self) -> None:
        if self.category not in TEMPORAL_RELATION_TYPES:
            raise ValueError("unsupported temporal cue category")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("temporal cue offsets must satisfy 0 <= start < end")
        if not self.text_hash.startswith("sha256:"):
            raise ValueError("temporal cue text_hash must be a SHA-256 hash")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible cue reference without raw cue text."""

        return {
            "category": self.category,
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
        }


@dataclass(frozen=True)
class TemporalRelationCandidate:
    """Typed directed TLINK candidate with safe extraction provenance."""

    relation_type: TemporalRelationType
    source: TemporalSpanReference
    target: TemporalSpanReference
    confidence: float
    cue: TemporalCueReference
    features: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.relation_type not in TEMPORAL_RELATION_TYPES:
            raise ValueError("unsupported temporal relation type")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("temporal relation confidence must be between 0 and 1")
        object.__setattr__(
            self,
            "features",
            MappingProxyType(
                {key: float(value) for key, value in sorted(self.features.items())}
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )

    def stable_key(self) -> tuple[int, int, int, int, int, int, int]:
        """Return the deterministic ordering key for this candidate."""

        return (
            self.source.start,
            self.target.start,
            TEMPORAL_RELATION_TYPES.index(self.relation_type),
            self.source.end,
            self.target.end,
            self.cue.start,
            self.cue.end,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible candidate without raw clinical text."""

        return {
            "relation_type": self.relation_type,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "confidence": self.confidence,
            "cue": self.cue.to_dict(),
            "features": dict(self.features),
            "provenance": dict(self.provenance),
        }


def extract_tlink_candidates(
    text: str,
    spans: Iterable[EntitySpan | Mapping[str, Any]],
    *,
    max_char_distance: int | None = 240,
) -> tuple[TemporalRelationCandidate, ...]:
    """Extract typed directed EVENT-EVENT and EVENT-TIMEX candidates.

    Args:
        text: Source document used only during in-memory extraction.
        spans: Existing clinical spans with character offsets into ``text``.
        max_char_distance: Maximum gap allowed between a candidate pair. Pass
            ``None`` to allow any within-sentence distance.

    Returns:
        Stable TLINK candidates with offsets, labels, confidence, cue offsets,
        and hashes. No returned object contains raw note or cue text.
    """

    if max_char_distance is not None and max_char_distance < 0:
        raise ValueError("max_char_distance must be non-negative or None")
    distance_limit = len(text) if max_char_distance is None else max_char_distance
    batch = build_relation_candidates(
        text,
        spans,
        _temporal_relation_rules(distance_limit),
        language="en",
    )

    candidates: list[TemporalRelationCandidate] = []
    seen: set[tuple[Any, ...]] = set()
    for edge in batch.candidates:
        candidate = _safe_temporal_candidate(text, batch, edge)
        if candidate is None:
            continue
        key = (
            candidate.relation_type,
            candidate.source.start,
            candidate.source.end,
            candidate.target.start,
            candidate.target.end,
            candidate.cue.start,
            candidate.cue.end,
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)

    return tuple(sorted(candidates, key=lambda candidate: candidate.stable_key()))


def decode_tlink_candidates(
    candidates: Iterable[TemporalRelationCandidate],
    *,
    min_confidence: float = 0.0,
) -> SpanGraph:
    """Decode typed TLINK candidates into a consistent reduced graph.

    ``AFTER(source, target)`` is canonicalized to
    ``BEFORE(target, source)`` so contradictory BEFORE/AFTER claims share one
    partial-order constraint. ``BEFORE`` and ``CONTAINS`` edges are decoded as
    an acyclic graph and transitively reduced. The original relation type,
    direction, cue offsets, hashes, and extraction provenance remain attached
    to each :class:`~openmed.core.decoding.SpanEdge` decision trace.

    Args:
        candidates: Privacy-safe typed TLINK candidates to decode.
        min_confidence: Inclusive confidence floor between zero and one.

    Returns:
        A deterministic span graph containing only reduced retained edges plus
        kept and pruned provenance for every supplied candidate.

    Raises:
        ValueError: If ``min_confidence`` is invalid or one span id is reused
            with conflicting span metadata.
    """

    if not math.isfinite(float(min_confidence)) or not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be finite and between 0 and 1")

    candidate_tuple = tuple(candidates)
    spans_by_id: dict[str, TemporalSpanReference] = {}
    for candidate in candidate_tuple:
        for span in (candidate.source, candidate.target):
            existing = spans_by_id.get(span.span_id)
            if existing is not None and existing != span:
                raise ValueError(
                    f"temporal span id {span.span_id!r} has conflicting metadata"
                )
            spans_by_id[span.span_id] = span

    nodes = tuple(_temporal_graph_node(span) for span in spans_by_id.values())
    edges = tuple(_temporal_graph_edge(candidate) for candidate in candidate_tuple)
    return decode_span_graph(
        nodes,
        edges,
        constraints=SpanGraphConstraints(
            allowed_edge_labels=_REDUCED_TEMPORAL_RELATION_TYPES,
            acyclic_edge_labels=_PARTIAL_ORDER_RELATION_TYPES,
            transitive_reduction_edge_labels=_PARTIAL_ORDER_RELATION_TYPES,
        ),
        min_edge_score=min_confidence,
    )


def _temporal_graph_node(span: TemporalSpanReference) -> SpanNode:
    return SpanNode(
        node_id=span.span_id,
        start=span.start,
        end=span.end,
        label=span.role,
        score=span.score,
        text_hash=span.text_hash,
        metadata={
            "schema_version": TEMPORAL_GRAPH_SCHEMA_VERSION,
            "source_label": span.label,
        },
    )


def _temporal_graph_edge(candidate: TemporalRelationCandidate) -> SpanEdge:
    source = candidate.source
    target = candidate.target
    relation_type = candidate.relation_type
    canonical_relation_type = relation_type
    if relation_type == "AFTER":
        source, target = target, source
        canonical_relation_type = "BEFORE"
    elif relation_type == "OVERLAP" and _temporal_span_key(target) < _temporal_span_key(
        source
    ):
        source, target = target, source

    return SpanEdge(
        head=source.span_id,
        tail=target.span_id,
        label=canonical_relation_type,
        score=candidate.confidence,
        metadata={
            "schema_version": TEMPORAL_GRAPH_SCHEMA_VERSION,
            "candidate_relation_type": relation_type,
            "candidate_source_id": candidate.source.span_id,
            "candidate_target_id": candidate.target.span_id,
            "cue": candidate.cue.to_dict(),
            "features": dict(candidate.features),
            "provenance": dict(candidate.provenance),
        },
    )


def _temporal_span_key(span: TemporalSpanReference) -> tuple[int, int, str, str]:
    return (span.start, span.end, span.role, span.span_id)


def _temporal_relation_rules(
    max_char_distance: int,
) -> tuple[RelationCandidateRule, ...]:
    common = {"max_character_distance": max_char_distance}
    return (
        RelationCandidateRule(
            relation_type="BEFORE",
            source_relation=_BEFORE_DIRECT,
            head_labels=_EVENT_LABELS,
            tail_labels=_TEMPORAL_LABELS,
            cues=(
                "followed by",
                "then",
                "prior to",
                "before",
                "precedes",
                "preceding",
            ),
            **common,
        ),
        RelationCandidateRule(
            relation_type="AFTER",
            source_relation=_AFTER_DIRECT,
            head_labels=_EVENT_LABELS,
            tail_labels=_TEMPORAL_LABELS,
            cues=("preceded by", "subsequent to", "after", "following"),
            **common,
        ),
        RelationCandidateRule(
            relation_type="OVERLAP",
            source_relation=_OVERLAP_DIRECT,
            head_labels=_EVENT_LABELS,
            tail_labels=_TEMPORAL_LABELS,
            cues=(
                "at the same time as",
                "simultaneous with",
                "overlapped with",
                "concurrent with",
                "overlaps with",
                "overlap with",
            ),
            **common,
        ),
        RelationCandidateRule(
            relation_type="CONTAINS",
            source_relation=_CONTAINS_DIRECT,
            head_labels=_TEMPORAL_LABELS,
            tail_labels=_TEMPORAL_LABELS,
            cues=("contained", "contains", "included", "includes"),
            **common,
        ),
        RelationCandidateRule(
            relation_type="CONTAINS",
            source_relation=_CONTAINS_REVERSE,
            head_labels=_TEMPORAL_LABELS,
            tail_labels=_TEMPORAL_LABELS,
            cues=("throughout", "during", "within"),
            **common,
        ),
        RelationCandidateRule(
            relation_type="BEGINS_ON",
            source_relation=_BEGINS_ON_DIRECT,
            head_labels=_EVENT_LABELS,
            tail_labels=_TIMEX_LABELS,
            cues=("commenced on", "started on", "began on", "begins on"),
            **common,
        ),
        RelationCandidateRule(
            relation_type="ENDS_ON",
            source_relation=_ENDS_ON_DIRECT,
            head_labels=_EVENT_LABELS,
            tail_labels=_TIMEX_LABELS,
            cues=(
                "discontinued on",
                "completed on",
                "resolved on",
                "stopped on",
                "ended on",
                "ends on",
            ),
            **common,
        ),
    )


def _safe_temporal_candidate(
    text: str,
    batch: RelationCandidateBatch,
    edge: SpanEdge,
) -> TemporalRelationCandidate | None:
    source = batch.spans_by_node_id[edge.head]
    target = batch.spans_by_node_id[edge.tail]
    source_role = _span_role(source)
    target_role = _span_role(target)
    if source_role is None or target_role is None:
        return None
    relation_type = cast(TemporalRelationType, edge.label)
    if not _valid_role_pair(relation_type, source_role, target_role):
        return None

    cue_start = _metadata_offset(edge.metadata, "cue_start")
    cue_end = _metadata_offset(edge.metadata, "cue_end")
    source_relation = str(edge.metadata.get("source_relation", ""))
    if cue_start is None or cue_end is None:
        return None
    if not _has_supported_orientation(
        source,
        target,
        cue_start=cue_start,
        cue_end=cue_end,
        source_relation=source_relation,
    ):
        return None

    source_ref = _safe_span(text, source, source_role)
    target_ref = _safe_span(text, target, target_role)
    return TemporalRelationCandidate(
        relation_type=relation_type,
        source=source_ref,
        target=target_ref,
        confidence=round(max(0.0, min(float(edge.score), 1.0)), 6),
        cue=TemporalCueReference(
            category=relation_type,
            start=cue_start,
            end=cue_end,
            text_hash=hash_text(text[cue_start:cue_end]),
        ),
        features={
            "cue_match": 1.0,
            "event_timex_pair": float({source_role, target_role} == {"EVENT", "TIMEX"}),
            "intervening_span_count": float(
                _intervening_span_count(batch, source, target)
            ),
            "pair_char_distance": float(edge.metadata["character_distance"]),
        },
        provenance={
            "schema_version": TEMPORAL_RELATION_SCHEMA_VERSION,
            "extractor": "deterministic_temporal_tlink",
        },
    )


def _has_supported_orientation(
    source: SpanReference,
    target: SpanReference,
    *,
    cue_start: int,
    cue_end: int,
    source_relation: str,
) -> bool:
    left, right = sorted(
        (source, target),
        key=lambda span: (span.start, span.end, normalize_label(span.label)),
    )
    if left.end > right.start:
        return False
    if not left.end <= cue_start < cue_end <= right.start:
        return False
    if source_relation in _DIRECT_RULES:
        return source is left and target is right
    if source_relation == _CONTAINS_REVERSE:
        return source is right and target is left
    return False


def _safe_span(
    text: str,
    span: SpanReference,
    role: TemporalSpanRole,
) -> TemporalSpanReference:
    label = _temporal_label(span.label)
    return TemporalSpanReference(
        span_id=f"{role.casefold()}:{label.casefold()}:{span.start}:{span.end}",
        label=label,
        role=role,
        start=span.start,
        end=span.end,
        score=round(max(0.0, min(float(span.score), 1.0)), 6),
        text_hash=hash_text(text[span.start : span.end]),
    )


def _span_role(span: SpanReference) -> TemporalSpanRole | None:
    label = _temporal_label(span.label)
    if label in _TIMEX_LABEL_ALIASES:
        return "TIMEX"
    if label in _EVENT_LABEL_ALIASES:
        return "EVENT"
    return None


def _valid_role_pair(
    relation_type: TemporalRelationType,
    source_role: TemporalSpanRole,
    target_role: TemporalSpanRole,
) -> bool:
    if relation_type in {"BEGINS_ON", "ENDS_ON"}:
        return source_role == "EVENT" and target_role == "TIMEX"
    if relation_type in {"BEFORE", "AFTER", "OVERLAP"}:
        return source_role == "EVENT"
    return "EVENT" in {source_role, target_role}


def _temporal_label(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", label.upper()).strip("_")


def _metadata_offset(metadata: Mapping[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _intervening_span_count(
    batch: RelationCandidateBatch,
    source: SpanReference,
    target: SpanReference,
) -> int:
    left, right = sorted((source, target), key=lambda span: (span.start, span.end))
    return sum(
        1
        for span in batch.spans_by_node_id.values()
        if span is not source
        and span is not target
        and left.end <= span.start
        and span.end <= right.start
    )


__all__ = [
    "TEMPORAL_GRAPH_SCHEMA_VERSION",
    "TEMPORAL_RELATION_SCHEMA_VERSION",
    "TEMPORAL_RELATION_TYPES",
    "TemporalCueReference",
    "TemporalRelationCandidate",
    "TemporalRelationType",
    "TemporalSpanReference",
    "TemporalSpanRole",
    "decode_tlink_candidates",
    "extract_tlink_candidates",
]
