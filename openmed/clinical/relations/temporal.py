"""Deterministic temporal TLINK candidate extraction."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from openmed.processing.advanced_ner import EntitySpan

from .candidate import (
    SpanPairCandidate,
    SpanReference,
    generate_span_pair_candidates,
    normalize_relation_label,
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

_EVENT_LABELS = {
    "condition",
    "diagnosis",
    "event",
    "finding",
    "medication_event",
    "problem",
    "procedure",
    "symptom",
}
_TIMEX_LABELS = {
    "date",
    "duration",
    "set",
    "time",
    "timex",
    "timex3",
    "temporal_expression",
}

_BEGINS_ON_RE = re.compile(
    r"\b(?:began|begins|begin|started|starts|onset|onset\s+on|developed)\b"
    r"(?:\s+\w+){0,2}?\s*(?:on|at|by)?\b",
    re.IGNORECASE,
)
_ENDS_ON_RE = re.compile(
    r"\b(?:ended|ends|resolved|resolves|stopped|stops|completed|discontinued)\b"
    r"(?:\s+\w+){0,2}?\s*(?:on|at|by)?\b",
    re.IGNORECASE,
)
_BEFORE_RE = re.compile(r"\b(?:before|prior\s+to|preceding)\b", re.IGNORECASE)
_AFTER_RE = re.compile(
    r"\b(?:after|following|subsequent\s+to)\b",
    re.IGNORECASE,
)
_FOLLOWED_BY_RE = re.compile(r"\bfollowed\s+by\b", re.IGNORECASE)
_PRECEDED_BY_RE = re.compile(r"\bpreceded\s+by\b", re.IGNORECASE)
_OVERLAP_RE = re.compile(
    r"\b(?:overlapped\s+with|overlaps?\s+with|concurrent\s+with|"
    r"simultaneous\s+with|while|when|at\s+the\s+same\s+time\s+as)\b",
    re.IGNORECASE,
)
_CONTAINS_RE = re.compile(
    r"\b(?:contained|contains|included|includes)\b",
    re.IGNORECASE,
)
_DURING_BETWEEN_RE = re.compile(r"\b(?:during|within)\b", re.IGNORECASE)
_DURING_PREFIX_RE = re.compile(r"\b(?:during|within)\s+$", re.IGNORECASE)


@dataclass(frozen=True)
class TemporalSpanReference:
    """PHI-safe reference to an event or TIMEX span."""

    span_id: str
    label: str
    role: TemporalSpanRole
    start: int
    end: int
    score: float
    text_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready span reference without raw source text."""

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
    """PHI-safe reference to the cue that supported a temporal relation."""

    category: str
    start: int
    end: int
    text_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready cue reference without raw source text."""

        return {
            "category": self.category,
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
        }


@dataclass(frozen=True)
class TemporalRelationCandidate:
    """Typed directed TLINK candidate with privacy-preserving provenance."""

    relation_type: TemporalRelationType
    source: TemporalSpanReference
    target: TemporalSpanReference
    confidence: float
    cue: TemporalCueReference
    features: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def stable_key(self) -> tuple[int, int, int, int, int]:
        """Return a deterministic candidate ordering key."""

        return (
            self.source.start,
            self.target.start,
            TEMPORAL_RELATION_TYPES.index(self.relation_type),
            self.source.end,
            self.target.end,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready candidate without raw clinical text."""

        return {
            "relation_type": self.relation_type,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "confidence": self.confidence,
            "cue": self.cue.to_dict(),
            "features": {key: self.features[key] for key in sorted(self.features)},
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class _ClassifiedPair:
    relation_type: TemporalRelationType
    source: SpanReference
    target: SpanReference
    cue_start: int
    cue_end: int


def extract_tlink_candidates(
    text: str,
    spans: Iterable[EntitySpan | Mapping[str, Any]],
    *,
    max_char_distance: int | None = 240,
    section_by_span: Mapping[tuple[int, int], str] | None = None,
) -> tuple[TemporalRelationCandidate, ...]:
    """Extract typed directed temporal relation candidates.

    The extractor is deterministic and local-only. It emits EVENT-EVENT and
    EVENT-TIMEX TLINK candidates with source/target offsets, hashes,
    confidence, and cue provenance, but never returns raw note text.
    """

    pairs = generate_span_pair_candidates(
        text,
        spans,
        label_predicate=_is_temporal_span_label,
        max_char_distance=max_char_distance,
        section_by_span=section_by_span,
    )
    candidates: list[TemporalRelationCandidate] = []
    seen: set[tuple[int, int, int, int, TemporalRelationType]] = set()
    for pair in pairs:
        classified = _classify_pair(text, pair)
        if classified is None:
            continue
        key = (
            classified.source.start,
            classified.source.end,
            classified.target.start,
            classified.target.end,
            classified.relation_type,
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(_to_candidate(text, pair, classified))
    return tuple(sorted(candidates, key=lambda candidate: candidate.stable_key()))


def _classify_pair(
    text: str,
    pair: SpanPairCandidate,
) -> _ClassifiedPair | None:
    left_role = _span_role(pair.left)
    right_role = _span_role(pair.right)
    if left_role is None or right_role is None:
        return None
    if left_role == "TIMEX" and right_role == "TIMEX":
        return None

    between_start = pair.left.end
    between_end = pair.right.start
    between = text[between_start:between_end]

    if left_role == "EVENT" and right_role == "TIMEX":
        begins = _search_between(_BEGINS_ON_RE, between, base_offset=between_start)
        if begins is not None:
            return _classified("BEGINS_ON", pair.left, pair.right, begins)
        ends = _search_between(_ENDS_ON_RE, between, base_offset=between_start)
        if ends is not None:
            return _classified("ENDS_ON", pair.left, pair.right, ends)

    contains = _search_between(_CONTAINS_RE, between, base_offset=between_start)
    if contains is not None:
        return _classified("CONTAINS", pair.left, pair.right, contains)

    followed_by = _search_between(_FOLLOWED_BY_RE, between, base_offset=between_start)
    if followed_by is not None:
        return _classified("BEFORE", pair.left, pair.right, followed_by)

    preceded_by = _search_between(_PRECEDED_BY_RE, between, base_offset=between_start)
    if preceded_by is not None:
        return _classified("AFTER", pair.left, pair.right, preceded_by)

    before = _search_between(_BEFORE_RE, between, base_offset=between_start)
    if before is not None:
        return _classified("BEFORE", pair.left, pair.right, before)

    after = _search_between(_AFTER_RE, between, base_offset=between_start)
    if after is not None:
        return _classified("AFTER", pair.left, pair.right, after)

    overlap = _search_between(_OVERLAP_RE, between, base_offset=between_start)
    if overlap is not None:
        return _classified("OVERLAP", pair.left, pair.right, overlap)

    during = _search_between(_DURING_BETWEEN_RE, between, base_offset=between_start)
    if during is not None:
        return _classified("CONTAINS", pair.right, pair.left, during)

    prefix_start = max(0, pair.left.start - 24)
    prefix = text[prefix_start : pair.left.start]
    prefix_match = _DURING_PREFIX_RE.search(prefix)
    if prefix_match is not None:
        cue = (prefix_start + prefix_match.start(), prefix_start + prefix_match.end())
        return _classified("CONTAINS", pair.left, pair.right, cue)

    return None


def _classified(
    relation_type: TemporalRelationType,
    source: SpanReference,
    target: SpanReference,
    cue_offsets: tuple[int, int],
) -> _ClassifiedPair:
    return _ClassifiedPair(
        relation_type=relation_type,
        source=source,
        target=target,
        cue_start=cue_offsets[0],
        cue_end=cue_offsets[1],
    )


def _search_between(
    pattern: re.Pattern[str],
    value: str,
    *,
    base_offset: int,
) -> tuple[int, int] | None:
    match = pattern.search(value)
    if match is None:
        return None
    return base_offset + match.start(), base_offset + match.end()


def _to_candidate(
    text: str,
    pair: SpanPairCandidate,
    classified: _ClassifiedPair,
) -> TemporalRelationCandidate:
    features = {
        "cue_match": 1.0,
        "event_timex_pair": (
            1.0
            if {_span_role(classified.source), _span_role(classified.target)}
            == {"EVENT", "TIMEX"}
            else 0.0
        ),
        "intervening_span_count": float(pair.intervening_span_count),
        "pair_char_distance": float(pair.char_distance),
    }
    return TemporalRelationCandidate(
        relation_type=classified.relation_type,
        source=_safe_span(text, classified.source),
        target=_safe_span(text, classified.target),
        confidence=_confidence(pair),
        cue=TemporalCueReference(
            category=classified.relation_type,
            start=classified.cue_start,
            end=classified.cue_end,
            text_hash=_hash_text(text[classified.cue_start : classified.cue_end]),
        ),
        features=features,
        provenance={
            "schema_version": TEMPORAL_RELATION_SCHEMA_VERSION,
            "extractor": "deterministic_temporal_tlink",
        },
    )


def _safe_span(text: str, span: SpanReference) -> TemporalSpanReference:
    role = _span_role(span)
    if role is None:
        msg = f"unsupported temporal span label: {span.label!r}"
        raise ValueError(msg)
    return TemporalSpanReference(
        span_id=f"{role.lower()}:{span.start}:{span.end}",
        label=span.label,
        role=role,
        start=span.start,
        end=span.end,
        score=span.score,
        text_hash=_hash_text(text[span.start : span.end]),
    )


def _confidence(pair: SpanPairCandidate) -> float:
    penalty = min(pair.char_distance, 160) / 1000.0
    penalty += min(pair.intervening_span_count, 4) * 0.03
    return round(max(0.5, 0.94 - penalty), 6)


def _is_temporal_span_label(label: str) -> bool:
    return _span_role_from_label(label) is not None


def _span_role(span: SpanReference) -> TemporalSpanRole | None:
    return _span_role_from_label(span.label)


def _span_role_from_label(label: str) -> TemporalSpanRole | None:
    normalized = normalize_relation_label(label)
    if normalized in _TIMEX_LABELS or "timex" in normalized:
        return "TIMEX"
    if normalized in _EVENT_LABELS or normalized.endswith("_event"):
        return "EVENT"
    return None


def _hash_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "TEMPORAL_RELATION_SCHEMA_VERSION",
    "TEMPORAL_RELATION_TYPES",
    "TemporalCueReference",
    "TemporalRelationCandidate",
    "TemporalRelationType",
    "TemporalSpanReference",
    "TemporalSpanRole",
    "extract_tlink_candidates",
]
