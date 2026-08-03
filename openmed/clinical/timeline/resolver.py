"""Clinical timeline resolution with relative-date normalization."""

from __future__ import annotations

import calendar
import heapq
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from openmed.clinical.context import (
    reconcile_temporality_with_interval,
    resolve_temporality,
)
from openmed.clinical.relations.temporal import (
    TemporalCueReference,
    TemporalRelationCandidate,
    TemporalRelationType,
    TemporalSpanReference,
    extract_tlink_candidates,
)
from openmed.clinical.temporal_normalizer import NormalizedTimex, normalize_temporal
from openmed.clinical.timeline.timex import (
    TemporalExpression,
    detect_timexes,
)
from openmed.core.audit import hash_text
from openmed.core.decoding import (
    EdgeDecisionTrace,
    SpanEdge,
    SpanGraphConstraints,
    SpanNode,
    decode_span_graph,
)
from openmed.processing.advanced_ner import EntitySpan

TIMELINE_ASSISTIVE_DISCLAIMER = (
    "Clinical timeline normalization is assistive and is not a clinical "
    "decision, diagnosis, treatment recommendation, or substitute for "
    "clinician review."
)
EVENT_ANCHORING_ADVISORY = (
    "Clinical event anchoring is deterministic assistive metadata for review "
    "and is not a clinical decision or substitute for clinician verification."
)

TimelineRelationKind = Literal["before", "after", "overlap", "unknown"]
EventAnchorSource = Literal["timex", "dct_fallback"]
TimelineEdgeStatus = Literal["kept", "pruned"]

ORDER_EVENTS_SCHEMA_VERSION = 1

_TEMPORAL_ORDER_EDGE = "TEMPORAL_PRECEDES"
_EVENT_SPAN_LABELS = frozenset(
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
_TIMEX_SPAN_LABELS = frozenset(
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

_ANCHOR_TERMS = {
    "admission": ("admission", "admitted", "hospitalization"),
    "surgery": ("surgery", "operation", "operative", "post-op", "postoperative"),
    "procedure": ("procedure",),
    "discharge": ("discharge", "discharged"),
    "visit": ("visit", "clinic"),
    "onset": ("onset", "started", "began"),
}
_REFERENCE_DEPENDENT_DIRECTIONS = {
    "same",
    "past",
    "future",
    "since",
}


@dataclass(frozen=True)
class NormalizedInterval:
    """A normalized ISO day-level interval with explicit uncertainty bounds."""

    start: date
    end: date
    lower_bound: date
    upper_bound: date
    precision: str = "day"
    uncertainty_days: int = 0

    @property
    def iso_value(self) -> str:
        """Return ``YYYY-MM-DD/YYYY-MM-DD`` interval text."""

        return f"{self.start.isoformat()}/{self.end.isoformat()}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready interval payload."""

        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "value": self.iso_value,
            "lower_bound": self.lower_bound.isoformat(),
            "upper_bound": self.upper_bound.isoformat(),
            "precision": self.precision,
            "uncertainty_days": self.uncertainty_days,
        }


@dataclass(frozen=True)
class TimelineEvent:
    """One timeline event derived from a temporal expression."""

    event_id: str
    text: str
    start: int
    end: int
    timex: TemporalExpression
    interval: NormalizedInterval | None
    temporality: str
    reference_date_dependent: bool
    reference_date_provenance: Mapping[str, Any]
    relation_anchor: str | None = None
    relative_offset_days: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def normalized_value(self) -> str | None:
        """Return the normalized interval value when resolved."""

        return self.interval.iso_value if self.interval is not None else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready event payload."""

        return {
            "id": self.event_id,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "timex": self.timex.to_dict(),
            "interval": self.interval.to_dict() if self.interval else None,
            "normalized_value": self.normalized_value,
            "temporality": self.temporality,
            "reference_date_dependent": self.reference_date_dependent,
            "reference_date_provenance": dict(self.reference_date_provenance),
            "relation_anchor": self.relation_anchor,
            "relative_offset_days": self.relative_offset_days,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class TimelineRelation:
    """A partial-order relation between two events or anchors."""

    source_id: str
    target_id: str
    relation: TimelineRelationKind
    evidence: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready relation payload."""

        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class TimexAnchorReference:
    """Privacy-safe reference to a normalized TIMEX source span."""

    start: int
    end: int
    text_hash: str
    timex_type: str
    normalized_value: str
    granularity_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return TIMEX metadata without raw source text."""

        return {
            "span": [self.start, self.end],
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
            "type": self.timex_type,
            "normalized_value": self.normalized_value,
            "granularity_flags": list(self.granularity_flags),
        }


@dataclass(frozen=True)
class EventTemporalAnchor:
    """One EVENT span anchored to DCT and optionally to a resolved TIMEX."""

    event_start: int
    event_end: int
    event_text_hash: str
    anchor_source: EventAnchorSource
    anchor_value: str
    dct_position: TimelineRelationKind
    timex: TimexAnchorReference | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return anchoring metadata containing no raw note text."""

        return {
            "event": {
                "span": [self.event_start, self.event_end],
                "start": self.event_start,
                "end": self.event_end,
                "text_hash": self.event_text_hash,
            },
            "anchor_source": self.anchor_source,
            "anchor_value": self.anchor_value,
            "dct_position": self.dct_position,
            "timex": self.timex.to_dict() if self.timex is not None else None,
        }


@dataclass(frozen=True)
class EventAnchoringResult:
    """Privacy-safe DCT anchoring output for a collection of EVENT spans."""

    document_creation_time: str
    anchors: tuple[EventTemporalAnchor, ...]
    disclaimer: str = EVENT_ANCHORING_ADVISORY

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-ready anchoring output."""

        return {
            "document_creation_time": self.document_creation_time,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "disclaimer": self.disclaimer,
        }


@dataclass(frozen=True)
class ResolvedTimeline:
    """Resolved timeline with ordered events and partial-order relations."""

    events: tuple[TimelineEvent, ...]
    relations: tuple[TimelineRelation, ...]
    reference_date: date | None
    reference_date_provenance: Mapping[str, Any]
    disclaimer: str = TIMELINE_ASSISTIVE_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready timeline payload."""

        return {
            "events": [event.to_dict() for event in self.events],
            "relations": [relation.to_dict() for relation in self.relations],
            "reference_date": (
                self.reference_date.isoformat()
                if self.reference_date is not None
                else None
            ),
            "reference_date_provenance": dict(self.reference_date_provenance),
            "disclaimer": self.disclaimer,
        }


@dataclass(frozen=True)
class OrderedTimelineEvent:
    """Privacy-safe EVENT position in a decoded document timeline."""

    event_id: str
    label: str
    start: int
    end: int
    text_hash: str
    position: int
    confidence: float

    def __post_init__(self) -> None:
        if not self.event_id or not self.label:
            raise ValueError("ordered timeline event id and label must be non-empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("event offsets must satisfy 0 <= start < end")
        if not self.text_hash.startswith("sha256:"):
            raise ValueError("event text_hash must be a SHA-256 hash")
        if self.position < 0:
            raise ValueError("event position must be non-negative")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("event confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready event containing no raw note text."""

        return {
            "id": self.event_id,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
            "position": self.position,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TimelineEdgeProvenance:
    """Privacy-safe kept or pruned TLINK decision provenance."""

    relation_type: TemporalRelationType
    source: TemporalSpanReference
    target: TemporalSpanReference
    confidence: float
    cue: TemporalCueReference
    status: TimelineEdgeStatus
    reason: str
    constraint: str | None = None
    features: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("edge confidence must be between 0 and 1")
        if self.status not in {"kept", "pruned"}:
            raise ValueError("edge status must be kept or pruned")
        if not self.reason:
            raise ValueError("edge decision reason must be non-empty")
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready edge decision containing no raw note text."""

        return {
            "relation_type": self.relation_type,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "confidence": self.confidence,
            "cue": self.cue.to_dict(),
            "status": self.status,
            "reason": self.reason,
            "constraint": self.constraint,
            "features": dict(self.features),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class Timeline:
    """Cycle-free, privacy-safe ordering of supplied clinical EVENT spans."""

    events: tuple[OrderedTimelineEvent, ...]
    edges: tuple[TimelineEdgeProvenance, ...]
    disclaimer: str = TIMELINE_ASSISTIVE_DISCLAIMER
    schema_version: int = ORDER_EVENTS_SCHEMA_VERSION

    @property
    def edge_provenance(self) -> tuple[TimelineEdgeProvenance, ...]:
        """Return decisions for every retained or pruned TLINK candidate."""

        return self.edges

    @property
    def kept_edges(self) -> tuple[TimelineEdgeProvenance, ...]:
        """Return TLINK candidates retained by graph decoding."""

        return tuple(edge for edge in self.edges if edge.status == "kept")

    @property
    def retained_edges(self) -> tuple[TimelineEdgeProvenance, ...]:
        """Return an alias for retained TLINK candidates."""

        return self.kept_edges

    @property
    def pruned_edges(self) -> tuple[TimelineEdgeProvenance, ...]:
        """Return TLINK candidates rejected by graph decoding."""

        return tuple(edge for edge in self.edges if edge.status == "pruned")

    @property
    def is_cycle_free(self) -> bool:
        """Return whether retained BEFORE/AFTER relations are acyclic."""

        return _timeline_edges_are_acyclic(self.kept_edges)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready timeline containing no raw clinical text."""

        return {
            "schema_version": self.schema_version,
            "events": [event.to_dict() for event in self.events],
            "edges": [edge.to_dict() for edge in self.edges],
            "cycle_free": self.is_cycle_free,
            "disclaimer": self.disclaimer,
        }


@dataclass(frozen=True)
class TimelineEvaluationResult:
    """Value-accuracy and ordering-consistency metrics for gold timelines."""

    value_accuracy: float
    ordering_consistency: float
    value_correct: int
    value_total: int
    ordering_correct: int
    ordering_total: int
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready evaluation payload."""

        return {
            "value_accuracy": self.value_accuracy,
            "ordering_consistency": self.ordering_consistency,
            "value_correct": self.value_correct,
            "value_total": self.value_total,
            "ordering_correct": self.ordering_correct,
            "ordering_total": self.ordering_total,
            "failures": list(self.failures),
        }


def resolve_timeline(
    text: str,
    *,
    reference_date: str | date | datetime | None = None,
) -> ResolvedTimeline:
    """Resolve temporal expressions into a normalized clinical timeline.

    Args:
        text: Source clinical note text.
        reference_date: Optional document reference date.  Relative
            expressions that depend on this anchor remain relative-only when it
            is absent; the resolver never substitutes the current wall clock.

    Returns:
        A ``ResolvedTimeline`` carrying normalized events, partial-order
        relations, reference-date provenance, and the assistive disclaimer.
    """

    document_reference = _coerce_date(reference_date)
    reference_provenance = _reference_date_provenance(
        document_reference=document_reference,
        required=False,
    )
    timexes = detect_timexes(text)
    sentence_windows = _sentence_windows(text)
    anchor_dates: dict[str, date] = {}
    previous_interval_start: date | None = None
    events: list[TimelineEvent] = []

    for index, timex in enumerate(timexes, start=1):
        sentence_start, sentence_end = _sentence_for_offsets(
            sentence_windows,
            timex.start,
            timex.end,
        )
        sentence_text = text[sentence_start:sentence_end].strip()
        interval = _resolve_interval(
            timex,
            document_reference=document_reference,
            anchor_dates=anchor_dates,
            previous_interval_start=previous_interval_start,
        )
        reference_required = _reference_date_required(timex)
        event_reference_provenance = _reference_date_provenance(
            document_reference=document_reference,
            required=reference_required,
        )
        temporality = resolve_temporality(
            {
                "text": sentence_text,
                "context": text,
                "start": sentence_start,
                "end": sentence_end,
            }
        )
        if interval is not None and document_reference is not None:
            temporality = reconcile_temporality_with_interval(
                temporality=temporality,
                interval_start=interval.start,
                interval_end=interval.end,
                reference_date=document_reference,
            )
        event = TimelineEvent(
            event_id=f"t{index}",
            text=sentence_text,
            start=sentence_start,
            end=sentence_end,
            timex=timex,
            interval=interval,
            temporality=temporality,
            reference_date_dependent=reference_required,
            reference_date_provenance=event_reference_provenance,
            relation_anchor=timex.anchor,
            relative_offset_days=_relative_offset_days(timex),
            provenance={
                "source": "openmed.clinical.timeline",
                "timex_start": timex.start,
                "timex_end": timex.end,
                "timex_text": timex.text,
            },
        )
        events.append(event)

        if interval is not None:
            previous_interval_start = interval.start
            _update_anchor_dates(anchor_dates, sentence_text, interval.start)
        elif timex.anchor and timex.anchor in anchor_dates:
            previous_interval_start = anchor_dates[timex.anchor]

    relations = _timeline_relations(events, document_reference)
    ordered_events = tuple(sorted(events, key=_event_sort_key))
    return ResolvedTimeline(
        events=ordered_events,
        relations=tuple(relations),
        reference_date=document_reference,
        reference_date_provenance={
            **reference_provenance,
            "required": any(event.reference_date_dependent for event in events),
        },
    )


def anchor_events(
    text: str,
    event_spans: Iterable[Mapping[str, object] | Sequence[int]],
    document_creation_time: str | date | datetime,
    timex_spans: Iterable[NormalizedTimex | Mapping[str, object] | Sequence[int]]
    | None = None,
    *,
    max_character_distance: int = 160,
) -> EventAnchoringResult:
    """Anchor every supplied EVENT span to DCT and a nearby resolved TIMEX.

    When ``timex_spans`` is omitted, deterministic TIMEX detection supplies
    candidate offsets. Passing an empty iterable disables detection. Only
    resolved ``DATE`` and ``TIME`` records can anchor an event; other or
    unresolved TIMEX values cause a DCT fallback. Candidate links never cross
    a sentence boundary and the nearest candidate wins deterministically.

    Args:
        text: Source document used only in memory for offsets and hashing.
        event_spans: EVENT mappings with ``start``/``end`` offsets or two-item
            offset sequences.
        document_creation_time: Required ISO DCT used by relative-date
            normalization and DCT-position calculation.
        timex_spans: Optional normalized TIMEX records or source offsets. When
            omitted, offsets are detected locally before normalization.
        max_character_distance: Largest character gap allowed between an
            EVENT and its TIMEX candidate within the same sentence.

    Returns:
        An ``EventAnchoringResult`` whose source evidence consists only of
        offsets and hashes. Every input EVENT has exactly one output anchor.

    Raises:
        TypeError: If the text, DCT, or a span has an unsupported type.
        ValueError: If the DCT, distance, or a span is invalid.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(max_character_distance, bool) or max_character_distance < 0:
        raise ValueError("max_character_distance must be non-negative")

    dct_value, dct_date = _coerce_document_creation_time(document_creation_time)
    events = tuple(
        _coerce_source_span(span, text_length=len(text)) for span in event_spans
    )
    normalized_timexes = _event_anchor_timexes(
        text,
        timex_spans,
        document_creation_time=document_creation_time,
    )

    anchors: list[EventTemporalAnchor] = []
    for event_start, event_end in events:
        timex = _nearest_event_timex(
            text,
            event_start=event_start,
            event_end=event_end,
            timexes=normalized_timexes,
            max_character_distance=max_character_distance,
        )
        event_hash = hash_text(text[event_start:event_end])
        if timex is None:
            anchors.append(
                EventTemporalAnchor(
                    event_start=event_start,
                    event_end=event_end,
                    event_text_hash=event_hash,
                    anchor_source="dct_fallback",
                    anchor_value=dct_value,
                    dct_position="overlap",
                )
            )
            continue

        position = _normalized_timex_dct_position(timex, dct_date)
        if position == "unknown" or timex.value is None:
            anchors.append(
                EventTemporalAnchor(
                    event_start=event_start,
                    event_end=event_end,
                    event_text_hash=event_hash,
                    anchor_source="dct_fallback",
                    anchor_value=dct_value,
                    dct_position="overlap",
                )
            )
            continue

        anchors.append(
            EventTemporalAnchor(
                event_start=event_start,
                event_end=event_end,
                event_text_hash=event_hash,
                anchor_source="timex",
                anchor_value=timex.value,
                dct_position=position,
                timex=TimexAnchorReference(
                    start=timex.start,
                    end=timex.end,
                    text_hash=hash_text(text[timex.start : timex.end]),
                    timex_type=timex.timex_type,
                    normalized_value=timex.value,
                    granularity_flags=timex.granularity_flags,
                ),
            )
        )

    return EventAnchoringResult(
        document_creation_time=dct_value,
        anchors=tuple(anchors),
    )


def order_events(
    text: str,
    spans: Iterable[EntitySpan | Mapping[str, Any]],
) -> Timeline:
    """Order supplied EVENT spans using privacy-safe typed TLINK candidates.

    Typed candidates are converted to a single chronological edge direction
    before the shared span-graph decoder applies its acyclicity constraint.
    ``AFTER`` candidates are therefore reversed only for decoding; the public
    provenance retains the original relation type and source/target roles.

    Args:
        text: Source clinical note used only during in-memory extraction.
        spans: Existing EVENT and TIMEX spans with offsets into ``text``.

    Returns:
        A cycle-free ``Timeline``. Events contain normalized labels, offsets,
        hashes, zero-based positions, and confidence. Edge provenance records
        every kept or pruned candidate without retaining raw clinical text.
    """

    span_items = tuple(spans)
    candidates = extract_tlink_candidates(text, span_items)
    references = {
        reference.span_id: reference
        for reference in _safe_ordering_span_references(text, span_items)
    }
    for candidate in candidates:
        _keep_stronger_reference(references, candidate.source)
        _keep_stronger_reference(references, candidate.target)

    nodes = tuple(_timeline_span_node(reference) for reference in references.values())
    candidate_edges = tuple(
        _timeline_candidate_edge(candidate, candidate_index=index)
        for index, candidate in enumerate(candidates)
    )
    graph = decode_span_graph(
        nodes,
        candidate_edges,
        constraints=SpanGraphConstraints(
            allowed_edge_labels={
                _TEMPORAL_ORDER_EDGE,
                "BEGINS_ON",
                "CONTAINS",
                "ENDS_ON",
                "OVERLAP",
            },
            acyclic_edge_labels=(_TEMPORAL_ORDER_EDGE,),
        ),
    )

    ordered_node_ids = _topologically_order_timeline_nodes(graph.nodes, graph.edges)
    order_index = {node_id: index for index, node_id in enumerate(ordered_node_ids)}
    event_references = sorted(
        (reference for reference in references.values() if reference.role == "EVENT"),
        key=lambda reference: order_index[reference.span_id],
    )
    events = tuple(
        OrderedTimelineEvent(
            event_id=reference.span_id,
            label=reference.label,
            start=reference.start,
            end=reference.end,
            text_hash=reference.text_hash,
            position=position,
            confidence=reference.score,
        )
        for position, reference in enumerate(event_references)
    )
    edges = _timeline_edge_provenance(graph.decisions, candidates)
    timeline = Timeline(events=events, edges=edges)
    if not timeline.is_cycle_free:  # Defensive invariant around decoder changes.
        raise RuntimeError("decoded temporal timeline contains a cycle")
    return timeline


def evaluate_timeline_gold(
    cases_or_path: str | Path | Iterable[Mapping[str, Any]],
) -> TimelineEvaluationResult:
    """Evaluate ``resolve_timeline`` against synthetic gold cases."""

    cases = _load_gold_cases(cases_or_path)
    value_correct = 0
    value_total = 0
    ordering_correct = 0
    ordering_total = 0
    failures: list[str] = []

    for case in cases:
        timeline = resolve_timeline(
            str(case["text"]),
            reference_date=case.get("reference_date"),
        )
        events_by_timex = {
            _norm(event.timex.text): event
            for event in timeline.events
            if event.timex.text
        }

        for expected in case.get("expected_events", ()):
            timex_text = str(expected["timex_text"])
            expected_value = expected.get("normalized_value")
            event = events_by_timex.get(_norm(timex_text))
            if expected_value is None:
                continue
            value_total += 1
            if event is not None and event.normalized_value == expected_value:
                value_correct += 1
            else:
                actual = event.normalized_value if event is not None else None
                failures.append(
                    f"{case['id']} value {timex_text!r}: "
                    f"expected {expected_value!r}, got {actual!r}"
                )

        for expected in case.get("expected_relations", ()):
            source = events_by_timex.get(_norm(str(expected["source"])))
            target = events_by_timex.get(_norm(str(expected["target"])))
            ordering_total += 1
            actual_relation = (
                _event_pair_relation(source, target)
                if source is not None and target is not None
                else "unknown"
            )
            if actual_relation == expected["relation"]:
                ordering_correct += 1
            else:
                failures.append(
                    f"{case['id']} relation {expected['source']!r} -> "
                    f"{expected['target']!r}: expected {expected['relation']!r}, "
                    f"got {actual_relation!r}"
                )

    return TimelineEvaluationResult(
        value_accuracy=_rate(value_correct, value_total),
        ordering_consistency=_rate(ordering_correct, ordering_total),
        value_correct=value_correct,
        value_total=value_total,
        ordering_correct=ordering_correct,
        ordering_total=ordering_total,
        failures=tuple(failures),
    )


def _safe_ordering_span_references(
    text: str,
    spans: Sequence[EntitySpan | Mapping[str, Any]],
) -> tuple[TemporalSpanReference, ...]:
    references: dict[str, TemporalSpanReference] = {}
    for item in spans:
        if isinstance(item, TemporalSpanReference):
            start = item.start
            end = item.end
            raw_label = item.label
            score = float(item.score)
        elif isinstance(item, EntitySpan):
            start = item.start
            end = item.end
            raw_label = item.label
            score = float(item.score)
        else:
            metadata = item.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                metadata = {}
            start = int(item.get("start", item.get("start_char", -1)))
            end = int(item.get("end", item.get("end_char", -1)))
            raw_label = str(item.get("label", item.get("entity", "")))
            score = float(item.get("score", metadata.get("confidence", 1.0)))

        label = _timeline_span_label(raw_label)
        if label in _EVENT_SPAN_LABELS:
            role: Literal["EVENT", "TIMEX"] = "EVENT"
        elif label in _TIMEX_SPAN_LABELS:
            role = "TIMEX"
        else:
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        if not math.isfinite(score):
            raise ValueError("temporal span score must be finite")

        reference = TemporalSpanReference(
            span_id=f"{role.casefold()}:{label.casefold()}:{start}:{end}",
            label=label,
            role=role,
            start=start,
            end=end,
            score=round(max(0.0, min(score, 1.0)), 6),
            text_hash=hash_text(text[start:end]),
        )
        _keep_stronger_reference(references, reference)

    return tuple(
        sorted(
            references.values(),
            key=lambda reference: (
                reference.start,
                reference.end,
                reference.label,
                reference.span_id,
            ),
        )
    )


def _keep_stronger_reference(
    references: dict[str, TemporalSpanReference],
    candidate: TemporalSpanReference,
) -> None:
    current = references.get(candidate.span_id)
    if current is None or candidate.score > current.score:
        references[candidate.span_id] = candidate


def _timeline_span_node(reference: TemporalSpanReference) -> SpanNode:
    return SpanNode(
        node_id=reference.span_id,
        start=reference.start,
        end=reference.end,
        label=reference.role,
        score=reference.score,
        text_hash=reference.text_hash,
        metadata={
            "schema_version": ORDER_EVENTS_SCHEMA_VERSION,
            "source_label": reference.label,
        },
    )


def _timeline_candidate_edge(
    candidate: TemporalRelationCandidate,
    *,
    candidate_index: int,
) -> SpanEdge:
    if candidate.relation_type in {"BEFORE", "AFTER"}:
        head, tail = _chronological_endpoints(
            candidate.relation_type,
            candidate.source.span_id,
            candidate.target.span_id,
        )
        label = _TEMPORAL_ORDER_EDGE
    else:
        head = candidate.source.span_id
        tail = candidate.target.span_id
        label = candidate.relation_type
    return SpanEdge(
        head=head,
        tail=tail,
        label=label,
        score=candidate.confidence,
        metadata={"candidate_index": candidate_index},
    )


def _timeline_edge_provenance(
    decisions: Sequence[EdgeDecisionTrace],
    candidates: Sequence[TemporalRelationCandidate],
) -> tuple[TimelineEdgeProvenance, ...]:
    provenance: list[TimelineEdgeProvenance] = []
    for decision in decisions:
        candidate_index = decision.edge.metadata.get("candidate_index")
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or not 0 <= candidate_index < len(candidates)
        ):
            raise RuntimeError("temporal graph decision lost candidate provenance")
        candidate = candidates[candidate_index]
        provenance.append(
            TimelineEdgeProvenance(
                relation_type=candidate.relation_type,
                source=candidate.source,
                target=candidate.target,
                confidence=candidate.confidence,
                cue=candidate.cue,
                status=decision.status,
                reason=decision.reason,
                constraint=decision.constraint,
                features=candidate.features,
                provenance={
                    "candidate": dict(candidate.provenance),
                    "decoder": "span_graph",
                    "schema_version": ORDER_EVENTS_SCHEMA_VERSION,
                },
            )
        )
    return tuple(provenance)


def _topologically_order_timeline_nodes(
    nodes: Sequence[SpanNode],
    edges: Sequence[SpanEdge],
) -> tuple[str, ...]:
    nodes_by_id = {node.node_id: node for node in nodes}
    adjacency = {node.node_id: set() for node in nodes}
    indegree = {node.node_id: 0 for node in nodes}
    for edge in edges:
        if edge.label != _TEMPORAL_ORDER_EDGE:
            continue
        if edge.tail in adjacency[edge.head]:
            continue
        adjacency[edge.head].add(edge.tail)
        indegree[edge.tail] += 1

    ready = [
        (*_timeline_node_sort_key(node), node.node_id)
        for node in nodes
        if indegree[node.node_id] == 0
    ]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        *_, node_id = heapq.heappop(ready)
        ordered.append(node_id)
        for target_id in sorted(
            adjacency[node_id],
            key=lambda item: _timeline_node_sort_key(nodes_by_id[item]),
        ):
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                target = nodes_by_id[target_id]
                heapq.heappush(
                    ready,
                    (*_timeline_node_sort_key(target), target.node_id),
                )

    if len(ordered) != len(nodes):
        raise RuntimeError("decoded temporal graph contains a cycle")
    return tuple(ordered)


def _timeline_node_sort_key(node: SpanNode) -> tuple[int, int, str]:
    return node.start, node.end, node.label


def _timeline_edges_are_acyclic(
    edges: Sequence[TimelineEdgeProvenance],
) -> bool:
    adjacency: dict[str, set[str]] = {}
    indegree: dict[str, int] = {}
    for edge in edges:
        if edge.status != "kept" or edge.relation_type not in {"BEFORE", "AFTER"}:
            continue
        head, tail = _chronological_endpoints(
            edge.relation_type,
            edge.source.span_id,
            edge.target.span_id,
        )
        adjacency.setdefault(head, set())
        adjacency.setdefault(tail, set())
        indegree.setdefault(head, 0)
        indegree.setdefault(tail, 0)
        if tail not in adjacency[head]:
            adjacency[head].add(tail)
            indegree[tail] += 1

    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        node_id = ready.pop()
        visited += 1
        for target_id in adjacency[node_id]:
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                ready.append(target_id)
    return visited == len(indegree)


def _chronological_endpoints(
    relation_type: TemporalRelationType,
    source_id: str,
    target_id: str,
) -> tuple[str, str]:
    if relation_type == "AFTER":
        return target_id, source_id
    return source_id, target_id


def _timeline_span_label(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", label.upper()).strip("_")


def _resolve_interval(
    timex: TemporalExpression,
    *,
    document_reference: date | None,
    anchor_dates: Mapping[str, date],
    previous_interval_start: date | None,
) -> NormalizedInterval | None:
    if timex.timex_type == "SET":
        return None
    if timex.timex_type == "DURATION":
        if timex.metadata.get("history_duration") and document_reference is not None:
            start = _add_duration(
                document_reference,
                amount=-(timex.amount or 0),
                unit=timex.unit or "day",
            )
            return _interval(
                start,
                document_reference,
                uncertainty_days=timex.uncertainty_days,
            )
        return None
    if timex.direction == "none":
        return _absolute_interval(timex)
    if timex.direction == "same" and document_reference is not None:
        return _interval(document_reference, document_reference)
    if timex.direction in {"past", "future"} and document_reference is not None:
        amount = timex.amount or 0
        if timex.direction == "past":
            amount = -amount
        target = _add_duration(document_reference, amount=amount, unit=timex.unit)
        return _interval(target, target, uncertainty_days=timex.uncertainty_days)
    if timex.direction in {"after_previous", "before_previous"}:
        if previous_interval_start is None:
            return None
        amount = timex.amount or 0
        if timex.direction == "before_previous":
            amount = -amount
        target = _add_duration(previous_interval_start, amount=amount, unit=timex.unit)
        return _interval(target, target, uncertainty_days=timex.uncertainty_days)
    if timex.direction == "postop_day":
        anchor = _lookup_anchor(anchor_dates, "surgery")
        if anchor is None:
            return None
        target = _add_duration(anchor, amount=timex.amount or 0, unit="day")
        return _interval(target, target, uncertainty_days=timex.uncertainty_days)
    if timex.direction == "since":
        anchor = _lookup_anchor(anchor_dates, timex.anchor)
        if anchor is None or document_reference is None:
            return None
        return _interval(
            min(anchor, document_reference),
            max(anchor, document_reference),
            uncertainty_days=timex.uncertainty_days,
        )
    return None


def _absolute_interval(timex: TemporalExpression) -> NormalizedInterval | None:
    if timex.value is None:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", timex.value):
        value = date.fromisoformat(timex.value)
        return _interval(
            value,
            value,
            precision=str(timex.metadata.get("precision", "day")),
            uncertainty_days=timex.uncertainty_days,
        )
    if re.fullmatch(r"\d{4}-\d{2}", timex.value):
        year, month = (int(part) for part in timex.value.split("-"))
        start = date(year, month, 1)
        end = date(year, month, calendar.monthrange(year, month)[1])
        return _interval(
            start,
            end,
            precision="month",
            uncertainty_days=timex.uncertainty_days,
        )
    return None


def _interval(
    start: date,
    end: date,
    *,
    precision: str = "day",
    uncertainty_days: int = 0,
) -> NormalizedInterval:
    if end < start:
        start, end = end, start
    return NormalizedInterval(
        start=start,
        end=end,
        lower_bound=start - timedelta(days=uncertainty_days),
        upper_bound=end + timedelta(days=uncertainty_days),
        precision=precision,
        uncertainty_days=uncertainty_days,
    )


def _add_duration(value: date, *, amount: int, unit: str | None) -> date:
    if unit == "day":
        return value + timedelta(days=amount)
    if unit == "week":
        return value + timedelta(weeks=amount)
    if unit == "month":
        return _add_months(value, amount)
    if unit == "year":
        return _add_months(value, amount * 12)
    return value


def _add_months(value: date, amount: int) -> date:
    month_index = value.month - 1 + amount
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _relative_offset_days(timex: TemporalExpression) -> int | None:
    if timex.amount is None or timex.unit is None:
        return None
    days_per_unit = {
        "day": 1,
        "week": 7,
        "month": 30,
        "year": 365,
    }.get(timex.unit)
    if days_per_unit is None:
        return None
    amount = timex.amount * days_per_unit
    if timex.direction in {"past", "before_previous", "before_anchor"}:
        return -amount
    if timex.direction in {"future", "after_previous", "after_anchor", "postop_day"}:
        return amount
    if timex.direction == "same":
        return 0
    return None


def _reference_date_required(timex: TemporalExpression) -> bool:
    if timex.direction in _REFERENCE_DEPENDENT_DIRECTIONS:
        return True
    return bool(timex.metadata.get("history_duration"))


def _reference_date_provenance(
    *,
    document_reference: date | None,
    required: bool,
) -> dict[str, Any]:
    return {
        "required": required,
        "provided": document_reference is not None,
        "source": "user_supplied" if document_reference is not None else "not_supplied",
        "value": document_reference.isoformat() if document_reference else None,
    }


def _timeline_relations(
    events: Sequence[TimelineEvent],
    document_reference: date | None,
) -> list[TimelineRelation]:
    relations: list[TimelineRelation] = []
    for event in events:
        if event.reference_date_dependent:
            relations.extend(_document_reference_relations(event, document_reference))
        if event.interval is None and event.relation_anchor is not None:
            relations.append(
                TimelineRelation(
                    source_id=event.event_id,
                    target_id=f"anchor:{event.relation_anchor}",
                    relation=_anchor_relation(event),
                    evidence=event.timex.text,
                )
            )

    for left, right in combinations(events, 2):
        relation = _event_pair_relation(left, right)
        if relation != "unknown":
            relations.append(
                TimelineRelation(
                    source_id=left.event_id,
                    target_id=right.event_id,
                    relation=relation,
                    evidence=f"{left.timex.text} | {right.timex.text}",
                )
            )
    return relations


def _document_reference_relations(
    event: TimelineEvent,
    document_reference: date | None,
) -> list[TimelineRelation]:
    if event.interval is not None and document_reference is not None:
        relation = _interval_relation(
            event.interval.start,
            event.interval.end,
            document_reference,
            document_reference,
        )
    elif event.relative_offset_days is not None:
        if event.relative_offset_days < 0:
            relation = "before"
        elif event.relative_offset_days > 0:
            relation = "after"
        else:
            relation = "overlap"
    else:
        relation = "unknown"
    if relation == "unknown":
        return []
    return [
        TimelineRelation(
            source_id=event.event_id,
            target_id="document_reference",
            relation=relation,
            evidence=event.timex.text,
        )
    ]


def _event_pair_relation(
    left: TimelineEvent | None,
    right: TimelineEvent | None,
) -> TimelineRelationKind:
    if left is None or right is None:
        return "unknown"
    if left.interval is not None and right.interval is not None:
        return _interval_relation(
            left.interval.start,
            left.interval.end,
            right.interval.start,
            right.interval.end,
        )
    if left.relative_offset_days is not None and right.relative_offset_days is not None:
        if left.relative_offset_days < right.relative_offset_days:
            return "before"
        if left.relative_offset_days > right.relative_offset_days:
            return "after"
        return "overlap"
    return "unknown"


def _interval_relation(
    left_start: date,
    left_end: date,
    right_start: date,
    right_end: date,
) -> TimelineRelationKind:
    if left_end < right_start:
        return "before"
    if right_end < left_start:
        return "after"
    return "overlap"


def _anchor_relation(event: TimelineEvent) -> TimelineRelationKind:
    if event.timex.direction in {"before_anchor", "before_previous"}:
        return "before"
    if event.timex.direction in {"after_anchor", "after_previous", "postop_day"}:
        return "after"
    if event.timex.direction == "since":
        return "after"
    return "unknown"


def _update_anchor_dates(
    anchor_dates: dict[str, date],
    sentence_text: str,
    anchor_date: date,
) -> None:
    normalized = sentence_text.casefold()
    for anchor, terms in _ANCHOR_TERMS.items():
        if any(term in normalized for term in terms):
            anchor_dates[anchor] = anchor_date
            if anchor == "surgery":
                anchor_dates["operation"] = anchor_date
            if anchor == "admission":
                anchor_dates["last admission"] = anchor_date


def _lookup_anchor(anchor_dates: Mapping[str, date], anchor: str | None) -> date | None:
    if anchor is None:
        return None
    if anchor in anchor_dates:
        return anchor_dates[anchor]
    if anchor == "operation":
        return anchor_dates.get("surgery")
    if anchor == "last admission":
        return anchor_dates.get("admission")
    return None


def _event_sort_key(event: TimelineEvent) -> tuple[int, Any, int]:
    if event.interval is not None:
        return (0, event.interval.start, event.start)
    if event.relative_offset_days is not None:
        return (1, event.relative_offset_days, event.start)
    return (2, event.start, event.start)


def _coerce_date(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _coerce_document_creation_time(
    value: str | date | datetime,
) -> tuple[str, date]:
    if isinstance(value, datetime):
        return value.isoformat(), value.date()
    if isinstance(value, date):
        return value.isoformat(), value
    if not isinstance(value, str):
        raise TypeError(
            "document_creation_time must be an ISO string, date, or datetime"
        )
    normalized = value.strip()
    if not normalized:
        raise ValueError("document_creation_time must not be empty")
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
            parsed_date = date.fromisoformat(normalized)
            return parsed_date.isoformat(), parsed_date
        parsed_datetime = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "document_creation_time must be a valid ISO date or datetime"
        ) from exc
    return parsed_datetime.isoformat(), parsed_datetime.date()


def _coerce_source_span(
    raw_span: Mapping[str, object] | Sequence[int],
    *,
    text_length: int,
) -> tuple[int, int]:
    if isinstance(raw_span, Mapping):
        raw_start = raw_span.get("start", raw_span.get("start_char"))
        raw_end = raw_span.get("end", raw_span.get("end_char"))
    elif isinstance(raw_span, Sequence) and not isinstance(raw_span, (str, bytes)):
        if len(raw_span) != 2:
            raise ValueError("span sequences must contain exactly start and end")
        raw_start, raw_end = raw_span
    else:
        raise TypeError("spans must be mappings or two-item sequences")
    if raw_start is None or raw_end is None:
        raise ValueError("spans require integer start and end offsets")
    if isinstance(raw_start, bool) or isinstance(raw_end, bool):
        raise ValueError("spans require integer start and end offsets")
    try:
        start = int(str(raw_start))
        end = int(str(raw_end))
    except (TypeError, ValueError) as exc:
        raise ValueError("spans require integer start and end offsets") from exc
    if start < 0 or end <= start or end > text_length:
        raise ValueError("span offsets must satisfy 0 <= start < end <= len(text)")
    return start, end


def _event_anchor_timexes(
    text: str,
    timex_spans: Iterable[NormalizedTimex | Mapping[str, object] | Sequence[int]]
    | None,
    *,
    document_creation_time: str | date | datetime,
) -> tuple[NormalizedTimex, ...]:
    if timex_spans is None:
        offsets: list[Mapping[str, object] | Sequence[int]] = [
            (timex.start, timex.end) for timex in detect_timexes(text)
        ]
    else:
        offsets = []
        for timex in timex_spans:
            if isinstance(timex, NormalizedTimex):
                offsets.append((timex.start, timex.end))
            else:
                offsets.append(timex)
    records = normalize_temporal(text, offsets, document_creation_time)
    return tuple(
        record
        for record in records
        if record.timex_type in {"DATE", "TIME"} and record.value is not None
    )


def _nearest_event_timex(
    text: str,
    *,
    event_start: int,
    event_end: int,
    timexes: Sequence[NormalizedTimex],
    max_character_distance: int,
) -> NormalizedTimex | None:
    candidates: list[tuple[tuple[int, int, int, int], NormalizedTimex]] = []
    for timex in timexes:
        if event_end <= timex.start:
            distance = timex.start - event_end
            between = text[event_end : timex.start]
            follows_event = 1
        elif timex.end <= event_start:
            distance = event_start - timex.end
            between = text[timex.end : event_start]
            follows_event = 0
        else:
            distance = 0
            between = ""
            follows_event = 0
        if distance > max_character_distance or re.search(r"[.!?。！？\n]", between):
            continue
        rank = (distance, follows_event, timex.start, timex.end)
        candidates.append((rank, timex))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _normalized_timex_dct_position(
    timex: NormalizedTimex,
    dct: date,
) -> TimelineRelationKind:
    if timex.value is None:
        return "unknown"
    bounds = _normalized_value_date_bounds(timex.value)
    if bounds is None:
        return "unknown"
    start, end = bounds
    if "since" in timex.granularity_flags:
        end = max(end, dct)
    return _interval_relation(start, end, dct, dct)


def _normalized_value_date_bounds(value: str) -> tuple[date, date] | None:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        parsed = date.fromisoformat(value)
        return parsed, parsed
    if match := re.fullmatch(r"(?P<year>\d{4})-(?P<month>\d{2})", value):
        year = int(match.group("year"))
        month = int(match.group("month"))
        start = date(year, month, 1)
        end = date(year, month, calendar.monthrange(year, month)[1])
        return start, end
    if match := re.fullmatch(r"(?P<year>\d{4})-W(?P<week>\d{2})", value):
        start = date.fromisocalendar(
            int(match.group("year")),
            int(match.group("week")),
            1,
        )
        return start, start + timedelta(days=6)
    if re.fullmatch(r"\d{4}", value):
        year = int(value)
        return date(year, 1, 1), date(year, 12, 31)
    try:
        parsed_datetime = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    parsed_date = parsed_datetime.date()
    return parsed_date, parsed_date


def _sentence_windows(text: str) -> tuple[tuple[int, int], ...]:
    if not text:
        return ()
    windows: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?]+(?:\s+|$)|\n+", text):
        _append_sentence_window(windows, text, start, match.end())
        start = match.end()
    _append_sentence_window(windows, text, start, len(text))
    return tuple(windows)


def _append_sentence_window(
    windows: list[tuple[int, int]],
    text: str,
    start: int,
    end: int,
) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        windows.append((start, end))


def _sentence_for_offsets(
    windows: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int]:
    for sentence_start, sentence_end in windows:
        if start >= sentence_start and end <= sentence_end:
            return sentence_start, sentence_end
    return 0, end


def _load_gold_cases(
    cases_or_path: str | Path | Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if isinstance(cases_or_path, (str, Path)):
        raw = json.loads(Path(cases_or_path).read_text(encoding="utf-8"))
        if raw.get("synthetic") is not True:
            raise ValueError("timeline gold corpus must be marked synthetic")
        cases = raw.get("cases")
        if not isinstance(cases, list):
            raise ValueError("timeline gold corpus must contain a cases list")
        return cases
    return list(cases_or_path)


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


__all__ = [
    "EVENT_ANCHORING_ADVISORY",
    "EventAnchorSource",
    "EventAnchoringResult",
    "EventTemporalAnchor",
    "NormalizedInterval",
    "ORDER_EVENTS_SCHEMA_VERSION",
    "OrderedTimelineEvent",
    "ResolvedTimeline",
    "TIMELINE_ASSISTIVE_DISCLAIMER",
    "Timeline",
    "TimelineEdgeProvenance",
    "TimelineEdgeStatus",
    "TimelineEvaluationResult",
    "TimelineEvent",
    "TimelineRelation",
    "TimelineRelationKind",
    "TimexAnchorReference",
    "anchor_events",
    "evaluate_timeline_gold",
    "order_events",
    "resolve_timeline",
]
