"""Privacy-safe evidence-linked clinical timeline graphs.

The graph is a small, deterministic composition layer for callers that already
have event spans, assertion context, and temporal evidence.  Source text may
be supplied while building the graph, but it is used only in memory to derive
content hashes.  Graph records never retain source text or arbitrary caller
metadata.
"""

from __future__ import annotations

import heapq
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, Literal, cast

from openmed.clinical.context import (
    AFFIRMED,
    CERTAIN,
    CERTAINTY_VALUES,
    NEGATION_VALUES,
    RECENT,
    TEMPORALITY_VALUES,
    ClinicalAssertion,
)
from openmed.core.audit import hash_text

TimelineRelationKind = Literal["before", "after", "overlap"]

TIMELINE_GRAPH_SCHEMA_VERSION = 1
TIMELINE_GRAPH_ADVISORY = (
    "Evidence-linked clinical timelines are deterministic assistive records "
    "for review and are not a clinical decision, diagnosis, treatment "
    "recommendation, or substitute for clinician verification."
)

_HASH_PREFIXES = ("sha256:", "hmac-sha256:")
_RELATION_ALIASES = {
    "before": "before",
    "precedes": "before",
    "preceded_by": "after",
    "after": "after",
    "follows": "after",
    "overlap": "overlap",
    "overlaps": "overlap",
}
_TIMEX_TYPES = frozenset({"DATE", "TIME", "DURATION", "SET"})


class TimelineGraphCycleError(ValueError):
    """Raised when temporal precedence links contain a directed cycle."""


@dataclass(frozen=True)
class TimelineEvidence:
    """A source span supporting an event's normalized temporal value.

    ``start`` and ``end`` are inclusive/exclusive offsets into the source
    document.  ``text_hash`` is optional for callers that already provide
    offsets without source text; when source text is available, the builder
    fills it with a SHA-256 fingerprint.  The surface text is intentionally
    not a field on this record.
    """

    start: int
    end: int
    normalized_value: str | None = None
    text_hash: str | None = None
    timex_type: str | None = None
    relation: str = "temporal_anchor"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        _validate_offset(self.start, self.end, "evidence")
        if self.normalized_value is not None:
            if not isinstance(self.normalized_value, str):
                raise TypeError("evidence normalized value must be a string")
            normalized_value = self.normalized_value.strip()
            if not normalized_value:
                raise ValueError("evidence normalized value must not be empty")
            object.__setattr__(self, "normalized_value", normalized_value)
        object.__setattr__(self, "text_hash", _validate_hash(self.text_hash))

        timex_type = self.timex_type
        if timex_type is not None:
            if not isinstance(timex_type, str) or not timex_type.strip():
                raise TypeError("evidence TIMEX type must be a non-empty string")
            normalized_type = timex_type.strip().upper()
            if normalized_type not in _TIMEX_TYPES:
                raise ValueError("unsupported evidence TIMEX type")
            object.__setattr__(self, "timex_type", normalized_type)

        if not isinstance(self.relation, str) or not self.relation.strip():
            raise TypeError("evidence relation must be a non-empty string")
        object.__setattr__(self, "relation", self.relation.strip().casefold())
        _validate_confidence(self.confidence, "evidence")
        object.__setattr__(self, "confidence", float(self.confidence))

    @property
    def source_offsets(self) -> tuple[int, int]:
        """Return the inclusive/exclusive source offsets."""

        return (self.start, self.end)

    @property
    def value(self) -> str | None:
        """Return the normalized temporal value."""

        return self.normalized_value

    @property
    def type(self) -> str | None:
        """Return the normalized TIMEX type."""

        return self.timex_type

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record without source text."""

        return {
            "span": [self.start, self.end],
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
            "normalized_value": self.normalized_value,
            "type": self.timex_type,
            "relation": self.relation,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TimelineGraphEvent:
    """One typed clinical event with assertion and temporal provenance."""

    event_id: str
    event_type: str
    start: int
    end: int
    timestamp: str | None = None
    assertion: ClinicalAssertion = field(
        default_factory=lambda: ClinicalAssertion(
            temporality=RECENT,
            certainty=CERTAIN,
            negation=AFFIRMED,
        )
    )
    temporal_evidence: tuple[TimelineEvidence, ...] = ()
    text_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise TypeError("event id must be a non-empty string")
        if not isinstance(self.event_type, str) or not self.event_type.strip():
            raise TypeError("event type must be a non-empty string")
        object.__setattr__(self, "event_id", self.event_id.strip())
        object.__setattr__(self, "event_type", self.event_type.strip().casefold())
        _validate_offset(self.start, self.end, "event")
        object.__setattr__(
            self,
            "timestamp",
            _normalize_temporal_value(self.timestamp),
        )
        object.__setattr__(self, "assertion", _coerce_assertion(self.assertion))
        if self.temporal_evidence is None:
            evidence = ()
        elif isinstance(
            self.temporal_evidence, (TimelineEvidence, Mapping)
        ) or _looks_like_timex(self.temporal_evidence):
            evidence = (_coerce_evidence(self.temporal_evidence),)
        else:
            evidence = tuple(_coerce_evidence(item) for item in self.temporal_evidence)
        object.__setattr__(
            self,
            "temporal_evidence",
            tuple(sorted(evidence, key=_evidence_sort_key)),
        )
        if self.timestamp is None and evidence:
            object.__setattr__(self, "timestamp", evidence[0].normalized_value)
        object.__setattr__(self, "text_hash", _validate_hash(self.text_hash))

    @property
    def id(self) -> str:
        """Return the stable event identifier."""

        return self.event_id

    @property
    def type(self) -> str:
        """Return the canonical event type."""

        return self.event_type

    @property
    def source_offsets(self) -> tuple[int, int]:
        """Return the event's inclusive/exclusive source offsets."""

        return (self.start, self.end)

    @property
    def source_span(self) -> tuple[int, int]:
        """Alias for :attr:`source_offsets`."""

        return self.source_offsets

    @property
    def event_time(self) -> str | None:
        """Return the normalized event timestamp, when available."""

        return self.timestamp

    @property
    def normalized_time(self) -> str | None:
        """Alias for :attr:`event_time`."""

        return self.timestamp

    @property
    def assertion_context(self) -> ClinicalAssertion:
        """Return assertion axes attached to this event."""

        return self.assertion

    @property
    def evidence(self) -> tuple[TimelineEvidence, ...]:
        """Return temporal evidence links attached to this event."""

        return self.temporal_evidence

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready event without raw source text."""

        return {
            "id": self.event_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "start": self.start,
            "end": self.end,
            "source_offsets": [self.start, self.end],
            "text_hash": self.text_hash,
            "timestamp": self.timestamp,
            "assertion": self.assertion.to_dict(),
            "temporal_evidence": [
                evidence.to_dict() for evidence in self.temporal_evidence
            ],
        }


@dataclass(frozen=True)
class TimelineTemporalLink:
    """A typed temporal relation between two graph events."""

    source_id: str
    target_id: str
    relation: TimelineRelationKind
    evidence: tuple[TimelineEvidence, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise TypeError("temporal link source id must be a non-empty string")
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise TypeError("temporal link target id must be a non-empty string")
        object.__setattr__(self, "source_id", self.source_id.strip())
        object.__setattr__(self, "target_id", self.target_id.strip())
        object.__setattr__(self, "relation", _normalize_relation(self.relation))
        if self.evidence is None:
            evidence = ()
        elif isinstance(
            self.evidence, (TimelineEvidence, Mapping)
        ) or _looks_like_timex(self.evidence):
            evidence = (_coerce_evidence(self.evidence),)
        else:
            evidence = tuple(_coerce_evidence(item) for item in self.evidence)
        object.__setattr__(
            self,
            "evidence",
            tuple(sorted(evidence, key=_evidence_sort_key)),
        )
        _validate_confidence(self.confidence, "temporal link")
        object.__setattr__(self, "confidence", float(self.confidence))

    @property
    def relation_type(self) -> TimelineRelationKind:
        """Return the typed relation name."""

        return self.relation

    @property
    def source(self) -> str:
        """Return the source event id."""

        return self.source_id

    @property
    def target(self) -> str:
        """Return the target event id."""

        return self.target_id

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready temporal link without raw source text."""

        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "confidence": self.confidence,
            "evidence": [evidence.to_dict() for evidence in self.evidence],
        }


@dataclass(frozen=True)
class TimelineGraph:
    """A cycle-free, deterministically ordered clinical timeline graph."""

    events: tuple[TimelineGraphEvent, ...]
    temporal_links: tuple[TimelineTemporalLink, ...] = ()
    disclaimer: str = TIMELINE_GRAPH_ADVISORY
    schema_version: int = TIMELINE_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        events = tuple(_coerce_event(event) for event in self.events)
        links = tuple(_coerce_link(link) for link in self.temporal_links)
        if len({event.event_id for event in events}) != len(events):
            raise ValueError("timeline event ids must be unique")
        event_ids = {event.event_id for event in events}
        if any(
            link.source_id not in event_ids or link.target_id not in event_ids
            for link in links
        ):
            raise ValueError("temporal links must reference timeline events")

        ordered_events = _ordered_events(events, links)
        ordered_links = tuple(sorted(links, key=_link_sort_key))
        object.__setattr__(self, "events", ordered_events)
        object.__setattr__(self, "temporal_links", ordered_links)
        if self.schema_version != TIMELINE_GRAPH_SCHEMA_VERSION:
            raise ValueError("unsupported timeline graph schema version")

    @property
    def links(self) -> tuple[TimelineTemporalLink, ...]:
        """Return temporal links using the shorter graph terminology."""

        return self.temporal_links

    @property
    def ordered_events(self) -> tuple[TimelineGraphEvent, ...]:
        """Return events in deterministic temporal/topological order."""

        return self.events

    @property
    def ordered_event_ids(self) -> tuple[str, ...]:
        """Return ordered event ids for compact consumers."""

        return tuple(event.event_id for event in self.events)

    @property
    def is_cycle_free(self) -> bool:
        """Return whether precedence links are acyclic."""

        return True

    @property
    def cycle_free(self) -> bool:
        """Alias for :attr:`is_cycle_free`."""

        return self.is_cycle_free

    def event(self, event_id: str) -> TimelineGraphEvent:
        """Return one event by id without exposing source text in errors."""

        for event in self.events:
            if event.event_id == event_id:
                return event
        raise KeyError("timeline event was not found")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, privacy-safe graph payload."""

        return {
            "schema_version": self.schema_version,
            "events": [event.to_dict() for event in self.events],
            "temporal_links": [link.to_dict() for link in self.temporal_links],
            "ordered_event_ids": list(self.ordered_event_ids),
            "cycle_free": self.is_cycle_free,
            "disclaimer": self.disclaimer,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Return deterministic JSON for local storage or transport."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
        )


def build_timeline_graph(
    events: Iterable[TimelineGraphEvent | Mapping[str, Any]],
    temporal_links: Iterable[TimelineTemporalLink | Mapping[str, Any]] | None = None,
    *,
    links: Iterable[TimelineTemporalLink | Mapping[str, Any]] | None = None,
    document_text: str | None = None,
    text: str | None = None,
) -> TimelineGraph:
    """Build a deterministic graph from typed event and temporal records.

    Args:
        events: Event records or mappings. Mappings may use ``id``/``event_id``,
            ``type``/``event_type``, ``start`` + ``end`` or
            ``source_offsets``, and ``timestamp``/``event_time``. Assertion
            axes can be nested under ``assertion`` or supplied at the top
            level. A mapping's source ``text`` is hashed and discarded.
        temporal_links: Optional typed links. ``links=`` is an alias for
            callers that use graph terminology.
        document_text: Optional source document used only to hash event and
            evidence spans that do not already provide a hash. ``text=`` is an
            alias for this argument.

    Returns:
        A cycle-free graph whose events are topologically ordered and then
        tie-broken by timestamp, source offsets, event type, and event id.

    Raises:
        TimelineGraphCycleError: If ``before``/``after`` links form a cycle.
        ValueError: If a link references an unknown event or an input span is
            invalid.

    The function is pure and local. It does not use the wall clock, network,
    environment state, or a model service.
    """

    source_text = _coerce_document_text(document_text, text)
    if isinstance(events, (Mapping, TimelineGraphEvent)):
        event_values = (events,)
    else:
        event_values = events
    event_items = tuple(
        _coerce_event(item, document_text=source_text) for item in event_values
    )
    link_items = temporal_links if links is None else links
    if temporal_links is not None and links is not None:
        raise ValueError("provide temporal links through only one argument")
    if isinstance(link_items, (Mapping, TimelineTemporalLink)) or (
        link_items is not None and not isinstance(link_items, Iterable)
    ):
        link_values = (link_items,)
    else:
        link_values = link_items or ()
    link_records = tuple(_coerce_link(item) for item in link_values)
    return TimelineGraph(events=event_items, temporal_links=link_records)


def _coerce_document_text(
    document_text: str | None,
    text: str | None,
) -> str | None:
    if document_text is not None and text is not None:
        raise ValueError("provide source text through only one argument")
    source_text = document_text if document_text is not None else text
    if source_text is not None and not isinstance(source_text, str):
        raise TypeError("source text must be a string")
    return source_text


def _coerce_event(
    event: TimelineGraphEvent | Mapping[str, Any],
    *,
    document_text: str | None = None,
) -> TimelineGraphEvent:
    if isinstance(event, TimelineGraphEvent):
        if event.text_hash is None and document_text is not None:
            return replace(
                event, text_hash=hash_text(document_text[event.start : event.end])
            )
        return event
    if not isinstance(event, Mapping):
        raise TypeError("timeline events must be event records or mappings")

    start, end = _mapping_offset(event, "event")
    event_type = _mapping_string(
        event,
        ("event_type", "type", "label"),
        default="event",
    )
    event_id = _mapping_string(event, ("event_id", "id", "node_id"))
    if event_id is None:
        event_id = f"event-{start}-{end}-{event_type.casefold()}"
    raw_text = _mapping_optional_string(event, ("text", "surface"))
    text_hash = _mapping_optional_string(event, ("text_hash", "content_hash"))
    if text_hash is None:
        source_surface = (
            document_text[start:end] if document_text is not None else raw_text
        )
        if source_surface is not None:
            text_hash = hash_text(source_surface)

    timestamp = _mapping_value(
        event,
        ("timestamp", "event_time", "normalized_time", "date", "time"),
    )
    assertion_value = _mapping_value(event, ("assertion", "assertion_context"))
    if assertion_value is None:
        possible_axes = {
            key: event[key]
            for key in ("temporality", "certainty", "negation", "experiencer")
            if key in event
        }
        assertion_value = possible_axes or None
    evidence_value = _mapping_value(
        event,
        ("temporal_evidence", "evidence", "timex_evidence", "timex"),
    )
    evidence_items = _coerce_evidence_collection(
        evidence_value,
        document_text=document_text,
    )
    return TimelineGraphEvent(
        event_id=event_id,
        event_type=event_type,
        start=start,
        end=end,
        timestamp=timestamp,
        assertion=_coerce_assertion(assertion_value),
        temporal_evidence=evidence_items,
        text_hash=text_hash,
    )


def _coerce_link(
    link: TimelineTemporalLink | Mapping[str, Any] | Any,
) -> TimelineTemporalLink:
    if isinstance(link, TimelineTemporalLink):
        return link

    relation_type = getattr(link, "relation_type", getattr(link, "relation", None))
    source_id = getattr(link, "source_id", None)
    target_id = getattr(link, "target_id", None)
    if source_id is None:
        source = getattr(link, "source", None)
        source_id = getattr(source, "span_id", getattr(source, "event_id", None))
    if target_id is None:
        target = getattr(link, "target", None)
        target_id = getattr(target, "span_id", getattr(target, "event_id", None))
    confidence = getattr(link, "confidence", 1.0)
    evidence_value = getattr(link, "evidence", None)
    cue = getattr(link, "cue", None)
    if evidence_value is None and cue is not None:
        cue_start = getattr(cue, "start", None)
        cue_end = getattr(cue, "end", None)
        if cue_start is not None and cue_end is not None:
            evidence_value = {
                "start": cue_start,
                "end": cue_end,
                "text_hash": getattr(cue, "text_hash", None),
                "type": None,
                "relation": getattr(cue, "category", "temporal_anchor"),
            }

    if source_id is None and target_id is None and isinstance(link, Mapping):
        source_id = _mapping_value(link, ("source_id", "source", "from"))
        target_id = _mapping_value(link, ("target_id", "target", "to"))
        relation_type = _mapping_value(link, ("relation", "relation_type", "type"))
        confidence = link.get("confidence", 1.0)
        evidence_value = _mapping_value(
            link,
            ("evidence", "temporal_evidence", "cue"),
        )
        if evidence_value is None and "evidence_start" in link:
            evidence_value = {
                "start": link.get("evidence_start"),
                "end": link.get("evidence_end"),
                "text_hash": link.get("evidence_hash"),
                "normalized_value": link.get("evidence_value"),
            }

    if source_id is None or target_id is None or relation_type is None:
        raise TypeError("temporal links must provide source, target, and relation")
    evidence = _coerce_evidence_collection(evidence_value)
    return TimelineTemporalLink(
        source_id=_coerce_identifier(source_id, "temporal link source id"),
        target_id=_coerce_identifier(target_id, "temporal link target id"),
        relation=cast(TimelineRelationKind, _normalize_relation(relation_type)),
        evidence=evidence,
        confidence=confidence,
    )


def _coerce_evidence_collection(
    value: Any,
    *,
    document_text: str | None = None,
) -> tuple[TimelineEvidence, ...]:
    if value is None:
        return ()
    if isinstance(value, (TimelineEvidence, Mapping)) or _looks_like_timex(value):
        values = (value,)
    elif isinstance(value, str):
        # A raw evidence string has no safe offset representation. Do not
        # copy it into an exception or report; callers should provide a span.
        return ()
    else:
        try:
            values = tuple(value)
        except TypeError as error:
            raise TypeError("temporal evidence must be a record or iterable") from error
    return tuple(_coerce_evidence(item, document_text=document_text) for item in values)


def _coerce_evidence(
    value: TimelineEvidence | Mapping[str, Any] | Any,
    *,
    document_text: str | None = None,
) -> TimelineEvidence:
    if isinstance(value, TimelineEvidence):
        if value.text_hash is None and document_text is not None:
            return replace(
                value,
                text_hash=hash_text(document_text[value.start : value.end]),
            )
        return value

    if _looks_like_timex(value):
        start = getattr(value, "start")
        end = getattr(value, "end")
        normalized_value = getattr(value, "value", None)
        timex_type = getattr(value, "timex_type", getattr(value, "type", None))
        raw_text = getattr(value, "text", None)
        text_hash = hash_text(raw_text) if isinstance(raw_text, str) else None
        return TimelineEvidence(
            start=start,
            end=end,
            normalized_value=normalized_value,
            text_hash=text_hash,
            timex_type=timex_type,
        )

    if not isinstance(value, Mapping):
        raise TypeError("temporal evidence must be evidence records")
    start, end = _mapping_offset(value, "evidence")
    raw_text = _mapping_optional_string(value, ("text", "surface"))
    text_hash = _mapping_optional_string(value, ("text_hash", "content_hash"))
    if text_hash is None:
        source_surface = (
            document_text[start:end] if document_text is not None else raw_text
        )
        if source_surface is not None:
            text_hash = hash_text(source_surface)
    normalized_value = _mapping_value(
        value,
        ("normalized_value", "value", "timestamp", "time"),
    )
    timex_type = _mapping_value(value, ("timex_type", "type"))
    relation = _mapping_string(
        value,
        ("relation", "kind"),
        default="temporal_anchor",
    )
    confidence = value.get("confidence", 1.0)
    return TimelineEvidence(
        start=start,
        end=end,
        normalized_value=_optional_temporal_string(normalized_value),
        text_hash=text_hash,
        timex_type=_optional_temporal_string(timex_type),
        relation=relation,
        confidence=confidence,
    )


def _coerce_assertion(value: Any) -> ClinicalAssertion:
    if value is None:
        return ClinicalAssertion(
            temporality=RECENT,
            certainty=CERTAIN,
            negation=AFFIRMED,
        )
    nested = getattr(value, "assertion", None)
    if isinstance(nested, ClinicalAssertion):
        value = nested
    if isinstance(value, ClinicalAssertion):
        axes = {
            "temporality": value.temporality,
            "certainty": value.certainty,
            "negation": value.negation,
            "experiencer": value.experiencer,
        }
    elif isinstance(value, Mapping):
        axes = {
            "temporality": value.get("temporality", RECENT),
            "certainty": value.get("certainty", CERTAIN),
            "negation": value.get("negation", AFFIRMED),
            "experiencer": value.get("experiencer"),
        }
    else:
        raise TypeError("assertion context must be a ClinicalAssertion or mapping")

    temporality = _axis_string(axes["temporality"], "temporality")
    certainty = _axis_string(axes["certainty"], "certainty")
    negation = (
        None if axes["negation"] is None else _axis_string(axes["negation"], "negation")
    )
    if temporality not in TEMPORALITY_VALUES:
        raise ValueError("unsupported assertion temporality")
    if certainty not in CERTAINTY_VALUES:
        raise ValueError("unsupported assertion certainty")
    if negation is not None and negation not in NEGATION_VALUES:
        raise ValueError("unsupported assertion negation")
    experiencer = axes["experiencer"]
    if experiencer is not None:
        experiencer = _axis_string(experiencer, "experiencer")
    return ClinicalAssertion(
        temporality=temporality,
        certainty=cast(Any, certainty),
        negation=cast(Any, negation),
        experiencer=experiencer,
    )


def _ordered_events(
    events: Sequence[TimelineGraphEvent],
    links: Sequence[TimelineTemporalLink],
) -> tuple[TimelineGraphEvent, ...]:
    by_id = {event.event_id: event for event in events}
    adjacency: dict[str, set[str]] = {event.event_id: set() for event in events}
    indegree = {event.event_id: 0 for event in events}
    for link in links:
        source, target = _precedence_endpoints(link)
        if source is None or target is None or target in adjacency[source]:
            continue
        adjacency[source].add(target)
        indegree[target] += 1

    ready: list[tuple[tuple[Any, ...], str]] = []
    for event_id, degree in indegree.items():
        if degree == 0:
            heapq.heappush(ready, (_event_sort_key(by_id[event_id]), event_id))

    ordered_ids: list[str] = []
    while ready:
        _, event_id = heapq.heappop(ready)
        ordered_ids.append(event_id)
        for target_id in sorted(adjacency[event_id]):
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                heapq.heappush(
                    ready,
                    (_event_sort_key(by_id[target_id]), target_id),
                )

    if len(ordered_ids) != len(events):
        raise TimelineGraphCycleError("temporal precedence links contain a cycle")
    return tuple(by_id[event_id] for event_id in ordered_ids)


def _precedence_endpoints(
    link: TimelineTemporalLink,
) -> tuple[str | None, str | None]:
    if link.relation == "before":
        return link.source_id, link.target_id
    if link.relation == "after":
        return link.target_id, link.source_id
    return None, None


def _event_sort_key(event: TimelineGraphEvent) -> tuple[Any, ...]:
    timestamp_missing, timestamp_value = _timestamp_sort_key(event.timestamp)
    return (
        timestamp_missing,
        timestamp_value,
        event.start,
        event.end,
        event.event_type,
        event.event_id,
    )


def _link_sort_key(link: TimelineTemporalLink) -> tuple[Any, ...]:
    evidence_key = tuple(_evidence_sort_key(item) for item in link.evidence)
    return (
        link.source_id,
        link.target_id,
        link.relation,
        evidence_key,
        link.confidence,
    )


def _evidence_sort_key(evidence: TimelineEvidence) -> tuple[Any, ...]:
    return (
        evidence.start,
        evidence.end,
        evidence.normalized_value or "",
        evidence.timex_type or "",
        evidence.relation,
        evidence.text_hash or "",
    )


def _timestamp_sort_key(value: str | None) -> tuple[int, str]:
    if value is None:
        return (1, "")
    candidate = value.split("/", 1)[0]
    try:
        if len(candidate) == 10:
            return (0, date.fromisoformat(candidate).isoformat())
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        return (0, parsed.isoformat())
    except ValueError:
        return (0, candidate)


def _normalize_temporal_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value") and not isinstance(value, (str, date, datetime)):
        value = getattr(value, "value")
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise TypeError("temporal value must be a string, date, or datetime")
    normalized = value.strip()
    if not normalized:
        raise ValueError("temporal value must not be empty")
    return normalized


def _optional_temporal_string(value: Any) -> str | None:
    if value is None:
        return None
    return _normalize_temporal_value(value)


def _normalize_relation(value: Any) -> TimelineRelationKind:
    if not isinstance(value, str):
        raise TypeError("temporal relation must be a string")
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    normalized = _RELATION_ALIASES.get(normalized, normalized)
    if normalized not in {"before", "after", "overlap"}:
        raise ValueError("unsupported temporal relation")
    return cast(TimelineRelationKind, normalized)


def _validate_offset(start: Any, end: Any, label: str) -> None:
    if not isinstance(start, int) or isinstance(start, bool):
        raise TypeError(f"{label} start offset must be an integer")
    if not isinstance(end, int) or isinstance(end, bool):
        raise TypeError(f"{label} end offset must be an integer")
    if start < 0 or end <= start:
        raise ValueError(f"{label} offsets must satisfy 0 <= start < end")


def _validate_confidence(value: Any, label: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{label} confidence must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} confidence must be numeric") from error
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{label} confidence must be between 0 and 1")


def _validate_hash(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("text hash must be a string")
    normalized = value.strip()
    if not normalized.startswith(_HASH_PREFIXES):
        raise ValueError("text hash must be a SHA-256 or HMAC-SHA-256 value")
    return normalized


def _mapping_offset(mapping: Mapping[str, Any], label: str) -> tuple[int, int]:
    raw_offset = mapping.get("source_offsets", mapping.get("offset"))
    if raw_offset is None:
        raw_offset = mapping.get("source_span")
    if raw_offset is not None:
        if (
            not isinstance(raw_offset, Sequence)
            or isinstance(raw_offset, str)
            or len(raw_offset) != 2
        ):
            raise TypeError(f"{label} offsets must be a two-item sequence")
        start, end = raw_offset
    else:
        start, end = mapping.get("start"), mapping.get("end")
    _validate_offset(start, end, label)
    return cast(int, start), cast(int, end)


def _mapping_value(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _mapping_string(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
    *,
    default: str | None = None,
) -> str | None:
    value = _mapping_value(mapping, keys)
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise TypeError("timeline string field must be non-empty")
    return value.strip()


def _mapping_optional_string(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
) -> str | None:
    value = _mapping_value(mapping, keys)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("timeline text or hash field must be a string")
    normalized = value.strip()
    return normalized or None


def _coerce_identifier(value: Any, label: str) -> str:
    if isinstance(value, Mapping):
        value = _mapping_value(value, ("event_id", "id", "span_id"))
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string")
    return value.strip()


def _axis_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"assertion {label} must be a non-empty string")
    return value.strip().casefold()


def _looks_like_timex(value: Any) -> bool:
    return all(hasattr(value, attribute) for attribute in ("start", "end")) and (
        hasattr(value, "value") or hasattr(value, "normalized_value")
    )


# Names used by callers that prefer the domain-specific terms over the graph
# implementation names.  They intentionally point at the same immutable
# records so serialized schemas stay singular and stable.
AssertionContext = ClinicalAssertion
TimelineAssertionContext = ClinicalAssertion
TemporalEvidence = TimelineEvidence
TemporalEvidenceLink = TimelineEvidence
TimelineEvent = TimelineGraphEvent
EvidenceLinkedTimelineEvent = TimelineGraphEvent
TimelineEdge = TimelineTemporalLink
TemporalLink = TimelineTemporalLink
create_timeline_graph = build_timeline_graph
build_evidence_linked_timeline = build_timeline_graph


__all__ = [
    "AssertionContext",
    "EvidenceLinkedTimelineEvent",
    "TIMELINE_GRAPH_ADVISORY",
    "TIMELINE_GRAPH_SCHEMA_VERSION",
    "TemporalEvidence",
    "TemporalEvidenceLink",
    "TemporalLink",
    "TimelineAssertionContext",
    "TimelineEdge",
    "TimelineEvent",
    "TimelineEvidence",
    "TimelineGraph",
    "TimelineGraphCycleError",
    "TimelineGraphEvent",
    "TimelineRelationKind",
    "TimelineTemporalLink",
    "build_evidence_linked_timeline",
    "build_timeline_graph",
    "create_timeline_graph",
]
