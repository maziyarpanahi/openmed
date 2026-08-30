"""Assemble privacy-safe clinical events into an ordered timeline.

The assembler is deliberately a hand-off layer.  It does not extract entity
mentions, normalize temporal expressions, or infer assertions.  Those inputs
are supplied by the clinical extraction layers and are joined here by stable
ids and source offsets.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import Any

from openmed.clinical.context import (
    AFFIRMED,
    CERTAIN,
    HYPOTHETICAL,
    NEGATED,
    RECENT,
    ClinicalAssertion,
)
from openmed.clinical.timeline.resolver import Timeline as ResolvedTimeline

SpanOffset = tuple[int, int]

CLINICAL_EVENT_TIMELINE_SCHEMA_VERSION = 1
CLINICAL_EVENT_TIMELINE_ADVISORY = (
    "Clinical event timelines are deterministic assistive annotations for "
    "review and downstream organization, not a clinical decision, diagnosis, "
    "treatment recommendation, or substitute for clinician verification."
)

_DEFAULT_EVENT_KIND = "observation"
_MISSING = object()
_WHITESPACE_RE = re.compile(r"\s+")
_DATE_PREFIX_RE = re.compile(
    r"^(?P<year>\d{4})(?:-(?P<month>\d{2}))?(?:-(?P<day>\d{2}))?"
)
_PROVENANCE_KEYS = frozenset(
    {
        "coreference_chain",
        "entity_label",
        "event_kinds",
        "input_index",
        "mention_count",
        "source_span",
        "source_spans",
    }
)


@dataclass(frozen=True)
class ClinicalEvent:
    """One normalized clinical event without persisted source text.

    ``entity`` is the caller's stable entity or concept identifier.  A source
    surface string is never stored as a separate field; ``source_span`` and
    the semantic ``label`` in provenance are sufficient for review joins.
    """

    entity: str
    event_kind: str
    normalized_time: str | None
    section: str | None
    assertion: ClinicalAssertion
    source_span: SpanOffset
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        entity = _clean_label(self.entity, field_name="entity")
        event_kind = _clean_label(self.event_kind, field_name="event_kind")
        section = _clean_optional_label(self.section)
        normalized_time = _normalize_time_value(self.normalized_time)
        assertion = _coerce_assertion(self.assertion)
        source_span = _coerce_offset(self.source_span)
        object.__setattr__(self, "entity", entity)
        object.__setattr__(self, "event_kind", event_kind)
        object.__setattr__(self, "section", section)
        object.__setattr__(self, "normalized_time", normalized_time)
        object.__setattr__(self, "assertion", assertion)
        object.__setattr__(self, "source_span", source_span)
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(_safe_provenance(self.provenance)),
        )

    @property
    def start(self) -> int:
        """Return the inclusive source offset."""

        return self.source_span[0]

    @property
    def end(self) -> int:
        """Return the exclusive source offset."""

        return self.source_span[1]

    @property
    def span(self) -> SpanOffset:
        """Return the source offset under the common ``span`` alias."""

        return self.source_span

    @property
    def source_offsets(self) -> SpanOffset:
        """Return source offsets without exposing source text."""

        return self.source_span

    @property
    def label(self) -> str:
        """Return the semantic entity label, when supplied."""

        value = self.provenance.get("entity_label")
        return str(value) if value is not None else self.entity

    @property
    def entity_id(self) -> str:
        """Return the canonical entity identity used for deduplication."""

        return self.entity

    @property
    def event_type(self) -> str:
        """Return ``event_kind`` under the event-frame naming convention."""

        return self.event_kind

    @property
    def time(self) -> str | None:
        """Return the normalized timestamp under the concise alias."""

        return self.normalized_time

    @property
    def normalized_timestamp(self) -> str | None:
        """Return ``normalized_time`` under the timestamp alias."""

        return self.normalized_time

    @property
    def offset(self) -> SpanOffset:
        """Return the source span under the offset alias."""

        return self.source_span

    @property
    def is_anchored(self) -> bool:
        """Return whether ``normalized_time`` is an absolute sortable value."""

        return _anchor_key(self.normalized_time) is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready event containing no raw source text."""

        return {
            "entity": self.entity,
            "label": self.label,
            "event_kind": self.event_kind,
            "normalized_time": self.normalized_time,
            "section": self.section,
            "assertion": self.assertion.to_dict(),
            "source_span": {"start": self.start, "end": self.end},
            "provenance": _json_safe_provenance(self.provenance),
        }


@dataclass(frozen=True)
class ClinicalEventTimeline(ResolvedTimeline):
    """Chronologically ordered :class:`ClinicalEvent` records.

    The class is a timeline-compatible subtype of the existing temporal
    resolver result.  Its ``edges`` tuple is intentionally empty: this layer
    represents anchored order plus an explicit unanchored partial-order bucket,
    while graph TLINK decoding remains owned by ``order_events``.
    """

    events: tuple[ClinicalEvent, ...]
    edges: tuple[Any, ...] = ()
    disclaimer: str = CLINICAL_EVENT_TIMELINE_ADVISORY
    schema_version: int = CLINICAL_EVENT_TIMELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.document_creation_time is not None:
            raise ValueError(
                "assembled timelines do not accept a document_creation_time; "
                "supply normalized timestamps instead"
            )
        object.__setattr__(self, "events", tuple(self.events))
        if any(not isinstance(event, ClinicalEvent) for event in self.events):
            raise TypeError("clinical event timelines require ClinicalEvent records")

    @property
    def anchored_events(self) -> tuple[ClinicalEvent, ...]:
        """Return events with absolute sortable timestamps."""

        return tuple(event for event in self.events if event.is_anchored)

    @property
    def unanchored_events(self) -> tuple[ClinicalEvent, ...]:
        """Return the stable partial-order bucket for unanchored events."""

        return tuple(event for event in self.events if not event.is_anchored)

    @property
    def partial_order_events(self) -> tuple[ClinicalEvent, ...]:
        """Return unanchored events under the explicit partial-order alias."""

        return self.unanchored_events

    @property
    def partial_order_bucket(self) -> tuple[ClinicalEvent, ...]:
        """Return the unanchored partial-order bucket."""

        return self.unanchored_events

    @property
    def partial_order(self) -> tuple[ClinicalEvent, ...]:
        """Return the unanchored events as the partial-order view."""

        return self.unanchored_events

    @property
    def unanchored(self) -> tuple[ClinicalEvent, ...]:
        """Return unanchored events under the concise alias."""

        return self.unanchored_events

    @property
    def is_totally_ordered(self) -> bool:
        """Return whether all events have an absolute timestamp."""

        return not self.unanchored_events

    @property
    def provenance_offsets(self) -> tuple[SpanOffset, ...]:
        """Return one primary source offset for each ordered event."""

        return tuple(event.source_span for event in self.events)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready timeline without raw note text."""

        unanchored_indices = [
            index for index, event in enumerate(self.events) if not event.is_anchored
        ]
        return {
            "schema_version": self.schema_version,
            "events": [event.to_dict() for event in self.events],
            "anchored_event_indices": [
                index for index, event in enumerate(self.events) if event.is_anchored
            ],
            "unanchored_event_indices": unanchored_indices,
            "partial_order": {"unanchored_event_indices": unanchored_indices},
            "disclaimer": self.disclaimer,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Return deterministic JSON for the assembled timeline."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


# These aliases make the assembled result discoverable without changing the
# established ``Timeline`` class used by the temporal graph resolver.
AssembledTimeline = ClinicalEventTimeline
ClinicalTimeline = ClinicalEventTimeline
EventTimeline = ClinicalEventTimeline


@dataclass(frozen=True)
class _EntityRecord:
    input_index: int
    document_id: str | None
    mention_id: str
    entity_key: str
    label: str
    event_kind: str
    section: str | None
    source_span: SpanOffset
    direct_time: Any = None
    direct_assertion: Any = None

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(value for value in (self.mention_id, self.entity_key) if value)


@dataclass(frozen=True)
class _TimeRecord:
    input_index: int
    keys: tuple[str, ...]
    source_span: SpanOffset | None
    value: str | None
    unanchored: bool


@dataclass(frozen=True)
class _AssertionRecord:
    input_index: int
    keys: tuple[str, ...]
    source_span: SpanOffset | None
    assertion: ClinicalAssertion


def assemble_timeline(
    entities: Iterable[Any] | Mapping[Any, Any],
    normalized_times: Iterable[Any] | Mapping[Any, Any] | None = None,
    assertions: Iterable[Any] | Mapping[Any, Any] | None = None,
    chains: Any = None,
) -> ClinicalEventTimeline:
    """Assemble extracted entities, normalized times, and assertions.

    Args:
        entities: Entity mappings or span-like objects.  Records should carry
            ``start``/``end`` (or ``span``), a semantic label, and optionally a
            stable ``id``/``entity_id``, ``event_kind``, and ``section``.
        normalized_times: Normalized TIMEX records, an id-keyed mapping, or an
            offset-keyed iterable.  Absolute ISO dates are chronologically
            anchored; relative or unresolved values remain in the partial
            order bucket.
        assertions: ``ClinicalAssertion`` records, assertion mappings, or an
            id-keyed/offset-keyed collection.  Negated and non-patient
            experiencers are retained as assertion axes.
        chains: Optional coreference chains, cluster result, or mention-to-
            cluster mapping.  Chain ids become the deduplication identity.

    Returns:
        A privacy-safe timeline whose events are sorted by anchored timestamp,
        source offsets, and semantic labels.  Unanchored events remain in a
        deterministic trailing partial-order bucket.

    Raises:
        TypeError: If an input record has an unsupported shape.
        ValueError: If a source span is invalid or an entity lacks offsets.
    """

    entity_records = tuple(_coerce_entities(entities))
    time_records = tuple(_coerce_time_records(normalized_times))
    assertion_records = tuple(_coerce_assertion_records(assertions))
    chain_index = _coreference_index(chains)

    prepared: list[
        tuple[_EntityRecord, str, str | None, ClinicalAssertion, str | None]
    ] = []
    for entity in entity_records:
        chain_id = _chain_for_entity(entity, chain_index)
        entity_key = chain_id or entity.entity_key
        time = _time_for_entity(entity, time_records)
        assertion = _assertion_for_entity(entity, assertion_records)
        normalized_time = (
            time.value
            if time is not None
            else _normalize_time_value(entity.direct_time)
        )
        prepared.append((entity, entity_key, normalized_time, assertion, chain_id))

    grouped: dict[tuple[str, str | None, tuple[str, ...]], list[tuple[Any, ...]]] = {}
    for item in prepared:
        entity, entity_key, normalized_time, assertion, chain_id = item
        key = (
            entity_key.casefold(),
            normalized_time,
            _assertion_key(assertion),
        )
        grouped.setdefault(key, []).append(item)

    events: list[ClinicalEvent] = []
    for members in grouped.values():
        ordered_members = sorted(
            members,
            key=lambda item: (
                item[0].source_span[0],
                item[0].source_span[1],
                item[0].input_index,
            ),
        )
        primary, entity_key, normalized_time, assertion, chain_id = ordered_members[0]
        source_spans = tuple(
            sorted({member[0].source_span for member in ordered_members})
        )
        event_kinds = tuple(
            sorted({member[0].event_kind for member in ordered_members})
        )
        provenance: dict[str, Any] = {
            "entity_label": primary.label,
            "event_kinds": event_kinds,
            "mention_count": len(ordered_members),
            "source_span": primary.source_span,
            "source_spans": source_spans,
        }
        if chain_id is not None:
            provenance["coreference_chain"] = chain_id
        events.append(
            ClinicalEvent(
                entity=entity_key,
                event_kind=primary.event_kind,
                normalized_time=normalized_time,
                section=primary.section,
                assertion=assertion,
                source_span=primary.source_span,
                provenance=provenance,
            )
        )

    events.sort(key=_event_sort_key)
    return ClinicalEventTimeline(events=tuple(events))


def _coerce_entities(
    entities: Iterable[Any] | Mapping[Any, Any],
) -> Iterable[_EntityRecord]:
    for index, (raw, mapping_key) in enumerate(
        _iter_records(entities, _ENTITY_RECORD_KEYS)
    ):
        yield _coerce_entity(raw, index, mapping_key)


_ENTITY_RECORD_KEYS = frozenset(
    {
        "start",
        "end",
        "span",
        "source_span",
        "offset",
        "label",
        "entity",
        "entity_id",
        "event_kind",
        "event_type",
    }
)


def _coerce_entity(raw: Any, index: int, mapping_key: Any) -> _EntityRecord:
    source_span = _coerce_optional_offset(
        _first_value(raw, ("source_span", "span", "offset"), _MISSING)
    )
    if source_span is None:
        start = _first_value(raw, ("start",), _MISSING)
        end = _first_value(raw, ("end",), _MISSING)
        source_span = _coerce_optional_offset((start, end))
    if source_span is None:
        source_span = _coerce_optional_offset(mapping_key)
    if source_span is None:
        raise ValueError("timeline entity requires a non-empty source span")

    mention_id = (
        _clean_optional_label(
            _first_value(
                raw,
                ("mention_id", "span_id", "source_id", "id", "entity_id"),
                mapping_key,
            )
        )
        or f"mention-{index}"
    )
    label = (
        _clean_optional_label(
            _first_value(
                raw,
                ("label", "canonical_label", "entity_type", "category", "type"),
                None,
            )
        )
        or "ENTITY"
    )
    entity_value = _first_value(
        raw,
        ("entity_id", "concept_id", "canonical_id", "code", "entity"),
        mapping_key,
    )
    entity_key = _clean_optional_label(entity_value) or mention_id
    event_kind = _clean_optional_label(
        _first_value(raw, ("event_kind", "event_type", "kind", "event"), None)
    )
    metadata = _first_value(raw, ("metadata",), {})
    if isinstance(metadata, Mapping):
        event_kind = event_kind or _clean_optional_label(
            _first_value(metadata, ("event_kind", "event_type", "kind"), None)
        )
    event_kind = event_kind or _DEFAULT_EVENT_KIND
    section = _clean_optional_label(
        _first_value(raw, ("section", "section_label", "canonical_section"), None)
    )
    document_id = _clean_optional_label(
        _first_value(raw, ("document_id", "doc_id"), None)
    )
    direct_time = _first_value(
        raw,
        ("normalized_time", "normalized_timestamp", "time", "timestamp"),
        None,
    )
    direct_assertion = _first_value(raw, ("assertion", "clinical_assertion"), None)
    if direct_assertion is None and isinstance(metadata, Mapping):
        direct_assertion = _first_value(
            metadata, ("clinical_context", "assertion"), None
        )
    return _EntityRecord(
        input_index=index,
        document_id=document_id,
        mention_id=mention_id,
        entity_key=entity_key,
        label=label,
        event_kind=_normalize_event_kind(event_kind),
        section=section,
        source_span=source_span,
        direct_time=direct_time,
        direct_assertion=direct_assertion,
    )


def _coerce_time_records(
    normalized_times: Iterable[Any] | Mapping[Any, Any] | None,
) -> Iterable[_TimeRecord]:
    if normalized_times is None:
        return ()
    records: list[_TimeRecord] = []
    for index, (raw, mapping_key) in enumerate(
        _iter_records(normalized_times, _TIME_RECORD_KEYS)
    ):
        records.append(_coerce_time_record(raw, index, mapping_key))
    return tuple(records)


_TIME_RECORD_KEYS = frozenset(
    {
        "value",
        "normalized_value",
        "normalized_time",
        "timestamp",
        "start",
        "end",
        "span",
        "source_span",
        "granularity_flags",
        "timex_type",
    }
)


def _coerce_time_record(raw: Any, index: int, mapping_key: Any) -> _TimeRecord:
    source_span = _coerce_optional_offset(
        _first_value(raw, ("source_span", "span", "offset"), _MISSING)
    )
    if source_span is None:
        start = _first_value(raw, ("start",), _MISSING)
        end = _first_value(raw, ("end",), _MISSING)
        source_span = _coerce_optional_offset((start, end))
    if source_span is None:
        source_span = _coerce_optional_offset(mapping_key)
    value = _first_value(
        raw,
        ("normalized_time", "normalized_value", "value", "timestamp", "time"),
        raw if isinstance(raw, (str, date, datetime)) else None,
    )
    flags = _first_value(raw, ("granularity_flags", "flags"), ())
    if isinstance(flags, str):
        flags = (flags,)
    flags = tuple(str(flag).casefold() for flag in (flags or ()))
    normalized_value = _normalize_time_value(value)
    return _TimeRecord(
        input_index=index,
        keys=_record_keys(raw, mapping_key),
        source_span=source_span,
        value=normalized_value,
        unanchored="unanchored" in flags or _anchor_key(normalized_value) is None,
    )


def _coerce_assertion_records(
    assertions: Iterable[Any] | Mapping[Any, Any] | None,
) -> Iterable[_AssertionRecord]:
    if assertions is None:
        return ()
    records: list[_AssertionRecord] = []
    for index, (raw, mapping_key) in enumerate(
        _iter_records(assertions, _ASSERTION_RECORD_KEYS)
    ):
        source_span = _coerce_optional_offset(
            _first_value(raw, ("source_span", "span", "offset"), _MISSING)
        )
        if source_span is None:
            start = _first_value(raw, ("start",), _MISSING)
            end = _first_value(raw, ("end",), _MISSING)
            source_span = _coerce_optional_offset((start, end))
        if source_span is None:
            source_span = _coerce_optional_offset(mapping_key)
        records.append(
            _AssertionRecord(
                input_index=index,
                keys=_record_keys(raw, mapping_key),
                source_span=source_span,
                assertion=_coerce_assertion(raw),
            )
        )
    return tuple(records)


_ASSERTION_RECORD_KEYS = frozenset(
    {
        "temporality",
        "certainty",
        "negation",
        "experiencer",
        "assertion",
        "clinical_context",
        "start",
        "end",
        "span",
        "source_span",
    }
)


def _iter_records(
    values: Iterable[Any] | Mapping[Any, Any], record_keys: set[str] | frozenset[str]
) -> Iterable[tuple[Any, Any]]:
    if isinstance(values, Mapping):
        if any(key in values for key in record_keys):
            yield values, None
            return
        for key, value in values.items():
            yield value, key
        return
    if isinstance(values, (str, bytes)):
        yield values, None
        return
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise TypeError("timeline inputs must be iterable or mappings") from exc
    for value in iterator:
        yield value, None


def _time_for_entity(
    entity: _EntityRecord,
    records: Sequence[_TimeRecord],
) -> _TimeRecord | None:
    if not records:
        return None
    candidates: list[tuple[int, int, _TimeRecord]] = []
    for record in records:
        key_rank = 0 if set(entity.aliases) & set(record.keys) else 3
        span_rank = 1 if record.source_span == entity.source_span else 2
        if key_rank == 3 and record.source_span != entity.source_span:
            continue
        candidates.append((key_rank, span_rank, record))
    if not candidates:
        aligned = [
            record
            for record in records
            if record.input_index == entity.input_index
            and not record.keys
            and record.source_span is None
        ]
        return aligned[0] if aligned else None
    return min(candidates, key=lambda item: (item[0], item[1], item[2].input_index))[2]


def _assertion_for_entity(
    entity: _EntityRecord,
    records: Sequence[_AssertionRecord],
) -> ClinicalAssertion:
    if entity.direct_assertion is not None:
        return _coerce_assertion(entity.direct_assertion)
    candidates: list[tuple[int, int, _AssertionRecord]] = []
    for record in records:
        key_rank = 0 if set(entity.aliases) & set(record.keys) else 3
        span_rank = 1 if record.source_span == entity.source_span else 2
        if key_rank == 3 and record.source_span != entity.source_span:
            continue
        candidates.append((key_rank, span_rank, record))
    if not candidates:
        aligned = [
            record
            for record in records
            if record.input_index == entity.input_index
            and not record.keys
            and record.source_span is None
        ]
        return aligned[0].assertion if aligned else _coerce_assertion(None)
    return min(candidates, key=lambda item: (item[0], item[1], item[2].input_index))[
        2
    ].assertion


def _coreference_index(chains: Any) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    if chains is None:
        return index

    for method_name in ("cluster_ids_by_offset", "entity_ids_by_offset"):
        method = getattr(chains, method_name, None)
        if callable(method):
            for key, value in method().items():
                if isinstance(key, tuple) and len(key) == 2:
                    document_id, offset = key
                    if isinstance(offset, (tuple, list)) and len(offset) == 2:
                        _add_chain_alias(index, offset, value, document_id)
            return index

    if isinstance(chains, Mapping):
        for key, value in chains.items():
            if _is_scalar(value):
                _add_chain_alias(index, key, value)
                continue
            if isinstance(value, Iterable) and not isinstance(value, Mapping):
                _add_chain_members(index, key, value)
                continue
            chain_id = _first_value(value, ("chain_id", "cluster_id", "entity_id"), key)
            _add_chain_alias(index, key, chain_id)
            members = _first_value(value, ("members", "member_offsets", "mentions"), ())
            _add_chain_members(index, chain_id, members)
        return index

    clusters = _first_value(chains, ("clusters",), _MISSING)
    if clusters is not _MISSING:
        chains = clusters
    if isinstance(chains, (str, bytes)):
        return index
    try:
        iterator = iter(chains)
    except TypeError:
        iterator = iter((chains,))
    for chain in iterator:
        chain_id = _first_value(chain, ("chain_id", "cluster_id", "entity_id"), None)
        if chain_id is None:
            continue
        members = _first_value(
            chain, ("members", "member_offsets", "member_spans", "mentions"), ()
        )
        _add_chain_members(index, chain_id, members)
    return index


def _add_chain_members(
    index: dict[tuple[str, str], str], chain_id: Any, members: Any
) -> None:
    chain = _clean_optional_label(chain_id)
    if not chain:
        return
    if isinstance(members, (str, bytes)):
        members = (members,)
    for member in members or ():
        _add_chain_alias(index, member, chain)


def _add_chain_alias(
    index: dict[tuple[str, str], str],
    alias: Any,
    chain_id: Any,
    document_id: Any = None,
) -> None:
    chain = _clean_optional_label(chain_id)
    if not chain:
        return
    if isinstance(alias, (tuple, list)) and len(alias) == 2:
        offset = _coerce_optional_offset(alias)
        if offset is not None:
            index[("offset", f"{offset[0]}:{offset[1]}")] = chain
            if document_id is not None:
                doc = _clean_optional_label(document_id)
                if doc:
                    index[("doc-offset", f"{doc}:{offset[0]}:{offset[1]}")] = chain
            return
    if isinstance(alias, Mapping) or hasattr(alias, "__dict__"):
        member_id = _first_value(
            alias,
            ("mention_id", "span_id", "source_id", "id", "entity_id"),
            None,
        )
        if member_id is not None:
            index[("id", str(member_id))] = chain
        member_doc = _first_value(alias, ("document_id", "doc_id"), document_id)
        offset = _coerce_optional_offset(
            _first_value(alias, ("source_span", "span", "offset"), _MISSING)
        )
        if offset is None:
            start = _first_value(alias, ("start",), _MISSING)
            end = _first_value(alias, ("end",), _MISSING)
            offset = _coerce_optional_offset((start, end))
        if offset is not None:
            index[("offset", f"{offset[0]}:{offset[1]}")] = chain
            doc = _clean_optional_label(member_doc)
            if doc:
                index[("doc-offset", f"{doc}:{offset[0]}:{offset[1]}")] = chain
        return
    index[("id", str(alias))] = chain


def _chain_for_entity(
    entity: _EntityRecord,
    index: Mapping[tuple[str, str], str],
) -> str | None:
    for alias in entity.aliases:
        chain = index.get(("id", alias))
        if chain is not None:
            return chain
    start, end = entity.source_span
    if entity.document_id:
        chain = index.get(("doc-offset", f"{entity.document_id}:{start}:{end}"))
        if chain is not None:
            return chain
    return index.get(("offset", f"{start}:{end}"))


def _assertion_key(assertion: ClinicalAssertion) -> tuple[str, ...]:
    values = assertion.to_dict()
    return tuple(
        str(values.get(key, ""))
        for key in (
            "temporality",
            "certainty",
            "negation",
            "experiencer",
        )
    )


def _event_sort_key(event: ClinicalEvent) -> tuple[Any, ...]:
    anchor = _anchor_key(event.normalized_time)
    if anchor is None:
        return (
            1,
            0,
            0,
            event.start,
            event.end,
            event.entity.casefold(),
            event.event_kind,
        )
    return (
        0,
        *anchor,
        event.start,
        event.end,
        event.entity.casefold(),
        event.event_kind,
    )


def _anchor_key(value: str | None) -> tuple[int, int, str] | None:
    if not value:
        return None
    candidate = value.strip()
    if "/" in candidate:
        candidate = candidate.split("/", 1)[0]
    match = _DATE_PREFIX_RE.fullmatch(candidate)
    if match is None:
        try:
            parsed_datetime = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed_date = date.fromisoformat(candidate)
            except ValueError:
                return None
            return parsed_date.toordinal(), 0, value
        return parsed_datetime.date().toordinal(), 1, value
    year = int(match.group("year"))
    month = int(match.group("month") or 1)
    day = int(match.group("day") or 1)
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    precision = 2 if match.group("day") else 1 if match.group("month") else 0
    return parsed.toordinal(), precision, value


def _coerce_assertion(value: Any) -> ClinicalAssertion:
    if isinstance(value, ClinicalAssertion):
        return value
    if value is None:
        return ClinicalAssertion(
            temporality=RECENT,
            certainty=CERTAIN,
            negation=AFFIRMED,
        )
    if hasattr(value, "to_dict") and not isinstance(value, Mapping):
        value = value.to_dict()
    if isinstance(value, str):
        value = {"assertion": value}
    if not isinstance(value, Mapping):
        raise TypeError("assertions must be ClinicalAssertion records or mappings")
    data = dict(value)
    nested = data.get("clinical_context") or data.get("context")
    if isinstance(nested, Mapping):
        merged = dict(nested)
        merged.update(data)
        data = merged
    assertion_value = data.get("assertion")
    if isinstance(assertion_value, ClinicalAssertion):
        return assertion_value
    if isinstance(assertion_value, Mapping):
        merged = dict(assertion_value)
        merged.update(data)
        data = merged
    if isinstance(assertion_value, str):
        token = assertion_value.casefold().strip()
        if token in {AFFIRMED, NEGATED}:
            data.setdefault("negation", token)
        elif token in {RECENT, "historical", HYPOTHETICAL}:
            data.setdefault("temporality", token)
        elif token in {CERTAIN, "uncertain"}:
            data.setdefault("certainty", token)
        elif token in {"family", "relative", "other"}:
            data.setdefault("experiencer", token)
    negation = _axis_value(data, ("negation", "polarity", "assertion_status"), AFFIRMED)
    if negation not in {AFFIRMED, NEGATED}:
        negation = AFFIRMED
    temporality = _axis_value(data, ("temporality", "temporal_status"), RECENT)
    if temporality not in {RECENT, "historical", HYPOTHETICAL}:
        temporality = RECENT
    certainty = _axis_value(data, ("certainty", "uncertainty"), CERTAIN)
    if certainty == "possible":
        certainty = "uncertain"
    if certainty not in {CERTAIN, "uncertain"}:
        certainty = CERTAIN
    experiencer = _clean_optional_label(
        _first_value(data, ("experiencer", "subject", "experiencer_type"), None)
    )
    return ClinicalAssertion(
        temporality=temporality,
        certainty=certainty,
        negation=negation,
        experiencer=experiencer,
    )


def _axis_value(data: Mapping[str, Any], keys: Sequence[str], default: str) -> str:
    value = _first_value(data, keys, default)
    return str(value).strip().casefold() if value is not None else default


def _record_keys(raw: Any, mapping_key: Any) -> tuple[str, ...]:
    values: list[str] = []
    if mapping_key is not None:
        values.append(str(mapping_key))
    for key in (
        "mention_id",
        "span_id",
        "source_id",
        "id",
        "entity_id",
        "event_id",
        "timex_id",
    ):
        value = _first_value(raw, (key,), None)
        if value is not None:
            values.append(str(value))
    return tuple(dict.fromkeys(values))


def _first_value(raw: Any, keys: Sequence[str], default: Any = None) -> Any:
    if raw is _MISSING:
        return default
    if isinstance(raw, Mapping):
        for key in keys:
            if key in raw:
                return raw[key]
        return default
    for key in keys:
        value = getattr(raw, key, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _coerce_offset(value: Any) -> SpanOffset:
    offset = _coerce_optional_offset(value)
    if offset is None:
        raise ValueError("timeline entity requires a non-empty source span")
    return offset


def _coerce_optional_offset(value: Any) -> SpanOffset | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, Mapping):
        start = value.get("start", _MISSING)
        end = value.get("end", _MISSING)
        if start is _MISSING or end is _MISSING:
            return None
        value = (start, end)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if any(item is _MISSING or item is None for item in value):
        return None
    if len(value) != 2 or not all(isinstance(item, int) for item in value):
        raise TypeError("timeline source spans must contain two integer offsets")
    start, end = int(value[0]), int(value[1])
    if start < 0 or end <= start:
        raise ValueError("timeline source span must satisfy 0 <= start < end")
    return start, end


def _clean_label(value: Any, *, field_name: str) -> str:
    cleaned = _clean_optional_label(value)
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty label")
    return cleaned


def _clean_optional_label(value: Any) -> str | None:
    if value is None or value is _MISSING:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = _WHITESPACE_RE.sub(" ", value.strip())
    return cleaned or None


def _normalize_event_kind(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _normalize_time_value(value: Any) -> str | None:
    if value is None or value is _MISSING:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        nested = _first_value(
            value,
            ("normalized_time", "normalized_value", "value", "timestamp", "date"),
            None,
        )
        return _normalize_time_value(nested)
    iso_value = getattr(value, "iso_value", _MISSING)
    if iso_value is not _MISSING:
        return _normalize_time_value(iso_value)
    object_value = getattr(value, "value", _MISSING)
    if object_value is not _MISSING:
        return _normalize_time_value(object_value)
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    return cleaned or None


def _safe_provenance(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    safe: dict[str, Any] = {}
    for key in _PROVENANCE_KEYS:
        if key not in value:
            continue
        item = value[key]
        if key in {"source_span", "source_spans"}:
            if key == "source_span":
                offset = _coerce_optional_offset(item)
                if offset is not None:
                    safe[key] = offset
            else:
                offsets = []
                for candidate in item or ():
                    offset = _coerce_optional_offset(candidate)
                    if offset is not None:
                        offsets.append(offset)
                safe[key] = tuple(offsets)
        elif key == "event_kinds":
            safe[key] = tuple(sorted(str(candidate) for candidate in (item or ())))
        elif key in {"input_index", "mention_count"}:
            if isinstance(item, int) and item >= 0:
                safe[key] = item
        else:
            cleaned = _clean_optional_label(item)
            if cleaned:
                safe[key] = cleaned
    return safe


def _json_safe_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key == "source_span":
            result[key] = {"start": item[0], "end": item[1]}
        elif key == "source_spans":
            result[key] = [{"start": offset[0], "end": offset[1]} for offset in item]
        elif isinstance(item, tuple):
            result[key] = list(item)
        else:
            result[key] = item
    return result


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


__all__ = [
    "AssembledTimeline",
    "CLINICAL_EVENT_TIMELINE_ADVISORY",
    "CLINICAL_EVENT_TIMELINE_SCHEMA_VERSION",
    "ClinicalEvent",
    "ClinicalEventTimeline",
    "ClinicalTimeline",
    "EventTimeline",
    "assemble_timeline",
]
