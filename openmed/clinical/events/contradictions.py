"""Privacy-safe contradiction reports for typed clinical event timelines.

The comparator keeps interval values in memory only long enough to compare
them.  Report evidence contains source offsets, fingerprints, and controlled
type or status labels; it never contains source text, event values, or date
intervals.  The implementation is deterministic and rules-only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

from openmed.core.audit import hash_text

CONTRADICTION_REPORT_SCHEMA_VERSION = 1

EVENT_CONTRADICTION_ADVISORY = (
    "Clinical event contradiction reports are deterministic review signals; "
    "they do not select a clinical truth or make a clinical decision."
)

ContradictionKind = Literal[
    "overlap",
    "impossible_order",
    "conflicting_status",
]
DateLike = date | datetime | str

_HASH_RE = re.compile(r"^(?:sha256|hmac-sha256):[0-9a-f]{64}$")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,96}$")
_TOKEN_RE = re.compile(r"[^a-z0-9_.:/-]+")

_STATUS_ALIASES = {
    "active": "active",
    "affirmed": "active",
    "current": "active",
    "ongoing": "active",
    "present": "active",
    "started": "active",
    "start": "active",
    "on": "active",
    "inactive": "inactive",
    "historical": "inactive",
    "history": "inactive",
    "resolved": "inactive",
    "ended": "inactive",
    "stopped": "inactive",
    "stop": "inactive",
    "off": "inactive",
    "refuted": "refuted",
    "negated": "refuted",
    "absent": "refuted",
    "否定": "refuted",
    "unconfirmed": "unconfirmed",
    "uncertain": "unconfirmed",
    "hypothetical": "unconfirmed",
    "provisional": "unconfirmed",
}
_CONFLICTING_STATUS_PAIRS = frozenset(
    {
        frozenset(("active", "inactive")),
        frozenset(("active", "refuted")),
        frozenset(("active", "unconfirmed")),
        frozenset(("inactive", "refuted")),
    }
)
_START_EVENT_TYPES = frozenset(
    {
        "admission",
        "begin",
        "beginning",
        "initiation",
        "onset",
        "start",
        "started",
    }
)
_END_EVENT_TYPES = frozenset(
    {
        "discharge",
        "end",
        "ended",
        "resolution",
        "resolved",
        "stop",
        "stopped",
    }
)


@dataclass(frozen=True)
class EventInterval:
    """A typed event interval with privacy-safe source provenance.

    ``interval_start`` and ``interval_end`` are used only during comparison.
    They are intentionally absent from :meth:`to_dict`, which is safe for
    reports and logs.  ``fingerprint`` should normally be a SHA-256 or
    HMAC-SHA256 fingerprint of the source span.  If omitted, a deterministic
    metadata fingerprint is generated.
    """

    event_id: str
    event_type: str
    interval_start: DateLike
    interval_end: DateLike
    source_start: int = 0
    source_end: int = 0
    fingerprint: str = ""
    entity_id: str | None = None
    sequence: int | None = None
    status: str | None = None
    precedes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        event_id = _safe_identifier(self.event_id, fallback="event")
        event_type = _safe_token(self.event_type, fallback="event")
        start = _coerce_date(self.interval_start)
        end = _coerce_date(self.interval_end)
        source_start, source_end = _validate_offsets(
            self.source_start,
            self.source_end,
        )
        fingerprint = _coerce_fingerprint(
            self.fingerprint,
            {
                "event_id": event_id,
                "event_type": event_type,
                "interval_start": start.isoformat(),
                "interval_end": end.isoformat(),
                "source_start": source_start,
                "source_end": source_end,
            },
        )
        entity_id = (
            _safe_identifier(self.entity_id, fallback="")
            if self.entity_id is not None
            else None
        )
        sequence = _coerce_sequence(self.sequence)
        status = _coerce_status(self.status) if self.status is not None else None
        precedes = tuple(
            safe_value
            for value in _string_sequence(self.precedes)
            if (safe_value := _safe_identifier(value, fallback=""))
        )

        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "interval_start", start)
        object.__setattr__(self, "interval_end", end)
        object.__setattr__(self, "source_start", source_start)
        object.__setattr__(self, "source_end", source_end)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "precedes", precedes)

    @property
    def entity_key(self) -> str | None:
        """Return the optional logical entity key used for comparisons."""

        return self.entity_id

    @property
    def start(self) -> date:
        """Return the normalized interval start."""

        return self.interval_start

    @property
    def end(self) -> date:
        """Return the normalized interval end."""

        return self.interval_end

    @property
    def text_hash(self) -> str:
        """Return the source fingerprint under the common span vocabulary."""

        return self.fingerprint

    @property
    def source_offsets(self) -> tuple[int, int]:
        """Return the inclusive/exclusive source span offsets."""

        return self.source_start, self.source_end

    def to_dict(self) -> dict[str, Any]:
        """Return a report-safe representation without interval values."""

        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_offsets": [self.source_start, self.source_end],
            "fingerprint": self.fingerprint,
        }
        if self.status is not None:
            payload["status"] = self.status
        return payload


# The longer name is useful when the caller wants to make the typed nature of
# the input explicit.  Keep one implementation so both names serialize alike.
TypedEventInterval = EventInterval


@dataclass(frozen=True)
class EventStatusAssertion:
    """A status assertion tied to a source span and optional time interval."""

    entity_id: str
    status: str
    source_start: int = 0
    source_end: int = 0
    fingerprint: str = ""
    interval_start: DateLike | None = None
    interval_end: DateLike | None = None
    assertion_id: str | None = None
    event_id: str | None = None

    def __post_init__(self) -> None:
        entity_id = _safe_identifier(self.entity_id, fallback="entity")
        status = _coerce_status(self.status)
        source_start, source_end = _validate_offsets(
            self.source_start,
            self.source_end,
        )
        interval_start = (
            _coerce_date(self.interval_start)
            if self.interval_start is not None
            else None
        )
        interval_end = (
            _coerce_date(self.interval_end) if self.interval_end is not None else None
        )
        if (interval_start is None) != (interval_end is None):
            raise ValueError("status interval requires both endpoints")
        fingerprint = _coerce_fingerprint(
            self.fingerprint,
            {
                "entity_id": entity_id,
                "status": status,
                "source_start": source_start,
                "source_end": source_end,
            },
        )
        assertion_id = (
            _safe_identifier(self.assertion_id, fallback="")
            if self.assertion_id is not None
            else None
        )
        event_id = (
            _safe_identifier(self.event_id, fallback="")
            if self.event_id is not None
            else None
        )

        object.__setattr__(self, "entity_id", entity_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "source_start", source_start)
        object.__setattr__(self, "source_end", source_end)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "interval_start", interval_start)
        object.__setattr__(self, "interval_end", interval_end)
        object.__setattr__(self, "assertion_id", assertion_id)
        object.__setattr__(self, "event_id", event_id)

    @property
    def entity_key(self) -> str:
        """Return the logical entity key used for status comparison."""

        return self.entity_id

    @property
    def source_offsets(self) -> tuple[int, int]:
        """Return the inclusive/exclusive source span offsets."""

        return self.source_start, self.source_end

    @property
    def text_hash(self) -> str:
        """Return the source fingerprint under the common span vocabulary."""

        return self.fingerprint

    def to_dict(self) -> dict[str, Any]:
        """Return a report-safe representation without interval values."""

        payload: dict[str, Any] = {
            "status": self.status,
            "source_offsets": [self.source_start, self.source_end],
            "fingerprint": self.fingerprint,
        }
        if self.assertion_id is not None:
            payload["assertion_id"] = self.assertion_id
        if self.event_id is not None:
            payload["event_id"] = self.event_id
        return payload


StatusAssertion = EventStatusAssertion


@dataclass(frozen=True)
class ContradictionEvidence:
    """Privacy-safe evidence reference attached to one contradiction."""

    source_start: int
    source_end: int
    fingerprint: str
    event_id: str | None = None
    event_type: str | None = None
    status: str | None = None

    def __post_init__(self) -> None:
        source_start, source_end = _validate_offsets(
            self.source_start,
            self.source_end,
        )
        fingerprint = _coerce_fingerprint(
            self.fingerprint,
            {
                "source_start": source_start,
                "source_end": source_end,
            },
        )
        event_id = (
            _safe_identifier(self.event_id, fallback="")
            if self.event_id is not None
            else None
        )
        event_type = (
            _safe_token(self.event_type, fallback="event")
            if self.event_type is not None
            else None
        )
        status = _coerce_status(self.status) if self.status is not None else None
        object.__setattr__(self, "source_start", source_start)
        object.__setattr__(self, "source_end", source_end)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "status", status)

    @property
    def source_offsets(self) -> tuple[int, int]:
        """Return the inclusive/exclusive source span offsets."""

        return self.source_start, self.source_end

    def to_dict(self) -> dict[str, Any]:
        """Return evidence containing offsets and fingerprints only."""

        payload: dict[str, Any] = {
            "source_offsets": [self.source_start, self.source_end],
            "fingerprint": self.fingerprint,
        }
        if self.event_id is not None:
            payload["event_id"] = self.event_id
        if self.event_type is not None:
            payload["event_type"] = self.event_type
        if self.status is not None:
            payload["status"] = self.status
        return payload


@dataclass(frozen=True)
class EventContradiction:
    """One deterministic contradiction finding."""

    kind: ContradictionKind
    evidence: tuple[ContradictionEvidence, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.kind not in {
            "overlap",
            "impossible_order",
            "conflicting_status",
        }:
            raise ValueError("unsupported contradiction kind")
        if not self.evidence:
            raise ValueError("contradiction evidence must not be empty")
        if not self.reason:
            raise ValueError("contradiction reason must not be empty")
        object.__setattr__(self, "evidence", tuple(self.evidence))

    @property
    def contradiction_type(self) -> ContradictionKind:
        """Return ``kind`` under the terminology used by report consumers."""

        return self.kind

    @property
    def type(self) -> ContradictionKind:
        """Return ``kind`` for compact consumers."""

        return self.kind

    @property
    def sources(self) -> tuple[ContradictionEvidence, ...]:
        """Return the evidence references."""

        return self.evidence

    @property
    def left(self) -> ContradictionEvidence:
        """Return the first deterministically ordered evidence reference."""

        return self.evidence[0]

    @property
    def right(self) -> ContradictionEvidence | None:
        """Return the second evidence reference when this is a pair finding."""

        return self.evidence[1] if len(self.evidence) > 1 else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready, source-text-free finding."""

        return {
            "type": self.kind,
            "kind": self.kind,
            "reason": self.reason,
            "evidence": [item.to_dict() for item in self.evidence],
        }


Contradiction = EventContradiction


@dataclass(frozen=True)
class EventContradictionReport:
    """Deterministic report containing all detected contradiction findings."""

    contradictions: tuple[EventContradiction, ...]
    events_checked: int
    status_assertions_checked: int
    disclaimer: str = EVENT_CONTRADICTION_ADVISORY
    schema_version: int = CONTRADICTION_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.events_checked < 0 or self.status_assertions_checked < 0:
            raise ValueError("report counts must be non-negative")
        object.__setattr__(self, "contradictions", tuple(self.contradictions))

    @property
    def findings(self) -> tuple[EventContradiction, ...]:
        """Return contradiction findings under the report vocabulary."""

        return self.contradictions

    @property
    def has_contradictions(self) -> bool:
        """Return whether at least one contradiction was detected."""

        return bool(self.contradictions)

    @property
    def counts(self) -> dict[ContradictionKind, int]:
        """Return deterministic counts by contradiction kind."""

        counts: dict[ContradictionKind, int] = {
            "conflicting_status": 0,
            "impossible_order": 0,
            "overlap": 0,
        }
        for contradiction in self.contradictions:
            counts[contradiction.kind] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report with no raw source values."""

        return {
            "schema_version": self.schema_version,
            "events_checked": self.events_checked,
            "status_assertions_checked": self.status_assertions_checked,
            "counts": dict(self.counts),
            "contradictions": [
                contradiction.to_dict() for contradiction in self.contradictions
            ],
            "disclaimer": self.disclaimer,
        }


def report_event_contradictions(
    events: Iterable[EventInterval | Mapping[str, Any] | Any],
    status_assertions: Iterable[EventStatusAssertion | Mapping[str, Any]] = (),
) -> EventContradictionReport:
    """Compare event intervals and status assertions without choosing truth.

    Events with the same explicit ``entity_id`` are compared together.  When
    no entity key is supplied, events of the same controlled event type form a
    conservative comparison group.  Intervals are inclusive day ranges: two
    ranges sharing a day are reported as ``"overlap"``.  An interval whose
    start follows its end is reported as ``"impossible_order"``.  Explicit
    ``sequence`` values, start/end event types, and ``precedes`` references can
    add pairwise impossible-order findings.

    Statuses are compared only for the same entity.  A status pair is checked
    when its optional intervals overlap, or when either assertion is
    unbounded.  An active/inactive, active/refuted, active/unconfirmed, or
    inactive/refuted pair is reported as ``"conflicting_status"``; no status
    is selected as authoritative.

    Args:
        events: ``EventInterval`` records, compatible mappings, or timeline
            event records carrying a normalized ``interval``.
        status_assertions: Optional ``EventStatusAssertion`` records or
            compatible mappings.

    Returns:
        A deterministic report whose serialized evidence contains source
        offsets and fingerprints but no source text or interval values.
    """

    raw_events = _iter_records(events)
    interval_records: list[EventInterval] = []
    embedded_statuses: list[EventStatusAssertion] = []
    for index, raw_event in enumerate(raw_events):
        event = _coerce_event(raw_event, index=index)
        if event is None:
            continue
        interval_records.append(event)
        if event.status is not None:
            embedded_statuses.append(
                EventStatusAssertion(
                    entity_id=event.entity_id or event.event_id,
                    status=event.status,
                    source_start=event.source_start,
                    source_end=event.source_end,
                    fingerprint=event.fingerprint,
                    interval_start=event.interval_start,
                    interval_end=event.interval_end,
                    assertion_id=event.event_id,
                    event_id=event.event_id,
                )
            )

    statuses = [
        assertion
        for index, raw_assertion in enumerate(
            [*embedded_statuses, *_iter_records(status_assertions)]
        )
        if (assertion := _coerce_status_assertion(raw_assertion, index=index))
        is not None
    ]

    findings = [*(_interval_findings(interval_records)), *(_status_findings(statuses))]
    findings.sort(key=_contradiction_sort_key)
    return EventContradictionReport(
        contradictions=tuple(findings),
        events_checked=len(interval_records),
        status_assertions_checked=len(statuses),
    )


def compare_event_intervals(
    events: Iterable[EventInterval | Mapping[str, Any] | Any],
    status_assertions: Iterable[EventStatusAssertion | Mapping[str, Any]] = (),
) -> EventContradictionReport:
    """Alias for :func:`report_event_contradictions`."""

    return report_event_contradictions(events, status_assertions)


def detect_event_contradictions(
    events: Iterable[EventInterval | Mapping[str, Any] | Any],
    status_assertions: Iterable[EventStatusAssertion | Mapping[str, Any]] = (),
) -> EventContradictionReport:
    """Alias for :func:`report_event_contradictions`."""

    return report_event_contradictions(events, status_assertions)


def _interval_findings(events: Sequence[EventInterval]) -> list[EventContradiction]:
    findings: list[EventContradiction] = []
    ordered_events = sorted(events, key=_event_sort_key)
    seen_invalid: set[tuple[Any, ...]] = set()
    for event in ordered_events:
        if event.interval_start <= event.interval_end:
            continue
        signature = _event_signature(event)
        if signature in seen_invalid:
            continue
        seen_invalid.add(signature)
        findings.append(
            _contradiction(
                "impossible_order",
                (_event_evidence(event),),
            )
        )

    for left_index, left in enumerate(ordered_events):
        for right in ordered_events[left_index + 1 :]:
            if _same_event(left, right):
                continue
            if (
                left.interval_start > left.interval_end
                or right.interval_start > right.interval_end
            ):
                continue
            evidence = (_event_evidence(left), _event_evidence(right))
            same_group = _comparison_group(left) == _comparison_group(right)
            if same_group and _intervals_overlap(left, right):
                findings.append(_contradiction("overlap", evidence))
            if _pair_has_impossible_order(left, right):
                findings.append(_contradiction("impossible_order", evidence))
    return findings


def _status_findings(
    assertions: Sequence[EventStatusAssertion],
) -> list[EventContradiction]:
    findings: list[EventContradiction] = []
    ordered = sorted(assertions, key=_status_sort_key)
    groups: dict[str, list[EventStatusAssertion]] = {}
    for assertion in ordered:
        groups.setdefault(assertion.entity_id, []).append(assertion)

    for group_assertions in groups.values():
        for left_index, left in enumerate(group_assertions):
            for right in group_assertions[left_index + 1 :]:
                if _same_status_assertion(left, right):
                    continue
                if not _status_intervals_overlap(left, right):
                    continue
                if not _statuses_conflict(left.status, right.status):
                    continue
                findings.append(
                    _contradiction(
                        "conflicting_status",
                        (_status_evidence(left), _status_evidence(right)),
                    )
                )
    return findings


def _coerce_event(raw: Any, *, index: int) -> EventInterval | None:
    if isinstance(raw, EventInterval):
        return raw
    if not isinstance(raw, Mapping):
        interval = getattr(raw, "interval", None)
        if interval is None:
            return None
        text = getattr(raw, "text", "")
        start = getattr(interval, "start", None)
        end = getattr(interval, "end", None)
        if start is None or end is None:
            return None
        source_start = getattr(raw, "start", 0)
        source_end = getattr(raw, "end", 0)
        event_type = getattr(raw, "event_type", None) or "event"
        fingerprint = getattr(raw, "text_hash", "") or (
            hash_text(text) if isinstance(text, str) and text else ""
        )
        event_id = getattr(raw, "event_id", None)
        if not event_id:
            event_id = _derived_reference(
                "event",
                {
                    "event_type": event_type,
                    "source_start": source_start,
                    "source_end": source_end,
                    "fingerprint": fingerprint,
                },
            )
        return EventInterval(
            event_id=event_id,
            event_type=event_type,
            interval_start=start,
            interval_end=end,
            source_start=source_start,
            source_end=source_end,
            fingerprint=fingerprint,
            entity_id=getattr(raw, "entity_id", None),
            sequence=getattr(raw, "sequence", None),
            status=getattr(raw, "status", None),
        )

    data = dict(raw)
    interval = data.get("interval")
    interval_start, interval_end = _interval_values(interval, data)
    if interval_start is None or interval_end is None:
        return None
    source_start, source_end = _mapping_offsets(data, has_interval=interval is not None)
    raw_text = _first_value(data, "text", "surface", "value", "source_text")
    fingerprint = _first_value(
        data,
        "fingerprint",
        "text_hash",
        "source_hash",
        "content_hash",
        "hash",
    )
    if not fingerprint and isinstance(raw_text, str):
        fingerprint = hash_text(raw_text)
    event_id = _first_value(data, "event_id", "id")
    if not event_id:
        event_id = _derived_reference(
            "event",
            {
                "event_type": _first_value(
                    data,
                    "event_type",
                    "type",
                    "kind",
                    "label",
                )
                or "event",
                "source_start": source_start,
                "source_end": source_end,
                "fingerprint": fingerprint or "",
            },
        )
    return EventInterval(
        event_id=event_id,
        event_type=_first_value(data, "event_type", "type", "kind", "label") or "event",
        interval_start=interval_start,
        interval_end=interval_end,
        source_start=source_start,
        source_end=source_end,
        fingerprint=fingerprint or "",
        entity_id=_first_value(
            data,
            "entity_id",
            "entity_key",
            "concept_id",
            "subject_id",
            "group_id",
            "entity",
        ),
        sequence=_first_value(data, "sequence", "order", "ordinal", "position"),
        status=_status_value(data),
        precedes=_string_sequence(data.get("precedes")),
    )


def _coerce_status_assertion(
    raw: Any,
    *,
    index: int,
) -> EventStatusAssertion | None:
    if isinstance(raw, EventStatusAssertion):
        return raw
    if not isinstance(raw, Mapping):
        return None
    data = dict(raw)
    entity_id = _first_value(
        data,
        "entity_id",
        "entity_key",
        "concept_id",
        "subject_id",
        "event_id",
        "entity",
    )
    status = _status_value(data)
    if not entity_id or status is None:
        return None
    interval = data.get("interval")
    interval_start, interval_end = _interval_values(interval, data)
    source_start, source_end = _mapping_offsets(data, has_interval=interval is not None)
    raw_text = _first_value(data, "text", "surface", "value", "source_text")
    fingerprint = _first_value(
        data,
        "fingerprint",
        "text_hash",
        "source_hash",
        "content_hash",
        "hash",
    )
    if not fingerprint and isinstance(raw_text, str):
        fingerprint = hash_text(raw_text)
    assertion_id = _first_value(data, "assertion_id", "id")
    if not assertion_id:
        assertion_id = _derived_reference(
            "assertion",
            {
                "entity_id": entity_id,
                "status": status,
                "source_start": source_start,
                "source_end": source_end,
                "fingerprint": fingerprint or "",
            },
        )
    return EventStatusAssertion(
        entity_id=entity_id,
        status=status,
        source_start=source_start,
        source_end=source_end,
        fingerprint=fingerprint or "",
        interval_start=interval_start,
        interval_end=interval_end,
        assertion_id=assertion_id,
        event_id=_first_value(data, "event_id"),
    )


def _interval_values(
    interval: Any,
    data: Mapping[str, Any],
) -> tuple[DateLike | None, DateLike | None]:
    if interval is not None:
        if isinstance(interval, Mapping):
            start = _first_value(interval, "start", "interval_start", "from")
            end = _first_value(interval, "end", "interval_end", "to")
            if start is None and end is None:
                value = interval.get("value")
                if isinstance(value, str) and "/" in value:
                    start, end = value.split("/", 1)
            return start, end
        if isinstance(interval, Sequence) and not isinstance(interval, (str, bytes)):
            if len(interval) >= 2:
                return interval[0], interval[1]
        start = getattr(interval, "start", None)
        end = getattr(interval, "end", None)
        return start, end

    start = _first_value(data, "interval_start", "start_date", "date_start")
    end = _first_value(data, "interval_end", "end_date", "date_end")
    if start is not None or end is not None:
        return start, end
    direct_start = data.get("start")
    direct_end = data.get("end")
    if _looks_like_date(direct_start) and _looks_like_date(direct_end):
        return direct_start, direct_end
    return None, None


def _mapping_offsets(
    data: Mapping[str, Any],
    *,
    has_interval: bool,
) -> tuple[int, int]:
    offsets = _first_value(data, "source_offsets", "offset", "span", "source_span")
    if isinstance(offsets, Mapping):
        start = _first_value(offsets, "start", "source_start")
        end = _first_value(offsets, "end", "source_end")
        if start is not None or end is not None:
            return _validate_offsets(start or 0, end or 0)
    if isinstance(offsets, Sequence) and not isinstance(offsets, (str, bytes)):
        if len(offsets) >= 2:
            return _validate_offsets(offsets[0], offsets[1])
    start = _first_value(data, "source_start", "start_offset", "char_start")
    end = _first_value(data, "source_end", "end_offset", "char_end")
    if start is not None or end is not None:
        return _validate_offsets(start or 0, end or 0)
    if (
        has_interval
        and isinstance(data.get("start"), int)
        and isinstance(data.get("end"), int)
    ):
        return _validate_offsets(data["start"], data["end"])
    return 0, 0


def _iter_records(records: Any) -> list[Any]:
    if isinstance(records, Mapping):
        if _looks_like_record(records):
            return [records]
        return list(records.values())
    if isinstance(records, (str, bytes)):
        raise TypeError("records must be an iterable of event records")
    try:
        return list(records)
    except TypeError as exc:
        raise TypeError("records must be an iterable of event records") from exc


def _looks_like_record(value: Mapping[str, Any]) -> bool:
    return any(
        key in value
        for key in (
            "event_id",
            "event_type",
            "type",
            "interval",
            "interval_start",
            "start",
            "date_start",
            "status",
            "entity_id",
            "entity",
        )
    )


def _first_value(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _status_value(data: Mapping[str, Any]) -> str | None:
    for key in ("status", "clinical_status", "clinicalStatus", "status_value"):
        value = data.get(key)
        if value is not None:
            return str(value)
    assertion = data.get("assertion")
    if isinstance(assertion, Mapping):
        value = _first_value(assertion, "status", "clinical_status", "clinicalStatus")
        if value is not None:
            return str(value)
    return None


def _string_sequence(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


def _coerce_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        try:
            return date.fromisoformat(candidate[:10])
        except ValueError as exc:
            raise ValueError("interval values must be ISO dates") from exc
    raise TypeError("interval values must be dates or ISO date strings")


def _looks_like_date(value: Any) -> bool:
    if isinstance(value, (date, datetime)):
        return True
    if not isinstance(value, str):
        return False
    try:
        _coerce_date(value)
    except (TypeError, ValueError):
        return False
    return True


def _validate_offsets(start: Any, end: Any) -> tuple[int, int]:
    if isinstance(start, bool) or isinstance(end, bool):
        raise TypeError("source offsets must be integers")
    try:
        normalized_start = int(start)
        normalized_end = int(end)
    except (TypeError, ValueError) as exc:
        raise TypeError("source offsets must be integers") from exc
    if normalized_start < 0 or normalized_end < normalized_start:
        raise ValueError("source offsets must satisfy 0 <= start <= end")
    return normalized_start, normalized_end


def _coerce_sequence(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("event sequence must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("event sequence must be an integer") from exc


def _coerce_status(value: Any) -> str:
    normalized = _safe_token(value, fallback="unknown")
    return _STATUS_ALIASES.get(normalized, "unknown")


def _safe_identifier(value: Any, *, fallback: str) -> str:
    text = str(value).strip() if value is not None else ""
    if _SAFE_IDENTIFIER_RE.fullmatch(text):
        return text
    if not text:
        return fallback
    return "ref:" + hash_text(text)[7:23]


def _safe_token(value: Any, *, fallback: str) -> str:
    text = str(value).strip().casefold() if value is not None else ""
    text = _TOKEN_RE.sub("_", text).strip("_:/.-")
    return text[:96] or fallback


def _coerce_fingerprint(value: Any, fallback_payload: Mapping[str, Any]) -> str:
    if isinstance(value, str) and _HASH_RE.fullmatch(value):
        return value
    if value:
        return hash_text(str(value))
    payload = json.dumps(
        dict(fallback_payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hash_text(payload)


def _derived_reference(prefix: str, payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{prefix}-ref:{hash_text(serialized)[7:23]}"


def _comparison_group(event: EventInterval) -> str:
    return event.entity_id or f"event-type:{event.event_type}"


def _event_sort_key(event: EventInterval) -> tuple[Any, ...]:
    return (
        _comparison_group(event),
        event.interval_start,
        event.interval_end,
        event.sequence if event.sequence is not None else 2**31 - 1,
        event.source_start,
        event.source_end,
        event.event_type,
        event.event_id,
        event.fingerprint,
    )


def _status_sort_key(assertion: EventStatusAssertion) -> tuple[Any, ...]:
    return (
        assertion.entity_id,
        assertion.interval_start or date.max,
        assertion.interval_end or date.max,
        assertion.source_start,
        assertion.source_end,
        assertion.status,
        assertion.fingerprint,
        assertion.assertion_id or "",
    )


def _event_signature(event: EventInterval) -> tuple[Any, ...]:
    return (
        event.event_id,
        event.event_type,
        event.interval_start,
        event.interval_end,
        event.source_start,
        event.source_end,
        event.fingerprint,
    )


def _same_event(left: EventInterval, right: EventInterval) -> bool:
    return _event_signature(left) == _event_signature(right)


def _same_status_assertion(
    left: EventStatusAssertion,
    right: EventStatusAssertion,
) -> bool:
    return (
        left.entity_id,
        left.status,
        left.source_start,
        left.source_end,
        left.fingerprint,
    ) == (
        right.entity_id,
        right.status,
        right.source_start,
        right.source_end,
        right.fingerprint,
    )


def _intervals_overlap(left: EventInterval, right: EventInterval) -> bool:
    return (
        left.interval_start <= right.interval_end
        and right.interval_start <= left.interval_end
    )


def _status_intervals_overlap(
    left: EventStatusAssertion,
    right: EventStatusAssertion,
) -> bool:
    if left.interval_start is None or right.interval_start is None:
        return True
    if left.interval_end is None or right.interval_end is None:
        return True
    if (
        left.interval_start > left.interval_end
        or right.interval_start > right.interval_end
    ):
        return True
    return (
        left.interval_start <= right.interval_end
        and right.interval_start <= left.interval_end
    )


def _pair_has_impossible_order(
    left: EventInterval,
    right: EventInterval,
) -> bool:
    if left.event_id in right.precedes or right.event_id in left.precedes:
        if left.event_id in right.precedes:
            return left.interval_start >= right.interval_start
        return right.interval_start >= left.interval_start

    if left.sequence is not None and right.sequence is not None:
        if left.sequence < right.sequence:
            return left.interval_start > right.interval_start
        if right.sequence < left.sequence:
            return right.interval_start > left.interval_start

    left_type = left.event_type.rsplit(":", 1)[-1]
    right_type = right.event_type.rsplit(":", 1)[-1]
    if left_type in _START_EVENT_TYPES and right_type in _END_EVENT_TYPES:
        return left.interval_start > right.interval_start
    if right_type in _START_EVENT_TYPES and left_type in _END_EVENT_TYPES:
        return right.interval_start > left.interval_start
    return False


def _statuses_conflict(left: str, right: str) -> bool:
    if left == right or "unknown" in {left, right}:
        return False
    return frozenset((left, right)) in _CONFLICTING_STATUS_PAIRS


def _event_evidence(event: EventInterval) -> ContradictionEvidence:
    return ContradictionEvidence(
        source_start=event.source_start,
        source_end=event.source_end,
        fingerprint=event.fingerprint,
        event_id=event.event_id,
        event_type=event.event_type,
        status=event.status,
    )


def _status_evidence(assertion: EventStatusAssertion) -> ContradictionEvidence:
    return ContradictionEvidence(
        source_start=assertion.source_start,
        source_end=assertion.source_end,
        fingerprint=assertion.fingerprint,
        event_id=assertion.event_id or assertion.assertion_id,
        status=assertion.status,
    )


def _contradiction(
    kind: ContradictionKind,
    evidence: tuple[ContradictionEvidence, ...],
) -> EventContradiction:
    reasons = {
        "overlap": "typed event intervals overlap",
        "impossible_order": "typed event ordering is impossible",
        "conflicting_status": "status assertions conflict",
    }
    ordered_evidence = tuple(
        sorted(
            evidence,
            key=lambda item: (
                item.source_start,
                item.source_end,
                item.event_id or "",
                item.fingerprint,
            ),
        )
    )
    return EventContradiction(kind, ordered_evidence, reasons[kind])


def _contradiction_sort_key(
    contradiction: EventContradiction,
) -> tuple[Any, ...]:
    first = contradiction.evidence[0]
    second = contradiction.evidence[1] if len(contradiction.evidence) > 1 else None
    return (
        contradiction.kind,
        first.source_start,
        first.source_end,
        second.source_start if second else -1,
        second.source_end if second else -1,
        first.fingerprint,
        second.fingerprint if second else "",
    )


__all__ = [
    "CONTRADICTION_REPORT_SCHEMA_VERSION",
    "EVENT_CONTRADICTION_ADVISORY",
    "Contradiction",
    "ContradictionEvidence",
    "ContradictionKind",
    "EventContradiction",
    "EventContradictionReport",
    "EventInterval",
    "EventStatusAssertion",
    "StatusAssertion",
    "TypedEventInterval",
    "compare_event_intervals",
    "detect_event_contradictions",
    "report_event_contradictions",
]
