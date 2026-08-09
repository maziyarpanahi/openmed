"""Value-free provenance export for clinical timeline events.

The export is intentionally an allow-list rather than a filtered copy of a
timeline.  It retains document-local event identifiers, half-open source
offsets, a small controlled assertion status, temporal confidence, and a
policy fingerprint.  Source text, normalized values, arbitrary metadata, and
other protected values are never copied to the result.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

TIMELINE_PROVENANCE_SCHEMA_VERSION: Final[int] = 1
TIMELINE_PROVENANCE_DISCLAIMER: Final[str] = (
    "Clinical timeline provenance is value-free assistive metadata for review; "
    "it is not a clinical decision or a compliance certification."
)

_DEFAULT_POLICY_DESCRIPTOR: Final[dict[str, Any]] = {
    "schema_version": TIMELINE_PROVENANCE_SCHEMA_VERSION,
    "protected_values": "omit",
    "ordering": "timeline_position_then_source_offsets_then_event_id",
}
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_MISSING = object()

DEFAULT_TIMELINE_POLICY_FINGERPRINT: Final[str]


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _hash_value(value: Any) -> str:
    if isinstance(value, bytes):
        return _sha256_bytes(value)
    if isinstance(value, str):
        return _sha256_bytes(value.encode("utf-8"))
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _canonical_json(value: Any) -> str:
    """Serialize a value for a stable fingerprint without exposing it."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            default=str,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError):
        return json.dumps(type(value).__name__, ensure_ascii=True)


DEFAULT_TIMELINE_POLICY_FINGERPRINT = _hash_value(_DEFAULT_POLICY_DESCRIPTOR)


@dataclass(frozen=True)
class TimelineProvenanceEvent:
    """Allow-listed provenance for one timeline event.

    ``source_offsets`` uses Python's half-open ``[start, end)`` convention.
    ``value_hash`` is optional and is only populated when the caller opts into
    hashing a source value; the default export omits it.
    """

    event_id: str
    source_offsets: tuple[int, int]
    assertion_status: str = "unknown"
    temporal_confidence: float | None = None
    policy_fingerprint: str = DEFAULT_TIMELINE_POLICY_FINGERPRINT
    value_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _event_id(self.event_id))
        object.__setattr__(
            self,
            "source_offsets",
            _offset_pair(self.source_offsets),
        )
        object.__setattr__(
            self,
            "assertion_status",
            _assertion_status(self.assertion_status),
        )
        object.__setattr__(
            self,
            "temporal_confidence",
            _probability(self.temporal_confidence, "temporal_confidence"),
        )
        object.__setattr__(
            self,
            "policy_fingerprint",
            _fingerprint(self.policy_fingerprint),
        )
        if self.value_hash is not None:
            object.__setattr__(self, "value_hash", _value_hash(self.value_hash))

    @property
    def start(self) -> int:
        """Return the inclusive source start offset."""

        return self.source_offsets[0]

    @property
    def end(self) -> int:
        """Return the exclusive source end offset."""

        return self.source_offsets[1]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record containing no raw source value."""

        payload: dict[str, Any] = {
            "event_id": self.event_id,
            "source_offsets": {"start": self.start, "end": self.end},
            "assertion_status": self.assertion_status,
            "temporal_confidence": self.temporal_confidence,
            "policy_fingerprint": self.policy_fingerprint,
        }
        if self.value_hash is not None:
            payload["value_hash"] = self.value_hash
        return payload


@dataclass(frozen=True)
class TimelineProvenanceExport:
    """Deterministic, value-free clinical timeline provenance payload."""

    events: tuple[TimelineProvenanceEvent, ...]
    policy_fingerprint: str = DEFAULT_TIMELINE_POLICY_FINGERPRINT
    schema_version: int = TIMELINE_PROVENANCE_SCHEMA_VERSION
    disclaimer: str = TIMELINE_PROVENANCE_DISCLAIMER

    def __post_init__(self) -> None:
        if self.schema_version != TIMELINE_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("unsupported timeline provenance schema version")
        object.__setattr__(
            self,
            "events",
            tuple(sorted(self.events, key=_event_sort_key)),
        )
        object.__setattr__(
            self,
            "policy_fingerprint",
            _fingerprint(self.policy_fingerprint),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready export containing no raw source value."""

        return {
            "schema_version": self.schema_version,
            "policy_fingerprint": self.policy_fingerprint,
            "events": [event.to_dict() for event in self.events],
            "disclaimer": self.disclaimer,
        }


TimelineProvenanceRecord = TimelineProvenanceEvent


def build_timeline_provenance_export(
    timeline: Any,
    *,
    policy: Any = None,
    policy_fingerprint: str | None = None,
    include_value_hashes: bool = False,
    hash_values: bool | None = None,
) -> TimelineProvenanceExport:
    """Build a value-free provenance export from a clinical timeline.

    ``timeline`` may be an existing timeline object exposing ``events``, a
    mapping containing an ``events`` collection, or an iterable of event
    mappings/objects.  Event records may provide ``event_id``/``id``,
    ``start``/``end`` or a two-item ``source_offsets`` value, an assertion
    mapping or status string, and ``temporal_confidence``.  A missing policy
    uses the deterministic value-free export policy fingerprint.

    Args:
        timeline: Timeline-like object or event collection to export.
        policy: Optional caller-supplied policy object or mapping.  Its
            fingerprint or stable content hash is emitted, never its values.
        policy_fingerprint: Optional explicit policy fingerprint.  Non-digest
            values are hashed before they are emitted.
        include_value_hashes: Include SHA-256 hashes for available event
            ``text``/``value`` fields.  Raw values remain omitted.
        hash_values: Alias for ``include_value_hashes``.

    Returns:
        An immutable :class:`TimelineProvenanceExport`.

    Raises:
        TypeError: If the timeline or event collection cannot be iterated.
        ValueError: If an event has invalid or missing offsets, or a confidence
            is outside the closed interval from zero to one.
    """

    if hash_values is not None:
        include_value_hashes = hash_values
    if not isinstance(include_value_hashes, bool):
        raise TypeError("include_value_hashes must be a boolean")

    export_policy_fingerprint = _resolve_policy_fingerprint(
        policy=policy,
        policy_fingerprint=policy_fingerprint,
    )
    records: list[tuple[tuple[Any, ...], TimelineProvenanceEvent]] = []
    for raw_event in _timeline_events(timeline):
        event = _coerce_event(
            raw_event,
            default_policy_fingerprint=export_policy_fingerprint,
            include_value_hash=include_value_hashes,
        )
        records.append((_sort_key(raw_event, event), event))

    ordered_events = tuple(
        event for _, event in sorted(records, key=lambda item: item[0])
    )
    return TimelineProvenanceExport(
        events=ordered_events,
        policy_fingerprint=export_policy_fingerprint,
    )


def export_timeline_provenance(
    timeline: Any,
    *,
    policy: Any = None,
    policy_fingerprint: str | None = None,
    include_value_hashes: bool = False,
    hash_values: bool | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready, value-free clinical timeline provenance export."""

    return build_timeline_provenance_export(
        timeline,
        policy=policy,
        policy_fingerprint=policy_fingerprint,
        include_value_hashes=include_value_hashes,
        hash_values=hash_values,
    ).to_dict()


export_clinical_timeline_provenance = export_timeline_provenance
build_value_free_timeline_provenance = build_timeline_provenance_export


def _timeline_events(timeline: Any) -> tuple[Any, ...]:
    if isinstance(timeline, TimelineProvenanceExport):
        return timeline.events

    raw_events = _read(timeline, ("events",), default=_MISSING)
    if raw_events is _MISSING:
        if isinstance(timeline, Mapping):
            raw_events = (timeline,)
        else:
            raw_events = timeline
    if isinstance(raw_events, (str, bytes, bytearray)):
        raise TypeError("timeline events must be an iterable of event records")
    try:
        return tuple(raw_events)
    except TypeError as exc:
        raise TypeError("timeline events must be an iterable of event records") from exc


def _coerce_event(
    raw_event: Any,
    *,
    default_policy_fingerprint: str,
    include_value_hash: bool,
) -> TimelineProvenanceEvent:
    offsets = _event_offsets(raw_event)
    raw_event_id = _event_value(raw_event, ("event_id", "id", "frame_id", "span_id"))
    event_id = (
        _event_id(raw_event_id)
        if raw_event_id is not _MISSING and raw_event_id is not None
        else f"event:{offsets[0]}:{offsets[1]}"
    )

    raw_assertion = _event_value(
        raw_event,
        (
            "assertion_status",
            "assertion",
            "clinical_assertion",
            "status",
        ),
    )
    if raw_assertion is _MISSING:
        assertion_axes = {
            name: _event_value(raw_event, (name,))
            for name in ("negation", "certainty", "temporality")
        }
        assertion_axes = {
            name: value
            for name, value in assertion_axes.items()
            if value is not _MISSING
        }
        if assertion_axes:
            raw_assertion = assertion_axes
    raw_confidence = _event_value(
        raw_event,
        ("temporal_confidence", "temporal_score", "anchor_confidence"),
    )
    if raw_confidence is _MISSING:
        raw_confidence = _event_value(raw_event, ("confidence",))
    raw_policy_fingerprint = _event_value(raw_event, ("policy_fingerprint",))
    if raw_policy_fingerprint is _MISSING or raw_policy_fingerprint is None:
        raw_policy = _event_value(raw_event, ("policy",))
        raw_policy_fingerprint = (
            _fingerprint_from_policy(raw_policy)
            if raw_policy is not _MISSING and raw_policy is not None
            else default_policy_fingerprint
        )
    value_hash = _event_value_hash(raw_event) if include_value_hash else None

    return TimelineProvenanceEvent(
        event_id=event_id,
        source_offsets=offsets,
        assertion_status=(
            "unknown" if raw_assertion is _MISSING else _assertion_status(raw_assertion)
        ),
        temporal_confidence=(
            None
            if raw_confidence is _MISSING or raw_confidence is None
            else _probability(raw_confidence, "temporal_confidence")
        ),
        policy_fingerprint=str(raw_policy_fingerprint),
        value_hash=value_hash,
    )


def _event_offsets(event: Any) -> tuple[int, int]:
    raw_offsets = _event_value(
        event,
        ("source_offsets", "source_offset", "offset", "span"),
    )
    if raw_offsets is not _MISSING:
        return _offset_pair(raw_offsets)

    start = _event_value(event, ("source_start", "start"))
    end = _event_value(event, ("source_end", "end"))
    if start is _MISSING or end is _MISSING:
        raise ValueError("timeline event source offsets are required")
    return _offset_pair((start, end))


def _event_value(event: Any, names: Sequence[str]) -> Any:
    direct = _read(event, names, default=_MISSING)
    if direct is not _MISSING:
        return direct
    metadata = _read(event, ("metadata",), default=_MISSING)
    if isinstance(metadata, Mapping):
        return _read(metadata, names, default=_MISSING)
    return _MISSING


def _read(value: Any, names: Sequence[str], *, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        try:
            candidate = getattr(value, name)
        except AttributeError:
            continue
        if not callable(candidate):
            return candidate
    return default


def _offset_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, Mapping):
        start = value.get("start", value.get("source_start", _MISSING))
        end = value.get("end", value.get("source_end", _MISSING))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 2:
            raise ValueError("timeline event source offsets must contain start and end")
        start, end = value
    else:
        raise TypeError("timeline event source offsets must contain start and end")

    if (
        start is _MISSING
        or end is _MISSING
        or isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise ValueError("timeline event source offsets must satisfy 0 <= start < end")
    return start, end


def _event_id(value: Any) -> str:
    if value is None:
        raise ValueError("timeline event_id must be non-empty")
    candidate = str(value).strip()
    if not candidate:
        raise ValueError("timeline event_id must be non-empty")
    if _SAFE_EVENT_ID_RE.fullmatch(candidate):
        return candidate
    return _hash_value(candidate)


def _probability(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number between 0 and 1") from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field_name} must be a number between 0 and 1")
    return probability


def _assertion_status(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, Mapping):
        direct = value.get("assertion_status", value.get("status", _MISSING))
        if direct is not _MISSING:
            return _assertion_status(direct)
        negation = _normalised_token(value.get("negation"))
        temporality = _normalised_token(value.get("temporality"))
        certainty = _normalised_token(value.get("certainty"))
        if negation in {"negated", "refuted", "absent"}:
            return "negated"
        if temporality in {"hypothetical", "conditional"}:
            return "hypothetical"
        if certainty in {"uncertain", "possible"}:
            return "uncertain"
        if temporality in {"historical", "resolved", "past"}:
            return "historical"
        if any(token is not None for token in (negation, temporality, certainty)):
            return "affirmed"
        return "unknown"

    for attribute in ("assertion_status", "status"):
        nested = _read(value, (attribute,), default=_MISSING)
        if nested is not _MISSING and nested is not value:
            return _assertion_status(nested)
    axes = {
        name: _read(value, (name,), default=_MISSING)
        for name in ("negation", "temporality", "certainty")
    }
    if any(axis is not _MISSING for axis in axes.values()):
        return _assertion_status(
            {name: axis for name, axis in axes.items() if axis is not _MISSING}
        )

    token = _normalised_token(value)
    aliases = {
        "affirmed": "affirmed",
        "active": "affirmed",
        "certain": "affirmed",
        "confirmed": "affirmed",
        "present": "affirmed",
        "negated": "negated",
        "absent": "negated",
        "refuted": "negated",
        "not_present": "negated",
        "uncertain": "uncertain",
        "possible": "uncertain",
        "maybe": "uncertain",
        "historical": "historical",
        "past": "historical",
        "resolved": "historical",
        "hypothetical": "hypothetical",
        "conditional": "hypothetical",
        "unknown": "unknown",
        "unasserted": "unknown",
    }
    return aliases.get(token, "unknown")


def _normalised_token(value: Any) -> str | None:
    if value is None or value is _MISSING:
        return None
    token = str(value).strip().casefold()
    return token or None


def _resolve_policy_fingerprint(
    *,
    policy: Any,
    policy_fingerprint: str | None,
) -> str:
    if policy_fingerprint is not None:
        return _fingerprint(policy_fingerprint)
    if policy is None:
        return DEFAULT_TIMELINE_POLICY_FINGERPRINT
    return _fingerprint_from_policy(policy)


def _fingerprint_from_policy(policy: Any) -> str:
    if isinstance(policy, str):
        return _hash_value({"policy": policy.strip()})

    direct = _read(policy, ("policy_fingerprint", "fingerprint"), default=_MISSING)
    if direct is not _MISSING and direct is not None:
        return _fingerprint(direct)
    plan = _read(policy, ("plan",), default=_MISSING)
    if plan is not _MISSING:
        plan_fingerprint = _read(plan, ("fingerprint",), default=_MISSING)
        if plan_fingerprint is not _MISSING and plan_fingerprint is not None:
            return _fingerprint(plan_fingerprint)

    to_dict = _read(policy, ("to_dict",), default=_MISSING)
    value = to_dict() if callable(to_dict) else policy
    return _hash_value({"policy": value})


def _fingerprint(value: Any) -> str:
    if value is None:
        return DEFAULT_TIMELINE_POLICY_FINGERPRINT
    if isinstance(value, str):
        candidate = value.strip()
        if _HASH_RE.fullmatch(candidate):
            return candidate
        if _HEX_HASH_RE.fullmatch(candidate):
            return f"sha256:{candidate}"
    return _hash_value(value)


def _value_hash(value: Any) -> str:
    if isinstance(value, str):
        candidate = value.strip()
        if _HASH_RE.fullmatch(candidate):
            return candidate
        if _HEX_HASH_RE.fullmatch(candidate):
            return f"sha256:{candidate}"
    return _hash_value(value)


def _event_value_hash(event: Any) -> str | None:
    existing = _event_value(event, ("value_hash", "text_hash", "source_hash"))
    if existing is not _MISSING and existing is not None:
        return _value_hash(existing)
    raw_value = _event_value(
        event,
        ("value", "normalized_value", "text", "surface", "raw_value"),
    )
    if raw_value is _MISSING or raw_value is None:
        return None
    return _hash_value(raw_value)


def _sort_key(raw_event: Any, event: TimelineProvenanceEvent) -> tuple[Any, ...]:
    position = _event_value(raw_event, ("position", "order", "index"))
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        position = None
    position_missing = position is None
    return (
        position_missing,
        position if position is not None else 0,
        event.start,
        event.end,
        event.event_id,
        event.assertion_status,
        event.temporal_confidence if event.temporal_confidence is not None else -1.0,
        event.policy_fingerprint,
        event.value_hash or "",
    )


def _event_sort_key(event: TimelineProvenanceEvent) -> tuple[Any, ...]:
    """Return the canonical sort key for an already-sanitized event."""

    return (
        event.start,
        event.end,
        event.event_id,
        event.assertion_status,
        event.temporal_confidence if event.temporal_confidence is not None else -1.0,
        event.policy_fingerprint,
        event.value_hash or "",
    )


__all__ = [
    "DEFAULT_TIMELINE_POLICY_FINGERPRINT",
    "TIMELINE_PROVENANCE_DISCLAIMER",
    "TIMELINE_PROVENANCE_SCHEMA_VERSION",
    "TimelineProvenanceEvent",
    "TimelineProvenanceExport",
    "TimelineProvenanceRecord",
    "build_timeline_provenance_export",
    "build_value_free_timeline_provenance",
    "export_clinical_timeline_provenance",
    "export_timeline_provenance",
]
