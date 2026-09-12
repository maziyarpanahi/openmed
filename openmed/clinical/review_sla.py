"""Deterministic, PHI-safe SLA summaries for human-review queues.

The module deliberately requires an injected clock.  Queue entries are
classified locally using their enqueue time, priority, and expiry deadline;
the report contains counts only.  Detailed records retain an opaque,
deterministic key instead of the input case key.

This is an operational review aid, not a compliance certification or a
clinical decision mechanism.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from math import isfinite
from types import MappingProxyType
from typing import Any, TypeAlias

__all__ = [
    "AGE_BUCKETS",
    "DEFAULT_OPAQUE_KEY_NAMESPACE",
    "DEFAULT_PRIORITY_SLA",
    "EXPIRY_BUCKETS",
    "OVERDUE_BUCKETS",
    "PRIORITY_LEVELS",
    "REVIEW_SLA_SCHEMA_VERSION",
    "ReviewQueueCase",
    "ReviewSLARecord",
    "ReviewSLAReport",
    "build_review_sla_report",
    "build_sla_report",
    "compute_review_sla",
    "opaque_case_key",
    "render_review_sla_report",
    "render_sla_report",
]

REVIEW_SLA_SCHEMA_VERSION = "review-sla.v1"
DEFAULT_OPAQUE_KEY_NAMESPACE = "openmed-review-sla.v1"

PRIORITY_LEVELS = ("urgent", "high", "normal", "low")
DEFAULT_PRIORITY_SLA: Mapping[str, timedelta] = MappingProxyType(
    {
        "urgent": timedelta(hours=4),
        "high": timedelta(hours=8),
        "normal": timedelta(hours=24),
        "low": timedelta(hours=72),
    }
)

AGE_BUCKETS = ("0-1h", "1-4h", "4-24h", "24h+")
EXPIRY_BUCKETS = ("expired", "due-within-4h", "due-after-4h")
OVERDUE_BUCKETS = ("on-time", "0-24h-overdue", "24h+-overdue")

_EXPIRY_SOON = timedelta(hours=4)
_OVERDUE_DAY = timedelta(hours=24)
_MISSING = object()
_CASE_KEY_FIELDS = ("case_key", "case_id", "id", "key")
_QUEUED_AT_FIELDS = ("queued_at", "enqueued_at", "created_at", "submitted_at")
_PRIORITY_FIELDS = ("priority", "urgency")
_EXPIRY_FIELDS = ("expires_at", "expiry_at", "due_at")
_SLA_FIELDS = ("sla", "sla_duration", "sla_seconds")

Clock: TypeAlias = Callable[[], datetime]


@dataclass(frozen=True)
class ReviewQueueCase:
    """Input-only queue entry used by :func:`compute_review_sla`.

    ``case_key`` is accepted for local classification but is never emitted by
    this module.  ``expires_at`` takes precedence over ``sla``; when neither
    is supplied, the default duration for the canonical priority is used.
    """

    case_key: str | int = field(repr=False)
    queued_at: datetime
    priority: str = "normal"
    expires_at: datetime | None = None
    sla: timedelta | int | float | None = None


@dataclass(frozen=True)
class ReviewSLARecord:
    """PHI-safe classification for one queue entry.

    The ``case_key`` property and serialized ``case_key`` field are opaque
    hashes.  No raw input key, case contents, reviewer identity, or metadata
    is retained.
    """

    opaque_case_key: str
    queued_at: str
    priority: str
    age_seconds: int
    age_bucket: str
    expires_at: str
    expiry_bucket: str
    overdue_seconds: int
    overdue_bucket: str

    @property
    def case_key(self) -> str:
        """Return the stable opaque key used in this record."""

        return self.opaque_case_key

    @property
    def is_overdue(self) -> bool:
        """Whether the entry is past its expiry deadline."""

        return self.overdue_seconds > 0 or self.overdue_bucket != "on-time"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record without raw case data."""

        return {
            "case_key": self.opaque_case_key,
            "queued_at": self.queued_at,
            "priority": self.priority,
            "age_seconds": self.age_seconds,
            "age_bucket": self.age_bucket,
            "expires_at": self.expires_at,
            "expiry_bucket": self.expiry_bucket,
            "overdue_seconds": self.overdue_seconds,
            "overdue_bucket": self.overdue_bucket,
        }


@dataclass(frozen=True)
class ReviewSLAReport:
    """Counts-only SLA report for one injected point in time."""

    as_of: str
    total_cases: int
    priority_counts: Mapping[str, int]
    age_counts: Mapping[str, int]
    expiry_counts: Mapping[str, int]
    overdue_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if (
            isinstance(self.total_cases, bool)
            or not isinstance(self.total_cases, int)
            or self.total_cases < 0
        ):
            raise ValueError("total_cases must be a non-negative integer")
        for field_name, labels in (
            ("priority_counts", PRIORITY_LEVELS),
            ("age_counts", AGE_BUCKETS),
            ("expiry_counts", EXPIRY_BUCKETS),
            ("overdue_counts", OVERDUE_BUCKETS),
        ):
            counts = getattr(self, field_name)
            normalized: dict[str, int] = {}
            for label in labels:
                value = counts.get(label, 0)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("report counts must be non-negative integers")
                normalized[label] = value
            if set(counts) - set(labels):
                raise ValueError("report contains an unsupported bucket")
            object.__setattr__(self, field_name, MappingProxyType(normalized))

    @property
    def priority_buckets(self) -> Mapping[str, int]:
        """Alias for priority counts."""

        return self.priority_counts

    @property
    def age_buckets(self) -> Mapping[str, int]:
        """Alias for age counts."""

        return self.age_counts

    @property
    def expiry_buckets(self) -> Mapping[str, int]:
        """Alias for expiry counts."""

        return self.expiry_counts

    @property
    def overdue_buckets(self) -> Mapping[str, int]:
        """Alias for overdue counts."""

        return self.overdue_counts

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic counts-only report payload."""

        return {
            "schema_version": REVIEW_SLA_SCHEMA_VERSION,
            "as_of": self.as_of,
            "total_cases": self.total_cases,
            "priority_counts": dict(self.priority_counts),
            "age_counts": dict(self.age_counts),
            "expiry_counts": dict(self.expiry_counts),
            "overdue_counts": dict(self.overdue_counts),
        }

    def to_json(self) -> str:
        """Serialize the report with a stable key and bucket order."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
        )


def opaque_case_key(
    case_key: str | int,
    *,
    namespace: str = DEFAULT_OPAQUE_KEY_NAMESPACE,
) -> str:
    """Return a stable SHA-256 key without exposing the input value.

    The namespace is part of the digest domain so the same synthetic key can
    be used safely by separate local report types.  Callers that need a
    deployment-specific pseudonym can provide a stable, non-sensitive
    namespace; no random value is generated by default.
    """

    normalized_key = _normalize_case_key(case_key)
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    digest = sha256(
        namespace.encode("utf-8") + b"\0" + normalized_key.encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def compute_review_sla(
    cases: Iterable[ReviewQueueCase | Mapping[str, Any] | object],
    *,
    now: datetime | str | None = None,
    clock: Clock | object | None = None,
    priority_sla: Mapping[str, timedelta | int | float] | None = None,
    namespace: str = DEFAULT_OPAQUE_KEY_NAMESPACE,
) -> tuple[ReviewSLARecord, ...]:
    """Classify queue entries at an explicitly injected point in time.

    Args:
        cases: Queue entries or mappings containing a case key, a
            ``queued_at``/``enqueued_at`` timestamp, and an optional priority.
            Mappings may provide ``expires_at`` or a positive ``sla`` duration
            in seconds.  The input key is hashed and never copied to output.
        now: Fixed clock value.  Mutually exclusive with ``clock``.
        clock: Callable returning a clock value, or an object exposing a
            callable ``now()`` method.  One of ``now`` or ``clock`` is
            required; the system clock is never consulted.
        priority_sla: Optional overrides, in seconds or timedeltas, merged
            with :data:`DEFAULT_PRIORITY_SLA`.
        namespace: Stable digest namespace for opaque case keys.

    Returns:
        Deterministically ordered, PHI-safe records.

    Raises:
        ValueError: If the clock, timestamps, priorities, durations, or case
            keys are invalid, or if a case key is duplicated.
        TypeError: If the case collection or a case field has an unsupported
            type.
    """

    as_of = _resolve_clock(now, clock)
    durations = _priority_durations(priority_sla)
    source_cases = _as_case_iterable(cases)
    records: list[ReviewSLARecord] = []
    seen_keys: set[str] = set()

    for source_case in source_cases:
        case = _coerce_case(source_case)
        key = opaque_case_key(case.case_key, namespace=namespace)
        if key in seen_keys:
            raise ValueError("case keys must be unique")
        seen_keys.add(key)

        queued_at = _coerce_datetime(case.queued_at, "queued_at")
        if queued_at > as_of:
            raise ValueError("queued_at cannot be after the injected clock")

        priority = _normalize_priority(case.priority)
        expiry = _resolve_expiry(case, queued_at, priority, durations)
        if expiry < queued_at:
            raise ValueError("expiry cannot be before queued_at")

        age = as_of - queued_at
        overdue = as_of - expiry
        age_seconds = int(age.total_seconds())
        overdue_seconds = max(0, int(overdue.total_seconds()))

        records.append(
            ReviewSLARecord(
                opaque_case_key=key,
                queued_at=queued_at.isoformat(),
                priority=priority,
                age_seconds=age_seconds,
                age_bucket=_age_bucket(age),
                expires_at=expiry.isoformat(),
                expiry_bucket=_expiry_bucket(as_of, expiry),
                overdue_seconds=overdue_seconds,
                overdue_bucket=_overdue_bucket(overdue),
            )
        )

    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.queued_at,
                record.priority,
                record.opaque_case_key,
            ),
        )
    )


def build_review_sla_report(
    cases: Iterable[ReviewQueueCase | Mapping[str, Any] | object],
    *,
    now: datetime | str | None = None,
    clock: Clock | object | None = None,
    priority_sla: Mapping[str, timedelta | int | float] | None = None,
    namespace: str = DEFAULT_OPAQUE_KEY_NAMESPACE,
) -> ReviewSLAReport:
    """Build a counts-only SLA report from a local synthetic queue."""

    as_of = _resolve_clock(now, clock)
    records = compute_review_sla(
        cases,
        now=as_of,
        priority_sla=priority_sla,
        namespace=namespace,
    )
    return ReviewSLAReport(
        as_of=as_of.isoformat(),
        total_cases=len(records),
        priority_counts=_count_records(records, "priority", PRIORITY_LEVELS),
        age_counts=_count_records(records, "age_bucket", AGE_BUCKETS),
        expiry_counts=_count_records(records, "expiry_bucket", EXPIRY_BUCKETS),
        overdue_counts=_count_records(records, "overdue_bucket", OVERDUE_BUCKETS),
    )


def render_review_sla_report(
    cases: Iterable[ReviewQueueCase | Mapping[str, Any] | object],
    *,
    now: datetime | str | None = None,
    clock: Clock | object | None = None,
    priority_sla: Mapping[str, timedelta | int | float] | None = None,
    namespace: str = DEFAULT_OPAQUE_KEY_NAMESPACE,
) -> str:
    """Render a deterministic JSON counts-only report."""

    return build_review_sla_report(
        cases,
        now=now,
        clock=clock,
        priority_sla=priority_sla,
        namespace=namespace,
    ).to_json()


def _as_case_iterable(
    cases: Iterable[ReviewQueueCase | Mapping[str, Any] | object],
) -> Iterable[ReviewQueueCase | Mapping[str, Any] | object]:
    if isinstance(cases, (ReviewQueueCase, Mapping)):
        return (cases,)
    if isinstance(cases, (str, bytes)):
        raise TypeError("cases must be an iterable of queue entries")
    try:
        iter(cases)
    except TypeError as exc:
        raise TypeError("cases must be an iterable of queue entries") from exc
    return cases


def _coerce_case(
    source: ReviewQueueCase | Mapping[str, Any] | object,
) -> ReviewQueueCase:
    if isinstance(source, ReviewQueueCase):
        return source

    case_key = _field_value(source, _CASE_KEY_FIELDS)
    queued_at = _field_value(source, _QUEUED_AT_FIELDS)
    if case_key is _MISSING:
        raise ValueError("a case key is required")
    if queued_at is _MISSING:
        raise ValueError("queued_at is required")

    priority = _field_value(source, _PRIORITY_FIELDS)
    expires_at = _field_value(source, _EXPIRY_FIELDS)
    sla = _field_value(source, _SLA_FIELDS)
    return ReviewQueueCase(
        case_key=case_key,
        queued_at=queued_at,
        priority="normal" if priority is _MISSING else priority,
        expires_at=None if expires_at is _MISSING else expires_at,
        sla=None if sla is _MISSING else sla,
    )


def _field_value(source: object, names: tuple[str, ...]) -> object:
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    return _MISSING


def _normalize_case_key(case_key: str | int) -> str:
    if isinstance(case_key, bool):
        raise TypeError("case key must be a non-empty string or integer")
    if isinstance(case_key, int):
        normalized = str(case_key)
    elif isinstance(case_key, str):
        normalized = case_key
    else:
        raise TypeError("case key must be a non-empty string or integer")
    if not normalized:
        raise ValueError("case key must be a non-empty string or integer")
    return normalized


def _coerce_datetime(value: datetime | str, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"{name} must be a valid datetime") from None
    else:
        raise TypeError(f"{name} must be a datetime or ISO datetime string")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve_clock(
    now: datetime | str | None,
    clock: Clock | object | None,
) -> datetime:
    if now is not None and clock is not None:
        raise ValueError("provide either now or clock, not both")
    if now is not None:
        return _coerce_datetime(now, "now")
    if clock is None:
        raise ValueError("an injected clock is required")
    if callable(clock):
        observed = clock()
    else:
        now_method = getattr(clock, "now", None)
        if not callable(now_method):
            raise TypeError("clock must be callable or expose now()")
        observed = now_method()
    return _coerce_datetime(observed, "clock")


_PRIORITY_ALIASES = {
    "urgent": "urgent",
    "critical": "urgent",
    "immediate": "urgent",
    "p0": "urgent",
    "high": "high",
    "p1": "high",
    "normal": "normal",
    "standard": "normal",
    "medium": "normal",
    "default": "normal",
    "p2": "normal",
    "low": "low",
    "background": "low",
    "p3": "low",
}


def _normalize_priority(priority: str | int) -> str:
    if isinstance(priority, bool) or not isinstance(priority, (str, int)):
        raise TypeError("priority must be a supported string")
    token = str(priority).strip().lower().replace("_", "-").replace(" ", "-")
    try:
        return _PRIORITY_ALIASES[token]
    except KeyError:
        raise ValueError(
            "priority must be one of urgent, high, normal, or low"
        ) from None


def _duration(value: timedelta | int | float, name: str) -> timedelta:
    if isinstance(value, timedelta):
        duration = value
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive duration")
    elif not isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a positive duration")
    else:
        duration = timedelta(seconds=float(value))
    if duration <= timedelta(0) or not isfinite(duration.total_seconds()):
        raise ValueError(f"{name} must be a positive duration")
    return duration


def _priority_durations(
    overrides: Mapping[str, timedelta | int | float] | None,
) -> dict[str, timedelta]:
    durations = dict(DEFAULT_PRIORITY_SLA)
    if overrides is None:
        return durations
    if not isinstance(overrides, Mapping):
        raise TypeError("priority_sla must be a mapping")
    for priority, value in overrides.items():
        canonical = _normalize_priority(priority)
        durations[canonical] = _duration(value, "priority SLA")
    return durations


def _resolve_expiry(
    case: ReviewQueueCase,
    queued_at: datetime,
    priority: str,
    durations: Mapping[str, timedelta],
) -> datetime:
    if case.expires_at is not None:
        return _coerce_datetime(case.expires_at, "expires_at")
    if case.sla is not None:
        return queued_at + _duration(case.sla, "sla")
    return queued_at + durations[priority]


def _age_bucket(age: timedelta) -> str:
    if age < timedelta(hours=1):
        return AGE_BUCKETS[0]
    if age < timedelta(hours=4):
        return AGE_BUCKETS[1]
    if age < timedelta(hours=24):
        return AGE_BUCKETS[2]
    return AGE_BUCKETS[3]


def _expiry_bucket(as_of: datetime, expires_at: datetime) -> str:
    remaining = expires_at - as_of
    if remaining < timedelta(0):
        return EXPIRY_BUCKETS[0]
    if remaining <= _EXPIRY_SOON:
        return EXPIRY_BUCKETS[1]
    return EXPIRY_BUCKETS[2]


def _overdue_bucket(overdue: timedelta) -> str:
    if overdue <= timedelta(0):
        return OVERDUE_BUCKETS[0]
    if overdue <= _OVERDUE_DAY:
        return OVERDUE_BUCKETS[1]
    return OVERDUE_BUCKETS[2]


def _count_records(
    records: Iterable[ReviewSLARecord],
    attribute: str,
    labels: tuple[str, ...],
) -> dict[str, int]:
    counts = Counter(getattr(record, attribute) for record in records)
    return {label: counts[label] for label in labels}


build_sla_report = build_review_sla_report
render_sla_report = render_review_sla_report
