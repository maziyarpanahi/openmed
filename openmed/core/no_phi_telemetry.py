"""Deterministic, aggregate-only telemetry for local OpenMed pipelines.

This module is a deliberately small privacy boundary.  Callers can record
typed counters and latency observations, but the exported representation is
limited to fixed metric names and finite dimension values.  It does not log,
persist, or send anything; an application must explicitly transport the
returned snapshot if it needs external collection.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any

SCHEMA_VERSION = 1
OTHER_DIMENSION_VALUE = "other"
UNKNOWN_EXCEPTION_CATEGORY = "unknown"

DEFAULT_LATENCY_BUCKETS_SECONDS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
)
MAX_LATENCY_BUCKETS = 32


class CounterName(str, Enum):
    """The only counter families accepted by the exporter."""

    PIPELINE_RUNS = "openmed_pipeline_runs_total"
    PIPELINE_FAILURES = "openmed_pipeline_failures_total"
    PIPELINE_REJECTIONS = "openmed_pipeline_rejections_total"
    PIPELINE_ENTITIES = "openmed_pipeline_entities_total"


class DimensionName(str, Enum):
    """Dimension keys that may appear in an exported sample."""

    STAGE = "stage"
    STATUS = "status"
    METHOD = "method"
    EXCEPTION_CATEGORY = "exception_category"


PIPELINE_LATENCY_NAME = "openmed_pipeline_latency_seconds"

PIPELINE_STAGE_VALUES: tuple[str, ...] = (
    "normalize",
    "language_script",
    "doc_type_section",
    "deterministic_detectors",
    "fast_pii_model",
    "clinical_phi_model",
    "span_arbitration",
    "policy_actions",
    "safety_sweep",
    "emit",
    "pipeline",
    OTHER_DIMENSION_VALUE,
)
PIPELINE_STATUS_VALUES: tuple[str, ...] = (
    "success",
    "error",
    "cancelled",
    "rejected",
    OTHER_DIMENSION_VALUE,
)
METHOD_VALUES: tuple[str, ...] = (
    "mask",
    "aadhaar_mask",
    "remove",
    "replace",
    "hash",
    "shift_dates",
    "format_preserve",
    OTHER_DIMENSION_VALUE,
)
EXCEPTION_CATEGORY_VALUES: tuple[str, ...] = (
    "cancelled",
    "capacity",
    "configuration",
    "dependency",
    "internal",
    "network",
    "timeout",
    "validation",
    UNKNOWN_EXCEPTION_CATEGORY,
)

_DIMENSION_NAMES = frozenset(item.value for item in DimensionName)
_DIMENSION_VALUES = {
    DimensionName.STAGE.value: frozenset(PIPELINE_STAGE_VALUES),
    DimensionName.STATUS.value: frozenset(PIPELINE_STATUS_VALUES),
    DimensionName.METHOD.value: frozenset(METHOD_VALUES),
    DimensionName.EXCEPTION_CATEGORY.value: frozenset(EXCEPTION_CATEGORY_VALUES),
}
_FAILURE_STATUSES = frozenset({"error", "cancelled", "rejected"})
_EVENT_KEYS = frozenset(
    {
        "amount",
        "counter",
        "dimensions",
        "entity_count",
        "exception",
        "exception_category",
        "latency_ms",
        "latency_seconds",
        "method",
        "name",
        "stage",
        "status",
        "value",
    }
)


class TelemetrySchemaError(ValueError):
    """Raised when input does not match the safe telemetry schema."""


class UnapprovedTelemetryKeyError(TelemetrySchemaError):
    """Raised when a caller supplies a field outside the safe allowlist."""


def sanitize_exception_category(value: object) -> str:
    """Return a bounded category without reading an exception message.

    Exception instances are classified from their type only.  A caller may
    also provide one of the documented category strings; every other string or
    object becomes ``"unknown"``.  In particular, ``str(value)`` is never
    called, so a message containing a prompt or identifier cannot enter a
    telemetry payload or an error message.
    """

    if value is None:
        return UNKNOWN_EXCEPTION_CATEGORY

    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in _DIMENSION_VALUES[DimensionName.EXCEPTION_CATEGORY.value]:
            return candidate
        return UNKNOWN_EXCEPTION_CATEGORY

    exception_type: type[BaseException] | None = None
    if isinstance(value, BaseException):
        exception_type = type(value)
    elif isinstance(value, type):
        try:
            if issubclass(value, BaseException):
                exception_type = value
        except TypeError:
            exception_type = None

    if exception_type is None:
        return UNKNOWN_EXCEPTION_CATEGORY

    if issubclass(exception_type, asyncio.CancelledError):
        return "cancelled"
    if issubclass(exception_type, TimeoutError):
        return "timeout"
    if issubclass(exception_type, MemoryError):
        return "capacity"
    if issubclass(exception_type, ConnectionError):
        return "network"
    if issubclass(exception_type, ImportError):
        return "dependency"
    if issubclass(exception_type, (ValueError, TypeError, KeyError)):
        return "validation"
    if issubclass(exception_type, BaseException):
        return "internal"
    return UNKNOWN_EXCEPTION_CATEGORY


@dataclass(frozen=True, slots=True)
class CounterSample:
    """One immutable aggregate counter sample."""

    name: CounterName
    value: int
    dimensions: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable counter sample."""

        return {
            "name": self.name.value,
            "value": self.value,
            "dimensions": dict(self.dimensions),
        }


@dataclass(frozen=True, slots=True)
class LatencySample:
    """One immutable latency histogram sample."""

    name: str
    count: int
    sum_seconds: float
    buckets: tuple[tuple[str, int], ...]
    dimensions: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable latency sample."""

        return {
            "name": self.name,
            "count": self.count,
            "sum_seconds": self.sum_seconds,
            "buckets": dict(self.buckets),
            "dimensions": dict(self.dimensions),
        }


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """A deterministic, transport-neutral telemetry snapshot."""

    schema_version: int
    counters: tuple[CounterSample, ...]
    latencies: tuple[LatencySample, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the snapshot without timestamps or source text."""

        return {
            "schema_version": self.schema_version,
            "counters": [sample.to_dict() for sample in self.counters],
            "latencies": [sample.to_dict() for sample in self.latencies],
        }

    def to_json(self) -> str:
        """Return canonical JSON with stable key and sample ordering."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass
class _LatencyAggregate:
    """Mutable internal histogram state guarded by the exporter lock."""

    count: int
    sum_seconds: float
    bucket_counts: list[int]


class NoPHITelemetryExporter:
    """Collect only allowlisted, aggregate pipeline health telemetry.

    The exporter is intentionally local and pull-oriented.  ``record`` and
    ``record_pipeline`` update in-memory counters; ``export`` returns a fresh
    dictionary and performs no I/O.  Dimension values outside the finite
    allowlists are mapped to ``"other"``.  Unknown field names are rejected so
    callers cannot accidentally turn a prompt, entity, model id, or request
    body into a label.
    """

    def __init__(
        self,
        *,
        latency_buckets_seconds: tuple[float, ...] | None = None,
    ) -> None:
        self._latency_buckets = _coerce_latency_buckets(latency_buckets_seconds)
        self._counters: dict[tuple[CounterName, tuple[tuple[str, str], ...]], int] = {}
        self._latencies: dict[tuple[tuple[str, str], ...], _LatencyAggregate] = {}
        self._lock = threading.RLock()

    @property
    def latency_buckets_seconds(self) -> tuple[float, ...]:
        """Return the fixed bucket boundaries used by this exporter."""

        return self._latency_buckets

    def increment(
        self,
        counter: CounterName | str,
        *,
        amount: int = 1,
        dimensions: Mapping[str, object] | None = None,
    ) -> None:
        """Increment an approved counter by a positive integer amount."""

        counter_name = _coerce_counter_name(counter)
        safe_amount = _coerce_positive_int(amount, "counter amount")
        safe_dimensions = _normalize_dimensions(dimensions)
        with self._lock:
            key = (counter_name, safe_dimensions)
            self._counters[key] = self._counters.get(key, 0) + safe_amount

    def observe_latency_seconds(
        self,
        seconds: Real,
        *,
        dimensions: Mapping[str, object] | None = None,
    ) -> None:
        """Record one finite, non-negative pipeline latency in seconds."""

        observed = _coerce_duration(seconds, "latency")
        safe_dimensions = _normalize_dimensions(dimensions)
        with self._lock:
            aggregate = self._latencies.get(safe_dimensions)
            if aggregate is None:
                aggregate = _LatencyAggregate(
                    count=0,
                    sum_seconds=0.0,
                    bucket_counts=[0] * (len(self._latency_buckets) + 1),
                )
                self._latencies[safe_dimensions] = aggregate
            aggregate.count += 1
            aggregate.sum_seconds += observed
            for index, boundary in enumerate(self._latency_buckets):
                if observed <= boundary:
                    aggregate.bucket_counts[index] += 1
            aggregate.bucket_counts[-1] += 1

    def observe_latency_ms(
        self,
        milliseconds: Real,
        *,
        dimensions: Mapping[str, object] | None = None,
    ) -> None:
        """Record one finite, non-negative pipeline latency in milliseconds."""

        observed = _coerce_duration(milliseconds, "latency") / 1000.0
        self.observe_latency_seconds(observed, dimensions=dimensions)

    def record_pipeline(
        self,
        *,
        stage: str = "pipeline",
        status: str = "success",
        method: str = OTHER_DIMENSION_VALUE,
        latency_ms: Real | None = None,
        latency_seconds: Real | None = None,
        entity_count: int | None = None,
        exception: object | None = None,
        exception_category: object | None = None,
    ) -> None:
        """Record one pipeline outcome using only aggregate safe fields.

        ``exception`` may be an exception instance or class.  Its message is
        intentionally ignored.  ``latency_ms`` and ``latency_seconds`` are
        mutually exclusive conveniences for callers with either unit.
        """

        if latency_ms is not None and latency_seconds is not None:
            raise TelemetrySchemaError("telemetry latency has multiple units")

        safe_status = _normalize_dimension(DimensionName.STATUS.value, status)
        safe_stage = _normalize_dimension(DimensionName.STAGE.value, stage)
        safe_method = _normalize_dimension(DimensionName.METHOD.value, method)
        safe_exception = sanitize_exception_category(
            exception if exception is not None else exception_category
        )
        dimensions: dict[str, object] = {
            DimensionName.STAGE.value: safe_stage,
            DimensionName.STATUS.value: safe_status,
            DimensionName.METHOD.value: safe_method,
            DimensionName.EXCEPTION_CATEGORY.value: safe_exception,
        }

        safe_entity_count = 0
        if entity_count is not None:
            safe_entity_count = _coerce_non_negative_int(entity_count, "entity count")

        if latency_ms is not None:
            safe_latency_seconds = _coerce_duration(latency_ms, "latency") / 1000.0
        elif latency_seconds is not None:
            safe_latency_seconds = _coerce_duration(latency_seconds, "latency")
        else:
            safe_latency_seconds = None

        with self._lock:
            self._increment_locked(CounterName.PIPELINE_RUNS, dimensions)
            if safe_status in _FAILURE_STATUSES:
                self._increment_locked(CounterName.PIPELINE_FAILURES, dimensions)
            if safe_status == "rejected":
                self._increment_locked(CounterName.PIPELINE_REJECTIONS, dimensions)
            if safe_entity_count:
                self._increment_locked(
                    CounterName.PIPELINE_ENTITIES,
                    dimensions,
                    amount=safe_entity_count,
                )
            if safe_latency_seconds is not None:
                self._observe_latency_locked(safe_latency_seconds, dimensions)

    def record_pipeline_result(
        self,
        result: object,
        *,
        status: str = "success",
        method: str = OTHER_DIMENSION_VALUE,
        exception: object | None = None,
    ) -> None:
        """Record aggregate fields from a pipeline result without inspecting text.

        Only ``stage_durations_ms`` and the length of ``spans`` are read.  The
        result may contain source or redacted text; neither is copied, hashed,
        logged, or serialized.
        """

        durations = getattr(result, "stage_durations_ms", {})
        safe_durations: list[tuple[object, float]] = []
        if isinstance(durations, Mapping):
            for stage, duration in durations.items():
                try:
                    safe_duration = _coerce_duration(duration, "stage latency")
                except TelemetrySchemaError:
                    continue
                safe_durations.append((stage, safe_duration))

        try:
            entity_count = len(getattr(result, "spans", ()))
        except (TypeError, AttributeError):
            entity_count = 0

        total_seconds = sum(duration for _, duration in safe_durations)
        self.record_pipeline(
            status=status,
            method=method,
            latency_seconds=total_seconds if safe_durations else None,
            entity_count=entity_count,
            exception=exception,
        )

        for stage, duration in safe_durations:
            self.observe_latency_seconds(
                duration,
                dimensions={
                    DimensionName.STAGE.value: stage,
                    DimensionName.STATUS.value: status,
                    DimensionName.METHOD.value: method,
                    DimensionName.EXCEPTION_CATEGORY.value: sanitize_exception_category(
                        exception
                    ),
                },
            )

    def record(self, event: Mapping[str, object]) -> None:
        """Record a schema-checked event supplied as a mapping.

        Supported keys are intentionally finite.  ``counter``/``name`` and
        ``amount``/``value`` are equivalent spellings; latency may be supplied
        in milliseconds or seconds.  This method is useful at integration
        boundaries where a typed call is inconvenient, while retaining the
        same allowlist and bounded-value behavior.
        """

        if not isinstance(event, Mapping):
            raise TelemetrySchemaError("telemetry event must be a mapping")
        if any(not isinstance(key, str) or key not in _EVENT_KEYS for key in event):
            raise UnapprovedTelemetryKeyError(
                "telemetry event contains unapproved fields"
            )

        counter = _coalesced_event_value(event, "counter", "name")
        amount = _coalesced_event_value(event, "amount", "value", default=1)
        latency_ms = event.get("latency_ms")
        latency_seconds = event.get("latency_seconds")
        if latency_ms is not None and latency_seconds is not None:
            raise TelemetrySchemaError("telemetry latency has multiple units")

        dimensions = _event_dimensions(event)
        if counter is None and latency_ms is None and latency_seconds is None:
            raise TelemetrySchemaError("telemetry event has no approved measurement")

        if counter is not None:
            self.increment(counter, amount=amount, dimensions=dimensions)
        if latency_ms is not None:
            self.observe_latency_ms(latency_ms, dimensions=dimensions)
        if latency_seconds is not None:
            self.observe_latency_seconds(latency_seconds, dimensions=dimensions)

        if "entity_count" in event:
            entity_count = _coerce_non_negative_int(
                event["entity_count"], "entity count"
            )
            if entity_count:
                self.increment(
                    CounterName.PIPELINE_ENTITIES,
                    amount=entity_count,
                    dimensions=dimensions,
                )

    record_event = record

    def clear(self) -> None:
        """Remove all in-memory samples."""

        with self._lock:
            self._counters.clear()
            self._latencies.clear()

    def snapshot(self) -> TelemetrySnapshot:
        """Return a stable, immutable copy of the current samples."""

        with self._lock:
            counters = tuple(
                CounterSample(name, value, dimensions)
                for (name, dimensions), value in sorted(
                    self._counters.items(),
                    key=lambda item: (item[0][0].value, item[0][1]),
                )
            )
            latencies: list[LatencySample] = []
            for dimensions, aggregate in sorted(self._latencies.items()):
                bucket_values = tuple(
                    (str(boundary), aggregate.bucket_counts[index])
                    for index, boundary in enumerate(self._latency_buckets)
                ) + (("+Inf", aggregate.bucket_counts[-1]),)
                latencies.append(
                    LatencySample(
                        name=PIPELINE_LATENCY_NAME,
                        count=aggregate.count,
                        sum_seconds=aggregate.sum_seconds,
                        buckets=bucket_values,
                        dimensions=dimensions,
                    )
                )
            return TelemetrySnapshot(
                schema_version=SCHEMA_VERSION,
                counters=counters,
                latencies=tuple(latencies),
            )

    def export(self) -> dict[str, Any]:
        """Return a fresh deterministic dictionary with no external I/O."""

        return self.snapshot().to_dict()

    def export_json(self) -> str:
        """Return the canonical JSON representation of ``export``."""

        return self.snapshot().to_json()

    def render_prometheus(self) -> str:
        """Render safe samples as deterministic Prometheus text.

        This is formatting only.  It does not contact a collector or read an
        endpoint configuration.
        """

        snapshot = self.snapshot()
        lines: list[str] = []
        rendered_counter_families: set[CounterName] = set()
        for counter in snapshot.counters:
            if counter.name not in rendered_counter_families:
                if counter.name is CounterName.PIPELINE_RUNS:
                    help_text = "Aggregate OpenMed pipeline runs."
                elif counter.name is CounterName.PIPELINE_FAILURES:
                    help_text = "Aggregate OpenMed pipeline failures."
                elif counter.name is CounterName.PIPELINE_REJECTIONS:
                    help_text = "Aggregate OpenMed pipeline rejections."
                else:
                    help_text = "Aggregate OpenMed pipeline entities."
                lines.append(f"# HELP {counter.name.value} {help_text}")
                lines.append(f"# TYPE {counter.name.value} counter")
                rendered_counter_families.add(counter.name)
            labels = _label_suffix(dict(counter.dimensions))
            lines.append(f"{counter.name.value}{labels} {counter.value}")

        if snapshot.latencies:
            lines.append(f"# HELP {PIPELINE_LATENCY_NAME} Pipeline latency.")
            lines.append(f"# TYPE {PIPELINE_LATENCY_NAME} histogram")
            for latency in snapshot.latencies:
                base_labels = dict(latency.dimensions)
                for boundary, value in latency.buckets:
                    labels = dict(base_labels)
                    labels["le"] = boundary
                    lines.append(
                        f"{PIPELINE_LATENCY_NAME}_bucket{_label_suffix(labels)} {value}"
                    )
                labels = _label_suffix(base_labels)
                lines.append(f"{PIPELINE_LATENCY_NAME}_count{labels} {latency.count}")
                lines.append(
                    f"{PIPELINE_LATENCY_NAME}_sum{labels} "
                    f"{_format_float(latency.sum_seconds)}"
                )

        return "\n".join(lines) + ("\n" if lines else "")

    def _increment_locked(
        self,
        counter: CounterName,
        dimensions: Mapping[str, object],
        *,
        amount: int = 1,
    ) -> None:
        safe_dimensions = _normalize_dimensions(dimensions)
        key = (counter, safe_dimensions)
        self._counters[key] = self._counters.get(key, 0) + amount

    def _observe_latency_locked(
        self,
        seconds: float,
        dimensions: Mapping[str, object],
    ) -> None:
        safe_dimensions = _normalize_dimensions(dimensions)
        aggregate = self._latencies.get(safe_dimensions)
        if aggregate is None:
            aggregate = _LatencyAggregate(
                count=0,
                sum_seconds=0.0,
                bucket_counts=[0] * (len(self._latency_buckets) + 1),
            )
            self._latencies[safe_dimensions] = aggregate
        aggregate.count += 1
        aggregate.sum_seconds += seconds
        for index, boundary in enumerate(self._latency_buckets):
            if seconds <= boundary:
                aggregate.bucket_counts[index] += 1
        aggregate.bucket_counts[-1] += 1


def _coerce_latency_buckets(
    buckets: tuple[float, ...] | None,
) -> tuple[float, ...]:
    selected = DEFAULT_LATENCY_BUCKETS_SECONDS if buckets is None else buckets
    if not isinstance(selected, tuple) or not selected:
        raise TelemetrySchemaError("latency buckets must be a non-empty tuple")
    if len(selected) > MAX_LATENCY_BUCKETS:
        raise TelemetrySchemaError("latency buckets exceed the safe limit")

    normalized: list[float] = []
    for bucket in selected:
        if isinstance(bucket, bool) or not isinstance(bucket, Real):
            raise TelemetrySchemaError("latency buckets must be finite numbers")
        value = float(bucket)
        if not math.isfinite(value) or value <= 0:
            raise TelemetrySchemaError("latency buckets must be finite and positive")
        normalized.append(value)
    if normalized != sorted(set(normalized)):
        raise TelemetrySchemaError("latency buckets must be strictly increasing")
    return tuple(normalized)


def _coerce_counter_name(value: object) -> CounterName:
    if isinstance(value, CounterName):
        return value
    if isinstance(value, str):
        try:
            return CounterName(value)
        except ValueError:
            pass
    raise TelemetrySchemaError("telemetry counter is not approved")


def _coerce_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TelemetrySchemaError(f"{field_name} must be a positive integer")
    return value


def _coerce_non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TelemetrySchemaError(f"{field_name} must be a non-negative integer")
    return value


def _coerce_duration(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TelemetrySchemaError(f"{field_name} must be a finite number")
    observed = float(value)
    if not math.isfinite(observed) or observed < 0:
        raise TelemetrySchemaError(f"{field_name} must be finite and non-negative")
    return observed


def _normalize_dimension(name: str, value: object) -> str:
    allowed = _DIMENSION_VALUES[name]
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return OTHER_DIMENSION_VALUE
    candidate = value.strip().lower()
    return candidate if candidate in allowed else OTHER_DIMENSION_VALUE


def _normalize_dimensions(
    dimensions: Mapping[str, object] | None,
) -> tuple[tuple[str, str], ...]:
    if dimensions is None:
        return ()
    if not isinstance(dimensions, Mapping):
        raise TelemetrySchemaError("telemetry dimensions must be a mapping")
    if any(
        not isinstance(name, str) or name not in _DIMENSION_NAMES for name in dimensions
    ):
        raise UnapprovedTelemetryKeyError(
            "telemetry dimensions contain unapproved fields"
        )
    return tuple(
        sorted(
            (
                name,
                _normalize_dimension(name, value),
            )
            for name, value in dimensions.items()
        )
    )


def _coalesced_event_value(
    event: Mapping[str, object],
    first_name: str,
    second_name: str,
    *,
    default: object | None = None,
) -> object | None:
    first_present = first_name in event
    second_present = second_name in event
    if first_present and second_present:
        raise TelemetrySchemaError("telemetry event contains duplicate fields")
    if first_present:
        return event[first_name]
    if second_present:
        return event[second_name]
    return default


def _event_dimensions(event: Mapping[str, object]) -> dict[str, object]:
    raw_dimensions = event.get("dimensions")
    if raw_dimensions is None:
        dimensions: dict[str, object] = {}
    elif isinstance(raw_dimensions, Mapping):
        dimensions = dict(raw_dimensions)
    else:
        raise TelemetrySchemaError("telemetry dimensions must be a mapping")

    for name in (
        DimensionName.STAGE.value,
        DimensionName.STATUS.value,
        DimensionName.METHOD.value,
    ):
        if name in event:
            if name in dimensions:
                raise TelemetrySchemaError("telemetry event contains duplicate fields")
            dimensions[name] = event[name]

    if "exception" in event or "exception_category" in event:
        if DimensionName.EXCEPTION_CATEGORY.value in dimensions:
            raise TelemetrySchemaError("telemetry event contains duplicate fields")
        value = event.get("exception")
        if value is None:
            value = event.get("exception_category")
        dimensions[DimensionName.EXCEPTION_CATEGORY.value] = (
            sanitize_exception_category(value)
        )
    return dimensions


def _label_suffix(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    rendered = ",".join(
        f'{name}="{_escape_label_value(labels[name])}"' for name in sorted(labels)
    )
    return "{" + rendered + "}"


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_float(value: float) -> str:
    return f"{value:.12g}"


__all__ = [
    "CounterName",
    "CounterSample",
    "DEFAULT_LATENCY_BUCKETS_SECONDS",
    "DimensionName",
    "EXCEPTION_CATEGORY_VALUES",
    "LatencySample",
    "METHOD_VALUES",
    "NoPHITelemetryExporter",
    "OTHER_DIMENSION_VALUE",
    "PIPELINE_LATENCY_NAME",
    "PIPELINE_STAGE_VALUES",
    "PIPELINE_STATUS_VALUES",
    "SCHEMA_VERSION",
    "TelemetrySchemaError",
    "TelemetrySnapshot",
    "UnapprovedTelemetryKeyError",
    "sanitize_exception_category",
]
