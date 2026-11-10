"""Deterministic, aggregate-only telemetry for local OpenMed pipelines.

This module is a deliberately small privacy boundary.  Callers can record
typed counters and latency observations, but the exported representation is
limited to fixed metric names and finite dimension values.  It does not log,
persist, or send anything; an application must explicitly transport the
returned snapshot if it needs external collection.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import threading
from collections.abc import Mapping, Sized
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from numbers import Real
from typing import Any, Final

SCHEMA_VERSION: Final = 1
OTHER_DIMENSION_VALUE: Final = "other"
UNKNOWN_EXCEPTION_CATEGORY: Final = "unknown"
MAX_COUNTER_VALUE: Final = (1 << 63) - 1
MAX_ENTITY_COUNT: Final = 10_000_000
MAX_LATENCY_SECONDS: Final = 604_800.0
MAX_AGGREGATE_LATENCY_SECONDS: Final = 1_000_000_000_000_000.0
MAX_RESULT_STAGE_DURATIONS: Final = 64
MAX_SNAPSHOT_SAMPLES: Final = 40_000

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
MAX_LATENCY_BUCKETS: Final = 32


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


def _snapshot_mapping(
    value: Mapping[object, object],
    *,
    field_name: str,
    max_items: int,
) -> dict[str, object]:
    """Copy a bounded mapping while sanitizing protocol failures."""

    try:
        items = list(itertools.islice(value.items(), max_items + 1))
    except Exception:  # noqa: BLE001 - mappings are caller-controlled protocols.
        raise TelemetrySchemaError(f"{field_name} could not be read") from None
    if len(items) > max_items:
        raise UnapprovedTelemetryKeyError(f"{field_name} contains unapproved fields")

    result: dict[str, object] = {}
    for item in items:
        if type(item) not in {list, tuple} or len(item) != 2:
            raise TelemetrySchemaError(f"{field_name} contains an invalid entry")
        key, item_value = item
        if type(key) is not str or key in result:
            raise UnapprovedTelemetryKeyError(
                f"{field_name} contains unapproved fields"
            )
        result[key] = item_value
    return result


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

    if type(value) is str:
        if len(value) > 64:
            return UNKNOWN_EXCEPTION_CATEGORY
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

    def __post_init__(self) -> None:
        if type(self.name) is not CounterName:
            raise TelemetrySchemaError("counter sample name is not approved")
        value = _coerce_positive_int(self.value, "counter sample value")
        dimensions = _validate_dimension_tuple(self.dimensions)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "dimensions", dimensions)

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

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name != PIPELINE_LATENCY_NAME:
            raise TelemetrySchemaError("latency sample name is not approved")
        count = _coerce_positive_int(self.count, "latency sample count")
        sum_seconds = _coerce_bounded_float(
            self.sum_seconds,
            "latency sample sum",
            maximum=MAX_AGGREGATE_LATENCY_SECONDS,
        )
        if sum_seconds > count * MAX_LATENCY_SECONDS:
            raise TelemetrySchemaError("latency sample sum is inconsistent")
        dimensions = _validate_dimension_tuple(self.dimensions)
        buckets = _validate_bucket_tuple(self.buckets, count=count)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "sum_seconds", sum_seconds)
        object.__setattr__(self, "buckets", buckets)
        object.__setattr__(self, "dimensions", dimensions)

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

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
        ):
            raise TelemetrySchemaError("telemetry snapshot schema is unsupported")
        if type(self.counters) is not tuple or type(self.latencies) is not tuple:
            raise TelemetrySchemaError("telemetry snapshot samples must be tuples")
        if len(self.counters) + len(self.latencies) > MAX_SNAPSHOT_SAMPLES:
            raise TelemetrySchemaError("telemetry snapshot exceeds the safe limit")
        if any(type(sample) is not CounterSample for sample in self.counters):
            raise TelemetrySchemaError("telemetry snapshot counters are invalid")
        if any(type(sample) is not LatencySample for sample in self.latencies):
            raise TelemetrySchemaError("telemetry snapshot latencies are invalid")

        counter_keys = tuple(
            (sample.name.value, sample.dimensions) for sample in self.counters
        )
        latency_keys = tuple(sample.dimensions for sample in self.latencies)
        if counter_keys != tuple(sorted(set(counter_keys))):
            raise TelemetrySchemaError("telemetry snapshot counters are not canonical")
        if latency_keys != tuple(sorted(set(latency_keys))):
            raise TelemetrySchemaError("telemetry snapshot latencies are not canonical")

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
    sum_seconds: Decimal
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
            self._record_batch_locked(
                counter_updates=((counter_name, safe_dimensions, safe_amount),),
                latency_updates=(),
            )

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
            self._record_batch_locked(
                counter_updates=(),
                latency_updates=((observed, safe_dimensions),),
            )

    def observe_latency_ms(
        self,
        milliseconds: Real,
        *,
        dimensions: Mapping[str, object] | None = None,
    ) -> None:
        """Record one finite, non-negative pipeline latency in milliseconds."""

        observed = _coerce_duration_ms(milliseconds, "latency")
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
        if exception is not None and exception_category is not None:
            raise TelemetrySchemaError(
                "telemetry exception category has multiple sources"
            )

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
            safe_entity_count = _coerce_non_negative_int(
                entity_count,
                "entity count",
                maximum=MAX_ENTITY_COUNT,
            )

        if latency_ms is not None:
            safe_latency_seconds = _coerce_duration_ms(latency_ms, "latency")
        elif latency_seconds is not None:
            safe_latency_seconds = _coerce_duration(latency_seconds, "latency")
        else:
            safe_latency_seconds = None

        safe_dimensions = _normalize_dimensions(dimensions)
        counter_updates = [(CounterName.PIPELINE_RUNS, safe_dimensions, 1)]
        if safe_status in _FAILURE_STATUSES:
            counter_updates.append((CounterName.PIPELINE_FAILURES, safe_dimensions, 1))
        if safe_status == "rejected":
            counter_updates.append(
                (CounterName.PIPELINE_REJECTIONS, safe_dimensions, 1)
            )
        if safe_entity_count:
            counter_updates.append(
                (CounterName.PIPELINE_ENTITIES, safe_dimensions, safe_entity_count)
            )
        latency_updates = (
            ()
            if safe_latency_seconds is None
            else ((safe_latency_seconds, safe_dimensions),)
        )
        with self._lock:
            self._record_batch_locked(
                counter_updates=tuple(counter_updates),
                latency_updates=latency_updates,
            )

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

        try:
            durations = getattr(result, "stage_durations_ms", {})
            spans = getattr(result, "spans", ())
        except Exception:  # noqa: BLE001 - result attributes are caller-controlled.
            raise TelemetrySchemaError("pipeline result could not be read") from None
        if not isinstance(durations, Mapping):
            raise TelemetrySchemaError("pipeline result durations must be a mapping")
        if isinstance(spans, (str, bytes, bytearray, Mapping)) or not isinstance(
            spans, Sized
        ):
            raise TelemetrySchemaError("pipeline result spans must be a sequence")

        duration_items = _snapshot_mapping(
            durations,
            field_name="pipeline result durations",
            max_items=MAX_RESULT_STAGE_DURATIONS,
        )
        safe_durations = tuple(
            (
                _normalize_dimension(DimensionName.STAGE.value, stage),
                _coerce_duration_ms(duration, "stage latency"),
            )
            for stage, duration in duration_items.items()
        )
        try:
            entity_count = len(spans)
        except Exception:  # noqa: BLE001 - span containers are caller-controlled.
            raise TelemetrySchemaError(
                "pipeline result spans could not be read"
            ) from None
        safe_entity_count = _coerce_non_negative_int(
            entity_count,
            "entity count",
            maximum=MAX_ENTITY_COUNT,
        )

        safe_status = _normalize_dimension(DimensionName.STATUS.value, status)
        safe_method = _normalize_dimension(DimensionName.METHOD.value, method)
        safe_exception = sanitize_exception_category(exception)

        def dimensions_for(stage: str) -> tuple[tuple[str, str], ...]:
            return _normalize_dimensions(
                {
                    DimensionName.STAGE.value: stage,
                    DimensionName.STATUS.value: safe_status,
                    DimensionName.METHOD.value: safe_method,
                    DimensionName.EXCEPTION_CATEGORY.value: safe_exception,
                }
            )

        pipeline_dimensions = dimensions_for("pipeline")
        counter_updates = [(CounterName.PIPELINE_RUNS, pipeline_dimensions, 1)]
        if safe_status in _FAILURE_STATUSES:
            counter_updates.append(
                (CounterName.PIPELINE_FAILURES, pipeline_dimensions, 1)
            )
        if safe_status == "rejected":
            counter_updates.append(
                (CounterName.PIPELINE_REJECTIONS, pipeline_dimensions, 1)
            )
        if safe_entity_count:
            counter_updates.append(
                (
                    CounterName.PIPELINE_ENTITIES,
                    pipeline_dimensions,
                    safe_entity_count,
                )
            )

        latency_updates = [
            (duration, dimensions_for(stage)) for stage, duration in safe_durations
        ]
        if safe_durations:
            total_seconds = _coerce_duration(
                math.fsum(duration for _, duration in safe_durations),
                "pipeline latency",
            )
            latency_updates.append((total_seconds, pipeline_dimensions))

        with self._lock:
            self._record_batch_locked(
                counter_updates=tuple(counter_updates),
                latency_updates=tuple(latency_updates),
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
        values = _snapshot_mapping(
            event,
            field_name="telemetry event",
            max_items=len(_EVENT_KEYS),
        )
        if set(values) - _EVENT_KEYS:
            raise UnapprovedTelemetryKeyError(
                "telemetry event contains unapproved fields"
            )

        counter = _coalesced_event_value(values, "counter", "name")
        amount = _coalesced_event_value(values, "amount", "value", default=1)
        latency_ms = values.get("latency_ms")
        latency_seconds = values.get("latency_seconds")
        if latency_ms is not None and latency_seconds is not None:
            raise TelemetrySchemaError("telemetry latency has multiple units")

        dimensions = _normalize_dimensions(_event_dimensions(values))
        if counter is None and latency_ms is None and latency_seconds is None:
            raise TelemetrySchemaError("telemetry event has no approved measurement")
        if counter is None and ({"amount", "value"} & set(values)):
            raise TelemetrySchemaError("telemetry amount requires a counter")

        counter_updates: list[tuple[CounterName, tuple[tuple[str, str], ...], int]] = []
        latency_updates: list[tuple[float, tuple[tuple[str, str], ...]]] = []

        if counter is not None:
            counter_updates.append(
                (
                    _coerce_counter_name(counter),
                    dimensions,
                    _coerce_positive_int(amount, "counter amount"),
                )
            )
        if latency_ms is not None:
            latency_updates.append(
                (_coerce_duration_ms(latency_ms, "latency"), dimensions)
            )
        if latency_seconds is not None:
            latency_updates.append(
                (_coerce_duration(latency_seconds, "latency"), dimensions)
            )

        if "entity_count" in values:
            entity_count = _coerce_non_negative_int(
                values["entity_count"],
                "entity count",
                maximum=MAX_ENTITY_COUNT,
            )
            if entity_count:
                counter_updates.append(
                    (CounterName.PIPELINE_ENTITIES, dimensions, entity_count)
                )

        with self._lock:
            self._record_batch_locked(
                counter_updates=tuple(counter_updates),
                latency_updates=tuple(latency_updates),
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
                        sum_seconds=float(aggregate.sum_seconds),
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

    def _record_batch_locked(
        self,
        *,
        counter_updates: tuple[
            tuple[CounterName, tuple[tuple[str, str], ...], int], ...
        ],
        latency_updates: tuple[tuple[float, tuple[tuple[str, str], ...]], ...],
    ) -> None:
        """Validate and apply one all-or-nothing telemetry update batch."""

        counter_deltas: dict[tuple[CounterName, tuple[tuple[str, str], ...]], int] = {}
        for counter, dimensions, amount in counter_updates:
            key = (counter, dimensions)
            delta = counter_deltas.get(key, 0) + amount
            if delta > MAX_COUNTER_VALUE:
                raise TelemetrySchemaError("telemetry counter exceeds the safe limit")
            counter_deltas[key] = delta

        latency_groups: dict[tuple[tuple[str, str], ...], list[float]] = {}
        for observed, dimensions in latency_updates:
            latency_groups.setdefault(dimensions, []).append(observed)

        for key, delta in counter_deltas.items():
            if self._counters.get(key, 0) > MAX_COUNTER_VALUE - delta:
                raise TelemetrySchemaError("telemetry counter exceeds the safe limit")
        projected_latency_sums: dict[tuple[tuple[str, str], ...], Decimal] = {}
        for dimensions, observations in latency_groups.items():
            aggregate = self._latencies.get(dimensions)
            current_count = 0 if aggregate is None else aggregate.count
            if current_count > MAX_COUNTER_VALUE - len(observations):
                raise TelemetrySchemaError(
                    "telemetry latency count exceeds the safe limit"
                )
            current_sum = Decimal(0) if aggregate is None else aggregate.sum_seconds
            projected_sum = current_sum + sum(
                (Decimal(str(value)) for value in observations),
                start=Decimal(0),
            )
            if projected_sum > Decimal(str(MAX_AGGREGATE_LATENCY_SECONDS)):
                raise TelemetrySchemaError(
                    "telemetry latency sum exceeds the safe limit"
                )
            projected_latency_sums[dimensions] = projected_sum

        for key, delta in counter_deltas.items():
            self._counters[key] = self._counters.get(key, 0) + delta
        for dimensions, observations in latency_groups.items():
            aggregate = self._latencies.get(dimensions)
            if aggregate is None:
                aggregate = _LatencyAggregate(
                    count=0,
                    sum_seconds=Decimal(0),
                    bucket_counts=[0] * (len(self._latency_buckets) + 1),
                )
                self._latencies[dimensions] = aggregate
            aggregate.count += len(observations)
            aggregate.sum_seconds = projected_latency_sums[dimensions]
            for observed in observations:
                for index, boundary in enumerate(self._latency_buckets):
                    if observed <= boundary:
                        aggregate.bucket_counts[index] += 1
                aggregate.bucket_counts[-1] += 1


def _coerce_latency_buckets(
    buckets: tuple[float, ...] | None,
) -> tuple[float, ...]:
    selected = DEFAULT_LATENCY_BUCKETS_SECONDS if buckets is None else buckets
    if type(selected) is not tuple or not selected:
        raise TelemetrySchemaError("latency buckets must be a non-empty tuple")
    if len(selected) > MAX_LATENCY_BUCKETS:
        raise TelemetrySchemaError("latency buckets exceed the safe limit")

    normalized: list[float] = []
    for bucket in selected:
        value = _coerce_bounded_float(
            bucket,
            "latency bucket",
            maximum=MAX_LATENCY_SECONDS,
        )
        if value <= 0:
            raise TelemetrySchemaError("latency buckets must be finite and positive")
        normalized.append(value)
    if normalized != sorted(set(normalized)):
        raise TelemetrySchemaError("latency buckets must be strictly increasing")
    return tuple(normalized)


def _coerce_counter_name(value: object) -> CounterName:
    if type(value) is CounterName:
        return value
    if type(value) is str:
        try:
            return CounterName(value)
        except ValueError:
            pass
    raise TelemetrySchemaError("telemetry counter is not approved")


def _coerce_positive_int(
    value: object,
    field_name: str,
    *,
    maximum: int = MAX_COUNTER_VALUE,
) -> int:
    if type(value) is not int or not (0 < value <= maximum):
        raise TelemetrySchemaError(f"{field_name} must be a positive integer")
    return value


def _coerce_non_negative_int(
    value: object,
    field_name: str,
    *,
    maximum: int = MAX_COUNTER_VALUE,
) -> int:
    if type(value) is not int or not (0 <= value <= maximum):
        raise TelemetrySchemaError(f"{field_name} must be a non-negative integer")
    return value


def _coerce_duration(value: object, field_name: str) -> float:
    return _coerce_bounded_float(
        value,
        field_name,
        maximum=MAX_LATENCY_SECONDS,
    )


def _coerce_duration_ms(value: object, field_name: str) -> float:
    milliseconds = _coerce_bounded_float(
        value,
        field_name,
        maximum=MAX_LATENCY_SECONDS * 1000.0,
    )
    return milliseconds / 1000.0


def _coerce_bounded_float(
    value: object,
    field_name: str,
    *,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TelemetrySchemaError(f"{field_name} must be a finite number")
    try:
        observed = float(value)
    except Exception:  # noqa: BLE001 - numeric protocols are caller-controlled.
        raise TelemetrySchemaError(f"{field_name} must be a finite number") from None
    if not math.isfinite(observed) or not (0 <= observed <= maximum):
        raise TelemetrySchemaError(
            f"{field_name} must be finite, non-negative, and bounded"
        )
    return observed


def _normalize_dimension(name: str, value: object) -> str:
    allowed = _DIMENSION_VALUES[name]
    if isinstance(value, Enum):
        try:
            value = value.value
        except Exception:  # noqa: BLE001 - enums may be caller-controlled.
            return OTHER_DIMENSION_VALUE
    if type(value) is not str or len(value) > 64:
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
    values = _snapshot_mapping(
        dimensions,
        field_name="telemetry dimensions",
        max_items=len(_DIMENSION_NAMES),
    )
    if set(values) - _DIMENSION_NAMES:
        raise UnapprovedTelemetryKeyError(
            "telemetry dimensions contain unapproved fields"
        )
    return tuple(
        sorted(
            (
                name,
                _normalize_dimension(name, value),
            )
            for name, value in values.items()
        )
    )


def _validate_dimension_tuple(
    dimensions: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if type(dimensions) is not tuple or len(dimensions) > len(_DIMENSION_NAMES):
        raise TelemetrySchemaError("sample dimensions are invalid")
    normalized: list[tuple[str, str]] = []
    for item in dimensions:
        if type(item) is not tuple or len(item) != 2:
            raise TelemetrySchemaError("sample dimensions are invalid")
        name, value = item
        if (
            type(name) is not str
            or name not in _DIMENSION_NAMES
            or type(value) is not str
            or len(value) > 64
            or value not in _DIMENSION_VALUES[name]
        ):
            raise TelemetrySchemaError("sample dimensions are invalid")
        normalized.append((name, value))
    result = tuple(normalized)
    if result != tuple(sorted(set(result))):
        raise TelemetrySchemaError("sample dimensions are not canonical")
    return result


def _validate_bucket_tuple(
    buckets: tuple[tuple[str, int], ...],
    *,
    count: int,
) -> tuple[tuple[str, int], ...]:
    if type(buckets) is not tuple or not (1 < len(buckets) <= MAX_LATENCY_BUCKETS + 1):
        raise TelemetrySchemaError("latency sample buckets are invalid")

    normalized: list[tuple[str, int]] = []
    boundaries: list[float] = []
    previous_count = -1
    for index, item in enumerate(buckets):
        if type(item) is not tuple or len(item) != 2:
            raise TelemetrySchemaError("latency sample buckets are invalid")
        boundary, bucket_count = item
        if type(boundary) is not str or len(boundary) > 32:
            raise TelemetrySchemaError("latency sample buckets are invalid")
        safe_count = _coerce_non_negative_int(
            bucket_count,
            "latency bucket count",
            maximum=count,
        )
        if safe_count < previous_count:
            raise TelemetrySchemaError("latency sample buckets are inconsistent")
        previous_count = safe_count

        if index == len(buckets) - 1:
            if boundary != "+Inf" or safe_count != count:
                raise TelemetrySchemaError("latency sample buckets are inconsistent")
        else:
            try:
                numeric_boundary = float(boundary)
            except (TypeError, ValueError):
                raise TelemetrySchemaError(
                    "latency sample buckets are invalid"
                ) from None
            if not math.isfinite(numeric_boundary) or numeric_boundary <= 0:
                raise TelemetrySchemaError("latency sample buckets are invalid")
            boundaries.append(numeric_boundary)
        normalized.append((boundary, safe_count))

    if boundaries != sorted(set(boundaries)):
        raise TelemetrySchemaError("latency sample buckets are not canonical")
    return tuple(normalized)


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
        dimensions = _snapshot_mapping(
            raw_dimensions,
            field_name="telemetry dimensions",
            max_items=len(_DIMENSION_NAMES),
        )
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

    if "exception" in event and "exception_category" in event:
        raise TelemetrySchemaError("telemetry event contains duplicate fields")
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
    "MAX_AGGREGATE_LATENCY_SECONDS",
    "MAX_COUNTER_VALUE",
    "MAX_ENTITY_COUNT",
    "MAX_LATENCY_BUCKETS",
    "MAX_LATENCY_SECONDS",
    "MAX_RESULT_STAGE_DURATIONS",
    "MAX_SNAPSHOT_SAMPLES",
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
