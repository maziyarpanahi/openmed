"""Optional no-PHI observability for the core privacy pipeline.

The OpenTelemetry API is imported only after an explicit opt-in. This module
never creates an SDK provider, reader, processor, or exporter, so enabling it
does not create a network path. When OpenTelemetry is unavailable, the public
runtime degrades to a no-op and the core package remains dependency-free.

Only fixed pipeline stage names, numeric aggregates, canonical OpenMed labels,
and durations can reach spans or metric attributes. Raw document text,
detected surfaces, document identifiers, replacements, exception messages, and
stack traces are deliberately outside the attribute contract.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Any, Iterator, Mapping, Sequence

from .labels import CANONICAL_LABELS

TELEMETRY_ENABLED_ENV_VAR = "OPENMED_TELEMETRY_ENABLED"
TRACER_NAME = "openmed.pipeline"
METER_NAME = "openmed.pipeline"
SPAN_NAME_PREFIX = "openmed.pipeline"

DURATION_METRIC_NAME = "openmed.pipeline.stage.duration"
SPAN_COUNT_METRIC_NAME = "openmed.pipeline.stage.span_count"
ENTITY_COUNT_METRIC_NAME = "openmed.pipeline.stage.entity_count"

PIPELINE_STAGE_NAMES: tuple[str, ...] = (
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
)

_ENABLED_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})
_DISABLED_VALUES = frozenset({"0", "false", "no", "off", "disabled"})
_INTEGER_ATTRIBUTE_KEYS = frozenset(
    {
        "openmed.stage.index",
        "openmed.stage.span_count",
        "openmed.stage.entity_count",
        "openmed.stage.input_length",
        "openmed.stage.redacted_length",
        "openmed.stage.offset.start_min",
        "openmed.stage.offset.end_max",
    }
)
_ALLOWED_ATTRIBUTE_KEYS = frozenset(
    {
        "openmed.stage",
        "openmed.stage.duration_ms",
        "openmed.stage.failed",
        "openmed.stage.labels",
        *_INTEGER_ATTRIBUTE_KEYS,
    }
)


@dataclass(frozen=True)
class _OpenTelemetryModules:
    trace: Any = None
    metrics: Any = None


@lru_cache(maxsize=1)
def _load_otel() -> _OpenTelemetryModules:
    """Lazily import the optional OpenTelemetry API modules."""
    try:
        trace = import_module("opentelemetry.trace")
        metrics = import_module("opentelemetry.metrics")
    except ImportError:
        return _OpenTelemetryModules()
    return _OpenTelemetryModules(trace=trace, metrics=metrics)


def parse_telemetry_enabled(raw_value: str | None) -> bool:
    """Parse an explicit telemetry opt-in value.

    Args:
        raw_value: Environment or configuration value. ``None``, an empty
            string, and explicit false tokens keep telemetry disabled.

    Returns:
        Whether telemetry was explicitly enabled.

    Raises:
        ValueError: If a non-empty value is not a recognized boolean token.
    """
    if raw_value is None:
        return False
    normalized = raw_value.strip().lower()
    if not normalized or normalized in _DISABLED_VALUES:
        return False
    if normalized in _ENABLED_VALUES:
        return True
    raise ValueError(
        f"{TELEMETRY_ENABLED_ENV_VAR} must be a boolean value like 'true' or 'false'"
    )


def telemetry_enabled_from_env() -> bool:
    """Return whether core pipeline telemetry is explicitly enabled."""
    return parse_telemetry_enabled(os.getenv(TELEMETRY_ENABLED_ENV_VAR))


def otel_available() -> bool:
    """Return whether the optional OpenTelemetry trace and metrics APIs load."""
    modules = _load_otel()
    return modules.trace is not None and modules.metrics is not None


def safe_stage_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Return the no-PHI subset of candidate stage attributes.

    String values are never accepted generically. Stage names must be one of
    the ten fixed pipeline stages, while label values must be members of
    OpenMed's canonical taxonomy. All remaining values are bounded numeric or
    boolean aggregates.

    Args:
        attributes: Candidate attributes to validate.

    Returns:
        A new mapping containing only allowlisted, validated values.
    """
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        if key not in _ALLOWED_ATTRIBUTE_KEYS or value is None:
            continue
        if key == "openmed.stage":
            if value in PIPELINE_STAGE_NAMES:
                safe[key] = value
            continue
        if key == "openmed.stage.labels":
            labels = _safe_labels(value)
            if labels:
                safe[key] = labels
            continue
        if key == "openmed.stage.failed":
            if isinstance(value, bool):
                safe[key] = value
            continue
        if key in _INTEGER_ATTRIBUTE_KEYS:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                safe[key] = value
            continue
        if key == "openmed.stage.duration_ms":
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0.0
            ):
                safe[key] = float(value)
    return safe


def _safe_labels(value: Any) -> tuple[str, ...]:
    values: Sequence[Any]
    if isinstance(value, (list, tuple, set, frozenset)):
        values = tuple(value)
    else:
        values = (value,)
    return tuple(
        sorted(
            {
                item
                for item in values
                if isinstance(item, str) and item in CANONICAL_LABELS
            }
        )
    )


def _nonnegative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


class StageTelemetry:
    """No-PHI recorder for one pipeline stage.

    Instances are created by :meth:`PipelineTelemetry.stage_span`. Setters
    accept only aggregate values and route every span attribute through
    :func:`safe_stage_attributes`.
    """

    def __init__(
        self,
        *,
        index: int,
        name: str,
        span: Any = None,
        duration_histogram: Any = None,
        span_count_histogram: Any = None,
        entity_count_histogram: Any = None,
    ) -> None:
        self.index = index
        self.name = name
        self._span = span
        self._duration_histogram = duration_histogram
        self._span_count_histogram = span_count_histogram
        self._entity_count_histogram = entity_count_histogram
        self._span_count = 0
        self._entity_count = 0
        self._finished = False

    @property
    def active(self) -> bool:
        """Return whether this recorder has a trace or metric sink."""
        return bool(
            self._span is not None
            or self._duration_histogram is not None
            or self._span_count_histogram is not None
            or self._entity_count_histogram is not None
        )

    def _set_attributes(self, attributes: Mapping[str, Any]) -> None:
        setter = getattr(self._span, "set_attributes", None)
        safe = safe_stage_attributes(attributes)
        if safe and callable(setter):
            setter(safe)

    def set_span_count(self, count: int) -> None:
        """Record how many canonical spans this stage produced."""
        self._span_count = _nonnegative_int(count, name="span count")
        self._set_attributes({"openmed.stage.span_count": self._span_count})

    def set_entity_count(self, count: int) -> None:
        """Record how many entities this stage produced."""
        self._entity_count = _nonnegative_int(count, name="entity count")
        self._set_attributes({"openmed.stage.entity_count": self._entity_count})

    def set_labels(self, labels: Sequence[str]) -> None:
        """Record a set of canonical category labels, never entity text."""
        self._set_attributes({"openmed.stage.labels": labels})

    def set_input_length(self, length: int) -> None:
        """Record a stage input character count."""
        self._set_attributes(
            {
                "openmed.stage.input_length": _nonnegative_int(
                    length, name="input length"
                )
            }
        )

    def set_redacted_length(self, length: int) -> None:
        """Record the emitted redacted character count."""
        self._set_attributes(
            {
                "openmed.stage.redacted_length": _nonnegative_int(
                    length, name="redacted length"
                )
            }
        )

    def set_offset_range(self, start: int, end: int) -> None:
        """Record aggregate output bounds without storing a detected surface."""
        start = _nonnegative_int(start, name="start offset")
        end = _nonnegative_int(end, name="end offset")
        if end < start:
            raise ValueError("end offset must be greater than or equal to start")
        self._set_attributes(
            {
                "openmed.stage.offset.start_min": start,
                "openmed.stage.offset.end_max": end,
            }
        )

    def mark_failed(self) -> None:
        """Mark a failed stage without recording an exception or message."""
        self._set_attributes({"openmed.stage.failed": True})

    def finish(self, duration_ms: float) -> None:
        """Finish the stage and record its duration and aggregate metrics."""
        if self._finished:
            return
        self._finished = True
        if not self.active:
            return
        duration = float(duration_ms)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("duration_ms must be a finite non-negative number")
        self._set_attributes({"openmed.stage.duration_ms": duration})

        metric_attributes = safe_stage_attributes(
            {
                "openmed.stage": self.name,
                "openmed.stage.index": self.index,
            }
        )
        if self._duration_histogram is not None:
            self._duration_histogram.record(duration, attributes=metric_attributes)
        if self._span_count_histogram is not None:
            self._span_count_histogram.record(
                self._span_count,
                attributes=metric_attributes,
            )
        if self._entity_count_histogram is not None:
            self._entity_count_histogram.record(
                self._entity_count,
                attributes=metric_attributes,
            )


class PipelineTelemetry:
    """Opt-in OpenTelemetry spans and metrics for core pipeline stages.

    Args:
        enabled: Explicit opt-in. The default is ``False``.
        tracer: Optional caller-owned OpenTelemetry tracer. When omitted after
            opt-in, the global API tracer is used if OpenTelemetry is installed.
        meter: Optional caller-owned OpenTelemetry meter. When omitted after
            opt-in, the global API meter is used if OpenTelemetry is installed.

    OpenMed never configures the providers behind these objects and therefore
    never creates an exporter or network destination.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        tracer: Any = None,
        meter: Any = None,
    ) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        self.enabled = enabled
        self.tracer = tracer
        self.meter = meter
        self._duration_histogram = None
        self._span_count_histogram = None
        self._entity_count_histogram = None

        if not self.enabled:
            return

        if self.tracer is None or self.meter is None:
            modules = _load_otel()
            if self.tracer is None and modules.trace is not None:
                self.tracer = modules.trace.get_tracer(TRACER_NAME)
            if self.meter is None and modules.metrics is not None:
                self.meter = modules.metrics.get_meter(METER_NAME)

        if self.tracer is None and self.meter is None:
            self.enabled = False
            return

        if self.meter is not None:
            self._duration_histogram = self.meter.create_histogram(
                DURATION_METRIC_NAME,
                unit="ms",
                description="No-PHI wall-clock duration of a core pipeline stage.",
            )
            self._span_count_histogram = self.meter.create_histogram(
                SPAN_COUNT_METRIC_NAME,
                unit="1",
                description="Canonical spans produced by a core pipeline stage.",
            )
            self._entity_count_histogram = self.meter.create_histogram(
                ENTITY_COUNT_METRIC_NAME,
                unit="1",
                description="Entities produced by a core pipeline stage.",
            )

    @classmethod
    def from_env(
        cls,
        *,
        tracer: Any = None,
        meter: Any = None,
    ) -> "PipelineTelemetry":
        """Build telemetry from the explicit environment opt-in."""
        return cls(
            enabled=telemetry_enabled_from_env(),
            tracer=tracer,
            meter=meter,
        )

    @classmethod
    def disabled(cls) -> "PipelineTelemetry":
        """Return an explicitly disabled no-op runtime."""
        return cls(enabled=False)

    @contextmanager
    def stage_span(self, index: int, name: str) -> Iterator[StageTelemetry]:
        """Create one fixed-name, no-PHI span for a pipeline stage."""
        index = _nonnegative_int(index, name="stage index")
        if not 1 <= index <= len(PIPELINE_STAGE_NAMES):
            raise ValueError("stage index must be between 1 and 10")
        expected_name = PIPELINE_STAGE_NAMES[index - 1]
        if name != expected_name:
            raise ValueError(f"stage {index} must use the fixed name {expected_name!r}")

        span_manager: Any = nullcontext(None)
        if self.enabled and self.tracer is not None:
            attributes = safe_stage_attributes(
                {
                    "openmed.stage": name,
                    "openmed.stage.index": index,
                }
            )
            span_manager = self.tracer.start_as_current_span(
                f"{SPAN_NAME_PREFIX}.{name}",
                attributes=attributes,
                record_exception=False,
                set_status_on_exception=False,
            )

        with span_manager as span:
            recorder = StageTelemetry(
                index=index,
                name=name,
                span=span,
                duration_histogram=(self._duration_histogram if self.enabled else None),
                span_count_histogram=(
                    self._span_count_histogram if self.enabled else None
                ),
                entity_count_histogram=(
                    self._entity_count_histogram if self.enabled else None
                ),
            )
            try:
                yield recorder
            except BaseException:
                recorder.mark_failed()
                raise


__all__ = [
    "DURATION_METRIC_NAME",
    "ENTITY_COUNT_METRIC_NAME",
    "METER_NAME",
    "PIPELINE_STAGE_NAMES",
    "PipelineTelemetry",
    "SPAN_COUNT_METRIC_NAME",
    "StageTelemetry",
    "TELEMETRY_ENABLED_ENV_VAR",
    "TRACER_NAME",
    "otel_available",
    "parse_telemetry_enabled",
    "safe_stage_attributes",
    "telemetry_enabled_from_env",
]
