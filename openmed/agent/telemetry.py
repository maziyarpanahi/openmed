"""Opt-in, exporter-free OpenTelemetry bridge for value-free agent events.

Only the existing event attribute contract and closed action phases are accepted.
Identifiers are omitted; optionally the opaque run ID becomes a domain-separated
digest. This module never configures an SDK provider or a network destination.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager, nullcontext
from importlib import import_module
from importlib.util import find_spec
from time import perf_counter
from typing import Any, Callable, Iterator, Literal

from .action_phases import ActionPhase, ActionPhaseError
from .event_attributes import (
    ALLOWED_ATTRIBUTES,
    AttributeKind,
    EventAttributeError,
    EventAttributes,
)

INSTRUMENTATION_NAME = "openmed.agent"
EVENT_COUNTER_NAME = "openmed.agent.events"
DURATION_METRIC_NAME = "openmed.agent.event.duration"

_VALUE_KINDS = frozenset(
    {
        AttributeKind.EXECUTION_STAGE,
        AttributeKind.OUTCOME_CLASS,
        AttributeKind.OUTCOME_REASON,
        AttributeKind.DIGEST,
        AttributeKind.COUNT,
        AttributeKind.DURATION_MS,
        AttributeKind.FLAG,
    }
)
_METRIC_KINDS = frozenset(
    {
        AttributeKind.EXECUTION_STAGE,
        AttributeKind.OUTCOME_CLASS,
        AttributeKind.OUTCOME_REASON,
        AttributeKind.FLAG,
    }
)


def _phase(value: ActionPhase | str) -> ActionPhase:
    if isinstance(value, ActionPhase):
        return value
    if type(value) is str:
        try:
            return ActionPhase(value)
        except ValueError:
            pass
    raise ActionPhaseError("unknown_phase", "phase")


def _attributes(value: EventAttributes) -> EventAttributes:
    if type(value) is not EventAttributes:
        raise EventAttributeError("invalid_attribute_type")
    # Revalidate at the sink boundary, even for a tampered frozen instance.
    return EventAttributes(values=value.values, schema_version=value.schema_version)


class AgentTelemetry:
    """Opt-in spans, event counters, and latency for existing agent contracts.

    Args:
        enabled: Explicit boolean opt-in, defaulting to false. There is no
            implicit environment opt-in inherited from pipeline telemetry.
        run_id_mode: ``omit`` (default) or ``hash``. Hash mode emits a
            domain-separated SHA-256 digest under ``openmed.agent.run_id``;
            all other identifier kinds are always omitted.
        tracer: Optional caller-owned tracer. Otherwise use the global API
            tracer after opt-in, if OpenTelemetry is available.
        meter: Optional caller-owned meter. Otherwise use the global API meter
            after opt-in, if OpenTelemetry is available.
        clock: Monotonic seconds source used when no event duration is supplied.

    OpenMed creates neither providers nor exporters. Caller-owned providers
    control sampling, export, and ambient parent context. Metric dimensions
    exclude identifiers, digests, sequence numbers, counts, and durations.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        run_id_mode: Literal["omit", "hash"] = "omit",
        tracer: Any = None,
        meter: Any = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if type(run_id_mode) is not str or run_id_mode not in {"omit", "hash"}:
            raise ValueError("run_id_mode must be omit or hash")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.enabled = enabled
        self._run_id_mode = run_id_mode
        self._tracer = tracer if enabled else None
        self._clock = clock
        self._counter = None
        self._duration = None
        if not enabled:
            return

        if (tracer is None or meter is None) and find_spec("opentelemetry") is not None:
            try:
                trace_api = import_module("opentelemetry.trace")
                metrics_api = import_module("opentelemetry.metrics")
            except ImportError:
                pass
            else:
                if self._tracer is None:
                    self._tracer = trace_api.get_tracer(INSTRUMENTATION_NAME)
                if meter is None:
                    meter = metrics_api.get_meter(INSTRUMENTATION_NAME)
        if self._tracer is None and meter is None:
            self.enabled = False
            return
        if meter is not None:
            self._counter = meter.create_counter(
                EVENT_COUNTER_NAME,
                unit="1",
                description="Value-free agent events by phase and outcome.",
            )
            self._duration = meter.create_histogram(
                DURATION_METRIC_NAME,
                unit="ms",
                description="Value-free agent event latency.",
            )

    @contextmanager
    def event_span(
        self, phase: ActionPhase | str, attributes: EventAttributes
    ) -> Iterator[None]:
        """Trace one event or operation using only validated metadata.

        Args:
            phase: Existing action phase enum or exact canonical string.
            attributes: Existing immutable event attribute contract. Supplied
                ``duration_ms`` wins; otherwise measure the context duration.

        Yields:
            None. The raw span is deliberately not exposed for arbitrary writes.

        Raises:
            ActionPhaseError: If the phase is unknown.
            EventAttributeError: If attributes or duration fail validation.

        One counter increment is emitted per context entry, including failed
        operations. Exceptions propagate without messages, stack traces, events,
        or status descriptions being copied to telemetry. This bridge observes
        phases; it does not enforce transitions or make approval decisions.
        """
        canonical_phase = _phase(phase)
        validated = _attributes(attributes)
        if not self.enabled:
            yield None
            return

        safe = {
            f"openmed.agent.{key}": value
            for key, value in sorted(validated.values.items())
            if ALLOWED_ATTRIBUTES[key] in _VALUE_KINDS
        }
        run_id = validated.get("run_id")
        if self._run_id_mode == "hash" and run_id is not None:
            digest = hashlib.sha256(
                b"openmed.agent.telemetry.run_id.v1\0" + run_id.encode("ascii")
            ).hexdigest()
            safe["openmed.agent.run_id"] = f"sha256:{digest}"
        metric_attributes = {
            key: value
            for key, value in safe.items()
            if ALLOWED_ATTRIBUTES[key.removeprefix("openmed.agent.")] in _METRIC_KINDS
        }
        # Phase is encoded in a fixed span name and metric dimension, never
        # accepted as a free-form event attribute.
        metric_attributes["openmed.agent.phase"] = canonical_phase.value
        manager = nullcontext(None)
        if self._tracer is not None:
            manager = self._tracer.start_as_current_span(
                f"openmed.agent.{canonical_phase.value}",
                attributes=safe,
                record_exception=False,
                set_status_on_exception=False,
            )
        duration = validated.get("duration_ms")
        start = self._clock() if duration is None else None
        with manager as span:
            if self._counter is not None:
                self._counter.add(1, attributes=metric_attributes)
            try:
                yield None
            finally:
                if duration is None:
                    duration = (self._clock() - start) * 1000.0
                # Share the bounded duration contract rather than inventing one.
                duration = EventAttributes.from_mapping({"duration_ms": duration}).get(
                    "duration_ms"
                )
                if span is not None:
                    span.set_attribute("openmed.agent.duration_ms", duration)
                if self._duration is not None:
                    self._duration.record(duration, attributes=metric_attributes)


__all__ = [
    "AgentTelemetry",
    "DURATION_METRIC_NAME",
    "EVENT_COUNTER_NAME",
    "INSTRUMENTATION_NAME",
]
