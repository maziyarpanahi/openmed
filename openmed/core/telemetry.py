"""Optional, no-PHI observability for the core de-identification pipeline.

This module adds developer opt-in tracing and lightweight metrics for the
staged privacy pipeline (:mod:`openmed.core.pipeline`). It is designed to
preserve OpenMed's local-first, no-telemetry-by-default guarantees:

- **Off by default.** Nothing is recorded unless the caller explicitly opts in,
  either by constructing :class:`PipelineTelemetry` with ``enabled=True`` or by
  setting the ``OPENMED_TELEMETRY_ENABLED`` environment variable to a truthy
  value.
- **OpenTelemetry is optional.** The OTel packages are imported lazily behind a
  guard. When they are not installed, a no-op backend is used so importing and
  using this module never raises and core stays dependency-free.
- **Local-first.** No network exporter is ever configured here. Callers wire
  their own span processors/exporters onto their OTel ``TracerProvider`` if and
  when they want to ship spans somewhere.
- **No raw PHI.** Only aggregate, non-reversible signals are ever emitted:
  stage names, span/entity counts, character offsets and lengths, label *sets*,
  hashes, and durations. Raw text and detected identifiers are never recorded.
  Attribute keys are constrained to an explicit allowlist and values are scrubbed
  by :func:`safe_stage_attributes`.

Example:
    >>> from openmed.core.telemetry import PipelineTelemetry
    >>> telemetry = PipelineTelemetry(enabled=False)  # no-op backend
    >>> with telemetry.stage_span(3, "deterministic_detectors") as stage:
    ...     stage.set_span_count(2)
    >>> telemetry.enabled
    False
"""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional, Sequence

try:  # pragma: no cover - exercised when the optional otel extra is absent.
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import SpanKind as _OTelSpanKind
    from opentelemetry.trace import Status as _OTelStatus
    from opentelemetry.trace import StatusCode as _OTelStatusCode
except ImportError as exc:  # pragma: no cover - depends on install extras.
    _otel_trace = None  # type: ignore[assignment]
    _OTelSpanKind = None  # type: ignore[assignment]
    _OTelStatus = None  # type: ignore[assignment]
    _OTelStatusCode = None  # type: ignore[assignment]
    _OTEL_IMPORT_ERROR: Optional[ImportError] = exc
else:
    _OTEL_IMPORT_ERROR = None

TELEMETRY_ENABLED_ENV_VAR = "OPENMED_TELEMETRY_ENABLED"
TRACER_NAME = "openmed.pipeline"
SPAN_NAME_PREFIX = "openmed.pipeline"

_ENABLED_VALUES = frozenset({"1", "true", "yes", "on", "enabled"})
_DISABLED_VALUES = frozenset({"0", "false", "no", "off", "disabled"})
_SAFE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.:/ -]{1,128}$")

# Only these no-PHI attribute keys may ever reach a span. Anything else — and in
# particular any key carrying raw surface text — is dropped by
# ``safe_stage_attributes``. This mirrors the allowlist used by the REST
# service tracer (``openmed.service.tracing``) so the two stay consistent.
_ALLOWED_ATTRIBUTE_KEYS = frozenset(
    {
        "openmed.pipeline.doc_id_hash",
        "openmed.pipeline.language",
        "openmed.pipeline.script",
        "openmed.pipeline.stage_count",
        "openmed.stage",
        "openmed.stage.index",
        "openmed.stage.span_count",
        "openmed.stage.span_delta",
        "openmed.stage.entity_count",
        "openmed.stage.labels",
        "openmed.stage.input_length",
        "openmed.stage.redacted_length",
        "openmed.stage.duration_ms",
        "openmed.error.type",
    }
)


def parse_telemetry_enabled(raw_value: Optional[str]) -> bool:
    """Parse the telemetry opt-in flag.

    Args:
        raw_value: Raw environment-variable value (or ``None`` when unset).

    Returns:
        ``True`` only when the value is an explicit truthy token. Absent,
        empty, or falsy values keep telemetry disabled — the safe default.

    Raises:
        ValueError: If the value is a non-empty, unrecognized token.
    """
    if raw_value is None:
        return False
    normalized = raw_value.strip().lower()
    if not normalized:
        return False
    if normalized in _ENABLED_VALUES:
        return True
    if normalized in _DISABLED_VALUES:
        return False
    raise ValueError(
        f"{TELEMETRY_ENABLED_ENV_VAR} must be a boolean value like 'true' or 'false'"
    )


def telemetry_enabled_from_env() -> bool:
    """Return whether telemetry is opted in via the environment (default off)."""
    return parse_telemetry_enabled(os.getenv(TELEMETRY_ENABLED_ENV_VAR))


def otel_available() -> bool:
    """Return whether the optional OpenTelemetry API is importable."""
    return _OTEL_IMPORT_ERROR is None


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return value[:256]
    return None


def _safe_label(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if _SAFE_LABEL_PATTERN.fullmatch(normalized):
        return normalized
    return None


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    return [value]


def safe_stage_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Return only approved, no-PHI scalar or scalar-list attributes.

    Keys outside :data:`_ALLOWED_ATTRIBUTE_KEYS` are dropped, so no raw surface
    text can ever reach a span even if a caller passes it in by mistake. Label
    lists are additionally constrained to a conservative character allowlist.

    Args:
        attributes: Candidate attribute mapping.

    Returns:
        A new dict containing only safe, allowlisted attributes.
    """
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        if key not in _ALLOWED_ATTRIBUTE_KEYS or value is None:
            continue
        if key == "openmed.stage.labels":
            labels = sorted(
                {
                    label
                    for raw in _as_sequence(value)
                    if (label := _safe_label(raw)) is not None
                }
            )
            if labels:
                safe[key] = tuple(labels)
            continue
        normalized = _safe_scalar(value)
        if normalized is not None:
            safe[key] = normalized
    return safe


@dataclass
class StageMetrics:
    """No-PHI metrics collected for one pipeline stage.

    Only aggregate signals are stored; no raw text is ever kept here.
    """

    index: int
    name: str
    span_count: int = 0
    entity_count: int = 0
    labels: tuple[str, ...] = ()
    input_length: Optional[int] = None
    redacted_length: Optional[int] = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly, no-PHI summary of this stage."""
        payload: dict[str, Any] = {
            "index": self.index,
            "name": self.name,
            "span_count": self.span_count,
            "entity_count": self.entity_count,
            "labels": list(self.labels),
            "duration_ms": round(self.duration_ms, 3),
        }
        if self.input_length is not None:
            payload["input_length"] = self.input_length
        if self.redacted_length is not None:
            payload["redacted_length"] = self.redacted_length
        return payload


class StageRecorder:
    """Collect no-PHI signals for one stage and mirror them onto a span.

    Instances are yielded by :meth:`PipelineTelemetry.stage_span`. Every setter
    routes values through :func:`safe_stage_attributes` before touching the
    active span, so it is impossible to attach a raw-text attribute here.
    """

    def __init__(
        self,
        metrics: StageMetrics,
        span: Any,
        *,
        span_count_baseline: int = 0,
    ) -> None:
        self._metrics = metrics
        self._span = span
        self._baseline = span_count_baseline

    @property
    def metrics(self) -> StageMetrics:
        """Return the underlying :class:`StageMetrics` accumulator."""
        return self._metrics

    def _set_attributes(self, attributes: Mapping[str, Any]) -> None:
        if self._span is None:
            return
        safe = safe_stage_attributes(attributes)
        setter = getattr(self._span, "set_attributes", None)
        if safe and callable(setter):
            setter(safe)

    def set_span_count(self, count: int) -> None:
        """Record the number of spans this stage produced."""
        value = int(count)
        self._metrics.span_count = value
        self._set_attributes(
            {
                "openmed.stage.span_count": value,
                "openmed.stage.span_delta": value - self._baseline,
            }
        )

    def set_entity_count(self, count: int) -> None:
        """Record the number of entities associated with this stage."""
        value = int(count)
        self._metrics.entity_count = value
        self._set_attributes({"openmed.stage.entity_count": value})

    def set_labels(self, labels: Sequence[str]) -> None:
        """Record the *set* of canonical labels seen in this stage (no text)."""
        safe_labels = tuple(
            sorted({label for raw in labels if (label := _safe_label(raw)) is not None})
        )
        self._metrics.labels = safe_labels
        if safe_labels:
            self._set_attributes({"openmed.stage.labels": safe_labels})

    def set_input_length(self, length: int) -> None:
        """Record the input character length processed by this stage."""
        value = int(length)
        self._metrics.input_length = value
        self._set_attributes({"openmed.stage.input_length": value})

    def set_redacted_length(self, length: int) -> None:
        """Record the redacted-output character length for this stage."""
        value = int(length)
        self._metrics.redacted_length = value
        self._set_attributes({"openmed.stage.redacted_length": value})


class _NullStageRecorder(StageRecorder):
    """Recorder used when telemetry is disabled: accumulates nothing extra."""

    def __init__(self, metrics: StageMetrics) -> None:
        super().__init__(metrics, span=None)


@dataclass
class PipelineTelemetry:
    """Opt-in, no-PHI telemetry for the staged de-identification pipeline.

    When ``enabled`` is ``False`` (the default) every method is a cheap no-op and
    no OpenTelemetry objects are touched. When ``enabled`` is ``True`` but the
    OpenTelemetry API is not installed, a no-op backend is used instead of
    raising, so callers can safely request telemetry without hard-depending on
    the optional extra.

    Attributes:
        enabled: Whether span/metric recording is active.
        tracer: Optional OpenTelemetry tracer. When ``None`` and ``enabled`` is
            ``True``, the module-global tracer is used if OTel is available.
        service_name: Logical component name; kept for parity with the REST
            tracer.
    """

    enabled: bool = False
    tracer: Any = None
    service_name: str = "openmed-pipeline"
    stage_metrics: list[StageMetrics] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.enabled and self.tracer is None and otel_available():
            self.tracer = _otel_trace.get_tracer(TRACER_NAME)
        # Keep ``enabled`` an honest predicate: it is only ``True`` when a span
        # will actually be created. If the caller opted in but no backend is
        # usable, degrade to the no-op path instead of raising.
        if self.enabled and self.tracer is None:
            self.enabled = False
        self._run_context: dict[str, Any] = {}

    @classmethod
    def from_env(cls, *, tracer: Any = None) -> "PipelineTelemetry":
        """Build telemetry honoring the ``OPENMED_TELEMETRY_ENABLED`` flag."""
        return cls(enabled=telemetry_enabled_from_env(), tracer=tracer)

    @classmethod
    def disabled(cls) -> "PipelineTelemetry":
        """Return an explicitly disabled, no-op telemetry instance."""
        return cls(enabled=False)

    def start_pipeline(self) -> None:
        """Reset per-run accumulators for a new pipeline invocation."""
        self.stage_metrics = []
        self._run_context = {}

    def set_run_context(
        self,
        *,
        language: Optional[str] = None,
        script: Optional[str] = None,
        doc_id_hash: Optional[str] = None,
    ) -> None:
        """Attach no-PHI run context applied to subsequent stage spans.

        Only non-reversible signals are retained: the language, script, and an
        already-hashed document id. Never a raw document id or text.
        """
        if language is not None:
            self._run_context["openmed.pipeline.language"] = _safe_label(language)
        if script is not None:
            self._run_context["openmed.pipeline.script"] = _safe_label(script)
        if doc_id_hash is not None and isinstance(doc_id_hash, str):
            self._run_context["openmed.pipeline.doc_id_hash"] = doc_id_hash

    @contextmanager
    def stage_span(
        self,
        index: int,
        name: str,
        *,
        span_count_baseline: int = 0,
    ) -> Iterator[StageRecorder]:
        """Trace one pipeline stage, timing it and recording no-PHI signals.

        Args:
            index: 1-based stage index.
            name: Stage name (one of :data:`openmed.core.pipeline.STAGE_NAMES`).
            span_count_baseline: Prior cumulative span count, so the stage can
                report its net contribution as ``openmed.stage.span_delta``.

        Yields:
            A :class:`StageRecorder` for attaching aggregate stage metrics.
        """
        metrics = StageMetrics(index=index, name=name)
        self.stage_metrics.append(metrics)

        if not self.enabled or self.tracer is None:
            recorder = _NullStageRecorder(metrics)
            start = time.perf_counter()
            try:
                yield recorder
            finally:
                metrics.duration_ms = (time.perf_counter() - start) * 1000
            return

        base_attributes = safe_stage_attributes(
            {
                "openmed.stage": name,
                "openmed.stage.index": index,
                **{
                    key: value
                    for key, value in self._run_context.items()
                    if value is not None
                },
            }
        )
        start = time.perf_counter()
        with self.tracer.start_as_current_span(
            f"{SPAN_NAME_PREFIX}.{name}",
            kind=_OTelSpanKind.INTERNAL,
            attributes=base_attributes,
        ) as span:
            recorder = StageRecorder(
                metrics,
                span,
                span_count_baseline=span_count_baseline,
            )
            try:
                yield recorder
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(_OTelStatus(_OTelStatusCode.ERROR))
                span.set_attributes(
                    safe_stage_attributes({"openmed.error.type": type(exc).__name__})
                )
                raise
            finally:
                metrics.duration_ms = (time.perf_counter() - start) * 1000
                span.set_attributes(
                    safe_stage_attributes(
                        {"openmed.stage.duration_ms": round(metrics.duration_ms, 3)}
                    )
                )

    def summary(self) -> dict[str, Any]:
        """Return an aggregate, no-PHI summary of the most recent run."""
        return {
            "enabled": self.enabled,
            "stage_count": len(self.stage_metrics),
            "total_duration_ms": round(
                sum(metric.duration_ms for metric in self.stage_metrics), 3
            ),
            "stages": [metric.to_dict() for metric in self.stage_metrics],
        }


__all__ = [
    "PipelineTelemetry",
    "StageMetrics",
    "StageRecorder",
    "TELEMETRY_ENABLED_ENV_VAR",
    "TRACER_NAME",
    "otel_available",
    "parse_telemetry_enabled",
    "safe_stage_attributes",
    "telemetry_enabled_from_env",
]
