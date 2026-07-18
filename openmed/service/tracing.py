"""OpenTelemetry tracing helpers for the OpenMed REST service."""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

try:  # pragma: no cover - exercised when the optional service extra is absent.
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )
except ImportError as exc:  # pragma: no cover
    trace = None  # type: ignore[assignment]
    OTLPSpanExporter = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    SpanKind = None  # type: ignore[assignment]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]
    TraceContextTextMapPropagator = None  # type: ignore[assignment]
    _OTEL_IMPORT_ERROR: Optional[ImportError] = exc
else:
    _OTEL_IMPORT_ERROR = None

TRACING_ENABLED_ENV_VAR = "OPENMED_SERVICE_TRACING_ENABLED"
OTLP_ENDPOINT_ENV_VAR = "OPENMED_SERVICE_OTLP_ENDPOINT"
OTLP_HEADERS_ENV_VAR = "OPENMED_SERVICE_OTLP_HEADERS"
OTLP_TIMEOUT_ENV_VAR = "OPENMED_SERVICE_OTLP_TIMEOUT_SECONDS"
DEFAULT_OTLP_TIMEOUT_SECONDS = 10.0
TRACER_NAME = "openmed.service"

_ENABLED_VALUES = {"1", "true", "yes", "on", "enabled"}
_DISABLED_VALUES = {"0", "false", "no", "off", "disabled"}
_SAFE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.:/ -]{1,128}$")
_CURRENT_TRACING: ContextVar[Optional["ServiceTracing"]] = ContextVar(
    "openmed_service_tracing",
    default=None,
)

_ALLOWED_ATTRIBUTE_KEYS = frozenset(
    {
        "http.request.method",
        "http.response.status_code",
        "http.route",
        "openmed.batch.size",
        "openmed.batching.enabled",
        "openmed.coalescing.enabled",
        "openmed.endpoint",
        "openmed.entity.count",
        "openmed.entity.labels",
        "openmed.error.type",
        "openmed.input.count",
        "openmed.input.length",
        "openmed.input.total_length",
        "openmed.model_name",
        "openmed.request_id",
        "openmed.service.name",
        "openmed.stage",
        "openmed.stage.duration_ms",
    }
)


@dataclass(frozen=True)
class ServiceTraceConfig:
    """Configuration for optional OpenTelemetry tracing."""

    enabled: bool = False
    service_name: str = "openmed-rest"
    otlp_endpoint: Optional[str] = None
    otlp_headers: Mapping[str, str] = field(default_factory=dict)
    otlp_timeout_seconds: float = DEFAULT_OTLP_TIMEOUT_SECONDS


@dataclass
class ServiceTracing:
    """Runtime OpenTelemetry objects owned by one FastAPI app instance."""

    config: ServiceTraceConfig
    tracer_provider: Any = None
    tracer: Any = None
    propagator: Any = None

    @property
    def enabled(self) -> bool:
        """Return whether request middleware should create spans."""
        return bool(self.config.enabled and self.tracer is not None)

    @classmethod
    def disabled(cls, config: Optional[ServiceTraceConfig] = None) -> "ServiceTracing":
        """Return a disabled tracing runtime."""
        return cls(config=config or ServiceTraceConfig(enabled=False))

    def shutdown(self) -> None:
        """Flush and stop span processors, suppressing shutdown failures."""
        provider = self.tracer_provider
        if provider is None:
            return
        try:
            provider.force_flush()
            provider.shutdown()
        except Exception:
            return


class OpenTelemetryMiddleware:
    """Create one server span per HTTP request and continue traceparent context."""

    def __init__(self, app: ASGIApp, *, tracing: ServiceTracing) -> None:
        self.app = app
        self.tracing = tracing

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.tracing.enabled:
            await self.app(scope, receive, send)
            return

        status_code = 500
        parent_context = self.tracing.propagator.extract(_carrier_from_scope(scope))
        attributes = safe_trace_attributes(
            {
                "http.request.method": str(scope.get("method", "")),
                "http.route": _route_template(scope),
                "openmed.service.name": self.tracing.config.service_name,
                "openmed.request_id": _current_request_id(),
            }
        )

        with self.tracing.tracer.start_as_current_span(
            "openmed.service.request",
            context=parent_context,
            kind=SpanKind.SERVER,
            attributes=attributes,
        ) as span:
            tracing_token = _CURRENT_TRACING.set(self.tracing)

            async def send_with_status(message: Message) -> None:
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = int(message["status"])
                await send(message)

            try:
                await self.app(scope, receive, send_with_status)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR))
                span.set_attributes(
                    safe_trace_attributes({"openmed.error.type": type(exc).__name__})
                )
                raise
            finally:
                span.set_attributes(
                    safe_trace_attributes(
                        {
                            "http.response.status_code": status_code,
                            "http.route": _route_template(scope),
                            "openmed.request_id": _current_request_id(),
                        }
                    )
                )
                if status_code >= 500:
                    span.set_status(Status(StatusCode.ERROR))
                _CURRENT_TRACING.reset(tracing_token)


def service_trace_config_from_env() -> ServiceTraceConfig:
    """Build tracing settings from environment variables."""
    endpoint = _clean_env_value(os.getenv(OTLP_ENDPOINT_ENV_VAR))
    enabled = parse_tracing_enabled(
        os.getenv(TRACING_ENABLED_ENV_VAR),
        otlp_endpoint=endpoint,
    )
    return ServiceTraceConfig(
        enabled=enabled,
        otlp_endpoint=endpoint,
        otlp_headers=parse_otlp_headers(os.getenv(OTLP_HEADERS_ENV_VAR)),
        otlp_timeout_seconds=parse_otlp_timeout(os.getenv(OTLP_TIMEOUT_ENV_VAR)),
    )


def service_tracing_from_env() -> ServiceTracing:
    """Create the app-local tracing runtime from environment variables."""
    config = service_trace_config_from_env()
    if not config.enabled:
        return ServiceTracing.disabled(config)
    if _OTEL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "OpenTelemetry tracing requires the OpenMed service dependencies."
        ) from _OTEL_IMPORT_ERROR

    provider = TracerProvider(
        resource=Resource.create({"service.name": config.service_name})
    )
    if config.otlp_endpoint:
        exporter = OTLPSpanExporter(
            endpoint=config.otlp_endpoint,
            headers=dict(config.otlp_headers),
            timeout=config.otlp_timeout_seconds,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))

    return ServiceTracing(
        config=config,
        tracer_provider=provider,
        tracer=provider.get_tracer(TRACER_NAME),
        propagator=TraceContextTextMapPropagator(),
    )


def parse_tracing_enabled(
    raw_value: Optional[str], *, otlp_endpoint: Any = None
) -> bool:
    """Parse the tracing feature flag, treating an endpoint as explicit opt-in."""
    if raw_value is None:
        return bool(otlp_endpoint)

    normalized = raw_value.strip().lower()
    if not normalized:
        return False
    if normalized in _ENABLED_VALUES:
        return True
    if normalized in _DISABLED_VALUES:
        return False
    raise ValueError(
        f"{TRACING_ENABLED_ENV_VAR} must be a boolean value like 'true' or 'false'"
    )


def parse_otlp_headers(raw_value: Optional[str]) -> dict[str, str]:
    """Parse comma-separated OTLP exporter headers."""
    if raw_value is None or not raw_value.strip():
        return {}

    headers: dict[str, str] = {}
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"{OTLP_HEADERS_ENV_VAR} entries must use key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"{OTLP_HEADERS_ENV_VAR} entries must include a key")
        headers[key] = value
    return headers


def parse_otlp_timeout(raw_value: Optional[str]) -> float:
    """Parse the OTLP exporter timeout in seconds."""
    if raw_value is None or not raw_value.strip():
        return DEFAULT_OTLP_TIMEOUT_SECONDS
    try:
        timeout = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{OTLP_TIMEOUT_ENV_VAR} must be a positive number") from exc
    if timeout <= 0:
        raise ValueError(f"{OTLP_TIMEOUT_ENV_VAR} must be a positive number")
    return timeout


@contextmanager
def trace_service_stage(
    stage: str,
    attributes: Optional[Mapping[str, Any]] = None,
) -> Iterator[Any]:
    """Create a child span for one no-PHI service stage when tracing is active."""
    tracing = _CURRENT_TRACING.get()
    if tracing is None or not tracing.enabled:
        yield None
        return

    span_attributes = {"openmed.stage": stage}
    if attributes:
        span_attributes.update(attributes)
    start_time = time.perf_counter()
    with tracing.tracer.start_as_current_span(
        f"openmed.service.{stage}",
        kind=SpanKind.INTERNAL,
        attributes=safe_trace_attributes(span_attributes),
    ) as span:
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attributes(
                safe_trace_attributes({"openmed.error.type": type(exc).__name__})
            )
            raise
        finally:
            span.set_attributes(
                safe_trace_attributes(
                    {
                        "openmed.stage.duration_ms": round(
                            (time.perf_counter() - start_time) * 1000,
                            3,
                        )
                    }
                )
            )


def set_current_span_attributes(attributes: Mapping[str, Any]) -> None:
    """Attach no-PHI attributes to the active span, if one exists."""
    if trace is None:
        return
    span = trace.get_current_span()
    setter = getattr(span, "set_attributes", None)
    if callable(setter):
        setter(safe_trace_attributes(attributes))


def result_summary_attributes(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return aggregate no-PHI attributes for a service result payload."""
    entities = _extract_entities(payload)
    attributes: dict[str, Any] = {"openmed.entity.count": len(entities)}
    labels = sorted(
        {
            label
            for entity in entities
            if (label := _safe_label(_entity_label(entity))) is not None
        }
    )
    if labels:
        attributes["openmed.entity.labels"] = labels
    return attributes


def safe_trace_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Return only approved scalar or scalar-list trace attributes."""
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        if key not in _ALLOWED_ATTRIBUTE_KEYS or value is None:
            continue
        normalized = _safe_attribute_value(value)
        if normalized is not None:
            safe[key] = normalized
    return safe


def _extract_entities(payload: Mapping[str, Any]) -> list[Any]:
    for key in ("entities", "pii_entities"):
        value = payload.get(key)
        if isinstance(value, list):
            return list(value)
    return []


def _entity_label(entity: Any) -> Optional[str]:
    if not isinstance(entity, Mapping):
        return None
    for key in ("label", "entity_type", "type"):
        value = entity.get(key)
        if value is not None:
            return str(value)
    return None


def _safe_attribute_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return value[:256]
    if isinstance(value, (list, tuple)):
        safe_items = [
            item
            for raw_item in value
            if (item := _safe_attribute_value(raw_item)) is not None
            and not isinstance(item, bool)
        ]
        return safe_items if safe_items else None
    return None


def _safe_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    if _SAFE_LABEL_PATTERN.fullmatch(normalized):
        return normalized
    return None


def _carrier_from_scope(scope: Scope) -> dict[str, str]:
    return dict(Headers(scope=scope))


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path
    return "unknown"


def _current_request_id() -> Optional[str]:
    from .logging import current_request_id

    return current_request_id()


def _clean_env_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned


__all__ = [
    "DEFAULT_OTLP_TIMEOUT_SECONDS",
    "OTLP_ENDPOINT_ENV_VAR",
    "OTLP_HEADERS_ENV_VAR",
    "OTLP_TIMEOUT_ENV_VAR",
    "OpenTelemetryMiddleware",
    "ServiceTraceConfig",
    "ServiceTracing",
    "TRACING_ENABLED_ENV_VAR",
    "parse_otlp_headers",
    "parse_otlp_timeout",
    "parse_tracing_enabled",
    "result_summary_attributes",
    "safe_trace_attributes",
    "service_trace_config_from_env",
    "service_tracing_from_env",
    "set_current_span_attributes",
    "trace_service_stage",
]
