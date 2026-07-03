"""Request correlation and structured access logging for the REST service."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

ACCESS_LOGGER_NAME = "openmed.service.access"
REQUEST_ID_HEADER = "X-Request-ID"
SERVICE_LOG_LEVEL_ENV_VAR = "OPENMED_SERVICE_LOG_LEVEL"
SERVICE_LOG_FORMAT_ENV_VAR = "OPENMED_SERVICE_LOG_FORMAT"
_MODEL_NAME_SCOPE_KEY = "openmed.access_log_model_name"

_REQUEST_ID: ContextVar[Optional[str]] = ContextVar(
    "openmed_service_request_id",
    default=None,
)


@dataclass(frozen=True)
class ServiceLogConfig:
    """Access-log rendering and severity settings."""

    level: int
    fmt: str = "json"

    @property
    def enabled(self) -> bool:
        return self.level <= logging.CRITICAL


class StructuredJsonLogFormatter(logging.Formatter):
    """Render OpenMed log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        access_log = getattr(record, "openmed_access_log", None)
        if isinstance(access_log, Mapping):
            return _json_dumps(access_log)
        return _json_dumps(
            {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


def service_log_config_from_env() -> ServiceLogConfig:
    """Return service logging settings from environment variables."""
    return ServiceLogConfig(
        level=_parse_log_level(os.getenv(SERVICE_LOG_LEVEL_ENV_VAR)),
        fmt=_parse_log_format(os.getenv(SERVICE_LOG_FORMAT_ENV_VAR)),
    )


def current_request_id() -> Optional[str]:
    """Return the request ID active in the current context, if any."""
    return _REQUEST_ID.get()


def set_access_log_model_name(request: Any, model_name: Optional[str]) -> None:
    """Attach the parsed model name to the current request scope for logging."""
    if model_name:
        request.scope[_MODEL_NAME_SCOPE_KEY] = str(model_name)


class CorrelationIdMiddleware:
    """Add request IDs and emit PHI-free structured access logs."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        log_config: Optional[ServiceLogConfig] = None,
    ) -> None:
        self.app = app
        self.log_config = log_config or service_log_config_from_env()
        self.logger = logging.getLogger(ACCESS_LOGGER_NAME)
        self.logger.setLevel(self.log_config.level)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from_scope(scope)
        request_token = _REQUEST_ID.set(request_id)
        start_time = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            try:
                emit_access_log(
                    self.logger,
                    self.log_config,
                    {
                        "method": str(scope.get("method", "")),
                        "route": _route_template(scope),
                        "status_code": status_code,
                        "duration_ms": round(duration_ms, 3),
                        "model_name": scope.get(_MODEL_NAME_SCOPE_KEY),
                        "request_id": request_id,
                    },
                )
            finally:
                _REQUEST_ID.reset(request_token)


def emit_access_log(
    logger: logging.Logger,
    config: ServiceLogConfig,
    payload: Mapping[str, Any],
) -> None:
    """Emit one access-log record, suppressing logging failures."""
    if not config.enabled:
        return
    try:
        logger.log(
            config.level,
            _format_access_log(payload, config),
            extra={"openmed_access_log": dict(payload)},
        )
    except Exception:
        return


def _request_id_from_scope(scope: Scope) -> str:
    inbound = Headers(scope=scope).get(REQUEST_ID_HEADER)
    if inbound:
        return inbound
    return str(uuid.uuid4())


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path
    return "unknown"


def _format_access_log(payload: Mapping[str, Any], config: ServiceLogConfig) -> str:
    if config.fmt == "plain":
        return (
            f"{payload.get('method', '')} {payload.get('route', 'unknown')} "
            f"{payload.get('status_code', 0)} "
            f"{payload.get('duration_ms', 0)}ms "
            f"request_id={payload.get('request_id', '')} "
            f"model_name={payload.get('model_name') or '-'}"
        )
    return _json_dumps(payload)


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _parse_log_level(raw_value: Optional[str]) -> int:
    if raw_value is None:
        return logging.INFO
    value = raw_value.strip().upper()
    if value in {"", "INFO"}:
        return logging.INFO
    if value in {"OFF", "NONE", "DISABLED"}:
        return logging.CRITICAL + 1
    parsed = logging.getLevelName(value)
    if isinstance(parsed, int):
        return parsed
    try:
        numeric = int(value)
    except ValueError:
        return logging.INFO
    if numeric < 0:
        return logging.INFO
    return numeric


def _parse_log_format(raw_value: Optional[str]) -> str:
    if raw_value is None:
        return "json"
    value = raw_value.strip().lower()
    if value in {"plain", "text"}:
        return "plain"
    return "json"


__all__ = [
    "ACCESS_LOGGER_NAME",
    "REQUEST_ID_HEADER",
    "SERVICE_LOG_FORMAT_ENV_VAR",
    "SERVICE_LOG_LEVEL_ENV_VAR",
    "CorrelationIdMiddleware",
    "ServiceLogConfig",
    "StructuredJsonLogFormatter",
    "current_request_id",
    "service_log_config_from_env",
    "set_access_log_model_name",
]
