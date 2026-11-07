"""Request correlation and structured access logging for the REST service."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

ACCESS_LOGGER_NAME = "openmed.service.access"
REQUEST_ID_HEADER = "X-Request-ID"
SERVICE_LOG_LEVEL_ENV_VAR = "OPENMED_SERVICE_LOG_LEVEL"
SERVICE_LOG_FORMAT_ENV_VAR = "OPENMED_SERVICE_LOG_FORMAT"
_MODEL_NAME_SCOPE_KEY = "openmed.access_log_model_name"
_IDENTITY_SCOPE_KEY = "openmed.access_log_identity"
_CREDENTIAL_TYPE_SCOPE_KEY = "openmed.access_log_credential_type"
_GROUNDING_INPUT_COUNT_SCOPE_KEY = "openmed.access_log_grounding_input_count"
_GROUNDING_RESULT_COUNT_SCOPE_KEY = "openmed.access_log_grounding_result_count"
_GROUNDING_SYSTEMS_SCOPE_KEY = "openmed.access_log_grounding_systems"
_GROUNDING_LANG_SCOPE_KEY = "openmed.access_log_grounding_lang"
_BRIEF_SCOPE_KEY = "openmed.access_log_brief"

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
        from .app import _get_service_runtime

        _get_service_runtime(request).validate_served_model(model_name)
        request.scope[_MODEL_NAME_SCOPE_KEY] = str(model_name)


def set_access_log_identity(
    request: Any,
    *,
    principal: str,
    credential_type: str,
) -> None:
    """Attach a non-PHI caller identity to the current request access log."""
    request.scope[_IDENTITY_SCOPE_KEY] = str(principal)
    request.scope[_CREDENTIAL_TYPE_SCOPE_KEY] = str(credential_type)


def set_access_log_grounding(
    request: Any,
    *,
    input_count: int,
    systems: list[str],
    lang: str,
    result_count: Optional[int] = None,
) -> None:
    """Attach content-free grounding dimensions to the request access log."""

    request.scope[_GROUNDING_INPUT_COUNT_SCOPE_KEY] = max(0, int(input_count))
    request.scope[_GROUNDING_SYSTEMS_SCOPE_KEY] = tuple(str(item) for item in systems)
    request.scope[_GROUNDING_LANG_SCOPE_KEY] = str(lang)
    if result_count is not None:
        request.scope[_GROUNDING_RESULT_COUNT_SCOPE_KEY] = max(0, int(result_count))


def set_access_log_brief(request: Any, response: Mapping[str, Any]) -> None:
    """Attach only bounded counts and closed-vocabulary brief outcomes."""
    from openmed.clinical.brief import BriefRefusal

    reason = response.get("refusal_reason")
    backend = response.get("backend_id")
    request.scope[_BRIEF_SCOPE_KEY] = {
        "brief_claim_count": len(response.get("verdicts", [])),
        "brief_summary_characters": int(response.get("summary_characters", 0)),
        "brief_refusal_reason": reason
        if reason in {item.value for item in BriefRefusal}
        else None,
        "brief_backend": backend
        if backend in {"deterministic-extractive", "local-mlx"}
        else "local",
    }


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
                payload = {
                    "method": str(scope.get("method", "")),
                    "route": _route_template(scope),
                    "status_code": status_code,
                    "duration_ms": round(duration_ms, 3),
                    "model_name": scope.get(_MODEL_NAME_SCOPE_KEY),
                    "identity": scope.get(_IDENTITY_SCOPE_KEY),
                    "credential_type": scope.get(_CREDENTIAL_TYPE_SCOPE_KEY),
                    "request_id": request_id,
                }
                if _GROUNDING_INPUT_COUNT_SCOPE_KEY in scope:
                    payload.update(
                        {
                            "grounding_input_count": scope[
                                _GROUNDING_INPUT_COUNT_SCOPE_KEY
                            ],
                            "grounding_result_count": scope.get(
                                _GROUNDING_RESULT_COUNT_SCOPE_KEY
                            ),
                            "grounding_systems": scope[_GROUNDING_SYSTEMS_SCOPE_KEY],
                            "grounding_lang": scope[_GROUNDING_LANG_SCOPE_KEY],
                        }
                    )
                payload.update(scope.get(_BRIEF_SCOPE_KEY, {}))
                emit_access_log(
                    self.logger,
                    self.log_config,
                    payload,
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
    if scope.get("app") is not None:
        # Lazy included routers may leave the request's route path empty.
        # Resolve only a registered effective template; never log raw targets.
        from starlette.requests import Request

        from .auth import route_key_for_request

        try:
            _, template = route_key_for_request(Request(scope))
            return template or "unknown"
        except Exception:
            return "unknown"
    return "unknown"


def _format_access_log(payload: Mapping[str, Any], config: ServiceLogConfig) -> str:
    if config.fmt == "plain":
        rendered = (
            f"{payload.get('method', '')} {payload.get('route', 'unknown')} "
            f"{payload.get('status_code', 0)} "
            f"{payload.get('duration_ms', 0)}ms "
            f"request_id={payload.get('request_id', '')} "
            f"model_name={payload.get('model_name') or '-'} "
            f"identity={payload.get('identity') or '-'} "
            f"credential_type={payload.get('credential_type') or '-'}"
        )
        if "grounding_input_count" in payload:
            rendered += (
                f" grounding_input_count={payload.get('grounding_input_count')}"
                f" grounding_result_count={payload.get('grounding_result_count')}"
                f" grounding_systems={payload.get('grounding_systems') or '-'}"
                f" grounding_lang={payload.get('grounding_lang') or '-'}"
            )
        return rendered
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
    "set_access_log_grounding",
    "set_access_log_identity",
    "set_access_log_model_name",
]


def _launcher_log_config() -> dict[str, Any]:
    """Return the reference launcher's logging configuration.

    Access records go through :class:`StructuredJsonLogFormatter`; raw
    ``uvicorn.access`` lines are dropped. ``uvicorn.error`` keeps its records and
    gains :class:`~openmed.service.request_limits.ConnectionLimitLogFilter`, which
    counts connection-limit rejections for ``/metrics``.
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "connection_limit": {
                "()": "openmed.service.request_limits.ConnectionLimitLogFilter"
            }
        },
        "formatters": {
            "access": {"()": "openmed.service.logging.StructuredJsonLogFormatter"}
        },
        "handlers": {
            "access": {
                "class": "logging.StreamHandler",
                "formatter": "access",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            ACCESS_LOGGER_NAME: {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {"handlers": [], "level": "CRITICAL", "propagate": False},
            "uvicorn.error": {"filters": ["connection_limit"]},
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Launch the REST service with structured access logs and transport bounds.

    The launcher never logs raw request targets. It also bounds concurrent
    connections, idle keep-alive time and graceful shutdown. Those bounds
    default to ``OPENMED_SERVICE_LIMIT_CONCURRENCY``,
    ``OPENMED_SERVICE_TIMEOUT_KEEP_ALIVE`` and
    ``OPENMED_SERVICE_TIMEOUT_GRACEFUL_SHUTDOWN`` (or the built-in defaults of
    128 connections, 5 seconds and 30 seconds), and the matching command-line
    flags override them.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
    """
    import argparse

    import uvicorn

    from .request_limits import (
        ServerConnectionLimits,
        server_connection_limits_from_env,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    try:
        defaults = server_connection_limits_from_env()
    except ValueError as exc:
        parser.error(str(exc))
    parser.add_argument("app", nargs="?", default="openmed.service.app:app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--ssl-keyfile")
    parser.add_argument("--ssl-certfile")
    parser.add_argument("--ssl-ca-certs")
    parser.add_argument("--ssl-cert-reqs", type=int, default=0)
    parser.add_argument(
        "--no-proxy-headers", dest="proxy_headers", action="store_false"
    )
    parser.add_argument("--forwarded-allow-ips")
    parser.add_argument(
        "--limit-concurrency",
        type=int,
        default=defaults.limit_concurrency,
        help="Connections or tasks per worker before answering 503.",
    )
    parser.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=defaults.timeout_keep_alive,
        help="Seconds an idle HTTP keep-alive connection stays open.",
    )
    parser.add_argument(
        "--timeout-graceful-shutdown",
        type=int,
        default=defaults.timeout_graceful_shutdown,
        help="Seconds shutdown waits for open connections before cancelling.",
    )
    options = vars(parser.parse_args(argv))
    try:
        connection_limits = ServerConnectionLimits(
            limit_concurrency=options.pop("limit_concurrency"),
            timeout_keep_alive=options.pop("timeout_keep_alive"),
            timeout_graceful_shutdown=options.pop("timeout_graceful_shutdown"),
        )
    except ValueError as exc:
        parser.error(str(exc))
    uvicorn.run(
        **options,
        **connection_limits.uvicorn_options(),
        access_log=False,
        log_config=_launcher_log_config(),
    )


if __name__ == "__main__":  # pragma: no cover - exercised by process-level tests
    main()
