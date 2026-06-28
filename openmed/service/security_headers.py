"""Security middleware configuration for the OpenMed REST service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

SERVICE_CORS_ORIGINS_ENV_VAR = "OPENMED_SERVICE_CORS_ORIGINS"
SERVICE_TRUSTED_HOSTS_ENV_VAR = "OPENMED_SERVICE_TRUSTED_HOSTS"
DEFAULT_TRUSTED_HOSTS = ("localhost", "127.0.0.1", "[::1]")
SERVICE_CORS_METHODS = ("GET", "POST")
SERVICE_CORS_HEADERS = (
    "Accept",
    "Accept-Language",
    "Content-Language",
    "Content-Type",
    "X-Request-ID",
)


@dataclass(frozen=True)
class ServiceSecurityConfig:
    """CORS and trusted-host allowlists for the REST service."""

    cors_origins: Tuple[str, ...] = ()
    trusted_hosts: Tuple[str, ...] = DEFAULT_TRUSTED_HOSTS


def _parse_csv_allowlist(raw_value: Optional[str]) -> Tuple[str, ...]:
    if raw_value is None:
        return ()

    values = []
    seen = set()
    for item in raw_value.split(","):
        value = item.strip()
        if not value or value in seen:
            continue
        values.append(value)
        seen.add(value)
    return tuple(values)


def parse_cors_origins(raw_value: Optional[str]) -> Tuple[str, ...]:
    """Parse exact CORS origins from a comma-separated environment value."""
    origins = _parse_csv_allowlist(raw_value)
    if any("*" in origin for origin in origins):
        raise ValueError(
            f"{SERVICE_CORS_ORIGINS_ENV_VAR} must contain exact origins, not wildcards"
        )
    return origins


def parse_trusted_hosts(raw_value: Optional[str]) -> Tuple[str, ...]:
    """Parse trusted hosts or return the loopback-only default allowlist."""
    hosts = _parse_csv_allowlist(raw_value)
    if not hosts:
        return DEFAULT_TRUSTED_HOSTS
    if "*" in hosts:
        raise ValueError(
            f"{SERVICE_TRUSTED_HOSTS_ENV_VAR} must not contain the global '*' host"
        )
    return hosts


def parse_service_security_config() -> ServiceSecurityConfig:
    """Read CORS and trusted-host settings from the process environment."""
    return ServiceSecurityConfig(
        cors_origins=parse_cors_origins(os.getenv(SERVICE_CORS_ORIGINS_ENV_VAR)),
        trusted_hosts=parse_trusted_hosts(os.getenv(SERVICE_TRUSTED_HOSTS_ENV_VAR)),
    )


class ErrorEnvelopeTrustedHostMiddleware(TrustedHostMiddleware):
    """Trusted-host middleware that returns the service JSON error envelope."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.allow_any or scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        host = _host_without_port(headers.get("host", ""))
        is_valid_host = any(
            host == pattern or (pattern.startswith("*.") and host.endswith(pattern[1:]))
            for pattern in self.allowed_hosts
        )
        if is_valid_host:
            await self.app(scope, receive, send)
            return

        message = "Invalid host header"
        response = JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "bad_request",
                    "message": message,
                    "details": {"reason": message},
                }
            },
        )
        await response(scope, receive, send)


def _host_without_port(host_header: str) -> str:
    if host_header.startswith("["):
        closing_bracket = host_header.find("]")
        if closing_bracket != -1:
            return host_header[: closing_bracket + 1]
    return host_header.split(":", 1)[0]
