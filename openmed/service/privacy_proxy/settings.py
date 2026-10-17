"""Inbound security settings for the local privacy proxy.

The proxy reuses the service trusted-host, request-body and authentication
contracts instead of keeping a parallel security stack.  Listener-specific
values use ``OPENMED_PRIVACY_PROXY_*`` settings; credentials reuse the shared
``OPENMED_SERVICE_AUTH_*`` settings and stay disabled unless configured.
Configuration errors name the setting only and never echo its value.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Optional

from openmed.guard.operational_limits import OperationalLimits

from ..auth import ServiceAuthConfig, parse_service_auth_config
from ..limits import DEFAULT_MAX_TEXT_LENGTH
from ..request_limits import request_limits_from_env
from ..security_headers import parse_trusted_hosts

PRIVACY_PROXY_TRUSTED_HOSTS_ENV_VAR: Final = "OPENMED_PRIVACY_PROXY_TRUSTED_HOSTS"
PRIVACY_PROXY_MAX_REQUEST_BYTES_ENV_VAR: Final = (
    "OPENMED_PRIVACY_PROXY_MAX_REQUEST_BYTES"
)
PRIVACY_PROXY_MAX_MESSAGES_ENV_VAR: Final = "OPENMED_PRIVACY_PROXY_MAX_MESSAGES"
PRIVACY_PROXY_MAX_TEXT_CHARS_ENV_VAR: Final = "OPENMED_PRIVACY_PROXY_MAX_TEXT_CHARS"
PRIVACY_PROXY_MAX_STREAM_SECONDS_ENV_VAR: Final = (
    "OPENMED_PRIVACY_PROXY_MAX_STREAM_SECONDS"
)

DEFAULT_MAX_MESSAGES: Final = 256
DEFAULT_MAX_TEXT_CHARS: Final = DEFAULT_MAX_TEXT_LENGTH
DEFAULT_MAX_STREAM_SECONDS: Final = 300.0
_MAX_MESSAGES_CEILING: Final = 100_000
_MAX_TEXT_CHARS_CEILING: Final = 64 * 1024 * 1024
_MAX_STREAM_SECONDS_CEILING: Final = 86_400.0

# Completion routes redact, call the transport and restore, which matches the
# REST privacy-gateway route, so they share its scope.
PRIVACY_PROXY_ROUTE_SCOPES: Final[Mapping[tuple[str, str], tuple[str, ...]]] = {
    ("POST", "/v1/chat/completions"): ("pii:write",),
    ("POST", "/chat/completions"): ("pii:write",),
}


@dataclass(frozen=True)
class PrivacyProxyLimits:
    """Work bounds checked before redaction or any transport call.

    Attributes:
        max_messages: Maximum number of chat messages in one request.
        max_text_chars: Maximum total characters across every text value the
            proxy would send to the local extractor.
        max_stream_seconds: Maximum wall-clock seconds one streaming response
            may take before it ends with a controlled timeout event.
    """

    max_messages: int = DEFAULT_MAX_MESSAGES
    max_text_chars: int = DEFAULT_MAX_TEXT_CHARS
    max_stream_seconds: float = DEFAULT_MAX_STREAM_SECONDS

    def __post_init__(self) -> None:
        for value, ceiling in (
            (self.max_messages, _MAX_MESSAGES_CEILING),
            (self.max_text_chars, _MAX_TEXT_CHARS_CEILING),
        ):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError("privacy proxy limit is outside the safe range")
        seconds = self.max_stream_seconds
        if (
            type(seconds) not in {int, float}
            or not math.isfinite(seconds)
            or not 0 < seconds <= _MAX_STREAM_SECONDS_CEILING
        ):
            raise ValueError("privacy proxy stream limit is outside the safe range")
        object.__setattr__(self, "max_stream_seconds", float(seconds))


def privacy_proxy_limits_from_env() -> PrivacyProxyLimits:
    """Load privacy-proxy work bounds from the process environment.

    Returns:
        Limits with any configured overrides applied.

    Raises:
        ValueError: If a configured value is not a positive number within the
            safe range. The message names the setting, never its value.
    """
    return PrivacyProxyLimits(
        max_messages=_positive_int_from_env(
            PRIVACY_PROXY_MAX_MESSAGES_ENV_VAR,
            default=DEFAULT_MAX_MESSAGES,
            ceiling=_MAX_MESSAGES_CEILING,
        ),
        max_text_chars=_positive_int_from_env(
            PRIVACY_PROXY_MAX_TEXT_CHARS_ENV_VAR,
            default=DEFAULT_MAX_TEXT_CHARS,
            ceiling=_MAX_TEXT_CHARS_CEILING,
        ),
        max_stream_seconds=_positive_float_from_env(
            PRIVACY_PROXY_MAX_STREAM_SECONDS_ENV_VAR,
            default=DEFAULT_MAX_STREAM_SECONDS,
            ceiling=_MAX_STREAM_SECONDS_CEILING,
        ),
    )


def privacy_proxy_request_limits_from_env() -> OperationalLimits:
    """Load the proxy request-byte limit through the shared service parser."""
    return request_limits_from_env(PRIVACY_PROXY_MAX_REQUEST_BYTES_ENV_VAR)


def resolve_trusted_hosts(hosts: Optional[Sequence[str]] = None) -> tuple[str, ...]:
    """Return the proxy trusted-host allowlist.

    Args:
        hosts: Explicit host names. ``None`` reads
            ``OPENMED_PRIVACY_PROXY_TRUSTED_HOSTS`` and falls back to the
            loopback-only service default.

    Returns:
        The de-duplicated allowlist.

    Raises:
        ValueError: If the allowlist is explicitly empty or contains the
            global ``*`` host.
    """
    if hosts is None:
        return parse_trusted_hosts(
            os.getenv(PRIVACY_PROXY_TRUSTED_HOSTS_ENV_VAR),
            env_var=PRIVACY_PROXY_TRUSTED_HOSTS_ENV_VAR,
        )
    if isinstance(hosts, str):
        hosts = (hosts,)
    normalized = tuple(dict.fromkeys(host.strip() for host in hosts if host.strip()))
    if not normalized or "*" in normalized:
        raise ValueError(
            f"trusted_hosts ({PRIVACY_PROXY_TRUSTED_HOSTS_ENV_VAR}) must list "
            "explicit hosts without the global '*' host"
        )
    return normalized


def resolve_auth_config(
    config: Optional[ServiceAuthConfig] = None,
) -> ServiceAuthConfig:
    """Return the shared service auth contract with proxy route scopes.

    Args:
        config: Explicit authentication settings. ``None`` reads the shared
            ``OPENMED_SERVICE_AUTH_*`` settings, which keep auth disabled
            unless ``OPENMED_SERVICE_AUTH_ENABLED`` is set.

    Returns:
        Settings whose route scopes declare both completion routes, so they
        require credentials even when deny-by-default is turned off. Scopes
        already present in ``config`` are kept.
    """
    active = config if config is not None else parse_service_auth_config()
    route_scopes = dict(active.route_scopes)
    for route, scopes in PRIVACY_PROXY_ROUTE_SCOPES.items():
        route_scopes.setdefault(route, scopes)
    return replace(active, route_scopes=route_scopes)


def _positive_int_from_env(env_var: str, *, default: int, ceiling: int) -> int:
    raw = os.getenv(env_var)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{env_var} must be a positive integer") from None
    if not 1 <= value <= ceiling:
        raise ValueError(f"{env_var} must be between 1 and {ceiling}")
    return value


def _positive_float_from_env(
    env_var: str,
    *,
    default: float,
    ceiling: float,
) -> float:
    raw = os.getenv(env_var)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{env_var} must be a positive number of seconds") from None
    if not math.isfinite(value) or not 0 < value <= ceiling:
        raise ValueError(f"{env_var} must be greater than 0 and at most {ceiling:g}")
    return value


__all__ = [
    "DEFAULT_MAX_MESSAGES",
    "DEFAULT_MAX_STREAM_SECONDS",
    "DEFAULT_MAX_TEXT_CHARS",
    "PRIVACY_PROXY_MAX_MESSAGES_ENV_VAR",
    "PRIVACY_PROXY_MAX_REQUEST_BYTES_ENV_VAR",
    "PRIVACY_PROXY_MAX_STREAM_SECONDS_ENV_VAR",
    "PRIVACY_PROXY_MAX_TEXT_CHARS_ENV_VAR",
    "PRIVACY_PROXY_ROUTE_SCOPES",
    "PRIVACY_PROXY_TRUSTED_HOSTS_ENV_VAR",
    "PrivacyProxyLimits",
    "privacy_proxy_limits_from_env",
    "privacy_proxy_request_limits_from_env",
    "resolve_auth_config",
    "resolve_trusted_hosts",
]
