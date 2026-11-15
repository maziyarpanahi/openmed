"""Opt-in inbound HMAC request-signature enforcement for service routes."""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .auth import ROUTE_POLICIES, _policy_path, _policy_routes, route_key_for_request
from .signing import (
    DEFAULT_MAX_SKEW_SECONDS,
    DEFAULT_NONCE_CACHE_SIZE,
    NONCE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    NonceCache,
    NonceCacheFullError,
    ReplayDetectedError,
    SignatureVerificationError,
    StaleTimestampError,
    verify_request_signature,
)

SERVICE_REQUEST_SIGNING_ROUTES_ENV_VAR = "OPENMED_SERVICE_REQUEST_SIGNING_ROUTES"
SERVICE_REQUEST_SIGNING_SECRET_FILE_ENV_VAR = (
    "OPENMED_SERVICE_REQUEST_SIGNING_SECRET_FILE"
)
SERVICE_REQUEST_SIGNING_MAX_SKEW_ENV_VAR = (
    "OPENMED_SERVICE_REQUEST_SIGNING_MAX_SKEW_SECONDS"
)
SERVICE_REQUEST_SIGNING_NONCE_CACHE_SIZE_ENV_VAR = (
    "OPENMED_SERVICE_REQUEST_SIGNING_NONCE_CACHE_SIZE"
)
MIN_REQUEST_SIGNING_SECRET_BYTES = 32
MAX_REQUEST_SIGNING_SECRET_BYTES = 4096
SIGNATURE_CHALLENGE = "OPENMED-HMAC-SHA256"

_SIGNATURE_HEADERS = (TIMESTAMP_HEADER, NONCE_HEADER, SIGNATURE_HEADER)
_FAILURE_MESSAGES = {
    "signature_missing": "Request signature headers were not provided",
    "signature_invalid": "Request signature is invalid",
    "signature_stale": "Request signature timestamp is outside the accepted window",
    "signature_replayed": "Request signature nonce has already been used",
}

ErrorResponseFactory = Callable[..., Response]


@dataclass(frozen=True)
class RequestSigningConfig:
    """Inbound request-signing settings for the REST service.

    Attributes:
        routes: ``(METHOD, route template)`` pairs that require a signature.
            An empty set disables enforcement.
        secret: Shared HMAC secret. It is excluded from ``repr``.
        max_skew_seconds: Accepted clock skew and nonce replay window.
        nonce_cache_size: Maximum number of active nonces held per process.
    """

    routes: frozenset[tuple[str, str]] = frozenset()
    secret: bytes = field(default=b"", repr=False)
    max_skew_seconds: int = DEFAULT_MAX_SKEW_SECONDS
    nonce_cache_size: int = DEFAULT_NONCE_CACHE_SIZE

    def __post_init__(self) -> None:
        if not self.routes:
            return
        if len(self.secret) < MIN_REQUEST_SIGNING_SECRET_BYTES:
            raise ValueError(
                "Request signing secret must contain at least "
                f"{MIN_REQUEST_SIGNING_SECRET_BYTES} bytes"
            )
        _require_positive_int(self.max_skew_seconds, "max_skew_seconds")
        _require_positive_int(self.nonce_cache_size, "nonce_cache_size")

    @property
    def enabled(self) -> bool:
        """Return whether any route requires a signature."""
        return bool(self.routes)


def parse_request_signing_config() -> RequestSigningConfig:
    """Read inbound request-signing settings from the process environment.

    Returns:
        A disabled configuration when neither routes nor a secret file is
        configured, otherwise a validated enabled configuration.

    Raises:
        ValueError: If the settings are incomplete or invalid. Messages name
            only the environment variable, never a path, route, or secret.
    """
    routes = parse_signed_routes(os.getenv(SERVICE_REQUEST_SIGNING_ROUTES_ENV_VAR))
    raw_secret_file = os.getenv(SERVICE_REQUEST_SIGNING_SECRET_FILE_ENV_VAR)
    has_secret_file = raw_secret_file is not None and bool(raw_secret_file.strip())
    if not routes:
        if has_secret_file:
            raise ValueError(
                f"{SERVICE_REQUEST_SIGNING_ROUTES_ENV_VAR} is required when "
                f"{SERVICE_REQUEST_SIGNING_SECRET_FILE_ENV_VAR} is set"
            )
        return RequestSigningConfig()
    if not has_secret_file:
        raise ValueError(
            f"{SERVICE_REQUEST_SIGNING_SECRET_FILE_ENV_VAR} is required when "
            f"{SERVICE_REQUEST_SIGNING_ROUTES_ENV_VAR} is set"
        )
    return RequestSigningConfig(
        routes=routes,
        secret=_read_secret_file(str(raw_secret_file)),
        max_skew_seconds=_parse_positive_int(
            os.getenv(SERVICE_REQUEST_SIGNING_MAX_SKEW_ENV_VAR),
            env_var=SERVICE_REQUEST_SIGNING_MAX_SKEW_ENV_VAR,
            default=DEFAULT_MAX_SKEW_SECONDS,
        ),
        nonce_cache_size=_parse_positive_int(
            os.getenv(SERVICE_REQUEST_SIGNING_NONCE_CACHE_SIZE_ENV_VAR),
            env_var=SERVICE_REQUEST_SIGNING_NONCE_CACHE_SIZE_ENV_VAR,
            default=DEFAULT_NONCE_CACHE_SIZE,
        ),
    )


def parse_signed_routes(raw_value: Optional[str]) -> frozenset[tuple[str, str]]:
    """Parse comma-separated ``METHOD /template`` entries.

    Every entry must name an HTTP route declared in the service route policy
    table.

    Args:
        raw_value: Raw setting such as ``"POST /jobs,GET /jobs/{job_id}"``.

    Returns:
        The configured ``(METHOD, template)`` pairs.

    Raises:
        ValueError: If an entry is malformed or names an unknown route.
    """
    if raw_value is None or not raw_value.strip():
        return frozenset()
    routes: set[tuple[str, str]] = set()
    for index, entry in enumerate(raw_value.split(",")):
        if not entry.strip():
            continue
        method, separator, path = entry.strip().partition(" ")
        method = method.upper()
        path = path.strip()
        if not separator or not method or not path.startswith("/"):
            raise ValueError(
                f"{SERVICE_REQUEST_SIGNING_ROUTES_ENV_VAR} entry {index} must "
                "look like 'POST /jobs'"
            )
        if method == "WEBSOCKET" or (method, path) not in ROUTE_POLICIES:
            raise ValueError(
                f"{SERVICE_REQUEST_SIGNING_ROUTES_ENV_VAR} entry {index} does not "
                "name a known HTTP service route"
            )
        routes.add((method, path))
    return frozenset(routes)


def validate_signed_routes(app: Any, routes: Iterable[tuple[str, str]]) -> None:
    """Fail closed when a signed route template is not registered on ``app``.

    Args:
        app: The Starlette or FastAPI application that will serve the routes.
        routes: Configured ``(METHOD, template)`` pairs.

    Raises:
        ValueError: If any configured pair has no matching registered route.
    """
    registered: set[tuple[str, str]] = set()
    for route in _policy_routes(app):
        path = _policy_path(route)
        for method in getattr(route, "methods", None) or ():
            registered.add((str(method).upper(), path))
    if any(route not in registered for route in routes):
        raise ValueError(
            f"{SERVICE_REQUEST_SIGNING_ROUTES_ENV_VAR} names a route that is not "
            "registered on this service"
        )


class RequestSignatureVerifier:
    """Verify inbound signatures for configured routes with one nonce cache.

    The nonce cache lives in this process only. Replicas and worker processes
    each hold their own cache, so replay protection is per process.
    """

    def __init__(
        self,
        config: RequestSigningConfig,
        *,
        error_response: ErrorResponseFactory,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not config.enabled:
            raise ValueError("Request signing requires at least one route")
        self.config = config
        self.clock = clock
        self.nonce_cache = NonceCache(
            max_entries=config.nonce_cache_size,
            window_seconds=config.max_skew_seconds,
        )
        self._error_response = error_response
        # A signed GET template also guards HEAD so a router that answers HEAD
        # with the GET handler cannot bypass the check.
        self._routes = config.routes | frozenset(
            ("HEAD", path) for method, path in config.routes if method == "GET"
        )
        self._methods = frozenset(method for method, _ in self._routes)

    def requires_signature(self, scope: Scope) -> bool:
        """Return whether the request's matched route template is configured."""
        if scope.get("type") != "http":
            return False
        if str(scope.get("method", "")).upper() not in self._methods:
            return False
        return route_key_for_request(Request(scope)) in self._routes

    def verify(self, scope: Scope, body: bytes) -> Optional[Response]:
        """Verify one buffered request.

        Args:
            scope: ASGI HTTP scope for the request.
            body: The exact request body bytes.

        Returns:
            ``None`` when the request is authentic and fresh, otherwise a
            value-free error response.
        """
        # Literal ``error_code`` assignments keep these codes discoverable by
        # the API compatibility gate's stable error-category scan.
        headers = Headers(scope=scope)
        if any(headers.get(name) is None for name in _SIGNATURE_HEADERS):
            error_code = "signature_missing"
            return self._reject(error_code)
        try:
            verify_request_signature(
                str(scope.get("method", "")),
                _request_target(scope),
                body,
                headers,
                secret=self.config.secret,
                nonce_cache=self.nonce_cache,
                max_skew_seconds=self.config.max_skew_seconds,
                now=self.clock(),
            )
        except NonceCacheFullError:
            return self._error_response(
                503,
                "service_busy",
                "Request signature replay protection is at capacity",
                details=None,
            )
        except StaleTimestampError:
            error_code = "signature_stale"
        except ReplayDetectedError:
            error_code = "signature_replayed"
        except (SignatureVerificationError, TypeError, ValueError):
            error_code = "signature_invalid"
        else:
            return None
        return self._reject(error_code)

    def body_too_large(self) -> Response:
        """Return the service envelope for an over-limit signed request body."""
        return self._error_response(
            413,
            "payload_too_large",
            "Request body exceeds the configured limit",
            details=None,
        )

    def _reject(self, code: str) -> Response:
        response = self._error_response(
            401,
            code,
            _FAILURE_MESSAGES[code],
            details=None,
        )
        response.headers["WWW-Authenticate"] = SIGNATURE_CHALLENGE
        return response


class RequestSigningMiddleware:
    """Require valid HMAC signatures before configured routes run.

    Install it inside the request-body limit middleware so oversized bodies
    are rejected first. The verified body is replayed to the downstream app
    unchanged. Requests for routes that are not configured pass through
    without being read.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        verifier: RequestSignatureVerifier,
        max_body_bytes: int,
    ) -> None:
        _require_positive_int(max_body_bytes, "max_body_bytes")
        self.app = app
        self.verifier = verifier
        self.max_body_bytes = int(max_body_bytes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.verifier.requires_signature(scope):
            await self.app(scope, receive, send)
            return

        messages: deque[Message] = deque()
        body = bytearray()
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > self.max_body_bytes:
                await self.verifier.body_too_large()(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        rejection = self.verifier.verify(scope, bytes(body))
        if rejection is not None:
            await rejection(scope, receive, send)
            return

        async def replay() -> Message:
            if messages:
                return messages.popleft()
            return await receive()

        await self.app(scope, replay, send)


def install_request_signing(
    app: Any,
    *,
    error_response: ErrorResponseFactory,
    max_body_bytes: int,
) -> Optional[RequestSignatureVerifier]:
    """Install signature enforcement when the environment configures it.

    Call this after every route is registered. When signing is not
    configured, the application is left unchanged.

    Args:
        app: The FastAPI application.
        error_response: Factory for the service error envelope.
        max_body_bytes: Upper bound on buffered signed request bodies.

    Returns:
        The installed verifier, or ``None`` when signing is disabled.

    Raises:
        ValueError: If the configuration is invalid or names a route that is
            not registered on ``app``.
    """
    config = parse_request_signing_config()
    app.state.request_signing = None
    if not config.enabled:
        return None
    validate_signed_routes(app, config.routes)
    verifier = RequestSignatureVerifier(config, error_response=error_response)
    app.state.request_signing = verifier
    app.add_middleware(
        RequestSigningMiddleware,
        verifier=verifier,
        max_body_bytes=max_body_bytes,
    )
    return verifier


def _request_target(scope: Scope) -> str:
    raw_path = scope.get("raw_path")
    if isinstance(raw_path, (bytes, bytearray)) and raw_path:
        target = bytes(raw_path).decode("latin-1")
    else:
        target = str(scope.get("path", ""))
    query = scope.get("query_string") or b""
    if query:
        target = f"{target}?{bytes(query).decode('latin-1')}"
    return target


def _read_secret_file(raw_path: str) -> bytes:
    env_var = SERVICE_REQUEST_SIGNING_SECRET_FILE_ENV_VAR
    try:
        with Path(raw_path.strip()).open("rb") as handle:
            data = handle.read(MAX_REQUEST_SIGNING_SECRET_BYTES + 1)
    except OSError:
        raise ValueError(f"{env_var} must name a readable file") from None
    if len(data) > MAX_REQUEST_SIGNING_SECRET_BYTES:
        raise ValueError(
            f"{env_var} must not exceed {MAX_REQUEST_SIGNING_SECRET_BYTES} bytes"
        )
    secret = data.strip()
    if not secret:
        raise ValueError(f"{env_var} must contain a non-empty secret")
    if len(secret) < MIN_REQUEST_SIGNING_SECRET_BYTES:
        raise ValueError(
            f"{env_var} secret must contain at least "
            f"{MIN_REQUEST_SIGNING_SECRET_BYTES} bytes"
        )
    return secret


def _parse_positive_int(raw_value: Optional[str], *, env_var: str, default: int) -> int:
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value.strip())
    except ValueError:
        raise ValueError(f"{env_var} must be a positive integer") from None
    if value < 1:
        raise ValueError(f"{env_var} must be a positive integer")
    return value


def _require_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "MIN_REQUEST_SIGNING_SECRET_BYTES",
    "RequestSignatureVerifier",
    "RequestSigningConfig",
    "RequestSigningMiddleware",
    "SERVICE_REQUEST_SIGNING_MAX_SKEW_ENV_VAR",
    "SERVICE_REQUEST_SIGNING_NONCE_CACHE_SIZE_ENV_VAR",
    "SERVICE_REQUEST_SIGNING_ROUTES_ENV_VAR",
    "SERVICE_REQUEST_SIGNING_SECRET_FILE_ENV_VAR",
    "SIGNATURE_CHALLENGE",
    "install_request_signing",
    "parse_request_signing_config",
    "parse_signed_routes",
    "validate_signed_routes",
]
