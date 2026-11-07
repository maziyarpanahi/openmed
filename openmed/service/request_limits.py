"""ASGI request-body and connection bounds shared by REST and GraphQL routes.

Every ``POST``, ``PUT`` and ``PATCH`` body is bounded in size and in receive
time before route parsing or model work. A body that stalls, or trickles in
below a minimum transfer rate, is rejected with ``408 request_body_timeout``
and its buffered chunks are released. The reference launcher
(``python -m openmed.service.logging``) also bounds concurrent connections,
idle keep-alive time and graceful shutdown through
:class:`ServerConnectionLimits`. Rejections carry only value-free codes and
counters are aggregate: no body content, header value or client address is
recorded.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol

from starlette.datastructures import Headers
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from openmed.guard.operational_limits import OperationalLimits

MAX_REQUEST_BYTES_ENV_VAR: Final = "OPENMED_SERVICE_MAX_REQUEST_BYTES"
REQUEST_BODY_TIMEOUT_ENV_VAR: Final = "OPENMED_SERVICE_REQUEST_BODY_TIMEOUT_SECONDS"
REQUEST_BODY_GRACE_ENV_VAR: Final = "OPENMED_SERVICE_REQUEST_BODY_GRACE_SECONDS"
REQUEST_BODY_MIN_RATE_ENV_VAR: Final = (
    "OPENMED_SERVICE_REQUEST_BODY_MIN_BYTES_PER_SECOND"
)
LIMIT_CONCURRENCY_ENV_VAR: Final = "OPENMED_SERVICE_LIMIT_CONCURRENCY"
TIMEOUT_KEEP_ALIVE_ENV_VAR: Final = "OPENMED_SERVICE_TIMEOUT_KEEP_ALIVE"
TIMEOUT_GRACEFUL_SHUTDOWN_ENV_VAR: Final = "OPENMED_SERVICE_TIMEOUT_GRACEFUL_SHUTDOWN"

REQUEST_SIZE_INVALID: Final = "request_size_invalid"
REQUEST_SIZE_EXCEEDED: Final = "request_size_exceeded"
REQUEST_BODY_TIMEOUT: Final = "request_body_timeout"
BODY_TIMEOUT_DEADLINE: Final = "deadline"
BODY_TIMEOUT_MIN_RATE: Final = "min_rate"
BODY_TIMEOUT_REASONS: Final = frozenset({BODY_TIMEOUT_DEADLINE, BODY_TIMEOUT_MIN_RATE})

DEFAULT_REQUEST_BODY_TIMEOUT_SECONDS: Final = 60.0
DEFAULT_REQUEST_BODY_GRACE_SECONDS: Final = 10.0
DEFAULT_REQUEST_BODY_MIN_BYTES_PER_SECOND: Final = 1024
DEFAULT_LIMIT_CONCURRENCY: Final = 128
DEFAULT_TIMEOUT_KEEP_ALIVE_SECONDS: Final = 5
DEFAULT_TIMEOUT_GRACEFUL_SHUTDOWN_SECONDS: Final = 30

_MAX_BODY_SECONDS: Final = 3600.0
_MAX_MIN_BYTES_PER_SECOND: Final = 1024 * 1024 * 1024
_MAX_LIMIT_CONCURRENCY: Final = 100_000
_MAX_SERVER_SECONDS: Final = 3600
_BODY_METHODS: Final = frozenset({"POST", "PUT", "PATCH"})
_REJECTION_STATUS: Final = {
    REQUEST_SIZE_INVALID: 400,
    REQUEST_SIZE_EXCEEDED: 413,
    REQUEST_BODY_TIMEOUT: 408,
}
# Uvicorn answers 503 itself at ``--limit-concurrency`` and reports it only with
# this fixed warning on ``uvicorn.error``; it carries no client address.
_CONNECTION_LIMIT_WARNING: Final = "Exceeded concurrency limit."


def request_limits_from_env() -> OperationalLimits:
    """Load the bounded request-byte override from the environment."""

    default = OperationalLimits()
    raw = os.getenv(MAX_REQUEST_BYTES_ENV_VAR)
    if raw is None or not raw.strip():
        return default
    try:
        maximum = int(raw)
    except ValueError:
        raise ValueError(
            f"{MAX_REQUEST_BYTES_ENV_VAR} must be an integer number of bytes"
        ) from None
    return OperationalLimits(max_request_bytes=maximum)


@dataclass(frozen=True, slots=True)
class RequestBodyReadLimits:
    """Time bounds for receiving one request body.

    The whole body must arrive within ``timeout_seconds``. Once the
    ``grace_seconds`` allowance is spent, the body must also keep up with
    ``min_bytes_per_second``: every received byte extends the allowance by
    ``1 / min_bytes_per_second`` seconds, so a stalled or trickling client is
    rejected while a steady upload is not.

    Attributes:
        timeout_seconds: Total seconds allowed to receive the whole body.
        grace_seconds: Seconds allowed before the minimum rate applies.
        min_bytes_per_second: Minimum average transfer rate. ``0`` disables
            the rate check and leaves only the total deadline.
    """

    timeout_seconds: float = DEFAULT_REQUEST_BODY_TIMEOUT_SECONDS
    grace_seconds: float = DEFAULT_REQUEST_BODY_GRACE_SECONDS
    min_bytes_per_second: int = DEFAULT_REQUEST_BODY_MIN_BYTES_PER_SECOND

    def __post_init__(self) -> None:
        for name in ("timeout_seconds", "grace_seconds"):
            value = getattr(self, name)
            if (
                type(value) not in {int, float}
                or not math.isfinite(value)
                or not 0 < value <= _MAX_BODY_SECONDS
            ):
                raise ValueError("request-body read limit is outside the safe range")
            object.__setattr__(self, name, float(value))
        rate = self.min_bytes_per_second
        if type(rate) is not int or not 0 <= rate <= _MAX_MIN_BYTES_PER_SECOND:
            raise ValueError("request-body read limit is outside the safe range")


def request_body_read_limits_from_env(
    environ: Mapping[str, str] | None = None,
) -> RequestBodyReadLimits:
    """Load request-body time bounds from the environment.

    Args:
        environ: Mapping to read instead of ``os.environ``.

    Returns:
        Limits built from ``OPENMED_SERVICE_REQUEST_BODY_TIMEOUT_SECONDS``,
        ``OPENMED_SERVICE_REQUEST_BODY_GRACE_SECONDS`` and
        ``OPENMED_SERVICE_REQUEST_BODY_MIN_BYTES_PER_SECOND``; unset or empty
        variables keep the defaults.

    Raises:
        ValueError: A variable is not a number inside its safe range. The
            message names the variable, never its value.
    """

    source = os.environ if environ is None else environ
    return RequestBodyReadLimits(
        timeout_seconds=_env_seconds(
            source, REQUEST_BODY_TIMEOUT_ENV_VAR, DEFAULT_REQUEST_BODY_TIMEOUT_SECONDS
        ),
        grace_seconds=_env_seconds(
            source, REQUEST_BODY_GRACE_ENV_VAR, DEFAULT_REQUEST_BODY_GRACE_SECONDS
        ),
        min_bytes_per_second=_env_int(
            source,
            REQUEST_BODY_MIN_RATE_ENV_VAR,
            DEFAULT_REQUEST_BODY_MIN_BYTES_PER_SECOND,
            minimum=0,
            maximum=_MAX_MIN_BYTES_PER_SECOND,
        ),
    )


class RequestBodyTimeout(Exception):
    """Raised when a request body misses its receive deadline.

    Attributes:
        reason: ``"deadline"`` when the total time ran out, ``"min_rate"``
            when the body fell below the minimum transfer rate.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(REQUEST_BODY_TIMEOUT)


class RequestBodyDeadline:
    """Receive one request body's ASGI messages within its time bounds."""

    def __init__(
        self,
        limits: RequestBodyReadLimits,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Start the deadline for one request body.

        Args:
            limits: Total deadline, grace allowance and minimum rate.
            clock: Monotonic seconds source; tests inject a fake clock.
        """

        self._limits = limits
        self._clock = clock
        self._started = clock()
        self._received = 0

    @property
    def received_bytes(self) -> int:
        """Return the body bytes received so far."""

        return self._received

    def remaining_seconds(self) -> float:
        """Return the seconds left before the body is rejected."""

        allowed = self._limits.timeout_seconds
        rate_allowance = self._rate_allowance()
        if rate_allowance is not None:
            allowed = min(allowed, rate_allowance)
        return allowed - (self._clock() - self._started)

    async def receive(self, receive: Receive) -> Message:
        """Await the next ASGI message without exceeding the deadline.

        Args:
            receive: The downstream ASGI receive callable.

        Returns:
            The next ASGI message.

        Raises:
            RequestBodyTimeout: No message arrived in time, or the bytes
                received so far fell behind the minimum transfer rate.
        """

        remaining = self.remaining_seconds()
        if remaining <= 0:
            raise RequestBodyTimeout(self._reason())
        timeout = asyncio.timeout(remaining)
        try:
            async with timeout:
                message = await receive()
        except TimeoutError:
            if not timeout.expired():
                raise
            raise RequestBodyTimeout(self._reason()) from None
        if message["type"] == "http.request":
            self._received += len(message.get("body", b""))
            if self.remaining_seconds() < 0:
                raise RequestBodyTimeout(self._reason())
        return message

    def _rate_allowance(self) -> float | None:
        rate = self._limits.min_bytes_per_second
        if not rate:
            return None
        return self._limits.grace_seconds + self._received / rate

    def _reason(self) -> str:
        # Name the bound that was binding, independent of timer precision.
        rate_allowance = self._rate_allowance()
        if rate_allowance is not None and (
            rate_allowance < self._limits.timeout_seconds
        ):
            return BODY_TIMEOUT_MIN_RATE
        return BODY_TIMEOUT_DEADLINE


class RequestBodyTimeoutRecorder(Protocol):
    """Aggregate sink for request-body timeouts, such as the metrics registry."""

    def record_request_body_timeout(self, *, reason: str) -> None:
        """Count one body rejected for ``reason``."""


BodyRejectionResponder = Callable[[str], Response]


def body_limit_response(code: str) -> Response:
    """Return the shared value-free envelope for a request-body rejection.

    Args:
        code: ``request_size_invalid``, ``request_size_exceeded`` or
            ``request_body_timeout``.

    Returns:
        A JSON error response with status ``400``, ``413`` or ``408``.
    """

    return _limit_response(_REJECTION_STATUS[code], code)


class BoundedRequestBodyMiddleware:
    """Bound request-body size and receive time before parsing or model work.

    The middleware buffers ``POST``, ``PUT`` and ``PATCH`` bodies, rejects a
    declared or observed size over the byte budget, and rejects a body that
    misses its :class:`RequestBodyReadLimits` deadline with ``408`` and
    ``Connection: close``. Buffered chunks are dropped before any rejection
    is sent. Accepted bodies are replayed to the app in their original order.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        limits: OperationalLimits | None = None,
        max_bytes: int | None = None,
        limited_paths: Iterable[str] | None = None,
        read_limits: RequestBodyReadLimits | None = None,
        rejection_response: BodyRejectionResponder | None = None,
        metrics: RequestBodyTimeoutRecorder | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wrap an ASGI app with request-body bounds.

        Args:
            app: The downstream ASGI application.
            limits: Operational limits whose ``max_request_bytes`` is the byte
                budget. Defaults to :class:`OperationalLimits`.
            max_bytes: Byte budget that replaces ``limits.max_request_bytes``,
                for path-scoped instances whose budget comes from elsewhere.
            limited_paths: Exact request paths to bound. ``None`` bounds every
                path.
            read_limits: Receive-time bounds. Defaults to
                :class:`RequestBodyReadLimits`.
            rejection_response: Builds the response for a rejection code.
                Defaults to :func:`body_limit_response`.
            metrics: Optional aggregate sink for body timeouts.
            clock: Monotonic seconds source; tests inject a fake clock.

        Raises:
            ValueError: ``max_bytes`` is not a positive integer.
        """

        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 1):
            raise ValueError("max request body bytes must be positive")
        self.app = app
        self.limits = limits if limits is not None else OperationalLimits()
        self.max_bytes = (
            self.limits.max_request_bytes if max_bytes is None else max_bytes
        )
        self.limited_paths = None if limited_paths is None else frozenset(limited_paths)
        self.read_limits = (
            read_limits if read_limits is not None else RequestBodyReadLimits()
        )
        self._rejection_response = rejection_response or body_limit_response
        self._metrics = metrics
        self._clock = clock

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") not in _BODY_METHODS
            or (
                self.limited_paths is not None
                and scope.get("path") not in self.limited_paths
            )
        ):
            await self.app(scope, receive, send)
            return

        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                await self._reject(REQUEST_SIZE_INVALID, scope, receive, send)
                return
            if declared < 0:
                await self._reject(REQUEST_SIZE_INVALID, scope, receive, send)
                return
            if declared > self.max_bytes:
                await self._reject(REQUEST_SIZE_EXCEEDED, scope, receive, send)
                return

        buffered = await self._buffer_body(receive)
        if buffered is None:
            return
        if isinstance(buffered, str):
            await self._reject(buffered, scope, receive, send)
            return
        messages = buffered

        async def replay() -> Message:
            if messages:
                return messages.popleft()
            return await receive()

        await self.app(scope, replay, send)

    async def _buffer_body(self, receive: Receive) -> deque[Message] | str | None:
        """Buffer a whole body, or return a rejection code with nothing held."""

        deadline = RequestBodyDeadline(self.read_limits, clock=self._clock)
        messages: deque[Message] = deque()
        while True:
            try:
                message = await deadline.receive(receive)
            except RequestBodyTimeout as timeout:
                messages.clear()
                if self._metrics is not None:
                    self._metrics.record_request_body_timeout(reason=timeout.reason)
                return REQUEST_BODY_TIMEOUT
            messages.append(message)
            if message["type"] == "http.disconnect":
                return None
            if message["type"] != "http.request":
                continue
            if deadline.received_bytes > self.max_bytes:
                messages.clear()
                return REQUEST_SIZE_EXCEEDED
            if not message.get("more_body", False):
                return messages

    async def _reject(
        self, code: str, scope: Scope, receive: Receive, send: Send
    ) -> None:
        response = self._rejection_response(code)
        if code == REQUEST_BODY_TIMEOUT:
            response.headers["connection"] = "close"
        await response(scope, receive, send)


@dataclass(frozen=True, slots=True)
class ServerConnectionLimits:
    """Uvicorn connection bounds applied by the reference launcher.

    Attributes:
        limit_concurrency: Concurrent connections or tasks per worker process
            before Uvicorn answers ``503`` without calling the app.
        timeout_keep_alive: Seconds an idle HTTP keep-alive connection stays
            open. Distinct from the model ``OPENMED_SERVICE_KEEP_ALIVE``.
        timeout_graceful_shutdown: Seconds shutdown waits for open
            connections and tasks before cancelling them.
    """

    limit_concurrency: int = DEFAULT_LIMIT_CONCURRENCY
    timeout_keep_alive: int = DEFAULT_TIMEOUT_KEEP_ALIVE_SECONDS
    timeout_graceful_shutdown: int = DEFAULT_TIMEOUT_GRACEFUL_SHUTDOWN_SECONDS

    def __post_init__(self) -> None:
        bounds = {
            "limit_concurrency": _MAX_LIMIT_CONCURRENCY,
            "timeout_keep_alive": _MAX_SERVER_SECONDS,
            "timeout_graceful_shutdown": _MAX_SERVER_SECONDS,
        }
        for name, maximum in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer from 1 to {maximum}")

    def uvicorn_options(self) -> dict[str, int]:
        """Return the matching ``uvicorn.run`` keyword arguments."""

        return {
            "limit_concurrency": self.limit_concurrency,
            "timeout_keep_alive": self.timeout_keep_alive,
            "timeout_graceful_shutdown": self.timeout_graceful_shutdown,
        }


def server_connection_limits_from_env(
    environ: Mapping[str, str] | None = None,
) -> ServerConnectionLimits:
    """Load the reference launcher's connection bounds from the environment.

    Args:
        environ: Mapping to read instead of ``os.environ``.

    Returns:
        Limits built from ``OPENMED_SERVICE_LIMIT_CONCURRENCY``,
        ``OPENMED_SERVICE_TIMEOUT_KEEP_ALIVE`` and
        ``OPENMED_SERVICE_TIMEOUT_GRACEFUL_SHUTDOWN``; unset or empty
        variables keep the defaults.

    Raises:
        ValueError: A variable is not an integer inside its safe range. The
            message names the variable, never its value.
    """

    source = os.environ if environ is None else environ
    return ServerConnectionLimits(
        limit_concurrency=_env_int(
            source,
            LIMIT_CONCURRENCY_ENV_VAR,
            DEFAULT_LIMIT_CONCURRENCY,
            minimum=1,
            maximum=_MAX_LIMIT_CONCURRENCY,
        ),
        timeout_keep_alive=_env_int(
            source,
            TIMEOUT_KEEP_ALIVE_ENV_VAR,
            DEFAULT_TIMEOUT_KEEP_ALIVE_SECONDS,
            minimum=1,
            maximum=_MAX_SERVER_SECONDS,
        ),
        timeout_graceful_shutdown=_env_int(
            source,
            TIMEOUT_GRACEFUL_SHUTDOWN_ENV_VAR,
            DEFAULT_TIMEOUT_GRACEFUL_SHUTDOWN_SECONDS,
            minimum=1,
            maximum=_MAX_SERVER_SECONDS,
        ),
    )


class _AggregateCounter:
    """Thread-safe process-wide count with no labels."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def increment(self) -> None:
        with self._lock:
            self._value += 1

    def value(self) -> int:
        with self._lock:
            return self._value


_CONNECTION_REJECTIONS = _AggregateCounter()


def connection_rejection_count() -> int:
    """Return requests this process rejected at the connection limit."""

    return _CONNECTION_REJECTIONS.value()


class ConnectionLimitLogFilter(logging.Filter):
    """Count Uvicorn connection-limit rejections from its fixed warning.

    Uvicorn answers ``503`` itself when ``--limit-concurrency`` is reached,
    before the ASGI app runs, and reports it only as a fixed warning on the
    ``uvicorn.error`` logger. The reference launcher attaches this filter to
    that logger. It counts the warning and passes every record through
    unchanged; the warning carries no client address or request content.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Count a connection-limit warning and keep every record."""

        if (
            record.levelno == logging.WARNING
            and record.msg == _CONNECTION_LIMIT_WARNING
        ):
            _CONNECTION_REJECTIONS.increment()
        return True


def _env_seconds(source: Mapping[str, str], name: str, default: float) -> float:
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or not 0 < value <= _MAX_BODY_SECONDS:
        raise ValueError(
            f"{name} must be a number of seconds above 0 and at most "
            f"{_MAX_BODY_SECONDS:g}"
        )
    return value


def _env_int(
    source: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = source.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        value = minimum - 1
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _limit_response(status_code: int, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "details": None,
                "message": "Request rejected by an operational limit",
            }
        },
    )


__all__ = [
    "BODY_TIMEOUT_DEADLINE",
    "BODY_TIMEOUT_MIN_RATE",
    "BODY_TIMEOUT_REASONS",
    "BoundedRequestBodyMiddleware",
    "ConnectionLimitLogFilter",
    "LIMIT_CONCURRENCY_ENV_VAR",
    "MAX_REQUEST_BYTES_ENV_VAR",
    "REQUEST_BODY_GRACE_ENV_VAR",
    "REQUEST_BODY_MIN_RATE_ENV_VAR",
    "REQUEST_BODY_TIMEOUT",
    "REQUEST_BODY_TIMEOUT_ENV_VAR",
    "REQUEST_SIZE_EXCEEDED",
    "REQUEST_SIZE_INVALID",
    "RequestBodyDeadline",
    "RequestBodyReadLimits",
    "RequestBodyTimeout",
    "RequestBodyTimeoutRecorder",
    "ServerConnectionLimits",
    "TIMEOUT_GRACEFUL_SHUTDOWN_ENV_VAR",
    "TIMEOUT_KEEP_ALIVE_ENV_VAR",
    "body_limit_response",
    "connection_rejection_count",
    "request_body_read_limits_from_env",
    "request_limits_from_env",
    "server_connection_limits_from_env",
]
