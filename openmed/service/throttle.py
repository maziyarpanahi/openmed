"""In-process throttling primitives for the OpenMed REST service."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass
from typing import Optional

from fastapi import Request
from starlette.responses import Response

from .batcher import BackpressureError
from .runtime import ServiceThrottleConfig

PROBE_PATHS = frozenset({"/health", "/livez", "/readyz"})
GLOBAL_THROTTLE_KEY = "global"
ErrorResponseFactory = Callable[..., Response]
CallNext = Callable[[Request], Awaitable[Response]]
BackpressureCheck = Callable[[Request], Awaitable[Optional[BackpressureError]]]


@dataclass
class RateLimitDecision:
    """Result of one token-bucket admission attempt."""

    allowed: bool
    retry_after_seconds: float = 0.0


@dataclass
class _BucketState:
    tokens: float
    updated_at: float


class TokenBucketRateLimiter:
    """Async-safe in-memory token bucket keyed by client identity."""

    def __init__(
        self,
        *,
        requests_per_second: float,
        burst_size: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_second < 0:
            raise ValueError("requests_per_second must be greater than or equal to 0")
        if burst_size < 0:
            raise ValueError("burst_size must be greater than or equal to 0")

        self.requests_per_second = float(requests_per_second)
        self.burst_size = int(burst_size)
        self._clock = clock
        self._lock = asyncio.Lock()
        self._buckets: dict[str, _BucketState] = {}

    @property
    def enabled(self) -> bool:
        """Return whether rate limiting is active."""
        return self.requests_per_second > 0 and self.burst_size > 0

    async def consume(self, key: str = GLOBAL_THROTTLE_KEY) -> RateLimitDecision:
        """Consume one token for *key*, returning when to retry if rejected."""
        if not self.enabled:
            return RateLimitDecision(allowed=True)

        async with self._lock:
            now = self._clock()
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _BucketState(tokens=float(self.burst_size), updated_at=now)
                self._buckets[key] = bucket

            elapsed = max(0.0, now - bucket.updated_at)
            bucket.tokens = min(
                float(self.burst_size),
                bucket.tokens + (elapsed * self.requests_per_second),
            )
            bucket.updated_at = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return RateLimitDecision(allowed=True)

            retry_after = (1.0 - bucket.tokens) / self.requests_per_second
            return RateLimitDecision(
                allowed=False,
                retry_after_seconds=max(0.0, retry_after),
            )


class _ConcurrencyPermit:
    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        self._semaphore = semaphore
        self._released = False

    def release(self) -> None:
        """Release the held semaphore once."""
        if self._released:
            return
        self._released = True
        self._semaphore.release()


class ConcurrencyGate:
    """Bound concurrent in-flight work with per-key semaphores."""

    def __init__(self, *, max_concurrency: int, wait_seconds: float) -> None:
        if max_concurrency < 0:
            raise ValueError("max_concurrency must be greater than or equal to 0")
        if wait_seconds < 0:
            raise ValueError("wait_seconds must be greater than or equal to 0")

        self.max_concurrency = int(max_concurrency)
        self.wait_seconds = float(wait_seconds)
        self._lock = asyncio.Lock()
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    @property
    def enabled(self) -> bool:
        """Return whether concurrency limiting is active."""
        return self.max_concurrency > 0

    async def acquire(
        self,
        key: str = GLOBAL_THROTTLE_KEY,
    ) -> Optional[_ConcurrencyPermit]:
        """Acquire one concurrency slot, or return None after the wait bound."""
        if not self.enabled:
            return None

        semaphore = await self._semaphore_for_key(key)
        acquired = await self._acquire_with_bound(semaphore)
        if not acquired:
            return None
        return _ConcurrencyPermit(semaphore)

    async def _semaphore_for_key(self, key: str) -> asyncio.Semaphore:
        async with self._lock:
            semaphore = self._semaphores.get(key)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self.max_concurrency)
                self._semaphores[key] = semaphore
            return semaphore

    async def _acquire_with_bound(self, semaphore: asyncio.Semaphore) -> bool:
        if self.wait_seconds <= 0:
            if semaphore.locked():
                return False
            await semaphore.acquire()
            return True

        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=self.wait_seconds)
        except asyncio.TimeoutError:
            return False
        return True


class ServiceThrottle:
    """FastAPI middleware dispatcher for service request throttling."""

    def __init__(
        self,
        config: ServiceThrottleConfig,
        *,
        error_response: ErrorResponseFactory,
        backpressure_check: Optional[BackpressureCheck] = None,
        limited_paths: Optional[Collection[str]] = None,
        exempt_paths: Collection[str] = PROBE_PATHS,
    ) -> None:
        self.config = config
        self.rate_limiter = TokenBucketRateLimiter(
            requests_per_second=config.rate_limit_rps,
            burst_size=config.rate_limit_burst,
        )
        self.concurrency_gate = ConcurrencyGate(
            max_concurrency=config.max_concurrency,
            wait_seconds=config.concurrency_wait_seconds,
        )
        self._error_response = error_response
        self._backpressure_check = backpressure_check
        self._limited_paths = frozenset(limited_paths) if limited_paths else None
        self._exempt_paths = frozenset(exempt_paths)

    @property
    def enabled(self) -> bool:
        """Return whether any throttle gate is active."""
        return self.config.enabled or self._backpressure_check is not None

    async def dispatch(self, request: Request, call_next: CallNext) -> Response:
        """Apply rate and concurrency gates around one HTTP request."""
        if not self.enabled or not self._should_throttle(request.url.path):
            return await call_next(request)

        key = client_identity(request, self.config.key_by)
        decision = await self.rate_limiter.consume(key)
        if not decision.allowed:
            response = self._error_response(
                429,
                "rate_limited",
                "Request rate limit exceeded",
                details={
                    "retry_after_seconds": decision.retry_after_seconds,
                    "key": self.config.key_by,
                },
            )
            response.headers["Retry-After"] = format_retry_after(
                decision.retry_after_seconds
            )
            return response

        if self._backpressure_check is not None:
            try:
                backpressure = await self._backpressure_check(request)
            except ValueError as exc:
                return self._error_response(
                    400,
                    "bad_request",
                    str(exc),
                    details={"reason": str(exc)},
                )
            if backpressure is not None:
                response = self._error_response(
                    503,
                    "backpressure",
                    str(backpressure),
                    details=backpressure.to_details(),
                )
                response.headers["Retry-After"] = format_retry_after(
                    backpressure.retry_after_seconds
                )
                return response

        permit = await self.concurrency_gate.acquire(key)
        if self.concurrency_gate.enabled and permit is None:
            return self._error_response(
                503,
                "service_busy",
                "Service is busy; retry later",
                details={
                    "max_concurrency": self.config.max_concurrency,
                    "wait_seconds": self.config.concurrency_wait_seconds,
                    "key": self.config.key_by,
                },
            )

        try:
            return await call_next(request)
        finally:
            if permit is not None:
                permit.release()

    def _should_throttle(self, path: str) -> bool:
        if path in self._exempt_paths:
            return False
        return self._limited_paths is None or path in self._limited_paths


def client_identity(request: Request, key_by: str) -> str:
    """Return the throttle bucket key for one request."""
    if key_by == "global":
        return GLOBAL_THROTTLE_KEY

    if key_by == "x-forwarded-for":
        forwarded_for = request.headers.get("x-forwarded-for", "")
        first_hop = forwarded_for.split(",", 1)[0].strip()
        if first_hop:
            return first_hop

    if key_by in {"x-forwarded-for", "peer"}:
        peer = peer_identity(request)
        if peer:
            return peer

    return "unknown"


def peer_identity(request: Request) -> Optional[str]:
    """Return the direct peer host for one request when available."""
    client = request.client
    if client is not None and client.host:
        return client.host
    return None


def format_retry_after(seconds: float) -> str:
    """Format an HTTP Retry-After delta-seconds header."""
    return str(max(1, int(math.ceil(max(0.0, seconds)))))
