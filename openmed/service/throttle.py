"""In-process throttling primitives for the OpenMed REST service."""

from __future__ import annotations

import asyncio
import ipaddress
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Collection
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import Request
from starlette.responses import Response

from .batcher import BackpressureError
from .runtime import (
    DEFAULT_SERVICE_THROTTLE_MAX_TRACKED_KEYS,
    ServiceThrottleConfig,
    TrustedProxyConfig,
)

PROBE_PATHS = frozenset({"/health", "/livez", "/readyz"})
GLOBAL_THROTTLE_KEY = "global"
# Idle buckets examined per admission once the key bound is reached.
_EVICTION_SCAN_LIMIT = 32
IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
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
    """Async-safe in-memory token bucket keyed by client identity.

    At most ``max_keys`` identities are tracked. A bucket is evicted only once
    it has refilled to the full burst, when forgetting it cannot change any
    admission decision; a depleted bucket is never reset. When the bound is
    reached and no tracked bucket is idle, new identities share one overflow
    bucket until capacity frees up.
    """

    def __init__(
        self,
        *,
        requests_per_second: float,
        burst_size: int,
        max_keys: int = DEFAULT_SERVICE_THROTTLE_MAX_TRACKED_KEYS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_second < 0:
            raise ValueError("requests_per_second must be greater than or equal to 0")
        if burst_size < 0:
            raise ValueError("burst_size must be greater than or equal to 0")
        if max_keys < 1:
            raise ValueError("max_keys must be greater than or equal to 1")

        self.requests_per_second = float(requests_per_second)
        self.burst_size = int(burst_size)
        self.max_keys = int(max_keys)
        self._clock = clock
        self._lock = asyncio.Lock()
        # Least recently updated first, so refilled buckets gather at the front.
        self._buckets: OrderedDict[str, _BucketState] = OrderedDict()
        self._overflow: Optional[_BucketState] = None

    @property
    def tracked_key_count(self) -> int:
        """Return how many per-identity buckets are currently retained."""
        return len(self._buckets)

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
            bucket = self._bucket_for(key, now)
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

    def _bucket_for(self, key: str, now: float) -> _BucketState:
        self._evict_idle_front(now)
        bucket = self._buckets.get(key)
        if bucket is not None:
            self._buckets.move_to_end(key)
            return bucket

        if len(self._buckets) >= self.max_keys:
            self._evict_idle_scan(now)
        if len(self._buckets) >= self.max_keys:
            if self._overflow is None:
                self._overflow = self._new_bucket(now)
            return self._overflow

        bucket = self._new_bucket(now)
        self._buckets[key] = bucket
        return bucket

    def _new_bucket(self, now: float) -> _BucketState:
        return _BucketState(tokens=float(self.burst_size), updated_at=now)

    def _is_idle(self, bucket: _BucketState, now: float) -> bool:
        elapsed = max(0.0, now - bucket.updated_at)
        refilled = bucket.tokens + (elapsed * self.requests_per_second)
        return refilled >= float(self.burst_size)

    def _evict_idle_front(self, now: float) -> None:
        while self._buckets:
            key, bucket = next(iter(self._buckets.items()))
            if not self._is_idle(bucket, now):
                return
            del self._buckets[key]

    def _evict_idle_scan(self, now: float) -> None:
        candidates = []
        for index, (key, bucket) in enumerate(self._buckets.items()):
            if index >= _EVICTION_SCAN_LIMIT:
                break
            if self._is_idle(bucket, now):
                candidates.append(key)
        for key in candidates:
            del self._buckets[key]


@dataclass
class _GateSlot:
    semaphore: asyncio.Semaphore
    # Permit holders plus waiters; the slot is dropped when this reaches zero.
    users: int = 0


class _ConcurrencyPermit:
    def __init__(self, gate: ConcurrencyGate, key: Optional[str], slot: _GateSlot):
        self._gate = gate
        self._key = key
        self._slot = slot
        self._released = False

    def release(self) -> None:
        """Release the held semaphore once."""
        if self._released:
            return
        self._released = True
        self._slot.semaphore.release()
        self._gate._leave(self._key, self._slot)


class ConcurrencyGate:
    """Bound concurrent in-flight work with per-key semaphores.

    A key's semaphore is retained only while a permit is held or a request is
    waiting for one, so idle identities never accumulate. At most
    ``max_keys`` identities are tracked at once; beyond that, new identities
    share one overflow semaphore. Slot bookkeeping runs on the event loop
    without awaiting, so it needs no lock.
    """

    def __init__(
        self,
        *,
        max_concurrency: int,
        wait_seconds: float,
        max_keys: int = DEFAULT_SERVICE_THROTTLE_MAX_TRACKED_KEYS,
    ) -> None:
        if max_concurrency < 0:
            raise ValueError("max_concurrency must be greater than or equal to 0")
        if wait_seconds < 0:
            raise ValueError("wait_seconds must be greater than or equal to 0")
        if max_keys < 1:
            raise ValueError("max_keys must be greater than or equal to 1")

        self.max_concurrency = int(max_concurrency)
        self.wait_seconds = float(wait_seconds)
        self.max_keys = int(max_keys)
        self._slots: dict[str, _GateSlot] = {}
        self._overflow: Optional[_GateSlot] = None

    @property
    def enabled(self) -> bool:
        """Return whether concurrency limiting is active."""
        return self.max_concurrency > 0

    @property
    def tracked_key_count(self) -> int:
        """Return how many per-identity semaphores are currently retained."""
        return len(self._slots)

    async def acquire(
        self,
        key: str = GLOBAL_THROTTLE_KEY,
    ) -> Optional[_ConcurrencyPermit]:
        """Acquire one concurrency slot, or return None after the wait bound."""
        if not self.enabled:
            return None

        slot_key, slot = self._enter(key)
        acquired = False
        try:
            acquired = await self._acquire_with_bound(slot.semaphore)
        finally:
            if not acquired:
                self._leave(slot_key, slot)
        if not acquired:
            return None
        return _ConcurrencyPermit(self, slot_key, slot)

    def _enter(self, key: str) -> tuple[Optional[str], _GateSlot]:
        slot = self._slots.get(key)
        slot_key: Optional[str] = key
        if slot is None:
            if len(self._slots) < self.max_keys:
                slot = _GateSlot(asyncio.Semaphore(self.max_concurrency))
                self._slots[key] = slot
            else:
                slot_key = None
                if self._overflow is None:
                    self._overflow = _GateSlot(asyncio.Semaphore(self.max_concurrency))
                slot = self._overflow
        slot.users += 1
        return slot_key, slot

    def _leave(self, key: Optional[str], slot: _GateSlot) -> None:
        slot.users -= 1
        if slot.users > 0:
            return
        if key is None:
            if self._overflow is slot:
                self._overflow = None
        elif self._slots.get(key) is slot:
            del self._slots[key]

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


class ServiceBusyError(RuntimeError):
    """Reject additional model work when the request's gate is full."""

    def __init__(self, gate: ConcurrencyGate, key_by: str) -> None:
        super().__init__("Service is busy; retry later")
        self.max_concurrency = gate.max_concurrency
        self.wait_seconds = gate.wait_seconds
        self.key_by = key_by


class RequestWork:
    """Keep worker permits and drain tokens until accepted work actually settles.

    All methods and completion callbacks run on the request's event loop.
    The first threadpool worker inherits the request's admission permit;
    parallel workers acquire their own. Each releases its permit on completion,
    allowing waiting fields to progress even before the response finishes.
    Batch submissions retain admission until removal or dispatch completion.
    """

    def __init__(
        self,
        permit: Optional[_ConcurrencyPermit],
        gate: ConcurrencyGate,
        key: str,
        key_by: str,
        metrics: Optional[Any],
    ) -> None:
        self._permit = permit
        self._gate = gate
        self._key = key
        self._key_by = key_by
        self._metrics = metrics
        self._tasks: set[asyncio.Future[Any]] = set()
        self._callbacks: list[Callable[[], None]] = []
        self._primary_claimed = False
        self._closed = False
        self._finished = False
        self._orphaned = False

    async def acquire_model_slot(self) -> Optional[Callable[[], None]]:
        """Use the request's slot, or admit an additional parallel resolver."""
        if self._permit is None:
            return None
        if not self._primary_claimed:
            self._primary_claimed = True
            return self._permit.release
        permit = await self._gate.acquire(self._key)
        if permit is None:
            raise ServiceBusyError(self._gate, self._key_by)
        return permit.release

    def track(self, task: asyncio.Future[Any]) -> None:
        """Retain ownership of one asynchronous operation through completion."""
        self._tasks.add(task)

        def completed(future: asyncio.Future[Any]) -> None:
            self._tasks.discard(future)
            self._finish_if_idle()

        task.add_done_callback(completed)

    def on_finished(self, callback: Callable[[], None]) -> None:
        """Release a lifecycle token only after the response and work finish."""
        if self._finished:
            callback()
        else:
            self._callbacks.append(callback)

    def close(self) -> None:
        """Detach the HTTP wait while retaining any outstanding work."""
        self._closed = True
        if any(not task.done() for task in self._tasks):
            self._orphaned = True
            if self._metrics is not None:
                self._metrics.orphaned_work_started()
        self._finish_if_idle()

    def _finish_if_idle(self) -> None:
        if not self._closed or self._tasks or self._finished:
            return
        self._finished = True
        if self._permit is not None:
            # A worker may already have released this permit; release() is idempotent.
            self._permit.release()
        if self._orphaned and self._metrics is not None:
            self._metrics.orphaned_work_finished()
        callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            callback()


_REQUEST_WORK: ContextVar[Optional[RequestWork]] = ContextVar(
    "openmed_request_work", default=None
)


def current_request_work() -> Optional[RequestWork]:
    """Return admission ownership inherited by this request's async work."""
    return _REQUEST_WORK.get()


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
            max_keys=config.max_tracked_keys,
        )
        self.concurrency_gate = ConcurrencyGate(
            max_concurrency=config.max_concurrency,
            wait_seconds=config.concurrency_wait_seconds,
            max_keys=config.max_tracked_keys,
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
        from .auth import request_needs_admission

        should_throttle = (
            request_needs_admission(request)
            if self._limited_paths is not None
            else self._should_throttle(request.url.path)
        )
        # Drain and orphaned-work accounting also apply with no throttle limits.
        if not should_throttle:
            return await call_next(request)

        key = client_identity(request, self.config.key_by, self.config.proxies)
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

        work = RequestWork(
            permit,
            self.concurrency_gate,
            key,
            self.config.key_by,
            getattr(request.app.state, "metrics", None),
        )
        token = _REQUEST_WORK.set(work)
        try:
            return await call_next(request)
        finally:
            _REQUEST_WORK.reset(token)
            work.close()

    def _should_throttle(self, path: str) -> bool:
        if path in self._exempt_paths:
            return False
        return self._limited_paths is None or path in self._limited_paths


def client_identity(
    request: Request,
    key_by: str,
    proxies: Optional[TrustedProxyConfig] = None,
) -> str:
    """Return the throttle bucket key for one request.

    In ``x-forwarded-for`` mode the header is honored only when the direct
    peer is a trusted proxy. Entries are read from right to left, skipping at
    most ``proxies.hops`` trusted proxies (the direct peer counts as the
    first), and the first untrusted address identifies the client. Entries a
    client prepends are therefore never reached. Without trusted proxies the
    direct peer is used, exactly as in ``peer`` mode.

    Args:
        request: Incoming HTTP request.
        key_by: ``global``, ``peer``, or ``x-forwarded-for``.
        proxies: Trusted reverse proxies; ``None`` trusts no proxy.

    Returns:
        A bucket key. Forwarded addresses are returned in canonical form; the
        key must never be logged or exposed in responses or metrics.
    """
    if key_by == "global":
        return GLOBAL_THROTTLE_KEY

    if key_by == "x-forwarded-for" and proxies is not None and proxies.enabled:
        forwarded_client = _forwarded_client(request, proxies)
        if forwarded_client is not None:
            return forwarded_client

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


def _forwarded_client(
    request: Request,
    proxies: TrustedProxyConfig,
) -> Optional[str]:
    peer = _parse_address(peer_identity(request) or "")
    if peer is None or not _is_trusted(peer, proxies):
        return None

    header_values = request.headers.getlist("x-forwarded-for")
    # Only the rightmost ``hops`` entries can be examined; never split the rest.
    entries = ",".join(header_values).rsplit(",", proxies.hops)
    candidate = peer
    skipped = 1
    for raw_entry in reversed(entries):
        address = _parse_forwarded_address(raw_entry)
        if address is None:
            # A malformed entry cannot identify anyone; key by the last trusted
            # hop rather than by attacker-influenced text.
            break
        candidate = address
        if skipped >= proxies.hops or not _is_trusted(address, proxies):
            break
        skipped += 1
    return str(candidate)


def _is_trusted(address: IPAddress, proxies: TrustedProxyConfig) -> bool:
    return any(
        address.version == network.version and address in network
        for network in proxies.networks
    )


def _parse_forwarded_address(raw_value: str) -> Optional[IPAddress]:
    value = raw_value.strip()
    if value.startswith("["):
        host, closed, rest = value[1:].partition("]")
        if not closed or (rest and not _is_port_suffix(rest)):
            return None
        return _parse_address(host)
    if value.count(":") == 1:
        host, _, port = value.partition(":")
        if not port.isdigit():
            return None
        value = host
    return _parse_address(value)


def _is_port_suffix(value: str) -> bool:
    return value.startswith(":") and value[1:].isdigit()


def _parse_address(value: str) -> Optional[IPAddress]:
    try:
        address = ipaddress.ip_address(value.strip())
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return address.ipv4_mapped
        if address.scope_id is not None:
            return ipaddress.IPv6Address(str(address).split("%", 1)[0])
    return address


def format_retry_after(seconds: float) -> str:
    """Format an HTTP Retry-After delta-seconds header."""
    return str(max(1, int(math.ceil(max(0.0, seconds)))))
