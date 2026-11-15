"""Fair, bounded request admission shared by every service transport.

The controller bounds in-flight and waiting work at three levels: one global
ceiling, one ceiling per authenticated principal and, optionally, one ceiling
per namespace. Requests that cannot start immediately wait in a bounded
per-principal FIFO queue. Freed capacity is handed out by round-robin rotation
over principals with waiting work, so one principal's backlog cannot starve
another principal's queued request.

The controller is transport-neutral. REST requests are admitted by
:class:`HttpAdmission`, GraphQL operations by the schema extension in
:mod:`openmed.service.graphql_schema`, and synchronous transports such as a
gRPC server interceptor call :meth:`AdmissionController.admit_blocking` with an
:class:`AdmissionTicket` built from their own authenticated principal. When a
:class:`~openmed.service.runtime.ServiceRuntime` is shared, every transport
draws on the same budget.

Principal and namespace identities are reduced to process-local keyed digests
before they are stored. Rejections, metrics and snapshots contain only static
reason codes and aggregate counts, never principal identifiers or other
tenants' load.

This module owns request admission only. Async job admission, model residency
limits and the dynamic-batching queues remain separate controls.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

ADMISSION_MAX_CONCURRENCY_ENV_VAR = "OPENMED_SERVICE_ADMISSION_MAX_CONCURRENCY"
ADMISSION_MAX_QUEUE_ENV_VAR = "OPENMED_SERVICE_ADMISSION_MAX_QUEUE"
ADMISSION_PRINCIPAL_MAX_CONCURRENCY_ENV_VAR = (
    "OPENMED_SERVICE_ADMISSION_PRINCIPAL_MAX_CONCURRENCY"
)
ADMISSION_PRINCIPAL_MAX_QUEUE_ENV_VAR = "OPENMED_SERVICE_ADMISSION_PRINCIPAL_MAX_QUEUE"
ADMISSION_NAMESPACE_MAX_CONCURRENCY_ENV_VAR = (
    "OPENMED_SERVICE_ADMISSION_NAMESPACE_MAX_CONCURRENCY"
)
ADMISSION_NAMESPACE_MAX_QUEUE_ENV_VAR = "OPENMED_SERVICE_ADMISSION_NAMESPACE_MAX_QUEUE"
ADMISSION_NAMESPACE_CLAIM_ENV_VAR = "OPENMED_SERVICE_ADMISSION_NAMESPACE_CLAIM"
ADMISSION_QUEUE_TIMEOUT_ENV_VAR = "OPENMED_SERVICE_ADMISSION_QUEUE_TIMEOUT_SECONDS"
ADMISSION_RETRY_AFTER_ENV_VAR = "OPENMED_SERVICE_ADMISSION_RETRY_AFTER_SECONDS"

DEFAULT_ADMISSION_QUEUE_TIMEOUT_SECONDS = 5.0
DEFAULT_ADMISSION_RETRY_AFTER_SECONDS = 1.0
MAX_NAMESPACE_VALUE_LENGTH = 128

ANONYMOUS_PRINCIPAL = "anonymous"
AUTH_SCOPE_KEY = "openmed.auth"

REASON_PRINCIPAL_LIMIT = "principal_limit"
REASON_NAMESPACE_LIMIT = "namespace_limit"
REASON_GLOBAL_LIMIT = "global_limit"
ADMISSION_REASONS = (
    REASON_PRINCIPAL_LIMIT,
    REASON_NAMESPACE_LIMIT,
    REASON_GLOBAL_LIMIT,
)

TRANSPORT_REST = "rest"
TRANSPORT_GRAPHQL = "graphql"
TRANSPORT_GRPC = "grpc"
TRANSPORT_OTHER = "other"
ADMISSION_TRANSPORTS = frozenset({TRANSPORT_REST, TRANSPORT_GRAPHQL, TRANSPORT_GRPC})

# GraphQL documents can select several model-backed fields in one HTTP request,
# so the GraphQL schema admits each field at execution time instead.
EXECUTION_ADMITTED_PATHS = frozenset({"/graphql"})
_ADMITTED_ROUTE_CLASSES = frozenset({"model", "heavy"})
_BLOCKING_CANCEL_POLL_SECONDS = 0.05

# A fresh key per process keeps stored bucket keys from being stable,
# reusable identifiers. Transports in one process share the same key.
_IDENTITY_KEY = secrets.token_bytes(32)

_WAITING = 0
_GRANTED = 1
_ABANDONED = 2


@dataclass(frozen=True)
class AdmissionConfig:
    """Ceilings for fair request admission.

    A concurrency ceiling of ``0`` leaves that level unlimited; that level's
    queue setting is then ignored. With a ceiling set, the matching queue
    setting bounds how many requests may wait at that level, and ``0`` means
    requests that cannot start are rejected at once. A request may wait only
    when it fits every limited level's queue.

    Attributes:
        max_concurrency: Global in-flight ceiling across all principals.
        max_queue: Global ceiling on waiting requests.
        principal_max_concurrency: In-flight ceiling for each principal.
            Anonymous callers share one pooled principal.
        principal_max_queue: Waiting ceiling for each principal.
        namespace_max_concurrency: In-flight ceiling for each namespace.
        namespace_max_queue: Waiting ceiling for each namespace.
        namespace_claim: JWT claim that names a principal's namespace. Unset
            disables namespace accounting.
        queue_timeout_seconds: Longest a request waits before rejection.
        retry_after_seconds: Constant ``Retry-After`` hint on rejections. It
            never reflects current load.
    """

    max_concurrency: int = 0
    max_queue: int = 0
    principal_max_concurrency: int = 0
    principal_max_queue: int = 0
    namespace_max_concurrency: int = 0
    namespace_max_queue: int = 0
    namespace_claim: Optional[str] = None
    queue_timeout_seconds: float = DEFAULT_ADMISSION_QUEUE_TIMEOUT_SECONDS
    retry_after_seconds: float = DEFAULT_ADMISSION_RETRY_AFTER_SECONDS

    def __post_init__(self) -> None:
        for name in (
            "max_concurrency",
            "max_queue",
            "principal_max_concurrency",
            "principal_max_queue",
            "namespace_max_concurrency",
            "namespace_max_queue",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Admission {name} must be a non-negative integer")
        for name in ("queue_timeout_seconds", "retry_after_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Admission {name} must be a positive number")
            if not value > 0 or value == float("inf"):
                raise ValueError(f"Admission {name} must be a positive number")

    @property
    def enabled(self) -> bool:
        """Return whether any admission ceiling is configured."""
        return (
            self.max_concurrency > 0
            or self.principal_max_concurrency > 0
            or self.namespace_max_concurrency > 0
        )

    @property
    def namespaces_enabled(self) -> bool:
        """Return whether namespace ceilings apply."""
        return bool(self.namespace_claim) and self.namespace_max_concurrency > 0


@dataclass(frozen=True)
class AdmissionTicket:
    """One request's admission identity, independent of transport.

    Attributes:
        principal: Opaque bucket key. Use :meth:`for_principal` rather than
            passing a raw subject.
        namespace: Opaque namespace key, or ``None`` when not namespaced.
        transport: Static transport label used only for aggregate metrics.
    """

    principal: str = ANONYMOUS_PRINCIPAL
    namespace: Optional[str] = None
    transport: str = TRANSPORT_OTHER

    @classmethod
    def for_principal(
        cls,
        principal: object | None,
        *,
        transport: str,
        namespace_claim: Optional[str] = None,
    ) -> AdmissionTicket:
        """Build a ticket from an authenticated principal.

        Args:
            principal: Object with a ``subject`` string and optional ``claims``
                mapping, such as :class:`~openmed.service.auth.AuthPrincipal`.
                ``None`` or a principal without a subject joins the shared
                anonymous bucket.
            transport: ``rest``, ``graphql`` or ``grpc``; other values are
                reported as ``other``.
            namespace_claim: Claim that names the principal's namespace.

        Returns:
            A ticket whose keys are process-local digests.
        """
        label = transport if transport in ADMISSION_TRANSPORTS else TRANSPORT_OTHER
        subject = getattr(principal, "subject", None)
        if not isinstance(subject, str) or not subject:
            return cls(ANONYMOUS_PRINCIPAL, None, label)
        namespace = None
        if namespace_claim:
            claims = getattr(principal, "claims", None)
            value = claims.get(namespace_claim) if isinstance(claims, Mapping) else None
            if isinstance(value, str) and 0 < len(value) <= MAX_NAMESPACE_VALUE_LENGTH:
                namespace = _identity_digest("n", value)
        return cls(_identity_digest("p", subject), namespace, label)


@dataclass(frozen=True)
class AdmissionStats:
    """Aggregate admission state without identifiers."""

    in_flight: int
    queued: int
    active_principals: int


class AdmissionRejected(RuntimeError):
    """A request could not be admitted within its bounds.

    Attributes:
        reason: One of :data:`ADMISSION_REASONS`.
        retry_after_seconds: Constant retry hint from configuration.
    """

    status_code = 503
    error_code = "admission_saturated"
    grpc_status = "UNAVAILABLE"

    def __init__(self, reason: str, *, retry_after_seconds: float) -> None:
        self.reason = reason
        self.retry_after_seconds = float(retry_after_seconds)
        super().__init__("Request admission limit reached; retry later")

    @property
    def retry_after_header(self) -> str:
        """Return the ``Retry-After`` delta-seconds value."""
        from .throttle import format_retry_after

        return format_retry_after(self.retry_after_seconds)

    def to_details(self) -> dict[str, Any]:
        """Return value-free API error details."""
        return {
            "reason": self.reason,
            "retry_after_seconds": self.retry_after_seconds,
        }

    def grpc_trailing_metadata(self) -> tuple[tuple[str, str], ...]:
        """Return value-free trailing metadata for a gRPC rejection."""
        return (
            ("retry-after", self.retry_after_header),
            ("openmed-admission-reason", self.reason),
        )


class PrincipalAdmissionLimited(AdmissionRejected):
    """The caller's principal or namespace is over its configured share."""

    status_code = 429
    error_code = "admission_limited"
    grpc_status = "RESOURCE_EXHAUSTED"


class AdmissionSaturated(AdmissionRejected):
    """The caller is within its share but shared capacity is exhausted."""

    status_code = 503
    error_code = "admission_saturated"
    grpc_status = "UNAVAILABLE"


class AdmissionCancelled(RuntimeError):
    """A blocking acquisition was abandoned because its caller cancelled."""


class AdmissionPermit:
    """Ownership of one admitted unit of work.

    :meth:`release` is idempotent: the first call returns the unit to the
    controller and later calls do nothing.
    """

    __slots__ = ("_controller", "_principal", "_namespace", "_released")

    def __init__(
        self,
        controller: Optional[AdmissionController],
        principal: str,
        namespace: Optional[str],
    ) -> None:
        self._controller = controller
        self._principal = principal
        self._namespace = namespace
        self._released = False

    @property
    def released(self) -> bool:
        """Return whether this permit has been released."""
        return self._released

    def release(self) -> bool:
        """Release the permit once.

        Returns:
            ``True`` for the call that released it, otherwise ``False``.
        """
        controller = self._controller
        if controller is None:
            if self._released:
                return False
            self._released = True
            return True
        return controller._release(self)


class _Bucket:
    __slots__ = ("key", "in_flight", "waiters")

    def __init__(self, key: str) -> None:
        self.key = key
        self.in_flight = 0
        self.waiters: deque[_Waiter] = deque()


class _NamespaceLoad:
    __slots__ = ("in_flight", "waiting")

    def __init__(self) -> None:
        self.in_flight = 0
        self.waiting = 0


class _Waiter:
    __slots__ = ("bucket", "namespace", "state", "permit", "wake")

    def __init__(
        self,
        bucket: _Bucket,
        namespace: Optional[str],
        wake: Callable[[], bool],
    ) -> None:
        self.bucket = bucket
        self.namespace = namespace
        self.state = _WAITING
        self.permit: Optional[AdmissionPermit] = None
        self.wake = wake


class AdmissionController:
    """Thread-safe fair admission accounting shared across transports.

    Every state transition happens under one lock, and a waiter's state decides
    who owns a granted unit, so a cancelled, expired or failed request releases
    each permit exactly once. Async callers wait on their own event loop;
    blocking callers wait on a thread event.

    Starvation bound: when shared capacity frees, it goes to the next principal
    in rotation whose head request fits its own ceilings. A principal with
    waiting work is therefore granted within ``k - 1`` grants to others, where
    ``k`` is the number of principals waiting, unless it is held back by its
    own principal or namespace ceiling. ``queue_timeout_seconds`` bounds every
    wait in time.
    """

    def __init__(
        self,
        config: Optional[AdmissionConfig] = None,
        *,
        metrics: Any = None,
    ) -> None:
        """Create a controller.

        Args:
            config: Admission ceilings; defaults to a disabled configuration.
            metrics: Optional registry with ``record_principal_admission_state``
                and ``record_principal_admission_rejection`` hooks.
        """
        self._config = config if config is not None else AdmissionConfig()
        self._metrics = metrics
        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}
        self._namespaces: dict[str, _NamespaceLoad] = {}
        self._rotation: deque[str] = deque()
        self._in_flight = 0
        self._waiting = 0
        if self.enabled:
            self._publish_state()

    @property
    def config(self) -> AdmissionConfig:
        """Return the active configuration."""
        return self._config

    @property
    def enabled(self) -> bool:
        """Return whether any ceiling is enforced."""
        return self._config.enabled

    def ticket(self, principal: object | None, *, transport: str) -> AdmissionTicket:
        """Build a ticket for ``principal`` using this controller's claim."""
        return AdmissionTicket.for_principal(
            principal,
            transport=transport,
            namespace_claim=self._config.namespace_claim,
        )

    def snapshot(self) -> AdmissionStats:
        """Return aggregate in-flight, waiting and active-principal counts."""
        with self._lock:
            return AdmissionStats(
                in_flight=self._in_flight,
                queued=self._waiting,
                active_principals=len(self._buckets),
            )

    async def acquire(
        self,
        ticket: AdmissionTicket,
        *,
        timeout: Optional[float] = None,
    ) -> AdmissionPermit:
        """Admit one unit of work, waiting fairly when necessary.

        Args:
            ticket: The caller's admission identity.
            timeout: Longest wait in seconds; defaults to the configured queue
                timeout.

        Returns:
            A permit to release when the work has settled.

        Raises:
            PrincipalAdmissionLimited: The caller is over its own share.
            AdmissionSaturated: Shared capacity is exhausted.
            asyncio.CancelledError: The waiting task was cancelled; any unit
                granted in the meantime has been released.
        """
        if not self.enabled:
            return AdmissionPermit(None, ticket.principal, ticket.namespace)
        loop = asyncio.get_running_loop()
        signal: asyncio.Future[None] = loop.create_future()

        def wake() -> bool:
            try:
                loop.call_soon_threadsafe(_resolve_signal, signal)
            except RuntimeError:
                return False
            return True

        outcome = self._arrive(ticket, wake)
        if isinstance(outcome, AdmissionPermit):
            return outcome
        wait = self._config.queue_timeout_seconds if timeout is None else timeout
        try:
            # ``asyncio.wait`` neither cancels the signal on timeout nor, unlike
            # ``wait_for`` on Python 3.11, swallows a cancellation that races
            # with the grant; the waiter state below decides the outcome.
            await asyncio.wait((signal,), timeout=max(float(wait), 0.0))
        except BaseException:
            self._abandon(outcome)
            raise
        return self._settle(outcome, ticket)

    def acquire_blocking(
        self,
        ticket: AdmissionTicket,
        *,
        timeout: Optional[float] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> AdmissionPermit:
        """Admit one unit of work from a worker thread.

        This is the adapter for synchronous transports, for example a gRPC
        server interceptor that maps :class:`AdmissionRejected` to
        ``grpc_status`` with :meth:`AdmissionRejected.grpc_trailing_metadata`.

        Args:
            ticket: The caller's admission identity.
            timeout: Longest wait in seconds; defaults to the queue timeout.
            cancelled: Optional probe, such as ``lambda: not
                context.is_active()``, polled while waiting.

        Returns:
            A permit to release when the work has settled.

        Raises:
            PrincipalAdmissionLimited: The caller is over its own share.
            AdmissionSaturated: Shared capacity is exhausted.
            AdmissionCancelled: ``cancelled`` reported a cancelled caller.
        """
        if not self.enabled:
            return AdmissionPermit(None, ticket.principal, ticket.namespace)
        event = threading.Event()

        def wake() -> bool:
            event.set()
            return True

        outcome = self._arrive(ticket, wake)
        if isinstance(outcome, AdmissionPermit):
            return outcome
        wait = self._config.queue_timeout_seconds if timeout is None else timeout
        deadline = time.monotonic() + max(float(wait), 0.0)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if cancelled is None:
                    event.wait(remaining)
                    break
                if event.wait(min(remaining, _BLOCKING_CANCEL_POLL_SECONDS)):
                    break
                if cancelled():
                    raise AdmissionCancelled("Admission wait was cancelled")
        except BaseException:
            self._abandon(outcome)
            raise
        return self._settle(outcome, ticket)

    @asynccontextmanager
    async def admit(
        self,
        ticket: AdmissionTicket,
        *,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[AdmissionPermit]:
        """Hold one permit for the duration of an ``async with`` block."""
        permit = await self.acquire(ticket, timeout=timeout)
        try:
            yield permit
        finally:
            permit.release()

    @contextmanager
    def admit_blocking(
        self,
        ticket: AdmissionTicket,
        *,
        timeout: Optional[float] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> Iterator[AdmissionPermit]:
        """Hold one permit for the duration of a ``with`` block."""
        permit = self.acquire_blocking(ticket, timeout=timeout, cancelled=cancelled)
        try:
            yield permit
        finally:
            permit.release()

    def _arrive(
        self,
        ticket: AdmissionTicket,
        wake: Callable[[], bool],
    ) -> Union[AdmissionPermit, _Waiter]:
        namespace_key = ticket.namespace if self._config.namespaces_enabled else None
        with self._lock:
            bucket = self._buckets.get(ticket.principal)
            if bucket is None:
                bucket = _Bucket(ticket.principal)
                self._buckets[ticket.principal] = bucket
            namespace = self._namespace_load(namespace_key)
            if not bucket.waiters and self._has_room(bucket, namespace):
                permit = self._grant(bucket, namespace_key, namespace)
                self._publish_state()
                return permit
            reason = self._arrival_rejection(bucket, namespace)
            if reason is not None:
                self._forget_if_idle(bucket, namespace_key)
                self._record_rejection(reason, ticket.transport)
                raise self._rejection(reason)
            waiter = _Waiter(bucket, namespace_key, wake)
            bucket.waiters.append(waiter)
            if len(bucket.waiters) == 1:
                self._rotation.append(bucket.key)
            self._waiting += 1
            if namespace is not None:
                namespace.waiting += 1
            self._publish_state()
            return waiter

    def _settle(self, waiter: _Waiter, ticket: AdmissionTicket) -> AdmissionPermit:
        woken: list[_Waiter] = []
        with self._lock:
            if waiter.state == _GRANTED and waiter.permit is not None:
                return waiter.permit
            namespace = (
                self._namespaces.get(waiter.namespace)
                if waiter.namespace is not None
                else None
            )
            reason = self._blocking_reason(waiter.bucket, namespace)
            if waiter.state == _WAITING:
                woken = self._remove_waiter(waiter)
            self._record_rejection(reason, ticket.transport)
            self._publish_state()
        self._wake_all(woken)
        raise self._rejection(reason)

    def _abandon(self, waiter: _Waiter) -> None:
        granted: Optional[AdmissionPermit] = None
        woken: list[_Waiter] = []
        with self._lock:
            if waiter.state == _GRANTED:
                granted = waiter.permit
            elif waiter.state == _WAITING:
                woken = self._remove_waiter(waiter)
                self._publish_state()
        if granted is not None:
            granted.release()
        self._wake_all(woken)

    def _release(self, permit: AdmissionPermit) -> bool:
        with self._lock:
            if permit._released:
                return False
            permit._released = True
            self._in_flight -= 1
            bucket = self._buckets[permit._principal]
            bucket.in_flight -= 1
            if permit._namespace is not None:
                self._namespaces[permit._namespace].in_flight -= 1
            woken = self._dispatch()
            self._forget_if_idle(bucket, permit._namespace)
            self._publish_state()
        self._wake_all(woken)
        return True

    def _remove_waiter(self, waiter: _Waiter) -> list[_Waiter]:
        bucket = waiter.bucket
        bucket.waiters.remove(waiter)
        waiter.state = _ABANDONED
        self._waiting -= 1
        if waiter.namespace is not None:
            self._namespaces[waiter.namespace].waiting -= 1
        if not bucket.waiters:
            self._rotation.remove(bucket.key)
        # Removing a head may unblock the request queued behind it.
        woken = self._dispatch()
        self._forget_if_idle(bucket, waiter.namespace)
        return woken

    def _dispatch(self) -> list[_Waiter]:
        woken: list[_Waiter] = []
        max_concurrency = self._config.max_concurrency
        while self._rotation:
            if max_concurrency and self._in_flight >= max_concurrency:
                break
            granted = False
            for _ in range(len(self._rotation)):
                key = self._rotation.popleft()
                bucket = self._buckets[key]
                head = bucket.waiters[0]
                namespace = self._namespace_load(head.namespace)
                if self._has_room(bucket, namespace):
                    bucket.waiters.popleft()
                    self._waiting -= 1
                    if namespace is not None:
                        namespace.waiting -= 1
                    head.permit = self._grant(bucket, head.namespace, namespace)
                    head.state = _GRANTED
                    woken.append(head)
                    if bucket.waiters:
                        self._rotation.append(key)
                    granted = True
                    break
                self._rotation.append(key)
            if not granted:
                break
        return woken

    def _wake_all(self, woken: list[_Waiter]) -> None:
        for waiter in woken:
            if not waiter.wake() and waiter.permit is not None:
                # The waiter's event loop is gone; nobody can consume the unit.
                waiter.permit.release()

    def _grant(
        self,
        bucket: _Bucket,
        namespace_key: Optional[str],
        namespace: Optional[_NamespaceLoad],
    ) -> AdmissionPermit:
        self._in_flight += 1
        bucket.in_flight += 1
        if namespace is not None:
            namespace.in_flight += 1
        return AdmissionPermit(self, bucket.key, namespace_key)

    def _namespace_load(self, key: Optional[str]) -> Optional[_NamespaceLoad]:
        if key is None:
            return None
        load = self._namespaces.get(key)
        if load is None:
            load = _NamespaceLoad()
            self._namespaces[key] = load
        return load

    def _has_room(
        self,
        bucket: _Bucket,
        namespace: Optional[_NamespaceLoad],
    ) -> bool:
        config = self._config
        if config.max_concurrency and self._in_flight >= config.max_concurrency:
            return False
        if (
            config.principal_max_concurrency
            and bucket.in_flight >= config.principal_max_concurrency
        ):
            return False
        return not (
            namespace is not None
            and namespace.in_flight >= config.namespace_max_concurrency
        )

    def _blocking_reason(
        self,
        bucket: _Bucket,
        namespace: Optional[_NamespaceLoad],
    ) -> str:
        config = self._config
        if (
            config.principal_max_concurrency
            and bucket.in_flight >= config.principal_max_concurrency
        ):
            return REASON_PRINCIPAL_LIMIT
        if (
            namespace is not None
            and namespace.in_flight >= config.namespace_max_concurrency
        ):
            return REASON_NAMESPACE_LIMIT
        return REASON_GLOBAL_LIMIT

    def _arrival_rejection(
        self,
        bucket: _Bucket,
        namespace: Optional[_NamespaceLoad],
    ) -> Optional[str]:
        config = self._config
        principal_queue_full = (
            config.principal_max_concurrency > 0
            and len(bucket.waiters) >= config.principal_max_queue
        )
        namespace_queue_full = (
            namespace is not None and namespace.waiting >= config.namespace_max_queue
        )
        global_queue_full = (
            config.max_concurrency > 0 and self._waiting >= config.max_queue
        )
        if not (principal_queue_full or namespace_queue_full or global_queue_full):
            return None
        reason = self._blocking_reason(bucket, namespace)
        if reason != REASON_GLOBAL_LIMIT:
            return reason
        if principal_queue_full:
            return REASON_PRINCIPAL_LIMIT
        if namespace_queue_full:
            return REASON_NAMESPACE_LIMIT
        return REASON_GLOBAL_LIMIT

    def _forget_if_idle(self, bucket: _Bucket, namespace_key: Optional[str]) -> None:
        if not bucket.in_flight and not bucket.waiters:
            self._buckets.pop(bucket.key, None)
        if namespace_key is not None:
            load = self._namespaces.get(namespace_key)
            if load is not None and not load.in_flight and not load.waiting:
                del self._namespaces[namespace_key]

    def _rejection(self, reason: str) -> AdmissionRejected:
        retry_after = self._config.retry_after_seconds
        if reason == REASON_GLOBAL_LIMIT:
            return AdmissionSaturated(reason, retry_after_seconds=retry_after)
        return PrincipalAdmissionLimited(reason, retry_after_seconds=retry_after)

    def _publish_state(self) -> None:
        record = getattr(self._metrics, "record_principal_admission_state", None)
        if callable(record):
            record(
                in_flight=self._in_flight,
                queued=self._waiting,
                active_principals=len(self._buckets),
            )

    def _record_rejection(self, reason: str, transport: str) -> None:
        record = getattr(self._metrics, "record_principal_admission_rejection", None)
        if callable(record):
            record(reason=reason, transport=transport)


ErrorResponseFactory = Callable[..., "Response"]
CallNext = Callable[["Request"], Awaitable["Response"]]


class HttpAdmission:
    """REST middleware dispatcher for fair principal admission.

    It admits routes whose declared policy is ``model`` or ``heavy``. GraphQL
    routes are skipped here because the GraphQL schema admits each
    model-backed field when it executes. The permit is held until the response
    and any work the request started have settled.
    """

    def __init__(
        self,
        controller: AdmissionController,
        *,
        error_response: ErrorResponseFactory,
    ) -> None:
        """Create the dispatcher.

        Args:
            controller: Shared admission controller, usually the runtime's.
            error_response: Factory for the service error envelope.
        """
        self.controller = controller
        self._error_response = error_response

    async def dispatch(self, request: Request, call_next: CallNext) -> Response:
        """Admit one HTTP request around ``call_next``."""
        controller = self.controller
        if not controller.enabled or not request_needs_principal_admission(request):
            return await call_next(request)
        ticket = controller.ticket(
            request.scope.get(AUTH_SCOPE_KEY), transport=TRANSPORT_REST
        )
        try:
            permit = await controller.acquire(ticket)
        except AdmissionRejected as exc:
            return self.rejection_response(exc)
        try:
            return await call_next(request)
        finally:
            release_when_settled(permit)

    def rejection_response(self, exc: AdmissionRejected) -> Response:
        """Render a value-free 429 or 503 envelope with ``Retry-After``."""
        response = self._error_response(
            exc.status_code,
            exc.error_code,
            str(exc),
            details=exc.to_details(),
        )
        response.headers["Retry-After"] = exc.retry_after_header
        return response


def request_needs_principal_admission(request: Request) -> bool:
    """Return whether REST middleware admits ``request``."""
    from .auth import ROUTE_POLICIES, route_key_for_request

    route_key = route_key_for_request(request)
    if route_key[1] in EXECUTION_ADMITTED_PATHS:
        return False
    policy = ROUTE_POLICIES.get(route_key)
    return policy is not None and policy.admission in _ADMITTED_ROUTE_CLASSES


def release_when_settled(permit: AdmissionPermit) -> None:
    """Release ``permit`` once the current request's accepted work settles.

    Inside a throttled request the release waits for the request's tracked
    work, so a timed-out or cancelled request whose model work still runs
    keeps its unit. Elsewhere the permit is released immediately.
    """
    from .throttle import current_request_work

    def release() -> None:
        permit.release()

    work = current_request_work()
    if work is None:
        release()
    else:
        work.on_finished(release)


def _identity_digest(prefix: str, value: str) -> str:
    digest = hmac.new(_IDENTITY_KEY, value.encode("utf-8"), hashlib.sha256)
    return f"{prefix}:{digest.hexdigest()[:32]}"


def _resolve_signal(signal: asyncio.Future[None]) -> None:
    if not signal.done():
        signal.set_result(None)
