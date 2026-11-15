"""Signed webhook delivery for async OpenMed service jobs."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx

from .signing import (
    NONCE_HEADER,
    SIGNATURE_HEADER,
    SIGNATURE_PREFIX,
    TIMESTAMP_HEADER,
    sign_request,
)

EVENT_HEADER = "X-OpenMed-Event"
DEFAULT_WEBHOOK_TIMEOUT_SECONDS = 10.0
DEFAULT_WEBHOOK_WORKERS = 2
DEFAULT_WEBHOOK_MAX_DELIVERY_SECONDS = 300.0
DEFAULT_WEBHOOK_DRAIN_SECONDS = 2.0

WEBHOOK_ERROR_TIMEOUT = "timeout"
WEBHOOK_ERROR_CONNECT = "connect"
WEBHOOK_ERROR_TRANSPORT = "transport"
WEBHOOK_ERROR_HTTP_STATUS = "http_status"
WEBHOOK_ERROR_CANCELLED = "cancelled"
WEBHOOK_ERROR_DEADLINE = "deadline_exceeded"
WEBHOOK_ERROR_SENDER = "sender_error"
WEBHOOK_ERROR_CODES = frozenset(
    {
        WEBHOOK_ERROR_TIMEOUT,
        WEBHOOK_ERROR_CONNECT,
        WEBHOOK_ERROR_TRANSPORT,
        WEBHOOK_ERROR_HTTP_STATUS,
        WEBHOOK_ERROR_CANCELLED,
        WEBHOOK_ERROR_DEADLINE,
        WEBHOOK_ERROR_SENDER,
    }
)
_ABANDONED_ERROR_CODES = frozenset({WEBHOOK_ERROR_CANCELLED, WEBHOOK_ERROR_DEADLINE})


@dataclass(frozen=True)
class WebhookDeliveryResult:
    """Terminal webhook delivery status.

    ``error`` is one of :data:`WEBHOOK_ERROR_CODES` or ``None``. Results never
    carry callback URLs, exception text, or payload content.
    """

    success: bool
    attempts: int
    status_code: Optional[int] = None
    error: Optional[str] = None

    @property
    def status(self) -> str:
        """Return ``delivered``, ``failed``, or ``abandoned``."""
        if self.success:
            return "delivered"
        if self.error in _ABANDONED_ERROR_CODES:
            return "abandoned"
        return "failed"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe delivery summary with controlled values only.

        Free-form ``error`` values, for example from a custom sender, are
        replaced with ``sender_error`` so no exception text is persisted.
        """
        error: Optional[str] = None
        if self.error is not None and not self.success:
            error = (
                self.error
                if self.error in WEBHOOK_ERROR_CODES
                else WEBHOOK_ERROR_SENDER
            )
        status_code = self.status_code
        if isinstance(status_code, bool) or not isinstance(status_code, int):
            status_code = None
        attempts = self.attempts
        if isinstance(attempts, bool) or not isinstance(attempts, int):
            attempts = 0
        return {
            "status": self.status,
            "success": bool(self.success),
            "attempts": max(attempts, 0),
            "status_code": status_code,
            "error": error,
        }


def pending_delivery_metadata() -> dict[str, Any]:
    """Return the delivery summary recorded while a webhook is in flight."""
    return {
        "status": "pending",
        "success": None,
        "attempts": 0,
        "status_code": None,
        "error": None,
    }


def _transport_error_code(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return WEBHOOK_ERROR_TIMEOUT
    if isinstance(exc, httpx.ConnectError):
        return WEBHOOK_ERROR_CONNECT
    return WEBHOOK_ERROR_TRANSPORT


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a webhook payload deterministically."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sign_webhook_payload(
    payload: Any,
    *,
    secret: str,
    path: str = "/",
    timestamp: Optional[int] = None,
    nonce: Optional[str] = None,
) -> tuple[bytes, str, str, str]:
    """Return body bytes plus timestamp, nonce, and signature header values."""
    body = canonical_json_bytes(payload)
    headers = sign_request(
        "POST",
        path,
        body,
        secret=secret,
        timestamp=timestamp,
        nonce=nonce,
    )
    return (
        body,
        headers[TIMESTAMP_HEADER],
        headers[NONCE_HEADER],
        headers[SIGNATURE_HEADER],
    )


def deliver_webhook(
    url: str,
    payload: dict[str, Any],
    *,
    secret: str,
    max_attempts: int = 3,
    backoff_seconds: float = 0.5,
    timeout_seconds: float = DEFAULT_WEBHOOK_TIMEOUT_SECONDS,
    client: Optional[httpx.Client] = None,
    sleeper: Optional[Callable[[float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    max_total_seconds: Optional[float] = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> WebhookDeliveryResult:
    """POST a signed webhook payload, retrying non-2xx responses and I/O errors.

    Each HTTP delivery attempt receives its own timestamp and nonce so a
    receiver can apply replay protection without rejecting a legitimate retry.
    ``timeout_seconds`` bounds each attempt's connect/read/write/pool I/O and is
    applied to every request, including when the caller supplies ``client``; a
    supplied client's own configuration and ownership are left unchanged.

    Args:
        url: Callback URL.
        payload: JSON-serializable webhook payload.
        secret: Shared HMAC signing secret.
        max_attempts: Maximum number of HTTP attempts.
        backoff_seconds: Initial exponential backoff between attempts.
        timeout_seconds: Per-attempt I/O timeout.
        client: Optional caller-owned ``httpx.Client``.
        sleeper: Optional backoff wait function. When omitted, backoff waits
            on ``cancel_event`` (or ``time.sleep`` without one).
        cancel_event: Optional event that cancels further attempts and wakes
            a pending backoff wait. A cancelled delivery is ``abandoned`` with
            error ``cancelled``.
        max_total_seconds: Optional ceiling on total delivery wall time. Each
            attempt timeout is capped at the remaining budget, and a backoff
            that would cross the ceiling stops delivery as ``abandoned`` with
            error ``deadline_exceeded``.
        monotonic: Clock used for the wall-time ceiling.

    Returns:
        A :class:`WebhookDeliveryResult` with a controlled error code.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds must be greater than or equal to 0")
    if max_total_seconds is not None and max_total_seconds < 0:
        raise ValueError("max_total_seconds must be greater than or equal to 0")

    request_path = httpx.URL(url).raw_path.decode("ascii")
    body = canonical_json_bytes(payload)
    base_headers = {
        "Content-Type": "application/json",
        EVENT_HEADER: str(payload.get("event", "job.terminal")),
    }
    started = monotonic()

    def _remaining() -> Optional[float]:
        if max_total_seconds is None:
            return None
        return max_total_seconds - (monotonic() - started)

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    owns_client = client is None
    active_client = client or httpx.Client(timeout=timeout_seconds)
    attempts = 0
    last_status_code: Optional[int] = None
    last_error: Optional[str] = None
    try:
        for attempt in range(1, max_attempts + 1):
            if _cancelled():
                last_error = WEBHOOK_ERROR_CANCELLED
                break
            remaining = _remaining()
            if remaining is not None and remaining <= 0:
                last_error = WEBHOOK_ERROR_DEADLINE
                break
            attempts = attempt
            signed_headers = sign_request(
                "POST",
                request_path,
                body,
                secret=secret,
            )
            headers = {**base_headers, **signed_headers}
            attempt_timeout = (
                timeout_seconds
                if remaining is None
                else min(timeout_seconds, remaining)
            )
            try:
                response = active_client.post(
                    url,
                    content=body,
                    headers=headers,
                    timeout=attempt_timeout,
                )
                last_status_code = response.status_code
                if 200 <= response.status_code < 300:
                    return WebhookDeliveryResult(
                        success=True,
                        attempts=attempts,
                        status_code=response.status_code,
                        error=None,
                    )
                last_error = WEBHOOK_ERROR_HTTP_STATUS
            except httpx.HTTPError as exc:
                last_error = _transport_error_code(exc)

            if attempt >= max_attempts:
                break
            delay = backoff_seconds * (2 ** (attempt - 1))
            remaining = _remaining()
            if remaining is not None and delay >= remaining:
                last_error = WEBHOOK_ERROR_DEADLINE
                break
            if delay:
                if sleeper is not None:
                    sleeper(delay)
                elif cancel_event is not None:
                    cancel_event.wait(delay)
                else:
                    time.sleep(delay)
    finally:
        if owns_client:
            active_client.close()

    return WebhookDeliveryResult(
        success=False,
        attempts=attempts,
        status_code=last_status_code,
        error=last_error,
    )


WebhookDelivery = Callable[[threading.Event, float], WebhookDeliveryResult]
WebhookSettle = Callable[[WebhookDeliveryResult], None]


class _ScheduledDelivery:
    __slots__ = ("deliver", "on_settle", "deadline", "settled")

    def __init__(
        self,
        deliver: WebhookDelivery,
        on_settle: WebhookSettle,
        deadline: float,
    ) -> None:
        self.deliver = deliver
        self.on_settle = on_settle
        self.deadline = deadline
        self.settled = False


def _unattempted(code: str) -> WebhookDeliveryResult:
    return WebhookDeliveryResult(
        success=False,
        attempts=0,
        status_code=None,
        error=code,
    )


class WebhookDispatcher:
    """Bounded background executor for terminal webhook deliveries.

    Deliveries run on at most ``max_workers`` daemon threads, so a slow or
    failing receiver never occupies a job worker and never keeps the
    interpreter alive after :meth:`shutdown`. Every delivery settles exactly
    once through its ``on_settle`` callback.

    Args:
        max_workers: Maximum number of concurrent deliveries.
        max_delivery_seconds: Server-side ceiling on the total wall time of
            one delivery, measured from scheduling and including retries and
            backoff waits. It applies on top of the per-request retry settings.
        drain_seconds: Default time :meth:`shutdown` waits for in-flight
            deliveries to settle before recording them as abandoned.
        monotonic: Clock used for deadlines and the drain bound.
    """

    def __init__(
        self,
        *,
        max_workers: int = DEFAULT_WEBHOOK_WORKERS,
        max_delivery_seconds: float = DEFAULT_WEBHOOK_MAX_DELIVERY_SECONDS,
        drain_seconds: float = DEFAULT_WEBHOOK_DRAIN_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if int(max_workers) < 1:
            raise ValueError("max_workers must be at least 1")
        if float(max_delivery_seconds) <= 0:
            raise ValueError("max_delivery_seconds must be greater than 0")
        if float(drain_seconds) < 0:
            raise ValueError("drain_seconds must be greater than or equal to 0")
        self.max_workers = int(max_workers)
        self.max_delivery_seconds = float(max_delivery_seconds)
        self.drain_seconds = float(drain_seconds)
        self._monotonic = monotonic
        self._cancel = threading.Event()
        self._condition = threading.Condition()
        self._pending: deque[_ScheduledDelivery] = deque()
        self._unsettled: list[_ScheduledDelivery] = []
        self._threads: list[threading.Thread] = []
        self._closed = False

    def submit(self, deliver: WebhookDelivery, on_settle: WebhookSettle) -> None:
        """Schedule one delivery without blocking the caller.

        Args:
            deliver: Callable receiving the shared cancellation event and the
                remaining wall-time budget in seconds.
            on_settle: Callable that records the final result. Called exactly
                once, from a dispatcher thread or from :meth:`shutdown`.
        """
        item = _ScheduledDelivery(
            deliver,
            on_settle,
            self._monotonic() + self.max_delivery_seconds,
        )
        with self._condition:
            closed = self._closed
            if not closed:
                self._unsettled.append(item)
                self._pending.append(item)
                if len(self._threads) < self.max_workers:
                    self._start_worker_locked()
                self._condition.notify_all()
        if closed:
            self._settle(item, _unattempted(WEBHOOK_ERROR_CANCELLED))

    def shutdown(self, drain_seconds: Optional[float] = None) -> None:
        """Cancel backoff waits and settle every delivery within a drain bound.

        Deliveries still running when the bound elapses, and deliveries that
        never started, are recorded as ``abandoned`` with error ``cancelled``.

        Args:
            drain_seconds: Override for the configured drain bound.
        """
        bound = self.drain_seconds if drain_seconds is None else float(drain_seconds)
        with self._condition:
            if not self._closed:
                self._closed = True
                self._cancel.set()
                self._condition.notify_all()
            deadline = self._monotonic() + max(bound, 0.0)
            while self._unsettled:
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            leftovers = list(self._unsettled)
        for item in leftovers:
            self._settle(item, _unattempted(WEBHOOK_ERROR_CANCELLED))

    def _start_worker_locked(self) -> None:
        thread = threading.Thread(
            target=self._work,
            name=f"openmed-webhook-{len(self._threads)}",
            daemon=True,
        )
        self._threads.append(thread)
        thread.start()

    def _work(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if not self._pending:
                    return
                item = self._pending.popleft()
            self._settle(item, self._run(item))

    def _run(self, item: _ScheduledDelivery) -> WebhookDeliveryResult:
        if self._cancel.is_set():
            return _unattempted(WEBHOOK_ERROR_CANCELLED)
        remaining = item.deadline - self._monotonic()
        if remaining <= 0:
            return _unattempted(WEBHOOK_ERROR_DEADLINE)
        try:
            result = item.deliver(self._cancel, remaining)
        except Exception:
            return _unattempted(WEBHOOK_ERROR_SENDER)
        if not isinstance(result, WebhookDeliveryResult):
            return _unattempted(WEBHOOK_ERROR_SENDER)
        return result

    def _settle(
        self,
        item: _ScheduledDelivery,
        result: WebhookDeliveryResult,
    ) -> None:
        with self._condition:
            if item.settled:
                return
            item.settled = True
        try:
            item.on_settle(result)
        except Exception:
            # A failed record write must not stop the worker or surface
            # exception text through the thread excepthook.
            pass
        finally:
            with self._condition:
                if item in self._unsettled:
                    self._unsettled.remove(item)
                self._condition.notify_all()
