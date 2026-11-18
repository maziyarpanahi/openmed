"""Signed webhook delivery for async OpenMed service jobs.

Each logical delivery has one stable ``X-OpenMed-Delivery-Id``; every retry is
a new attempt with its own ``X-OpenMed-Attempt`` number, timestamp, and nonce,
signed with the version 2 webhook-delivery contract under an explicit key id.
The V1 ``X-OpenMed-Signature`` header is still sent by default for receivers
that have not moved to V2.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx

from .egress import (
    EgressAdapter,
    EgressCredential,
    EgressDataClass,
    EgressDecision,
    EgressDenied,
    EgressOperation,
    EgressPolicy,
    EgressPurpose,
    egress_http_client,
    egress_scope,
)
from .signing import (
    ATTEMPT_HEADER,
    DELIVERY_ID_HEADER,
    EVENT_HEADER,
    MAX_WEBHOOK_ATTEMPT,
    NONCE_HEADER,
    SIGNATURE_HEADER,
    SIGNATURE_PREFIX,
    TIMESTAMP_HEADER,
    WEBHOOK_SIGNATURE_HEADER,
    WEBHOOK_SIGNATURE_VERSION,
    Secret,
    new_delivery_id,
    normalize_delivery_id,
    normalize_event_type,
    sign_request,
    sign_webhook_request,
)
from .signing_keys import SigningKeyError, SigningKeyResolver, SigningKeyring

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
WEBHOOK_ERROR_SIGNING_KEY_UNAVAILABLE = "signing_key_unavailable"
WEBHOOK_ERROR_EGRESS_DENIED = "egress_denied"
WEBHOOK_ERROR_CODES = frozenset(
    {
        WEBHOOK_ERROR_TIMEOUT,
        WEBHOOK_ERROR_CONNECT,
        WEBHOOK_ERROR_TRANSPORT,
        WEBHOOK_ERROR_HTTP_STATUS,
        WEBHOOK_ERROR_CANCELLED,
        WEBHOOK_ERROR_DEADLINE,
        WEBHOOK_ERROR_SENDER,
        WEBHOOK_ERROR_SIGNING_KEY_UNAVAILABLE,
        WEBHOOK_ERROR_EGRESS_DENIED,
    }
)
_ABANDONED_ERROR_CODES = frozenset({WEBHOOK_ERROR_CANCELLED, WEBHOOK_ERROR_DEADLINE})
DEFAULT_WEBHOOK_EVENT = "job.terminal"


@dataclass(frozen=True)
class WebhookAttemptReceipt:
    """Value-free record of one delivery attempt.

    Attributes:
        attempt: 1-based attempt number.
        key_id: Key id that signed the attempt; ``None`` if no key was usable.
        signed_at: Signed Unix timestamp, or ``None`` if nothing was sent.
        status_code: HTTP status of the response, if one arrived.
        error: Controlled code for a failed attempt, otherwise ``None``.
    """

    attempt: int
    key_id: Optional[str]
    signed_at: Optional[int]
    status_code: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe attempt receipt."""
        return {
            "attempt": self.attempt,
            "key_id": self.key_id,
            "signed_at": self.signed_at,
            "status_code": self.status_code,
            "error": self.error,
        }


@dataclass(frozen=True)
class WebhookDeliveryResult:
    """Terminal webhook delivery status.

    ``error`` is one of :data:`WEBHOOK_ERROR_CODES` or ``None``. Results never
    carry callback URLs, exception text, or payload content.

    ``delivery_id``, ``signature_version``, and ``attempt_receipts`` describe
    a delivery signed under the version 2 contract; they are included in
    :meth:`to_dict` when ``delivery_id`` is set.
    """

    success: bool
    attempts: int
    status_code: Optional[int] = None
    error: Optional[str] = None
    delivery_id: Optional[str] = None
    signature_version: Optional[str] = None
    attempt_receipts: tuple[WebhookAttemptReceipt, ...] = ()

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
        summary: dict[str, Any] = {
            "status": self.status,
            "success": bool(self.success),
            "attempts": max(attempts, 0),
            "status_code": status_code,
            "error": error,
        }
        if self.delivery_id is not None:
            summary["delivery_id"] = self.delivery_id
            summary["signature_version"] = self.signature_version
            summary["attempt_receipts"] = [
                receipt.to_dict() for receipt in self.attempt_receipts
            ]
        return summary


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


def _webhook_operation(
    url: str, egress_policy: Optional[EgressPolicy]
) -> EgressOperation:
    return EgressOperation(
        EgressAdapter.WEBHOOKS,
        policy=egress_policy,
        bound_destinations=(url,),
    )


def _authorize_webhook(
    operation: EgressOperation,
    url: str,
    timeout_seconds: float,
) -> EgressDecision:
    return operation.authorize(
        url,
        purpose=EgressPurpose.WEBHOOK,
        data_class=EgressDataClass.METADATA,
        credential=EgressCredential.SIGNATURE,
        timeout_seconds=timeout_seconds,
    )


def authorize_webhook_destination(
    url: str,
    *,
    timeout_seconds: float = DEFAULT_WEBHOOK_TIMEOUT_SECONDS,
    egress_policy: Optional[EgressPolicy] = None,
) -> EgressDecision:
    """Check a callback URL against the egress policy at job submission.

    Args:
        url: Callback URL supplied with the job.
        timeout_seconds: Per-attempt timeout the delivery will use.
        egress_policy: Policy to apply; defaults to the environment policy.

    Returns:
        The allowed decision. Literal-IP callbacks are classified here;
        hostnames are resolved and checked again before each connection.

    Raises:
        EgressDenied: When the policy refuses the callback destination.
    """

    return _authorize_webhook(
        _webhook_operation(url, egress_policy), url, timeout_seconds
    )


def deliver_webhook(
    url: str,
    payload: dict[str, Any],
    *,
    secret: Optional[Secret] = None,
    keyring: Optional[SigningKeyResolver] = None,
    key_id: Optional[str] = None,
    delivery_id: Optional[str] = None,
    include_legacy_signature: bool = True,
    max_attempts: int = 3,
    backoff_seconds: float = 0.5,
    timeout_seconds: float = DEFAULT_WEBHOOK_TIMEOUT_SECONDS,
    client: Optional[httpx.Client] = None,
    sleeper: Optional[Callable[[float], None]] = None,
    clock: Callable[[], float] = time.time,
    cancel_event: Optional[threading.Event] = None,
    max_total_seconds: Optional[float] = None,
    monotonic: Callable[[], float] = time.monotonic,
    egress_policy: Optional[EgressPolicy] = None,
) -> WebhookDeliveryResult:
    """POST a signed webhook payload, retrying non-2xx responses and I/O errors.

    Every attempt of one call shares a stable delivery id and gets its own
    attempt number, timestamp, and nonce, signed with the version 2
    webhook-delivery contract. The signing key is resolved again for each
    attempt, so a retry after a key rotation signs with the newly active key
    while receivers keep verifying the previous key inside its overlap window.

    ``timeout_seconds`` bounds each attempt's connect/read/write/pool I/O and is
    applied to every request, including when the caller supplies ``client``; a
    supplied client's own configuration and ownership are left unchanged.

    Every attempt is authorized by the egress policy (``egress_policy`` or the
    environment policy) under one logical operation bound to ``url``: the first
    check runs before the payload is serialized, and a denied attempt stops
    delivery with error ``egress_denied`` instead of retrying. A client created
    here connects only to addresses that passed the policy.

    Args:
        url: Callback URL.
        payload: JSON-serializable, PHI-free webhook body.
        secret: Shared signing secret. Pass exactly one of ``secret`` or
            ``keyring``.
        keyring: Caller-owned key resolver, such as a
            :class:`~openmed.service.signing_keys.SigningKeyringFile`.
        key_id: Public id for ``secret``; derived from it when omitted.
        delivery_id: Stable delivery id to reuse, for example one sealed with a
            terminal event so redelivery after a restart keeps it; a fresh id
            is generated when omitted.
        include_legacy_signature: Also send the V1 ``X-OpenMed-Signature``
            header, computed with the same key, for receivers that verify V1.
        max_attempts: Attempts for this call, at least 1.
        backoff_seconds: Base delay, doubled after each failed attempt.
        timeout_seconds: Per-attempt I/O timeout.
        client: Optional HTTP client; closed only when created here.
        sleeper: Optional backoff wait function. When omitted, backoff waits
            on ``cancel_event`` (or ``time.sleep`` without one).
        clock: Source of Unix time for signed timestamps and key resolution.
        cancel_event: Optional event that cancels further attempts and wakes
            a pending backoff wait. A cancelled delivery is ``abandoned`` with
            error ``cancelled``.
        max_total_seconds: Optional ceiling on total delivery wall time. Each
            attempt timeout is capped at the remaining budget, and a backoff
            that would cross the ceiling stops delivery as ``abandoned`` with
            error ``deadline_exceeded``.
        monotonic: Clock used for the wall-time ceiling.
        egress_policy: Egress policy to apply; defaults to the environment
            policy.

    Returns:
        A value-free delivery result with a controlled error code and
        per-attempt receipts.

    Raises:
        ValueError: If the arguments are invalid. Nothing is sent.
    """
    if (secret is None) == (keyring is None):
        raise ValueError("pass exactly one of secret or keyring")
    if keyring is not None and key_id is not None:
        raise ValueError("key_id applies only to secret")
    if max_attempts < 1 or max_attempts > MAX_WEBHOOK_ATTEMPT:
        raise ValueError(f"max_attempts must be between 1 and {MAX_WEBHOOK_ATTEMPT}")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds must be greater than or equal to 0")
    if max_total_seconds is not None and max_total_seconds < 0:
        raise ValueError("max_total_seconds must be greater than or equal to 0")

    resolver: SigningKeyResolver = (
        keyring if keyring is not None else SigningKeyring.single(secret, key_id=key_id)  # type: ignore[arg-type]
    )
    stable_delivery_id = (
        new_delivery_id() if delivery_id is None else normalize_delivery_id(delivery_id)
    )
    operation = _webhook_operation(url, egress_policy)
    try:
        decision = _authorize_webhook(operation, url, timeout_seconds)
    except EgressDenied:
        return WebhookDeliveryResult(
            success=False,
            attempts=0,
            status_code=None,
            error=WEBHOOK_ERROR_EGRESS_DENIED,
            delivery_id=stable_delivery_id,
            signature_version=WEBHOOK_SIGNATURE_VERSION,
        )
    event = normalize_event_type(str(payload.get("event", DEFAULT_WEBHOOK_EVENT)))
    request_path = httpx.URL(url).raw_path.decode("ascii")
    body = canonical_json_bytes(payload)
    base_headers = {"Content-Type": "application/json"}
    started = monotonic()

    def _remaining() -> Optional[float]:
        if max_total_seconds is None:
            return None
        return max_total_seconds - (monotonic() - started)

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    owns_client = client is None
    active_client = client or egress_http_client(timeout=timeout_seconds)
    attempts = 0
    last_status_code: Optional[int] = None
    last_error: Optional[str] = None
    receipts: list[WebhookAttemptReceipt] = []
    try:
        for attempt in range(1, max_attempts + 1):
            if _cancelled():
                last_error = WEBHOOK_ERROR_CANCELLED
                break
            remaining = _remaining()
            if remaining is not None and remaining <= 0:
                last_error = WEBHOOK_ERROR_DEADLINE
                break
            try:
                if attempt > 1:
                    decision = _authorize_webhook(operation, url, timeout_seconds)
                checked = decision.require_body(len(body))
            except EgressDenied:
                last_status_code = None
                last_error = WEBHOOK_ERROR_EGRESS_DENIED
                break
            attempts = attempt
            now = clock()
            try:
                signing_key = resolver.current().signing_key(now=now)
            except SigningKeyError:
                last_error = WEBHOOK_ERROR_SIGNING_KEY_UNAVAILABLE
                receipts.append(
                    WebhookAttemptReceipt(
                        attempt=attempt, key_id=None, signed_at=None, error=last_error
                    )
                )
            else:
                assert signing_key.secret is not None
                signed_at = int(now)
                headers = {
                    **base_headers,
                    **sign_webhook_request(
                        "POST",
                        request_path,
                        body,
                        secret=signing_key.secret,
                        key_id=signing_key.key_id,
                        event=event,
                        delivery_id=stable_delivery_id,
                        attempt=attempt,
                        timestamp=signed_at,
                        include_legacy_signature=include_legacy_signature,
                    ),
                }
                attempt_timeout = (
                    timeout_seconds
                    if remaining is None
                    else min(timeout_seconds, remaining)
                )
                status_code: Optional[int] = None
                try:
                    with egress_scope(checked):
                        response = active_client.post(
                            url,
                            content=body,
                            headers=headers,
                            timeout=attempt_timeout,
                        )
                    status_code = response.status_code
                    last_status_code = status_code
                    if 200 <= status_code < 300:
                        receipts.append(
                            WebhookAttemptReceipt(
                                attempt=attempt,
                                key_id=signing_key.key_id,
                                signed_at=signed_at,
                                status_code=status_code,
                            )
                        )
                        return WebhookDeliveryResult(
                            success=True,
                            attempts=attempts,
                            status_code=status_code,
                            error=None,
                            delivery_id=stable_delivery_id,
                            signature_version=WEBHOOK_SIGNATURE_VERSION,
                            attempt_receipts=tuple(receipts),
                        )
                    last_error = WEBHOOK_ERROR_HTTP_STATUS
                except EgressDenied:
                    last_status_code = None
                    last_error = WEBHOOK_ERROR_EGRESS_DENIED
                    receipts.append(
                        WebhookAttemptReceipt(
                            attempt=attempt,
                            key_id=signing_key.key_id,
                            signed_at=signed_at,
                            error=last_error,
                        )
                    )
                    break
                except httpx.HTTPError as exc:
                    last_error = _transport_error_code(exc)
                receipts.append(
                    WebhookAttemptReceipt(
                        attempt=attempt,
                        key_id=signing_key.key_id,
                        signed_at=signed_at,
                        status_code=status_code,
                        error=last_error,
                    )
                )

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
        delivery_id=stable_delivery_id,
        signature_version=WEBHOOK_SIGNATURE_VERSION,
        attempt_receipts=tuple(receipts),
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

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until every submitted delivery has settled.

        Args:
            timeout: Maximum seconds to wait; ``None`` waits without a bound.

        Returns:
            ``True`` when no delivery is queued or in flight, ``False`` if the
            timeout elapsed first.
        """
        with self._condition:
            return self._condition.wait_for(lambda: not self._unsettled, timeout)

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


__all__ = [
    "ATTEMPT_HEADER",
    "DEFAULT_WEBHOOK_DRAIN_SECONDS",
    "DEFAULT_WEBHOOK_EVENT",
    "DEFAULT_WEBHOOK_MAX_DELIVERY_SECONDS",
    "DEFAULT_WEBHOOK_TIMEOUT_SECONDS",
    "DEFAULT_WEBHOOK_WORKERS",
    "DELIVERY_ID_HEADER",
    "EVENT_HEADER",
    "NONCE_HEADER",
    "SIGNATURE_HEADER",
    "SIGNATURE_PREFIX",
    "TIMESTAMP_HEADER",
    "WEBHOOK_ERROR_CANCELLED",
    "WEBHOOK_ERROR_CODES",
    "WEBHOOK_ERROR_CONNECT",
    "WEBHOOK_ERROR_DEADLINE",
    "WEBHOOK_ERROR_HTTP_STATUS",
    "WEBHOOK_ERROR_SENDER",
    "WEBHOOK_ERROR_EGRESS_DENIED",
    "WEBHOOK_ERROR_SIGNING_KEY_UNAVAILABLE",
    "WEBHOOK_ERROR_TIMEOUT",
    "WEBHOOK_ERROR_TRANSPORT",
    "WEBHOOK_SIGNATURE_HEADER",
    "WebhookAttemptReceipt",
    "WebhookDelivery",
    "WebhookDeliveryResult",
    "WebhookDispatcher",
    "WebhookSettle",
    "authorize_webhook_destination",
    "canonical_json_bytes",
    "deliver_webhook",
    "new_delivery_id",
    "pending_delivery_metadata",
    "sign_request",
    "sign_webhook_payload",
]
