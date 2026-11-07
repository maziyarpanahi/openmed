"""Signed webhook delivery for async OpenMed service jobs.

Each logical delivery has one stable ``X-OpenMed-Delivery-Id``; every retry is
a new attempt with its own ``X-OpenMed-Attempt`` number, timestamp, and nonce,
signed with the version 2 webhook-delivery contract under an explicit key id.
The V1 ``X-OpenMed-Signature`` header is still sent by default for receivers
that have not moved to V2.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import httpx

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
DEFAULT_WEBHOOK_EVENT = "job.terminal"
WEBHOOK_ERROR_SIGNING_KEY_UNAVAILABLE = "signing_key_unavailable"


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

    ``error`` is value-free: ``"HTTP <status>"`` for a non-2xx response or a
    controlled code such as ``"timeout"`` or ``"transport_error"``. It never
    contains endpoint URLs, response bodies, or exception text.

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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe delivery summary."""
        summary: dict[str, Any] = {
            "success": self.success,
            "attempts": self.attempts,
            "status_code": self.status_code,
            "error": self.error,
        }
        if self.delivery_id is not None:
            summary["delivery_id"] = self.delivery_id
            summary["signature_version"] = self.signature_version
            summary["attempt_receipts"] = [
                receipt.to_dict() for receipt in self.attempt_receipts
            ]
        return summary


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
    secret: Optional[Secret] = None,
    keyring: Optional[SigningKeyResolver] = None,
    key_id: Optional[str] = None,
    delivery_id: Optional[str] = None,
    include_legacy_signature: bool = True,
    max_attempts: int = 3,
    backoff_seconds: float = 0.5,
    timeout_seconds: float = DEFAULT_WEBHOOK_TIMEOUT_SECONDS,
    client: Optional[httpx.Client] = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
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
        sleeper: Backoff sleep function.
        clock: Source of Unix time for signed timestamps and key resolution.

    Returns:
        A value-free delivery result with per-attempt receipts.

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

    resolver: SigningKeyResolver = (
        keyring if keyring is not None else SigningKeyring.single(secret, key_id=key_id)  # type: ignore[arg-type]
    )
    stable_delivery_id = (
        new_delivery_id() if delivery_id is None else normalize_delivery_id(delivery_id)
    )
    event = normalize_event_type(str(payload.get("event", DEFAULT_WEBHOOK_EVENT)))
    request_path = httpx.URL(url).raw_path.decode("ascii")
    body = canonical_json_bytes(payload)
    base_headers = {"Content-Type": "application/json"}

    owns_client = client is None
    active_client = client or httpx.Client(timeout=timeout_seconds)
    attempts = 0
    last_status_code: Optional[int] = None
    last_error: Optional[str] = None
    receipts: list[WebhookAttemptReceipt] = []
    try:
        for attempt in range(1, max_attempts + 1):
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
                status_code: Optional[int] = None
                try:
                    response = active_client.post(
                        url,
                        content=body,
                        headers=headers,
                        timeout=timeout_seconds,
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
                    last_error = f"HTTP {status_code}"
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

            if attempt < max_attempts and backoff_seconds:
                sleeper(backoff_seconds * (2 ** (attempt - 1)))
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


def _transport_error_code(exc: httpx.HTTPError) -> str:
    """Return a controlled code; exception text can echo endpoint URLs."""
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    return "transport_error"


__all__ = [
    "ATTEMPT_HEADER",
    "DEFAULT_WEBHOOK_EVENT",
    "DEFAULT_WEBHOOK_TIMEOUT_SECONDS",
    "DELIVERY_ID_HEADER",
    "EVENT_HEADER",
    "NONCE_HEADER",
    "SIGNATURE_HEADER",
    "SIGNATURE_PREFIX",
    "TIMESTAMP_HEADER",
    "WEBHOOK_ERROR_SIGNING_KEY_UNAVAILABLE",
    "WEBHOOK_SIGNATURE_HEADER",
    "WebhookAttemptReceipt",
    "WebhookDeliveryResult",
    "canonical_json_bytes",
    "deliver_webhook",
    "new_delivery_id",
    "sign_webhook_payload",
]
