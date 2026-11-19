"""HMAC-SHA256 request signing with bounded local replay protection.

Version 1 (``OPENMED-HMAC-SHA256-V1``) signs one request: method, target,
timestamp, nonce, and body digest. The version 2 webhook-delivery contract
(``OPENMED-WEBHOOK-HMAC-SHA256-V2``) wraps that exact V1 canonical string with
the key id, event type, stable delivery id, and attempt number, so a receiver
can tell one logical delivery from its retries and pick the right key.
"""

from __future__ import annotations

import hashlib
import heapq
import hmac
import math
import re
import secrets
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional, Union

SIGNATURE_HEADER = "X-OpenMed-Signature"
TIMESTAMP_HEADER = "X-OpenMed-Timestamp"
NONCE_HEADER = "X-OpenMed-Nonce"
SIGNATURE_PREFIX = "sha256="
DEFAULT_MAX_SKEW_SECONDS = 300
DEFAULT_NONCE_CACHE_SIZE = 10_000

EVENT_HEADER = "X-OpenMed-Event"
DELIVERY_ID_HEADER = "X-OpenMed-Delivery-Id"
ATTEMPT_HEADER = "X-OpenMed-Attempt"
WEBHOOK_SIGNATURE_HEADER = "X-OpenMed-Webhook-Signature"
LEGACY_SIGNATURE_VERSION = "v1"
WEBHOOK_SIGNATURE_VERSION = "v2"
MAX_WEBHOOK_ATTEMPT = 1000

_SIGNING_SCHEME = "OPENMED-HMAC-SHA256-V1"
_WEBHOOK_SIGNING_SCHEME = "OPENMED-WEBHOOK-HMAC-SHA256-V2"
_HTTP_METHOD_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_NONCE_RE = re.compile(r"^[\x21-\x7e]{1,256}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{64}$")
_KEY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_DELIVERY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$")
_EVENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ATTEMPT_RE = re.compile(r"^[1-9][0-9]{0,3}$")
_WEBHOOK_SIGNATURE_VALUE_RE = re.compile(
    r"^v(?P<version>[1-9][0-9]{0,2}),kid=(?P<kid>[^,]{1,64}),"
    r"sha256=(?P<digest>[0-9a-f]{64})$"
)

Body = Union[bytes, bytearray, memoryview, str]
Secret = Union[bytes, str]


class SignatureVerificationError(ValueError):
    """Base class for request-signature verification failures."""


class InvalidSignatureError(SignatureVerificationError):
    """Raised when signature headers or the HMAC digest are invalid."""


class StaleTimestampError(SignatureVerificationError):
    """Raised when a signed timestamp falls outside the accepted window."""


class ReplayDetectedError(SignatureVerificationError):
    """Raised when a nonce has already been accepted in the active window."""


class NonceCacheFullError(SignatureVerificationError):
    """Raised when replay protection cannot safely accept another nonce."""


class NonceCache:
    """Thread-safe, bounded in-memory cache for accepted request nonces.

    Active entries are never evicted merely to make space because that would
    allow their requests to be replayed. Expired entries are removed before a
    nonce is claimed; when all entries are active, the cache fails closed.
    """

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_NONCE_CACHE_SIZE,
        window_seconds: Union[int, float] = DEFAULT_MAX_SKEW_SECONDS,
    ) -> None:
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries < 1
        ):
            raise ValueError("max_entries must be at least 1")
        self.max_entries = max_entries
        self.window_seconds = _normalize_window(window_seconds)
        self._entries: dict[str, float] = {}
        self._expirations: list[tuple[float, int, str]] = []
        self._sequence = 0
        self._lock = threading.Lock()

    def check_and_store(
        self,
        nonce: str,
        *,
        timestamp: Optional[Union[int, float]] = None,
        now: Optional[Union[int, float]] = None,
        window_seconds: Optional[Union[int, float]] = None,
    ) -> bool:
        """Atomically claim ``nonce`` and return whether it was newly stored.

        The entry remains active until the signed timestamp's acceptance
        window has passed. ``False`` means the nonce is already active or its
        requested expiration is already past.
        """
        normalized_nonce = _normalize_nonce(nonce)
        current_time = _normalize_clock_value(now, default=time.time())
        signed_at = _normalize_clock_value(timestamp, default=current_time)
        active_window = (
            self.window_seconds
            if window_seconds is None
            else _normalize_window(window_seconds)
        )
        expires_at = signed_at + active_window

        with self._lock:
            self._purge_expired_locked(current_time)
            if normalized_nonce in self._entries or expires_at < current_time:
                return False
            if len(self._entries) >= self.max_entries:
                raise NonceCacheFullError(
                    "Nonce cache is full with unexpired entries; request rejected"
                )

            self._sequence += 1
            self._entries[normalized_nonce] = expires_at
            heapq.heappush(
                self._expirations,
                (expires_at, self._sequence, normalized_nonce),
            )
            return True

    def purge_expired(self, *, now: Optional[Union[int, float]] = None) -> int:
        """Remove entries past their replay window and return the count."""
        current_time = _normalize_clock_value(now, default=time.time())
        with self._lock:
            before = len(self._entries)
            self._purge_expired_locked(current_time)
            return before - len(self._entries)

    def clear(self) -> None:
        """Remove all cached nonces."""
        with self._lock:
            self._entries.clear()
            self._expirations.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _purge_expired_locked(self, now: float) -> None:
        while self._expirations and self._expirations[0][0] < now:
            expires_at, _, nonce = heapq.heappop(self._expirations)
            if self._entries.get(nonce) == expires_at:
                del self._entries[nonce]


def canonical_signing_string(
    method: str,
    path: str,
    timestamp: int,
    nonce: str,
    body: Body = b"",
) -> bytes:
    """Build the versioned request string covered by the HMAC.

    Only the body's SHA-256 digest is included, so the canonical string never
    contains raw request content or PHI.
    """
    normalized_method = _normalize_method(method)
    normalized_path = _normalize_path(path)
    normalized_timestamp = _normalize_timestamp(timestamp)
    normalized_nonce = _normalize_nonce(nonce)
    body_digest = hashlib.sha256(_body_bytes(body)).hexdigest()
    return "\n".join(
        (
            _SIGNING_SCHEME,
            normalized_method,
            normalized_path,
            str(normalized_timestamp),
            normalized_nonce,
            f"sha256:{body_digest}",
        )
    ).encode("utf-8")


def sign_request(
    method: str,
    path: str,
    body: Body = b"",
    *,
    secret: Secret,
    timestamp: Optional[int] = None,
    nonce: Optional[str] = None,
) -> dict[str, str]:
    """Return headers that sign the exact request method, target, and body."""
    issued_at = _normalize_timestamp(
        int(time.time()) if timestamp is None else timestamp
    )
    request_nonce = _normalize_nonce(
        secrets.token_urlsafe(24) if nonce is None else nonce
    )
    digest = _signature_digest(
        secret,
        canonical_signing_string(method, path, issued_at, request_nonce, body),
    )
    return {
        TIMESTAMP_HEADER: str(issued_at),
        NONCE_HEADER: request_nonce,
        SIGNATURE_HEADER: f"{SIGNATURE_PREFIX}{digest}",
    }


def sign_request_headers(
    method: str,
    path: str,
    body: Body = b"",
    *,
    secret: Secret,
    timestamp: Optional[int] = None,
    nonce: Optional[str] = None,
) -> dict[str, str]:
    """Alias for :func:`sign_request` for client-side header construction."""
    return sign_request(
        method,
        path,
        body,
        secret=secret,
        timestamp=timestamp,
        nonce=nonce,
    )


def verify_request_signature(
    method: str,
    path: str,
    body: Body,
    headers: Mapping[str, str],
    *,
    secret: Secret,
    nonce_cache: NonceCache,
    max_skew_seconds: Union[int, float] = DEFAULT_MAX_SKEW_SECONDS,
    now: Optional[Union[int, float]] = None,
) -> bool:
    """Verify request integrity, freshness, and one-time nonce use.

    Successful verification atomically consumes the nonce. Failures raise a
    :class:`SignatureVerificationError` subclass and do not claim the nonce.
    """
    active_window = _normalize_window(max_skew_seconds)
    current_time = _normalize_clock_value(now, default=time.time())
    timestamp_value = _required_header(headers, TIMESTAMP_HEADER)
    nonce = _required_header(headers, NONCE_HEADER)
    signature_value = _required_header(headers, SIGNATURE_HEADER)

    issued_at = _parse_timestamp_header(timestamp_value)
    try:
        normalized_nonce = _normalize_nonce(nonce)
    except (TypeError, ValueError) as exc:
        raise InvalidSignatureError("Invalid request signature headers") from exc

    if abs(current_time - issued_at) > active_window:
        raise StaleTimestampError("Signed timestamp is outside the accepted window")

    if not signature_value.startswith(SIGNATURE_PREFIX):
        raise InvalidSignatureError("Invalid request signature")
    provided_digest = signature_value[len(SIGNATURE_PREFIX) :]
    if not _SIGNATURE_RE.fullmatch(provided_digest):
        raise InvalidSignatureError("Invalid request signature")

    expected_digest = _signature_digest(
        secret,
        canonical_signing_string(
            method,
            path,
            issued_at,
            normalized_nonce,
            body,
        ),
    )
    if not hmac.compare_digest(provided_digest, expected_digest):
        raise InvalidSignatureError("Invalid request signature")

    if not nonce_cache.check_and_store(
        normalized_nonce,
        timestamp=issued_at,
        now=current_time,
        window_seconds=active_window,
    ):
        raise ReplayDetectedError("Request nonce has already been used")
    return True


@dataclass(frozen=True)
class WebhookSignature:
    """Parsed ``X-OpenMed-Webhook-Signature`` value.

    Attributes:
        version: Signature scheme version, such as ``"v2"``.
        key_id: Identifier of the signing key generation; never key material.
        digest: Lowercase hexadecimal HMAC-SHA256 digest.
    """

    version: str
    key_id: str
    digest: str = field(repr=False)


@dataclass(frozen=True)
class SignedWebhookHeaders:
    """Validated signature-bearing headers of one webhook delivery attempt.

    Attributes:
        timestamp: Signed Unix timestamp in whole seconds.
        nonce: One-time opaque attempt nonce.
        delivery_id: Stable opaque identifier shared by every retry.
        attempt: 1-based attempt number within the delivery.
        event: Event type, such as ``"job.done"``.
        signature: Parsed versioned signature.
    """

    timestamp: int
    nonce: str = field(repr=False)
    delivery_id: str
    attempt: int
    event: str
    signature: WebhookSignature


def new_delivery_id() -> str:
    """Return a fresh opaque delivery identifier.

    Returns:
        A random ``dlv_``-prefixed identifier. It carries no payload, endpoint,
        or job values, so it is safe to log and persist.
    """
    return f"dlv_{secrets.token_hex(16)}"


def normalize_key_id(key_id: str) -> str:
    """Validate a signing key identifier.

    Args:
        key_id: 1-64 characters from ``[A-Za-z0-9._-]``, starting with a letter
            or digit.

    Returns:
        The unchanged key identifier.

    Raises:
        TypeError: If ``key_id`` is not a string.
        ValueError: If ``key_id`` has an invalid format.
    """
    if not isinstance(key_id, str):
        raise TypeError("key_id must be a string")
    if not _KEY_ID_RE.fullmatch(key_id):
        raise ValueError("key_id must be 1-64 characters from [A-Za-z0-9._-]")
    return key_id


def normalize_delivery_id(delivery_id: str) -> str:
    """Validate an opaque webhook delivery identifier.

    Args:
        delivery_id: 8-128 characters from ``[A-Za-z0-9_-]``, starting with a
            letter or digit.

    Returns:
        The unchanged delivery identifier.

    Raises:
        TypeError: If ``delivery_id`` is not a string.
        ValueError: If ``delivery_id`` has an invalid format.
    """
    if not isinstance(delivery_id, str):
        raise TypeError("delivery_id must be a string")
    if not _DELIVERY_ID_RE.fullmatch(delivery_id):
        raise ValueError("delivery_id must be 8-128 characters from [A-Za-z0-9_-]")
    return delivery_id


def normalize_event_type(event: str) -> str:
    """Validate a webhook event type such as ``job.done``.

    Args:
        event: 1-128 characters from ``[A-Za-z0-9._:-]``, starting with a letter
            or digit.

    Returns:
        The unchanged event type.

    Raises:
        TypeError: If ``event`` is not a string.
        ValueError: If ``event`` has an invalid format.
    """
    if not isinstance(event, str):
        raise TypeError("event must be a string")
    if not _EVENT_RE.fullmatch(event):
        raise ValueError("event must be 1-128 characters from [A-Za-z0-9._:-]")
    return event


def webhook_signing_string(
    method: str,
    path: str,
    timestamp: int,
    nonce: str,
    body: Body = b"",
    *,
    key_id: str,
    event: str,
    delivery_id: str,
    attempt: int,
) -> bytes:
    """Build the version 2 webhook-delivery string covered by the HMAC.

    The string is five delivery-contract lines followed by the unchanged
    version 1 canonical string, so V1 canonicalization is preserved exactly
    and only the body digest, never the body, is included.

    Args:
        method: HTTP method, normally ``POST``.
        path: Exact origin-form callback path and query.
        timestamp: Signed Unix timestamp in whole seconds.
        nonce: One-time attempt nonce.
        body: Exact body bytes sent on the wire.
        key_id: Identifier of the signing key.
        event: Event type sent in ``X-OpenMed-Event``.
        delivery_id: Stable delivery identifier shared by every retry.
        attempt: 1-based attempt number.

    Returns:
        UTF-8 bytes of the version 2 signing string.
    """
    contract = "\n".join(
        (
            _WEBHOOK_SIGNING_SCHEME,
            f"key-id:{normalize_key_id(key_id)}",
            f"event:{normalize_event_type(event)}",
            f"delivery-id:{normalize_delivery_id(delivery_id)}",
            f"attempt:{_normalize_attempt(attempt)}",
        )
    ).encode("utf-8")
    return (
        contract
        + b"\n"
        + canonical_signing_string(method, path, timestamp, nonce, body)
    )


def webhook_signature_digest(secret: Secret, signing_string: bytes) -> str:
    """Return the lowercase hexadecimal HMAC-SHA256 digest of a signing string.

    Args:
        secret: Shared signing secret.
        signing_string: Output of :func:`webhook_signing_string` or
            :func:`canonical_signing_string`.

    Returns:
        The 64-character hexadecimal digest.
    """
    return _signature_digest(secret, signing_string)


def format_webhook_signature(key_id: str, digest: str) -> str:
    """Return the ``X-OpenMed-Webhook-Signature`` header value.

    Args:
        key_id: Identifier of the signing key.
        digest: Hexadecimal digest from :func:`webhook_signature_digest`.

    Returns:
        ``v2,kid=<key id>,sha256=<digest>``.

    Raises:
        ValueError: If ``key_id`` or ``digest`` has an invalid format.
    """
    if not isinstance(digest, str) or not _SIGNATURE_RE.fullmatch(digest):
        raise ValueError("digest must be 64 lowercase hexadecimal characters")
    normalized_key_id = normalize_key_id(key_id)
    return f"{WEBHOOK_SIGNATURE_VERSION},kid={normalized_key_id},sha256={digest}"


def parse_webhook_signature(value: str) -> WebhookSignature:
    """Parse an ``X-OpenMed-Webhook-Signature`` header value.

    Args:
        value: Header value in ``v<N>,kid=<key id>,sha256=<hex>`` form.

    Returns:
        The parsed signature. Unsupported versions parse successfully so the
        caller can reject them with a precise reason.

    Raises:
        InvalidSignatureError: If the value is malformed.
    """
    if not isinstance(value, str):
        raise InvalidSignatureError("Invalid webhook signature")
    match = _WEBHOOK_SIGNATURE_VALUE_RE.fullmatch(value)
    if match is None or not _KEY_ID_RE.fullmatch(match.group("kid")):
        raise InvalidSignatureError("Invalid webhook signature")
    return WebhookSignature(
        version=f"v{match.group('version')}",
        key_id=match.group("kid"),
        digest=match.group("digest"),
    )


def parse_webhook_headers(headers: Mapping[str, str]) -> SignedWebhookHeaders:
    """Read and validate the signature-bearing headers of a webhook attempt.

    Args:
        headers: Request headers; names are matched case-insensitively and
            each header must appear exactly once.

    Returns:
        The validated header values.

    Raises:
        InvalidSignatureError: If a header is missing, duplicated, or malformed.
    """
    signature = parse_webhook_signature(
        _required_header(headers, WEBHOOK_SIGNATURE_HEADER)
    )
    timestamp = _parse_timestamp_header(_required_header(headers, TIMESTAMP_HEADER))
    attempt_value = _required_header(headers, ATTEMPT_HEADER)
    if not _ATTEMPT_RE.fullmatch(attempt_value):
        raise InvalidSignatureError("Invalid webhook attempt header")
    try:
        nonce = _normalize_nonce(_required_header(headers, NONCE_HEADER))
        delivery_id = normalize_delivery_id(
            _required_header(headers, DELIVERY_ID_HEADER)
        )
        event = normalize_event_type(_required_header(headers, EVENT_HEADER))
        attempt = _normalize_attempt(int(attempt_value))
    except (TypeError, ValueError) as exc:
        raise InvalidSignatureError("Invalid webhook delivery headers") from exc
    return SignedWebhookHeaders(
        timestamp=timestamp,
        nonce=nonce,
        delivery_id=delivery_id,
        attempt=attempt,
        event=event,
        signature=signature,
    )


def sign_webhook_request(
    method: str,
    path: str,
    body: Body = b"",
    *,
    secret: Secret,
    key_id: str,
    event: str,
    delivery_id: str,
    attempt: int,
    timestamp: Optional[int] = None,
    nonce: Optional[str] = None,
    include_legacy_signature: bool = False,
) -> dict[str, str]:
    """Return the headers that sign one webhook delivery attempt.

    Args:
        method: HTTP method, normally ``POST``.
        path: Exact origin-form callback path and query.
        body: Exact body bytes sent on the wire.
        secret: Signing secret of the key named by ``key_id``.
        key_id: Identifier of the signing key generation.
        event: Event type.
        delivery_id: Stable identifier shared by every retry of the delivery.
        attempt: 1-based attempt number; each attempt also gets a fresh nonce.
        timestamp: Signed Unix timestamp; defaults to the current time.
        nonce: Attempt nonce; defaults to a cryptographically random value.
        include_legacy_signature: Also emit the V1 ``X-OpenMed-Signature``
            header, computed with the same key, for receivers that have not
            moved to the V2 contract yet.

    Returns:
        Event, delivery id, attempt, timestamp, nonce, and V2 signature headers,
        plus the V1 signature header when requested.
    """
    issued_at = _normalize_timestamp(
        int(time.time()) if timestamp is None else timestamp
    )
    attempt_nonce = _normalize_nonce(
        secrets.token_urlsafe(24) if nonce is None else nonce
    )
    digest = _signature_digest(
        secret,
        webhook_signing_string(
            method,
            path,
            issued_at,
            attempt_nonce,
            body,
            key_id=key_id,
            event=event,
            delivery_id=delivery_id,
            attempt=attempt,
        ),
    )
    headers = {
        EVENT_HEADER: event,
        DELIVERY_ID_HEADER: delivery_id,
        ATTEMPT_HEADER: str(attempt),
        TIMESTAMP_HEADER: str(issued_at),
        NONCE_HEADER: attempt_nonce,
        WEBHOOK_SIGNATURE_HEADER: format_webhook_signature(key_id, digest),
    }
    if include_legacy_signature:
        legacy_digest = _signature_digest(
            secret,
            canonical_signing_string(method, path, issued_at, attempt_nonce, body),
        )
        headers[SIGNATURE_HEADER] = f"{SIGNATURE_PREFIX}{legacy_digest}"
    return headers


def _required_header(headers: Mapping[str, str], target: str) -> str:
    values = [
        value
        for name, value in headers.items()
        if isinstance(name, str) and name.casefold() == target.casefold()
    ]
    if len(values) != 1 or not isinstance(values[0], str):
        raise InvalidSignatureError("Missing or duplicate request signature header")
    return values[0]


def _parse_timestamp_header(value: str) -> int:
    if not value.isascii() or not value.isdigit():
        raise InvalidSignatureError("Invalid signed timestamp")
    issued_at = int(value)
    if value != str(issued_at):
        raise InvalidSignatureError("Invalid signed timestamp")
    return issued_at


def _signature_digest(secret: Secret, signing_string: bytes) -> str:
    return hmac.new(_secret_bytes(secret), signing_string, hashlib.sha256).hexdigest()


def _secret_bytes(secret: Secret) -> bytes:
    if isinstance(secret, str):
        secret_bytes = secret.encode("utf-8")
    elif isinstance(secret, bytes):
        secret_bytes = secret
    else:
        raise TypeError("secret must be bytes or str")
    if not secret_bytes:
        raise ValueError("secret must not be blank")
    return secret_bytes


def _body_bytes(body: Body) -> bytes:
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, bytes):
        return body
    if isinstance(body, (bytearray, memoryview)):
        return bytes(body)
    raise TypeError("body must be bytes-like or str")


def _normalize_method(method: str) -> str:
    if not isinstance(method, str) or not _HTTP_METHOD_RE.fullmatch(method):
        raise ValueError("method must be a valid HTTP method token")
    return method.upper()


def _normalize_path(path: str) -> str:
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError("path must be an origin-form request target")
    if "#" in path or any(not "!" <= character <= "~" for character in path):
        raise ValueError(
            "path must be a printable ASCII request target without a fragment"
        )
    return path


def _normalize_timestamp(timestamp: int) -> int:
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise TypeError("timestamp must be an integer Unix timestamp")
    if timestamp < 0:
        raise ValueError("timestamp must not be negative")
    return timestamp


def _normalize_nonce(nonce: str) -> str:
    if not isinstance(nonce, str):
        raise TypeError("nonce must be a string")
    if not _NONCE_RE.fullmatch(nonce):
        raise ValueError("nonce must contain 1-256 printable ASCII characters")
    return nonce


def _normalize_attempt(attempt: int) -> int:
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise TypeError("attempt must be an integer")
    if not 1 <= attempt <= MAX_WEBHOOK_ATTEMPT:
        raise ValueError(f"attempt must be between 1 and {MAX_WEBHOOK_ATTEMPT}")
    return attempt


def _normalize_clock_value(
    value: Optional[Union[int, float]], *, default: Union[int, float]
) -> float:
    normalized = default if value is None else value
    if isinstance(normalized, bool) or not isinstance(normalized, (int, float)):
        raise TypeError("clock values must be numeric")
    clock_value = float(normalized)
    if not math.isfinite(clock_value):
        raise ValueError("clock values must be finite")
    return clock_value


def _normalize_window(value: Union[int, float]) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("max_skew_seconds must be greater than 0")
    return float(value)


__all__ = [
    "ATTEMPT_HEADER",
    "DEFAULT_MAX_SKEW_SECONDS",
    "DEFAULT_NONCE_CACHE_SIZE",
    "DELIVERY_ID_HEADER",
    "EVENT_HEADER",
    "InvalidSignatureError",
    "LEGACY_SIGNATURE_VERSION",
    "MAX_WEBHOOK_ATTEMPT",
    "NONCE_HEADER",
    "NonceCache",
    "NonceCacheFullError",
    "ReplayDetectedError",
    "SIGNATURE_HEADER",
    "SIGNATURE_PREFIX",
    "SignatureVerificationError",
    "SignedWebhookHeaders",
    "StaleTimestampError",
    "TIMESTAMP_HEADER",
    "WEBHOOK_SIGNATURE_HEADER",
    "WEBHOOK_SIGNATURE_VERSION",
    "WebhookSignature",
    "canonical_signing_string",
    "format_webhook_signature",
    "new_delivery_id",
    "normalize_delivery_id",
    "normalize_event_type",
    "normalize_key_id",
    "parse_webhook_headers",
    "parse_webhook_signature",
    "sign_request",
    "sign_request_headers",
    "sign_webhook_request",
    "verify_request_signature",
    "webhook_signature_digest",
    "webhook_signing_string",
]
