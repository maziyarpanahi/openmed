"""HMAC-SHA256 request signing with bounded local replay protection."""

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
from typing import Optional, Union

SIGNATURE_HEADER = "X-OpenMed-Signature"
TIMESTAMP_HEADER = "X-OpenMed-Timestamp"
NONCE_HEADER = "X-OpenMed-Nonce"
SIGNATURE_PREFIX = "sha256="
DEFAULT_MAX_SKEW_SECONDS = 300
DEFAULT_NONCE_CACHE_SIZE = 10_000

_SIGNING_SCHEME = "OPENMED-HMAC-SHA256-V1"
_HTTP_METHOD_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_NONCE_RE = re.compile(r"^[\x21-\x7e]{1,256}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{64}$")

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
        if isinstance(window_seconds, bool) or window_seconds <= 0:
            raise ValueError("window_seconds must be greater than 0")

        self.max_entries = max_entries
        self.window_seconds = float(window_seconds)
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
    request_nonce = _normalize_nonce(nonce or secrets.token_urlsafe(24))
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
    "DEFAULT_MAX_SKEW_SECONDS",
    "DEFAULT_NONCE_CACHE_SIZE",
    "InvalidSignatureError",
    "NONCE_HEADER",
    "NonceCache",
    "NonceCacheFullError",
    "ReplayDetectedError",
    "SIGNATURE_HEADER",
    "SIGNATURE_PREFIX",
    "SignatureVerificationError",
    "StaleTimestampError",
    "TIMESTAMP_HEADER",
    "canonical_signing_string",
    "sign_request",
    "sign_request_headers",
    "verify_request_signature",
]
