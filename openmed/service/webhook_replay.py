"""Receiver-side webhook verification with a durable replay ledger.

:class:`WebhookReceiver` verifies the version 2 webhook-delivery signature
(see :mod:`openmed.service.signing`), resolves the signing key through a
caller-owned :class:`~openmed.service.signing_keys.SigningKeyResolver`, and
claims each attempt in a :class:`FileReplayLedger` so that:

- one logical delivery is accepted once, however many retries arrive;
- a retransmitted attempt is rejected as a replay, including after restart;
- concurrent receivers sharing one ledger file cannot both claim a delivery.

Every call returns a typed :class:`WebhookVerification`. Failed verification
never touches the ledger, so a forged or stale request cannot consume a valid
delivery identity. The ledger fails closed: when its integrity or capacity
cannot be established, verification returns ``unavailable`` instead of
accepting.

The ledger file stores only keyed digests of delivery ids, attempt identities,
and nonces, plus key ids, attempt numbers, timestamps, and states, and an
HMAC-SHA256 over the whole document detects tampering and truncation. It never
stores payloads, signatures, secrets, or raw identifiers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

from ._state_files import atomic_write_private, exclusive_lock, read_bounded
from .signing import (
    DEFAULT_MAX_SKEW_SECONDS,
    LEGACY_SIGNATURE_VERSION,
    NONCE_HEADER,
    SIGNATURE_HEADER,
    SIGNATURE_PREFIX,
    TIMESTAMP_HEADER,
    WEBHOOK_SIGNATURE_HEADER,
    WEBHOOK_SIGNATURE_VERSION,
    Body,
    InvalidSignatureError,
    canonical_signing_string,
    parse_webhook_headers,
    webhook_signature_digest,
    webhook_signing_string,
)
from .signing_keys import (
    RetiredSigningKeyError,
    SigningKeyError,
    SigningKeyResolver,
    SigningKeyring,
    UnknownSigningKeyError,
)

LEDGER_FORMAT = "openmed.webhook-replay-ledger"
LEDGER_SCHEMA_VERSION = 1
DEFAULT_LEDGER_MAX_ENTRIES = 10_000
DEFAULT_DELIVERY_TTL_SECONDS = 24 * 60 * 60
MAX_DELIVERY_TTL_SECONDS = 30 * 24 * 60 * 60
MIN_LEDGER_KEY_BYTES = 32
DELIVERY_STATE_CLAIMED = "claimed"
DELIVERY_STATE_APPLIED = "applied"

REASON_NEW_DELIVERY = "new_delivery"
REASON_LEGACY_ACCEPTED = "legacy_v1_accepted"
REASON_DELIVERY_IN_PROGRESS = "delivery_in_progress"
REASON_DELIVERY_APPLIED = "delivery_applied"
REASON_ATTEMPT_REPLAYED = "attempt_replayed"
REASON_NONCE_REPLAYED = "nonce_replayed"
REASON_TIMESTAMP_OUTSIDE_WINDOW = "timestamp_outside_window"
REASON_HEADERS_INVALID = "headers_invalid"
REASON_VERSION_UNSUPPORTED = "signature_version_unsupported"
REASON_SIGNATURE_MISMATCH = "signature_mismatch"
REASON_REQUEST_INVALID = "request_invalid"
REASON_KEYRING_UNAVAILABLE = "keyring_unavailable"
LEDGER_INTEGRITY_FAILED = "ledger_integrity_failed"
LEDGER_ROLLBACK = "ledger_rollback"
LEDGER_FULL = "ledger_full"
LEDGER_UNREADABLE = "ledger_unreadable"
LEDGER_WRITE_FAILED = "ledger_write_failed"
LEDGER_LOCK_FAILED = "ledger_lock_failed"

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_MAC_PREFIX = "hmac-sha256:"
_DOCUMENT_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "sequence",
        "updated_at",
        "deliveries",
        "attempts",
        "nonces",
        "mac",
    }
)
_DELIVERY_FIELDS = frozenset(
    {"state", "key_id", "attempt", "claimed_at", "applied_at", "expires_at"}
)
# Generous per-entry bound so a corrupted or hostile file cannot exhaust memory.
_MAX_BYTES_PER_ENTRY = 512

_T = TypeVar("_T")


class WebhookOutcome(str, Enum):
    """Typed result of verifying one webhook delivery attempt."""

    ACCEPTED = "accepted"
    """New delivery; the caller should apply its effect, then mark it."""
    DUPLICATE = "duplicate"
    """A fresh retry of a delivery that is already claimed or applied."""
    REPLAYED = "replayed"
    """A retransmission of an attempt (or nonce) that was already seen."""
    STALE = "stale"
    """The signed timestamp is outside the accepted window."""
    BAD_SIGNATURE = "bad_signature"
    """Headers are missing or malformed, or the HMAC does not match."""
    UNKNOWN_KEY = "unknown_key"
    """The key id is unknown or the key is not active yet."""
    RETIRED_KEY = "retired_key"
    """The key is retired or its overlap window has closed."""
    UNAVAILABLE = "unavailable"
    """Replay state or keys could not be established; fails closed."""


_HTTP_STATUS = {
    WebhookOutcome.ACCEPTED: 200,
    WebhookOutcome.DUPLICATE: 200,
    WebhookOutcome.REPLAYED: 409,
    WebhookOutcome.STALE: 401,
    WebhookOutcome.BAD_SIGNATURE: 401,
    WebhookOutcome.UNKNOWN_KEY: 401,
    WebhookOutcome.RETIRED_KEY: 401,
    WebhookOutcome.UNAVAILABLE: 503,
}


class ReplayLedgerError(RuntimeError):
    """Value-free replay-ledger failure.

    Attributes:
        code: Controlled reason, such as ``"ledger_integrity_failed"``.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class WebhookVerification:
    """Value-free outcome of verifying one webhook delivery attempt.

    Identity fields (key id, delivery id, attempt, event, signed time) are only
    populated once the signature has verified; for rejected requests they stay
    ``None`` so unauthenticated input is never echoed back as trusted.

    Attributes:
        outcome: Typed outcome.
        reason: Controlled reason code for the outcome.
        signature_version: ``"v2"``, or ``"v1"`` for legacy compatibility.
        key_id: Key id that verified the signature.
        delivery_id: Stable delivery id shared by every retry.
        attempt: Attempt number of this transmission.
        event: Signed event type.
        signed_at: Signed Unix timestamp.
    """

    outcome: WebhookOutcome
    reason: str
    signature_version: Optional[str] = None
    key_id: Optional[str] = None
    delivery_id: Optional[str] = None
    attempt: Optional[int] = None
    event: Optional[str] = None
    signed_at: Optional[int] = None

    @property
    def accepted(self) -> bool:
        """Whether the caller should apply this delivery's effect now."""
        return self.outcome is WebhookOutcome.ACCEPTED

    @property
    def http_status(self) -> int:
        """Suggested response status.

        ``200`` for accepted and duplicate deliveries (so the sender stops
        retrying), ``409`` for replays, ``401`` for authentication failures, and
        ``503`` when the receiver fails closed (so the sender retries later).
        """
        return _HTTP_STATUS[self.outcome]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, value-free receipt."""
        return {
            "outcome": self.outcome.value,
            "reason": self.reason,
            "signature_version": self.signature_version,
            "key_id": self.key_id,
            "delivery_id": self.delivery_id,
            "attempt": self.attempt,
            "event": self.event,
            "signed_at": self.signed_at,
        }


@dataclass
class _LedgerState:
    sequence: int = 0
    deliveries: dict[str, dict[str, Any]] = field(default_factory=dict)
    attempts: dict[str, float] = field(default_factory=dict)
    nonces: dict[str, float] = field(default_factory=dict)

    def size(self) -> int:
        return len(self.deliveries) + len(self.attempts) + len(self.nonces)

    def purge(self, now: float) -> int:
        before = self.size()
        self.deliveries = {
            digest: entry
            for digest, entry in self.deliveries.items()
            if entry["expires_at"] >= now
        }
        self.attempts = {
            digest: expires
            for digest, expires in self.attempts.items()
            if expires >= now
        }
        self.nonces = {
            digest: expires for digest, expires in self.nonces.items() if expires >= now
        }
        return before - self.size()


class FileReplayLedger:
    """Durable, bounded, integrity-checked replay and idempotency ledger.

    Every operation runs as one transaction under a thread lock and a
    cross-process file lock: read and verify the file, apply the change, and
    replace the file atomically. Entries expire on their own TTL: attempt and
    nonce records when their signed timestamp leaves the acceptance window,
    delivery records ``delivery_ttl_seconds`` after they were claimed. Active
    entries are never evicted to make room; a full ledger fails closed.

    Args:
        path: Ledger file. A sibling ``<name>.lock`` file serializes writers.
        integrity_key: Secret of at least :data:`MIN_LEDGER_KEY_BYTES` bytes
            for the ledger HMAC and identifier digests. Keep it separate from
            webhook signing keys.
        max_entries: Upper bound on stored delivery, attempt, and nonce records.
        delivery_ttl_seconds: How long an accepted delivery is remembered; this
            is the receiver's idempotency horizon.
        clock: Source of Unix time.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike[str]],
        *,
        integrity_key: bytes,
        max_entries: int = DEFAULT_LEDGER_MAX_ENTRIES,
        delivery_ttl_seconds: float = DEFAULT_DELIVERY_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(integrity_key, bytes) or len(integrity_key) < (
            MIN_LEDGER_KEY_BYTES
        ):
            raise ValueError(
                f"integrity_key must be at least {MIN_LEDGER_KEY_BYTES} bytes"
            )
        if (
            isinstance(max_entries, bool)
            or not isinstance(max_entries, int)
            or max_entries < 3
        ):
            raise ValueError("max_entries must be an integer of at least 3")
        if (
            not _is_finite_number(delivery_ttl_seconds)
            or delivery_ttl_seconds <= 0
            or delivery_ttl_seconds > MAX_DELIVERY_TTL_SECONDS
        ):
            raise ValueError(
                "delivery_ttl_seconds must be greater than 0 and at most "
                f"{MAX_DELIVERY_TTL_SECONDS}"
            )
        self.path = Path(path)
        self.max_entries = max_entries
        self.delivery_ttl_seconds = float(delivery_ttl_seconds)
        self._key = integrity_key
        self._clock = clock
        self._lock_path = self.path.with_name(f"{self.path.name}.lock")
        self._thread_lock = threading.Lock()
        self._last_sequence = 0
        self._max_bytes = 4096 + max_entries * _MAX_BYTES_PER_ENTRY

    def claim_attempt(
        self,
        *,
        delivery_id: Optional[str],
        attempt: int,
        nonce: str,
        key_id: str,
        attempt_expires_at: float,
        now: Optional[float] = None,
    ) -> tuple[WebhookOutcome, str]:
        """Atomically record one verified attempt and classify it.

        Args:
            delivery_id: Stable delivery id, or ``None`` for a legacy V1
                request, which is tracked by nonce only.
            attempt: Attempt number.
            nonce: Attempt nonce.
            key_id: Key id that verified the attempt.
            attempt_expires_at: When the attempt's signed timestamp leaves the
                acceptance window; its records are kept until then.
            now: Unix time; defaults to the ledger clock.

        Returns:
            ``(outcome, reason)``: accepted, duplicate, replayed, or
            ``unavailable`` with ``ledger_full``.

        Raises:
            ReplayLedgerError: If the ledger cannot be read, verified, or
                written. Nothing is claimed in that case.
        """
        current_time = self._now(now)
        nonce_digest = self._digest(b"nonce", nonce)
        attempt_digest = (
            None
            if delivery_id is None
            else self._digest(b"attempt", f"{delivery_id}\x00{attempt}")
        )
        delivery_digest = (
            None if delivery_id is None else self._digest(b"delivery", delivery_id)
        )

        def claim(state: _LedgerState) -> tuple[tuple[WebhookOutcome, str], bool]:
            changed = state.purge(current_time) > 0
            if attempt_digest is not None and attempt_digest in state.attempts:
                return (WebhookOutcome.REPLAYED, REASON_ATTEMPT_REPLAYED), changed
            if nonce_digest in state.nonces:
                return (WebhookOutcome.REPLAYED, REASON_NONCE_REPLAYED), changed
            existing = (
                None
                if delivery_digest is None
                else state.deliveries.get(delivery_digest)
            )
            needed = 1 + (attempt_digest is not None)
            needed += delivery_digest is not None and existing is None
            if state.size() + needed > self.max_entries:
                return (WebhookOutcome.UNAVAILABLE, LEDGER_FULL), changed
            expires_at = float(attempt_expires_at)
            state.nonces[nonce_digest] = expires_at
            if attempt_digest is not None:
                state.attempts[attempt_digest] = expires_at
            if delivery_digest is None:
                return (WebhookOutcome.ACCEPTED, REASON_LEGACY_ACCEPTED), True
            if existing is not None:
                reason = (
                    REASON_DELIVERY_APPLIED
                    if existing["state"] == DELIVERY_STATE_APPLIED
                    else REASON_DELIVERY_IN_PROGRESS
                )
                return (WebhookOutcome.DUPLICATE, reason), True
            state.deliveries[delivery_digest] = {
                "state": DELIVERY_STATE_CLAIMED,
                "key_id": key_id,
                "attempt": attempt,
                "claimed_at": current_time,
                "applied_at": None,
                "expires_at": current_time + self.delivery_ttl_seconds,
            }
            return (WebhookOutcome.ACCEPTED, REASON_NEW_DELIVERY), True

        return self._transaction(claim, now=current_time)

    def mark_applied(self, delivery_id: str, *, now: Optional[float] = None) -> bool:
        """Record that a claimed delivery's effect was applied.

        Args:
            delivery_id: Delivery id from an accepted verification.
            now: Unix time; defaults to the ledger clock.

        Returns:
            ``True`` if a claimed delivery moved to ``applied``.

        Raises:
            ReplayLedgerError: If the ledger cannot be read or written.
        """
        current_time = self._now(now)
        digest = self._digest(b"delivery", delivery_id)

        def apply(state: _LedgerState) -> tuple[bool, bool]:
            changed = state.purge(current_time) > 0
            entry = state.deliveries.get(digest)
            if entry is None or entry["state"] != DELIVERY_STATE_CLAIMED:
                return False, changed
            entry["state"] = DELIVERY_STATE_APPLIED
            entry["applied_at"] = current_time
            return True, True

        return self._transaction(apply, now=current_time)

    def release(self, delivery_id: str, *, now: Optional[float] = None) -> bool:
        """Drop a claim whose effect was not applied, so a retry can claim it.

        The attempt and nonce records stay, so retransmitting the released
        attempt is still rejected as a replay; only a fresh retry can claim.

        Args:
            delivery_id: Delivery id from an accepted verification.
            now: Unix time; defaults to the ledger clock.

        Returns:
            ``True`` if a claimed delivery was released.

        Raises:
            ReplayLedgerError: If the ledger cannot be read or written.
        """
        current_time = self._now(now)
        digest = self._digest(b"delivery", delivery_id)

        def drop(state: _LedgerState) -> tuple[bool, bool]:
            changed = state.purge(current_time) > 0
            entry = state.deliveries.get(digest)
            if entry is None or entry["state"] != DELIVERY_STATE_CLAIMED:
                return False, changed
            del state.deliveries[digest]
            return True, True

        return self._transaction(drop, now=current_time)

    def purge_expired(self, *, now: Optional[float] = None) -> int:
        """Remove expired records and return how many were removed."""
        current_time = self._now(now)

        def purge(state: _LedgerState) -> tuple[int, bool]:
            removed = state.purge(current_time)
            return removed, removed > 0

        return self._transaction(purge, now=current_time)

    def stats(self) -> dict[str, int]:
        """Return value-free record counts and capacity.

        Raises:
            ReplayLedgerError: If the ledger cannot be read or verified.
        """

        def count(state: _LedgerState) -> tuple[dict[str, int], bool]:
            states = [entry["state"] for entry in state.deliveries.values()]
            return {
                "sequence": state.sequence,
                "deliveries_claimed": states.count(DELIVERY_STATE_CLAIMED),
                "deliveries_applied": states.count(DELIVERY_STATE_APPLIED),
                "attempts": len(state.attempts),
                "nonces": len(state.nonces),
                "entries": state.size(),
                "max_entries": self.max_entries,
            }, False

        return self._transaction(count, now=None)

    def _transaction(
        self,
        operation: Callable[[_LedgerState], tuple[_T, bool]],
        *,
        now: Optional[float],
    ) -> _T:
        with self._thread_lock:
            try:
                with exclusive_lock(self._lock_path):
                    state = self._read()
                    result, changed = operation(state)
                    if changed:
                        state.sequence += 1
                        self._write(state, now=self._now(now))
                    return result
            except OSError:
                raise ReplayLedgerError(LEDGER_LOCK_FAILED) from None

    def _read(self) -> _LedgerState:
        try:
            data = read_bounded(self.path, self._max_bytes)
        except OSError:
            raise ReplayLedgerError(LEDGER_UNREADABLE) from None
        if data is None:
            if self._last_sequence > 0:
                # The file vanished after this process saw it: never restart
                # replay protection from empty.
                raise ReplayLedgerError(LEDGER_ROLLBACK)
            return _LedgerState()
        state = self._decode(data)
        if state.sequence < self._last_sequence:
            raise ReplayLedgerError(LEDGER_ROLLBACK)
        self._last_sequence = state.sequence
        return state

    def _decode(self, data: bytes) -> _LedgerState:
        try:
            document = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ReplayLedgerError(LEDGER_INTEGRITY_FAILED) from None
        if not isinstance(document, dict) or set(document) != _DOCUMENT_FIELDS:
            raise ReplayLedgerError(LEDGER_INTEGRITY_FAILED)
        mac = document.pop("mac")
        try:
            expected = self._mac(document)
        except (TypeError, ValueError):
            raise ReplayLedgerError(LEDGER_INTEGRITY_FAILED) from None
        if (
            not isinstance(mac, str)
            or not mac.isascii()
            or not hmac.compare_digest(mac, expected)
        ):
            raise ReplayLedgerError(LEDGER_INTEGRITY_FAILED)
        try:
            return _state_from_document(document)
        except (KeyError, TypeError, ValueError):
            raise ReplayLedgerError(LEDGER_INTEGRITY_FAILED) from None

    def _write(self, state: _LedgerState, *, now: float) -> None:
        document: dict[str, Any] = {
            "format": LEDGER_FORMAT,
            "schema_version": LEDGER_SCHEMA_VERSION,
            "sequence": state.sequence,
            "updated_at": now,
            "deliveries": state.deliveries,
            "attempts": state.attempts,
            "nonces": state.nonces,
        }
        document["mac"] = self._mac(document)
        try:
            atomic_write_private(self.path, _canonical(document) + b"\n")
        except OSError:
            raise ReplayLedgerError(LEDGER_WRITE_FAILED) from None
        self._last_sequence = state.sequence

    def _mac(self, document: Mapping[str, Any]) -> str:
        digest = hmac.new(
            self._key, b"ledger-document\x00" + _canonical(document), hashlib.sha256
        )
        return f"{_MAC_PREFIX}{digest.hexdigest()}"

    def _digest(self, label: bytes, value: str) -> str:
        message = label + b"\x00" + value.encode("utf-8")
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def _now(self, now: Optional[float]) -> float:
        value = self._clock() if now is None else now
        if not _is_finite_number(value):
            raise ValueError("clock values must be finite numbers")
        return float(value)


class WebhookReceiver:
    """Verify webhook delivery attempts and claim them in a replay ledger.

    Typical use::

        verification = receiver.verify(method, path, body, headers)
        if verification.accepted:
            try:
                apply(body)
            except Exception:
                receiver.release(verification)
                raise
            receiver.mark_applied(verification)
        return verification.http_status

    Args:
        keyring: Resolver for the receiver's key generations, such as a
            :class:`~openmed.service.signing_keys.SigningKeyringFile`.
        ledger: Durable replay ledger.
        max_skew_seconds: Accepted distance between the signed timestamp and
            the receiver clock.
        accept_legacy_v1: Also accept requests that carry only the V1
            ``X-OpenMed-Signature`` header. V1 has no delivery id, so such
            requests get nonce replay protection but no retry de-duplication.
            Disabled by default.
        clock: Source of Unix time.
    """

    def __init__(
        self,
        *,
        keyring: Union[SigningKeyResolver, SigningKeyring],
        ledger: FileReplayLedger,
        max_skew_seconds: float = DEFAULT_MAX_SKEW_SECONDS,
        accept_legacy_v1: bool = False,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if (
            not _is_finite_number(max_skew_seconds)
            or max_skew_seconds <= 0
            or max_skew_seconds > ledger.delivery_ttl_seconds
        ):
            raise ValueError(
                "max_skew_seconds must be greater than 0 and no longer than the "
                "ledger delivery TTL"
            )
        self.keyring = keyring
        self.ledger = ledger
        self.max_skew_seconds = float(max_skew_seconds)
        self.accept_legacy_v1 = bool(accept_legacy_v1)
        self._clock = clock

    def verify(
        self,
        method: str,
        path: str,
        body: Body,
        headers: Mapping[str, str],
        *,
        now: Optional[float] = None,
    ) -> WebhookVerification:
        """Verify one delivery attempt and claim it on success.

        Args:
            method: Request method as received.
            path: Exact origin-form path and query as received.
            body: Exact body bytes as received.
            headers: Request headers.
            now: Unix time; defaults to the receiver clock.

        Returns:
            The typed outcome. This method does not raise for hostile input.
        """
        current_time = float(self._clock() if now is None else now)
        if _has_header(headers, WEBHOOK_SIGNATURE_HEADER):
            return self._verify_v2(method, path, body, headers, current_time)
        if _has_header(headers, SIGNATURE_HEADER):
            if self.accept_legacy_v1:
                return self._verify_legacy(method, path, body, headers, current_time)
            return _rejected(WebhookOutcome.BAD_SIGNATURE, REASON_VERSION_UNSUPPORTED)
        return _rejected(WebhookOutcome.BAD_SIGNATURE, REASON_HEADERS_INVALID)

    def mark_applied(self, verification: WebhookVerification) -> bool:
        """Mark an accepted delivery as applied.

        Args:
            verification: Result of :meth:`verify`.

        Returns:
            ``True`` if the delivery moved from claimed to applied.

        Raises:
            ReplayLedgerError: If the ledger cannot be updated.
        """
        if not verification.accepted or verification.delivery_id is None:
            return False
        return self.ledger.mark_applied(verification.delivery_id, now=self._clock())

    def release(self, verification: WebhookVerification) -> bool:
        """Release an accepted delivery whose effect could not be applied.

        Args:
            verification: Result of :meth:`verify`.

        Returns:
            ``True`` if the claim was released for a later retry.

        Raises:
            ReplayLedgerError: If the ledger cannot be updated.
        """
        if not verification.accepted or verification.delivery_id is None:
            return False
        return self.ledger.release(verification.delivery_id, now=self._clock())

    def _verify_v2(
        self,
        method: str,
        path: str,
        body: Body,
        headers: Mapping[str, str],
        now: float,
    ) -> WebhookVerification:
        try:
            signed = parse_webhook_headers(headers)
        except InvalidSignatureError:
            return _rejected(WebhookOutcome.BAD_SIGNATURE, REASON_HEADERS_INVALID)
        if signed.signature.version != WEBHOOK_SIGNATURE_VERSION:
            return _rejected(WebhookOutcome.BAD_SIGNATURE, REASON_VERSION_UNSUPPORTED)
        if abs(now - signed.timestamp) > self.max_skew_seconds:
            return _rejected(WebhookOutcome.STALE, REASON_TIMESTAMP_OUTSIDE_WINDOW)
        try:
            keyring = self.keyring.current()
        except SigningKeyError:
            return _rejected(WebhookOutcome.UNAVAILABLE, REASON_KEYRING_UNAVAILABLE)
        try:
            key = keyring.verification_key(signed.signature.key_id, now=now)
        except UnknownSigningKeyError as exc:
            return _rejected(WebhookOutcome.UNKNOWN_KEY, exc.code)
        except RetiredSigningKeyError as exc:
            return _rejected(WebhookOutcome.RETIRED_KEY, exc.code)
        assert key.secret is not None
        try:
            signing_string = webhook_signing_string(
                method,
                path,
                signed.timestamp,
                signed.nonce,
                body,
                key_id=key.key_id,
                event=signed.event,
                delivery_id=signed.delivery_id,
                attempt=signed.attempt,
            )
        except (TypeError, ValueError):
            return _rejected(WebhookOutcome.BAD_SIGNATURE, REASON_REQUEST_INVALID)
        expected = webhook_signature_digest(key.secret, signing_string)
        if not hmac.compare_digest(expected, signed.signature.digest):
            return _rejected(WebhookOutcome.BAD_SIGNATURE, REASON_SIGNATURE_MISMATCH)

        try:
            outcome, reason = self.ledger.claim_attempt(
                delivery_id=signed.delivery_id,
                attempt=signed.attempt,
                nonce=signed.nonce,
                key_id=key.key_id,
                attempt_expires_at=signed.timestamp + self.max_skew_seconds,
                now=now,
            )
        except ReplayLedgerError as exc:
            outcome, reason = WebhookOutcome.UNAVAILABLE, exc.code
        return WebhookVerification(
            outcome=outcome,
            reason=reason,
            signature_version=WEBHOOK_SIGNATURE_VERSION,
            key_id=key.key_id,
            delivery_id=signed.delivery_id,
            attempt=signed.attempt,
            event=signed.event,
            signed_at=signed.timestamp,
        )

    def _verify_legacy(
        self,
        method: str,
        path: str,
        body: Body,
        headers: Mapping[str, str],
        now: float,
    ) -> WebhookVerification:
        try:
            timestamp_value = _single_header(headers, TIMESTAMP_HEADER)
            nonce = _single_header(headers, NONCE_HEADER)
            signature_value = _single_header(headers, SIGNATURE_HEADER)
            if not timestamp_value.isascii() or not timestamp_value.isdigit():
                raise ValueError
            signed_at = int(timestamp_value)
            if str(signed_at) != timestamp_value:
                raise ValueError
            if not signature_value.startswith(SIGNATURE_PREFIX):
                raise ValueError
            provided = signature_value[len(SIGNATURE_PREFIX) :]
            if not _DIGEST_RE.fullmatch(provided):
                raise ValueError
            signing_string = canonical_signing_string(
                method, path, signed_at, nonce, body
            )
        except (TypeError, ValueError):
            return _rejected(WebhookOutcome.BAD_SIGNATURE, REASON_HEADERS_INVALID)
        if abs(now - signed_at) > self.max_skew_seconds:
            return _rejected(WebhookOutcome.STALE, REASON_TIMESTAMP_OUTSIDE_WINDOW)
        try:
            keys = self.keyring.current().verification_keys(now=now)
        except SigningKeyError:
            return _rejected(WebhookOutcome.UNAVAILABLE, REASON_KEYRING_UNAVAILABLE)
        # Compare against every usable key so timing does not reveal which
        # generation, if any, was closest.
        matched = None
        for key in keys:
            assert key.secret is not None
            digest = webhook_signature_digest(key.secret, signing_string)
            if hmac.compare_digest(digest, provided) and matched is None:
                matched = key
        if matched is None:
            return _rejected(WebhookOutcome.BAD_SIGNATURE, REASON_SIGNATURE_MISMATCH)
        try:
            outcome, reason = self.ledger.claim_attempt(
                delivery_id=None,
                attempt=1,
                nonce=nonce,
                key_id=matched.key_id,
                attempt_expires_at=signed_at + self.max_skew_seconds,
                now=now,
            )
        except ReplayLedgerError as exc:
            outcome, reason = WebhookOutcome.UNAVAILABLE, exc.code
        return WebhookVerification(
            outcome=outcome,
            reason=reason,
            signature_version=LEGACY_SIGNATURE_VERSION,
            key_id=matched.key_id,
            signed_at=signed_at,
        )


def _rejected(outcome: WebhookOutcome, reason: str) -> WebhookVerification:
    return WebhookVerification(outcome=outcome, reason=reason)


def _has_header(headers: Mapping[str, str], target: str) -> bool:
    folded = target.casefold()
    return any(
        isinstance(name, str) and name.casefold() == folded for name in headers.keys()
    )


def _single_header(headers: Mapping[str, str], target: str) -> str:
    folded = target.casefold()
    values = [
        value
        for name, value in headers.items()
        if isinstance(name, str) and name.casefold() == folded
    ]
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError("missing or duplicate header")
    return values[0]


def _state_from_document(document: Mapping[str, Any]) -> _LedgerState:
    if (
        document["format"] != LEDGER_FORMAT
        or document["schema_version"] != LEDGER_SCHEMA_VERSION
        or isinstance(document["schema_version"], bool)
    ):
        raise ValueError("unsupported ledger")
    sequence = document["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("invalid sequence")
    deliveries = document["deliveries"]
    attempts = document["attempts"]
    nonces = document["nonces"]
    if not all(isinstance(item, dict) for item in (deliveries, attempts, nonces)):
        raise TypeError("invalid ledger sections")
    for digest, entry in deliveries.items():
        if (
            not _DIGEST_RE.fullmatch(digest)
            or not isinstance(entry, dict)
            or set(entry) != _DELIVERY_FIELDS
            or entry["state"] not in {DELIVERY_STATE_CLAIMED, DELIVERY_STATE_APPLIED}
            or not isinstance(entry["key_id"], str)
            or isinstance(entry["attempt"], bool)
            or not isinstance(entry["attempt"], int)
            or not _is_finite_number(entry["claimed_at"])
            or not _is_finite_number(entry["expires_at"])
            or not (
                entry["applied_at"] is None or _is_finite_number(entry["applied_at"])
            )
        ):
            raise ValueError("invalid delivery entry")
    for section in (attempts, nonces):
        for digest, expires_at in section.items():
            if not _DIGEST_RE.fullmatch(digest) or not _is_finite_number(expires_at):
                raise ValueError("invalid attempt entry")
    return _LedgerState(
        sequence=sequence,
        deliveries=deliveries,
        attempts=attempts,
        nonces=nonces,
    )


def _canonical(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


__all__ = [
    "DEFAULT_DELIVERY_TTL_SECONDS",
    "DEFAULT_LEDGER_MAX_ENTRIES",
    "DELIVERY_STATE_APPLIED",
    "DELIVERY_STATE_CLAIMED",
    "FileReplayLedger",
    "LEDGER_FORMAT",
    "LEDGER_SCHEMA_VERSION",
    "MIN_LEDGER_KEY_BYTES",
    "ReplayLedgerError",
    "WebhookOutcome",
    "WebhookReceiver",
    "WebhookVerification",
]
