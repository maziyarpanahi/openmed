"""Protected payload lifecycle for asynchronous service jobs.

An async de-identification job carries PHI-bearing content from submission
until a worker consumes it (the documents plus the webhook target and secret),
and a terminal webhook event from completion until it is delivered.  This
module keeps that content out of job metadata and behind an opaque payload
reference with an explicit, bounded retention deadline.

Two stores implement one contract:

* :class:`MemoryJobPayloadStore` keeps canonical payload bytes in process
  memory only.  It writes nothing to disk and is the default when durable
  retention is not requested.
* :class:`EncryptedJobPayloadStore` retains payloads on local disk.  Every
  file holds only a versioned AEAD envelope from
  :mod:`openmed.core.aead_envelope`, sealed under a caller-supplied
  :class:`~openmed.core.key_provider.KeyProvider`; temporary files hold the
  same ciphertext, so a crash remnant is protected like the final payload.

Each payload is bound, as authenticated associated data, to its opaque
reference, job, kind, schema version, creation time, and retention deadline,
so a payload moved to another job or given a longer deadline fails closed.
Expired and deleted references can never be reopened.  Payloads that a
delivery has pinned are not removed until the pin is released.

Every lifecycle transition records a value-free :class:`PayloadReceipt` that
holds only an opaque reference, the payload kind, a ciphertext digest, a
timestamp, and a controlled reason code.  Errors are
:class:`~openmed.core.key_provider.EncryptionError` subclasses carrying one
controlled ``code``; messages never contain content, keys, or paths.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import secrets
import tempfile
import threading
from collections import Counter, OrderedDict, deque
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from openmed.core.aead_envelope import (
    EnvelopeCipher,
    envelope_from_text,
    envelope_to_text,
)
from openmed.core.key_provider import (
    EncryptionError,
    KeyProvider,
    register_error_messages,
)

__all__ = [
    "DEFAULT_PAYLOAD_RETENTION_SECONDS",
    "JOB_PAYLOAD_FILE_FORMAT",
    "JOB_PAYLOAD_KEY_PURPOSE",
    "JOB_PAYLOAD_SCHEMA_VERSION",
    "PAYLOAD_KIND_INPUT",
    "PAYLOAD_KIND_TERMINAL_EVENT",
    "PAYLOAD_MODE_ENCRYPTED",
    "PAYLOAD_MODE_MEMORY",
    "PAYLOAD_STATE_DELETED",
    "PAYLOAD_STATE_EXPIRED",
    "PAYLOAD_STATE_RETAINED",
    "EncryptedJobPayloadStore",
    "JobPayloadError",
    "JobPayloadStore",
    "MemoryJobPayloadStore",
    "PayloadRef",
    "PayloadReceipt",
]

JOB_PAYLOAD_SCHEMA_VERSION = 1
JOB_PAYLOAD_KEY_PURPOSE = "service-job-payloads"
JOB_PAYLOAD_FILE_FORMAT = "openmed.service.job-payload"
DEFAULT_PAYLOAD_RETENTION_SECONDS = 24 * 60 * 60

PAYLOAD_KIND_INPUT = "input"
PAYLOAD_KIND_TERMINAL_EVENT = "terminal_event"
PAYLOAD_KINDS = frozenset({PAYLOAD_KIND_INPUT, PAYLOAD_KIND_TERMINAL_EVENT})

PAYLOAD_MODE_MEMORY = "memory"
PAYLOAD_MODE_ENCRYPTED = "encrypted"

PAYLOAD_STATE_RETAINED = "retained"
PAYLOAD_STATE_DELETED = "deleted"
PAYLOAD_STATE_EXPIRED = "expired"

RECEIPT_CREATED = "created"
RECEIPT_READ = "read"
RECEIPT_READ_REFUSED = "read_refused"
RECEIPT_EXPIRED = "expired"
RECEIPT_DELETED = "deleted"
RECEIPT_CLEANUP_FAILED = "cleanup_failed"
RECEIPT_EVENTS = frozenset(
    {
        RECEIPT_CREATED,
        RECEIPT_READ,
        RECEIPT_READ_REFUSED,
        RECEIPT_EXPIRED,
        RECEIPT_DELETED,
        RECEIPT_CLEANUP_FAILED,
    }
)

REASON_RETENTION_ELAPSED = "retention_elapsed"
REASON_ORPHAN = "orphan"
REASON_CRASH_REMNANT = "crash_remnant"

_REF_RE = re.compile(r"jp_[0-9a-f]{32}\Z")
_JOB_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PAYLOAD_SUFFIX = ".payload.json"
_PAYLOAD_NAME_RE = re.compile(r"(jp_[0-9a-f]{32})\.payload\.json\Z")
_TEMP_NAME_RE = re.compile(r"\.(jp_[0-9a-f]{32})\.[A-Za-z0-9_]+\.tmp\Z")
_MAX_RECEIPTS = 1024
_MAX_TOMBSTONES = 4096
_FILE_FIELDS = frozenset(
    {
        "format",
        "schema_version",
        "ref",
        "kind",
        "job_id",
        "created_at",
        "expires_at",
        "digest",
        "envelope",
    }
)

register_error_messages(
    {
        "job_payload_failed": "job payload operation failed",
        "payload_ref_invalid": "job payload reference is invalid",
        "payload_missing": "job payload is unavailable",
        "payload_expired": "job payload retention has elapsed",
        "payload_deleted": "job payload was deleted",
        "payload_damaged": "stored job payload is damaged",
        "payload_read_failed": "stored job payload could not be read",
        "payload_write_failed": "job payload could not be stored",
        "payload_storage_full": "job payload storage is full",
        "payload_retention_invalid": (
            "job payload retention is outside the allowed range"
        ),
        "payload_body_invalid": "job payload body is not a JSON object",
    }
)


class JobPayloadError(EncryptionError):
    """Value-free failure of a job payload lifecycle operation.

    Attributes:
        code: Controlled identifier naming the failure, for example
            ``"payload_expired"``. It never contains content or paths.
    """

    default_code = "job_payload_failed"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _parse_timestamp(raw_value: object) -> Optional[datetime]:
    if not isinstance(raw_value, str) or not raw_value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(raw_value[:-1] + "+00:00")
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)


def _canonical_bytes(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _require_ref(ref: object) -> str:
    if not isinstance(ref, str) or _REF_RE.fullmatch(ref) is None:
        raise JobPayloadError("payload_ref_invalid")
    return ref


def _require_kind(kind: object) -> str:
    if kind not in PAYLOAD_KINDS:
        raise JobPayloadError("payload_ref_invalid")
    return str(kind)


def _require_job_id(job_id: object) -> str:
    if not isinstance(job_id, str) or _JOB_ID_RE.fullmatch(job_id) is None:
        raise JobPayloadError("payload_ref_invalid")
    return job_id


def _require_reason(reason: object) -> str:
    if not isinstance(reason, str) or _CODE_RE.fullmatch(reason) is None:
        raise ValueError("reason must be a controlled code")
    return reason


@dataclass(frozen=True)
class PayloadRef:
    """Opaque, value-free handle to one retained job payload.

    Attributes:
        ref: Random opaque reference, ``jp_`` followed by 32 hex characters.
        kind: ``"input"`` or ``"terminal_event"``.
        job_id: Opaque identifier of the job that owns the payload.
        created_at: UTC creation timestamp.
        expires_at: UTC retention deadline; the payload is unreadable after it.
        digest: ``sha256:`` digest of the stored ciphertext envelope, or
            ``None`` for memory-only payloads.
        schema_version: Payload envelope schema version.
    """

    ref: str
    kind: str
    job_id: str
    created_at: str
    expires_at: str
    digest: Optional[str] = None
    schema_version: int = JOB_PAYLOAD_SCHEMA_VERSION

    def to_record(
        self,
        *,
        state: str = PAYLOAD_STATE_RETAINED,
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        """Return the public, value-free job-record entry for this payload.

        Args:
            state: Lifecycle state to report.
            reason: Optional controlled reason code for the state.

        Returns:
            A JSON-safe mapping without content, keys, or storage paths.
        """

        return {
            "ref": self.ref,
            "schema_version": self.schema_version,
            "state": state,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "digest": self.digest,
            "reason": reason,
        }

    def context(self) -> dict[str, Any]:
        """Return the record identity bound into the envelope's AAD."""

        return {
            "ref": self.ref,
            "kind": self.kind,
            "job_id": self.job_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class PayloadReceipt:
    """Value-free evidence of one payload lifecycle transition.

    Attributes:
        event: One of ``created``, ``read``, ``read_refused``, ``expired``,
            ``deleted``, or ``cleanup_failed``.
        ref: Opaque payload reference.
        kind: Payload kind, or ``"unknown"`` for an unreadable remnant.
        at: UTC timestamp of the transition.
        reason: Controlled reason or failure code.
        digest: Ciphertext digest when known; never a digest of plaintext.
        schema_version: Payload envelope schema version.
    """

    event: str
    ref: str
    kind: str
    at: str
    reason: Optional[str] = None
    digest: Optional[str] = None
    schema_version: int = JOB_PAYLOAD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the receipt as a JSON-safe mapping."""

        return {
            "event": self.event,
            "ref": self.ref,
            "kind": self.kind,
            "at": self.at,
            "reason": self.reason,
            "digest": self.digest,
            "schema_version": self.schema_version,
        }


class JobPayloadStore:
    """Lifecycle contract shared by the memory and encrypted payload stores.

    The store owns an index of retained payloads, pins held by in-flight
    deliveries, and value-free receipts.  Subclasses provide only the storage
    primitives.  All public methods are thread-safe.

    Args:
        max_retention_seconds: Upper bound for any payload's retention.
        clock: Source of timezone-aware UTC timestamps.
    """

    mode = "abstract"
    retains_failed_deliveries = False

    def __init__(
        self,
        *,
        max_retention_seconds: int = DEFAULT_PAYLOAD_RETENTION_SECONDS,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if (
            isinstance(max_retention_seconds, bool)
            or not isinstance(max_retention_seconds, int)
            or max_retention_seconds < 1
        ):
            raise ValueError("max_retention_seconds must be a positive integer")
        self._max_retention_seconds = max_retention_seconds
        self._clock = clock
        self._lock = threading.RLock()
        self._index: dict[str, PayloadRef] = {}
        self._pins: dict[str, int] = {}
        self._deferred: dict[str, tuple[str, str]] = {}
        self._cleanup_pending: dict[str, tuple[PayloadRef, str, str]] = {}
        self._tombstones: OrderedDict[str, str] = OrderedDict()
        self._expired_unreported: list[PayloadRef] = []
        self._receipts: deque[PayloadReceipt] = deque(maxlen=_MAX_RECEIPTS)
        self._counts: Counter[str] = Counter()

    @property
    def max_retention_seconds(self) -> int:
        """Return the configured retention ceiling in seconds."""

        return self._max_retention_seconds

    def create(
        self,
        kind: str,
        job_id: str,
        body: Mapping[str, Any],
        *,
        retention_seconds: int,
        now: Optional[datetime] = None,
    ) -> PayloadRef:
        """Retain one immutable payload and return its opaque reference.

        Args:
            kind: ``"input"`` or ``"terminal_event"``.
            job_id: Opaque identifier of the owning job.
            body: JSON-serializable mapping. It is serialized once, so later
                changes to the caller's object never reach the payload.
            retention_seconds: Retention period, at most
                :attr:`max_retention_seconds`.
            now: Creation time; defaults to the store clock.

        Returns:
            The new payload reference.

        Raises:
            JobPayloadError: For an invalid body, an out-of-range retention, or
                a failed write. A failed write leaves no payload behind.
            EncryptionError: If the key provider cannot seal the payload.
        """

        kind = _require_kind(kind)
        job_id = _require_job_id(job_id)
        if (
            isinstance(retention_seconds, bool)
            or not isinstance(retention_seconds, int)
            or not 1 <= retention_seconds <= self._max_retention_seconds
        ):
            raise JobPayloadError("payload_retention_invalid")
        if not isinstance(body, Mapping):
            raise JobPayloadError("payload_body_invalid")
        moment = self._clock() if now is None else now
        ref = "jp_" + secrets.token_hex(16)
        entry = PayloadRef(
            ref=ref,
            kind=kind,
            job_id=job_id,
            created_at=_isoformat(moment),
            expires_at=_isoformat(moment + timedelta(seconds=retention_seconds)),
        )
        try:
            plaintext = _canonical_bytes(
                {
                    "schema_version": JOB_PAYLOAD_SCHEMA_VERSION,
                    "ref": ref,
                    "kind": kind,
                    "job_id": job_id,
                    "body": dict(body),
                }
            )
        except (TypeError, ValueError):
            raise JobPayloadError("payload_body_invalid") from None
        with self._lock:
            stored = self._store(entry, plaintext)
            self._index[ref] = stored
            self._emit(RECEIPT_CREATED, stored, reason=PAYLOAD_STATE_RETAINED)
            return stored

    def read(self, ref: str, *, job_id: str, kind: str) -> dict[str, Any]:
        """Authenticate and return a fresh copy of one retained payload body.

        Args:
            ref: Opaque payload reference.
            job_id: Job the payload must belong to.
            kind: Payload kind the payload must have.

        Returns:
            A new mapping on every call; mutating it never changes the payload.

        Raises:
            JobPayloadError: If the reference is invalid, unknown, deleted,
                expired, or damaged.
            EncryptionError: If the key is unavailable or the envelope fails
                authentication. Failed reads never change stored state.
        """

        ref = _require_ref(ref)
        kind = _require_kind(kind)
        job_id = _require_job_id(job_id)
        with self._lock:
            entry = self._index.get(ref)
            if entry is None:
                raise self._refusal_unknown(ref, kind)
            if entry.job_id != job_id or entry.kind != kind:
                raise self._refusal(entry, "payload_missing")
            if ref in self._deferred:
                raise self._refusal(entry, _state_code(self._deferred[ref][0]))
            if self._is_expired(entry):
                self._expire_locked(entry)
                raise self._refusal(entry, "payload_expired")
            try:
                plaintext = self._fetch(entry)
                body = _decode_document(plaintext, entry)
            except EncryptionError as exc:
                self._emit(RECEIPT_READ_REFUSED, entry, reason=exc.code)
                raise
            self._emit(RECEIPT_READ, entry, reason="authorized")
            return body

    def delete(self, ref: str, *, reason: str) -> None:
        """Delete one payload, or schedule its deletion while it is pinned.

        Deleting an unknown or already removed reference is a no-op. The
        reference is refused for reads from this call onward.

        Args:
            ref: Opaque payload reference.
            reason: Controlled reason code, for example ``"consumed"``.
        """

        ref = _require_ref(ref)
        reason = _require_reason(reason)
        with self._lock:
            entry = self._index.get(ref)
            if entry is None:
                return
            if self._pins.get(ref):
                self._deferred[ref] = (PAYLOAD_STATE_DELETED, reason)
                return
            self._remove_locked(entry, PAYLOAD_STATE_DELETED, reason)

    @contextmanager
    def pinned(self, ref: str) -> Iterator[PayloadRef]:
        """Hold a payload so cleanup cannot remove it while it is in use.

        Expiry and deletion requested during the pin take effect when the last
        pin is released. Reads remain bounded by the retention deadline.

        Args:
            ref: Opaque payload reference.

        Yields:
            The pinned payload reference.

        Raises:
            JobPayloadError: If the payload is unknown, deleted, or expired.
        """

        ref = _require_ref(ref)
        with self._lock:
            entry = self._index.get(ref)
            if entry is None:
                raise self._refusal_unknown(ref, "unknown")
            if ref in self._deferred:
                raise self._refusal(entry, _state_code(self._deferred[ref][0]))
            if self._is_expired(entry):
                self._expire_locked(entry)
                raise self._refusal(entry, "payload_expired")
            self._pins[ref] = self._pins.get(ref, 0) + 1
        try:
            yield entry
        finally:
            with self._lock:
                remaining = self._pins.get(ref, 1) - 1
                if remaining > 0:
                    self._pins[ref] = remaining
                else:
                    self._pins.pop(ref, None)
                    deferred = self._deferred.pop(ref, None)
                    current = self._index.get(ref)
                    if current is not None:
                        if deferred is not None:
                            self._remove_locked(current, *deferred)
                        elif self._is_expired(current):
                            self._expire_locked(current)

    def sweep_expired(self) -> tuple[PayloadRef, ...]:
        """Remove unpinned payloads past their deadline and retry cleanup.

        Returns:
            Every payload that expired since the previous sweep, including
            payloads expired by a read or by releasing a pin.
        """

        with self._lock:
            self._retry_cleanup_locked()
            now = self._clock()
            for entry in list(self._index.values()):
                if self._is_expired(entry, now):
                    self._expire_locked(entry)
            expired = tuple(self._expired_unreported)
            self._expired_unreported.clear()
            return expired

    def recover(self, live_refs: Iterable[str]) -> dict[str, int]:
        """Reconcile retained payloads with the job records after a restart.

        Payloads that no job record still references as retained are orphans
        (for example, from a crash between payload and record writes) and are
        deleted, as are crash remnants and expired payloads.

        Args:
            live_refs: References that job records still hold as retained.

        Returns:
            Value-free counts of retained, removed, and failed payloads.
        """

        live = {ref for ref in live_refs if isinstance(ref, str)}
        with self._lock:
            counts = Counter(self._recover_storage_locked(live))
            for entry in list(self._index.values()):
                if self._pins.get(entry.ref):
                    continue
                if self._is_expired(entry):
                    self._expire_locked(entry)
                    counts["expired_removed"] += 1
                elif entry.ref not in live:
                    self._remove_locked(entry, PAYLOAD_STATE_DELETED, REASON_ORPHAN)
                    counts["orphans_removed"] += 1
            counts["retained"] = len(self._index)
            counts["cleanup_failed"] = len(self._cleanup_pending)
            return dict(sorted(counts.items()))

    def contains(self, ref: str) -> bool:
        """Return whether ``ref`` is currently retained and readable."""

        with self._lock:
            entry = self._index.get(ref)
            return (
                entry is not None
                and ref not in self._deferred
                and not self._is_expired(entry)
            )

    def retained_count(self) -> int:
        """Return how many payloads the store currently retains."""

        with self._lock:
            return len(self._index)

    def receipts(self) -> tuple[PayloadReceipt, ...]:
        """Return the most recent lifecycle receipts, oldest first."""

        with self._lock:
            return tuple(self._receipts)

    def lifecycle_evidence(self) -> dict[str, Any]:
        """Return value-free lifecycle evidence for this store.

        Returns:
            Schema version, mode, retention ceiling, retained and pinned
            counts, per-event counts, and recent receipts. No content, keys,
            identifiers beyond opaque references, or paths are included.
        """

        with self._lock:
            return {
                "schema_version": JOB_PAYLOAD_SCHEMA_VERSION,
                "mode": self.mode,
                "max_retention_seconds": self._max_retention_seconds,
                "retained_count": len(self._index),
                "pinned_count": len(self._pins),
                "cleanup_pending_count": len(self._cleanup_pending),
                "event_counts": dict(sorted(self._counts.items())),
                "receipts": [receipt.to_dict() for receipt in self._receipts],
            }

    # Storage primitives -------------------------------------------------

    def _store(self, entry: PayloadRef, plaintext: bytes) -> PayloadRef:
        raise NotImplementedError

    def _fetch(self, entry: PayloadRef) -> bytes:
        raise NotImplementedError

    def _remove(self, ref: str) -> None:
        raise NotImplementedError

    def _recover_storage_locked(self, live: set[str]) -> dict[str, int]:
        del live
        return {}

    # Lifecycle helpers --------------------------------------------------

    def _is_expired(self, entry: PayloadRef, now: Optional[datetime] = None) -> bool:
        deadline = _parse_timestamp(entry.expires_at)
        moment = self._clock() if now is None else now
        return deadline is None or moment >= deadline

    def _expire_locked(self, entry: PayloadRef) -> None:
        if entry.ref not in self._index:
            return
        if self._pins.get(entry.ref):
            self._deferred.setdefault(
                entry.ref, (PAYLOAD_STATE_EXPIRED, REASON_RETENTION_ELAPSED)
            )
            return
        self._remove_locked(entry, PAYLOAD_STATE_EXPIRED, REASON_RETENTION_ELAPSED)

    def _remove_locked(self, entry: PayloadRef, state: str, reason: str) -> None:
        self._index.pop(entry.ref, None)
        self._deferred.pop(entry.ref, None)
        self._remember(entry.ref, state)
        if state == PAYLOAD_STATE_EXPIRED:
            self._expired_unreported.append(entry)
        try:
            self._remove(entry.ref)
        except OSError:
            self._cleanup_pending[entry.ref] = (entry, state, reason)
            self._emit(RECEIPT_CLEANUP_FAILED, entry, reason="storage_error")
            return
        self._cleanup_pending.pop(entry.ref, None)
        event = RECEIPT_EXPIRED if state == PAYLOAD_STATE_EXPIRED else RECEIPT_DELETED
        self._emit(event, entry, reason=reason)

    def _retry_cleanup_locked(self) -> None:
        for ref, (entry, state, reason) in list(self._cleanup_pending.items()):
            try:
                self._remove(ref)
            except OSError:
                continue
            del self._cleanup_pending[ref]
            event = (
                RECEIPT_EXPIRED if state == PAYLOAD_STATE_EXPIRED else RECEIPT_DELETED
            )
            self._emit(event, entry, reason=reason)

    def _remember(self, ref: str, state: str) -> None:
        self._tombstones[ref] = state
        self._tombstones.move_to_end(ref)
        while len(self._tombstones) > _MAX_TOMBSTONES:
            self._tombstones.popitem(last=False)

    def _refusal_unknown(self, ref: str, kind: str) -> JobPayloadError:
        state = self._tombstones.get(ref)
        code = "payload_missing" if state is None else _state_code(state)
        self._emit_raw(RECEIPT_READ_REFUSED, ref, kind, reason=code, digest=None)
        return JobPayloadError(code)

    def _refusal(self, entry: PayloadRef, code: str) -> JobPayloadError:
        self._emit(RECEIPT_READ_REFUSED, entry, reason=code)
        return JobPayloadError(code)

    def _emit(self, event: str, entry: PayloadRef, *, reason: str) -> None:
        self._emit_raw(event, entry.ref, entry.kind, reason=reason, digest=entry.digest)

    def _emit_raw(
        self,
        event: str,
        ref: str,
        kind: str,
        *,
        reason: Optional[str],
        digest: Optional[str],
    ) -> None:
        receipt = PayloadReceipt(
            event=event,
            ref=ref,
            kind=kind if kind in PAYLOAD_KINDS else "unknown",
            at=_isoformat(self._clock()),
            reason=reason if reason and _CODE_RE.fullmatch(reason) else None,
            digest=digest if digest and _DIGEST_RE.fullmatch(digest) else None,
        )
        self._receipts.append(receipt)
        self._counts[event] += 1

    def __repr__(self) -> str:
        with self._lock:
            return f"{type(self).__name__}(retained={len(self._index)})"


def _state_code(state: str) -> str:
    if state == PAYLOAD_STATE_EXPIRED:
        return "payload_expired"
    if state == PAYLOAD_STATE_DELETED:
        return "payload_deleted"
    return "payload_missing"


def _decode_document(plaintext: bytes, entry: PayloadRef) -> dict[str, Any]:
    try:
        document = json.loads(plaintext.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        raise JobPayloadError("payload_damaged") from None
    if (
        not isinstance(document, dict)
        or set(document) != {"schema_version", "ref", "kind", "job_id", "body"}
        or document["schema_version"] != entry.schema_version
        or document["ref"] != entry.ref
        or document["kind"] != entry.kind
        or document["job_id"] != entry.job_id
        or not isinstance(document["body"], dict)
    ):
        raise JobPayloadError("payload_damaged")
    return document["body"]


class MemoryJobPayloadStore(JobPayloadStore):
    """Process-local payload store that never writes payloads to disk.

    Payload bytes live only in this object's memory and are dropped as soon as
    they are deleted or expire. Use it when durable retention is not
    requested; payloads do not survive a restart.

    Args:
        max_retention_seconds: Upper bound for any payload's retention.
        clock: Source of timezone-aware UTC timestamps.
    """

    mode = PAYLOAD_MODE_MEMORY
    retains_failed_deliveries = False

    def __init__(
        self,
        *,
        max_retention_seconds: int = DEFAULT_PAYLOAD_RETENTION_SECONDS,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        super().__init__(max_retention_seconds=max_retention_seconds, clock=clock)
        self._blobs: dict[str, bytes] = {}

    def _store(self, entry: PayloadRef, plaintext: bytes) -> PayloadRef:
        self._blobs[entry.ref] = bytes(plaintext)
        return entry

    def _fetch(self, entry: PayloadRef) -> bytes:
        blob = self._blobs.get(entry.ref)
        if blob is None:
            raise JobPayloadError("payload_missing")
        return blob

    def _remove(self, ref: str) -> None:
        self._blobs.pop(ref, None)


class EncryptedJobPayloadStore(JobPayloadStore):
    """Local-disk payload store sealed with a caller-supplied key provider.

    Each payload is one JSON file named after its opaque reference. The file
    holds cleartext lifecycle metadata (reference, kind, job, timestamps, and
    ciphertext digest) next to an AEAD envelope whose associated data binds
    that metadata, so a sweep can expire payloads without any key while a
    tampered deadline or a payload copied to another job fails authentication.
    Writes go to a temporary file holding the same ciphertext, are flushed to
    disk, and are published with an atomic rename.

    Args:
        directory: Directory that holds payload files; created with mode
            ``0o700`` on first write.
        key_provider: Caller-supplied key provider. OpenMed never persists or
            caches its key material.
        max_retention_seconds: Upper bound for any payload's retention.
        clock: Source of timezone-aware UTC timestamps.
        purpose: Key purpose label used for every payload.

    Raises:
        TypeError: If ``key_provider`` does not implement the contract.
    """

    mode = PAYLOAD_MODE_ENCRYPTED
    retains_failed_deliveries = True

    def __init__(
        self,
        directory: Path,
        *,
        key_provider: KeyProvider,
        max_retention_seconds: int = DEFAULT_PAYLOAD_RETENTION_SECONDS,
        clock: Callable[[], datetime] = _utc_now,
        purpose: str = JOB_PAYLOAD_KEY_PURPOSE,
    ) -> None:
        super().__init__(max_retention_seconds=max_retention_seconds, clock=clock)
        self._cipher = EnvelopeCipher(key_provider, purpose=purpose)
        self._directory = Path(directory)
        self._remnants: set[str] = set()
        self._damaged: set[str] = set()
        self._load_index()

    @property
    def directory(self) -> Path:
        """Return the payload directory (never included in records or receipts)."""

        return self._directory

    def read(self, ref: str, *, job_id: str, kind: str) -> dict[str, Any]:
        """Return one payload body, refusing damaged files before decryption."""

        with self._lock:
            if isinstance(ref, str) and ref in self._damaged:
                self._emit_raw(
                    RECEIPT_READ_REFUSED,
                    ref,
                    kind if isinstance(kind, str) else "unknown",
                    reason="payload_damaged",
                    digest=None,
                )
                raise JobPayloadError("payload_damaged")
            return super().read(ref, job_id=job_id, kind=kind)

    def _path(self, ref: str) -> Path:
        return self._directory / f"{_require_ref(ref)}{_PAYLOAD_SUFFIX}"

    def _store(self, entry: PayloadRef, plaintext: bytes) -> PayloadRef:
        envelope = self._cipher.seal(plaintext, context=entry.context())
        stored = replace(entry, digest=_sha256_digest(envelope))
        document = {
            "format": JOB_PAYLOAD_FILE_FORMAT,
            "schema_version": stored.schema_version,
            "ref": stored.ref,
            "kind": stored.kind,
            "job_id": stored.job_id,
            "created_at": stored.created_at,
            "expires_at": stored.expires_at,
            "digest": stored.digest,
            "envelope": envelope_to_text(envelope),
        }
        self._write_atomic(stored.ref, _canonical_bytes(document) + b"\n")
        return stored

    def _write_atomic(self, ref: str, data: bytes) -> None:
        temp_name: Optional[str] = None
        failure: Optional[str] = None
        try:
            self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{ref}.",
                suffix=".tmp",
                dir=str(self._directory),
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self._path(ref))
            temp_name = None
            _fsync_directory(self._directory)
        except OSError as exc:
            failure = (
                "payload_storage_full"
                if exc.errno in {errno.ENOSPC, errno.EDQUOT}
                else "payload_write_failed"
            )
        finally:
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
        if failure is not None:
            raise JobPayloadError(failure)

    def _fetch(self, entry: PayloadRef) -> bytes:
        try:
            raw = self._path(entry.ref).read_bytes()
        except FileNotFoundError:
            raise JobPayloadError("payload_missing") from None
        except OSError:
            raise JobPayloadError("payload_read_failed") from None
        document = _parse_payload_file(raw)
        if document is None or _entry_from_document(document) != entry:
            raise JobPayloadError("payload_damaged")
        try:
            envelope = envelope_from_text(document["envelope"])
        except EncryptionError:
            raise JobPayloadError("payload_damaged") from None
        if _sha256_digest(envelope) != entry.digest:
            raise JobPayloadError("payload_damaged")
        return self._cipher.open(envelope, context=entry.context())

    def _remove(self, ref: str) -> None:
        self._path(ref).unlink(missing_ok=True)
        self._damaged.discard(ref)

    def _load_index(self) -> None:
        try:
            names = sorted(entry.name for entry in os.scandir(self._directory))
        except FileNotFoundError:
            return
        except OSError:
            raise JobPayloadError("payload_read_failed") from None
        for name in names:
            remnant = _TEMP_NAME_RE.fullmatch(name)
            if remnant is not None:
                self._remnants.add(name)
                continue
            match = _PAYLOAD_NAME_RE.fullmatch(name)
            if match is None:
                continue
            ref = match.group(1)
            try:
                raw = (self._directory / name).read_bytes()
            except OSError:
                self._damaged.add(ref)
                continue
            document = _parse_payload_file(raw)
            entry = None if document is None else _entry_from_document(document)
            if entry is None or entry.ref != ref:
                self._damaged.add(ref)
                continue
            self._index[ref] = entry

    def _recover_storage_locked(self, live: set[str]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for name in sorted(self._remnants):
            match = _TEMP_NAME_RE.fullmatch(name)
            if match is None:
                continue
            ref = match.group(1)
            try:
                (self._directory / name).unlink(missing_ok=True)
            except OSError:
                self._emit_raw(
                    RECEIPT_CLEANUP_FAILED,
                    ref,
                    "unknown",
                    reason="storage_error",
                    digest=None,
                )
                counts["remnants_failed"] += 1
                continue
            self._emit_raw(
                RECEIPT_DELETED,
                ref,
                "unknown",
                reason=REASON_CRASH_REMNANT,
                digest=None,
            )
            counts["remnants_removed"] += 1
        self._remnants.clear()
        for ref in sorted(self._damaged - live):
            try:
                self._path(ref).unlink(missing_ok=True)
            except OSError:
                counts["damaged_failed"] += 1
                continue
            self._damaged.discard(ref)
            self._remember(ref, PAYLOAD_STATE_DELETED)
            self._emit_raw(
                RECEIPT_DELETED, ref, "unknown", reason=REASON_ORPHAN, digest=None
            )
            counts["orphans_removed"] += 1
        counts["damaged_retained"] = len(self._damaged)
        return dict(counts)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"{type(self).__name__}(retained={len(self._index)}, "
                f"purpose={self._cipher.purpose!r})"
            )


def _parse_payload_file(raw: bytes) -> Optional[dict[str, Any]]:
    try:
        document = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(document, dict) or set(document) != _FILE_FIELDS:
        return None
    if document["format"] != JOB_PAYLOAD_FILE_FORMAT or not isinstance(
        document["envelope"], str
    ):
        return None
    return document


def _entry_from_document(document: Mapping[str, Any]) -> Optional[PayloadRef]:
    schema_version = document.get("schema_version")
    ref = document.get("ref")
    kind = document.get("kind")
    job_id = document.get("job_id")
    digest = document.get("digest")
    created_at = document.get("created_at")
    expires_at = document.get("expires_at")
    if (
        schema_version != JOB_PAYLOAD_SCHEMA_VERSION
        or isinstance(schema_version, bool)
        or not isinstance(ref, str)
        or _REF_RE.fullmatch(ref) is None
        or kind not in PAYLOAD_KINDS
        or not isinstance(job_id, str)
        or _JOB_ID_RE.fullmatch(job_id) is None
        or not isinstance(digest, str)
        or _DIGEST_RE.fullmatch(digest) is None
        or _parse_timestamp(created_at) is None
        or _parse_timestamp(expires_at) is None
    ):
        return None
    assert isinstance(created_at, str) and isinstance(expires_at, str)
    return PayloadRef(
        ref=ref,
        kind=str(kind),
        job_id=job_id,
        created_at=created_at,
        expires_at=expires_at,
        digest=digest,
        schema_version=JOB_PAYLOAD_SCHEMA_VERSION,
    )


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":  # pragma: no cover - directories cannot be opened on Windows
        return
    try:
        descriptor = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
