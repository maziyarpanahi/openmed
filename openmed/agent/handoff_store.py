"""Local concurrency control for metadata-only reviewer handoffs.

Accepted decisions record workflow metadata, not clinical authority. No source
payload, approval token, reviewer name, or free-text judgment enters this store.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from openmed.clinical.review_state_machine import ReviewState
from openmed.structured.store.protocols import StoreResult, StoreState

from .reviewer_handoff import ReviewerHandoffError, ReviewerHandoffPacket

MAX_REVIEW_LEASE_SECONDS = 86400
HANDOFF_STORE_SCHEMA_VERSION = 1


class HandoffStoreError(ValueError):
    """Value-safe validation or storage failure with a controlled code."""


def _opaque(value: object, prefix: str) -> None:
    if type(value) is not str or not re.fullmatch(prefix + r"_[0-9a-f]{32}", value):
        raise HandoffStoreError("invalid_opaque_reference")


def _digest(value: object) -> None:
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise HandoffStoreError("invalid_digest")


def _revision(value: object, *, allow_zero: bool = False) -> None:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise HandoffStoreError("invalid_revision")


def _duration(value: object) -> None:
    if type(value) is not int or not 1 <= value <= MAX_REVIEW_LEASE_SECONDS:
        raise HandoffStoreError("invalid_lease_duration")


def _timestamp(value: object) -> None:
    if type(value) is not int and type(value) is not float:
        raise HandoffStoreError("invalid_timestamp")
    if not math.isfinite(value) or value <= 0:
        raise HandoffStoreError("invalid_timestamp")


@dataclass(frozen=True, slots=True)
class HandoffRevision:
    """Exact action, packet and ordered evidence commitments for one revision.

    Args:
        action_id: Opaque ``act_`` reference followed by 32 lowercase hex digits.
        revision: Monotonically increasing revision assigned by the store.
        action_digest: SHA-256 commitment to the caller's exact proposed action.
        handoff_digest: SHA-256 commitment to the validated packet's canonical JSON.
        evidence_digest: SHA-256 commitment to its ordered evidence references.
        expires_at: Packet expiry, expressed as UTC Unix seconds.
    """

    action_id: str
    revision: int
    action_digest: str
    handoff_digest: str
    evidence_digest: str
    expires_at: float

    def __post_init__(self) -> None:
        _opaque(self.action_id, "act")
        _revision(self.revision)
        for value in (self.action_digest, self.handoff_digest, self.evidence_digest):
            _digest(value)
        _timestamp(self.expires_at)


@dataclass(frozen=True, slots=True)
class ReviewLease:
    """A bounded review window bound to an exact handoff revision.

    Args:
        lease_id: Store-generated opaque lease reference.
        binding: Exact handoff revision being reviewed.
        reviewer_role_ref: Opaque ``role_`` reference with 32 lowercase hex digits.
        reviewer_ref: Opaque ``rev_`` reference with 32 lowercase hex digits.
        expires_at: Lease expiry, capped by the packet's expiry.
    """

    lease_id: str
    binding: HandoffRevision
    reviewer_role_ref: str
    reviewer_ref: str
    expires_at: float

    def __post_init__(self) -> None:
        _opaque(self.lease_id, "lease")
        _opaque(self.reviewer_role_ref, "role")
        _opaque(self.reviewer_ref, "rev")
        if type(self.binding) is not HandoffRevision:
            raise HandoffStoreError("invalid_binding")
        _timestamp(self.expires_at)
        if self.expires_at > self.binding.expires_at:
            raise HandoffStoreError("invalid_lease_expiry")


@dataclass(frozen=True, slots=True)
class DecisionReceipt:
    """Append-only record of a caller-supplied approved or rejected review state.

    Args:
        lease: Exact persisted lease, including revision and reviewer references.
        decision: Caller-supplied review state; not an approval token.
        recorded_at: UTC Unix seconds at the accepted transaction.
    """

    lease: ReviewLease
    decision: ReviewState
    recorded_at: float

    def __post_init__(self) -> None:
        if type(self.lease) is not ReviewLease:
            raise HandoffStoreError("invalid_lease")
        if type(self.decision) is not ReviewState or self.decision not in (
            ReviewState.APPROVED,
            ReviewState.REJECTED,
        ):
            raise HandoffStoreError("invalid_decision")
        _timestamp(self.recorded_at)
        if self.recorded_at >= self.lease.expires_at:
            raise HandoffStoreError("expired")

    @property
    def authorizes_clinical_action(self) -> bool:
        """Return False: receipt acceptance never grants clinical authority."""
        return False


class HandoffStore(Protocol):
    """Narrow local adapter contract for revisioned human-review decisions."""

    def publish(
        self,
        action_id: str,
        action_digest: str,
        packet: ReviewerHandoffPacket,
        *,
        expected_revision: int,
    ) -> StoreResult[HandoffRevision]:
        """Create or supersede a handoff using an optimistic revision."""
        ...

    def acquire(
        self,
        action_id: str,
        *,
        expected_revision: int,
        reviewer_role_ref: str,
        reviewer_ref: str,
        seconds: int = 300,
    ) -> StoreResult[ReviewLease]:
        """Acquire a bounded lease for exactly one current revision."""
        ...

    def decide(
        self,
        lease: ReviewLease,
        decision: ReviewState,
    ) -> StoreResult[DecisionReceipt]:
        """Append one decision or return an explicit controlled refusal."""
        ...

    def current(self, action_id: str) -> StoreResult[HandoffRevision]:
        """Read the current revision, refusing expired packets."""
        ...

    def receipts(self, action_id: str) -> tuple[DecisionReceipt, ...]:
        """Read historical receipts; history alone never implies current approval."""
        ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS handoff_meta (version INTEGER NOT NULL, clock REAL NOT NULL);
CREATE TABLE IF NOT EXISTS handoff_revisions (
    action_id TEXT NOT NULL, revision INTEGER NOT NULL, action_digest TEXT NOT NULL,
    handoff_digest TEXT NOT NULL, evidence_digest TEXT NOT NULL, expires_at REAL NOT NULL,
    PRIMARY KEY (action_id, revision)
);
CREATE TABLE IF NOT EXISTS handoff_leases (
    lease_id TEXT PRIMARY KEY, action_id TEXT NOT NULL, revision INTEGER NOT NULL,
    reviewer_role_ref TEXT NOT NULL, reviewer_ref TEXT NOT NULL, expires_at REAL NOT NULL,
    FOREIGN KEY (action_id, revision) REFERENCES handoff_revisions(action_id, revision)
);
CREATE TABLE IF NOT EXISTS handoff_receipts (
    lease_id TEXT PRIMARY KEY REFERENCES handoff_leases(lease_id),
    action_id TEXT NOT NULL, revision INTEGER NOT NULL, decision TEXT NOT NULL,
    recorded_at REAL NOT NULL, UNIQUE(action_id, revision),
    FOREIGN KEY (action_id, revision) REFERENCES handoff_revisions(action_id, revision)
);
"""


class SQLiteHandoffStore:
    """Durable local adapter with atomic decisions across threads and processes.

    Args:
        path: Caller-owned dedicated SQLite file, or ``:memory:`` for tests.
        clock: Injected UTC clock; defaults to the local UTC wall clock.
        max_lease_seconds: Lease ceiling, from 1 through 86400 seconds.

    The authenticated caller owns reviewer identity, action commitments and
    clinical judgment. The database must be protected from untrusted writers.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        max_lease_seconds: int = 900,
    ) -> None:
        _duration(max_lease_seconds)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._max_lease_seconds = max_lease_seconds
        self._lock = threading.RLock()
        self._db: sqlite3.Connection | None = None
        try:
            self._db = sqlite3.connect(
                str(path), isolation_level=None, check_same_thread=False, timeout=5
            )
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.executescript(_SCHEMA)
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute("SELECT version FROM handoff_meta").fetchone()
            if row is None:
                self._db.execute(
                    "INSERT INTO handoff_meta VALUES (?, ?)",
                    (HANDOFF_STORE_SCHEMA_VERSION, 0),
                )
            elif row["version"] != HANDOFF_STORE_SCHEMA_VERSION:
                raise HandoffStoreError("unsupported_store_schema")
            # Revision, lease and receipt rows are immutable at the SQL boundary.
            for table in ("handoff_revisions", "handoff_leases", "handoff_receipts"):
                for operation in ("UPDATE", "DELETE"):
                    self._db.execute(
                        f"CREATE TRIGGER IF NOT EXISTS {table}_{operation.lower()} "
                        f"BEFORE {operation} ON {table} "
                        "BEGIN SELECT RAISE(ABORT, 'append_only'); END"
                    )
            self._db.commit()
        except (sqlite3.Error, HandoffStoreError):
            if self._db is not None:
                self._db.close()
            self._db = None
            raise HandoffStoreError("store_unavailable_or_incompatible") from None

    def close(self) -> None:
        """Close the local database connection."""
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    @contextmanager
    def _transaction(self) -> Iterator[tuple[sqlite3.Connection, float]]:
        with self._lock:
            if self._db is None:
                raise HandoffStoreError("store_closed")
            db = self._db
            try:
                db.execute("BEGIN IMMEDIATE")
                try:
                    value = self._clock()
                except Exception:
                    raise HandoffStoreError("clock_unavailable") from None
                if (
                    type(value) is not datetime
                    or value.tzinfo is None
                    or value.utcoffset() != timezone.utc.utcoffset(value)
                ):
                    raise HandoffStoreError("invalid_clock")
                now = value.timestamp()
                previous = db.execute("SELECT clock FROM handoff_meta").fetchone()[0]
                if now < previous:
                    raise HandoffStoreError("clock_regressed")
                db.execute("UPDATE handoff_meta SET clock=?", (now,))
                yield db, now
                db.commit()
            except sqlite3.Error:
                db.rollback()
                raise HandoffStoreError("store_unavailable") from None
            except BaseException:
                db.rollback()
                raise

    @staticmethod
    def _current(db: sqlite3.Connection, action_id: str) -> HandoffRevision | None:
        row = db.execute(
            "SELECT * FROM handoff_revisions WHERE action_id=? "
            "ORDER BY revision DESC LIMIT 1",
            (action_id,),
        ).fetchone()
        return None if row is None else HandoffRevision(**dict(row))

    @staticmethod
    def _refusal(
        db: sqlite3.Connection,
        binding: HandoffRevision | None,
        expected_revision: int,
        now: float,
    ) -> str | None:
        if binding is None:
            return "not_found"
        if binding.revision != expected_revision:
            return "superseded"
        if binding.expires_at <= now:
            return "expired"
        if db.execute(
            "SELECT 1 FROM handoff_receipts WHERE action_id=? AND revision=?",
            (binding.action_id, binding.revision),
        ).fetchone():
            return "already_decided"
        return None

    def publish(
        self,
        action_id: str,
        action_digest: str,
        packet: ReviewerHandoffPacket,
        *,
        expected_revision: int,
    ) -> StoreResult[HandoffRevision]:
        """Publish a revision; expected_revision=0 creates a new action.

        Corrections always append a new revision, even for identical digests.
        Previous leases and decisions remain historical and cannot be reused.

        Args:
            action_id: Opaque action reference, stable across corrections.
            action_digest: SHA-256 commitment to the exact caller-owned action.
            packet: Existing validated metadata-only handoff packet.
            expected_revision: Last observed revision, or zero for creation.

        Returns:
            The new binding or a revision conflict, expiry or future-issue refusal.

        Raises:
            HandoffStoreError: For unsafe inputs, unavailable storage or clocks.
        """
        _opaque(action_id, "act")
        _digest(action_digest)
        _revision(expected_revision, allow_zero=True)
        if type(packet) is not ReviewerHandoffPacket:
            raise HandoffStoreError("invalid_packet")
        with self._transaction() as (db, now):
            # Revalidate at the transaction clock, including expiry and metadata.
            try:
                validated = ReviewerHandoffPacket.from_dict(
                    packet.to_dict(), now=datetime.fromtimestamp(now, timezone.utc)
                )
            except ReviewerHandoffError as error:
                if error.code == "expired":
                    return StoreResult.outcome(StoreState.DENIED, "expired")
                raise HandoffStoreError("invalid_packet") from None
            if validated.issued_at.timestamp() > now:
                return StoreResult.outcome(StoreState.DENIED, "not_yet_valid")
            previous = self._current(db, action_id)
            if (0 if previous is None else previous.revision) != expected_revision:
                return StoreResult.outcome(StoreState.CONFLICT, "revision_conflict")
            evidence = json.dumps(
                [ref.to_dict() for ref in validated.evidence_references],
                sort_keys=True,
                separators=(",", ":"),
            )
            binding = HandoffRevision(
                action_id,
                expected_revision + 1,
                action_digest,
                hashlib.sha256(validated.to_json().encode()).hexdigest(),
                hashlib.sha256(evidence.encode()).hexdigest(),
                validated.expires_at.timestamp(),
            )
            db.execute(
                "INSERT INTO handoff_revisions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    binding.action_id,
                    binding.revision,
                    binding.action_digest,
                    binding.handoff_digest,
                    binding.evidence_digest,
                    binding.expires_at,
                ),
            )
            return StoreResult.success(binding, created=True, revision=binding.revision)

    def current(self, action_id: str) -> StoreResult[HandoffRevision]:
        """Return the latest binding or an explicit missing/expired outcome.

        Args:
            action_id: Opaque action reference.

        Returns:
            The current binding, or a missing/expired refusal. Success does not
            imply that a decision exists or that an action is authorized.
        """
        _opaque(action_id, "act")
        with self._transaction() as (db, now):
            binding = self._current(db, action_id)
            if binding is None:
                return StoreResult.outcome(StoreState.DENIED, "not_found")
            if binding.expires_at <= now:
                return StoreResult.outcome(StoreState.DENIED, "expired")
            return StoreResult.success(binding, revision=binding.revision)

    def acquire(
        self,
        action_id: str,
        *,
        expected_revision: int,
        reviewer_role_ref: str,
        reviewer_ref: str,
        seconds: int = 300,
    ) -> StoreResult[ReviewLease]:
        """Acquire a lease; competing reviewers may review the same revision.

        Args:
            action_id: Opaque action reference.
            expected_revision: Exact current revision to review.
            reviewer_role_ref: Opaque role reference supplied by the caller.
            reviewer_ref: Opaque reviewer reference supplied by the caller.
            seconds: Positive duration within the configured lease ceiling.

        Returns:
            A persisted lease or a missing, superseded, expired or decided refusal.

        Raises:
            HandoffStoreError: For unsafe inputs or unavailable storage/clocks.
        """
        _opaque(action_id, "act")
        _opaque(reviewer_role_ref, "role")
        _opaque(reviewer_ref, "rev")
        _revision(expected_revision)
        _duration(seconds)
        if seconds > self._max_lease_seconds:
            raise HandoffStoreError("lease_duration_exceeded")
        with self._transaction() as (db, now):
            binding = self._current(db, action_id)
            refusal = self._refusal(db, binding, expected_revision, now)
            if refusal:
                return StoreResult.outcome(StoreState.CONFLICT, refusal)
            assert binding is not None
            lease = ReviewLease(
                "lease_" + uuid.uuid4().hex,
                binding,
                reviewer_role_ref,
                reviewer_ref,
                min(now + seconds, binding.expires_at),
            )
            db.execute(
                "INSERT INTO handoff_leases VALUES (?, ?, ?, ?, ?, ?)",
                (
                    lease.lease_id,
                    action_id,
                    expected_revision,
                    reviewer_role_ref,
                    reviewer_ref,
                    lease.expires_at,
                ),
            )
            return StoreResult.success(lease, created=True, revision=expected_revision)

    def decide(
        self,
        lease: ReviewLease,
        decision: ReviewState,
    ) -> StoreResult[DecisionReceipt]:
        """Accept one exact leased decision, atomically, without issuing approval.

        Args:
            lease: Exact lease returned by this database, without modifications.
            decision: Caller-supplied APPROVED or REJECTED review state.

        Returns:
            An accepted receipt or a controlled lease, expiry or revision refusal.
            A repeat submission returns already_decided, never reusable approval.

        Raises:
            HandoffStoreError: For unsafe inputs or unavailable storage/clocks.
        """
        if type(lease) is not ReviewLease or type(lease.binding) is not HandoffRevision:
            raise HandoffStoreError("invalid_lease")
        if type(decision) is not ReviewState or decision not in (
            ReviewState.APPROVED,
            ReviewState.REJECTED,
        ):
            raise HandoffStoreError("invalid_decision")
        _opaque(lease.lease_id, "lease")
        _opaque(lease.binding.action_id, "act")
        with self._transaction() as (db, now):
            row = db.execute(
                "SELECT * FROM handoff_leases WHERE lease_id=?", (lease.lease_id,)
            ).fetchone()
            if row is None:
                return StoreResult.outcome(StoreState.DENIED, "invalid_lease")
            stored_binding = db.execute(
                "SELECT * FROM handoff_revisions WHERE action_id=? AND revision=?",
                (row["action_id"], row["revision"]),
            ).fetchone()
            stored = ReviewLease(
                row["lease_id"],
                HandoffRevision(**dict(stored_binding)),
                row["reviewer_role_ref"],
                row["reviewer_ref"],
                row["expires_at"],
            )
            if lease != stored:
                return StoreResult.outcome(StoreState.DENIED, "invalid_lease")
            refusal = self._refusal(
                db, self._current(db, row["action_id"]), row["revision"], now
            )
            # Even an accepted receipt cannot make an expired lease reusable.
            if refusal in (None, "already_decided") and lease.expires_at <= now:
                return StoreResult.outcome(StoreState.DENIED, "expired")
            if refusal:
                return StoreResult.outcome(StoreState.CONFLICT, refusal)
            receipt = DecisionReceipt(lease, decision, now)
            db.execute(
                "INSERT INTO handoff_receipts VALUES (?, ?, ?, ?, ?)",
                (
                    lease.lease_id,
                    row["action_id"],
                    row["revision"],
                    decision.value,
                    now,
                ),
            )
            return StoreResult.success(receipt, created=True, revision=row["revision"])

    def receipts(self, action_id: str) -> tuple[DecisionReceipt, ...]:
        """Return append-only historical decisions in revision order.

        This includes expired and superseded revisions. Consumers must check the
        exact current binding independently; history is never reusable authority.

        Args:
            action_id: Opaque action reference.

        Returns:
            Immutable receipts ordered by revision, or an empty tuple.
        """
        _opaque(action_id, "act")
        with self._transaction() as (db, _):
            rows = db.execute(
                "SELECT r.*, l.reviewer_role_ref, l.reviewer_ref, "
                "l.expires_at AS lease_expiry FROM handoff_receipts r "
                "JOIN handoff_leases l ON l.lease_id=r.lease_id "
                "WHERE r.action_id=? ORDER BY r.revision",
                (action_id,),
            ).fetchall()
            result = []
            for row in rows:
                revision = db.execute(
                    "SELECT * FROM handoff_revisions WHERE action_id=? AND revision=?",
                    (action_id, row["revision"]),
                ).fetchone()
                lease = ReviewLease(
                    row["lease_id"],
                    HandoffRevision(**dict(revision)),
                    row["reviewer_role_ref"],
                    row["reviewer_ref"],
                    row["lease_expiry"],
                )
                result.append(
                    DecisionReceipt(
                        lease, ReviewState(row["decision"]), row["recorded_at"]
                    )
                )
            return tuple(result)
