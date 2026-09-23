"""Local, metadata-only checkpoints for ordered FHIR Subscription intake.

A claim reserves one notification before workflow dispatch. The caller commits
it only after durable downstream acceptance. Unconfirmed claims fail closed on
retry; this module cannot make external workflow side effects transactional.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class SubscriptionCheckpointError(ValueError):
    """A value-free checkpoint validation or storage error."""


@dataclass(frozen=True, slots=True)
class IntakeDecision:
    """A payload-free intake outcome for one delivery."""

    status: Literal["claimed", "duplicate", "quarantined"]
    reason: str
    sequence: int


@dataclass(frozen=True, slots=True)
class QuarantinedDelivery:
    """Metadata-only reference for an unresolved delivery."""

    event_digest: str
    sequence: int
    reason: str


def _identity(value: str, name: str) -> bytes:
    if type(value) is not str or not 1 <= len(value) <= 512:
        raise SubscriptionCheckpointError(f"{name}: invalid_identity")
    return value.encode("utf-8")


def _sequence(value: int) -> int:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise SubscriptionCheckpointError("sequence: invalid_integer")
    return value


class SubscriptionCheckpoint:
    """Checkpoint one ordered subscription stream in a local SQLite file.

    Args:
        path: Local database path. Keep it in a private directory.
        secret: Caller-managed random HMAC key, at least 32 bytes. Reuse it on
            reopening; neither it nor raw identifiers are persisted.
        window: Number of committed sequences retained for deduplication.
        initial_sequence: First expected sequence for each new subscription.
        max_gap: Largest future sequence retained in quarantine.
        max_quarantine: Maximum unresolved records per stream.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        secret: bytes,
        window: int = 128,
        initial_sequence: int = 0,
        max_gap: int = 128,
        max_quarantine: int = 512,
    ) -> None:
        if type(secret) is not bytes or len(secret) < 32:
            raise SubscriptionCheckpointError("secret: invalid_key")
        self.initial_sequence = _sequence(initial_sequence)
        for name, value in (
            ("window", window),
            ("max_gap", max_gap),
            ("max_quarantine", max_quarantine),
        ):
            if type(value) is not int or not 1 <= value <= 100_000:
                raise SubscriptionCheckpointError(f"{name}: invalid_limit")
        self._secret = secret
        self.window = window
        self.max_gap = max_gap
        self.max_quarantine = max_quarantine
        try:
            self._connection = sqlite3.connect(
                str(path), isolation_level=None, timeout=5
            )
        except sqlite3.Error:
            raise SubscriptionCheckpointError("checkpoint: unavailable") from None
        self._connection.execute("PRAGMA journal_mode=DELETE")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                name TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS streams (
                scope TEXT PRIMARY KEY,
                committed INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS deliveries (
                scope TEXT NOT NULL,
                event_key TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                notification TEXT NOT NULL,
                resource TEXT NOT NULL,
                version TEXT NOT NULL,
                state TEXT NOT NULL,
                reason TEXT NOT NULL,
                PRIMARY KEY (scope, event_key)
            );
            CREATE INDEX IF NOT EXISTS deliveries_sequence
                ON deliveries(scope, sequence);
            CREATE INDEX IF NOT EXISTS deliveries_notification
                ON deliveries(scope, notification);
            CREATE INDEX IF NOT EXISTS deliveries_resource
                ON deliveries(scope, resource, version);
            """
        )
        verifier = hmac.new(secret, b"openmed.fhir.subscription.v1", hashlib.sha256)
        expected = verifier.hexdigest()
        self._connection.execute(
            "INSERT OR IGNORE INTO metadata VALUES ('key_verifier', ?)", (expected,)
        )
        actual = self._connection.execute(
            "SELECT value FROM metadata WHERE name = 'key_verifier'"
        ).fetchone()
        if actual is None or not hmac.compare_digest(actual[0], expected):
            self._connection.close()
            raise SubscriptionCheckpointError("secret: incompatible_key")

    def __repr__(self) -> str:
        return "SubscriptionCheckpoint(<redacted>)"

    def close(self) -> None:
        """Close the local checkpoint database."""

        self._connection.close()

    def __enter__(self) -> SubscriptionCheckpoint:
        """Return this checkpoint for context-managed use."""

        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _digest(self, value: bytes) -> str:
        return hmac.new(self._secret, value, hashlib.sha256).hexdigest()

    def _keys(
        self,
        subscription_id: str,
        notification_id: str,
        resource_id: str,
        resource_version: str,
        sequence: int,
    ) -> tuple[str, str, str, str, str]:
        parts = tuple(
            _identity(value, name)
            for name, value in (
                ("subscription_id", subscription_id),
                ("notification_id", notification_id),
                ("resource_id", resource_id),
                ("resource_version", resource_version),
            )
        )
        _sequence(sequence)
        scope, notification, resource, version = (
            self._digest(label + b"\0" + part)
            for label, part in zip(
                (b"scope", b"notification", b"resource", b"version"), parts
            )
        )
        event_parts = (*parts, str(sequence).encode("ascii"))
        event = self._digest(
            b"event\0"
            + b"".join(len(part).to_bytes(4, "big") + part for part in event_parts)
        )
        return scope, notification, resource, version, event

    def _record(
        self,
        scope: str,
        event: str,
        sequence: int,
        notification: str,
        resource: str,
        version: str,
        state: str,
        reason: str,
    ) -> None:
        self._connection.execute(
            "INSERT OR IGNORE INTO deliveries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (scope, event, sequence, notification, resource, version, state, reason),
        )

    def quarantines(self, subscription_id: str) -> tuple[QuarantinedDelivery, ...]:
        """List unresolved records without disclosing delivery identifiers."""

        scope = self._digest(b"scope\0" + _identity(subscription_id, "subscription_id"))
        rows = self._connection.execute(
            "SELECT event_key, sequence, "
            "CASE WHEN state = 'pending' THEN 'unconfirmed' ELSE reason END "
            "FROM deliveries WHERE scope = ? AND state != 'committed' "
            "ORDER BY sequence, event_key",
            (scope,),
        ).fetchall()
        return tuple(QuarantinedDelivery(*row) for row in rows)

    def discard_quarantine(self, subscription_id: str, event_digest: str) -> None:
        """Discard one quarantined record after explicit operator review.

        This never commits a pending workflow or advances delivery order.
        Re-delivery is evaluated from the current checkpoint state.
        """

        scope = self._digest(b"scope\0" + _identity(subscription_id, "subscription_id"))
        if type(event_digest) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", event_digest
        ):
            raise SubscriptionCheckpointError("event_digest: invalid_digest")
        cursor = self._connection.execute(
            "DELETE FROM deliveries WHERE scope = ? AND event_key = ? "
            "AND state = 'quarantined'",
            (scope, event_digest),
        )
        if cursor.rowcount != 1:
            raise SubscriptionCheckpointError("quarantine: missing_record")

    def claim(
        self,
        *,
        subscription_id: str,
        notification_id: str,
        resource_id: str,
        resource_version: str,
        sequence: int,
        replay_gap: bool = False,
    ) -> IntakeDecision:
        """Reserve a delivery or return a deterministic duplicate/quarantine.

        Sequence numbers begin at ``initial_sequence``. A gap can be retried with ``replay_gap``
        once earlier sequences have been committed. Other quarantines require
        operator reconciliation and are never silently retried.
        """

        scope, notification, resource, version, event = self._keys(
            subscription_id,
            notification_id,
            resource_id,
            resource_version,
            sequence,
        )
        if type(replay_gap) is not bool:
            raise SubscriptionCheckpointError("replay_gap: invalid_flag")
        db = self._connection
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute(
                "INSERT OR IGNORE INTO streams VALUES (?, ?)",
                (scope, self.initial_sequence - 1),
            )
            committed = db.execute(
                "SELECT committed FROM streams WHERE scope = ?", (scope,)
            ).fetchone()[0]
            existing = db.execute(
                "SELECT state, reason FROM deliveries WHERE scope = ? AND event_key = ?",
                (scope, event),
            ).fetchone()
            if existing is not None:
                state, reason = existing
                if state == "committed":
                    decision = IntakeDecision(
                        "duplicate", "already_committed", sequence
                    )
                elif (
                    state == "quarantined"
                    and reason == "gap"
                    and replay_gap
                    and sequence == committed + 1
                    and db.execute(
                        "SELECT 1 FROM deliveries WHERE scope = ? AND sequence = ? "
                        "AND event_key != ? LIMIT 1",
                        (scope, sequence, event),
                    ).fetchone()
                    is None
                ):
                    db.execute(
                        "UPDATE deliveries SET state = 'pending', reason = '' "
                        "WHERE scope = ? AND event_key = ?",
                        (scope, event),
                    )
                    decision = IntakeDecision("claimed", "gap_replay", sequence)
                else:
                    decision = IntakeDecision(
                        "quarantined", reason or "unconfirmed", sequence
                    )
            else:
                collision = db.execute(
                    "SELECT 1 FROM deliveries WHERE scope = ? AND "
                    "(notification = ? OR sequence = ?) LIMIT 1",
                    (scope, notification, sequence),
                ).fetchone()
                seen_version = db.execute(
                    "SELECT 1 FROM deliveries WHERE scope = ? AND resource = ? "
                    "AND version = ? AND state = 'committed' LIMIT 1",
                    (scope, resource, version),
                ).fetchone()
                if collision:
                    reason = "conflict"
                elif sequence <= committed:
                    reason = "outside_window"
                elif sequence > committed + self.max_gap + 1:
                    reason = "gap_exceeds_limit"
                elif sequence > committed + 1:
                    reason = "gap"
                else:
                    reason = ""
                unresolved = db.execute(
                    "SELECT count(*) FROM deliveries WHERE scope = ? "
                    "AND state != 'committed'",
                    (scope,),
                ).fetchone()[0]
                if reason and unresolved >= self.max_quarantine:
                    decision = IntakeDecision("quarantined", "capacity", sequence)
                elif reason:
                    self._record(
                        scope,
                        event,
                        sequence,
                        notification,
                        resource,
                        version,
                        "quarantined",
                        reason,
                    )
                    decision = IntakeDecision("quarantined", reason, sequence)
                elif seen_version:
                    self._record(
                        scope,
                        event,
                        sequence,
                        notification,
                        resource,
                        version,
                        "committed",
                        "",
                    )
                    db.execute(
                        "UPDATE streams SET committed = ? WHERE scope = ?",
                        (sequence, scope),
                    )
                    db.execute(
                        "DELETE FROM deliveries WHERE scope = ? AND state = 'committed' "
                        "AND sequence <= ?",
                        (scope, sequence - self.window),
                    )
                    decision = IntakeDecision("duplicate", "resource_version", sequence)
                else:
                    self._record(
                        scope,
                        event,
                        sequence,
                        notification,
                        resource,
                        version,
                        "pending",
                        "",
                    )
                    decision = IntakeDecision("claimed", "new", sequence)
            db.execute("COMMIT")
            return decision
        except BaseException:
            db.execute("ROLLBACK")
            raise

    def commit(
        self,
        *,
        subscription_id: str,
        notification_id: str,
        resource_id: str,
        resource_version: str,
        sequence: int,
    ) -> None:
        """Mark a claim accepted after durable downstream acknowledgment."""

        scope, _, _, _, event = self._keys(
            subscription_id,
            notification_id,
            resource_id,
            resource_version,
            sequence,
        )
        db = self._connection
        db.execute("BEGIN IMMEDIATE")
        try:
            row = db.execute(
                "SELECT state FROM deliveries WHERE scope = ? AND event_key = ?",
                (scope, event),
            ).fetchone()
            current = db.execute(
                "SELECT committed FROM streams WHERE scope = ?", (scope,)
            ).fetchone()
            if (
                row is None
                or current is None
                or row[0] != "pending"
                or sequence != current[0] + 1
            ):
                raise SubscriptionCheckpointError("commit: invalid_claim")
            db.execute(
                "UPDATE deliveries SET state = 'committed' "
                "WHERE scope = ? AND event_key = ?",
                (scope, event),
            )
            db.execute(
                "UPDATE streams SET committed = ? WHERE scope = ?",
                (sequence, scope),
            )
            db.execute(
                "DELETE FROM deliveries WHERE scope = ? AND state = 'committed' "
                "AND sequence <= ?",
                (scope, sequence - self.window),
            )
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
