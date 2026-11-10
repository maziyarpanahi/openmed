"""Append-only local persistence for terminology mappings and review items."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openmed.clinical.journey_contracts import canonical_digest, canonical_json
from openmed.structured.store import StoreResult, StoreState

from .resolution import (
    TERMINOLOGY_RESOLUTION_COMPATIBILITY_POLICY,
    TERMINOLOGY_RESOLUTION_SCHEMA_VERSION,
    TerminologyMappingResult,
    TerminologyResolutionError,
)

_DIGEST_RE = re.compile(r"^(?:sha256|hmac-sha256):[0-9a-f]{64}$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|\+00:00)$"
)
_REVIEW_STATES = frozenset({"ambiguous", "unmapped", "rejected"})


class TerminologyMappingStoreError(RuntimeError):
    """Base error for safe terminology-store lifecycle failures."""


@dataclass(frozen=True, slots=True)
class TerminologyReviewItem:
    """PHI-safe queue item for a mapping that needs explicit review."""

    queue_id: str
    mapping_id: str
    source_digest: str
    state: str
    candidate_count: int
    reason_code: str
    vocabulary: str
    vocabulary_version: str
    snapshot_digest: str
    created_at: str
    status: str = "open"
    schema_version: str = TERMINOLOGY_RESOLUTION_SCHEMA_VERSION
    compatibility_policy: str = TERMINOLOGY_RESOLUTION_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        for value in (
            self.queue_id,
            self.mapping_id,
            self.source_digest,
            self.snapshot_digest,
        ):
            if _DIGEST_RE.fullmatch(value) is None:
                raise TerminologyResolutionError("review identifiers must be digests")
        if self.state not in _REVIEW_STATES:
            raise TerminologyResolutionError("only non-mapped states enter review")
        if type(self.candidate_count) is not int or self.candidate_count < 0:
            raise TerminologyResolutionError("candidate_count must be non-negative")
        if self.status != "open":
            raise TerminologyResolutionError("new review item status must be open")
        if _TIMESTAMP_RE.fullmatch(self.created_at) is None:
            raise TerminologyResolutionError("created_at must be a UTC timestamp")

    @classmethod
    def from_result(
        cls,
        result: TerminologyMappingResult,
        *,
        created_at: str,
    ) -> "TerminologyReviewItem":
        """Build a queue item while retaining only safe result metadata."""

        if not result.review_required:
            raise TerminologyResolutionError("mapped results do not require review")
        queue_id = canonical_digest(
            {
                "mapping_id": result.mapping_id,
                "state": result.state,
                "vocabulary_version": result.snapshot.version,
            }
        )
        return cls(
            queue_id=queue_id,
            mapping_id=result.mapping_id,
            source_digest=result.source_digest,
            state=result.state,
            candidate_count=len(result.candidates),
            reason_code=result.reason_code,
            vocabulary=result.snapshot.vocabulary,
            vocabulary_version=result.snapshot.version,
            snapshot_digest=result.snapshot.digest,
            created_at=created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, source-free queue metadata."""

        return {
            "candidate_count": self.candidate_count,
            "compatibility_policy": self.compatibility_policy,
            "created_at": self.created_at,
            "mapping_id": self.mapping_id,
            "queue_id": self.queue_id,
            "reason_code": self.reason_code,
            "schema_version": self.schema_version,
            "snapshot_digest": self.snapshot_digest,
            "source_digest": self.source_digest,
            "state": self.state,
            "status": self.status,
            "vocabulary": self.vocabulary,
            "vocabulary_version": self.vocabulary_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TerminologyReviewItem":
        """Parse a serialized queue item."""

        try:
            return cls(
                queue_id=str(payload["queue_id"]),
                mapping_id=str(payload["mapping_id"]),
                source_digest=str(payload["source_digest"]),
                state=str(payload["state"]),
                candidate_count=int(payload["candidate_count"]),
                reason_code=str(payload["reason_code"]),
                vocabulary=str(payload["vocabulary"]),
                vocabulary_version=str(payload["vocabulary_version"]),
                snapshot_digest=str(payload["snapshot_digest"]),
                created_at=str(payload["created_at"]),
                status=str(payload["status"]),
                schema_version=str(payload["schema_version"]),
                compatibility_policy=str(payload["compatibility_policy"]),
            )
        except (KeyError, TypeError, ValueError):
            raise TerminologyResolutionError("invalid review item") from None


class SQLiteTerminologyMappingStore:
    """Append-only SQLite history with a visible non-success queue."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._closed = False
        self._prepare_path()
        try:
            self._connection = sqlite3.connect(
                self.path, isolation_level=None, check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
            os.chmod(self.path, 0o600)
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            self._migrate()
        except (OSError, sqlite3.Error) as exc:
            self._close_after_failed_open()
            raise TerminologyMappingStoreError(
                "terminology mapping store cannot be initialized"
            ) from exc

    @classmethod
    def open(cls, path: str | Path) -> StoreResult["SQLiteTerminologyMappingStore"]:
        """Open a store without leaking platform error details."""

        try:
            return StoreResult.success(cls(path))
        except TerminologyMappingStoreError:
            return StoreResult.outcome(StoreState.FAILURE, "mapping_store_open_failed")

    def __enter__(self) -> "SQLiteTerminologyMappingStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the store idempotently."""

        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def record(
        self,
        result: TerminologyMappingResult,
        *,
        recorded_at: str | datetime | None = None,
    ) -> StoreResult[TerminologyMappingResult]:
        """Append one immutable mapping and queue non-success states."""

        timestamp = _timestamp(recorded_at)
        payload_json = result.to_json()
        payload_hash = canonical_digest(json.loads(payload_json))
        review = (
            TerminologyReviewItem.from_result(result, created_at=timestamp)
            if result.review_required
            else None
        )
        with self._lock:
            if self._closed:
                return StoreResult.outcome(StoreState.FAILURE, "mapping_store_closed")
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                existing = self._connection.execute(
                    "SELECT payload_hash FROM terminology_mappings WHERE mapping_id = ?",
                    (result.mapping_id,),
                ).fetchone()
                if existing is not None:
                    self._connection.execute("ROLLBACK")
                    if str(existing["payload_hash"]) != payload_hash:
                        return StoreResult.outcome(
                            StoreState.CONFLICT, "mapping_identity_conflict"
                        )
                    return StoreResult.success(result, created=False)
                self._connection.execute(
                    """
                    INSERT INTO terminology_mappings (
                        mapping_id, source_digest, snapshot_digest,
                        vocabulary, vocabulary_version, state, recorded_at,
                        payload_hash, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.mapping_id,
                        result.source_digest,
                        result.snapshot.digest,
                        result.snapshot.vocabulary,
                        result.snapshot.version,
                        result.state,
                        timestamp,
                        payload_hash,
                        payload_json,
                    ),
                )
                if review is not None:
                    review_json = canonical_json(review.to_dict())
                    self._connection.execute(
                        """
                        INSERT INTO terminology_review_queue (
                            queue_id, mapping_id, source_digest, state,
                            created_at, payload_hash, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            review.queue_id,
                            review.mapping_id,
                            review.source_digest,
                            review.state,
                            review.created_at,
                            canonical_digest(review.to_dict()),
                            review_json,
                        ),
                    )
                self._connection.execute("COMMIT")
                return StoreResult.success(result, created=True)
            except sqlite3.IntegrityError:
                self._rollback()
                return StoreResult.outcome(
                    StoreState.CONFLICT, "mapping_write_conflict"
                )
            except sqlite3.Error:
                self._rollback()
                return StoreResult.outcome(StoreState.FAILURE, "mapping_write_failed")

    def history(self, source_digest: str) -> StoreResult[tuple[dict[str, Any], ...]]:
        """Read every snapshot-specific mapping for a source digest."""

        if _DIGEST_RE.fullmatch(source_digest) is None:
            return StoreResult.outcome(StoreState.FAILURE, "source_digest_invalid")
        with self._lock:
            if self._closed:
                return StoreResult.outcome(StoreState.FAILURE, "mapping_store_closed")
            try:
                rows = self._connection.execute(
                    """
                    SELECT payload_hash, payload_json
                    FROM terminology_mappings
                    WHERE source_digest = ?
                    ORDER BY recorded_at, mapping_id
                    """,
                    (source_digest,),
                ).fetchall()
            except sqlite3.Error:
                return StoreResult.outcome(StoreState.FAILURE, "mapping_read_failed")
        if not rows:
            return StoreResult.outcome(StoreState.UNKNOWN, "mapping_not_found")
        payloads: list[dict[str, Any]] = []
        for row in rows:
            verified = _verified_payload(row)
            if verified is None:
                return StoreResult.outcome(
                    StoreState.FAILURE, "mapping_integrity_failed"
                )
            payloads.append(verified)
        return StoreResult.success(tuple(payloads))

    def review_queue(
        self,
        *,
        states: set[str] | frozenset[str] | None = None,
    ) -> StoreResult[tuple[TerminologyReviewItem, ...]]:
        """Return the visible queue in stable creation order."""

        selected = frozenset(states or _REVIEW_STATES)
        if not selected or not selected <= _REVIEW_STATES:
            return StoreResult.outcome(StoreState.FAILURE, "review_state_invalid")
        placeholders = ",".join("?" for _ in selected)
        with self._lock:
            if self._closed:
                return StoreResult.outcome(StoreState.FAILURE, "mapping_store_closed")
            try:
                rows = self._connection.execute(
                    f"""
                    SELECT payload_hash, payload_json
                    FROM terminology_review_queue
                    WHERE state IN ({placeholders})
                    ORDER BY created_at, queue_id
                    """,  # noqa: S608 - placeholders are fixed from validated states
                    tuple(sorted(selected)),
                ).fetchall()
            except sqlite3.Error:
                return StoreResult.outcome(
                    StoreState.FAILURE, "review_queue_read_failed"
                )
        items: list[TerminologyReviewItem] = []
        for row in rows:
            payload = _verified_payload(row)
            if payload is None:
                return StoreResult.outcome(
                    StoreState.FAILURE, "review_queue_integrity_failed"
                )
            try:
                items.append(TerminologyReviewItem.from_dict(payload))
            except TerminologyResolutionError:
                return StoreResult.outcome(
                    StoreState.FAILURE, "review_queue_integrity_failed"
                )
        return StoreResult.success(tuple(items))

    def _prepare_path(self) -> None:
        try:
            if self.path.is_symlink():
                raise TerminologyMappingStoreError("mapping store path is unsafe")
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)
        except OSError as exc:
            raise TerminologyMappingStoreError(
                "mapping store path cannot be prepared"
            ) from exc

    def _migrate(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS terminology_mappings (
                mapping_id TEXT PRIMARY KEY,
                source_digest TEXT NOT NULL,
                snapshot_digest TEXT NOT NULL,
                vocabulary TEXT NOT NULL,
                vocabulary_version TEXT NOT NULL,
                state TEXT NOT NULL CHECK (
                    state IN ('mapped', 'ambiguous', 'unmapped', 'rejected')
                ),
                recorded_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS terminology_mapping_history
                ON terminology_mappings(source_digest, recorded_at, mapping_id);
            CREATE TABLE IF NOT EXISTS terminology_review_queue (
                queue_id TEXT PRIMARY KEY,
                mapping_id TEXT NOT NULL UNIQUE,
                source_digest TEXT NOT NULL,
                state TEXT NOT NULL CHECK (
                    state IN ('ambiguous', 'unmapped', 'rejected')
                ),
                created_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY(mapping_id) REFERENCES terminology_mappings(mapping_id)
            );
            CREATE INDEX IF NOT EXISTS terminology_review_state
                ON terminology_review_queue(state, created_at, queue_id);
            """
        )

    def _rollback(self) -> None:
        try:
            self._connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    def _close_after_failed_open(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                pass


def _timestamp(value: str | datetime | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise TerminologyResolutionError("recorded_at must include UTC timezone")
        normalized = value.astimezone(timezone.utc).isoformat()
    elif isinstance(value, str):
        normalized = value
    else:
        raise TypeError("recorded_at must be text or datetime")
    if _TIMESTAMP_RE.fullmatch(normalized) is None:
        raise TerminologyResolutionError("recorded_at must be a UTC timestamp")
    return normalized


def _verified_payload(row: sqlite3.Row) -> dict[str, Any] | None:
    try:
        payload = json.loads(str(row["payload_json"]))
        if not isinstance(payload, dict):
            return None
        if canonical_digest(payload) != str(row["payload_hash"]):
            return None
        return payload
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


__all__ = [
    "SQLiteTerminologyMappingStore",
    "TerminologyMappingStoreError",
    "TerminologyReviewItem",
]
