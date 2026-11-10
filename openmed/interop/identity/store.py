"""Durable local registry for exact identity links and review decisions."""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from openmed.clinical.journey_contracts import canonical_digest
from openmed.structured.store import StoreResult, StoreState

from .contracts import (
    IdentityContractError,
    IdentityLink,
    IdentityResolution,
    IdentityReviewDecision,
    SourceIdentityKey,
)

IDENTITY_STORE_SCHEMA_VERSION = 1

_MIGRATION_SQL = (
    """
    CREATE TABLE identity_links (
        link_id TEXT PRIMARY KEY,
        entity_type TEXT NOT NULL,
        source_fingerprint TEXT NOT NULL,
        canonical_key TEXT NOT NULL,
        active INTEGER NOT NULL,
        payload_hash TEXT NOT NULL UNIQUE,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX identity_links_source_idx
    ON identity_links(entity_type, source_fingerprint, active, canonical_key)
    """,
    """
    CREATE TABLE identity_resolutions (
        resolution_id TEXT PRIMARY KEY,
        request_digest TEXT NOT NULL,
        state TEXT NOT NULL,
        review_required INTEGER NOT NULL,
        payload_hash TEXT NOT NULL UNIQUE,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX identity_resolutions_request_idx
    ON identity_resolutions(request_digest, resolution_id)
    """,
    """
    CREATE TABLE identity_reviews (
        decision_id TEXT PRIMARY KEY,
        resolution_id TEXT NOT NULL UNIQUE
            REFERENCES identity_resolutions(resolution_id),
        action TEXT NOT NULL,
        payload_hash TEXT NOT NULL UNIQUE,
        payload_json TEXT NOT NULL
    )
    """,
)
_MIGRATION_CHECKSUM = canonical_digest(_MIGRATION_SQL)


class IdentityStoreCompatibilityError(RuntimeError):
    """Raised when persisted identity state uses an unsupported schema."""


class IdentityStoreError(RuntimeError):
    """Value-safe local identity-store lifecycle error."""


class IdentityResolutionStore:
    """SQLite registry with append-only resolutions and explicit reviews."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._closed = False
        try:
            if self.path.exists() and self.path.is_symlink():
                raise IdentityStoreError("identity store path is unsafe")
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)
            self._connection = sqlite3.connect(
                self.path,
                isolation_level=None,
                check_same_thread=False,
            )
            os.chmod(self.path, 0o600)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            self._apply_migration()
        except IdentityStoreCompatibilityError:
            self._close_failed()
            raise
        except (OSError, sqlite3.Error) as exc:
            self._close_failed()
            raise IdentityStoreError("identity store cannot be initialized") from exc

    @classmethod
    def open(cls, path: str | Path) -> StoreResult["IdentityResolutionStore"]:
        """Open with typed unsupported and failure outcomes."""

        try:
            return StoreResult.success(cls(path))
        except IdentityStoreCompatibilityError:
            return StoreResult.outcome(StoreState.UNSUPPORTED, "schema_unsupported")
        except IdentityStoreError:
            return StoreResult.outcome(StoreState.FAILURE, "identity_store_open_failed")

    def add_link(self, link: IdentityLink) -> StoreResult[IdentityLink]:
        """Persist one exact link idempotently."""

        payload_hash = canonical_digest(link.to_dict())
        with self._lock:
            try:
                existing = self._connection.execute(
                    "SELECT payload_hash FROM identity_links WHERE link_id = ?",
                    (link.link_id,),
                ).fetchone()
                if existing is not None:
                    if str(existing["payload_hash"]) == payload_hash:
                        return StoreResult.success(link, created=False)
                    return StoreResult.outcome(StoreState.CONFLICT, "link_id_conflict")
                self._connection.execute("BEGIN IMMEDIATE")
                self._insert_link(link)
                self._connection.execute("COMMIT")
                return StoreResult.success(link, created=True)
            except sqlite3.IntegrityError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                return StoreResult.outcome(
                    StoreState.CONFLICT, "identity_link_conflict"
                )
            except (IdentityContractError, sqlite3.Error):
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                return StoreResult.outcome(
                    StoreState.FAILURE, "identity_link_write_failed"
                )

    def active_links(
        self,
        source_keys: tuple[SourceIdentityKey, ...],
    ) -> StoreResult[Mapping[str, tuple[IdentityLink, ...]]]:
        """Return active exact links keyed by source-key fingerprint."""

        if not isinstance(source_keys, tuple) or not source_keys:
            return StoreResult.outcome(StoreState.FAILURE, "invalid_source_keys")
        result: dict[str, tuple[IdentityLink, ...]] = {}
        with self._lock:
            try:
                for key in source_keys:
                    if not isinstance(key, SourceIdentityKey):
                        return StoreResult.outcome(
                            StoreState.FAILURE,
                            "invalid_source_keys",
                        )
                    rows = self._connection.execute(
                        "SELECT payload_json FROM identity_links "
                        "WHERE entity_type = ? AND source_fingerprint = ? "
                        "AND active = 1 ORDER BY canonical_key, link_id",
                        (key.entity_type, key.fingerprint),
                    ).fetchall()
                    result[key.fingerprint] = tuple(
                        IdentityLink.from_json(str(row["payload_json"])) for row in rows
                    )
            except (IdentityContractError, sqlite3.Error):
                return StoreResult.outcome(
                    StoreState.FAILURE, "identity_link_read_failed"
                )
        return StoreResult.success(result)

    def save_resolution(
        self,
        resolution: IdentityResolution,
    ) -> StoreResult[IdentityResolution]:
        """Persist one deterministic outcome idempotently."""

        payload_hash = canonical_digest(resolution.to_dict())
        with self._lock:
            try:
                existing = self._connection.execute(
                    "SELECT payload_hash, payload_json FROM identity_resolutions "
                    "WHERE resolution_id = ?",
                    (resolution.resolution_id,),
                ).fetchone()
                if existing is not None:
                    if str(existing["payload_hash"]) == payload_hash:
                        return StoreResult.success(resolution, created=False)
                    persisted = IdentityResolution.from_json(
                        str(existing["payload_json"])
                    )
                    if _same_resolution_replay(persisted, resolution):
                        return StoreResult.success(persisted, created=False)
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "resolution_id_conflict",
                    )
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute(
                    "INSERT INTO identity_resolutions("
                    "resolution_id, request_digest, state, review_required, "
                    "payload_hash, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        resolution.resolution_id,
                        resolution.request_digest,
                        resolution.state,
                        int(resolution.review_required),
                        payload_hash,
                        resolution.to_json(),
                    ),
                )
                self._connection.execute("COMMIT")
                return StoreResult.success(resolution, created=True)
            except sqlite3.IntegrityError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                return StoreResult.outcome(
                    StoreState.CONFLICT,
                    "identity_resolution_conflict",
                )
            except (IdentityContractError, sqlite3.Error):
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                return StoreResult.outcome(
                    StoreState.FAILURE,
                    "identity_resolution_write_failed",
                )

    def get_resolution(self, resolution_id: str) -> StoreResult[IdentityResolution]:
        """Read one persisted resolution."""

        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT payload_json FROM identity_resolutions WHERE resolution_id = ?",
                    (resolution_id,),
                ).fetchone()
                if row is None:
                    return StoreResult.outcome(
                        StoreState.UNKNOWN,
                        "resolution_not_found",
                    )
                return StoreResult.success(
                    IdentityResolution.from_json(str(row["payload_json"]))
                )
            except (IdentityContractError, sqlite3.Error):
                return StoreResult.outcome(
                    StoreState.FAILURE,
                    "identity_resolution_read_failed",
                )

    def apply_review(
        self,
        decision: IdentityReviewDecision,
    ) -> StoreResult[IdentityReviewDecision]:
        """Record review and atomically replace active links when selected."""

        with self._lock:
            resolution_result = self.get_resolution(decision.resolution_id)
            if not resolution_result.ok or resolution_result.value is None:
                return StoreResult.outcome(
                    resolution_result.state,
                    resolution_result.code or "resolution_read_failed",
                )
            resolution = resolution_result.value
            if not resolution.review_required:
                return StoreResult.outcome(
                    StoreState.CONFLICT,
                    "review_not_required",
                )
            if (
                decision.policy_id != resolution.policy_id
                or decision.policy_version != resolution.policy_version
            ):
                return StoreResult.outcome(StoreState.DENIED, "policy_mismatch")
            selected = decision.selected_canonical_key
            if decision.action in {"confirm_match", "merge"} and selected not in (
                resolution.candidate_keys
            ):
                return StoreResult.outcome(
                    StoreState.CONFLICT,
                    "review_candidate_mismatch",
                )
            if decision.action == "split" and selected in resolution.candidate_keys:
                return StoreResult.outcome(
                    StoreState.CONFLICT,
                    "split_requires_new_key",
                )

            payload_hash = canonical_digest(decision.to_dict())
            try:
                existing = self._connection.execute(
                    "SELECT payload_hash FROM identity_reviews WHERE resolution_id = ?",
                    (resolution.resolution_id,),
                ).fetchone()
                if existing is not None:
                    if str(existing["payload_hash"]) == payload_hash:
                        return StoreResult.success(decision, created=False)
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "review_already_recorded",
                    )
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute(
                    "INSERT INTO identity_reviews("
                    "decision_id, resolution_id, action, payload_hash, payload_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        decision.decision_id,
                        decision.resolution_id,
                        decision.action,
                        payload_hash,
                        decision.to_json(),
                    ),
                )
                for key in resolution.source_keys:
                    self._connection.execute(
                        "UPDATE identity_links SET active = 0 "
                        "WHERE entity_type = ? AND source_fingerprint = ?",
                        (key.entity_type, key.fingerprint),
                    )
                    if selected is not None:
                        link = _review_link(key, selected, decision)
                        self._insert_link(link)
                self._connection.execute("COMMIT")
                return StoreResult.success(decision, created=True)
            except sqlite3.IntegrityError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                return StoreResult.outcome(StoreState.CONFLICT, "review_conflict")
            except sqlite3.Error:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                return StoreResult.outcome(StoreState.FAILURE, "review_write_failed")

    def integrity_check(self) -> StoreResult[Mapping[str, int]]:
        """Verify canonical hashes and foreign-key constraints."""

        tables = {
            "identity_links": IdentityLink.from_json,
            "identity_resolutions": IdentityResolution.from_json,
            "identity_reviews": IdentityReviewDecision.from_json,
        }
        counts: dict[str, int] = {}
        with self._lock:
            try:
                for table, parser in tables.items():
                    rows = self._connection.execute(
                        f"SELECT payload_hash, payload_json FROM {table}"  # noqa: S608
                    ).fetchall()
                    for row in rows:
                        record = parser(str(row["payload_json"]))
                        if canonical_digest(record.to_dict()) != str(
                            row["payload_hash"]
                        ):
                            raise IdentityContractError("stored payload hash differs")
                    counts[table] = len(rows)
                violations = self._connection.execute(
                    "PRAGMA foreign_key_check"
                ).fetchall()
                if violations:
                    raise IdentityContractError("identity foreign key differs")
            except (IdentityContractError, sqlite3.Error):
                return StoreResult.outcome(
                    StoreState.FAILURE,
                    "identity_integrity_failed",
                )
        return StoreResult.success(counts)

    def close(self) -> None:
        """Checkpoint and close the registry."""

        with self._lock:
            if self._closed:
                return
            try:
                self._connection.execute("PRAGMA wal_checkpoint(FULL)")
            finally:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "IdentityResolutionStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _insert_link(self, link: IdentityLink) -> None:
        self._connection.execute(
            "INSERT INTO identity_links("
            "link_id, entity_type, source_fingerprint, canonical_key, active, "
            "payload_hash, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                link.link_id,
                link.source_key.entity_type,
                link.source_key.fingerprint,
                link.canonical_key,
                int(link.active),
                canonical_digest(link.to_dict()),
                link.to_json(),
            ),
        )

    def _apply_migration(self) -> None:
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL)"
        )
        rows = self._connection.execute(
            "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
        if rows:
            if (
                len(rows) != 1
                or int(rows[0]["version"]) != IDENTITY_STORE_SCHEMA_VERSION
                or str(rows[0]["name"]) != "identity_resolution_store"
                or str(rows[0]["checksum"]) != _MIGRATION_CHECKSUM
            ):
                raise IdentityStoreCompatibilityError(
                    "identity store schema is unsupported"
                )
            return
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            for statement in _MIGRATION_SQL:
                self._connection.execute(statement)
            self._connection.execute(
                "INSERT INTO schema_migrations(version, name, checksum) "
                "VALUES (?, ?, ?)",
                (
                    IDENTITY_STORE_SCHEMA_VERSION,
                    "identity_resolution_store",
                    _MIGRATION_CHECKSUM,
                ),
            )
            self._connection.execute("COMMIT")
        except sqlite3.Error as exc:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise IdentityStoreError("identity store migration failed") from exc

    def _close_failed(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()


def _review_link(
    key: SourceIdentityKey,
    canonical_key: str,
    decision: IdentityReviewDecision,
) -> IdentityLink:
    digest = canonical_digest(
        {
            "canonical_key": canonical_key,
            "decision_id": decision.decision_id,
            "source_key": key.to_dict(),
        }
    )
    return IdentityLink(
        link_id=f"link_{digest.removeprefix('sha256:')[:32]}",
        source_key=key,
        canonical_key=canonical_key,
        evidence_digest=decision.evidence_digest,
        policy_id=decision.policy_id,
        policy_version=decision.policy_version,
        recorded_at=decision.decided_at,
    )


def _same_resolution_replay(
    persisted: IdentityResolution,
    proposed: IdentityResolution,
) -> bool:
    persisted_data = persisted.to_dict()
    proposed_data = proposed.to_dict()
    persisted_data.pop("resolved_at")
    proposed_data.pop("resolved_at")
    return persisted_data == proposed_data
