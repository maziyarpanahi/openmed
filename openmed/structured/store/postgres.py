"""PostgreSQL implementation of the Journey metadata-store contracts.

The adapter uses a dedicated Python DB-API connection and does not import a
PostgreSQL driver at module import time.  SQL, errors, and public results are
kept free of clinical values.
"""

from __future__ import annotations

import hashlib
import json
import re
import ssl
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Any, Protocol, cast
from urllib.parse import parse_qsl, unquote, urlsplit

from openmed.clinical.journey_contracts import canonical_digest

from .local import (
    LocalStoreError,
    SQLiteJourneyStore,
    SQLiteJourneyTransaction,
    StoreCompatibilityError,
    StoreConstraintError,
    StoreMigrationError,
    _valid_timestamp,
)
from .protocols import (
    AllowAllStoragePolicy,
    CommitStatusUnknown,
    StoragePolicy,
    StoreResult,
    StoreState,
    assert_contract_compatibility,
)

_SCHEMA_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_CONSTRAINT_SQLSTATES = frozenset({"23502", "23503", "23505", "23514", "23P01"})
_MIGRATION_LOCK_ID = 613_753_209


class DBAPICursor(Protocol):
    """Minimal cursor surface required by the PostgreSQL adapter."""

    description: Sequence[Sequence[Any]] | None
    rowcount: int

    def execute(self, operation: str, parameters: Sequence[Any] = ()) -> Any:
        """Execute one parameterized statement."""

    def executemany(
        self,
        operation: str,
        parameters: Sequence[Sequence[Any]],
    ) -> Any:
        """Execute one parameterized statement repeatedly."""

    def fetchone(self) -> Sequence[Any] | Mapping[str, Any] | None:
        """Fetch one result row."""

    def fetchall(self) -> Sequence[Sequence[Any] | Mapping[str, Any]]:
        """Fetch all result rows."""

    def close(self) -> None:
        """Close the cursor."""


class DBAPIConnection(Protocol):
    """Dedicated DB-API connection surface required by this store."""

    autocommit: bool

    def cursor(self) -> DBAPICursor:
        """Allocate a cursor."""

    def commit(self) -> None:
        """Commit the current transaction."""

    def rollback(self) -> None:
        """Roll back the current transaction."""

    def close(self) -> None:
        """Close the connection."""


class MigrationHealth(str, Enum):
    """Explicit health state for the PostgreSQL schema."""

    HEALTHY = "healthy"
    PENDING = "pending"
    UNSUPPORTED = "unsupported"
    DRIFTED = "drifted"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PostgresMigrationReport:
    """Value-safe migration state without connection or credential details."""

    state: MigrationHealth
    current_version: int
    target_version: int
    pending_versions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PostgresMigration:
    """One immutable ordered PostgreSQL migration."""

    version: int
    name: str
    statements: tuple[str, ...]

    @property
    def checksum(self) -> str:
        """Return a deterministic checksum for drift detection."""

        payload = "\n-- statement --\n".join(self.statements).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


POSTGRES_MIGRATIONS = (
    PostgresMigration(
        version=1,
        name="initial_append_only_store",
        statements=(
            """
            CREATE TABLE store_revisions (
                revision BIGINT PRIMARY KEY,
                committed_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE artifacts (
                artifact_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL UNIQUE,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_revision BIGINT NOT NULL
                    REFERENCES store_revisions(revision)
            )
            """,
            """
            CREATE TABLE evidence_locators (
                locator_id TEXT PRIMARY KEY,
                artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_revision BIGINT NOT NULL
                    REFERENCES store_revisions(revision)
            )
            """,
            """
            CREATE TABLE clinical_facts (
                fact_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_revision BIGINT NOT NULL
                    REFERENCES store_revisions(revision)
            )
            """,
            """
            CREATE TABLE fact_evidence (
                fact_id TEXT NOT NULL REFERENCES clinical_facts(fact_id),
                locator_id TEXT NOT NULL REFERENCES evidence_locators(locator_id),
                PRIMARY KEY (fact_id, locator_id)
            )
            """,
            """
            CREATE TABLE fact_parents (
                fact_id TEXT NOT NULL REFERENCES clinical_facts(fact_id),
                parent_fact_id TEXT NOT NULL REFERENCES clinical_facts(fact_id),
                PRIMARY KEY (fact_id, parent_fact_id)
            )
            """,
            """
            CREATE TABLE conflict_sets (
                conflict_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_revision BIGINT NOT NULL
                    REFERENCES store_revisions(revision)
            )
            """,
            """
            CREATE TABLE conflict_facts (
                conflict_id TEXT NOT NULL REFERENCES conflict_sets(conflict_id),
                fact_id TEXT NOT NULL REFERENCES clinical_facts(fact_id),
                PRIMARY KEY (conflict_id, fact_id)
            )
            """,
            """
            CREATE TABLE resolution_events (
                resolution_id TEXT PRIMARY KEY,
                conflict_id TEXT NOT NULL REFERENCES conflict_sets(conflict_id),
                supersedes_resolution_id TEXT
                    REFERENCES resolution_events(resolution_id),
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_revision BIGINT NOT NULL
                    REFERENCES store_revisions(revision)
            )
            """,
            """
            CREATE TABLE dataset_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_revision BIGINT NOT NULL
                    REFERENCES store_revisions(revision)
            )
            """,
            """
            CREATE TABLE canonical_record_versions (
                canonical_id TEXT NOT NULL,
                version BIGINT NOT NULL,
                subject_id TEXT NOT NULL,
                fact_id TEXT NOT NULL REFERENCES clinical_facts(fact_id),
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_revision BIGINT NOT NULL
                    REFERENCES store_revisions(revision),
                PRIMARY KEY (canonical_id, version),
                UNIQUE (canonical_id, payload_hash)
            )
            """,
            """
            CREATE TABLE job_metadata_versions (
                job_id TEXT NOT NULL,
                version BIGINT NOT NULL,
                state TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_revision BIGINT NOT NULL
                    REFERENCES store_revisions(revision),
                PRIMARY KEY (job_id, version),
                UNIQUE (job_id, payload_hash)
            )
            """,
            """
            CREATE INDEX clinical_facts_subject_revision_idx
            ON clinical_facts(subject_id, created_revision, fact_id)
            """,
            """
            CREATE INDEX canonical_records_revision_idx
            ON canonical_record_versions(canonical_id, created_revision, version)
            """,
            """
            CREATE INDEX job_metadata_revision_idx
            ON job_metadata_versions(job_id, created_revision, version)
            """,
        ),
    ),
    PostgresMigration(
        version=2,
        name="ingestion_control_plane",
        statements=(
            """
            CREATE TABLE ingestion_manifests (
                manifest_digest TEXT PRIMARY KEY,
                manifest_id TEXT NOT NULL UNIQUE,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE ingestion_jobs (
                job_id TEXT PRIMARY KEY,
                manifest_digest TEXT NOT NULL UNIQUE
                    REFERENCES ingestion_manifests(manifest_digest)
            )
            """,
            """
            CREATE TABLE ingestion_job_versions (
                job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                version BIGINT NOT NULL,
                state TEXT NOT NULL,
                checkpoint_sequence BIGINT NOT NULL,
                recorded_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (job_id, version),
                UNIQUE (job_id, payload_hash)
            )
            """,
            """
            CREATE TABLE ingestion_replay_audits (
                replay_id TEXT PRIMARY KEY,
                manifest_digest TEXT NOT NULL
                    REFERENCES ingestion_manifests(manifest_digest),
                job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                action TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE ingestion_leases (
                lease_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                worker_id TEXT NOT NULL,
                epoch BIGINT NOT NULL,
                acquired_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                UNIQUE (job_id, epoch)
            )
            """,
            """
            CREATE TABLE ingestion_lease_releases (
                lease_id TEXT PRIMARY KEY REFERENCES ingestion_leases(lease_id),
                released_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE ingestion_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                sequence BIGINT NOT NULL,
                step TEXT NOT NULL,
                input_digest TEXT NOT NULL,
                output_digest TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                UNIQUE (job_id, sequence),
                UNIQUE (job_id, step, input_digest)
            )
            """,
            """
            CREATE TABLE ingestion_retries (
                retry_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                classification TEXT NOT NULL,
                attempt BIGINT NOT NULL,
                recorded_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE ingestion_cancellations (
                cancellation_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL UNIQUE REFERENCES ingestion_jobs(job_id),
                requested_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE ingestion_quarantine_results (
                quarantine_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                classification TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE ingestion_quarantine_promotions (
                promotion_id TEXT PRIMARY KEY,
                quarantine_id TEXT NOT NULL UNIQUE
                    REFERENCES ingestion_quarantine_results(quarantine_id),
                promoted_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX ingestion_job_state_idx
            ON ingestion_job_versions(job_id, version, state)
            """,
            """
            CREATE INDEX ingestion_lease_job_epoch_idx
            ON ingestion_leases(job_id, epoch)
            """,
            """
            CREATE INDEX ingestion_checkpoint_job_sequence_idx
            ON ingestion_checkpoints(job_id, sequence)
            """,
        ),
    ),
    PostgresMigration(
        version=3,
        name="ingestion_pipeline_lineage",
        statements=(
            """
            CREATE TABLE ingestion_pipeline_stages (
                stage_manifest_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                stage TEXT NOT NULL,
                sequence BIGINT NOT NULL,
                state TEXT NOT NULL,
                input_digest TEXT NOT NULL,
                output_digest TEXT,
                recorded_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                UNIQUE (job_id, stage, input_digest)
            )
            """,
            """
            CREATE TABLE ingestion_pipeline_edges (
                parent_stage_manifest_id TEXT NOT NULL
                    REFERENCES ingestion_pipeline_stages(stage_manifest_id),
                child_stage_manifest_id TEXT NOT NULL
                    REFERENCES ingestion_pipeline_stages(stage_manifest_id),
                PRIMARY KEY (parent_stage_manifest_id, child_stage_manifest_id)
            )
            """,
            """
            CREATE TABLE ingestion_pipeline_invalidations (
                invalidation_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                stage_manifest_id TEXT NOT NULL
                    REFERENCES ingestion_pipeline_stages(stage_manifest_id),
                replacement_job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                recorded_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                UNIQUE (stage_manifest_id, replacement_job_id)
            )
            """,
            """
            CREATE INDEX ingestion_pipeline_job_sequence_idx
            ON ingestion_pipeline_stages(job_id, sequence, stage_manifest_id)
            """,
            """
            CREATE INDEX ingestion_pipeline_edge_child_idx
            ON ingestion_pipeline_edges(child_stage_manifest_id)
            """,
            """
            CREATE INDEX ingestion_pipeline_invalidation_job_idx
            ON ingestion_pipeline_invalidations(job_id, recorded_at)
            """,
        ),
    ),
)

LATEST_POSTGRES_MIGRATION_VERSION = POSTGRES_MIGRATIONS[-1].version


class PostgresStoreError(LocalStoreError):
    """Value-safe PostgreSQL lifecycle or operation failure."""


class _CursorResult:
    def __init__(self, cursor: DBAPICursor) -> None:
        self._cursor: DBAPICursor | None = cursor
        self._columns = tuple(
            _description_name(column) for column in (cursor.description or ())
        )
        self.rowcount = int(getattr(cursor, "rowcount", -1))
        if cursor.description is None:
            cursor.close()
            self._cursor = None

    def fetchone(self) -> Mapping[str, Any] | None:
        if self._cursor is None:
            return None
        try:
            row = self._cursor.fetchone()
            return None if row is None else self._mapping(row)
        finally:
            self._cursor.close()
            self._cursor = None

    def fetchall(self) -> list[Mapping[str, Any]]:
        if self._cursor is None:
            return []
        try:
            return [self._mapping(row) for row in self._cursor.fetchall()]
        finally:
            self._cursor.close()
            self._cursor = None

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return iter(self.fetchall())

    def _mapping(self, row: Sequence[Any] | Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(row, Mapping):
            return row
        return dict(zip(self._columns, row, strict=True))


class _PostgresConnectionAdapter:
    def __init__(self, connection: DBAPIConnection) -> None:
        self.raw = connection

    def execute(
        self,
        operation: str,
        parameters: Sequence[Any] = (),
    ) -> _CursorResult:
        cursor: DBAPICursor | None = None
        try:
            cursor = self.raw.cursor()
            cursor.execute(_postgres_sql(operation), parameters)
            return _CursorResult(cursor)
        except Exception as exc:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if _sqlstate(exc) in _CONSTRAINT_SQLSTATES:
                raise StoreConstraintError("postgres constraint conflict") from None
            raise PostgresStoreError("postgres operation failed") from None

    def executemany(
        self,
        operation: str,
        parameters: Sequence[Sequence[Any]] | Iterator[Sequence[Any]],
    ) -> None:
        cursor: DBAPICursor | None = None
        try:
            cursor = self.raw.cursor()
            cursor.executemany(_postgres_sql(operation), tuple(parameters))
            cursor.close()
        except Exception as exc:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if _sqlstate(exc) in _CONSTRAINT_SQLSTATES:
                raise StoreConstraintError("postgres constraint conflict") from None
            raise PostgresStoreError("postgres operation failed") from None


class PostgresJourneyTransaction(SQLiteJourneyTransaction):
    """One atomic PostgreSQL Journey graph transaction."""


class PostgresJourneyStore(SQLiteJourneyStore):
    """PostgreSQL parity implementation for Journey metadata protocols.

    The supplied connection must be dedicated to this store.  The adapter sets
    it to autocommit mode, creates or selects the configured schema, and closes
    it when the store closes.
    """

    def __init__(
        self,
        connection: DBAPIConnection,
        *,
        schema: str = "openmed_journey",
        policy: StoragePolicy | None = None,
    ) -> None:
        assert_contract_compatibility()
        if _SCHEMA_RE.fullmatch(schema) is None:
            raise ValueError("PostgreSQL schema must be a controlled identifier")
        for attribute in ("cursor", "commit", "rollback", "close"):
            if not callable(getattr(connection, attribute, None)):
                raise TypeError(
                    "connection does not implement the required DB-API surface"
                )
        if not hasattr(connection, "autocommit"):
            raise TypeError("connection must expose an autocommit attribute")
        self.schema = schema
        self.policy = policy or AllowAllStoragePolicy()
        self._lock = threading.RLock()
        self._closed = False
        self._raw_connection = connection
        self._connection = cast(Any, _PostgresConnectionAdapter(connection))
        self._migration_report = PostgresMigrationReport(
            state=MigrationHealth.PENDING,
            current_version=0,
            target_version=LATEST_POSTGRES_MIGRATION_VERSION,
            pending_versions=tuple(
                migration.version for migration in POSTGRES_MIGRATIONS
            ),
        )
        try:
            connection.autocommit = True
            self._apply_postgres_migrations()
            self._connection.execute(f"SET search_path TO {self.schema}")
        except StoreCompatibilityError:
            self._close_after_failed_open()
            raise
        except Exception:
            self._close_after_failed_open()
            raise PostgresStoreError("postgres store cannot be initialized") from None

    @classmethod
    def open(  # type: ignore[override]
        cls,
        connection: DBAPIConnection,
        *,
        schema: str = "openmed_journey",
        policy: StoragePolicy | None = None,
    ) -> StoreResult["PostgresJourneyStore"]:
        """Open a store with typed compatibility and failure outcomes."""

        try:
            return StoreResult.success(cls(connection, schema=schema, policy=policy))
        except StoreCompatibilityError:
            return StoreResult.outcome(StoreState.UNSUPPORTED, "schema_unsupported")
        except (PostgresStoreError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "store_open_failed")

    @classmethod
    def connect(
        cls,
        dsn: str,
        *,
        schema: str = "openmed_journey",
        policy: StoragePolicy | None = None,
        connect_options: Mapping[str, Any] | None = None,
    ) -> StoreResult["PostgresJourneyStore"]:
        """Connect lazily with pg8000 when the caller explicitly supplies a DSN.

        The DSN and connection options are never stored, logged, or included in
        result representations.
        """

        if not isinstance(dsn, str) or not dsn:
            return StoreResult.outcome(StoreState.FAILURE, "invalid_dsn")
        try:
            parameters = _pg8000_connect_parameters(dsn, connect_options)
        except (TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_dsn")
        try:
            pg8000 = import_module("pg8000.dbapi")
            connection = pg8000.connect(**parameters)
        except Exception:
            return StoreResult.outcome(StoreState.FAILURE, "postgres_connect_failed")
        return cls.open(connection, schema=schema, policy=policy)

    @property
    def migration_report(self) -> PostgresMigrationReport:
        """Return the verified migration state from this successful open."""

        return self._migration_report

    @property
    def schema_version(self) -> int:
        """Return the latest verified PostgreSQL migration version."""

        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
        ).fetchone()
        if row is None:
            raise PostgresStoreError("postgres migration state is unavailable")
        return int(row["version"])

    def close(self) -> None:
        """Close the dedicated PostgreSQL connection."""

        with self._lock:
            if self._closed:
                return
            try:
                self._raw_connection.rollback()
            except Exception:
                pass
            try:
                self._raw_connection.close()
            except Exception:
                pass
            self._closed = True

    def __enter__(self) -> "PostgresJourneyStore":
        return self

    @contextmanager
    def transaction(self, *, committed_at: str) -> Iterator[PostgresJourneyTransaction]:
        """Open one serializable, atomic PostgreSQL graph transaction."""

        if not _valid_timestamp(committed_at):
            raise ValueError("committed_at must be a timezone-aware ISO timestamp")
        self._require_open()
        with self._lock:
            try:
                self._connection.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
                self._connection.execute("LOCK TABLE store_revisions IN EXCLUSIVE MODE")
                row = self._connection.execute(
                    "INSERT INTO store_revisions(revision, committed_at) "
                    "SELECT COALESCE(MAX(revision), 0) + 1, ? "
                    "FROM store_revisions "
                    "RETURNING revision",
                    (committed_at,),
                ).fetchone()
                if row is None:
                    raise PostgresStoreError("postgres revision was not allocated")
                transaction = PostgresJourneyTransaction(self, int(row["revision"]))
                yield transaction
                if transaction.failed or not transaction.mutated:
                    self._raw_connection.rollback()
                else:
                    try:
                        self._raw_connection.commit()
                    except Exception:
                        raise CommitStatusUnknown(
                            "postgres commit status is unknown"
                        ) from None
            except BaseException:
                try:
                    self._raw_connection.rollback()
                except Exception:
                    pass
                raise

    def integrity_check(self) -> StoreResult[Mapping[str, int]]:
        """Validate constraints and payload hashes without returning values."""

        denied = self._denied("verify", "store")
        if denied is not None:
            return denied
        try:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM pg_constraint
                WHERE connamespace = current_schema()::regnamespace
                  AND NOT convalidated
                """
            ).fetchone()
            if row is None or int(row["count"]) != 0:
                return StoreResult.outcome(
                    StoreState.FAILURE,
                    "foreign_key_integrity_failed",
                )
            counts: dict[str, int] = {}
            for table in (
                "artifacts",
                "evidence_locators",
                "clinical_facts",
                "conflict_sets",
                "resolution_events",
                "dataset_snapshots",
                "canonical_record_versions",
                "job_metadata_versions",
            ):
                rows = self._connection.execute(
                    f"SELECT payload_hash, payload_json FROM {table}"  # noqa: S608
                ).fetchall()
                mismatched = any(
                    canonical_digest(json.loads(item["payload_json"]))
                    != item["payload_hash"]
                    for item in rows
                )
                if mismatched:
                    return StoreResult.outcome(
                        StoreState.FAILURE,
                        "payload_hash_mismatch",
                    )
                counts[table] = len(rows)
        except (PostgresStoreError, TypeError, ValueError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(counts)

    def _apply_postgres_migrations(self) -> None:
        try:
            self._connection.execute("BEGIN")
            self._connection.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            self._connection.execute(f"SET LOCAL search_path TO {self.schema}")
            self._connection.execute(
                "SELECT pg_advisory_xact_lock(?)",
                (_MIGRATION_LOCK_ID,),
            ).fetchone()
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version BIGINT PRIMARY KEY,
                    name TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._connection.execute("LOCK TABLE schema_migrations IN EXCLUSIVE MODE")
            applied = {
                int(row["version"]): (str(row["name"]), str(row["checksum"]))
                for row in self._connection.execute(
                    "SELECT version, name, checksum FROM schema_migrations "
                    "ORDER BY version"
                )
            }
            if applied and max(applied) > LATEST_POSTGRES_MIGRATION_VERSION:
                self._migration_report = PostgresMigrationReport(
                    state=MigrationHealth.UNSUPPORTED,
                    current_version=max(applied),
                    target_version=LATEST_POSTGRES_MIGRATION_VERSION,
                )
                raise StoreCompatibilityError(
                    "persisted PostgreSQL schema is unsupported"
                )
            known = {migration.version: migration for migration in POSTGRES_MIGRATIONS}
            for version, (name, checksum) in applied.items():
                migration = known.get(version)
                if (
                    migration is None
                    or migration.name != name
                    or migration.checksum != checksum
                ):
                    self._migration_report = PostgresMigrationReport(
                        state=MigrationHealth.DRIFTED,
                        current_version=max(applied, default=0),
                        target_version=LATEST_POSTGRES_MIGRATION_VERSION,
                    )
                    raise StoreCompatibilityError(
                        "persisted PostgreSQL migration checksum differs"
                    )
            pending = tuple(
                migration.version
                for migration in POSTGRES_MIGRATIONS
                if migration.version not in applied
            )
            self._migration_report = PostgresMigrationReport(
                state=MigrationHealth.PENDING if pending else MigrationHealth.HEALTHY,
                current_version=max(applied, default=0),
                target_version=LATEST_POSTGRES_MIGRATION_VERSION,
                pending_versions=pending,
            )
            for migration in POSTGRES_MIGRATIONS:
                if migration.version in applied:
                    continue
                for statement in migration.statements:
                    self._connection.execute(statement)
                self._connection.execute(
                    "INSERT INTO schema_migrations(version, name, checksum) "
                    "VALUES (?, ?, ?)",
                    (migration.version, migration.name, migration.checksum),
                )
            self._raw_connection.commit()
            self._migration_report = PostgresMigrationReport(
                state=MigrationHealth.HEALTHY,
                current_version=LATEST_POSTGRES_MIGRATION_VERSION,
                target_version=LATEST_POSTGRES_MIGRATION_VERSION,
            )
        except StoreCompatibilityError:
            try:
                self._raw_connection.rollback()
            except Exception:
                pass
            raise
        except (PostgresStoreError, StoreConstraintError):
            try:
                self._raw_connection.rollback()
            except Exception:
                pass
            self._migration_report = PostgresMigrationReport(
                state=MigrationHealth.FAILED,
                current_version=0,
                target_version=LATEST_POSTGRES_MIGRATION_VERSION,
            )
            raise StoreMigrationError("PostgreSQL store migration failed") from None

    def _close_after_failed_open(self) -> None:
        try:
            self._raw_connection.rollback()
        except Exception:
            pass
        try:
            self._raw_connection.close()
        except Exception:
            pass
        self._closed = True


def _postgres_sql(operation: str) -> str:
    """Translate the internal qmark parameter style to DB-API ``format``."""

    return operation.replace("?", "%s")


def _description_name(column: Sequence[Any] | Any) -> str:
    name = getattr(column, "name", None)
    if isinstance(name, str):
        return name
    return str(column[0])


def _sqlstate(error: Exception) -> str | None:
    value = getattr(error, "sqlstate", None)
    if isinstance(value, str):
        return value
    value = getattr(error, "pgcode", None)
    if isinstance(value, str):
        return value
    if error.args and isinstance(error.args[0], dict):
        code = error.args[0].get("C")
        return code if isinstance(code, str) else None
    return None


def _pg8000_connect_parameters(
    dsn: str,
    options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Convert a simple PostgreSQL URL into bounded pg8000 arguments."""

    parsed = urlsplit(dsn)
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or not parsed.hostname
        or not parsed.username
        or parsed.fragment
        or not parsed.path.startswith("/")
        or parsed.path.count("/") != 1
        or len(parsed.path) == 1
    ):
        raise ValueError("unsupported PostgreSQL connection URL")
    port = parsed.port if parsed.port is not None else 5432
    if not 1 <= port <= 65535:
        raise ValueError("invalid PostgreSQL port")
    supplied = dict(options or {})
    if set(supplied) - {"connect_timeout"}:
        raise ValueError("unsupported PostgreSQL connection option")
    timeout = supplied.get("connect_timeout", 3)
    if type(timeout) not in {int, float} or not 0 < timeout <= 30:
        raise ValueError("invalid PostgreSQL connection timeout")
    # Python 3.10 treats an empty string as a malformed field in strict mode.
    query = (
        parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
        if parsed.query
        else []
    )
    if len(query) > 1 or (query and query[0] != ("sslmode", "verify-full")):
        raise ValueError("unsupported PostgreSQL connection option")
    return {
        "database": unquote(parsed.path[1:]),
        "host": parsed.hostname,
        "password": unquote(parsed.password) if parsed.password is not None else None,
        "port": port,
        "ssl_context": ssl.create_default_context() if query else None,
        "timeout": timeout,
        "user": unquote(parsed.username),
    }


__all__ = [
    "DBAPIConnection",
    "LATEST_POSTGRES_MIGRATION_VERSION",
    "MigrationHealth",
    "POSTGRES_MIGRATIONS",
    "PostgresJourneyStore",
    "PostgresJourneyTransaction",
    "PostgresMigration",
    "PostgresMigrationReport",
    "PostgresStoreError",
]
