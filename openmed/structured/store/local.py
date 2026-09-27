"""Durable, offline-first local stores for longitudinal Journey records."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    ConflictSet,
    DatasetSnapshot,
    EvidenceLocator,
    ResolutionEvent,
    canonical_digest,
    sha256_digest,
)

from .migrations import LATEST_MIGRATION_VERSION, MIGRATIONS
from .protocols import (
    AllowAllStoragePolicy,
    CanonicalRecord,
    CanonicalRecordVersion,
    CommitStatusUnknown,
    JobMetadata,
    StoragePolicy,
    StorePoint,
    StoreResult,
    StoreState,
    assert_contract_compatibility,
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPAQUE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{16,128}$")
_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)

T = TypeVar("T")


class LocalStoreError(RuntimeError):
    """Base value-safe exception for local-store lifecycle failures."""


class StoreCompatibilityError(LocalStoreError):
    """Raised when persisted schema state is newer or has drifted."""


class StoreMigrationError(LocalStoreError):
    """Raised when a deterministic migration cannot be applied."""


class StoreConstraintError(LocalStoreError):
    """Raised by backend adapters for a value-safe constraint conflict."""


@dataclass(frozen=True, slots=True)
class IngestedGraph:
    """Value-safe counts for one committed source-to-fact graph."""

    artifact: ClinicalArtifact = field(repr=False)
    evidence_count: int
    fact_count: int
    conflict_count: int


class LocalArtifactStore:
    """Content-addressed filesystem store with verified reads and writes."""

    def __init__(
        self,
        root: str | Path,
        *,
        policy: StoragePolicy | None = None,
    ) -> None:
        self.root = Path(root)
        self.policy = policy or AllowAllStoragePolicy()
        self._lock = threading.RLock()
        self._prepare_directories()

    def put_bytes(
        self,
        artifact: ClinicalArtifact,
        content: bytes,
    ) -> StoreResult[ClinicalArtifact]:
        """Persist bytes only when size and content digest match metadata."""

        if not self.policy.allows("write", "artifact"):
            return StoreResult.outcome(StoreState.DENIED, "policy_denied")
        if not isinstance(content, bytes):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_content_type")
        if len(content) != artifact.byte_size:
            return StoreResult.outcome(StoreState.CONFLICT, "artifact_size_mismatch")
        if sha256_digest(content) != artifact.content_hash:
            return StoreResult.outcome(StoreState.CONFLICT, "artifact_hash_mismatch")

        target = self._blob_path(artifact.content_hash)
        with self._lock:
            existing = self._read_verified(artifact.content_hash)
            if existing.ok:
                return StoreResult.success(artifact, created=False)
            if existing.state not in {StoreState.UNKNOWN}:
                return StoreResult.outcome(
                    existing.state, existing.code or "read_failed"
                )
            try:
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                os.chmod(target.parent, 0o700)
                descriptor, temp_name = tempfile.mkstemp(
                    prefix=".journey-blob-",
                    dir=target.parent,
                )
                try:
                    if hasattr(os, "fchmod"):
                        os.fchmod(descriptor, 0o600)
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    try:
                        os.link(temp_name, target)
                        created = True
                    except FileExistsError:
                        created = False
                    finally:
                        Path(temp_name).unlink(missing_ok=True)
                    _fsync_directory(target.parent)
                except BaseException:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                    Path(temp_name).unlink(missing_ok=True)
                    raise
            except OSError:
                return StoreResult.outcome(StoreState.FAILURE, "artifact_write_failed")

            verified = self._read_verified(artifact.content_hash)
            if not verified.ok:
                if created:
                    target.unlink(missing_ok=True)
                return StoreResult.outcome(StoreState.FAILURE, "artifact_verify_failed")
            return StoreResult.success(artifact, created=created)

    def get_bytes(self, content_hash: str) -> StoreResult[bytes]:
        """Read bytes and revalidate the content-addressed digest."""

        if not self.policy.allows("read", "artifact"):
            return StoreResult.outcome(StoreState.DENIED, "policy_denied")
        return self._read_verified(content_hash)

    def _read_verified(self, content_hash: str) -> StoreResult[bytes]:
        """Read and verify a blob for internal integrity operations."""

        try:
            target = self._blob_path(content_hash)
        except ValueError:
            return StoreResult.outcome(StoreState.FAILURE, "invalid_content_hash")
        try:
            if target.is_symlink():
                return StoreResult.outcome(StoreState.FAILURE, "artifact_path_unsafe")
            content = target.read_bytes()
        except FileNotFoundError:
            return StoreResult.outcome(StoreState.UNKNOWN, "artifact_not_found")
        except OSError:
            return StoreResult.outcome(StoreState.FAILURE, "artifact_read_failed")
        if sha256_digest(content) != content_hash:
            return StoreResult.outcome(StoreState.FAILURE, "artifact_integrity_failed")
        return StoreResult.success(content)

    def discard_if_created(self, content_hash: str) -> None:
        """Remove a newly staged blob after a metadata transaction rolls back."""

        try:
            target = self._blob_path(content_hash)
            if target.is_file() and not target.is_symlink():
                target.unlink()
                _fsync_directory(target.parent)
        except (OSError, ValueError):
            return

    def _prepare_directories(self) -> None:
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(self.root, 0o700)
            blob_root = self.root / "blobs" / "sha256"
            blob_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(blob_root, 0o700)
        except OSError as exc:
            raise LocalStoreError("local artifact store cannot be initialized") from exc

    def _blob_path(self, content_hash: str) -> Path:
        if (
            not isinstance(content_hash, str)
            or _DIGEST_RE.fullmatch(content_hash) is None
        ):
            raise ValueError("content hash must be a normalized SHA-256 digest")
        digest = content_hash.removeprefix("sha256:")
        return self.root / "blobs" / "sha256" / digest[:2] / digest


class SQLiteJourneyStore:
    """Append-only SQLite store with revisions and deterministic migrations."""

    def __init__(
        self,
        path: str | Path,
        *,
        policy: StoragePolicy | None = None,
    ) -> None:
        assert_contract_compatibility()
        self.path = Path(path)
        self.policy = policy or AllowAllStoragePolicy()
        self._lock = threading.RLock()
        self._closed = False
        self._prepare_path()
        try:
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
            self._apply_migrations()
        except StoreCompatibilityError:
            self._close_after_failed_open()
            raise
        except (OSError, sqlite3.Error) as exc:
            self._close_after_failed_open()
            raise LocalStoreError("local metadata store cannot be initialized") from exc

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        policy: StoragePolicy | None = None,
    ) -> StoreResult["SQLiteJourneyStore"]:
        """Open a store with typed compatibility and failure outcomes."""

        try:
            return StoreResult.success(cls(path, policy=policy))
        except StoreCompatibilityError:
            return StoreResult.outcome(StoreState.UNSUPPORTED, "schema_unsupported")
        except LocalStoreError:
            return StoreResult.outcome(StoreState.FAILURE, "store_open_failed")

    @property
    def schema_version(self) -> int:
        """Return the latest verified migration version."""

        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_migrations"
        ).fetchone()
        return int(row["version"])

    @property
    def latest_revision(self) -> int | None:
        """Return the latest committed revision, if any."""

        row = self._connection.execute(
            "SELECT MAX(revision) AS revision FROM store_revisions"
        ).fetchone()
        return None if row["revision"] is None else int(row["revision"])

    def close(self) -> None:
        """Checkpoint and close the local database."""

        with self._lock:
            if self._closed:
                return
            try:
                self._connection.execute("PRAGMA wal_checkpoint(FULL)")
            finally:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "SQLiteJourneyStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self, *, committed_at: str) -> Iterator["SQLiteJourneyTransaction"]:
        """Open one atomic append-only transaction at a supplied timestamp."""

        if not _valid_timestamp(committed_at):
            raise ValueError("committed_at must be a timezone-aware ISO timestamp")
        self._require_open()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                cursor = self._connection.execute(
                    "INSERT INTO store_revisions(committed_at) VALUES (?)",
                    (committed_at,),
                )
                if cursor.lastrowid is None:
                    raise LocalStoreError("local store revision was not allocated")
                transaction = SQLiteJourneyTransaction(self, int(cursor.lastrowid))
                yield transaction
                if transaction.failed or not transaction.mutated:
                    self._connection.execute("ROLLBACK")
                else:
                    self._connection.execute("COMMIT")
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def put_artifact(
        self,
        artifact: ClinicalArtifact,
        *,
        committed_at: str,
    ) -> StoreResult[ClinicalArtifact]:
        """Persist artifact metadata in one auto-committed revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_artifact(artifact))

    def put_evidence(
        self,
        locator: EvidenceLocator,
        *,
        committed_at: str,
    ) -> StoreResult[EvidenceLocator]:
        """Persist evidence in one auto-committed revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_evidence(locator))

    def put_fact(
        self,
        fact: ClinicalFact,
        *,
        committed_at: str,
    ) -> StoreResult[ClinicalFact]:
        """Persist a fact in one auto-committed revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_fact(fact))

    def put_conflict(
        self,
        conflict: ConflictSet,
        *,
        committed_at: str,
    ) -> StoreResult[ConflictSet]:
        """Persist a conflict set in one auto-committed revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_conflict(conflict))

    def put_resolution(
        self,
        resolution: ResolutionEvent,
        *,
        committed_at: str,
    ) -> StoreResult[ResolutionEvent]:
        """Persist a resolution event in one auto-committed revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_resolution(resolution))

    def put_dataset(
        self,
        snapshot: DatasetSnapshot,
        *,
        committed_at: str,
    ) -> StoreResult[DatasetSnapshot]:
        """Persist a dataset snapshot in one auto-committed revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_dataset(snapshot))

    def put_canonical(
        self,
        record: CanonicalRecord,
        *,
        committed_at: str,
    ) -> StoreResult[CanonicalRecordVersion]:
        """Append a canonical-record version in one revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_canonical(record))

    def put_job(
        self,
        job: JobMetadata,
        *,
        committed_at: str,
    ) -> StoreResult[JobMetadata]:
        """Append PHI-free job metadata in one revision."""

        return self._auto_put(committed_at, lambda tx: tx.put_job(job))

    def get_artifact(
        self,
        artifact_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[ClinicalArtifact]:
        """Read artifact metadata visible at one revision."""

        return self._read_record(
            operation="read",
            record_type="artifact",
            table="artifacts",
            id_column="artifact_id",
            record_id=artifact_id,
            parser=ClinicalArtifact.from_json,
            as_of=as_of,
        )

    def get_artifact_by_hash(
        self,
        content_hash: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[ClinicalArtifact]:
        """Read artifact metadata by a normalized content digest."""

        if _DIGEST_RE.fullmatch(content_hash) is None:
            return StoreResult.outcome(StoreState.FAILURE, "invalid_content_hash")
        return self._read_record(
            operation="read",
            record_type="artifact",
            table="artifacts",
            id_column="content_hash",
            record_id=content_hash,
            parser=ClinicalArtifact.from_json,
            as_of=as_of,
        )

    def get_evidence(
        self,
        locator_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[EvidenceLocator]:
        """Read evidence visible at one revision."""

        return self._read_record(
            operation="read",
            record_type="evidence",
            table="evidence_locators",
            id_column="locator_id",
            record_id=locator_id,
            parser=EvidenceLocator.from_json,
            as_of=as_of,
        )

    def get_fact(
        self,
        fact_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[ClinicalFact]:
        """Read a fact visible at one revision."""

        return self._read_record(
            operation="read",
            record_type="fact",
            table="clinical_facts",
            id_column="fact_id",
            record_id=fact_id,
            parser=ClinicalFact.from_json,
            as_of=as_of,
        )

    def get_conflict(
        self,
        conflict_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[ConflictSet]:
        """Read a conflict visible at one revision."""

        return self._read_record(
            operation="read",
            record_type="conflict",
            table="conflict_sets",
            id_column="conflict_id",
            record_id=conflict_id,
            parser=ConflictSet.from_json,
            as_of=as_of,
        )

    def get_resolution(
        self,
        resolution_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[ResolutionEvent]:
        """Read a resolution visible at one revision."""

        return self._read_record(
            operation="read",
            record_type="resolution",
            table="resolution_events",
            id_column="resolution_id",
            record_id=resolution_id,
            parser=ResolutionEvent.from_json,
            as_of=as_of,
        )

    def get_dataset(
        self,
        snapshot_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[DatasetSnapshot]:
        """Read a dataset snapshot visible at one revision."""

        return self._read_record(
            operation="read",
            record_type="dataset",
            table="dataset_snapshots",
            id_column="snapshot_id",
            record_id=snapshot_id,
            parser=DatasetSnapshot.from_json,
            as_of=as_of,
        )

    def get_canonical(
        self,
        canonical_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[CanonicalRecordVersion]:
        """Read the newest canonical-record version visible at one revision."""

        denied = self._denied("read", "canonical")
        if denied is not None:
            return denied
        if not _valid_opaque_id(canonical_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        cutoff = self._cutoff(as_of)
        row = self._connection.execute(
            """
            SELECT payload_json, version, created_revision
            FROM canonical_record_versions
            WHERE canonical_id = ? AND created_revision <= ?
            ORDER BY created_revision DESC, version DESC
            LIMIT 1
            """,
            (canonical_id, cutoff),
        ).fetchone()
        if row is None:
            return StoreResult.outcome(StoreState.UNKNOWN, "canonical_not_found")
        try:
            record = CanonicalRecord.from_dict(json.loads(row["payload_json"]))
            version = CanonicalRecordVersion(
                record=record,
                version=int(row["version"]),
                revision=int(row["created_revision"]),
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(version, revision=version.revision)

    def list_canonical_versions(
        self,
        canonical_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[tuple[CanonicalRecordVersion, ...]]:
        """Return complete append-only canonical history through one revision."""

        denied = self._denied("read", "canonical")
        if denied is not None:
            return denied
        if not _valid_opaque_id(canonical_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        rows = self._connection.execute(
            """
            SELECT payload_json, version, created_revision
            FROM canonical_record_versions
            WHERE canonical_id = ? AND created_revision <= ?
            ORDER BY version
            """,
            (canonical_id, self._cutoff(as_of)),
        ).fetchall()
        if not rows:
            return StoreResult.outcome(StoreState.UNKNOWN, "canonical_not_found")
        try:
            versions = tuple(
                CanonicalRecordVersion(
                    record=CanonicalRecord.from_dict(json.loads(row["payload_json"])),
                    version=int(row["version"]),
                    revision=int(row["created_revision"]),
                )
                for row in rows
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(versions)

    def list_canonical_records(
        self,
        subject_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[tuple[CanonicalRecordVersion, ...]]:
        """Return each latest visible canonical pointer for one subject."""

        denied = self._denied("read", "canonical")
        if denied is not None:
            return denied
        if not _valid_opaque_id(subject_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        cutoff = self._cutoff(as_of)
        rows = self._connection.execute(
            """
            SELECT visible_record.payload_json, visible_record.version,
                   visible_record.created_revision
            FROM canonical_record_versions AS visible_record
            WHERE visible_record.subject_id = ?
              AND visible_record.created_revision <= ?
              AND visible_record.version = (
                  SELECT MAX(candidate.version)
                  FROM canonical_record_versions AS candidate
                  WHERE candidate.canonical_id = visible_record.canonical_id
                    AND candidate.created_revision <= ?
              )
            ORDER BY visible_record.canonical_id
            """,
            (subject_id, cutoff, cutoff),
        ).fetchall()
        try:
            records = tuple(
                CanonicalRecordVersion(
                    record=CanonicalRecord.from_dict(json.loads(row["payload_json"])),
                    version=int(row["version"]),
                    revision=int(row["created_revision"]),
                )
                for row in rows
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(records)

    def get_job(
        self,
        job_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[JobMetadata]:
        """Read the newest job metadata visible at one revision."""

        denied = self._denied("read", "job")
        if denied is not None:
            return denied
        if not _valid_opaque_id(job_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        row = self._connection.execute(
            """
            SELECT payload_json, created_revision
            FROM job_metadata_versions
            WHERE job_id = ? AND created_revision <= ?
            ORDER BY created_revision DESC, version DESC
            LIMIT 1
            """,
            (job_id, self._cutoff(as_of)),
        ).fetchone()
        if row is None:
            return StoreResult.outcome(StoreState.UNKNOWN, "job_not_found")
        try:
            job = JobMetadata.from_dict(json.loads(row["payload_json"]))
        except (ValueError, TypeError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(job, revision=int(row["created_revision"]))

    def list_job_versions(
        self,
        job_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[tuple[JobMetadata, ...]]:
        """Return append-only job metadata versions through one revision."""

        denied = self._denied("read", "job")
        if denied is not None:
            return denied
        if not _valid_opaque_id(job_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        rows = self._connection.execute(
            """
            SELECT payload_json
            FROM job_metadata_versions
            WHERE job_id = ? AND created_revision <= ?
            ORDER BY version, created_revision
            """,
            (job_id, self._cutoff(as_of)),
        ).fetchall()
        if not rows:
            return StoreResult.outcome(StoreState.UNKNOWN, "job_not_found")
        try:
            versions = tuple(
                JobMetadata.from_dict(json.loads(row["payload_json"])) for row in rows
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(versions)

    def list_jobs(
        self,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[tuple[JobMetadata, ...]]:
        """Return the latest visible version of each PHI-free job record."""

        denied = self._denied("read", "job")
        if denied is not None:
            return denied
        cutoff = self._cutoff(as_of)
        rows = self._connection.execute(
            """
            SELECT visible_job.payload_json
            FROM job_metadata_versions AS visible_job
            WHERE visible_job.created_revision <= ?
              AND visible_job.version = (
                  SELECT MAX(candidate.version)
                  FROM job_metadata_versions AS candidate
                  WHERE candidate.job_id = visible_job.job_id
                    AND candidate.created_revision <= ?
              )
            ORDER BY visible_job.job_id
            """,
            (cutoff, cutoff),
        ).fetchall()
        try:
            jobs = tuple(
                JobMetadata.from_dict(json.loads(row["payload_json"])) for row in rows
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(jobs)

    def list_facts(
        self,
        subject_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[tuple[ClinicalFact, ...]]:
        """Return deterministic fact history for one opaque subject."""

        denied = self._denied("read", "fact")
        if denied is not None:
            return denied
        if not _valid_opaque_id(subject_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        rows = self._connection.execute(
            """
            SELECT payload_json
            FROM clinical_facts
            WHERE subject_id = ? AND created_revision <= ?
            ORDER BY created_revision, fact_id
            """,
            (subject_id, self._cutoff(as_of)),
        ).fetchall()
        try:
            facts = tuple(ClinicalFact.from_json(row["payload_json"]) for row in rows)
        except ValueError:
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(facts)

    def list_conflicts(
        self,
        subject_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[tuple[ConflictSet, ...]]:
        """Return deterministic conflict history for one opaque subject."""

        denied = self._denied("read", "conflict")
        if denied is not None:
            return denied
        if not _valid_opaque_id(subject_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        rows = self._connection.execute(
            """
            SELECT payload_json
            FROM conflict_sets
            WHERE subject_id = ? AND created_revision <= ?
            ORDER BY created_revision, conflict_id
            """,
            (subject_id, self._cutoff(as_of)),
        ).fetchall()
        try:
            conflicts = tuple(
                ConflictSet.from_json(row["payload_json"]) for row in rows
            )
        except ValueError:
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(conflicts)

    def list_resolutions(
        self,
        conflict_id: str,
        *,
        as_of: StorePoint | None = None,
    ) -> StoreResult[tuple[ResolutionEvent, ...]]:
        """Return append-only resolution history for one conflict."""

        denied = self._denied("read", "resolution")
        if denied is not None:
            return denied
        if not _valid_opaque_id(conflict_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        rows = self._connection.execute(
            """
            SELECT payload_json
            FROM resolution_events
            WHERE conflict_id = ? AND created_revision <= ?
            ORDER BY created_revision, resolution_id
            """,
            (conflict_id, self._cutoff(as_of)),
        ).fetchall()
        try:
            resolutions = tuple(
                ResolutionEvent.from_json(row["payload_json"]) for row in rows
            )
        except ValueError:
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(resolutions)

    def integrity_check(self) -> StoreResult[Mapping[str, int]]:
        """Run local SQLite and payload-hash integrity checks without values."""

        denied = self._denied("verify", "store")
        if denied is not None:
            return denied
        row = self._connection.execute("PRAGMA integrity_check").fetchone()
        if row is None or row[0] != "ok":
            return StoreResult.outcome(StoreState.FAILURE, "sqlite_integrity_failed")
        if self._connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            return StoreResult.outcome(
                StoreState.FAILURE, "foreign_key_integrity_failed"
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
            try:
                mismatched = any(
                    canonical_digest(json.loads(item["payload_json"]))
                    != item["payload_hash"]
                    for item in rows
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
            if mismatched:
                return StoreResult.outcome(StoreState.FAILURE, "payload_hash_mismatch")
            counts[table] = len(rows)
        return StoreResult.success(counts)

    def _auto_put(
        self,
        committed_at: str,
        operation: Callable[["SQLiteJourneyTransaction"], StoreResult[T]],
    ) -> StoreResult[T]:
        result: StoreResult[T]
        try:
            with self.transaction(committed_at=committed_at) as transaction:
                result = operation(transaction)
        except CommitStatusUnknown:
            return StoreResult.outcome(StoreState.UNKNOWN, "commit_status_unknown")
        except (LocalStoreError, sqlite3.Error, ValueError, TypeError):
            return StoreResult.outcome(StoreState.FAILURE, "transaction_failed")
        if not result.ok:
            return result
        return StoreResult.success(
            result.value,  # type: ignore[arg-type]
            created=result.created,
            revision=transaction.revision if result.created else result.revision,
        )

    def _read_record(
        self,
        *,
        operation: str,
        record_type: str,
        table: str,
        id_column: str,
        record_id: str,
        parser: Callable[[str], T],
        as_of: StorePoint | None,
    ) -> StoreResult[T]:
        denied = self._denied(operation, record_type)
        if denied is not None:
            return denied
        if id_column != "content_hash" and not _valid_opaque_id(record_id):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_identifier")
        row = self._connection.execute(
            f"""
            SELECT payload_json, created_revision FROM {table}
            WHERE {id_column} = ? AND created_revision <= ?
            """,  # noqa: S608
            (record_id, self._cutoff(as_of)),
        ).fetchone()
        if row is None:
            return StoreResult.outcome(StoreState.UNKNOWN, f"{record_type}_not_found")
        try:
            record = parser(row["payload_json"])
        except (ValueError, TypeError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        return StoreResult.success(record, revision=int(row["created_revision"]))

    def _cutoff(self, as_of: StorePoint | None) -> int:
        if as_of is not None:
            return as_of.revision
        return self.latest_revision or 0

    def _denied(self, operation: str, record_type: str) -> StoreResult[Any] | None:
        if self.policy.allows(operation, record_type):
            return None
        return StoreResult.outcome(StoreState.DENIED, "policy_denied")

    def _prepare_path(self) -> None:
        try:
            if not self.path.parent.exists():
                self.path.parent.mkdir(mode=0o700, parents=True)
                os.chmod(self.path.parent, 0o700)
        except OSError as exc:
            raise LocalStoreError("local metadata path cannot be initialized") from exc

    def _apply_migrations(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                checksum TEXT NOT NULL
            )
            """
        )
        applied = {
            int(row["version"]): (str(row["name"]), str(row["checksum"]))
            for row in self._connection.execute(
                "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
            )
        }
        if applied and max(applied) > LATEST_MIGRATION_VERSION:
            raise StoreCompatibilityError("persisted schema major is unsupported")
        known = {migration.version: migration for migration in MIGRATIONS}
        for version, (name, checksum) in applied.items():
            migration = known.get(version)
            if (
                migration is None
                or migration.name != name
                or migration.checksum != checksum
            ):
                raise StoreCompatibilityError("persisted migration checksum differs")

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            for migration in MIGRATIONS:
                if migration.version in applied:
                    continue
                for statement in migration.statements:
                    self._connection.execute(statement)
                self._connection.execute(
                    "INSERT INTO schema_migrations(version, name, checksum) "
                    "VALUES (?, ?, ?)",
                    (migration.version, migration.name, migration.checksum),
                )
            self._connection.execute("COMMIT")
        except sqlite3.Error as exc:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise StoreMigrationError("local store migration failed") from exc

    def _close_after_failed_open(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()

    def _require_open(self) -> None:
        if self._closed:
            raise LocalStoreError("local metadata store is closed")


class SQLiteJourneyTransaction:
    """Mutation surface bound to one SQLite revision and transaction."""

    def __init__(self, store: SQLiteJourneyStore, revision: int) -> None:
        self.store = store
        self.revision = revision
        self.failed = False
        self.mutated = False

    def put_artifact(self, artifact: ClinicalArtifact) -> StoreResult[ClinicalArtifact]:
        """Persist artifact metadata idempotently."""

        denied = self._denied("write", "artifact")
        if denied is not None:
            return denied
        if not all(
            self._exists("artifacts", "artifact_id", item)
            for item in artifact.parent_artifact_ids
        ):
            return self._fail(StoreState.PARTIAL, "missing_parent_artifact")
        existing_hash = self.store._connection.execute(
            "SELECT payload_json, created_revision FROM artifacts WHERE content_hash = ?",
            (artifact.content_hash,),
        ).fetchone()
        if existing_hash is not None:
            return self._existing(existing_hash, artifact, ClinicalArtifact.from_json)
        return self._insert_simple(
            table="artifacts",
            id_column="artifact_id",
            record_id=artifact.artifact_id,
            payload_json=artifact.to_json(),
            extra_columns=("content_hash",),
            extra_values=(artifact.content_hash,),
            value=artifact,
        )

    def put_evidence(self, locator: EvidenceLocator) -> StoreResult[EvidenceLocator]:
        """Persist evidence after its artifact dependency exists."""

        denied = self._denied("write", "evidence")
        if denied is not None:
            return denied
        if not self._exists("artifacts", "artifact_id", locator.artifact_id):
            return self._fail(StoreState.PARTIAL, "missing_artifact")
        return self._insert_simple(
            table="evidence_locators",
            id_column="locator_id",
            record_id=locator.locator_id,
            payload_json=locator.to_json(),
            extra_columns=("artifact_id",),
            extra_values=(locator.artifact_id,),
            value=locator,
        )

    def put_fact(self, fact: ClinicalFact) -> StoreResult[ClinicalFact]:
        """Persist a fact after evidence and parent dependencies exist."""

        denied = self._denied("write", "fact")
        if denied is not None:
            return denied
        if not all(
            self._exists("evidence_locators", "locator_id", item)
            for item in fact.evidence_ids
        ):
            return self._fail(StoreState.PARTIAL, "missing_evidence")
        if fact.parent_fact_ids:
            placeholders = ",".join("?" for _ in fact.parent_fact_ids)
            parent_rows = self.store._connection.execute(
                "SELECT fact_id, subject_id FROM clinical_facts WHERE fact_id IN "
                f"({placeholders})",  # noqa: S608
                fact.parent_fact_ids,
            ).fetchall()
            if len(parent_rows) != len(fact.parent_fact_ids):
                return self._fail(StoreState.PARTIAL, "missing_parent_fact")
            if any(row["subject_id"] != fact.subject_id for row in parent_rows):
                return self._fail(StoreState.CONFLICT, "parent_subject_mismatch")
        result = self._insert_simple(
            table="clinical_facts",
            id_column="fact_id",
            record_id=fact.fact_id,
            payload_json=fact.to_json(),
            extra_columns=("subject_id", "fact_type"),
            extra_values=(fact.subject_id, fact.fact_type),
            value=fact,
        )
        if result.ok and result.created:
            self.store._connection.executemany(
                "INSERT INTO fact_evidence(fact_id, locator_id) VALUES (?, ?)",
                ((fact.fact_id, item) for item in fact.evidence_ids),
            )
            self.store._connection.executemany(
                "INSERT INTO fact_parents(fact_id, parent_fact_id) VALUES (?, ?)",
                ((fact.fact_id, item) for item in fact.parent_fact_ids),
            )
        return result

    def put_conflict(self, conflict: ConflictSet) -> StoreResult[ConflictSet]:
        """Persist a conflict after every referenced fact exists."""

        denied = self._denied("write", "conflict")
        if denied is not None:
            return denied
        rows = self.store._connection.execute(
            "SELECT fact_id, subject_id FROM clinical_facts WHERE fact_id IN "
            f"({','.join('?' for _ in conflict.fact_ids)})",  # noqa: S608
            conflict.fact_ids,
        ).fetchall()
        if len(rows) != len(conflict.fact_ids):
            return self._fail(StoreState.PARTIAL, "missing_conflict_fact")
        if any(row["subject_id"] != conflict.subject_id for row in rows):
            return self._fail(StoreState.CONFLICT, "conflict_subject_mismatch")
        result = self._insert_simple(
            table="conflict_sets",
            id_column="conflict_id",
            record_id=conflict.conflict_id,
            payload_json=conflict.to_json(),
            extra_columns=("subject_id",),
            extra_values=(conflict.subject_id,),
            value=conflict,
        )
        if result.ok and result.created:
            self.store._connection.executemany(
                "INSERT INTO conflict_facts(conflict_id, fact_id) VALUES (?, ?)",
                ((conflict.conflict_id, item) for item in conflict.fact_ids),
            )
        return result

    def put_resolution(
        self,
        resolution: ResolutionEvent,
    ) -> StoreResult[ResolutionEvent]:
        """Persist a resolution after conflict and supersession checks."""

        denied = self._denied("write", "resolution")
        if denied is not None:
            return denied
        conflict_row = self.store._connection.execute(
            "SELECT payload_json FROM conflict_sets WHERE conflict_id = ?",
            (resolution.conflict_id,),
        ).fetchone()
        if conflict_row is None:
            return self._fail(StoreState.PARTIAL, "missing_conflict")
        conflict = ConflictSet.from_json(conflict_row["payload_json"])
        selected = set(resolution.selected_fact_ids + resolution.rejected_fact_ids)
        if not selected.issubset(conflict.fact_ids):
            return self._fail(StoreState.CONFLICT, "resolution_fact_mismatch")
        if resolution.supersedes_resolution_id is not None:
            parent = self.store._connection.execute(
                "SELECT conflict_id FROM resolution_events WHERE resolution_id = ?",
                (resolution.supersedes_resolution_id,),
            ).fetchone()
            if parent is None:
                return self._fail(StoreState.PARTIAL, "missing_resolution_parent")
            if parent["conflict_id"] != resolution.conflict_id:
                return self._fail(StoreState.CONFLICT, "resolution_parent_mismatch")
        return self._insert_simple(
            table="resolution_events",
            id_column="resolution_id",
            record_id=resolution.resolution_id,
            payload_json=resolution.to_json(),
            extra_columns=("conflict_id", "supersedes_resolution_id"),
            extra_values=(
                resolution.conflict_id,
                resolution.supersedes_resolution_id,
            ),
            value=resolution,
        )

    def put_dataset(self, snapshot: DatasetSnapshot) -> StoreResult[DatasetSnapshot]:
        """Persist a dataset snapshot after source dependencies exist."""

        denied = self._denied("write", "dataset")
        if denied is not None:
            return denied
        checks = (
            ("artifacts", "artifact_id", snapshot.source_artifact_ids),
            ("clinical_facts", "fact_id", snapshot.source_fact_ids),
            ("dataset_snapshots", "snapshot_id", snapshot.parent_snapshot_ids),
        )
        for table, column, identifiers in checks:
            if not all(self._exists(table, column, item) for item in identifiers):
                return self._fail(StoreState.PARTIAL, "missing_dataset_source")
        return self._insert_simple(
            table="dataset_snapshots",
            id_column="snapshot_id",
            record_id=snapshot.snapshot_id,
            payload_json=snapshot.to_json(),
            value=snapshot,
        )

    def put_canonical(
        self,
        record: CanonicalRecord,
    ) -> StoreResult[CanonicalRecordVersion]:
        """Append a canonical-record version or return its identical version."""

        denied = self._denied("write", "canonical")
        if denied is not None:
            return denied
        fact_row = self.store._connection.execute(
            "SELECT subject_id FROM clinical_facts WHERE fact_id = ?",
            (record.fact_id,),
        ).fetchone()
        if fact_row is None:
            return self._fail(StoreState.PARTIAL, "missing_canonical_fact")
        if fact_row["subject_id"] != record.subject_id:
            return self._fail(StoreState.CONFLICT, "canonical_subject_mismatch")
        payload_json = record.to_json()
        payload_hash = canonical_digest(record.to_dict())
        existing = self.store._connection.execute(
            """
            SELECT version, created_revision FROM canonical_record_versions
            WHERE canonical_id = ? AND payload_hash = ?
            """,
            (record.canonical_id, payload_hash),
        ).fetchone()
        if existing is not None:
            version = CanonicalRecordVersion(
                record=record,
                version=int(existing["version"]),
                revision=int(existing["created_revision"]),
            )
            return StoreResult.success(
                version, created=False, revision=version.revision
            )
        row = self.store._connection.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 AS version "
            "FROM canonical_record_versions WHERE canonical_id = ?",
            (record.canonical_id,),
        ).fetchone()
        version_number = int(row["version"])
        self.store._connection.execute(
            """
            INSERT INTO canonical_record_versions(
                canonical_id, version, subject_id, fact_id, payload_hash,
                payload_json, created_revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.canonical_id,
                version_number,
                record.subject_id,
                record.fact_id,
                payload_hash,
                payload_json,
                self.revision,
            ),
        )
        version = CanonicalRecordVersion(
            record=record,
            version=version_number,
            revision=self.revision,
        )
        self.mutated = True
        return StoreResult.success(version, created=True, revision=self.revision)

    def put_job(self, job: JobMetadata) -> StoreResult[JobMetadata]:
        """Append a PHI-free job-state version idempotently."""

        denied = self._denied("write", "job")
        if denied is not None:
            return denied
        payload_json = job.to_json()
        payload_hash = canonical_digest(job.to_dict())
        existing = self.store._connection.execute(
            """
            SELECT payload_json, created_revision FROM job_metadata_versions
            WHERE job_id = ? AND payload_hash = ?
            """,
            (job.job_id, payload_hash),
        ).fetchone()
        if existing is not None:
            return StoreResult.success(
                JobMetadata.from_dict(json.loads(existing["payload_json"])),
                created=False,
                revision=int(existing["created_revision"]),
            )
        row = self.store._connection.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 AS version "
            "FROM job_metadata_versions WHERE job_id = ?",
            (job.job_id,),
        ).fetchone()
        self.store._connection.execute(
            """
            INSERT INTO job_metadata_versions(
                job_id, version, state, payload_hash, payload_json, created_revision
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job.job_id,
                int(row["version"]),
                job.state,
                payload_hash,
                payload_json,
                self.revision,
            ),
        )
        self.mutated = True
        return StoreResult.success(job, created=True, revision=self.revision)

    def _insert_simple(
        self,
        *,
        table: str,
        id_column: str,
        record_id: str,
        payload_json: str,
        value: T,
        extra_columns: tuple[str, ...] = (),
        extra_values: tuple[Any, ...] = (),
    ) -> StoreResult[T]:
        existing_id = self.store._connection.execute(
            f"SELECT payload_json, created_revision FROM {table} WHERE {id_column} = ?",  # noqa: S608
            (record_id,),
        ).fetchone()
        parser = cast(Callable[[str], T], getattr(type(value), "from_json"))
        if existing_id is not None:
            return self._existing(existing_id, value, parser)
        payload = json.loads(payload_json)
        payload_hash = canonical_digest(payload)
        existing_hash = self.store._connection.execute(
            f"SELECT payload_json, created_revision FROM {table} WHERE payload_hash = ?",  # noqa: S608
            (payload_hash,),
        ).fetchone()
        if existing_hash is not None:
            return self._existing(existing_hash, value, parser, require_equal=False)
        columns = (
            id_column,
            *extra_columns,
            "payload_hash",
            "payload_json",
            "created_revision",
        )
        placeholders = ",".join("?" for _ in columns)
        try:
            self.store._connection.execute(
                f"INSERT INTO {table}({','.join(columns)}) VALUES ({placeholders})",  # noqa: S608
                (record_id, *extra_values, payload_hash, payload_json, self.revision),
            )
        except (sqlite3.IntegrityError, StoreConstraintError):
            return self._fail(StoreState.CONFLICT, "record_conflict")
        self.mutated = True
        return StoreResult.success(value, created=True, revision=self.revision)

    def _existing(
        self,
        row: sqlite3.Row,
        requested: T,
        parser: Callable[[str], T],
        *,
        require_equal: bool = True,
    ) -> StoreResult[T]:
        try:
            existing = parser(row["payload_json"])
        except (ValueError, TypeError, json.JSONDecodeError):
            return self._fail(StoreState.FAILURE, "stored_payload_invalid")
        if require_equal and existing != requested:
            return self._fail(StoreState.CONFLICT, "record_conflict")
        revision = (
            int(row["created_revision"]) if "created_revision" in row.keys() else None
        )
        return StoreResult.success(existing, created=False, revision=revision)

    def _exists(self, table: str, column: str, value: str) -> bool:
        return (
            self.store._connection.execute(
                f"SELECT 1 FROM {table} WHERE {column} = ?",  # noqa: S608
                (value,),
            ).fetchone()
            is not None
        )

    def _denied(self, operation: str, record_type: str) -> StoreResult[Any] | None:
        denied = self.store._denied(operation, record_type)
        if denied is not None:
            self.failed = True
        return denied

    def _fail(self, state: StoreState, code: str) -> StoreResult[Any]:
        self.failed = True
        return StoreResult.outcome(state, code, revision=self.revision)


class LocalJourneyStore:
    """Facade joining verified blobs to one atomic SQLite metadata graph."""

    def __init__(
        self,
        root: str | Path,
        *,
        policy: StoragePolicy | None = None,
    ) -> None:
        root_path = Path(root)
        self.artifacts = LocalArtifactStore(root_path / "artifacts", policy=policy)
        self.metadata = SQLiteJourneyStore(root_path / "journey.sqlite3", policy=policy)

    def close(self) -> None:
        """Close the metadata store."""

        self.metadata.close()

    def __enter__(self) -> "LocalJourneyStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def ingest_graph(
        self,
        artifact: ClinicalArtifact,
        content: bytes,
        *,
        evidence: Iterable[EvidenceLocator] = (),
        facts: Iterable[ClinicalFact] = (),
        conflicts: Iterable[ConflictSet] = (),
        committed_at: str,
    ) -> StoreResult[IngestedGraph]:
        """Atomically persist one verified source-to-fact metadata graph."""

        locators = tuple(evidence)
        fact_records = tuple(facts)
        conflict_records = tuple(conflicts)
        blob = self.artifacts.put_bytes(artifact, content)
        if not blob.ok:
            return StoreResult.outcome(blob.state, blob.code or "artifact_write_failed")

        failure: StoreResult[Any] | None = None
        created_metadata = False
        try:
            with self.metadata.transaction(committed_at=committed_at) as transaction:
                artifact_result = transaction.put_artifact(artifact)
                if not artifact_result.ok:
                    failure = artifact_result
                else:
                    created_metadata = created_metadata or artifact_result.created
                for locator in locators:
                    if failure is not None:
                        break
                    evidence_result = transaction.put_evidence(locator)
                    if not evidence_result.ok:
                        failure = evidence_result
                    else:
                        created_metadata = created_metadata or evidence_result.created
                for fact in fact_records:
                    if failure is not None:
                        break
                    fact_result = transaction.put_fact(fact)
                    if not fact_result.ok:
                        failure = fact_result
                    else:
                        created_metadata = created_metadata or fact_result.created
                for conflict in conflict_records:
                    if failure is not None:
                        break
                    conflict_result = transaction.put_conflict(conflict)
                    if not conflict_result.ok:
                        failure = conflict_result
                    else:
                        created_metadata = created_metadata or conflict_result.created
        except (LocalStoreError, sqlite3.Error, TypeError, ValueError):
            failure = StoreResult.outcome(StoreState.FAILURE, "transaction_failed")

        if failure is not None:
            if blob.created:
                self.artifacts.discard_if_created(artifact.content_hash)
            return StoreResult.outcome(
                failure.state, failure.code or "transaction_failed"
            )
        graph = IngestedGraph(
            artifact=artifact,
            evidence_count=len(locators),
            fact_count=len(fact_records),
            conflict_count=len(conflict_records),
        )
        return StoreResult.success(
            graph,
            created=blob.created or created_metadata,
            revision=(
                transaction.revision
                if created_metadata
                else self.metadata.latest_revision
            ),
        )


def _valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        return False
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _valid_opaque_id(value: Any) -> bool:
    return isinstance(value, str) and _OPAQUE_ID_RE.fullmatch(value) is not None


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        return
    finally:
        os.close(descriptor)
