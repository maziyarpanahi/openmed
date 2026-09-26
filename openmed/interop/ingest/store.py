"""Durable SQLite and PostgreSQL ledgers for resumable ingestion jobs."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

from openmed.clinical.journey_contracts import canonical_digest
from openmed.structured.store import (
    CommitStatusUnknown,
    PostgresJourneyStore,
    SQLiteJourneyStore,
    StoreResult,
    StoreState,
)
from openmed.structured.store.local import StoreConstraintError

from .contracts import (
    Cancellation,
    Checkpoint,
    IngestionContractError,
    IngestionJob,
    Lease,
    QuarantinePromotion,
    QuarantineResult,
    ReplayAudit,
    Retry,
    SourceManifest,
)
from .pipeline_contracts import (
    PipelineLineageError,
    PipelineStageInvalidation,
    PipelineStageManifest,
    build_stage_invalidation,
)

T = TypeVar("T")

_TERMINAL_STATES = frozenset({"cancelled", "completed", "failed"})
_POSTGRES_LEDGER_LOCK_ID = 613_753_210


@dataclass(frozen=True, slots=True)
class ManifestRegistration:
    """A manifest registration plus its durable replay audit."""

    manifest: SourceManifest
    job: IngestionJob
    audit: ReplayAudit


@runtime_checkable
class IngestionLedger(Protocol):
    """Backend-neutral durable ingestion coordination surface."""

    def register_manifest(
        self,
        manifest: SourceManifest,
        *,
        recorded_at: str,
    ) -> StoreResult[ManifestRegistration]:
        """Register or audit an unchanged manifest replay."""

    def get_job(self, job_id: str) -> StoreResult[IngestionJob]:
        """Read the latest job state."""

    def acquire_lease(
        self,
        job_id: str,
        worker_id: str,
        *,
        acquired_at: str,
        duration_seconds: int,
    ) -> StoreResult[Lease]:
        """Acquire one exclusive live lease."""

    def commit_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        lease_id: str,
        recorded_at: str,
    ) -> StoreResult[Checkpoint]:
        """Commit one idempotent step boundary."""

    def get_step_checkpoint(
        self,
        job_id: str,
        step: str,
        input_digest: str,
    ) -> StoreResult[Checkpoint]:
        """Read a completed step by its idempotency identity."""

    def complete_job(
        self,
        job_id: str,
        *,
        lease_id: str,
        completed_at: str,
    ) -> StoreResult[IngestionJob]:
        """Complete a job under its live lease."""

    def record_retry(
        self,
        retry: Retry,
        *,
        lease_id: str,
    ) -> StoreResult[Retry]:
        """Persist one retry classification."""

    def cancel_job(self, cancellation: Cancellation) -> StoreResult[Cancellation]:
        """Cancel a job explicitly."""

    def quarantine(
        self,
        result: QuarantineResult,
        *,
        lease_id: str,
    ) -> StoreResult[QuarantineResult]:
        """Quarantine untrusted output and stop the job."""

    def promote_quarantine(
        self,
        promotion: QuarantinePromotion,
    ) -> StoreResult[QuarantinePromotion]:
        """Explicitly promote one quarantined result for resumed processing."""


class _IngestionStoreMixin:
    _connection: Any
    _lock: threading.RLock
    _raw_connection: Any
    policy: Any

    def _ledger_transaction(self) -> AbstractContextManager[None]:
        raise NotImplementedError

    def register_manifest(
        self,
        manifest: SourceManifest,
        *,
        recorded_at: str,
    ) -> StoreResult[ManifestRegistration]:
        """Register a manifest or record an auditable unchanged replay."""

        denied = self._ingestion_denied("write")
        if denied is not None:
            return denied
        try:
            _parse_time(recorded_at, "recorded_at")
            with self._ledger_transaction():
                existing_row = self._connection.execute(
                    "SELECT payload_json FROM ingestion_manifests "
                    "WHERE manifest_digest = ?",
                    (manifest.manifest_digest,),
                ).fetchone()
                created = existing_row is None
                if created:
                    conflicting = self._connection.execute(
                        "SELECT manifest_digest FROM ingestion_manifests "
                        "WHERE manifest_id = ?",
                        (manifest.manifest_id,),
                    ).fetchone()
                    if conflicting is not None:
                        return StoreResult.outcome(
                            StoreState.CONFLICT,
                            "manifest_id_conflict",
                        )
                    self._insert_payload(
                        "ingestion_manifests",
                        (
                            "manifest_digest",
                            "manifest_id",
                            "payload_hash",
                            "payload_json",
                        ),
                        (
                            manifest.manifest_digest,
                            manifest.manifest_id,
                            canonical_digest(manifest.to_dict()),
                            manifest.to_json(),
                        ),
                    )
                    job = IngestionJob(
                        job_id=_derived_id("job", manifest.manifest_digest),
                        manifest_digest=manifest.manifest_digest,
                        state="queued",
                        checkpoint_sequence=0,
                        created_at=recorded_at,
                        updated_at=recorded_at,
                    )
                    self._connection.execute(
                        "INSERT INTO ingestion_jobs(job_id, manifest_digest) "
                        "VALUES (?, ?)",
                        (job.job_id, job.manifest_digest),
                    )
                    self._append_job_version(job, recorded_at)
                    stored_manifest = manifest
                else:
                    stored_manifest = self._parse_record(
                        existing_row,
                        SourceManifest.from_json,
                    )
                    job_row = self._connection.execute(
                        "SELECT job_id FROM ingestion_jobs WHERE manifest_digest = ?",
                        (manifest.manifest_digest,),
                    ).fetchone()
                    if job_row is None:
                        return StoreResult.outcome(
                            StoreState.FAILURE,
                            "stored_job_missing",
                        )
                    stored_job = self._get_job_locked(str(job_row["job_id"]))
                    if stored_job is None:
                        return StoreResult.outcome(
                            StoreState.FAILURE,
                            "stored_job_invalid",
                        )
                    job = stored_job

                audit = ReplayAudit(
                    replay_id=f"replay_{uuid.uuid4().hex}",
                    manifest_digest=manifest.manifest_digest,
                    job_id=job.job_id,
                    action="created" if created else "noop",
                    recorded_at=recorded_at,
                )
                self._insert_payload(
                    "ingestion_replay_audits",
                    (
                        "replay_id",
                        "manifest_digest",
                        "job_id",
                        "action",
                        "recorded_at",
                        "payload_hash",
                        "payload_json",
                    ),
                    (
                        audit.replay_id,
                        audit.manifest_digest,
                        audit.job_id,
                        audit.action,
                        audit.recorded_at,
                        canonical_digest(audit.to_dict()),
                        audit.to_json(),
                    ),
                )
            return StoreResult.success(
                ManifestRegistration(stored_manifest, job, audit),
                created=created,
            )
        except CommitStatusUnknown:
            return StoreResult.outcome(StoreState.UNKNOWN, "commit_status_unknown")
        except (IngestionContractError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_manifest")
        except (sqlite3.Error, StoreConstraintError, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "manifest_write_failed")

    def get_job(self, job_id: str) -> StoreResult[IngestionJob]:
        """Return the latest append-only job version."""

        denied = self._ingestion_denied("read")
        if denied is not None:
            return denied
        try:
            with self._lock:
                job = self._get_job_locked(job_id)
        except (IngestionContractError, TypeError, ValueError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_job_invalid")
        except (sqlite3.Error, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "job_read_failed")
        if job is None:
            return StoreResult.outcome(StoreState.UNKNOWN, "job_not_found")
        return StoreResult.success(job)

    def acquire_lease(
        self,
        job_id: str,
        worker_id: str,
        *,
        acquired_at: str,
        duration_seconds: int,
    ) -> StoreResult[Lease]:
        """Acquire or idempotently reuse one worker's live lease."""

        denied = self._ingestion_denied("write")
        if denied is not None:
            return denied
        try:
            acquired = _parse_time(acquired_at, "acquired_at")
            if type(duration_seconds) is not int or duration_seconds < 1:
                raise ValueError("duration_seconds must be positive")
            with self._ledger_transaction():
                job = self._get_job_locked(job_id)
                if job is None:
                    return StoreResult.outcome(StoreState.UNKNOWN, "job_not_found")
                if job.state in _TERMINAL_STATES:
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "job_not_leasable",
                    )
                if job.state == "quarantined":
                    return StoreResult.outcome(
                        StoreState.DENIED,
                        "quarantine_promotion_required",
                    )
                active = self._latest_unreleased_lease(job_id)
                if (
                    active is not None
                    and _parse_time(
                        active.expires_at,
                        "expires_at",
                    )
                    > acquired
                ):
                    if active.worker_id == worker_id:
                        return StoreResult.success(active, created=False)
                    return StoreResult.outcome(StoreState.CONFLICT, "lease_held")
                if active is not None:
                    self._release_lease_locked(active.lease_id, acquired_at)
                epoch_row = self._connection.execute(
                    "SELECT COALESCE(MAX(epoch), 0) + 1 AS epoch "
                    "FROM ingestion_leases WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                if epoch_row is None:
                    return StoreResult.outcome(
                        StoreState.FAILURE,
                        "lease_epoch_failed",
                    )
                epoch = int(epoch_row["epoch"])
                expires_at = _format_time(
                    acquired + timedelta(seconds=duration_seconds)
                )
                lease = Lease(
                    lease_id=_derived_id("lease", job_id, worker_id, str(epoch)),
                    job_id=job_id,
                    worker_id=worker_id,
                    epoch=epoch,
                    acquired_at=acquired_at,
                    expires_at=expires_at,
                )
                self._insert_payload(
                    "ingestion_leases",
                    (
                        "lease_id",
                        "job_id",
                        "worker_id",
                        "epoch",
                        "acquired_at",
                        "expires_at",
                        "payload_hash",
                        "payload_json",
                    ),
                    (
                        lease.lease_id,
                        lease.job_id,
                        lease.worker_id,
                        lease.epoch,
                        lease.acquired_at,
                        lease.expires_at,
                        canonical_digest(lease.to_dict()),
                        lease.to_json(),
                    ),
                )
                if job.state != "running":
                    self._append_job_version(
                        replace(job, state="running", updated_at=acquired_at),
                        acquired_at,
                    )
            return StoreResult.success(lease, created=True)
        except CommitStatusUnknown:
            return StoreResult.outcome(StoreState.UNKNOWN, "commit_status_unknown")
        except (IngestionContractError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_lease")
        except (sqlite3.Error, StoreConstraintError, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "lease_write_failed")

    def get_active_lease(self, job_id: str, *, at: str) -> StoreResult[Lease]:
        """Return the live unreleased lease at one explicit time."""

        denied = self._ingestion_denied("read")
        if denied is not None:
            return denied
        try:
            instant = _parse_time(at, "at")
            with self._lock:
                lease = self._latest_unreleased_lease(job_id)
            if lease is None or _parse_time(lease.expires_at, "expires_at") <= instant:
                return StoreResult.outcome(StoreState.UNKNOWN, "live_lease_not_found")
            return StoreResult.success(lease)
        except (IngestionContractError, TypeError, ValueError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_lease_invalid")
        except (sqlite3.Error, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "lease_read_failed")

    def release_lease(
        self,
        lease_id: str,
        *,
        released_at: str,
    ) -> StoreResult[Lease]:
        """Release one lease idempotently."""

        denied = self._ingestion_denied("write")
        if denied is not None:
            return denied
        try:
            _parse_time(released_at, "released_at")
            with self._ledger_transaction():
                lease = self._get_lease_locked(lease_id)
                if lease is None:
                    return StoreResult.outcome(StoreState.UNKNOWN, "lease_not_found")
                existing = self._connection.execute(
                    "SELECT released_at FROM ingestion_lease_releases "
                    "WHERE lease_id = ?",
                    (lease_id,),
                ).fetchone()
                if existing is not None:
                    return StoreResult.success(lease, created=False)
                self._connection.execute(
                    "INSERT INTO ingestion_lease_releases(lease_id, released_at) "
                    "VALUES (?, ?)",
                    (lease_id, released_at),
                )
            return StoreResult.success(lease, created=True)
        except CommitStatusUnknown:
            return StoreResult.outcome(StoreState.UNKNOWN, "commit_status_unknown")
        except (IngestionContractError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_lease_release")
        except (sqlite3.Error, StoreConstraintError, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "lease_release_failed")

    def get_step_checkpoint(
        self,
        job_id: str,
        step: str,
        input_digest: str,
    ) -> StoreResult[Checkpoint]:
        """Return an acknowledged step matching its idempotency identity."""

        denied = self._ingestion_denied("read")
        if denied is not None:
            return denied
        try:
            row = self._connection.execute(
                "SELECT payload_json FROM ingestion_checkpoints "
                "WHERE job_id = ? AND step = ? AND input_digest = ?",
                (job_id, step, input_digest),
            ).fetchone()
            if row is None:
                return StoreResult.outcome(StoreState.UNKNOWN, "checkpoint_not_found")
            return StoreResult.success(
                self._parse_record(row, Checkpoint.from_json),
                created=False,
            )
        except (IngestionContractError, TypeError, ValueError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_checkpoint_invalid")
        except (sqlite3.Error, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "checkpoint_read_failed")

    def commit_checkpoint(
        self,
        checkpoint: Checkpoint,
        *,
        lease_id: str,
        recorded_at: str,
    ) -> StoreResult[Checkpoint]:
        """Commit one step exactly once at the logical checkpoint boundary."""

        denied = self._ingestion_denied("write")
        if denied is not None:
            return denied
        try:
            instant = _parse_time(recorded_at, "recorded_at")
            with self._ledger_transaction():
                lease_result = self._live_lease_locked(
                    checkpoint.job_id,
                    lease_id,
                    instant,
                )
                if lease_result is not None:
                    return lease_result
                job = self._get_job_locked(checkpoint.job_id)
                if job is None:
                    return StoreResult.outcome(StoreState.UNKNOWN, "job_not_found")
                if checkpoint.manifest_digest != job.manifest_digest:
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "checkpoint_manifest_mismatch",
                    )
                existing = self._connection.execute(
                    "SELECT payload_json FROM ingestion_checkpoints "
                    "WHERE checkpoint_id = ? OR (job_id = ? AND sequence = ?)",
                    (
                        checkpoint.checkpoint_id,
                        checkpoint.job_id,
                        checkpoint.sequence,
                    ),
                ).fetchone()
                if existing is not None:
                    stored = self._parse_record(existing, Checkpoint.from_json)
                    if stored == checkpoint:
                        return StoreResult.success(stored, created=False)
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "checkpoint_conflict",
                    )
                step_row = self._connection.execute(
                    "SELECT payload_json FROM ingestion_checkpoints "
                    "WHERE job_id = ? AND step = ? AND input_digest = ?",
                    (checkpoint.job_id, checkpoint.step, checkpoint.input_digest),
                ).fetchone()
                if step_row is not None:
                    stored = self._parse_record(step_row, Checkpoint.from_json)
                    if (
                        stored.output_digest == checkpoint.output_digest
                        and stored.committed_revision == checkpoint.committed_revision
                    ):
                        return StoreResult.success(stored, created=False)
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "checkpoint_replay_conflict",
                    )
                if checkpoint.sequence != job.checkpoint_sequence + 1:
                    return StoreResult.outcome(StoreState.PARTIAL, "checkpoint_gap")
                self._insert_payload(
                    "ingestion_checkpoints",
                    (
                        "checkpoint_id",
                        "job_id",
                        "sequence",
                        "step",
                        "input_digest",
                        "output_digest",
                        "completed_at",
                        "payload_hash",
                        "payload_json",
                    ),
                    (
                        checkpoint.checkpoint_id,
                        checkpoint.job_id,
                        checkpoint.sequence,
                        checkpoint.step,
                        checkpoint.input_digest,
                        checkpoint.output_digest,
                        checkpoint.completed_at,
                        canonical_digest(checkpoint.to_dict()),
                        checkpoint.to_json(),
                    ),
                )
                self._append_job_version(
                    replace(
                        job,
                        state="running",
                        checkpoint_sequence=checkpoint.sequence,
                        updated_at=recorded_at,
                    ),
                    recorded_at,
                )
            return StoreResult.success(checkpoint, created=True)
        except CommitStatusUnknown:
            return StoreResult.outcome(StoreState.UNKNOWN, "commit_status_unknown")
        except (IngestionContractError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_checkpoint")
        except (sqlite3.Error, StoreConstraintError, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "checkpoint_write_failed")

    def complete_job(
        self,
        job_id: str,
        *,
        lease_id: str,
        completed_at: str,
    ) -> StoreResult[IngestionJob]:
        """Mark a leased job complete and release the lease atomically."""

        denied = self._ingestion_denied("write")
        if denied is not None:
            return denied
        try:
            instant = _parse_time(completed_at, "completed_at")
            with self._ledger_transaction():
                job = self._get_job_locked(job_id)
                if job is None:
                    return StoreResult.outcome(StoreState.UNKNOWN, "job_not_found")
                if job.state == "completed":
                    return StoreResult.success(job, created=False)
                lease_result = self._live_lease_locked(job_id, lease_id, instant)
                if lease_result is not None:
                    return lease_result
                if job.state != "running":
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "job_not_running",
                    )
                completed = replace(job, state="completed", updated_at=completed_at)
                self._append_job_version(completed, completed_at)
                self._release_lease_locked(lease_id, completed_at)
            return StoreResult.success(completed, created=True)
        except CommitStatusUnknown:
            return StoreResult.outcome(StoreState.UNKNOWN, "commit_status_unknown")
        except (IngestionContractError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_job_completion")
        except (sqlite3.Error, StoreConstraintError, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "job_completion_failed")

    def record_retry(
        self,
        retry: Retry,
        *,
        lease_id: str,
    ) -> StoreResult[Retry]:
        """Persist one stable retry classification and release its lease."""

        denied = self._ingestion_denied("write")
        if denied is not None:
            return denied
        try:
            instant = _parse_time(retry.recorded_at, "recorded_at")
            with self._ledger_transaction():
                lease_result = self._live_lease_locked(retry.job_id, lease_id, instant)
                if lease_result is not None:
                    return lease_result
                existing = self._connection.execute(
                    "SELECT payload_json FROM ingestion_retries WHERE retry_id = ?",
                    (retry.retry_id,),
                ).fetchone()
                if existing is not None:
                    stored = self._parse_record(existing, Retry.from_json)
                    if stored == retry:
                        return StoreResult.success(stored, created=False)
                    return StoreResult.outcome(StoreState.CONFLICT, "retry_conflict")
                self._insert_payload(
                    "ingestion_retries",
                    (
                        "retry_id",
                        "job_id",
                        "classification",
                        "attempt",
                        "recorded_at",
                        "payload_hash",
                        "payload_json",
                    ),
                    (
                        retry.retry_id,
                        retry.job_id,
                        retry.classification,
                        retry.attempt,
                        retry.recorded_at,
                        canonical_digest(retry.to_dict()),
                        retry.to_json(),
                    ),
                )
                self._release_lease_locked(lease_id, retry.recorded_at)
            return StoreResult.success(retry, created=True)
        except CommitStatusUnknown:
            return StoreResult.outcome(StoreState.UNKNOWN, "commit_status_unknown")
        except (IngestionContractError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_retry")
        except (sqlite3.Error, StoreConstraintError, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "retry_write_failed")

    def cancel_job(self, cancellation: Cancellation) -> StoreResult[Cancellation]:
        """Persist an explicit cancellation and stop future work."""

        denied = self._ingestion_denied("write")
        if denied is not None:
            return denied
        try:
            _parse_time(cancellation.requested_at, "requested_at")
            with self._ledger_transaction():
                job = self._get_job_locked(cancellation.job_id)
                if job is None:
                    return StoreResult.outcome(StoreState.UNKNOWN, "job_not_found")
                existing = self._connection.execute(
                    "SELECT payload_json FROM ingestion_cancellations WHERE job_id = ?",
                    (cancellation.job_id,),
                ).fetchone()
                if existing is not None:
                    stored = self._parse_record(existing, Cancellation.from_json)
                    if stored == cancellation:
                        return StoreResult.success(stored, created=False)
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "cancellation_conflict",
                    )
                if job.state == "completed":
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "completed_job_cannot_cancel",
                    )
                self._insert_payload(
                    "ingestion_cancellations",
                    (
                        "cancellation_id",
                        "job_id",
                        "requested_at",
                        "payload_hash",
                        "payload_json",
                    ),
                    (
                        cancellation.cancellation_id,
                        cancellation.job_id,
                        cancellation.requested_at,
                        canonical_digest(cancellation.to_dict()),
                        cancellation.to_json(),
                    ),
                )
                self._append_job_version(
                    replace(
                        job,
                        state="cancelled",
                        updated_at=cancellation.requested_at,
                    ),
                    cancellation.requested_at,
                )
                active = self._latest_unreleased_lease(cancellation.job_id)
                if active is not None:
                    self._release_lease_locked(
                        active.lease_id,
                        cancellation.requested_at,
                    )
            return StoreResult.success(cancellation, created=True)
        except CommitStatusUnknown:
            return StoreResult.outcome(StoreState.UNKNOWN, "commit_status_unknown")
        except (IngestionContractError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_cancellation")
        except (sqlite3.Error, StoreConstraintError, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "cancellation_write_failed")

    def quarantine(
        self,
        result: QuarantineResult,
        *,
        lease_id: str,
    ) -> StoreResult[QuarantineResult]:
        """Persist untrusted output without promoting any candidate fact."""

        denied = self._ingestion_denied("write")
        if denied is not None:
            return denied
        try:
            instant = _parse_time(result.created_at, "created_at")
            with self._ledger_transaction():
                lease_result = self._live_lease_locked(
                    result.job_id,
                    lease_id,
                    instant,
                )
                if lease_result is not None:
                    return lease_result
                job = self._get_job_locked(result.job_id)
                if job is None:
                    return StoreResult.outcome(StoreState.UNKNOWN, "job_not_found")
                if result.manifest_digest != job.manifest_digest:
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "quarantine_manifest_mismatch",
                    )
                existing = self._connection.execute(
                    "SELECT payload_json FROM ingestion_quarantine_results "
                    "WHERE quarantine_id = ?",
                    (result.quarantine_id,),
                ).fetchone()
                if existing is not None:
                    stored = self._parse_record(existing, QuarantineResult.from_json)
                    if stored == result:
                        return StoreResult.success(stored, created=False)
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "quarantine_conflict",
                    )
                self._insert_payload(
                    "ingestion_quarantine_results",
                    (
                        "quarantine_id",
                        "job_id",
                        "classification",
                        "created_at",
                        "payload_hash",
                        "payload_json",
                    ),
                    (
                        result.quarantine_id,
                        result.job_id,
                        result.classification,
                        result.created_at,
                        canonical_digest(result.to_dict()),
                        result.to_json(),
                    ),
                )
                self._append_job_version(
                    replace(job, state="quarantined", updated_at=result.created_at),
                    result.created_at,
                )
                self._release_lease_locked(lease_id, result.created_at)
            return StoreResult.success(result, created=True)
        except CommitStatusUnknown:
            return StoreResult.outcome(StoreState.UNKNOWN, "commit_status_unknown")
        except (IngestionContractError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_quarantine")
        except (sqlite3.Error, StoreConstraintError, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "quarantine_write_failed")

    def get_quarantine(self, quarantine_id: str) -> StoreResult[QuarantineResult]:
        """Read one quarantined result by opaque identifier."""

        denied = self._ingestion_denied("read")
        if denied is not None:
            return denied
        try:
            row = self._connection.execute(
                "SELECT payload_json FROM ingestion_quarantine_results "
                "WHERE quarantine_id = ?",
                (quarantine_id,),
            ).fetchone()
            if row is None:
                return StoreResult.outcome(StoreState.UNKNOWN, "quarantine_not_found")
            return StoreResult.success(
                self._parse_record(row, QuarantineResult.from_json)
            )
        except (IngestionContractError, TypeError, ValueError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_quarantine_invalid")
        except (sqlite3.Error, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "quarantine_read_failed")

    def promote_quarantine(
        self,
        promotion: QuarantinePromotion,
    ) -> StoreResult[QuarantinePromotion]:
        """Record explicit review promotion before processing can resume."""

        denied = self._ingestion_denied("write")
        if denied is not None:
            return denied
        try:
            _parse_time(promotion.promoted_at, "promoted_at")
            with self._ledger_transaction():
                quarantine_row = self._connection.execute(
                    "SELECT job_id FROM ingestion_quarantine_results "
                    "WHERE quarantine_id = ?",
                    (promotion.quarantine_id,),
                ).fetchone()
                if quarantine_row is None:
                    return StoreResult.outcome(
                        StoreState.UNKNOWN,
                        "quarantine_not_found",
                    )
                existing = self._connection.execute(
                    "SELECT payload_json FROM ingestion_quarantine_promotions "
                    "WHERE quarantine_id = ?",
                    (promotion.quarantine_id,),
                ).fetchone()
                if existing is not None:
                    stored = self._parse_record(
                        existing,
                        QuarantinePromotion.from_json,
                    )
                    if stored == promotion:
                        return StoreResult.success(stored, created=False)
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "promotion_conflict",
                    )
                job = self._get_job_locked(str(quarantine_row["job_id"]))
                if job is None:
                    return StoreResult.outcome(StoreState.FAILURE, "stored_job_missing")
                if job.state != "quarantined":
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "job_not_quarantined",
                    )
                self._insert_payload(
                    "ingestion_quarantine_promotions",
                    (
                        "promotion_id",
                        "quarantine_id",
                        "promoted_at",
                        "payload_hash",
                        "payload_json",
                    ),
                    (
                        promotion.promotion_id,
                        promotion.quarantine_id,
                        promotion.promoted_at,
                        canonical_digest(promotion.to_dict()),
                        promotion.to_json(),
                    ),
                )
                self._append_job_version(
                    replace(job, state="queued", updated_at=promotion.promoted_at),
                    promotion.promoted_at,
                )
            return StoreResult.success(promotion, created=True)
        except CommitStatusUnknown:
            return StoreResult.outcome(StoreState.UNKNOWN, "commit_status_unknown")
        except (IngestionContractError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_promotion")
        except (sqlite3.Error, StoreConstraintError, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "promotion_write_failed")

    def ingestion_integrity_check(self) -> StoreResult[Mapping[str, int]]:
        """Validate ingestion payload hashes and return counts only."""

        denied = self._ingestion_denied("verify")
        if denied is not None:
            return denied
        parsers: tuple[tuple[str, Callable[[str], Any]], ...] = (
            ("ingestion_manifests", SourceManifest.from_json),
            ("ingestion_job_versions", IngestionJob.from_json),
            ("ingestion_replay_audits", ReplayAudit.from_json),
            ("ingestion_leases", Lease.from_json),
            ("ingestion_checkpoints", Checkpoint.from_json),
            ("ingestion_retries", Retry.from_json),
            ("ingestion_cancellations", Cancellation.from_json),
            ("ingestion_quarantine_results", QuarantineResult.from_json),
            ("ingestion_quarantine_promotions", QuarantinePromotion.from_json),
            ("ingestion_pipeline_stages", PipelineStageManifest.from_json),
            (
                "ingestion_pipeline_invalidations",
                PipelineStageInvalidation.from_json,
            ),
        )
        counts: dict[str, int] = {}
        try:
            for table, parser in parsers:
                rows = self._connection.execute(
                    f"SELECT payload_hash, payload_json FROM {table}"  # noqa: S608
                ).fetchall()
                for row in rows:
                    record = parser(row["payload_json"])
                    if canonical_digest(record.to_dict()) != row["payload_hash"]:
                        return StoreResult.outcome(
                            StoreState.FAILURE,
                            "payload_hash_mismatch",
                        )
                counts[table] = len(rows)
            edge_rows = self._connection.execute(
                "SELECT parent_stage_manifest_id, child_stage_manifest_id "
                "FROM ingestion_pipeline_edges"
            ).fetchall()
            counts["ingestion_pipeline_edges"] = len(edge_rows)
        except (
            IngestionContractError,
            PipelineLineageError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return StoreResult.outcome(StoreState.FAILURE, "stored_payload_invalid")
        except (sqlite3.Error, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "integrity_check_failed")
        return StoreResult.success(counts)

    def put_pipeline_stage(
        self,
        manifest: PipelineStageManifest,
    ) -> StoreResult[PipelineStageManifest]:
        """Persist one safe stage manifest and its derivation edges."""

        denied = self._ingestion_denied("write")
        if denied is not None:
            return denied
        try:
            input_digest = canonical_digest(list(manifest.input_digests))
            output_digest = (
                canonical_digest(list(manifest.output_digests))
                if manifest.output_digests
                else None
            )
            with self._ledger_transaction():
                if self._get_job_locked(manifest.job_id) is None:
                    return StoreResult.outcome(StoreState.UNKNOWN, "job_not_found")
                existing = self._connection.execute(
                    "SELECT payload_json FROM ingestion_pipeline_stages "
                    "WHERE stage_manifest_id = ? OR "
                    "(job_id = ? AND stage = ? AND input_digest = ?)",
                    (
                        manifest.stage_manifest_id,
                        manifest.job_id,
                        manifest.stage,
                        input_digest,
                    ),
                ).fetchone()
                if existing is not None:
                    stored = self._parse_record(
                        existing,
                        PipelineStageManifest.from_json,
                    )
                    if stored == manifest:
                        return StoreResult.success(stored, created=False)
                    return StoreResult.outcome(
                        StoreState.CONFLICT,
                        "stage_manifest_conflict",
                    )
                for parent_id in manifest.parent_stage_manifest_ids:
                    parent = self._connection.execute(
                        "SELECT 1 FROM ingestion_pipeline_stages "
                        "WHERE stage_manifest_id = ?",
                        (parent_id,),
                    ).fetchone()
                    if parent is None:
                        return StoreResult.outcome(
                            StoreState.PARTIAL,
                            "stage_parent_missing",
                        )
                self._insert_payload(
                    "ingestion_pipeline_stages",
                    (
                        "stage_manifest_id",
                        "job_id",
                        "stage",
                        "sequence",
                        "state",
                        "input_digest",
                        "output_digest",
                        "recorded_at",
                        "payload_hash",
                        "payload_json",
                    ),
                    (
                        manifest.stage_manifest_id,
                        manifest.job_id,
                        manifest.stage,
                        manifest.sequence,
                        manifest.state,
                        input_digest,
                        output_digest,
                        manifest.recorded_at,
                        manifest.manifest_digest,
                        manifest.to_json(),
                    ),
                )
                if manifest.parent_stage_manifest_ids:
                    self._connection.executemany(
                        "INSERT INTO ingestion_pipeline_edges("
                        "parent_stage_manifest_id, child_stage_manifest_id"
                        ") VALUES (?, ?)",
                        tuple(
                            (parent_id, manifest.stage_manifest_id)
                            for parent_id in manifest.parent_stage_manifest_ids
                        ),
                    )
            return StoreResult.success(manifest, created=True)
        except CommitStatusUnknown:
            return StoreResult.outcome(StoreState.UNKNOWN, "commit_status_unknown")
        except (PipelineLineageError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_stage_manifest")
        except (sqlite3.Error, StoreConstraintError, RuntimeError):
            return StoreResult.outcome(
                StoreState.FAILURE, "stage_manifest_write_failed"
            )

    def list_pipeline_stages(
        self,
        job_id: str,
    ) -> StoreResult[tuple[PipelineStageManifest, ...]]:
        """List stage manifests for one job in deterministic execution order."""

        denied = self._ingestion_denied("read")
        if denied is not None:
            return denied
        try:
            rows = self._connection.execute(
                "SELECT payload_json FROM ingestion_pipeline_stages "
                "WHERE job_id = ? ORDER BY sequence, stage_manifest_id",
                (job_id,),
            ).fetchall()
            manifests = tuple(
                self._parse_record(row, PipelineStageManifest.from_json) for row in rows
            )
        except (PipelineLineageError, TypeError, ValueError, json.JSONDecodeError):
            return StoreResult.outcome(StoreState.FAILURE, "stored_stage_invalid")
        except (sqlite3.Error, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "stage_manifest_read_failed")
        if not manifests:
            return StoreResult.outcome(StoreState.UNKNOWN, "stage_manifest_not_found")
        return StoreResult.success(manifests)

    def invalidate_pipeline_descendants(
        self,
        job_id: str,
        *,
        from_stage_manifest_id: str,
        replacement_job_id: str,
        recorded_at: str,
    ) -> StoreResult[tuple[PipelineStageInvalidation, ...]]:
        """Invalidate a replaced stage and descendants without touching ancestors."""

        denied = self._ingestion_denied("write")
        if denied is not None:
            return denied
        try:
            _parse_time(recorded_at, "recorded_at")
            with self._ledger_transaction():
                if self._get_job_locked(replacement_job_id) is None:
                    return StoreResult.outcome(
                        StoreState.UNKNOWN,
                        "replacement_job_not_found",
                    )
                rows = self._connection.execute(
                    "SELECT payload_json FROM ingestion_pipeline_stages "
                    "WHERE job_id = ? ORDER BY sequence, stage_manifest_id",
                    (job_id,),
                ).fetchall()
                manifests = {
                    item.stage_manifest_id: item
                    for item in (
                        self._parse_record(row, PipelineStageManifest.from_json)
                        for row in rows
                    )
                }
                if from_stage_manifest_id not in manifests:
                    return StoreResult.outcome(
                        StoreState.UNKNOWN,
                        "stage_manifest_not_found",
                    )
                edge_rows = self._connection.execute(
                    "SELECT parent_stage_manifest_id, child_stage_manifest_id "
                    "FROM ingestion_pipeline_edges"
                ).fetchall()
                children: dict[str, set[str]] = {}
                for row in edge_rows:
                    parent_id = str(row["parent_stage_manifest_id"])
                    child_id = str(row["child_stage_manifest_id"])
                    if child_id in manifests:
                        children.setdefault(parent_id, set()).add(child_id)
                selected: set[str] = set()
                pending = [from_stage_manifest_id]
                while pending:
                    current = pending.pop()
                    if current in selected:
                        continue
                    selected.add(current)
                    pending.extend(sorted(children.get(current, ())))
                invalidations = tuple(
                    build_stage_invalidation(
                        manifests[stage_id],
                        replacement_job_id=replacement_job_id,
                        reason_code="stage_reprocessed",
                        recorded_at=recorded_at,
                    )
                    for stage_id in sorted(
                        selected,
                        key=lambda item: (
                            manifests[item].sequence,
                            item,
                        ),
                    )
                )
                for invalidation in invalidations:
                    existing = self._connection.execute(
                        "SELECT payload_json FROM ingestion_pipeline_invalidations "
                        "WHERE invalidation_id = ? OR "
                        "(stage_manifest_id = ? AND replacement_job_id = ?)",
                        (
                            invalidation.invalidation_id,
                            invalidation.stage_manifest_id,
                            invalidation.replacement_job_id,
                        ),
                    ).fetchone()
                    if existing is not None:
                        stored = self._parse_record(
                            existing,
                            PipelineStageInvalidation.from_json,
                        )
                        if stored != invalidation:
                            return StoreResult.outcome(
                                StoreState.CONFLICT,
                                "stage_invalidation_conflict",
                            )
                        continue
                    self._insert_payload(
                        "ingestion_pipeline_invalidations",
                        (
                            "invalidation_id",
                            "job_id",
                            "stage_manifest_id",
                            "replacement_job_id",
                            "recorded_at",
                            "payload_hash",
                            "payload_json",
                        ),
                        (
                            invalidation.invalidation_id,
                            invalidation.job_id,
                            invalidation.stage_manifest_id,
                            invalidation.replacement_job_id,
                            invalidation.recorded_at,
                            canonical_digest(invalidation.to_dict()),
                            invalidation.to_json(),
                        ),
                    )
            return StoreResult.success(invalidations, created=True)
        except CommitStatusUnknown:
            return StoreResult.outcome(StoreState.UNKNOWN, "commit_status_unknown")
        except (PipelineLineageError, TypeError, ValueError):
            return StoreResult.outcome(StoreState.FAILURE, "invalid_stage_invalidation")
        except (sqlite3.Error, StoreConstraintError, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "stage_invalidation_failed")

    def list_pipeline_invalidations(
        self,
        job_id: str,
    ) -> StoreResult[tuple[PipelineStageInvalidation, ...]]:
        """List stage invalidations for one job."""

        denied = self._ingestion_denied("read")
        if denied is not None:
            return denied
        try:
            rows = self._connection.execute(
                "SELECT payload_json FROM ingestion_pipeline_invalidations "
                "WHERE job_id = ? ORDER BY recorded_at, invalidation_id",
                (job_id,),
            ).fetchall()
            invalidations = tuple(
                self._parse_record(row, PipelineStageInvalidation.from_json)
                for row in rows
            )
        except (PipelineLineageError, TypeError, ValueError, json.JSONDecodeError):
            return StoreResult.outcome(
                StoreState.FAILURE, "stored_invalidation_invalid"
            )
        except (sqlite3.Error, RuntimeError):
            return StoreResult.outcome(StoreState.FAILURE, "invalidation_read_failed")
        return StoreResult.success(invalidations)

    def _get_job_locked(self, job_id: str) -> IngestionJob | None:
        row = self._connection.execute(
            "SELECT payload_json FROM ingestion_job_versions "
            "WHERE job_id = ? ORDER BY version DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        return None if row is None else self._parse_record(row, IngestionJob.from_json)

    def _ingestion_denied(self, operation: str) -> StoreResult[Any] | None:
        if self.policy.allows(operation, "ingestion"):
            return None
        return StoreResult.outcome(StoreState.DENIED, "policy_denied")

    def _append_job_version(
        self,
        job: IngestionJob,
        recorded_at: str,
    ) -> tuple[IngestionJob, bool]:
        payload_hash = canonical_digest(job.to_dict())
        existing = self._connection.execute(
            "SELECT payload_json FROM ingestion_job_versions "
            "WHERE job_id = ? AND payload_hash = ?",
            (job.job_id, payload_hash),
        ).fetchone()
        if existing is not None:
            return self._parse_record(existing, IngestionJob.from_json), False
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 AS version "
            "FROM ingestion_job_versions WHERE job_id = ?",
            (job.job_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("job version allocation failed")
        self._insert_payload(
            "ingestion_job_versions",
            (
                "job_id",
                "version",
                "state",
                "checkpoint_sequence",
                "recorded_at",
                "payload_hash",
                "payload_json",
            ),
            (
                job.job_id,
                int(row["version"]),
                job.state,
                job.checkpoint_sequence,
                recorded_at,
                payload_hash,
                job.to_json(),
            ),
        )
        return job, True

    def _latest_unreleased_lease(self, job_id: str) -> Lease | None:
        row = self._connection.execute(
            """
            SELECT leases.payload_json
            FROM ingestion_leases AS leases
            LEFT JOIN ingestion_lease_releases AS releases
              ON releases.lease_id = leases.lease_id
            WHERE leases.job_id = ? AND releases.lease_id IS NULL
            ORDER BY leases.epoch DESC
            LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        return None if row is None else self._parse_record(row, Lease.from_json)

    def _get_lease_locked(self, lease_id: str) -> Lease | None:
        row = self._connection.execute(
            "SELECT payload_json FROM ingestion_leases WHERE lease_id = ?",
            (lease_id,),
        ).fetchone()
        return None if row is None else self._parse_record(row, Lease.from_json)

    def _live_lease_locked(
        self,
        job_id: str,
        lease_id: str,
        instant: datetime,
    ) -> StoreResult[Any] | None:
        lease = self._get_lease_locked(lease_id)
        if lease is None or lease.job_id != job_id:
            return StoreResult.outcome(StoreState.DENIED, "live_lease_required")
        release = self._connection.execute(
            "SELECT 1 FROM ingestion_lease_releases WHERE lease_id = ?",
            (lease_id,),
        ).fetchone()
        if (
            release is not None
            or _parse_time(lease.expires_at, "expires_at") <= instant
        ):
            return StoreResult.outcome(StoreState.DENIED, "live_lease_required")
        active = self._latest_unreleased_lease(job_id)
        if active is None or active.lease_id != lease_id:
            return StoreResult.outcome(StoreState.DENIED, "live_lease_required")
        return None

    def _release_lease_locked(self, lease_id: str, released_at: str) -> None:
        existing = self._connection.execute(
            "SELECT 1 FROM ingestion_lease_releases WHERE lease_id = ?",
            (lease_id,),
        ).fetchone()
        if existing is None:
            self._connection.execute(
                "INSERT INTO ingestion_lease_releases(lease_id, released_at) "
                "VALUES (?, ?)",
                (lease_id, released_at),
            )

    def _insert_payload(
        self,
        table: str,
        columns: tuple[str, ...],
        values: tuple[Any, ...],
    ) -> None:
        placeholders = ",".join("?" for _ in columns)
        self._connection.execute(
            f"INSERT INTO {table}({','.join(columns)}) VALUES ({placeholders})",  # noqa: S608
            values,
        )

    @staticmethod
    def _parse_record(
        row: Mapping[str, Any],
        parser: Callable[[str], T],
    ) -> T:
        return parser(str(row["payload_json"]))


class SQLiteIngestionStore(_IngestionStoreMixin, SQLiteJourneyStore):
    """SQLite Journey store extended with durable ingestion coordination."""

    @contextmanager
    def _ledger_transaction(self) -> Iterator[None]:
        self._require_open()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield
                self._connection.execute("COMMIT")
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise


class PostgresIngestionStore(_IngestionStoreMixin, PostgresJourneyStore):
    """PostgreSQL Journey store with cross-worker ingestion coordination."""

    @contextmanager
    def _ledger_transaction(self) -> Iterator[None]:
        self._require_open()
        with self._lock:
            try:
                self._connection.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
                self._connection.execute(
                    "SELECT pg_advisory_xact_lock(?)",
                    (_POSTGRES_LEDGER_LOCK_ID,),
                ).fetchone()
                yield
                try:
                    self._raw_connection.commit()
                except Exception:
                    raise CommitStatusUnknown(
                        "postgres ingestion commit status is unknown"
                    ) from None
            except BaseException:
                try:
                    self._raw_connection.rollback()
                except Exception:
                    pass
                raise


def _derived_id(prefix: str, *values: str) -> str:
    digest = canonical_digest({"prefix": prefix, "values": list(values)})
    return f"{prefix}_{digest.removeprefix('sha256:')[:32]}"


def _parse_time(value: str, field_name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be timezone-aware")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        raise ValueError(f"{field_name} must be timezone-aware") from None
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace(
            "+00:00",
            "Z",
        )
    )


__all__ = [
    "IngestionLedger",
    "ManifestRegistration",
    "PostgresIngestionStore",
    "SQLiteIngestionStore",
]
