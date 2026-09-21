"""Tests for durable ingestion registration, leases, and quarantine."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.clinical.journey_contracts import canonical_digest
from openmed.interop.ingest import (
    Cancellation,
    Checkpoint,
    IngestionLedger,
    Lease,
    QuarantinePromotion,
    QuarantineResult,
    Retry,
    SourceManifest,
    SQLiteIngestionStore,
)
from openmed.structured.store import MIGRATIONS, DenyStorageOperations, StoreState

T0 = "2026-01-02T03:04:05Z"
T0_LIVE = "2026-01-02T03:04:15Z"
T1 = "2026-01-02T03:05:05Z"
T2 = "2026-01-02T04:04:05Z"
DIGEST_A = canonical_digest({"synthetic": "a"})
DIGEST_B = canonical_digest({"synthetic": "b"})


def _manifest() -> SourceManifest:
    return SourceManifest(
        manifest_id="manifest_aaaaaaaaaaaaaaaa",
        source_id="source_aaaaaaaaaaaaaaaa",
        artifact_digests=(DIGEST_A,),
        policy_digest=canonical_digest({"policy": "synthetic"}),
        pipeline_digest=canonical_digest({"pipeline": "synthetic"}),
        created_at=T0,
    )


def _registered(store: SQLiteIngestionStore):
    registration = store.register_manifest(_manifest(), recorded_at=T0)
    assert registration.ok and registration.value is not None
    return registration.value


def _lease(
    store: SQLiteIngestionStore,
    job_id: str,
    *,
    worker_id: str = "worker_aaaaaaaaaaaaaaaa",
    acquired_at: str = T0,
    duration_seconds: int = 3600,
) -> Lease:
    result = store.acquire_lease(
        job_id,
        worker_id,
        acquired_at=acquired_at,
        duration_seconds=duration_seconds,
    )
    assert result.ok and result.value is not None
    return result.value


def _checkpoint(job_id: str, manifest_digest: str, sequence: int = 1) -> Checkpoint:
    return Checkpoint(
        checkpoint_id=f"checkpoint_{sequence:016d}",
        job_id=job_id,
        manifest_digest=manifest_digest,
        step=f"step_{sequence}",
        sequence=sequence,
        input_digest=canonical_digest({"input": sequence}),
        output_digest=canonical_digest({"output": sequence}),
        completed_at=T1,
        committed_revision=sequence,
    )


def test_manifest_replay_is_audited_noop_and_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "journey.sqlite3"
    store = SQLiteIngestionStore(path)
    assert isinstance(store, IngestionLedger)

    first = store.register_manifest(_manifest(), recorded_at=T0)
    replayed_manifest = replace(
        _manifest(),
        manifest_id="manifest_bbbbbbbbbbbbbbbb",
        created_at=T1,
    )
    second = store.register_manifest(replayed_manifest, recorded_at=T1)

    assert first.ok and first.created and first.value is not None
    assert second.ok and not second.created and second.value is not None
    assert second.value.audit.action == "noop"
    assert second.value.job.job_id == first.value.job.job_id
    rows = store._connection.execute(
        "SELECT action FROM ingestion_replay_audits ORDER BY recorded_at"
    ).fetchall()
    assert [row["action"] for row in rows] == ["created", "noop"]
    store.close()

    reopened = SQLiteIngestionStore(path)
    assert reopened.get_job(first.value.job.job_id).value == first.value.job
    assert reopened.ingestion_integrity_check().ok
    reopened.close()


def test_existing_v1_journey_database_migrates_to_ingestion_ledger(
    tmp_path: Path,
) -> None:
    path = tmp_path / "journey-v1.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE schema_migrations ("
        "version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL)"
    )
    migration = MIGRATIONS[0]
    for statement in migration.statements:
        connection.execute(statement)
    connection.execute(
        "INSERT INTO schema_migrations(version, name, checksum) VALUES (?, ?, ?)",
        (migration.version, migration.name, migration.checksum),
    )
    connection.commit()
    connection.close()

    store = SQLiteIngestionStore(path)
    versions = store._connection.execute(
        "SELECT version FROM schema_migrations ORDER BY version"
    ).fetchall()
    tables = {
        row["name"]
        for row in store._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }

    assert [row["version"] for row in versions] == [1, 2]
    assert "ingestion_jobs" in tables
    assert store.ingestion_integrity_check().ok
    store.close()


def test_manifest_id_cannot_be_reused_for_changed_contents(tmp_path: Path) -> None:
    store = SQLiteIngestionStore(tmp_path / "journey.sqlite3")
    assert store.register_manifest(_manifest(), recorded_at=T0).ok
    changed = replace(_manifest(), artifact_digests=(DIGEST_B,))

    result = store.register_manifest(changed, recorded_at=T1)

    assert result.state is StoreState.CONFLICT
    assert result.code == "manifest_id_conflict"
    store.close()


def test_only_one_worker_holds_a_live_lease_and_expiry_allows_takeover(
    tmp_path: Path,
) -> None:
    store = SQLiteIngestionStore(tmp_path / "journey.sqlite3")
    job = _registered(store).job
    first = _lease(store, job.job_id, duration_seconds=30)

    same_worker = store.acquire_lease(
        job.job_id,
        first.worker_id,
        acquired_at=T0,
        duration_seconds=30,
    )
    blocked = store.acquire_lease(
        job.job_id,
        "worker_bbbbbbbbbbbbbbbb",
        acquired_at=T0_LIVE,
        duration_seconds=30,
    )
    takeover = store.acquire_lease(
        job.job_id,
        "worker_bbbbbbbbbbbbbbbb",
        acquired_at=T2,
        duration_seconds=30,
    )

    assert same_worker.ok and not same_worker.created
    assert blocked.state is StoreState.CONFLICT
    assert blocked.code == "lease_held"
    assert takeover.ok and takeover.created and takeover.value is not None
    assert takeover.value.epoch == first.epoch + 1
    assert store.get_active_lease(job.job_id, at=T2).value == takeover.value
    store.close()


def test_checkpoint_is_gap_safe_idempotent_and_advances_job(tmp_path: Path) -> None:
    store = SQLiteIngestionStore(tmp_path / "journey.sqlite3")
    registration = _registered(store)
    lease = _lease(store, registration.job.job_id)
    checkpoint = _checkpoint(
        registration.job.job_id,
        registration.job.manifest_digest,
    )

    gap = store.commit_checkpoint(
        replace(checkpoint, checkpoint_id="checkpoint_0000000000000002", sequence=2),
        lease_id=lease.lease_id,
        recorded_at=T1,
    )
    first = store.commit_checkpoint(
        checkpoint,
        lease_id=lease.lease_id,
        recorded_at=T1,
    )
    replay = store.commit_checkpoint(
        checkpoint,
        lease_id=lease.lease_id,
        recorded_at=T1,
    )
    conflict = store.commit_checkpoint(
        replace(checkpoint, output_digest=DIGEST_B),
        lease_id=lease.lease_id,
        recorded_at=T1,
    )

    assert gap.state is StoreState.PARTIAL and gap.code == "checkpoint_gap"
    assert first.ok and first.created
    assert replay.ok and not replay.created
    assert conflict.state is StoreState.CONFLICT
    assert store.get_job(registration.job.job_id).value.checkpoint_sequence == 1
    store.close()


def test_completion_and_cancellation_are_idempotent_and_terminal(
    tmp_path: Path,
) -> None:
    store = SQLiteIngestionStore(tmp_path / "journey.sqlite3")
    registration = _registered(store)
    lease = _lease(store, registration.job.job_id)

    completed = store.complete_job(
        registration.job.job_id,
        lease_id=lease.lease_id,
        completed_at=T1,
    )
    replay = store.complete_job(
        registration.job.job_id,
        lease_id=lease.lease_id,
        completed_at=T1,
    )
    cancellation = Cancellation(
        cancellation_id="cancellation_aaaaaaaaaaaaaaaa",
        job_id=registration.job.job_id,
        actor_digest=DIGEST_A,
        reason_code="owner_requested",
        requested_at=T2,
    )

    assert completed.ok and completed.created
    assert replay.ok and not replay.created
    assert store.cancel_job(cancellation).state is StoreState.CONFLICT
    assert (
        store.acquire_lease(
            registration.job.job_id,
            "worker_bbbbbbbbbbbbbbbb",
            acquired_at=T2,
            duration_seconds=30,
        ).code
        == "job_not_leasable"
    )
    store.close()


def test_quarantine_requires_explicit_promotion_before_new_lease(
    tmp_path: Path,
) -> None:
    store = SQLiteIngestionStore(tmp_path / "journey.sqlite3")
    registration = _registered(store)
    lease = _lease(store, registration.job.job_id)
    quarantine = QuarantineResult(
        quarantine_id="quarantine_aaaaaaaaaaaaaaaa",
        job_id=registration.job.job_id,
        manifest_digest=registration.job.manifest_digest,
        classification="partial",
        reason_code="partial_parse",
        candidate_count=2,
        failure_count=1,
        created_at=T1,
        output_digest=DIGEST_B,
    )
    assert store.quarantine(quarantine, lease_id=lease.lease_id).ok

    blocked = store.acquire_lease(
        registration.job.job_id,
        "worker_bbbbbbbbbbbbbbbb",
        acquired_at=T2,
        duration_seconds=30,
    )
    promotion = QuarantinePromotion(
        promotion_id="promotion_aaaaaaaaaaaaaaaa",
        quarantine_id=quarantine.quarantine_id,
        reviewer_digest=DIGEST_A,
        evidence_digest=DIGEST_B,
        promoted_at=T2,
    )
    promoted = store.promote_quarantine(promotion)
    resumed = store.acquire_lease(
        registration.job.job_id,
        "worker_bbbbbbbbbbbbbbbb",
        acquired_at=T2,
        duration_seconds=30,
    )

    assert blocked.state is StoreState.DENIED
    assert blocked.code == "quarantine_promotion_required"
    assert promoted.ok and promoted.created
    assert resumed.ok and resumed.created
    assert store.get_quarantine(quarantine.quarantine_id).value == quarantine
    store.close()


def test_retry_records_are_value_free_and_release_lease(tmp_path: Path) -> None:
    store = SQLiteIngestionStore(tmp_path / "journey.sqlite3")
    registration = _registered(store)
    lease = _lease(store, registration.job.job_id)
    retry = Retry(
        retry_id="retry_aaaaaaaaaaaaaaaa",
        job_id=registration.job.job_id,
        classification="transient",
        reason_code="operation_timeout",
        attempt=1,
        recorded_at=T1,
        retry_after=T2,
    )

    recorded = store.record_retry(retry, lease_id=lease.lease_id)

    assert recorded.ok and recorded.created
    assert (
        store.get_active_lease(registration.job.job_id, at=T1).state
        is StoreState.UNKNOWN
    )
    row = store._connection.execute(
        "SELECT payload_json FROM ingestion_retries WHERE retry_id = ?",
        (retry.retry_id,),
    ).fetchone()
    assert json.loads(row["payload_json"])["reason_code"] == "operation_timeout"
    store.close()


def test_integrity_check_detects_tampered_ingestion_payload(tmp_path: Path) -> None:
    store = SQLiteIngestionStore(tmp_path / "journey.sqlite3")
    _registered(store)
    store._connection.execute(
        "UPDATE ingestion_manifests SET payload_json = ?",
        ('{"synthetic":"tampered"}',),
    )

    result = store.ingestion_integrity_check()

    assert result.state is StoreState.FAILURE
    assert result.code == "stored_payload_invalid"
    store.close()


def test_ingestion_policy_denials_are_typed(tmp_path: Path) -> None:
    path = tmp_path / "journey.sqlite3"
    denied_write = SQLiteIngestionStore(
        path,
        policy=DenyStorageOperations(frozenset({"write"})),
    )
    result = denied_write.register_manifest(_manifest(), recorded_at=T0)
    assert result.state is StoreState.DENIED
    assert result.code == "policy_denied"
    denied_write.close()

    allowed = SQLiteIngestionStore(path)
    registration = _registered(allowed)
    allowed.close()

    denied_read = SQLiteIngestionStore(
        path,
        policy=DenyStorageOperations(frozenset({"read", "verify"})),
    )
    assert denied_read.get_job(registration.job.job_id).state is StoreState.DENIED
    assert denied_read.ingestion_integrity_check().state is StoreState.DENIED
    denied_read.close()
