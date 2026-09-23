"""Synthetic end-to-end recovery tests for duplicate-safe ingestion."""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    EvidenceLocator,
    canonical_digest,
    sha256_digest,
)
from openmed.interop.ingest import (
    IngestionCoordinator,
    PostgresIngestionStore,
    QuarantinePromotion,
    QuarantineRequest,
    SourceManifest,
    SQLiteIngestionStore,
    StepOutput,
)
from openmed.structured.store import StoreState

pytestmark = pytest.mark.integration

T0 = "2026-01-02T03:04:05Z"
T1 = "2026-01-02T04:04:05Z"
T2 = "2026-01-03T03:04:05Z"
CONTENT = b"synthetic resumable ingestion source"


class WorkerKilled(BaseException):
    """Synthetic abrupt worker termination that bypasses normal handling."""


def _artifact() -> ClinicalArtifact:
    return ClinicalArtifact(
        artifact_id="artifact_ffffffffffffffff",
        artifact_type="clinical_note",
        media_type="text/plain",
        content_hash=sha256_digest(CONTENT),
        byte_size=len(CONTENT),
        source_id="source_ffffffffffffffff",
        recorded_at=T0,
        subject_id="subject_ffffffffffffffff",
        encounter_id="encounter_ffffffffffffffff",
    )


def _evidence() -> EvidenceLocator:
    return EvidenceLocator(
        locator_id="evidence_ffffffffffffffff",
        artifact_id=_artifact().artifact_id,
        location_type="text_span",
        location={"start": 0, "end": 9},
    )


def _fact(index: int) -> ClinicalFact:
    fact_id = f"fact_{index:016d}"
    return ClinicalFact(
        fact_id=fact_id,
        subject_id="subject_ffffffffffffffff",
        fact_type="condition",
        value={"code": f"synthetic-{index}", "state": "active"},
        status="active",
        evidence_ids=(_evidence().locator_id,),
        derivation_hash=canonical_digest({"fact": index, "synthetic": True}),
        encounter_id="encounter_ffffffffffffffff",
    )


def _manifest() -> SourceManifest:
    return SourceManifest(
        manifest_id="manifest_ffffffffffffffff",
        source_id="source_ffffffffffffffff",
        artifact_digests=(_artifact().content_hash,),
        policy_digest=canonical_digest({"policy": "synthetic"}),
        pipeline_digest=canonical_digest({"pipeline": "synthetic"}),
        created_at=T0,
    )


def _prepare_store(path: Path) -> tuple[SQLiteIngestionStore, str]:
    store = SQLiteIngestionStore(path)
    with store.transaction(committed_at=T0) as transaction:
        assert transaction.put_artifact(_artifact()).ok
        assert transaction.put_evidence(_evidence()).ok
    registration = store.register_manifest(_manifest(), recorded_at=T0)
    assert registration.ok and registration.value is not None
    return store, registration.value.job.job_id


@pytest.mark.parametrize("crash_step", (0, 1, 2))
def test_worker_killed_at_each_checkpoint_resumes_without_duplicate_facts(
    tmp_path: Path,
    crash_step: int,
) -> None:
    path = tmp_path / f"crash-{crash_step}.sqlite3"
    store, job_id = _prepare_store(path)
    coordinator = IngestionCoordinator(store)
    crashed = False

    def operation(index: int, *, crash: bool, timestamp: str):
        def run() -> StepOutput:
            nonlocal crashed
            fact = _fact(index)
            result = store.put_fact(fact, committed_at=timestamp)
            assert result.ok
            if crash and not crashed:
                crashed = True
                raise WorkerKilled
            return StepOutput(
                output_digest=canonical_digest(fact.to_dict()),
                committed_revision=result.revision,
            )

        return run

    for index in range(crash_step):
        result = coordinator.execute_step(
            job_id=job_id,
            worker_id="worker_aaaaaaaaaaaaaaaa",
            step=f"step_{index}",
            input_digest=canonical_digest({"input": index}),
            acquired_at=T1,
            lease_seconds=3600,
            attempt=1,
            operation=operation(index, crash=False, timestamp=T1),
        )
        assert result.ok and result.created

    with pytest.raises(WorkerKilled):
        coordinator.execute_step(
            job_id=job_id,
            worker_id="worker_aaaaaaaaaaaaaaaa",
            step=f"step_{crash_step}",
            input_digest=canonical_digest({"input": crash_step}),
            acquired_at=T1,
            lease_seconds=3600,
            attempt=1,
            operation=operation(crash_step, crash=True, timestamp=T1),
        )
    store.close()

    store = SQLiteIngestionStore(path)
    coordinator = IngestionCoordinator(store)
    for index in range(3):
        result = coordinator.execute_step(
            job_id=job_id,
            worker_id="worker_bbbbbbbbbbbbbbbb",
            step=f"step_{index}",
            input_digest=canonical_digest({"input": index}),
            acquired_at=T2,
            lease_seconds=3600,
            attempt=2,
            operation=operation(index, crash=False, timestamp=T2),
        )
        assert result.ok
        assert result.value is not None
        assert result.value.replayed is (index < crash_step)

    facts = store.list_facts("subject_ffffffffffffffff")
    assert facts.ok and facts.value is not None
    assert tuple(fact.fact_id for fact in facts.value) == tuple(
        f"fact_{index:016d}" for index in range(3)
    )
    checkpoint_count = store._connection.execute(
        "SELECT COUNT(*) AS count FROM ingestion_checkpoints WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    assert int(checkpoint_count["count"]) == 3
    assert store.ingestion_integrity_check().ok

    lease = store.get_active_lease(job_id, at=T2)
    assert lease.ok and lease.value is not None
    completed = coordinator.complete(
        job_id,
        lease_id=lease.value.lease_id,
        completed_at=T2,
    )
    assert completed.ok and completed.value.state == "completed"
    replay = coordinator.register(_manifest(), recorded_at=T2)
    assert replay.ok and not replay.created
    assert replay.value.job.state == "completed"
    store.close()


@pytest.mark.parametrize(
    ("classification", "reason_code"),
    (("partial", "partial_parse"), ("malformed", "malformed_source")),
)
def test_untrusted_parse_stays_quarantined_until_explicit_promotion(
    tmp_path: Path,
    classification: str,
    reason_code: str,
) -> None:
    store, job_id = _prepare_store(tmp_path / f"quarantine-{classification}.sqlite3")
    coordinator = IngestionCoordinator(store)

    result = coordinator.execute_step(
        job_id=job_id,
        worker_id="worker_aaaaaaaaaaaaaaaa",
        step="parse",
        input_digest=canonical_digest({"input": "partial"}),
        acquired_at=T1,
        lease_seconds=3600,
        attempt=1,
        operation=lambda: QuarantineRequest(
            classification=classification,
            reason_code=reason_code,
            candidate_count=2,
            failure_count=1,
            output_digest=canonical_digest({"candidates": 2}),
        ),
    )

    assert result.state is StoreState.PARTIAL
    assert result.value is not None and result.value.quarantine is not None
    assert store.get_job(job_id).value.state == "quarantined"
    assert store.list_facts("subject_ffffffffffffffff").value == ()
    blocked = coordinator.execute_step(
        job_id=job_id,
        worker_id="worker_bbbbbbbbbbbbbbbb",
        step="normalize",
        input_digest=canonical_digest({"input": "blocked"}),
        acquired_at=T2,
        lease_seconds=3600,
        attempt=1,
        operation=lambda: StepOutput(canonical_digest({"should": "not-run"})),
    )
    assert blocked.state is StoreState.DENIED
    assert blocked.code == "quarantine_promotion_required"

    quarantine = result.value.quarantine
    promotion = QuarantinePromotion(
        promotion_id="promotion_ffffffffffffffff",
        quarantine_id=quarantine.quarantine_id,
        reviewer_digest=canonical_digest({"reviewer": "synthetic"}),
        evidence_digest=canonical_digest({"evidence": "synthetic"}),
        promoted_at=T2,
    )
    assert coordinator.promote(promotion).ok
    assert store.get_job(job_id).value.state == "queued"
    store.close()


def test_failure_diagnostic_never_echoes_exception_text(tmp_path: Path) -> None:
    store, job_id = _prepare_store(tmp_path / "failure.sqlite3")
    coordinator = IngestionCoordinator(store)
    canary = "raw-clinical-canary-must-not-persist"

    def fail() -> StepOutput:
        raise RuntimeError(canary)

    result = coordinator.execute_step(
        job_id=job_id,
        worker_id="worker_aaaaaaaaaaaaaaaa",
        step="parse",
        input_digest=canonical_digest({"input": "failure"}),
        acquired_at=T1,
        lease_seconds=3600,
        attempt=1,
        operation=fail,
    )

    assert result.state is StoreState.FAILURE
    assert result.code == "unclassified_failure"
    assert result.value is not None and result.value.retry is not None
    assert canary not in result.value.retry.to_json()
    assert canary not in repr(result)
    stored = store._connection.execute(
        "SELECT payload_json FROM ingestion_retries"
    ).fetchone()
    assert canary not in stored["payload_json"]
    store.close()


@pytest.fixture
def postgres_ingestion_runtime() -> Iterator[tuple[Any, str, str]]:
    dsn = os.environ.get("OPENMED_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("OPENMED_TEST_POSTGRES_DSN is not configured")
    psycopg = pytest.importorskip("psycopg")
    schema = f"openmed_ingest_test_{uuid.uuid4().hex}"
    try:
        yield psycopg, dsn, schema
    finally:
        connection = psycopg.connect(dsn, autocommit=True)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        finally:
            connection.close()


def test_postgres_two_worker_lease_race_has_one_winner_and_recovers(
    postgres_ingestion_runtime: tuple[Any, str, str],
) -> None:
    psycopg, dsn, schema = postgres_ingestion_runtime
    first = PostgresIngestionStore(psycopg.connect(dsn), schema=schema)
    second = PostgresIngestionStore(psycopg.connect(dsn), schema=schema)
    registration = first.register_manifest(_manifest(), recorded_at=T0)
    assert registration.ok and registration.value is not None
    job_id = registration.value.job.job_id
    barrier = threading.Barrier(2)

    def acquire(store: PostgresIngestionStore, worker_id: str):
        barrier.wait(timeout=5)
        return store.acquire_lease(
            job_id,
            worker_id,
            acquired_at=T1,
            duration_seconds=3600,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            left = executor.submit(
                acquire,
                first,
                "worker_aaaaaaaaaaaaaaaa",
            )
            right = executor.submit(
                acquire,
                second,
                "worker_bbbbbbbbbbbbbbbb",
            )
            results = (left.result(timeout=10), right.result(timeout=10))
        assert sum(result.ok for result in results) == 1
        assert sum(result.code == "lease_held" for result in results) == 1
        winner = next(result.value for result in results if result.ok)
        assert winner is not None
        active = first.get_active_lease(job_id, at=T1)
        assert active.ok and active.value == winner
        assert first.ingestion_integrity_check().ok
    finally:
        first.close()
        second.close()

    reopened = PostgresIngestionStore(psycopg.connect(dsn), schema=schema)
    try:
        assert reopened.get_job(job_id).value.state == "running"
        assert reopened.get_active_lease(job_id, at=T1).ok
    finally:
        reopened.close()
