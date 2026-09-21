"""PostgreSQL conformance tests using a caller-provided ephemeral database."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from dataclasses import replace
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
from openmed.structured.store import (
    CanonicalRecord,
    DenyStorageOperations,
    MigrationHealth,
    PointInTimeReader,
    PostgresJourneyStore,
    PostgresMigration,
    PostgresStoreError,
    SQLiteJourneyStore,
    StorePoint,
    TransactionalJourneyStore,
)
from openmed.structured.store import postgres as postgres_module

pytestmark = pytest.mark.integration

CONTENT = b"synthetic postgres journey artifact"
COMMITTED_1 = "2026-01-02T03:04:05Z"
COMMITTED_2 = "2026-01-02T04:04:05Z"
COMMITTED_3 = "2026-01-02T05:04:05Z"
COMMITTED_4 = "2026-01-02T06:04:05Z"


def _artifact() -> ClinicalArtifact:
    return ClinicalArtifact(
        artifact_id="artifact_dddddddddddddddd",
        artifact_type="clinical_note",
        media_type="text/plain",
        content_hash=sha256_digest(CONTENT),
        byte_size=len(CONTENT),
        source_id="source_dddddddddddddddd",
        recorded_at=COMMITTED_1,
        subject_id="subject_dddddddddddddddd",
        encounter_id="encounter_dddddddddddddddd",
    )


def _evidence() -> EvidenceLocator:
    return EvidenceLocator(
        locator_id="evidence_dddddddddddddddd",
        artifact_id=_artifact().artifact_id,
        location_type="text_span",
        location={"start": 0, "end": 9},
    )


def _fact(
    fact_id: str = "fact_dddddddddddddddd",
    *,
    state: str = "active",
    parent_fact_ids: tuple[str, ...] = (),
) -> ClinicalFact:
    return ClinicalFact(
        fact_id=fact_id,
        subject_id="subject_dddddddddddddddd",
        fact_type="condition",
        value={"code": "synthetic-condition", "state": state},
        status=state,
        evidence_ids=(_evidence().locator_id,),
        derivation_hash=canonical_digest(
            {"fact_id": fact_id, "state": state, "synthetic": True}
        ),
        encounter_id="encounter_dddddddddddddddd",
        parent_fact_ids=parent_fact_ids,
    )


def _canonical(fact_id: str, state: str) -> CanonicalRecord:
    return CanonicalRecord(
        canonical_id="canonical_dddddddddddddddd",
        subject_id="subject_dddddddddddddddd",
        fact_id=fact_id,
        record_type="condition",
        state=state,
        effective_at=COMMITTED_1,
        reason_code="source_priority",
        metadata={"review": {"required": False}},
    )


@pytest.fixture
def postgres_runtime() -> Iterator[tuple[Any, str, str]]:
    dsn = os.environ.get("OPENMED_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("OPENMED_TEST_POSTGRES_DSN is not configured")
    psycopg = pytest.importorskip("psycopg")
    schema = f"openmed_test_{uuid.uuid4().hex}"
    try:
        yield psycopg, dsn, schema
    finally:
        connection = psycopg.connect(dsn, autocommit=True)
        try:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        finally:
            connection.close()


def _exercise_contract(store: Any) -> dict[str, Any]:
    assert isinstance(store, TransactionalJourneyStore)
    assert isinstance(store, PointInTimeReader)

    with store.transaction(committed_at=COMMITTED_1) as transaction:
        assert transaction.put_artifact(_artifact()).ok
        assert transaction.put_evidence(_evidence()).ok
        assert transaction.put_fact(_fact()).ok
    first_revision = transaction.revision

    first_canonical = store.put_canonical(
        _canonical(_fact().fact_id, "active"),
        committed_at=COMMITTED_2,
    )
    assert first_canonical.ok and first_canonical.created
    historical_point = StorePoint(first_canonical.revision or 0)

    second = _fact(
        "fact_eeeeeeeeeeeeeeee",
        state="inactive",
        parent_fact_ids=(_fact().fact_id,),
    )
    assert store.put_fact(second, committed_at=COMMITTED_3).ok
    corrected = store.put_canonical(
        _canonical(second.fact_id, "inactive"),
        committed_at=COMMITTED_4,
    )
    assert corrected.ok and corrected.created

    replay = store.put_fact(second, committed_at=COMMITTED_4)
    assert replay.ok and not replay.created

    before_conflict = store.latest_revision
    conflict = store.put_artifact(
        replace(_artifact(), media_type="application/pdf"),
        committed_at=COMMITTED_4,
    )
    assert not conflict.ok
    assert store.latest_revision == before_conflict

    missing = _fact("fact_ffffffffffffffff")
    missing = ClinicalFact.from_dict(
        {
            **missing.to_dict(),
            "evidence_ids": ["evidence_ffffffffffffffff"],
        }
    )
    before_failed = store.latest_revision
    failed = store.put_fact(missing, committed_at=COMMITTED_4)
    assert not failed.ok
    assert store.latest_revision == before_failed
    assert not store.get_fact(missing.fact_id).ok

    historical = store.get_canonical(
        "canonical_dddddddddddddddd",
        as_of=historical_point,
    )
    latest = store.get_canonical("canonical_dddddddddddddddd")
    integrity = store.integrity_check()
    assert historical.ok and latest.ok and integrity.ok
    assert historical.value is not None and latest.value is not None

    return {
        "first_revision": first_revision,
        "facts": tuple(
            fact.to_json()
            for fact in store.list_facts("subject_dddddddddddddddd").value or ()
        ),
        "historical": historical.value.record.to_json(),
        "latest": latest.value.record.to_json(),
        "canonical_versions": tuple(
            version.record.to_json()
            for version in (
                store.list_canonical_versions("canonical_dddddddddddddddd").value or ()
            )
        ),
        "integrity": dict(integrity.value or {}),
    }


def test_postgres_matches_sqlite_logical_contract(
    postgres_runtime: tuple[Any, str, str],
    tmp_path: Path,
) -> None:
    psycopg, dsn, schema = postgres_runtime
    with SQLiteJourneyStore(tmp_path / "journey.sqlite3") as local:
        local_result = _exercise_contract(local)

    postgres = PostgresJourneyStore(psycopg.connect(dsn), schema=schema)
    try:
        postgres_result = _exercise_contract(postgres)
        assert postgres.migration_report.state is MigrationHealth.HEALTHY
    finally:
        postgres.close()

    assert postgres_result == local_result

    reopened = PostgresJourneyStore(psycopg.connect(dsn), schema=schema)
    try:
        assert reopened.migration_report.state is MigrationHealth.HEALTHY
        assert (
            reopened.get_canonical("canonical_dddddddddddddddd").value.record.state
            == "inactive"
        )
        assert reopened.integrity_check().ok
    finally:
        reopened.close()

    denied = PostgresJourneyStore(
        psycopg.connect(dsn),
        schema=schema,
        policy=DenyStorageOperations(frozenset({"read"})),
    )
    try:
        result = denied.get_fact("fact_dddddddddddddddd")
        assert not result.ok and result.code == "policy_denied"
    finally:
        denied.close()


def test_interrupted_migration_rolls_back_and_recovers(
    postgres_runtime: tuple[Any, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psycopg, dsn, schema = postgres_runtime
    original_migrations = postgres_module.POSTGRES_MIGRATIONS
    original_latest = postgres_module.LATEST_POSTGRES_MIGRATION_VERSION
    broken = PostgresMigration(
        version=2,
        name="synthetic_interruption",
        statements=(
            "CREATE TABLE migration_interruption_marker(id BIGINT PRIMARY KEY)",
            "THIS IS NOT VALID POSTGRESQL",
        ),
    )
    monkeypatch.setattr(
        postgres_module,
        "POSTGRES_MIGRATIONS",
        (*original_migrations, broken),
    )
    monkeypatch.setattr(postgres_module, "LATEST_POSTGRES_MIGRATION_VERSION", 2)

    with pytest.raises(PostgresStoreError):
        PostgresJourneyStore(psycopg.connect(dsn), schema=schema)

    inspector = psycopg.connect(dsn)
    try:
        with inspector.cursor() as cursor:
            cursor.execute("SELECT to_regnamespace(%s)", (schema,))
            assert cursor.fetchone()[0] is None
    finally:
        inspector.close()

    monkeypatch.setattr(postgres_module, "POSTGRES_MIGRATIONS", original_migrations)
    monkeypatch.setattr(
        postgres_module,
        "LATEST_POSTGRES_MIGRATION_VERSION",
        original_latest,
    )
    recovered = PostgresJourneyStore(psycopg.connect(dsn), schema=schema)
    try:
        assert recovered.migration_report.state is MigrationHealth.HEALTHY
        assert recovered.schema_version == original_latest
    finally:
        recovered.close()
