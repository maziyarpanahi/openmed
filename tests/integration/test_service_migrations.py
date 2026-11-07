"""Offline rehearsal of forward and rollback-safe migrations for durable schemas.

The full rehearsal runs once per module against synthetic fixtures: every
present component and supported source version goes through clean upgrade,
repeat, a killed or interrupted migration, drift, a newer-binary schema, the
previous release's binary, and restore-based rollback through a verified
recovery point.  Further tests prove the rehearsal catches regressions (an
engine that accepts newer versions, an activation check that migrates
restored state, an unclassified or edited migration), that the PostgreSQL
deployment readiness gate stays closed until migrations complete, and that a
preflight plan is a pure dry run.  A real PostgreSQL server is rehearsed only
when ``OPENMED_TEST_POSTGRES_DSN`` is set.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from openmed.deploy.journey import ComponentState, migration_health
from openmed.service import recovery, schema_registry
from openmed.service.migration_rehearsal import (
    RehearsalReport,
    Scenario,
    ScenarioOutcome,
    SQLitePostgresDouble,
    rehearse_service_migrations,
)
from openmed.service.recovery import (
    ServiceStateSources,
    create_recovery_point,
    journey_store_check,
)
from openmed.service.schema_registry import (
    JOURNEY_SQLITE,
    Presence,
    durable_schemas,
    plan_service_migrations,
    registry_digest,
    registry_findings,
)
from openmed.structured.store import (
    PostgresJourneyStore,
    SQLiteJourneyStore,
)
from openmed.structured.store.migrations import MIGRATIONS
from openmed.structured.store.postgres import POSTGRES_MIGRATIONS
from tests.fixtures.service_recovery import (
    SENTINEL,
    backup_provider,
    build_service_state,
    tree_bytes,
)

pytest.importorskip("cryptography")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        sys.platform == "win32", reason="atomic symlink activation is POSIX-only"
    ),
]

_CLOCK = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
_SIGNING_KEY = bytes(range(32))


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the migration rehearsal must not use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(scope="module")
def full_report(tmp_path_factory: pytest.TempPathFactory) -> RehearsalReport:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the migration rehearsal must not use the network")

    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("OPENMED_PROFILE", raising=False)
        patch.setattr(socket.socket, "connect", refuse)
        patch.setattr(socket, "create_connection", refuse)
        return rehearse_service_migrations(
            tmp_path_factory.mktemp("rehearsal") / "workspace",
            clock=lambda: _CLOCK,
        )


def _rows(report: RehearsalReport, component: str) -> list[Any]:
    item = next(row for row in report.components if row.component == component)
    return list(item.scenarios)


def _row(report: RehearsalReport, component: str, scenario: Scenario, source: int):
    matches = [
        row
        for row in _rows(report, component)
        if row.scenario is scenario and row.source_version == source
    ]
    assert len(matches) == 1, (component, scenario, source)
    return matches[0]


def test_full_rehearsal_passes_the_release_gate(full_report: RehearsalReport) -> None:
    assert full_report.gate_state == "passed", full_report.gate_codes
    assert full_report.gate_codes == ()
    assert full_report.registry_digest == registry_digest()
    assert full_report.previous_release == schema_registry.PREVIOUS_RELEASE
    assert full_report.generated_at == "2026-11-07T12:00:00Z"
    assert registry_findings() == ()


def test_every_component_and_source_version_has_a_complete_matrix(
    full_report: RehearsalReport,
) -> None:
    reported = {item.component for item in full_report.components}
    assert reported == {schema.component for schema in durable_schemas()}
    for schema in durable_schemas():
        if schema.presence is not Presence.PRESENT:
            continue
        for source in schema.source_versions:
            for scenario in Scenario:
                if scenario is Scenario.PREVIOUS_BINARY:
                    continue
                row = _row(full_report, schema.component, scenario, source)
                assert row.outcome is not ScenarioOutcome.FAILED, row.to_dict()
                if scenario in {
                    Scenario.FORWARD_UPGRADE,
                    Scenario.REPEAT,
                    Scenario.UNKNOWN_VERSION,
                }:
                    assert row.outcome is ScenarioOutcome.PASSED, row.to_dict()
        previous = _row(
            full_report,
            schema.component,
            Scenario.PREVIOUS_BINARY,
            schema.current_version or 0,
        )
        assert previous.outcome is not ScenarioOutcome.FAILED, previous.to_dict()


@pytest.mark.parametrize(
    ("component", "sources", "evidence"),
    [
        (JOURNEY_SQLITE, (0, 1, 2, 3), "killed_process"),
        ("journey_postgres", (0, 1, 2), "connection_lost"),
        ("journey_artifact_marker", (0,), "killed_process"),
        ("model_registry", (1,), "killed_process"),
        ("recovery_manifest", (0,), "fault_injected"),
    ],
)
def test_interrupted_migrations_resume_without_a_half_upgraded_schema(
    full_report: RehearsalReport,
    component: str,
    sources: tuple[int, ...],
    evidence: str,
) -> None:
    for source in sources:
        row = _row(full_report, component, Scenario.INTERRUPTED, source)
        assert row.outcome is ScenarioOutcome.PASSED, row.to_dict()
        assert row.evidence == evidence
        assert row.checks["interrupted"] is True
        assert row.checks["not_ready_after_interruption"] is True
        assert row.checks["no_half_upgraded_schema"] is True
        assert row.checks["retry_ready"] is True
        if "plan_not_ready" in row.checks:
            assert row.checks["plan_not_ready"] is True
        if source:
            assert row.checks["state_unchanged"] is True


def test_drift_and_newer_versions_are_refused_with_registry_codes(
    full_report: RehearsalReport,
) -> None:
    executed = 0
    for schema in durable_schemas():
        if schema.presence is not Presence.PRESENT:
            continue
        for row in _rows(full_report, schema.component):
            if row.scenario not in {Scenario.DRIFT, Scenario.UNKNOWN_VERSION}:
                continue
            if row.outcome is ScenarioOutcome.NOT_APPLICABLE:
                assert row.scenario is Scenario.DRIFT
                continue
            key = "drift" if row.scenario is Scenario.DRIFT else "unknown_version"
            assert row.outcome is ScenarioOutcome.PASSED, row.to_dict()
            assert row.code == schema.refusal_codes[key]
            assert row.checks["refused_before_mutation"] is True
            executed += 1
    assert executed >= 30


def test_restore_rolls_back_to_the_state_the_previous_binary_reads(
    full_report: RehearsalReport,
) -> None:
    for source in (1, 2, 3, 4):
        restored = _row(full_report, JOURNEY_SQLITE, Scenario.RESTORE_ROLLBACK, source)
        assert restored.outcome is ScenarioOutcome.PASSED, restored.to_dict()
        assert restored.observed_version == source
        assert restored.checks["not_ready_before_activation"] is True
        assert restored.checks["restored_state_identical"] is True
        retry = _row(full_report, JOURNEY_SQLITE, Scenario.RETRY_AFTER_RECOVERY, source)
        assert retry.outcome is ScenarioOutcome.PASSED
        assert retry.observed_version == MIGRATIONS[-1].version
    previous = _row(full_report, JOURNEY_SQLITE, Scenario.RESTORE_ROLLBACK, 3)
    assert previous.checks["previous_binary_reads_restored"] is True
    postgres = _row(full_report, "journey_postgres", Scenario.RESTORE_ROLLBACK, 3)
    assert postgres.outcome is ScenarioOutcome.EXTERNAL
    assert postgres.code == "operator_database_backup"


def test_previous_release_binary_behaves_as_declared(
    full_report: RehearsalReport,
) -> None:
    sqlite = _row(full_report, JOURNEY_SQLITE, Scenario.PREVIOUS_BINARY, 4)
    assert sqlite.outcome is ScenarioOutcome.PASSED
    assert sqlite.evidence == "previous_release_chain"
    assert sqlite.code == "schema_unsupported"
    postgres = _row(full_report, "journey_postgres", Scenario.PREVIOUS_BINARY, 3)
    assert postgres.outcome is ScenarioOutcome.PASSED
    assert postgres.code == "previous_release_reads"
    marker = _row(full_report, "journey_artifact_marker", Scenario.PREVIOUS_BINARY, 1)
    assert marker.outcome is ScenarioOutcome.DECLARED
    assert marker.code == "artifact_integrity_failed"


def test_report_is_value_free_signed_and_tamper_evident(
    full_report: RehearsalReport,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    text = full_report.to_json()
    assert str(tmp_path_factory.getbasetemp()) not in text
    assert "rehearsal_" not in text
    assert SENTINEL not in text
    for item in json.loads(text)["components"]:
        if item["presence"] == "not_present":
            assert item["tracking_issue"] in {4918, 5066, 3599}
            assert item["scenarios"] == []

    signed = full_report.sign(_SIGNING_KEY, key_id="release-gate-1")
    assert signed.verify(_SIGNING_KEY)
    assert not signed.verify(bytes(32))
    assert not full_report.verify(_SIGNING_KEY)
    restored = json.loads(signed.to_json())
    assert restored["signature"]["algorithm"] == "HMAC-SHA256"
    assert restored["packet_digest"] == signed.recompute_packet_digest()
    tampered = RehearsalReport(
        components=signed.components,
        gate_state="blocked",
        gate_codes=signed.gate_codes,
        registry_digest=signed.registry_digest,
        generated_at=signed.generated_at,
        packet_digest=signed.packet_digest,
        signature=signed.signature,
    )
    assert not tampered.verify(_SIGNING_KEY)
    with pytest.raises(ValueError):
        full_report.sign(b"short", key_id="release-gate-1")
    with pytest.raises(ValueError):
        full_report.sign(_SIGNING_KEY, key_id="../path")
    assert "migration rehearsal passed" in full_report.summary()


def test_rehearsal_catches_a_job_store_that_accepts_newer_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from openmed.service import jobs

    monkeypatch.setattr(jobs, "JOB_STORE_SCHEMA_VERSION", 99)
    report = rehearse_service_migrations(tmp_path / "ws", components=["job_store"])

    assert report.gate_state == "blocked"
    assert any(
        code.startswith("job_store:unknown_version:") for code in report.gate_codes
    )


def test_rehearsal_catches_an_activation_check_that_migrates_restored_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Restore the old behaviour: every staged store is opened (and migrated).
    monkeypatch.setattr(
        recovery,
        "_staged_schema_version",
        lambda database: recovery.LATEST_MIGRATION_VERSION,
    )
    report = rehearse_service_migrations(tmp_path / "ws", components=[JOURNEY_SQLITE])

    assert report.gate_state == "blocked"
    for source in (1, 2, 3):
        row = _row(report, JOURNEY_SQLITE, Scenario.RESTORE_ROLLBACK, source)
        assert row.outcome is ScenarioOutcome.FAILED
        assert row.checks["restored_state_identical"] is False
    assert _row(report, JOURNEY_SQLITE, Scenario.RESTORE_ROLLBACK, 4).outcome is (
        ScenarioOutcome.PASSED
    )


def test_unclassified_or_edited_migrations_block_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(schema_registry, "_NEW_STEP_CLASSES", {})
    edited = {
        **schema_registry.RELEASED_CHECKSUMS,
        "journey_postgres": {
            **schema_registry.RELEASED_CHECKSUMS["journey_postgres"],
            2: "0" * 64,
        },
    }
    monkeypatch.setattr(schema_registry, "RELEASED_CHECKSUMS", edited)

    report = rehearse_service_migrations(tmp_path / "ws", components=["audit_chain"])

    assert report.gate_state == "blocked"
    assert "journey_sqlite:step_4_unclassified" in report.gate_codes
    assert "journey_sqlite:rollback_unclassified" in report.gate_codes
    assert "journey_postgres:released_step_2_changed" in report.gate_codes


def test_postgres_readiness_gate_stays_closed_until_migration_completes(
    tmp_path: Path,
) -> None:
    double = SQLitePostgresDouble(tmp_path / "schema.sqlite3")
    fresh = migration_health(PostgresJourneyStore.open(double.connect()))
    assert fresh.ready
    assert fresh.components["migration"].schema_version == (
        POSTGRES_MIGRATIONS[-1].version
    )

    connection = double.connect()
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO schema_migrations(version, name, checksum) VALUES (%s, %s, %s)",
        (POSTGRES_MIGRATIONS[-1].version + 1, "future_release", "f" * 64),
    )
    cursor.close()
    connection.close()
    newer = migration_health(PostgresJourneyStore.open(double.connect()))
    assert not newer.ready
    assert newer.components["migration"].state is ComponentState.INCOMPATIBLE
    assert newer.components["migration"].code == "schema_unsupported"
    assert newer.components["store"].state is ComponentState.INCOMPATIBLE


def _seed_journey(root: Path, version: int) -> Path:
    database = root / "journey" / "journey.sqlite3"
    database.parent.mkdir(parents=True)
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        connection.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, "
            "name TEXT NOT NULL, checksum TEXT NOT NULL)"
        )
        for migration in MIGRATIONS[:version]:
            for statement in migration.statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, name, checksum) "
                "VALUES (?, ?, ?)",
                (migration.version, migration.name, migration.checksum),
            )
        connection.execute(
            "INSERT INTO store_revisions(committed_at) VALUES (?)",
            ("2026-01-01T00:00:00Z",),
        )
    finally:
        connection.close()
    return database


def test_journey_activation_check_never_migrates_an_older_restored_store(
    tmp_path: Path,
) -> None:
    from tests.fixtures.service_recovery import journey_encryption

    database = _seed_journey(tmp_path / "staged", 3)
    before = database.read_bytes()

    result = journey_store_check(journey_encryption())(tmp_path / "staged")

    assert result.passed, result
    assert database.read_bytes() == before
    assert sorted(path.name for path in database.parent.iterdir()) == [
        "journey.sqlite3"
    ]

    connection = sqlite3.connect(database, isolation_level=None)
    connection.execute(
        "UPDATE schema_migrations SET checksum = ? WHERE version = 2", ("0" * 64,)
    )
    connection.close()
    drifted = journey_store_check(journey_encryption())(tmp_path / "staged")
    assert not drifted.passed
    assert drifted.code == "schema_incompatible"


def test_preflight_is_a_dry_run_that_requires_a_matching_recovery_point(
    tmp_path: Path,
) -> None:
    current = build_service_state(tmp_path / "current")
    current.close()
    before = tree_bytes(current.root)
    plan = plan_service_migrations(
        ServiceStateSources.from_state_root(current.root), state_root=current.root
    )
    assert plan.admission == "ready", plan.to_dict()
    assert tree_bytes(current.root) == before

    root = tmp_path / "previous"
    _seed_journey(root, 3)
    sources = ServiceStateSources.from_state_root(root)
    pending = plan_service_migrations(sources, state_root=root)
    assert pending.admission == "blocked"
    assert pending.blocking_codes == ("recovery_point_required",)
    action = next(item for item in pending.actions if item.component == JOURNEY_SQLITE)
    assert (action.action, action.source_version, action.target_version) == (
        "migrate",
        3,
        MIGRATIONS[-1].version,
    )
    assert action.rollback_class == "restore_required"
    assert action.recovery_point == "service_recovery_point"

    provider = backup_provider()
    point = tmp_path / "point"
    receipt = create_recovery_point(
        sources, point, key_provider=provider, writers_stopped=True
    )
    assert receipt.completed, receipt.reason
    snapshot = tree_bytes(root)
    ready = plan_service_migrations(
        sources,
        state_root=root,
        recovery_point=point,
        backup_key_provider=provider,
    )
    assert ready.admission == "migration_required", ready.to_dict()
    assert not ready.ready
    assert tree_bytes(root) == snapshot

    connection = sqlite3.connect(root / "journey" / "journey.sqlite3")
    connection.execute(
        "INSERT INTO store_revisions(committed_at) VALUES (?)",
        ("2026-01-02T00:00:00Z",),
    )
    connection.commit()
    connection.close()
    stale = plan_service_migrations(
        sources,
        state_root=root,
        recovery_point=point,
        backup_key_provider=provider,
    )
    assert stale.blocking_codes == ("recovery_point_stale",)

    older_root = tmp_path / "older"
    _seed_journey(older_root, 2)
    older_point = tmp_path / "older-point"
    assert create_recovery_point(
        ServiceStateSources.from_state_root(older_root),
        older_point,
        key_provider=provider,
        writers_stopped=True,
    ).completed
    mismatched = plan_service_migrations(
        sources,
        state_root=root,
        recovery_point=older_point,
        backup_key_provider=provider,
    )
    assert mismatched.blocking_codes == ("recovery_point_version_mismatch",)

    migrated = SQLiteJourneyStore.open(
        root / "journey" / "journey.sqlite3", encryption=current.encryption
    )
    assert migrated.ok, migrated.code
    migrated.value.close()
    after = plan_service_migrations(sources, state_root=root)
    assert after.admission == "ready", after.to_dict()


@pytest.fixture
def postgres_connect() -> Iterator[Any]:
    dsn = os.environ.get("OPENMED_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("OPENMED_TEST_POSTGRES_DSN is not configured")
    psycopg = pytest.importorskip("psycopg")
    yield lambda: psycopg.connect(dsn)


def test_postgres_server_rehearsal(
    tmp_path: Path, postgres_connect: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    report = rehearse_service_migrations(
        tmp_path / "ws",
        components=["journey_postgres"],
        postgres_connect=postgres_connect,
    )

    item = report.components[0]
    assert item.engine == "native_engine_postgresql_server"
    assert report.gate_state == "partial", report.gate_codes
    assert not item.failed, [row.to_dict() for row in item.failed]
