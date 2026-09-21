"""Offline tests for PostgreSQL Journey-store boundaries and migrations."""

from __future__ import annotations

import re

import pytest

from openmed.structured.store import (
    LATEST_POSTGRES_MIGRATION_VERSION,
    POSTGRES_MIGRATIONS,
    MigrationHealth,
    PostgresJourneyStore,
    PostgresMigration,
    PostgresMigrationReport,
    StoreState,
)
from openmed.structured.store.postgres import _postgres_sql


def test_postgres_migrations_are_ordered_deterministic_and_native() -> None:
    versions = tuple(migration.version for migration in POSTGRES_MIGRATIONS)

    assert versions == tuple(range(1, LATEST_POSTGRES_MIGRATION_VERSION + 1))
    assert len({migration.checksum for migration in POSTGRES_MIGRATIONS}) == len(
        POSTGRES_MIGRATIONS
    )
    for migration in POSTGRES_MIGRATIONS:
        assert migration.checksum == migration.checksum
        sql = "\n".join(migration.statements)
        assert "revision BIGINT PRIMARY KEY" in sql
        assert "BIGSERIAL" not in sql
        assert "AUTOINCREMENT" not in sql
        assert "PRAGMA" not in sql


def test_postgres_migration_checksum_changes_with_statement() -> None:
    first = PostgresMigration(2, "synthetic", ("SELECT 1",))
    second = PostgresMigration(2, "synthetic", ("SELECT 2",))

    assert re.fullmatch(r"[0-9a-f]{64}", first.checksum)
    assert first.checksum != second.checksum


def test_migration_report_exposes_only_value_safe_state() -> None:
    report = PostgresMigrationReport(
        state=MigrationHealth.PENDING,
        current_version=0,
        target_version=1,
        pending_versions=(1,),
    )

    assert report.state is MigrationHealth.PENDING
    assert report.pending_versions == (1,)


def test_qmark_translation_preserves_parameterized_values() -> None:
    sql = _postgres_sql(
        "SELECT payload_json FROM clinical_facts "
        "WHERE fact_id = ? AND created_revision <= ?"
    )

    assert sql.endswith("WHERE fact_id = %s AND created_revision <= %s")


@pytest.mark.parametrize(
    "schema",
    (
        "OpenMed",
        "journey-store",
        "journey.store",
        "journey schema",
        "a" * 64,
    ),
)
def test_schema_identifiers_are_strictly_bounded(schema: str) -> None:
    with pytest.raises(ValueError, match="controlled identifier"):
        PostgresJourneyStore(object(), schema=schema)  # type: ignore[arg-type]


def test_invalid_connection_and_dsn_return_typed_failure() -> None:
    opened = PostgresJourneyStore.open(object())  # type: ignore[arg-type]
    connected = PostgresJourneyStore.connect("")

    assert opened.state is StoreState.FAILURE
    assert opened.code == "store_open_failed"
    assert connected.state is StoreState.FAILURE
    assert connected.code == "invalid_dsn"
    assert "object" not in repr(opened)
