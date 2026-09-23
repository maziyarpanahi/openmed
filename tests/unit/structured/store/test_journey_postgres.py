"""Offline tests for PostgreSQL Journey-store boundaries and migrations."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.structured.store import (
    LATEST_POSTGRES_MIGRATION_VERSION,
    POSTGRES_MIGRATIONS,
    MigrationHealth,
    PostgresJourneyStore,
    PostgresMigration,
    PostgresMigrationReport,
    StoreResult,
    StoreState,
)
from openmed.structured.store.postgres import (
    _pg8000_connect_parameters,
    _postgres_sql,
    _sqlstate,
)


def test_postgres_migrations_are_ordered_deterministic_and_native() -> None:
    versions = tuple(migration.version for migration in POSTGRES_MIGRATIONS)

    assert versions == tuple(range(1, LATEST_POSTGRES_MIGRATION_VERSION + 1))
    assert len({migration.checksum for migration in POSTGRES_MIGRATIONS}) == len(
        POSTGRES_MIGRATIONS
    )
    for migration in POSTGRES_MIGRATIONS:
        assert migration.checksum == migration.checksum
        sql = "\n".join(migration.statements)
        assert "BIGSERIAL" not in sql
        assert "AUTOINCREMENT" not in sql
        assert "PRAGMA" not in sql
        if migration.version == 1:
            assert "revision BIGINT PRIMARY KEY" in sql


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


def test_pg8000_connection_url_decodes_credentials_and_bounds_options() -> None:
    parameters = _pg8000_connect_parameters(
        "postgresql://openmed:p%40ss%3Aword@postgres:5432/openmed",
        {"connect_timeout": 3},
    )

    assert parameters == {
        "database": "openmed",
        "host": "postgres",
        "password": "p@ss:word",
        "port": 5432,
        "ssl_context": None,
        "timeout": 3,
        "user": "openmed",
    }


def test_postgres_connect_uses_reviewed_driver_without_exposing_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    connection = object()

    def fake_connect(**kwargs: Any) -> object:
        observed.update(kwargs)
        return connection

    def fake_import(name: str) -> SimpleNamespace:
        assert name == "pg8000.dbapi"
        return SimpleNamespace(connect=fake_connect)

    monkeypatch.setattr(
        "openmed.structured.store.postgres.import_module",
        fake_import,
    )
    monkeypatch.setattr(
        PostgresJourneyStore,
        "open",
        classmethod(lambda cls, raw, **kwargs: StoreResult.success(raw)),
    )

    result = PostgresJourneyStore.connect(
        "postgresql://openmed:synthetic-secret@postgres/openmed"
    )

    assert result.ok
    assert result.value is connection
    assert observed["password"] == "synthetic-secret"
    assert "synthetic-secret" not in repr(result)


@pytest.mark.parametrize(
    "dsn",
    (
        "https://openmed:secret@postgres/openmed",
        "postgresql://openmed:secret@postgres:0/openmed",
        "postgresql://openmed:secret@postgres/openmed?sslmode=disable",
        "postgresql://openmed:secret@postgres/openmed?sslmode=require",
        "postgresql://openmed:secret@postgres/openmed?sslmode=verify-full&x=y",
    ),
)
def test_pg8000_connection_url_rejects_unsupported_or_unsafe_options(
    dsn: str,
) -> None:
    result = PostgresJourneyStore.connect(dsn)

    assert result.state is StoreState.FAILURE
    assert result.code == "invalid_dsn"
    assert "secret" not in repr(result)


def test_pg8000_connection_url_requires_tls_verification_when_requested() -> None:
    parameters = _pg8000_connect_parameters(
        "postgresql://openmed@db.example/openmed?sslmode=verify-full",
        None,
    )

    assert parameters["ssl_context"].check_hostname is True
    assert parameters["ssl_context"].verify_mode.name == "CERT_REQUIRED"


def test_pg8000_constraint_sqlstate_remains_typed() -> None:
    assert _sqlstate(ValueError({"C": "23505", "M": "synthetic sensitive"})) == (
        "23505"
    )
