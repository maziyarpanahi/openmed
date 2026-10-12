"""Offline tests for PostgreSQL Journey-store boundaries and migrations."""

from __future__ import annotations

import json
import logging
import re
import ssl
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

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
from openmed.structured.store import postgres as postgres_module
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
        profile="dev",
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
        "postgresql://openmed:synthetic-secret@postgres/openmed",
        profile="dev",
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
        "postgresql://openmed:secret@postgres/openmed?sslmode=require",
        "postgresql://openmed:secret@postgres/openmed?sslmode=verify-full&x=y",
    ),
)
def test_pg8000_connection_url_rejects_unsupported_or_unsafe_options(
    dsn: str,
) -> None:
    result = PostgresJourneyStore.connect(dsn, profile="dev")

    assert result.state is StoreState.FAILURE
    assert result.code == "invalid_dsn"
    assert "secret" not in repr(result)


def test_pg8000_connection_url_requires_tls_verification_when_requested() -> None:
    parameters = _pg8000_connect_parameters(
        "postgresql://openmed@db.example/openmed?sslmode=verify-full",
        None,
        profile="prod",
    )

    assert parameters["ssl_context"].check_hostname is True
    assert parameters["ssl_context"].verify_mode.name == "CERT_REQUIRED"


def test_pg8000_constraint_sqlstate_remains_typed() -> None:
    assert _sqlstate(ValueError({"C": "23505", "M": "synthetic sensitive"})) == (
        "23505"
    )


# Synthetic sentinels that must never reach results, health output, or logs.
_SENTINEL_USER = "sentineluser3915"
_SENTINEL_PASSWORD = "sentinelpassword3915"
_SENTINEL_HOST = "sentinel-host-3915.invalid"
_SENTINEL_DATABASE = "sentineldb3915"
_SENTINEL_CA_NAME = "OpenMed Synthetic Journey CA 3915"
_SENTINEL_CONTENT = "SENTINEL-CERTIFICATE-CONTENT-3915"
_SENTINELS = (
    _SENTINEL_USER,
    _SENTINEL_PASSWORD,
    _SENTINEL_HOST,
    _SENTINEL_DATABASE,
    _SENTINEL_CA_NAME,
    _SENTINEL_CONTENT,
)


def _sentinel_dsn(query: str = "") -> str:
    return (
        f"postgresql://{_SENTINEL_USER}:{_SENTINEL_PASSWORD}@{_SENTINEL_HOST}:5432/"
        f"{_SENTINEL_DATABASE}{query}"
    )


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _pem_key(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


@pytest.fixture
def synthetic_tls(tmp_path: Path) -> SimpleNamespace:
    """Write a synthetic CA bundle plus a client certificate and key."""

    now = datetime.now(timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_certificate = (
        x509.CertificateBuilder()
        .subject_name(_name(_SENTINEL_CA_NAME))
        .issuer_name(_name(_SENTINEL_CA_NAME))
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    client_key = ec.generate_private_key(ec.SECP256R1())
    client_certificate = (
        x509.CertificateBuilder()
        .subject_name(_name("openmed-synthetic-client"))
        .issuer_name(ca_certificate.subject)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(hours=1))
        .not_valid_after(now + timedelta(days=7))
        .sign(ca_key, hashes.SHA256())
    )
    ca_path = tmp_path / "synthetic-ca.pem"
    cert_path = tmp_path / "synthetic-client.pem"
    key_path = tmp_path / "synthetic-client.key"
    ca_path.write_bytes(ca_certificate.public_bytes(serialization.Encoding.PEM))
    cert_path.write_bytes(client_certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(_pem_key(client_key))
    return SimpleNamespace(ca=str(ca_path), cert=str(cert_path), key=str(key_path))


@pytest.fixture
def fake_pg8000(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Inject a fake pg8000 driver and record every import and connect call."""

    calls = SimpleNamespace(imports=[], connects=[], connection=object())

    def fake_connect(**kwargs: Any) -> object:
        calls.connects.append(kwargs)
        return calls.connection

    def fake_import(name: str) -> SimpleNamespace:
        calls.imports.append(name)
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
    return calls


def _assert_value_free(*outputs: str) -> None:
    for output in outputs:
        for sentinel in _SENTINELS:
            assert sentinel not in output


def test_operator_ca_bundle_builds_verify_full_context_for_driver(
    fake_pg8000: SimpleNamespace,
    synthetic_tls: SimpleNamespace,
) -> None:
    result = PostgresJourneyStore.connect(
        _sentinel_dsn("?sslmode=verify-full"),
        connect_options={"connect_timeout": 3, "sslrootcert": synthetic_tls.ca},
        profile="prod",
    )

    assert result.ok
    assert len(fake_pg8000.connects) == 1
    context = fake_pg8000.connects[0]["ssl_context"]
    assert isinstance(context, ssl.SSLContext)
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.minimum_version >= ssl.TLSVersion.TLSv1_2
    trusted = context.get_ca_certs()
    assert len(trusted) == 1
    assert ((("commonName", _SENTINEL_CA_NAME),),) == trusted[0]["subject"]
    assert fake_pg8000.connects[0]["host"] == _SENTINEL_HOST
    _assert_value_free(repr(result))


def test_operator_ca_bundle_alone_selects_verified_tls(
    fake_pg8000: SimpleNamespace,
    synthetic_tls: SimpleNamespace,
) -> None:
    result = PostgresJourneyStore.connect(
        _sentinel_dsn(),
        connect_options={"sslrootcert": synthetic_tls.ca},
        profile="prod",
    )

    assert result.ok
    context = fake_pg8000.connects[0]["ssl_context"]
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED


def test_optional_client_certificate_and_key_are_loaded(
    monkeypatch: pytest.MonkeyPatch,
    fake_pg8000: SimpleNamespace,
    synthetic_tls: SimpleNamespace,
) -> None:
    loaded: list[tuple[Any, ...]] = []
    original = ssl.SSLContext.load_cert_chain

    def spy(self: ssl.SSLContext, certfile: Any, keyfile: Any = None, *args: Any):
        loaded.append((certfile, keyfile))
        return original(self, certfile, keyfile, *args)

    monkeypatch.setattr(ssl.SSLContext, "load_cert_chain", spy)

    result = PostgresJourneyStore.connect(
        _sentinel_dsn("?sslmode=verify-full"),
        connect_options={
            "sslrootcert": synthetic_tls.ca,
            "sslcert": synthetic_tls.cert,
            "sslkey": synthetic_tls.key,
        },
        profile="prod",
    )

    assert result.ok
    assert loaded == [(synthetic_tls.cert, synthetic_tls.key)]
    assert fake_pg8000.connects[0]["ssl_context"].check_hostname is True


@pytest.mark.parametrize(
    "options",
    (
        {"sslkey": "/synthetic/client.key"},
        {"sslrootcert": ""},
        {"sslrootcert": 3},
        {"sslpassword": "synthetic"},
    ),
)
def test_incomplete_or_unknown_tls_options_fail_closed(
    fake_pg8000: SimpleNamespace,
    options: dict[str, Any],
) -> None:
    result = PostgresJourneyStore.connect(
        _sentinel_dsn("?sslmode=verify-full"),
        connect_options=options,
        profile="prod",
    )

    assert result.state is StoreState.FAILURE
    assert result.code == "invalid_dsn"
    assert fake_pg8000.imports == []


def test_plaintext_mode_conflicting_with_ca_bundle_fails_closed(
    fake_pg8000: SimpleNamespace,
    synthetic_tls: SimpleNamespace,
) -> None:
    result = PostgresJourneyStore.connect(
        _sentinel_dsn("?sslmode=disable"),
        connect_options={"sslrootcert": synthetic_tls.ca},
        profile="dev",
    )

    assert result.code == "invalid_dsn"
    assert fake_pg8000.imports == []


@pytest.mark.parametrize(
    "query",
    (
        "",
        "?sslmode=disable",
        "?sslmode=allow",
        "?sslmode=prefer",
        "?sslmode=require",
        "?sslmode=verify-ca",
    ),
)
@pytest.mark.parametrize("profile", ("prod", "production", "fast", None))
def test_production_refuses_plaintext_before_driver_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    fake_pg8000: SimpleNamespace,
    query: str,
    profile: str | None,
) -> None:
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)
    caplog.set_level(logging.DEBUG)

    result = PostgresJourneyStore.connect(_sentinel_dsn(query), profile=profile)

    assert result.state is StoreState.DENIED
    assert result.code == "postgres_tls_required"
    assert fake_pg8000.imports == []
    assert fake_pg8000.connects == []
    _assert_value_free(repr(result), caplog.text)


def test_production_profile_is_read_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    fake_pg8000: SimpleNamespace,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "prod")

    result = PostgresJourneyStore.connect(_sentinel_dsn())

    assert result.code == "postgres_tls_required"
    assert fake_pg8000.connects == []


@pytest.mark.parametrize("query", ("", "?sslmode=disable"))
@pytest.mark.parametrize("profile", ("dev", "test"))
def test_development_plaintext_requires_an_explicit_profile(
    monkeypatch: pytest.MonkeyPatch,
    fake_pg8000: SimpleNamespace,
    query: str,
    profile: str,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", profile)

    from_environment = PostgresJourneyStore.connect(_sentinel_dsn(query))
    explicit = PostgresJourneyStore.connect(_sentinel_dsn(query), profile=profile)

    assert from_environment.ok
    assert explicit.ok
    assert [call["ssl_context"] for call in fake_pg8000.connects] == [None, None]


def test_development_profile_still_rejects_non_verifying_tls(
    fake_pg8000: SimpleNamespace,
) -> None:
    result = PostgresJourneyStore.connect(
        _sentinel_dsn("?sslmode=require"),
        profile="dev",
    )

    assert result.code == "invalid_dsn"
    assert fake_pg8000.imports == []


@pytest.mark.parametrize("contents", (None, _SENTINEL_CONTENT))
def test_unusable_ca_bundle_is_value_free_and_never_connects(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    fake_pg8000: SimpleNamespace,
    contents: str | None,
) -> None:
    caplog.set_level(logging.DEBUG)
    bundle = tmp_path / f"{_SENTINEL_CONTENT}.pem"
    if contents is not None:
        bundle.write_text(contents, encoding="utf-8")

    result = PostgresJourneyStore.connect(
        _sentinel_dsn("?sslmode=verify-full"),
        connect_options={"sslrootcert": str(bundle)},
        profile="prod",
    )

    assert result.state is StoreState.FAILURE
    assert result.code == "postgres_tls_config_invalid"
    assert fake_pg8000.imports == []
    _assert_value_free(repr(result), caplog.text)


def test_tls_parameter_errors_do_not_carry_connection_values(
    synthetic_tls: SimpleNamespace,
) -> None:
    for dsn, options in (
        (_sentinel_dsn(), None),
        (_sentinel_dsn("?sslmode=require"), None),
        (_sentinel_dsn("?sslmode=verify-full"), {"sslrootcert": _SENTINEL_CONTENT}),
    ):
        with pytest.raises(ValueError) as caught:
            _pg8000_connect_parameters(dsn, options, profile="prod")
        error = caught.value
        assert error.__cause__ is None
        _assert_value_free(str(error), repr(error))


def test_transport_posture_is_exported_without_connection_values(
    fake_pg8000: SimpleNamespace,
    synthetic_tls: SimpleNamespace,
) -> None:
    posture = postgres_module.postgres_transport_posture

    verified = posture(
        _sentinel_dsn("?sslmode=verify-full"),
        {"sslrootcert": synthetic_tls.ca, "sslcert": synthetic_tls.cert},
        profile="prod",
    )
    system = posture(_sentinel_dsn("?sslmode=verify-full"), profile="prod")
    refused = posture(_sentinel_dsn(), profile="prod")
    unverified = posture(_sentinel_dsn("?sslmode=require"), profile="prod")
    development = posture(_sentinel_dsn(), profile="dev")
    invalid = posture("https://synthetic.invalid/db", profile="prod")

    assert verified.to_dict() == {
        "production": True,
        "tls": "verify_full",
        "trust_anchor": "operator_ca_bundle",
        "client_certificate": True,
        "code": "postgres_tls_verified",
        "ok": True,
    }
    assert system.to_dict()["trust_anchor"] == "system_trust_store"
    assert refused.to_dict() == {
        "production": True,
        "tls": "disabled",
        "trust_anchor": "none",
        "client_certificate": False,
        "code": "postgres_tls_required",
        "ok": False,
    }
    assert unverified.tls == "unverified"
    assert unverified.code == "postgres_tls_required"
    assert development.to_dict()["code"] == "postgres_plaintext_development"
    assert development.ok is True
    assert invalid.code == "invalid_dsn"
    assert invalid.ok is False
    assert fake_pg8000.imports == []
    for item in (verified, system, refused, unverified, development, invalid):
        _assert_value_free(repr(item), json.dumps(item.to_dict()))
        assert synthetic_tls.ca not in repr(item)
