"""Tests for the shared production Journey deployment contract."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from openmed.deploy import journey
from openmed.structured.store import (
    MigrationHealth,
    PostgresMigrationReport,
    StoreResult,
)
from openmed.structured.store.protocols import StoreState

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = (
    ROOT
    / "openmed"
    / "core"
    / "schemas"
    / "json"
    / "journey_deployment_health.schema.json"
)


class _HealthyStore:
    migration_report = PostgresMigrationReport(
        state=MigrationHealth.HEALTHY,
        current_version=3,
        target_version=3,
    )
    schema_version = 3

    def close(self) -> None:
        return None


class _GoldenStore:
    def __init__(self) -> None:
        self.artifact = None
        self.closed = False

    def put_artifact(self, artifact, *, committed_at: str):
        assert committed_at == "2026-01-01T00:00:00Z"
        self.artifact = artifact
        return StoreResult.success(artifact, created=True)

    def get_artifact(self, artifact_id: str):
        assert self.artifact is not None
        assert artifact_id == self.artifact.artifact_id
        return StoreResult.success(self.artifact)

    def close(self) -> None:
        self.closed = True


class _GoldenObjects:
    def __init__(self) -> None:
        self.content = None

    def put_bytes(self, artifact, content: bytes):
        self.content = content
        return StoreResult.success(artifact, created=True)

    def get_bytes(self, content_hash: str):
        assert content_hash.startswith("sha256:")
        return StoreResult.success(self.content)


def test_health_contract_is_versioned_typed_and_schema_valid() -> None:
    ready = journey.ComponentHealth(journey.ComponentState.READY, "store_ready")
    disabled = journey.ComponentHealth(
        journey.ComponentState.DISABLED,
        "model_disabled",
    )
    report = journey.DeploymentHealth(
        components={
            "migration": ready,
            "store": ready,
            "artifact": ready,
            "worker": ready,
            "model": disabled,
        }
    )

    payload = report.to_dict()
    jsonschema.validate(payload, json.loads(SCHEMA.read_text(encoding="utf-8")))
    assert payload["state"] == "ready"
    assert payload["compatibility_major"] == 1


def test_probe_distinguishes_each_component(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        journey.PostgresJourneyStore,
        "connect",
        lambda *args, **kwargs: StoreResult.success(_HealthyStore()),
    )
    monkeypatch.setattr(
        journey,
        "_probe_artifact",
        lambda root: journey.ComponentHealth(
            journey.ComponentState.UNAVAILABLE,
            "artifact_store_unavailable",
        ),
    )
    monkeypatch.setattr(
        journey,
        "_probe_http",
        lambda url, *, component: journey.ComponentHealth(
            journey.ComponentState.READY
            if component == "worker"
            else journey.ComponentState.PENDING,
            f"{component}_test",
        ),
    )

    report = journey.probe_deployment(
        "postgresql://not-emitted",
        worker_url="http://worker/readyz",
        model_url="http://model/readyz",
    ).to_dict()

    assert report["state"] == "not_ready"
    assert report["components"] == {
        "migration": {
            "state": "ready",
            "code": "migration_current",
            "schema_version": 3,
        },
        "store": {"state": "ready", "code": "store_ready"},
        "artifact": {
            "state": "unavailable",
            "code": "artifact_store_unavailable",
        },
        "worker": {"state": "ready", "code": "worker_test"},
        "model": {"state": "pending", "code": "model_test"},
    }
    assert "not-emitted" not in json.dumps(report)


def test_store_failure_is_value_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        journey.PostgresJourneyStore,
        "connect",
        lambda *args, **kwargs: StoreResult.outcome(
            StoreState.FAILURE,
            "postgres_connect_failed",
        ),
    )

    report = journey.run_migrations("postgresql://secret-value").to_dict()

    assert report["components"]["migration"] == {
        "state": "unavailable",
        "code": "postgres_connect_failed",
    }
    assert "secret-value" not in json.dumps(report)


def test_artifact_probe_verifies_write_and_read(tmp_path: Path) -> None:
    first = journey._probe_artifact(str(tmp_path / "artifacts"))
    second = journey._probe_artifact(str(tmp_path / "artifacts"))

    assert first == journey.ComponentHealth(
        journey.ComponentState.READY,
        "artifact_store_ready",
    )
    assert second == first
    blobs = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert len(blobs) == 1


def test_cli_requires_dsn_without_echoing_environment_name(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENMED_JOURNEY_POSTGRES_DSN", raising=False)

    assert journey.main(["migrate"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "schema_version": "openmed.journey.deployment-health.v1",
        "state": "not_ready",
        "code": "dsn_not_configured",
    }


def test_golden_journey_verifies_metadata_and_object_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _GoldenStore()
    objects = _GoldenObjects()
    monkeypatch.setattr(
        journey.PostgresJourneyStore,
        "connect",
        lambda *args, **kwargs: StoreResult.success(store),
    )
    monkeypatch.setattr(journey, "FsspecArtifactStore", lambda root: objects)

    result = journey.run_golden_journey(
        "postgresql://must-not-appear",
        artifact_root="/synthetic/artifacts",
    )

    assert result == {
        "schema_version": "openmed.journey.golden.v1",
        "compatibility_major": 1,
        "state": "passed",
        "code": "golden_journey_verified",
        "synthetic": True,
        "created": True,
        "artifact_count": 1,
    }
    assert store.closed is True
    assert "must-not-appear" not in json.dumps(result)


_TLS_SENTINELS = (
    "sentineluser3915",
    "sentinelpassword3915",
    "sentinel-host-3915.invalid",
    "sentineldb3915",
    "sentinel-ca-path-3915",
)
_PLAINTEXT_DSN = (
    "postgresql://sentineluser3915:sentinelpassword3915@"
    "sentinel-host-3915.invalid:5432/sentineldb3915"
)


def _fail_on_driver_import(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    imports: list[str] = []

    def fake_import(name: str) -> None:
        imports.append(name)
        raise AssertionError("the PostgreSQL driver must not be loaded")

    monkeypatch.setattr(
        "openmed.structured.store.postgres.import_module",
        fake_import,
    )
    return imports


def test_transport_settings_read_ca_bundle_and_client_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_JOURNEY_POSTGRES_SSLROOTCERT", "/tls/ca.crt")
    monkeypatch.setenv("OPENMED_JOURNEY_POSTGRES_SSLCERT", "/tls/tls.crt")
    monkeypatch.setenv("OPENMED_JOURNEY_POSTGRES_SSLKEY", "/tls/tls.key")

    settings = journey.PostgresTransportSettings.from_env()

    assert settings.connect_options() == {
        "connect_timeout": 3,
        "sslrootcert": "/tls/ca.crt",
        "sslcert": "/tls/tls.crt",
        "sslkey": "/tls/tls.key",
    }
    assert "/tls/" not in repr(settings)


def test_transport_settings_omit_unset_tls_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "OPENMED_JOURNEY_POSTGRES_SSLROOTCERT",
        "OPENMED_JOURNEY_POSTGRES_SSLCERT",
        "OPENMED_JOURNEY_POSTGRES_SSLKEY",
    ):
        monkeypatch.setenv(name, " ")

    assert journey.PostgresTransportSettings.from_env().connect_options() == {
        "connect_timeout": 3
    }


def test_entrypoints_pass_operator_ca_bundle_to_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict] = []

    def fake_connect(*args, **kwargs):
        observed.append(kwargs)
        return StoreResult.success(_HealthyStore())

    monkeypatch.setattr(journey.PostgresJourneyStore, "connect", fake_connect)
    monkeypatch.setenv("OPENMED_JOURNEY_POSTGRES_SSLROOTCERT", "/tls/ca.crt")

    journey.run_migrations("postgresql://not-emitted")

    assert observed[0]["connect_options"] == {
        "connect_timeout": 3,
        "sslrootcert": "/tls/ca.crt",
    }


@pytest.mark.parametrize("command", ("migrate", "probe"))
def test_production_cli_refuses_plaintext_dsn_without_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    command: str,
) -> None:
    imports = _fail_on_driver_import(monkeypatch)
    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    monkeypatch.setenv("OPENMED_JOURNEY_POSTGRES_DSN", _PLAINTEXT_DSN)
    monkeypatch.delenv("OPENMED_JOURNEY_POSTGRES_SSLROOTCERT", raising=False)

    status = journey.main(
        ["--artifact-root", str(tmp_path / "artifacts"), command],
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    jsonschema.validate(payload, json.loads(SCHEMA.read_text(encoding="utf-8")))
    assert status == 1
    assert imports == []
    assert payload["components"]["store"] == {
        "state": "incompatible",
        "code": "postgres_tls_required",
    }
    assert payload["components"]["migration"]["code"] == "postgres_tls_required"
    for sentinel in _TLS_SENTINELS:
        assert sentinel not in captured.out
        assert sentinel not in captured.err


def test_golden_journey_refuses_plaintext_dsn_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports = _fail_on_driver_import(monkeypatch)
    monkeypatch.setenv("OPENMED_PROFILE", "prod")

    result = journey.run_golden_journey(
        _PLAINTEXT_DSN,
        artifact_root="/synthetic/artifacts",
    )

    assert imports == []
    assert result["state"] == "failed"
    assert result["code"] == "postgres_tls_required"
    for sentinel in _TLS_SENTINELS:
        assert sentinel not in json.dumps(result)


def test_unusable_ca_bundle_is_reported_without_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    imports = _fail_on_driver_import(monkeypatch)
    missing = tmp_path / "sentinel-ca-path-3915.pem"

    report = journey.run_migrations(
        f"{_PLAINTEXT_DSN}?sslmode=verify-full",
        transport=journey.PostgresTransportSettings(
            sslrootcert=str(missing),
            profile="prod",
        ),
    ).to_dict()

    assert imports == []
    assert report["components"]["store"] == {
        "state": "unavailable",
        "code": "postgres_tls_config_invalid",
    }
    for sentinel in _TLS_SENTINELS:
        assert sentinel not in json.dumps(report)


def test_development_profile_keeps_explicit_plaintext_store_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict] = []

    def fake_connect(**kwargs):
        observed.append(kwargs)
        return object()

    monkeypatch.setattr(
        "openmed.structured.store.postgres.import_module",
        lambda name: type("Driver", (), {"connect": staticmethod(fake_connect)}),
    )
    monkeypatch.setattr(
        journey.PostgresJourneyStore,
        "open",
        classmethod(lambda cls, raw, **kwargs: StoreResult.success(_HealthyStore())),
    )
    monkeypatch.setenv("OPENMED_PROFILE", "dev")

    report = journey.run_migrations(_PLAINTEXT_DSN)

    assert report.ready
    assert observed[0]["ssl_context"] is None


def test_transport_posture_export_is_value_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_on_driver_import(monkeypatch)
    monkeypatch.setenv("OPENMED_PROFILE", "prod")

    refused = journey.journey_transport_posture(_PLAINTEXT_DSN)
    verified = journey.journey_transport_posture(
        f"{_PLAINTEXT_DSN}?sslmode=verify-full",
        transport=journey.PostgresTransportSettings(
            sslrootcert="/tls/sentinel-ca-path-3915.pem",
        ),
    )

    assert refused.to_dict()["code"] == "postgres_tls_required"
    assert refused.ok is False
    assert verified.to_dict() == {
        "production": True,
        "tls": "verify_full",
        "trust_anchor": "operator_ca_bundle",
        "client_certificate": False,
        "code": "postgres_tls_verified",
        "ok": True,
    }
    for sentinel in _TLS_SENTINELS:
        assert sentinel not in json.dumps(refused.to_dict())
        assert sentinel not in json.dumps(verified.to_dict())
        assert sentinel not in repr(verified)
