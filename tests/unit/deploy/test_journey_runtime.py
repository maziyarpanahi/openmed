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
