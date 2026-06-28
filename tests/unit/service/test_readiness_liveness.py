"""Readiness, liveness, and shutdown tests for the REST service."""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from openmed.service import runtime as service_runtime
from openmed.service.app import create_app

_SERVICE_ENV_VARS = (
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_KEEP_ALIVE",
    "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
    "OPENMED_SERVICE_MAX_TEXT_LENGTH",
    "OPENMED_SERVICE_BATCHING_ENABLED",
    "OPENMED_SERVICE_BATCH_MAX_SIZE",
    "OPENMED_SERVICE_BATCH_MAX_WAIT_MS",
    "OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
    "OPENMED_SERVICE_THROTTLE_KEY",
    "OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS",
)


class FakeLoader:
    """Minimal model loader double for service startup tests."""

    def __init__(self, config: Any):
        self.config = config

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        return object()

    def resolve_model_name(self, model_name: str) -> str:
        return model_name


@pytest.fixture(autouse=True)
def clean_service_env(monkeypatch):
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for env_var in _SERVICE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


@pytest.fixture(autouse=True)
def fake_loader(monkeypatch):
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)


def _assert_error_payload(response, status_code: int, code: str) -> dict[str, Any]:
    assert response.status_code == status_code
    payload = response.json()
    assert payload["error"]["code"] == code
    assert "message" in payload["error"]
    assert "details" in payload["error"]
    return payload


def test_livez_ok_before_preload():
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    try:
        response = client.get("/livez")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "openmed-rest"}


def test_readyz_503_before_preload():
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    try:
        response = client.get("/readyz")
    finally:
        client.close()

    _assert_error_payload(response, 503, "not_ready")


def test_readyz_200_after_preload():
    app = create_app()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "service": "openmed-rest"}


def test_health_alias_still_ok():
    app = create_app()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "openmed-rest"
    assert payload["profile"] == "test"


def test_shutdown_flips_ready_and_drains(monkeypatch):
    monkeypatch.setenv("OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS", "0.2")
    app = create_app()

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/readyz")
        assert response.status_code == 200
        client.app.state.inflight = 1
        shutdown_start = time.perf_counter()

    elapsed = time.perf_counter() - shutdown_start
    assert app.state.ready is False
    assert app.state.shutting_down is True
    assert elapsed >= 0.15


def test_model_request_rejected_during_shutdown():
    app = create_app()

    with TestClient(app, raise_server_exceptions=False) as client:
        client.app.state.shutting_down = True
        response = client.post("/analyze", json={"text": "sample"})

    _assert_error_payload(response, 503, "not_ready")
