"""Unit tests for REST rate and concurrency throttling."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.processing.outputs import PredictionResult
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
    """Minimal model loader double for throttled service tests."""

    def __init__(self, config: Any):
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        return object()

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


@pytest.fixture(autouse=True)
def clean_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for env_var in _SERVICE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


@pytest.fixture(autouse=True)
def fake_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)


def _prediction_result(text: str) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name="disease_detection_superclinical",
        timestamp=datetime.now().isoformat(),
        processing_time=0.01,
    )


def _assert_error_payload(response, status_code: int, code: str) -> dict[str, Any]:
    assert response.status_code == status_code
    payload = response.json()
    assert payload["error"]["code"] == code
    assert "message" in payload["error"]
    assert "details" in payload["error"]
    return payload


def test_service_throttle_config_defaults_to_disabled() -> None:
    config = service_runtime.parse_service_throttle_config()

    assert config.enabled is False
    assert config.rate_limit_rps == 0.0
    assert config.rate_limit_burst == 0
    assert config.max_concurrency == 0
    assert config.key_by == "global"


def test_service_throttle_config_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_RPS", "2.5")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_BURST", "4")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS", "0.125")
    monkeypatch.setenv("OPENMED_SERVICE_THROTTLE_KEY", "xff")

    config = service_runtime.parse_service_throttle_config()

    assert config.rate_limit_enabled is True
    assert config.concurrency_enabled is True
    assert config.rate_limit_rps == 2.5
    assert config.rate_limit_burst == 4
    assert config.max_concurrency == 3
    assert config.concurrency_wait_seconds == 0.125
    assert config.key_by == "x-forwarded-for"


def test_rate_limit_returns_429_with_retry_after(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_RPS", "1")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_BURST", "1")
    calls: list[str] = []

    def fake_analyze(text: str, **_: Any) -> PredictionResult:
        calls.append(text)
        return _prediction_result(text)

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    app = create_app()

    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post("/analyze", json={"text": "first"})
        second = client.post("/analyze", json={"text": "second"})

    assert first.status_code == 200
    _assert_error_payload(second, 429, "rate_limited")
    assert int(second.headers["Retry-After"]) >= 1
    assert calls == ["first"]


def test_concurrency_limit_returns_503_after_bounded_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS", "0.02")
    first_started = threading.Event()
    release_first = threading.Event()
    calls: list[str] = []
    calls_lock = threading.Lock()

    def fake_analyze(text: str, **_: Any) -> PredictionResult:
        with calls_lock:
            calls.append(text)
            call_index = len(calls)
        if call_index == 1:
            first_started.set()
            assert release_first.wait(timeout=2)
        return _prediction_result(text)

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    app = create_app()

    async def scenario():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                first_task = asyncio.create_task(
                    client.post("/analyze", json={"text": "held"})
                )
                assert await asyncio.to_thread(first_started.wait, 1)
                busy = await client.post("/analyze", json={"text": "queued"})
                release_first.set()
                first = await first_task
                return first, busy

    first, busy = asyncio.run(scenario())

    assert first.status_code == 200
    _assert_error_payload(busy, 503, "service_busy")
    assert calls == ["held"]


def test_probe_paths_do_not_consume_rate_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_RPS", "1")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_BURST", "1")

    def fake_analyze(text: str, **_: Any) -> PredictionResult:
        return _prediction_result(text)

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    app = create_app()

    with TestClient(app, raise_server_exceptions=False) as client:
        for _ in range(3):
            assert client.get("/health").status_code == 200
            assert client.get("/livez").status_code == 200
            assert client.get("/readyz").status_code == 200
        first = client.post("/analyze", json={"text": "first"})
        second = client.post("/analyze", json={"text": "second"})

    assert first.status_code == 200
    _assert_error_payload(second, 429, "rate_limited")


def test_throttle_middleware_is_noop_when_limits_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_analyze(text: str, **_: Any) -> PredictionResult:
        calls.append(text)
        return _prediction_result(text)

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    app = create_app()

    with TestClient(app, raise_server_exceptions=False) as client:
        responses = [
            client.post("/analyze", json={"text": text})
            for text in ("one", "two", "three")
        ]
        assert client.app.state.throttle.enabled is False

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert calls == ["one", "two", "three"]
