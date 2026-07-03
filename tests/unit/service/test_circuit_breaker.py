"""Unit tests for service retry and circuit-breaker behavior."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.metrics import (
    CIRCUIT_BREAKER_OPEN_NAME,
    METRICS_ENABLED_ENV_VAR,
)
from openmed.service.resilience import (
    CIRCUIT_CLOSED,
    CIRCUIT_HALF_OPEN,
    CIRCUIT_OPEN,
    CircuitBreakerOpenError,
    ResilienceManager,
    ServiceResilienceConfig,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FailingLoadLoader:
    """Loader double that simulates repeated model-load failures."""

    instances: list["FailingLoadLoader"] = []

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.create_pipeline_calls = 0
        FailingLoadLoader.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        self.create_pipeline_calls += 1
        raise RuntimeError("synthetic load failure")

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


def _clear_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCHING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_SIZE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_TRUSTED_HOSTS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_COALESCING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_RPS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_BURST", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_THROTTLE_KEY", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RESILIENCE_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RETRY_MAX_ATTEMPTS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RETRY_BACKOFF_INITIAL_SECONDS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RETRY_BACKOFF_MULTIPLIER", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RETRY_BACKOFF_MAX_SECONDS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RETRY_BACKOFF_JITTER_SECONDS", raising=False)
    monkeypatch.delenv(
        "OPENMED_SERVICE_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
        raising=False,
    )
    monkeypatch.delenv(
        "OPENMED_SERVICE_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS",
        raising=False,
    )
    monkeypatch.delenv(METRICS_ENABLED_ENV_VAR, raising=False)


def test_retries_use_backoff_jitter_and_attempt_cap() -> None:
    sleeps: list[float] = []
    attempts = 0
    manager = ResilienceManager(
        ServiceResilienceConfig(
            max_attempts=3,
            backoff_initial_seconds=0.1,
            backoff_multiplier=3.0,
            backoff_max_seconds=0.25,
            backoff_jitter_seconds=0.02,
            failure_threshold=5,
        ),
        sleep=sleeps.append,
        jitter=lambda upper: upper / 2,
    )

    def flaky_operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("temporary backend failure")
        return "ok"

    assert manager.execute("model-a", flaky_operation) == "ok"
    assert attempts == 3
    assert sleeps == pytest.approx([0.11, 0.26])
    assert manager.state_counts()[CIRCUIT_CLOSED] == 1


def test_value_error_is_not_retried_or_counted_as_backend_failure() -> None:
    sleeps: list[float] = []
    attempts = 0
    manager = ResilienceManager(
        ServiceResilienceConfig(
            max_attempts=3,
            failure_threshold=1,
        ),
        sleep=sleeps.append,
        jitter=lambda _: 0.0,
    )

    def bad_request_operation() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        manager.execute("model-a", bad_request_operation)

    assert attempts == 1
    assert sleeps == []
    assert manager.state_counts()[CIRCUIT_OPEN] == 0


def test_open_breaker_short_circuits_until_half_open_probe_recovers() -> None:
    clock = FakeClock()
    manager = ResilienceManager(
        ServiceResilienceConfig(
            max_attempts=1,
            failure_threshold=1,
            recovery_timeout_seconds=5.0,
        ),
        clock=clock,
        sleep=lambda _: None,
        jitter=lambda _: 0.0,
    )

    def fail_operation() -> str:
        raise RuntimeError("temporary backend failure")

    with pytest.raises(RuntimeError):
        manager.execute("remote-backend-a", fail_operation)

    assert manager.state_counts()[CIRCUIT_OPEN] == 1
    with pytest.raises(CircuitBreakerOpenError) as exc_info:
        manager.execute("remote-backend-a", lambda: "ok")
    assert exc_info.value.retry_after_seconds == 5

    clock.advance(5.0)
    assert manager.state_counts()[CIRCUIT_HALF_OPEN] == 1

    assert manager.execute("remote-backend-a", lambda: "ok") == "ok"
    assert manager.state_counts()[CIRCUIT_CLOSED] == 1


def test_service_repeated_load_failures_open_breaker_with_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_service_env(monkeypatch)
    monkeypatch.setenv(METRICS_ENABLED_ENV_VAR, "true")
    monkeypatch.setenv("OPENMED_SERVICE_RETRY_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("OPENMED_SERVICE_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("OPENMED_SERVICE_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS", "5")
    FailingLoadLoader.reset()
    monkeypatch.setattr(service_runtime, "ModelLoader", FailingLoadLoader)

    def fake_analyze(*_: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs["loader"].create_pipeline(kwargs["model_name"])
        return {"entities": []}

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)

    with TestClient(
        create_app(),
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        first = client.post("/analyze", json={"text": "Patient one"})
        second = client.post("/analyze", json={"text": "Patient two"})
        open_response = client.post("/analyze", json={"text": "Patient three"})
        metrics_response = client.get("/metrics")

    assert first.status_code == 500
    assert second.status_code == 500
    assert open_response.status_code == 503
    assert open_response.headers["Retry-After"] == "5"

    payload = open_response.json()
    assert payload["error"]["code"] == "circuit_breaker_open"
    assert payload["error"]["details"] == {
        "state": CIRCUIT_OPEN,
        "retry_after_seconds": 5,
    }
    assert "Patient" not in json.dumps(payload)
    assert "disease_detection_superclinical" not in json.dumps(payload)

    metrics_text = metrics_response.text
    assert f"{CIRCUIT_BREAKER_OPEN_NAME} 1" in metrics_text
    assert "disease_detection_superclinical" not in metrics_text
    assert "Patient" not in metrics_text
    assert FailingLoadLoader.instances[0].create_pipeline_calls == 2


def test_service_resilience_config_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_RESILIENCE_ENABLED", "false")
    monkeypatch.setenv("OPENMED_SERVICE_RETRY_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("OPENMED_SERVICE_RETRY_BACKOFF_INITIAL_SECONDS", "0.2")
    monkeypatch.setenv("OPENMED_SERVICE_RETRY_BACKOFF_MULTIPLIER", "2.5")
    monkeypatch.setenv("OPENMED_SERVICE_RETRY_BACKOFF_MAX_SECONDS", "3")
    monkeypatch.setenv("OPENMED_SERVICE_RETRY_BACKOFF_JITTER_SECONDS", "0.05")
    monkeypatch.setenv("OPENMED_SERVICE_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "7")
    monkeypatch.setenv(
        "OPENMED_SERVICE_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS",
        "12",
    )

    config = service_runtime.parse_service_resilience_config()

    assert config.enabled is False
    assert config.max_attempts == 4
    assert config.backoff_initial_seconds == 0.2
    assert config.backoff_multiplier == 2.5
    assert config.backoff_max_seconds == 3.0
    assert config.backoff_jitter_seconds == 0.05
    assert config.failure_threshold == 7
    assert config.recovery_timeout_seconds == 12.0
