"""Tests for the optional REST Prometheus metrics endpoint."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

import openmed
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.metrics import (
    BATCH_QUEUE_DEPTH_NAME,
    BATCH_QUEUE_WAIT_NAME,
    BATCH_SHED_NAME,
    METRICS_ENABLED_ENV_VAR,
    MODEL_EVICTION_NAME,
    MODEL_LOAD_NAME,
    SPECULATIVE_ACCEPTANCE_RATE_NAME,
    SPECULATIVE_ACCEPTED_TOKEN_NAME,
    SPECULATIVE_DECODE_NAME,
    SPECULATIVE_DRAFT_TOKEN_NAME,
    SPECULATIVE_FALLBACK_NAME,
    SPECULATIVE_ROLLBACK_NAME,
    PrometheusMetricsRegistry,
)
from openmed.service.warm_pool import WarmPool

LOOPBACK_BASE_URL = "http://127.0.0.1"


class FakeLoader:
    """Loader double that keeps REST tests away from model downloads."""

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


class WarmPoolLoader:
    """Loader double that records warm-pool cache releases."""

    def __init__(self) -> None:
        self.pipelines: dict[tuple[Any, ...], object] = {}

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **kwargs: Any) -> object:
        key = (
            model_name,
            tuple(sorted((name, repr(value)) for name, value in kwargs.items())),
        )
        self.pipelines.setdefault(key, object())
        return self.pipelines[key]

    def unload_model(self, model_name: str) -> dict[str, Any]:
        keys = [key for key in self.pipelines if key[0] == model_name]
        for key in keys:
            self.pipelines.pop(key, None)
        return {
            "model_name": model_name,
            "models": 0,
            "tokenizers": 0,
            "pipelines": len(keys),
        }


def _configure_service_env(monkeypatch, *, metrics_enabled: bool) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCHING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_SIZE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_QUEUE_SIZE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_TRUSTED_HOSTS", raising=False)
    if metrics_enabled:
        monkeypatch.setenv(METRICS_ENABLED_ENV_VAR, "true")
    else:
        monkeypatch.delenv(METRICS_ENABLED_ENV_VAR, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)


def _metric_label_names(line: str) -> set[str]:
    labels = line.split("{", 1)[1].split("}", 1)[0]
    return {item.split("=", 1)[0] for item in labels.split(",")}


def test_metrics_endpoint_is_disabled_by_default(monkeypatch) -> None:
    _configure_service_env(monkeypatch, metrics_enabled=False)

    with TestClient(
        create_app(),
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/metrics")

    assert response.status_code == 404


def test_metrics_endpoint_renders_prometheus_text_when_enabled(monkeypatch) -> None:
    _configure_service_env(monkeypatch, metrics_enabled=True)

    with TestClient(
        create_app(),
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        health = client.get("/health")
        response = client.get("/metrics")

    assert health.status_code == 200
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE openmed_service_request_total counter" in response.text
    assert (
        'openmed_service_request_total{route="/health",status_code="200"} 1'
        in response.text
    )
    assert "# TYPE openmed_service_request_duration_seconds histogram" in response.text
    assert (
        'openmed_service_request_duration_seconds_count{route="/health"} 1'
        in response.text
    )
    assert "openmed_service_inflight_requests" in response.text
    assert f"{MODEL_LOAD_NAME} 0" in response.text
    assert f"{MODEL_EVICTION_NAME} 0" in response.text


def test_served_request_updates_metrics_without_phi_labels(monkeypatch) -> None:
    _configure_service_env(monkeypatch, metrics_enabled=True)

    def fake_analyze(*_: Any, **__: Any) -> dict[str, Any]:
        return {
            "text": "Paciente: Maria Garcia",
            "entities": [
                {
                    "text": "Maria Garcia",
                    "label": "NAME",
                    "start": 10,
                    "end": 22,
                }
            ],
            "model_name": "disease_detection_superclinical",
        }

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)

    with TestClient(
        create_app(),
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        served = client.post(
            "/analyze",
            json={
                "text": "Paciente: Maria Garcia",
                "model_name": "disease_detection_superclinical",
            },
        )
        response = client.get("/metrics")

    assert served.status_code == 200
    metrics_text = response.text
    assert 'openmed_service_request_total{route="/analyze",status_code="200"} 1' in (
        metrics_text
    )
    assert (
        'openmed_service_request_duration_seconds_count{route="/analyze"} 1'
        in metrics_text
    )
    assert "Paciente" not in metrics_text
    assert "Maria" not in metrics_text
    assert "NAME" not in metrics_text
    assert "disease_detection_superclinical" not in metrics_text

    metric_lines = [
        line
        for line in metrics_text.splitlines()
        if line.startswith("openmed_") and "{" in line
    ]
    assert metric_lines
    for line in metric_lines:
        assert _metric_label_names(line) <= {
            "route",
            "status_code",
            "le",
            "priority",
        }


def test_warm_pool_model_counters_are_aggregate_only() -> None:
    registry = PrometheusMetricsRegistry()
    loader = WarmPoolLoader()
    pool = WarmPool(lambda: loader, metrics=registry)

    pool.create_pipeline("model-a", aggregation_strategy="simple")
    pool.unload_model("model-a")

    metrics_text = registry.render()
    assert f"{MODEL_LOAD_NAME} 1" in metrics_text
    assert f"{MODEL_EVICTION_NAME} 1" in metrics_text
    assert "model-a" not in metrics_text


def test_batch_queue_metrics_are_exported_per_priority() -> None:
    registry = PrometheusMetricsRegistry(duration_buckets=(0.01, 0.1))

    registry.record_batch_queue_depth(priority="interactive", depth=2)
    registry.record_batch_queue_depth(priority="bulk", depth=1)
    registry.record_batch_queue_wait(priority="interactive", wait_seconds=0.02)
    registry.record_batch_shed(priority="bulk")

    metrics_text = registry.render()

    assert f'{BATCH_QUEUE_DEPTH_NAME}{{priority="interactive"}} 2' in metrics_text
    assert f'{BATCH_QUEUE_DEPTH_NAME}{{priority="bulk"}} 1' in metrics_text
    assert f'{BATCH_QUEUE_WAIT_NAME}_count{{priority="interactive"}} 1' in metrics_text
    assert f'{BATCH_SHED_NAME}{{priority="bulk"}} 1' in metrics_text


def test_speculative_decode_metrics_are_aggregate_only() -> None:
    registry = PrometheusMetricsRegistry()

    registry.record_speculative_decode(
        {
            "drafted_tokens": 6,
            "accepted_tokens": 4,
            "rollback_count": 1,
            "fallback_reason": "tokenizer_mismatch",
        }
    )
    metrics_text = registry.render()

    assert f"{SPECULATIVE_DECODE_NAME} 1" in metrics_text
    assert f"{SPECULATIVE_DRAFT_TOKEN_NAME} 6" in metrics_text
    assert f"{SPECULATIVE_ACCEPTED_TOKEN_NAME} 4" in metrics_text
    assert f"{SPECULATIVE_ROLLBACK_NAME} 1" in metrics_text
    assert f"{SPECULATIVE_FALLBACK_NAME} 1" in metrics_text
    assert f"{SPECULATIVE_ACCEPTANCE_RATE_NAME} 0.666666666667" in metrics_text
    assert "tokenizer_mismatch" not in metrics_text
