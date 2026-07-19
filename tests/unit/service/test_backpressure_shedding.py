"""Tests for bounded batching admission and hysteretic load shedding."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime
from typing import Any

import httpx
import pytest

import openmed
from openmed.processing.outputs import PredictionResult
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.backpressure import AdmissionQueue, BackpressureError
from openmed.service.batcher import DynamicBatcher
from openmed.service.metrics import (
    ADMISSION_QUEUE_DEPTH_NAME,
    ADMISSION_QUEUE_SHEDDING_NAME,
    ADMISSION_QUEUE_WAIT_NAME,
    ADMISSION_SHED_NAME,
    PrometheusMetricsRegistry,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"


class FakeLoader:
    """Minimal loader double that keeps the burst test offline."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        return object()

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


def _prediction_result(text: str) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name="disease_detection_superclinical",
        timestamp=datetime.now().isoformat(),
        processing_time=0.01,
    )


def test_admission_queue_sheds_until_depth_reaches_low_watermark() -> None:
    queue = AdmissionQueue(
        queue_name="analyze",
        high_watermark=3,
        low_watermark=1,
        max_wait_ms=50,
    )

    for _ in range(3):
        queue.admit(priority="interactive")

    assert queue.snapshot().shedding is True
    with pytest.raises(BackpressureError) as at_high:
        queue.admit(priority="interactive")
    assert at_high.value.reason == "high_watermark"

    queue.release()
    assert queue.snapshot().depth == 2
    assert queue.snapshot().shedding is True
    with pytest.raises(BackpressureError):
        queue.admit(priority="interactive")

    queue.release()
    assert queue.snapshot().depth == 1
    assert queue.snapshot().shedding is False
    queue.admit(priority="interactive")
    assert queue.snapshot().depth == 2


def test_synthetic_burst_sheds_while_inflight_batch_completes_within_slo() -> None:
    async def scenario() -> tuple[list[str], BackpressureError, float]:
        dispatch_started = asyncio.Event()
        release_dispatch = asyncio.Event()

        async def dispatch(items: list[str]) -> list[str]:
            dispatch_started.set()
            await release_dispatch.wait()
            return list(items)

        batcher = DynamicBatcher(
            dispatch,
            max_batch_size=2,
            max_wait_ms=1000,
            max_queue_size_per_priority=8,
            high_watermark=2,
            low_watermark=0,
            max_queue_wait_ms=500,
            queue_name="analyze",
        )
        admitted = [
            asyncio.create_task(batcher.submit("first")),
            asyncio.create_task(batcher.submit("second")),
        ]
        await asyncio.wait_for(dispatch_started.wait(), timeout=0.2)

        snapshot = await batcher.admission_snapshot()
        assert snapshot.depth == 2
        assert snapshot.shedding is True
        with pytest.raises(BackpressureError) as shed:
            await batcher.submit("excess")

        released_at = time.perf_counter()
        release_dispatch.set()
        completed = await asyncio.wait_for(asyncio.gather(*admitted), timeout=0.5)
        completion_latency = time.perf_counter() - released_at

        drained = await batcher.admission_snapshot()
        assert drained.depth == 0
        assert drained.shedding is False
        return completed, shed.value, completion_latency

    completed, shed, completion_latency = asyncio.run(scenario())

    assert completed == ["first", "second"]
    assert shed.reason == "high_watermark"
    assert shed.queue_depth == 2
    assert completion_latency < 0.5


def test_request_waiting_beyond_maximum_is_shed_and_releases_admission() -> None:
    async def dispatch(items: list[str]) -> list[str]:
        return list(items)

    async def scenario() -> tuple[BackpressureError, int, bool]:
        batcher = DynamicBatcher(
            dispatch,
            max_batch_size=8,
            max_wait_ms=1000,
            high_watermark=2,
            low_watermark=0,
            max_queue_wait_ms=10,
            queue_name="analyze",
        )
        with pytest.raises(BackpressureError) as expired:
            await batcher.submit("stale")
        snapshot = await batcher.admission_snapshot()
        return expired.value, snapshot.depth, snapshot.shedding

    expired, depth, shedding = asyncio.run(scenario())

    assert expired.reason == "max_wait"
    assert expired.max_wait_ms == 10
    assert depth == 0
    assert shedding is False


def test_admission_metrics_expose_depth_state_wait_and_shed_count() -> None:
    metrics = PrometheusMetricsRegistry()
    queue = AdmissionQueue(
        queue_name="analyze",
        high_watermark=2,
        low_watermark=0,
        max_wait_ms=25,
        metrics=metrics,
    )

    queue.admit(priority="interactive")
    queue.admit(priority="interactive")
    with pytest.raises(BackpressureError):
        queue.admit(priority="interactive")
    queue.record_wait(0.012)

    rendered = metrics.render()
    assert f'{ADMISSION_QUEUE_DEPTH_NAME}{{queue="analyze"}} 2' in rendered
    assert f'{ADMISSION_QUEUE_SHEDDING_NAME}{{queue="analyze"}} 1' in rendered
    assert f'{ADMISSION_QUEUE_WAIT_NAME}{{queue="analyze"}} 0.012' in rendered
    assert f'{ADMISSION_SHED_NAME}{{queue="analyze"}} 1' in rendered


def test_batch_backpressure_config_reads_watermarks_and_max_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_BATCHING_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_QUEUE_SIZE", "20")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_HIGH_WATERMARK", "12")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_LOW_WATERMARK", "4")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_QUEUE_WAIT_MS", "75.5")

    config = service_runtime.parse_service_batching_config()

    assert config.enabled is True
    assert config.max_queue_size == 20
    assert config.high_watermark == 12
    assert config.low_watermark == 4
    assert config.max_queue_wait_ms == 75.5


def test_batch_backpressure_config_rejects_invalid_watermark_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_HIGH_WATERMARK", "4")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_LOW_WATERMARK", "4")

    with pytest.raises(ValueError, match="must be less than"):
        service_runtime.parse_service_batching_config()


def test_rest_burst_returns_503_retry_after_without_interrupting_inflight_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_BATCHING_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_SIZE", "1")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", "0")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_QUEUE_SIZE", "8")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_HIGH_WATERMARK", "1")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_LOW_WATERMARK", "0")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_QUEUE_WAIT_MS", "1000")
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_COALESCING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_RPS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_BURST", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", raising=False)

    first_started = threading.Event()
    release_first = threading.Event()

    def fake_analyze(text: str, **_: Any) -> PredictionResult:
        first_started.set()
        assert release_first.wait(timeout=2)
        return _prediction_result(text)

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    app = create_app()

    async def scenario() -> tuple[httpx.Response, httpx.Response, str, float]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url=LOOPBACK_BASE_URL,
            ) as client:
                first_task = asyncio.create_task(
                    client.post("/analyze", json={"text": "inflight"})
                )
                assert await asyncio.to_thread(first_started.wait, 1)
                shed = await client.post("/analyze", json={"text": "excess"})
                metrics_text = (await client.get("/metrics")).text
                released_at = time.perf_counter()
                release_first.set()
                first = await asyncio.wait_for(first_task, timeout=0.5)
                return (
                    first,
                    shed,
                    metrics_text,
                    time.perf_counter() - released_at,
                )

    first, shed, metrics_text, completion_latency = asyncio.run(scenario())

    assert first.status_code == 200
    assert shed.status_code == 503
    assert int(shed.headers["Retry-After"]) >= 1
    error = shed.json()["error"]
    assert error["code"] == "backpressure"
    assert error["details"]["queue"] == "analyze"
    assert error["details"]["reason"] == "high_watermark"
    assert "inflight" not in shed.text
    assert "excess" not in shed.text
    assert f'{ADMISSION_QUEUE_DEPTH_NAME}{{queue="analyze"}} 1' in metrics_text
    assert f'{ADMISSION_QUEUE_SHEDDING_NAME}{{queue="analyze"}} 1' in metrics_text
    assert f'{ADMISSION_SHED_NAME}{{queue="analyze"}} 1' in metrics_text
    assert completion_latency < 0.5
