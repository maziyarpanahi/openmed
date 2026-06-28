"""Unit tests for REST dynamic request batching."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import httpx
import pytest

from openmed.core import pii as pii_core
from openmed.processing.outputs import PredictionResult
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.batcher import DynamicBatcher


class FakeLoader:
    """Minimal loader double for service runtime tests."""

    instances: list["FakeLoader"] = []

    def __init__(self, config):
        self.config = config
        self.pipeline_calls: list[list[str]] = []
        FakeLoader.instances.append(self)

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **kwargs: Any):
        del model_name, kwargs

        def pipeline(texts, **call_kwargs):
            del call_kwargs
            batch = list(texts)
            self.pipeline_calls.append(batch)
            return [
                [
                    {
                        "word": text,
                        "entity_group": "TEST",
                        "score": 0.99,
                        "start": 0,
                        "end": len(text),
                    }
                ]
                for text in batch
            ]

        return pipeline

    def get_max_sequence_length(self, *_: Any, **__: Any) -> int:
        return 512

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


def _prediction_result(text: str) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name="pii-model",
        timestamp=datetime.now().isoformat(),
        processing_time=0.01,
    )


def _clear_batching_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENMED_SERVICE_BATCHING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_SIZE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_RPS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_BURST", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_THROTTLE_KEY", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS", raising=False)


def _enable_batching_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_RPS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_BURST", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_THROTTLE_KEY", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS", raising=False)
    monkeypatch.setenv("OPENMED_SERVICE_BATCHING_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_SIZE", "10")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", "25")


def test_concurrent_jobs_within_wait_window_dispatch_once():
    calls: list[list[int]] = []

    async def dispatch(items):
        calls.append(list(items))
        return [{"value": item} for item in items]

    async def scenario():
        batcher = DynamicBatcher(dispatch, max_batch_size=8, max_wait_ms=25)
        return await asyncio.gather(*(batcher.submit(index) for index in range(3)))

    results = asyncio.run(scenario())

    assert calls == [[0, 1, 2]]
    assert results == [{"value": 0}, {"value": 1}, {"value": 2}]


def test_max_batch_size_flushes_without_waiting_for_window():
    calls: list[list[str]] = []

    async def dispatch(items):
        calls.append(list(items))
        return list(items)

    async def scenario():
        batcher = DynamicBatcher(dispatch, max_batch_size=2, max_wait_ms=1000)
        return await asyncio.gather(batcher.submit("a"), batcher.submit("b"))

    results = asyncio.run(scenario())

    assert calls == [["a", "b"]]
    assert results == ["a", "b"]


def test_job_after_wait_window_forms_new_batch():
    calls: list[list[str]] = []

    async def dispatch(items):
        calls.append(list(items))
        return list(items)

    async def scenario():
        batcher = DynamicBatcher(dispatch, max_batch_size=8, max_wait_ms=5)
        first = await batcher.submit("first")
        second = await batcher.submit("second")
        return first, second

    results = asyncio.run(scenario())

    assert calls == [["first"], ["second"]]
    assert results == ("first", "second")


def test_per_request_error_result_does_not_fail_rest_of_batch():
    async def dispatch(items):
        return [ValueError("bad input"), {"value": items[1]}]

    async def scenario():
        batcher = DynamicBatcher(dispatch, max_batch_size=8, max_wait_ms=5)
        return await asyncio.gather(
            batcher.submit("bad"),
            batcher.submit("good"),
            return_exceptions=True,
        )

    bad, good = asyncio.run(scenario())

    assert isinstance(bad, ValueError)
    assert str(bad) == "bad input"
    assert good == {"value": "good"}


def test_service_batching_config_defaults_to_disabled(monkeypatch):
    _clear_batching_env(monkeypatch)

    config = service_runtime.parse_service_batching_config()

    assert config.enabled is False
    assert config.max_batch_size == service_runtime.DEFAULT_SERVICE_BATCH_MAX_SIZE
    assert config.max_wait_ms == service_runtime.DEFAULT_SERVICE_BATCH_MAX_WAIT_MS


def test_service_batching_config_reads_env(monkeypatch):
    monkeypatch.setenv("OPENMED_SERVICE_BATCHING_ENABLED", "on")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_SIZE", "4")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", "7.5")

    config = service_runtime.parse_service_batching_config()

    assert config.enabled is True
    assert config.max_batch_size == 4
    assert config.max_wait_ms == 7.5


def test_pii_extract_endpoint_batches_concurrent_requests(monkeypatch):
    _enable_batching_env(monkeypatch)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    calls: list[tuple[list[str], int]] = []

    def fake_extract_batch(texts, **kwargs: Any):
        calls.append((list(texts), kwargs["batch_size"]))
        return [_prediction_result(text) for text in texts]

    monkeypatch.setattr(pii_core, "_extract_pii_batch", fake_extract_batch)
    app = create_app()

    async def scenario():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                return await asyncio.gather(
                    client.post("/pii/extract", json={"text": "alpha"}),
                    client.post("/pii/extract", json={"text": "bravo"}),
                    client.post("/pii/extract", json={"text": "charlie"}),
                )

    responses = asyncio.run(scenario())

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert [response.json()["text"] for response in responses] == [
        "alpha",
        "bravo",
        "charlie",
    ]
    assert calls == [(["alpha", "bravo", "charlie"], 3)]


def test_analyze_endpoint_batches_compatible_concurrent_requests(monkeypatch):
    _enable_batching_env(monkeypatch)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    FakeLoader.instances.clear()
    app = create_app()

    async def scenario():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                payload = {"sentence_detection": False}
                return await asyncio.gather(
                    client.post("/analyze", json={"text": "alpha", **payload}),
                    client.post("/analyze", json={"text": "bravo", **payload}),
                    client.post("/analyze", json={"text": "charlie", **payload}),
                )

    responses = asyncio.run(scenario())

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert [response.json()["text"] for response in responses] == [
        "alpha",
        "bravo",
        "charlie",
    ]
    assert FakeLoader.instances[0].pipeline_calls == [["alpha", "bravo", "charlie"]]


def test_pii_extract_batches_only_compatible_request_shapes(monkeypatch):
    _enable_batching_env(monkeypatch)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    calls: list[tuple[list[str], str]] = []

    def fake_extract_batch(texts, **kwargs: Any):
        calls.append((list(texts), kwargs["lang"]))
        return [_prediction_result(text) for text in texts]

    monkeypatch.setattr(pii_core, "_extract_pii_batch", fake_extract_batch)
    app = create_app()

    async def scenario():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                return await asyncio.gather(
                    client.post("/pii/extract", json={"text": "alpha", "lang": "en"}),
                    client.post("/pii/extract", json={"text": "bravo", "lang": "es"}),
                    client.post("/pii/extract", json={"text": "charlie", "lang": "en"}),
                )

    responses = asyncio.run(scenario())

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert calls == [(["alpha", "charlie"], "en"), (["bravo"], "es")]


def test_pii_extract_endpoint_isolates_one_failed_batched_input(monkeypatch):
    _enable_batching_env(monkeypatch)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    calls: list[list[str]] = []

    def fake_extract_batch(texts, **kwargs: Any):
        del kwargs
        batch = list(texts)
        calls.append(batch)
        if "bad" in batch:
            raise ValueError("bad input")
        return [_prediction_result(text) for text in batch]

    monkeypatch.setattr(pii_core, "_extract_pii_batch", fake_extract_batch)
    app = create_app()

    async def scenario():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                return await asyncio.gather(
                    client.post("/pii/extract", json={"text": "good"}),
                    client.post("/pii/extract", json={"text": "bad"}),
                    client.post("/pii/extract", json={"text": "also good"}),
                )

    good, bad, also_good = asyncio.run(scenario())

    assert good.status_code == 200
    assert good.json()["text"] == "good"
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "bad_request"
    assert also_good.status_code == 200
    assert also_good.json()["text"] == "also good"
    assert calls[0] == ["good", "bad", "also good"]
    assert ["good"] in calls
    assert ["bad"] in calls
    assert ["also good"] in calls
