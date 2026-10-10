"""Unit tests for REST rate and concurrency throttling."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.processing.outputs import PredictionResult
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app

LOOPBACK_BASE_URL = "http://127.0.0.1"

_SERVICE_ENV_VARS = (
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_KEEP_ALIVE",
    "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
    "OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES",
    "OPENMED_SERVICE_DEFAULT_MODEL_FOOTPRINT_BYTES",
    "OPENMED_SERVICE_MODEL_ADMISSION_WAIT_SECONDS",
    "OPENMED_SERVICE_MAX_TEXT_LENGTH",
    "OPENMED_SERVICE_CORS_ORIGINS",
    "OPENMED_SERVICE_TRUSTED_HOSTS",
    "OPENMED_SERVICE_BATCHING_ENABLED",
    "OPENMED_SERVICE_BATCH_MAX_SIZE",
    "OPENMED_SERVICE_BATCH_MAX_WAIT_MS",
    "OPENMED_SERVICE_BATCH_MAX_QUEUE_SIZE",
    "OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
    "OPENMED_SERVICE_THROTTLE_KEY",
    "OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS",
    "OPENMED_SERVICE_COALESCING_ENABLED",
    "OPENMED_SERVICE_METRICS_ENABLED",
    "OPENMED_SERVICE_RETRY_MAX_ATTEMPTS",
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

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
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
                base_url=LOOPBACK_BASE_URL,
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


def test_batch_backpressure_returns_retry_after_before_model_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_BATCHING_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_SIZE", "8")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", "100")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_QUEUE_SIZE", "1")
    calls: list[str] = []
    calls_lock = threading.Lock()

    def fake_analyze(text: str, **_: Any) -> PredictionResult:
        with calls_lock:
            calls.append(text)
        return _prediction_result(text)

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    app = create_app()

    async def scenario():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url=LOOPBACK_BASE_URL,
            ) as client:
                first_task = asyncio.create_task(
                    client.post(
                        "/analyze",
                        json={"text": "held"},
                        headers={"x-openmed-priority": "bulk"},
                    )
                )
                await asyncio.sleep(0.02)
                shed = await client.post(
                    "/analyze",
                    json={"text": "shed"},
                    headers={"x-openmed-priority": "bulk"},
                )
                first = await first_task
                return first, shed

    first, shed = asyncio.run(scenario())

    assert first.status_code == 200
    payload = _assert_error_payload(shed, 503, "backpressure")
    assert int(shed.headers["Retry-After"]) >= 1
    assert payload["error"]["details"]["priority"] == "bulk"
    assert payload["error"]["details"]["queue_depth"] == 1
    assert payload["error"]["details"]["queue_capacity"] == 1
    assert calls == ["held"]


def test_probe_paths_do_not_consume_rate_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_RPS", "1")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_BURST", "1")

    def fake_analyze(text: str, **_: Any) -> PredictionResult:
        return _prediction_result(text)

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
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

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        responses = [
            client.post("/analyze", json={"text": text})
            for text in ("one", "two", "three")
        ]
        assert client.app.state.throttle.enabled is False

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert calls == ["one", "two", "three"]


def _orphaned_work(app) -> int:
    line = next(
        line
        for line in app.state.metrics.render().splitlines()
        if line.startswith("openmed_service_orphaned_work ")
    )
    return int(line.split()[1])


async def _wait_for_work_to_finish(app) -> None:
    async def wait():
        while app.state.inflight or _orphaned_work(app):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), 2)


def _model_request(mode: str, text: str) -> tuple[str, dict[str, Any]]:
    if mode == "graphql":
        return "/graphql", {
            "query": "query($input: AnalyzeInput!) { analyze(input: $input) { modelName } }",
            "variables": {"input": {"text": text}},
        }
    return "/analyze", {"text": text}


@pytest.mark.parametrize("mode", ["rest", "batch", "graphql"])
@pytest.mark.parametrize("worker_fails", [False, True])
def test_timeout_retains_capacity_until_model_work_finishes(
    monkeypatch: pytest.MonkeyPatch, caplog, mode: str, worker_fails: bool
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS", "0.01")
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_RETRY_MAX_ATTEMPTS", "1")
    if mode == "batch":
        monkeypatch.setenv("OPENMED_SERVICE_BATCHING_ENABLED", "true")
        monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_SIZE", "1")
        monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", "0")
    release = threading.Event()
    started = threading.Event()
    lock = threading.Lock()
    state = {"active": 0, "peak": 0, "calls": 0}

    def model(text: str, **_: Any) -> PredictionResult:
        with lock:
            state["calls"] += 1
            index = state["calls"]
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            if index == 1:
                started.set()
                assert release.wait(5)
                if worker_fails:
                    raise RuntimeError("SYNTHETIC_PRIVATE_FAILURE")
            return _prediction_result(text)
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(openmed, "analyze_text", model)
    app = create_app()

    async def scenario():
        async with app.router.lifespan_context(app):
            app.state.runtime.config.timeout = 0.1
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url=LOOPBACK_BASE_URL
            ) as client:
                path, body = _model_request(mode, "SYNTHETIC_PRIVATE_INPUT")
                try:
                    first = await client.post(path, json=body)
                    assert started.is_set()
                    if mode == "graphql":
                        assert first.status_code == 200
                        assert (
                            first.json()["errors"][0]["extensions"]["code"]
                            == "OPENMED_RESOLVER_ERROR"
                        )
                    else:
                        payload = _assert_error_payload(first, 504, "timeout")
                        assert payload["error"]["details"] == {"timeout_seconds": 0.1}
                    assert "SYNTHETIC_PRIVATE_INPUT" not in first.text
                    assert _orphaned_work(app) == 1
                    assert app.state.inflight == 1
                    for _ in range(2):
                        busy = await client.post(path, json=body)
                        _assert_error_payload(busy, 503, "service_busy")
                    assert state == {"active": 1, "peak": 1, "calls": 1}
                finally:
                    release.set()
                    await _wait_for_work_to_finish(app)
                assert state["active"] == 0
                recovered = await client.post(path, json=body)
                assert recovered.status_code == 200
                if mode == "graphql":
                    assert "errors" not in recovered.json()
                await _wait_for_work_to_finish(app)

    asyncio.run(scenario())
    assert state == {"active": 0, "peak": 1, "calls": 2}
    assert "SYNTHETIC_PRIVATE_FAILURE" not in caplog.text


@pytest.mark.parametrize("mode", ["rest", "batch", "coalesced"])
def test_cancelled_http_wait_retains_running_work(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    if mode == "batch":
        monkeypatch.setenv("OPENMED_SERVICE_BATCHING_ENABLED", "true")
        monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_SIZE", "1")
        monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", "0")
    if mode == "coalesced":
        monkeypatch.setenv("OPENMED_SERVICE_COALESCING_ENABLED", "true")
    release = threading.Event()
    started = threading.Event()
    calls = []

    def model(text: str, **_: Any) -> PredictionResult:
        calls.append(text)
        started.set()
        assert release.wait(5)
        return _prediction_result(text)

    monkeypatch.setattr(openmed, "analyze_text", model)
    app = create_app()

    async def scenario():
        async with app.router.lifespan_context(app):
            app.state.runtime.config.timeout = 0
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url=LOOPBACK_BASE_URL
            ) as client:
                task = asyncio.create_task(
                    client.post("/analyze", json={"text": "synthetic"})
                )
                try:
                    assert await asyncio.to_thread(started.wait, 2)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert _orphaned_work(app) == 1
                    assert app.state.inflight == 1
                    busy = await client.post("/analyze", json={"text": "synthetic"})
                    _assert_error_payload(busy, 503, "service_busy")
                    assert calls == ["synthetic"]
                finally:
                    release.set()
                    await _wait_for_work_to_finish(app)

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", [1, 2])
def test_parallel_graphql_resolvers_obey_model_concurrency_bound(
    monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", str(limit))
    monkeypatch.setenv("OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS", "0.01")
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    release = threading.Event()
    started = threading.Event()
    lock = threading.Lock()
    state = {"active": 0, "peak": 0, "calls": 0}

    def model(text: str, **_: Any) -> PredictionResult:
        with lock:
            state["calls"] += 1
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            if state["active"] == limit:
                started.set()
        try:
            assert release.wait(5)
            return _prediction_result(text)
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(openmed, "analyze_text", model)
    app = create_app()

    async def scenario():
        async with app.router.lifespan_context(app):
            app.state.runtime.config.timeout = 0
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url=LOOPBACK_BASE_URL
            ) as client:
                request = asyncio.create_task(
                    client.post(
                        "/graphql",
                        json={
                            "query": """query($input: AnalyzeInput!) {
                            first: analyze(input: $input) { modelName }
                            second: analyze(input: $input) { modelName }
                        }""",
                            "variables": {"input": {"text": "synthetic"}},
                        },
                    )
                )
                try:
                    assert await asyncio.to_thread(started.wait, 2)
                    assert state["calls"] == limit
                    assert state["peak"] == limit
                    busy = await client.post("/analyze", json={"text": "synthetic"})
                    _assert_error_payload(busy, 503, "service_busy")
                finally:
                    release.set()
                    response = await asyncio.wait_for(request, 2)
                    await _wait_for_work_to_finish(app)
                assert response.status_code == 200
                if limit == 1:
                    assert len(response.json()["errors"]) == 1
                    assert (
                        response.json()["errors"][0]["extensions"]["code"]
                        == "OPENMED_RESOLVER_ERROR"
                    )
                else:
                    assert "errors" not in response.json()

    asyncio.run(scenario())
    assert state["active"] == 0


@pytest.mark.parametrize("cancel_wait", [False, True])
def test_abandoned_queued_batch_does_not_start_model_work(
    monkeypatch: pytest.MonkeyPatch, cancel_wait: bool
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_BATCHING_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_SIZE", "8")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", "1000")
    calls = []

    def model(text: str, **_: Any) -> PredictionResult:
        calls.append(text)
        return _prediction_result(text)

    monkeypatch.setattr(openmed, "analyze_text", model)
    app = create_app()

    async def scenario():
        async with app.router.lifespan_context(app):
            app.state.runtime.config.timeout = 0 if cancel_wait else 0.1
            batcher = app.state.analyze_batcher
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url=LOOPBACK_BASE_URL
            ) as client:
                request = asyncio.create_task(
                    client.post("/analyze", json={"text": "abandoned"})
                )
                if cancel_wait:

                    async def wait_for_queue():
                        while not sum((await batcher.queue_depths()).values()):
                            await asyncio.sleep(0)

                    await asyncio.wait_for(wait_for_queue(), 2)
                    request.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await request
                else:
                    _assert_error_payload(await request, 504, "timeout")
                assert sum((await batcher.queue_depths()).values()) == 0
                await _wait_for_work_to_finish(app)
                assert calls == []
                batcher.max_wait_ms = 0
                app.state.runtime.config.timeout = 1
                recovered = await client.post("/analyze", json={"text": "recovered"})
                assert recovered.status_code == 200
                await _wait_for_work_to_finish(app)
                assert calls == ["recovered"]

    asyncio.run(scenario())


def test_parallel_graphql_fields_reuse_capacity_after_worker_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS", "1")
    lock = threading.Lock()
    state = {"active": 0, "peak": 0, "calls": 0}

    def model(text: str, **_: Any) -> PredictionResult:
        with lock:
            state["active"] += 1
            state["calls"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            time.sleep(0.03)
            return _prediction_result(text)
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(openmed, "analyze_text", model)
    app = create_app()

    async def scenario():
        async with app.router.lifespan_context(app):
            app.state.runtime.config.timeout = 0
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url=LOOPBACK_BASE_URL
            ) as client:
                response = await client.post(
                    "/graphql",
                    json={
                        "query": """query($input: AnalyzeInput!) {
                            first: analyze(input: $input) { modelName }
                            second: analyze(input: $input) { modelName }
                        }""",
                        "variables": {"input": {"text": "synthetic"}},
                    },
                )
                assert response.status_code == 200
                assert "errors" not in response.json()
                assert response.json()["data"]["first"] is not None
                assert response.json()["data"]["second"] is not None
                assert app.state.inflight == 0

    asyncio.run(scenario())
    assert state == {"active": 0, "peak": 1, "calls": 2}
