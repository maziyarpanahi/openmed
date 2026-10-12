"""Tests for the streaming REST de-identification endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from datetime import datetime
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.core.streaming import StreamingDeidentificationEvent
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app

LOOPBACK_BASE_URL = "http://127.0.0.1"


class _FakeLoader:
    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, Any]:
        return {}

    def unload_model(self, model_name: str) -> dict[str, Any]:
        return {"model_name": model_name, "models": 0, "tokenizers": 0, "pipelines": 0}

    def unload_all_models(self) -> dict[str, int]:
        return {"models": 0, "tokenizers": 0, "pipelines": 0}


def _empty_prediction(text: str, *args: Any, **kwargs: Any) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name=str(kwargs.get("model_name") or "stub"),
        timestamp=datetime.now().isoformat(),
    )


def _name_prediction(text: str, *args: Any, **kwargs: Any) -> PredictionResult:
    marker = "Maria Garcia"
    start = text.find(marker)
    entities = []
    if start >= 0:
        entities.append(
            EntityPrediction(
                text=marker,
                label="NAME",
                confidence=0.99,
                start=start,
                end=start + len(marker),
            )
        )
    return PredictionResult(
        text=text,
        entities=entities,
        model_name=str(kwargs.get("model_name") or "stub"),
        timestamp=datetime.now().isoformat(),
    )


@pytest.fixture(autouse=True)
def _service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_LOG_LEVEL", "INFO")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCHING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_COALESCING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_TRUSTED_HOSTS", raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", _FakeLoader)


@pytest.fixture
def client() -> Any:
    app = create_app()
    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client


def _stream_events(client: TestClient, payload: dict[str, Any]) -> list[dict[str, Any]]:
    with client.stream("POST", "/pii/deidentify/stream", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        return [json.loads(line) for line in response.iter_lines() if line]


@patch("openmed.core.pii.extract_pii", side_effect=_empty_prediction)
def test_long_stream_matches_single_shot_byte_for_byte(
    _mock_extract: Any,
    client: TestClient,
) -> None:
    text = (
        "Routine synthetic follow-up note without identifiers. " * 100
        + "Contact jane.patient@example.com before discharge."
    )
    single = client.post(
        "/pii/deidentify",
        json={"text": text, "method": "mask"},
    )
    assert single.status_code == 200

    chunk_size = 257
    events = _stream_events(
        client,
        {"text": text, "method": "mask", "chunk_size": chunk_size},
    )
    chunks = [event for event in events if event["type"] == "chunk"]

    assert len(chunks) >= 2
    assert (
        "".join(event["redacted_text"] for event in chunks)
        == (single.json()["deidentified_text"])
    )
    assert events[-1]["type"] == "final"
    assert events[-1]["audit"]["stream"]["chunks"] == math.ceil(len(text) / chunk_size)


@patch("openmed.core.pii.extract_pii", side_effect=_empty_prediction)
def test_boundary_identifier_is_fully_redacted_and_never_logged(
    _mock_extract: Any,
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    identifier = "jane.patient@example.com"
    text = f"Contact {identifier} before the appointment tomorrow."
    events = _stream_events(
        client,
        {
            "text": text,
            "method": "mask",
            "chunk_size": len("Contact jane.patient@"),
        },
    )
    redacted = "".join(
        event["redacted_text"] for event in events if event["type"] == "chunk"
    )

    assert redacted == "Contact [email] before the appointment tomorrow."
    assert identifier not in redacted
    assert "jane.patient" not in redacted
    assert "example.com" not in redacted
    assert events[-1]["type"] == "final"
    assert events[-1]["audit"]["span_count"] == 1
    assert identifier not in json.dumps(events[-1]["audit"])
    assert identifier not in caplog.text


@patch("openmed.core.pii.extract_pii", side_effect=_name_prediction)
def test_final_line_preserves_opt_in_single_shot_mapping(
    _mock_extract: Any,
    client: TestClient,
) -> None:
    text = "Patient Maria Garcia was discharged."
    request = {"text": text, "method": "mask", "keep_mapping": True}
    single = client.post("/pii/deidentify", json=request)
    assert single.status_code == 200

    events = _stream_events(client, {**request, "chunk_size": 7})
    final = events[-1]

    assert final["type"] == "final"
    assert final["mapping"] == single.json()["mapping"]
    assert (
        "".join(event["redacted_text"] for event in events if event["type"] == "chunk")
        == single.json()["deidentified_text"]
    )
    assert final["spans"] == final["audit"]["spans"]


def test_blocking_core_iterator_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_threads: list[int] = []
    worker_started = threading.Event()
    worker_release = threading.Event()
    worker_released: list[bool] = []

    def slow_stream(*args: Any, **kwargs: Any):
        worker_threads.append(threading.get_ident())
        worker_started.set()
        worker_released.append(worker_release.wait(timeout=10.0))
        yield StreamingDeidentificationEvent(redacted_text="safe output")
        yield StreamingDeidentificationEvent(
            redacted_text="",
            final=True,
            audit_record={"span_count": 0, "spans": []},
        )

    monkeypatch.setattr("openmed.service.streaming.deidentify_stream", slow_stream)
    app = create_app()

    async def scenario() -> tuple[httpx.Response, httpx.Response, int]:
        async with app.router.lifespan_context(app):
            loop_thread = threading.get_ident()
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url=LOOPBACK_BASE_URL,
            ) as async_client:
                stream_task = asyncio.create_task(
                    async_client.post(
                        "/pii/deidentify/stream",
                        json={"text": "synthetic input", "chunk_size": 4},
                    )
                )

                async def wait_until_worker_starts() -> None:
                    while not worker_started.is_set():
                        await asyncio.sleep(0)

                await asyncio.wait_for(wait_until_worker_starts(), timeout=10.0)
                try:
                    live = await asyncio.wait_for(
                        async_client.get("/livez"), timeout=10.0
                    )
                finally:
                    worker_release.set()
                streamed = await asyncio.wait_for(
                    stream_task,
                    timeout=10.0,
                )
                return streamed, live, loop_thread

    streamed, live, loop_thread = asyncio.run(scenario())

    assert streamed.status_code == 200
    assert live.status_code == 200
    assert worker_released == [True]
    assert worker_threads and worker_threads[0] != loop_thread


def test_stream_timeout_uses_safe_ndjson_error_line(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def slow_stream(*args: Any, **kwargs: Any):
        time.sleep(0.05)
        yield StreamingDeidentificationEvent(redacted_text="safe output")

    monkeypatch.setattr("openmed.service.streaming.deidentify_stream", slow_stream)
    client.app.state.runtime.config.timeout = 0.01

    events = _stream_events(client, {"text": "synthetic input", "chunk_size": 4})

    assert events == [
        {
            "type": "error",
            "error": {
                "code": "timeout",
                "message": "Request exceeded configured timeout of 0.01 seconds",
                "details": {"timeout_seconds": 0.01},
            },
        }
    ]


SYNTHETIC_STREAM_TEXT = "Synthetic patient Zed Quillborne visited clinic."
LATE_FRAGMENT = "late redacted fragment"
KEEP_ALIVE = "30s"


class _ResidentFakeLoader(_FakeLoader):
    """Stub loader that reports every requested model as resident."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.resident: set[str] = set()

    def resolve_model_name(self, model_name: str) -> str:
        self.resident.add(model_name)
        return model_name

    def loaded_models(self) -> dict[str, Any]:
        return {
            name: {"models": 1, "tokenizers": 1, "pipelines": 1}
            for name in sorted(self.resident)
        }


class _SlowSecondWindow:
    """Synthetic core stream whose second window blocks until released."""

    def __init__(self) -> None:
        self.second_window_started = threading.Event()
        self.release = threading.Event()
        self.window_returned = threading.Event()
        self.closed = threading.Event()

    def __call__(self, *args: Any, **kwargs: Any):
        try:
            yield StreamingDeidentificationEvent(redacted_text="first window ")
            self.second_window_started.set()
            self.release.wait(timeout=10.0)
            self.window_returned.set()
            yield StreamingDeidentificationEvent(redacted_text=LATE_FRAGMENT)
            yield StreamingDeidentificationEvent(
                redacted_text="",
                final=True,
                audit_record={"span_count": 0, "spans": []},
            )
        finally:
            self.closed.set()


def _model_state(runtime: Any) -> dict[str, Any]:
    models = runtime.loaded_models()["models"]
    assert len(models) == 1
    return next(iter(models.values()))


def _wait_until(predicate: Any, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


async def _async_wait_until(predicate: Any, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())


def _stream_runtime(monkeypatch: pytest.MonkeyPatch, *, timeout: float) -> Any:
    monkeypatch.setattr(service_runtime, "ModelLoader", _ResidentFakeLoader)
    runtime = service_runtime.ServiceRuntime.from_env()
    runtime.config.timeout = timeout
    return runtime


def _record_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    runtime: Any,
) -> list[type[BaseException] | None]:
    outcomes: list[type[BaseException] | None] = []
    finish = runtime.finish_model_request

    def recording_finish(
        model_key: str,
        keep_alive: Any,
        error: BaseException | None = None,
    ) -> None:
        outcomes.append(None if error is None else type(error))
        finish(model_key, keep_alive, error)

    monkeypatch.setattr(runtime, "finish_model_request", recording_finish)
    return outcomes


def _stream_payload() -> Any:
    from openmed.service.streaming import PIIDeidentifyStreamRequest

    return PIIDeidentifyStreamRequest(
        text=SYNTHETIC_STREAM_TEXT,
        chunk_size=8,
        keep_alive=KEEP_ALIVE,
    )


def test_timeout_mid_window_releases_model_after_window_returns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from openmed.service.streaming import deidentify_ndjson_stream

    caplog.set_level(logging.DEBUG)
    core = _SlowSecondWindow()
    monkeypatch.setattr("openmed.service.streaming.deidentify_stream", core)
    runtime = _stream_runtime(monkeypatch, timeout=0.5)
    outcomes = _record_outcomes(monkeypatch, runtime)

    async def scenario() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        lines = [
            json.loads(line)
            async for line in deidentify_ndjson_stream(_stream_payload(), runtime)
        ]
        # The worker thread is still inside the second window, so the model
        # reference must stay held until that window returns.
        held = dict(_model_state(runtime))
        assert outcomes == []
        core.release.set()
        released = await _async_wait_until(
            lambda: _model_state(runtime)["active_requests"] == 0
        )
        assert released
        return lines, held

    events, held = asyncio.run(scenario())

    assert core.second_window_started.is_set()
    assert core.window_returned.is_set()
    assert held["active_requests"] == 1
    assert held["keep_alive_seconds_remaining"] is None
    assert events == [
        {"type": "chunk", "index": 0, "redacted_text": "first window "},
        {
            "type": "error",
            "error": {
                "code": "timeout",
                "message": "Request exceeded configured timeout of 0.5 seconds",
                "details": {"timeout_seconds": 0.5},
            },
        },
    ]
    state = _model_state(runtime)
    assert state["active_requests"] == 0
    assert state["keep_alive_seconds_remaining"] is not None
    assert 0 < state["keep_alive_seconds_remaining"] <= 30
    assert outcomes == [asyncio.TimeoutError]
    assert core.closed.is_set()
    assert LATE_FRAGMENT not in json.dumps(events)
    assert SYNTHETIC_STREAM_TEXT not in caplog.text
    assert "Zed Quillborne" not in caplog.text
    assert "generator already executing" not in caplog.text


def test_client_disconnect_mid_window_releases_model_after_window_returns(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from openmed.service.streaming import deidentify_ndjson_stream

    caplog.set_level(logging.DEBUG)
    core = _SlowSecondWindow()
    monkeypatch.setattr("openmed.service.streaming.deidentify_stream", core)
    runtime = _stream_runtime(monkeypatch, timeout=0)
    outcomes = _record_outcomes(monkeypatch, runtime)

    async def scenario() -> tuple[str, dict[str, Any]]:
        stream = deidentify_ndjson_stream(_stream_payload(), runtime)
        first = await stream.__anext__()
        pending = asyncio.ensure_future(stream.__anext__())
        started = await _async_wait_until(core.second_window_started.is_set)
        assert started
        # Starlette cancels the response body task when the client disconnects.
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()
        held = dict(_model_state(runtime))
        assert outcomes == []
        core.release.set()
        released = await _async_wait_until(
            lambda: _model_state(runtime)["active_requests"] == 0
        )
        assert released
        return first, held

    first, held = asyncio.run(scenario())

    assert json.loads(first) == {
        "type": "chunk",
        "index": 0,
        "redacted_text": "first window ",
    }
    assert held["active_requests"] == 1
    state = _model_state(runtime)
    assert state["active_requests"] == 0
    assert state["keep_alive_seconds_remaining"] is not None
    assert outcomes == [asyncio.CancelledError]
    assert core.closed.is_set()
    assert SYNTHETIC_STREAM_TEXT not in caplog.text
    assert "generator already executing" not in caplog.text


def test_completed_stream_releases_model_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openmed.service.streaming import deidentify_ndjson_stream

    core = _SlowSecondWindow()
    core.release.set()
    monkeypatch.setattr("openmed.service.streaming.deidentify_stream", core)
    runtime = _stream_runtime(monkeypatch, timeout=5.0)
    outcomes = _record_outcomes(monkeypatch, runtime)

    async def scenario() -> list[dict[str, Any]]:
        lines = [
            json.loads(line)
            async for line in deidentify_ndjson_stream(_stream_payload(), runtime)
        ]
        # Negative control: no deferred cleanup is needed once the core
        # iterator is exhausted, so the reference is free before the loop
        # gets another turn.
        assert _model_state(runtime)["active_requests"] == 0
        assert outcomes == [None]
        return lines

    events = asyncio.run(scenario())

    assert [event["type"] for event in events] == ["chunk", "chunk", "final"]
    assert events[1]["redacted_text"] == LATE_FRAGMENT
    state = _model_state(runtime)
    assert state["active_requests"] == 0
    assert state["keep_alive_seconds_remaining"] is not None
    assert core.closed.is_set()


def test_http_stream_timeout_mid_window_ends_with_one_error_and_releases(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    core = _SlowSecondWindow()
    monkeypatch.setattr("openmed.service.streaming.deidentify_stream", core)
    monkeypatch.setattr(service_runtime, "ModelLoader", _ResidentFakeLoader)
    app = create_app()
    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=True,
    ) as test_client:
        runtime = test_client.app.state.runtime
        runtime.config.timeout = 0.5
        events = _stream_events(
            test_client,
            {
                "text": SYNTHETIC_STREAM_TEXT,
                "chunk_size": 8,
                "keep_alive": KEEP_ALIVE,
            },
        )
        held = dict(_model_state(runtime))
        core.release.set()
        assert _wait_until(lambda: _model_state(runtime)["active_requests"] == 0)
        state = _model_state(runtime)

    assert core.second_window_started.is_set()
    assert held["active_requests"] == 1
    assert [event["type"] for event in events] == ["chunk", "error"]
    assert events[-1]["error"]["code"] == "timeout"
    assert LATE_FRAGMENT not in json.dumps(events)
    assert state["active_requests"] == 0
    assert state["keep_alive_seconds_remaining"] is not None
    assert core.closed.is_set()
    assert SYNTHETIC_STREAM_TEXT not in caplog.text
    assert "Zed Quillborne" not in caplog.text
    assert "generator already executing" not in caplog.text
