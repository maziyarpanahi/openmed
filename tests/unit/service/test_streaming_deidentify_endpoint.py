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

    def slow_stream(*args: Any, **kwargs: Any):
        worker_threads.append(threading.get_ident())
        time.sleep(0.3)
        yield StreamingDeidentificationEvent(redacted_text="safe output")
        yield StreamingDeidentificationEvent(
            redacted_text="",
            final=True,
            audit_record={"span_count": 0, "spans": []},
        )

    monkeypatch.setattr("openmed.service.streaming.deidentify_stream", slow_stream)
    app = create_app()

    async def scenario() -> tuple[httpx.Response, httpx.Response, float, int]:
        async with app.router.lifespan_context(app):
            loop_thread = threading.get_ident()
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url=LOOPBACK_BASE_URL,
            ) as async_client:
                started = time.perf_counter()
                stream_task = asyncio.create_task(
                    async_client.post(
                        "/pii/deidentify/stream",
                        json={"text": "synthetic input", "chunk_size": 4},
                    )
                )
                await asyncio.sleep(0.02)
                live = await async_client.get("/livez")
                live_elapsed = time.perf_counter() - started
                streamed = await stream_task
                return streamed, live, live_elapsed, loop_thread

    streamed, live, live_elapsed, loop_thread = asyncio.run(scenario())

    assert streamed.status_code == 200
    assert live.status_code == 200
    assert live_elapsed < 0.2
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
