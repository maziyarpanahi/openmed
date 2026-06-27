"""Unit tests for REST request coalescing."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import httpx
import pytest

import openmed
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.coalesce import RequestCoalescer, coalescing_key
from openmed.service.schemas import AnalyzeRequest


class FakeLoader:
    """Minimal loader double for service runtime tests."""

    def __init__(self, config: Any):
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


def _enable_coalescing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCHING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_SIZE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", raising=False)
    monkeypatch.setenv("OPENMED_SERVICE_COALESCING_ENABLED", "true")


def _analyze_response(text: str, model_name: str) -> dict[str, Any]:
    return {"text": text, "model_name": model_name, "entities": []}


def test_service_coalescing_config_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("OPENMED_SERVICE_COALESCING_ENABLED", raising=False)

    config = service_runtime.parse_service_coalescing_config()

    assert config.enabled is False


def test_service_coalescing_config_reads_env(monkeypatch):
    monkeypatch.setenv("OPENMED_SERVICE_COALESCING_ENABLED", "on")

    config = service_runtime.parse_service_coalescing_config()

    assert config.enabled is True


def test_coalescing_key_uses_normalized_text_and_options() -> None:
    key = coalescing_key("/analyze", AnalyzeRequest(text="  alpha  "))
    same_key = coalescing_key("/analyze", AnalyzeRequest(text="alpha"))
    different_text = coalescing_key("/analyze", AnalyzeRequest(text="bravo"))
    different_options = coalescing_key(
        "/analyze",
        AnalyzeRequest(text="alpha", confidence_threshold=0.9),
    )

    assert same_key == key
    assert different_text != key
    assert different_options != key


def test_waiter_after_completion_before_eviction_receives_result() -> None:
    calls = 0

    async def scenario() -> tuple[dict[str, bool], dict[str, bool], int]:
        nonlocal calls
        coalescer = RequestCoalescer(eviction_delay_seconds=0.5)

        async def operation() -> dict[str, bool]:
            nonlocal calls
            calls += 1
            return {"ok": True}

        first = await coalescer.run("same", operation)
        second = await coalescer.run(
            "same",
            lambda: {"ok": False},
        )
        return first, second, calls

    first, second, call_count = asyncio.run(scenario())

    assert first == {"ok": True}
    assert second == {"ok": True}
    assert call_count == 1


def test_identical_concurrent_requests_trigger_one_model_call(monkeypatch) -> None:
    _enable_coalescing_env(monkeypatch)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    calls = 0

    def fake_analyze(text: str, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        with lock:
            calls += 1
        entered.set()
        assert release.wait(2.0)
        return _analyze_response(text, kwargs["model_name"])

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    app = create_app()

    async def scenario() -> list[httpx.Response]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                first = asyncio.create_task(
                    client.post("/analyze", json={"text": "alpha"})
                )
                assert await asyncio.to_thread(entered.wait, 2.0)
                second = asyncio.create_task(
                    client.post("/analyze", json={"text": "alpha"})
                )
                await asyncio.sleep(0.05)
                release.set()
                return list(await asyncio.gather(first, second))

    responses = asyncio.run(scenario())

    assert [response.status_code for response in responses] == [200, 200]
    assert [response.json()["text"] for response in responses] == ["alpha", "alpha"]
    assert calls == 1


@pytest.mark.parametrize(
    ("first_payload", "second_payload"),
    [
        ({"text": "alpha"}, {"text": "bravo"}),
        (
            {"text": "alpha", "confidence_threshold": 0.1},
            {"text": "alpha", "confidence_threshold": 0.9},
        ),
    ],
)
def test_non_identical_requests_do_not_coalesce(
    monkeypatch,
    first_payload,
    second_payload,
) -> None:
    _enable_coalescing_env(monkeypatch)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    lock = threading.Lock()
    calls: list[tuple[str, float | None]] = []

    def fake_analyze(text: str, **kwargs: Any) -> dict[str, Any]:
        with lock:
            calls.append((text, kwargs["confidence_threshold"]))
        return _analyze_response(text, kwargs["model_name"])

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    app = create_app()

    async def scenario() -> list[httpx.Response]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                return list(
                    await asyncio.gather(
                        client.post("/analyze", json=first_payload),
                        client.post("/analyze", json=second_payload),
                    )
                )

    responses = asyncio.run(scenario())

    assert [response.status_code for response in responses] == [200, 200]
    assert len(calls) == 2


def test_leader_error_propagates_to_all_joined_waiters(monkeypatch) -> None:
    _enable_coalescing_env(monkeypatch)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    calls = 0

    def fake_analyze(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        del args, kwargs
        with lock:
            calls += 1
        entered.set()
        assert release.wait(2.0)
        raise ValueError("bad input")

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    app = create_app()

    async def scenario() -> list[httpx.Response]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                first = asyncio.create_task(
                    client.post("/analyze", json={"text": "alpha"})
                )
                assert await asyncio.to_thread(entered.wait, 2.0)
                second = asyncio.create_task(
                    client.post("/analyze", json={"text": "alpha"})
                )
                await asyncio.sleep(0.05)
                release.set()
                return list(await asyncio.gather(first, second))

    responses = asyncio.run(scenario())

    assert [response.status_code for response in responses] == [400, 400]
    assert [response.json()["error"]["code"] for response in responses] == [
        "bad_request",
        "bad_request",
    ]
    assert calls == 1
