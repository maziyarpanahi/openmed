"""Deadline, write-idle, and event-cap bounds for the REST NDJSON streams."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime
from typing import Any, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.core.streaming import StreamingDeidentificationEvent
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.service import runtime as service_runtime
from openmed.service import streaming as service_streaming
from openmed.service.app import create_app
from openmed.service.throttle import ConcurrencyGate, RequestWork

LOOPBACK_BASE_URL = "http://127.0.0.1"
EXTRACT_PATH = "/pii/extract/stream"
DEIDENTIFY_PATH = "/pii/deidentify/stream"
# Bounds every wait for a worker thread or a stream to settle. Passing runs
# return at once; the bound only stops a regression from hanging the suite.
_SETTLE_SECONDS = 10.0
_MARKER = "QXSYNTHPT"
_CHUNK = "x" * 7 + _MARKER  # 16 characters: one model window per chunk.
_CHUNK_COUNT = 6
_TEXT = _CHUNK * _CHUNK_COUNT
_STALL_CHUNK_COUNT = 50

_SERVICE_ENV_VARS = (
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_KEEP_ALIVE",
    "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
    "OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES",
    "OPENMED_SERVICE_BATCHING_ENABLED",
    "OPENMED_SERVICE_COALESCING_ENABLED",
    "OPENMED_SERVICE_CORS_ORIGINS",
    "OPENMED_SERVICE_TRUSTED_HOSTS",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_THROTTLE_KEY",
    "OPENMED_SERVICE_STREAM_WRITE_IDLE_SECONDS",
    "OPENMED_SERVICE_STREAM_MAX_EVENTS",
)


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


class _Classifier:
    """Synthetic PII classifier whose ``block_on_call`` window blocks."""

    def __init__(self, block_on_call: Optional[int] = None) -> None:
        self.block_on_call = block_on_call
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, text: str, *args: Any, **kwargs: Any) -> PredictionResult:
        with self._lock:
            self.calls += 1
            index = self.calls
        if index == self.block_on_call:
            self.started.set()
            self.release.wait(_SETTLE_SECONDS)
        return _marker_prediction(text)


def _marker_prediction(text: str) -> PredictionResult:
    entities = []
    start = text.find(_MARKER)
    while start >= 0:
        entities.append(
            EntityPrediction(
                text=_MARKER,
                label="NAME",
                confidence=0.99,
                start=start,
                end=start + len(_MARKER),
            )
        )
        start = text.find(_MARKER, start + len(_MARKER))
    return PredictionResult(
        text=text,
        entities=entities,
        model_name="stub",
        timestamp=datetime.now().isoformat(),
    )


class _CoreStream:
    """Synthetic de-identification core stream that records how far it ran."""

    def __init__(self, chunks: int) -> None:
        self.chunks = chunks
        self.produced = 0
        self.closed = False

    def __call__(self, *args: Any, **kwargs: Any):
        try:
            for index in range(self.chunks):
                self.produced += 1
                yield StreamingDeidentificationEvent(redacted_text=f"[NAME] {index} ")
            yield StreamingDeidentificationEvent(
                redacted_text="",
                final=True,
                audit_record={"span_count": 0, "spans": []},
            )
        finally:
            self.closed = True


class _StallingConsumer:
    """Raw ASGI peer that reads ``read_lines`` body lines, then stops reading.

    A blocked ``send`` models a socket whose peer no longer reads: it never
    completes on its own, only cancellation ends it. With ``resume=True`` the
    consumer stalls once and then reads normally again (a slow consumer).
    """

    def __init__(self, body: bytes, *, read_lines: int, resume: bool = False):
        self.body = body
        self.read_lines = read_lines
        self.resume = resume
        self.messages: list[dict[str, Any]] = []
        self.stalled = asyncio.Event()
        self._request_sent = False
        self._disconnected = asyncio.Event()

    async def receive(self) -> dict[str, Any]:
        if not self._request_sent:
            self._request_sent = True
            return {"type": "http.request", "body": self.body, "more_body": False}
        await self._disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(self, message: dict[str, Any]) -> None:
        stall = (
            message["type"] == "http.response.body"
            and len(self.lines()) >= self.read_lines
            and not (self.resume and self.stalled.is_set())
        )
        if stall:
            self.stalled.set()
            await asyncio.Event().wait()
        self.messages.append(message)

    def lines(self) -> list[dict[str, Any]]:
        body = b"".join(
            message.get("body", b"")
            for message in self.messages
            if message["type"] == "http.response.body"
        )
        return [json.loads(line) for line in body.splitlines() if line]

    def completed(self) -> bool:
        return any(
            message["type"] == "http.response.body"
            and not message.get("more_body", False)
            for message in self.messages
        )


@pytest.fixture(autouse=True)
def _service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for env_var in _SERVICE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    # One admission slot and metrics, so permit and orphaned-work accounting
    # are observable from outside the stream.
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS", "0.01")
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_RETRY_MAX_ATTEMPTS", "1")
    monkeypatch.setattr(service_runtime, "ModelLoader", _FakeLoader)


def _extract_body(text: str = _TEXT) -> dict[str, Any]:
    return {
        "text": text,
        "chunk_size": len(_CHUNK),
        "window_chars": 64,
        "tokenizer_context_chars": 16,
        "max_entity_chars": 32,
    }


def _deidentify_body(text: str = _TEXT) -> dict[str, Any]:
    return {"text": text, "method": "mask", "chunk_size": len(_CHUNK)}


def _ndjson_lines(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line]


def _active_requests(app: Any) -> int:
    models = app.state.runtime.loaded_models()["models"]
    return sum(int(state.get("active_requests", 0)) for state in models.values())


def _orphaned_work(app: Any) -> int:
    line = next(
        line
        for line in app.state.metrics.render().splitlines()
        if line.startswith("openmed_service_orphaned_work ")
    )
    return int(line.split()[1])


def _at_baseline(app: Any) -> bool:
    return not (app.state.inflight or _orphaned_work(app) or _active_requests(app))


async def _wait_for_baseline(app: Any) -> None:
    async def wait() -> None:
        while not _at_baseline(app):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), _SETTLE_SECONDS)


def _advance_loop_clock(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """Make the running loop's pending timers due *seconds* early."""
    loop = asyncio.get_running_loop()
    clock = loop.time
    monkeypatch.setattr(loop, "time", lambda: clock() + seconds)


async def _assert_admission_busy(client: httpx.AsyncClient) -> None:
    busy = await client.post("/pii/extract", json={"text": "synthetic probe"})
    assert busy.status_code == 503
    assert busy.json()["error"]["code"] == "service_busy"


async def _assert_admission_free(client: httpx.AsyncClient) -> None:
    probe = await client.post("/pii/extract", json={"text": "synthetic probe"})
    assert probe.status_code == 200


def _http_scope(path: str, body: bytes, spec_version: str) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"127.0.0.1"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        "client": ("127.0.0.1", 50123),
        "server": ("127.0.0.1", 80),
    }


def _assert_logs_are_value_free(caplog: pytest.LogCaptureFixture) -> None:
    assert _MARKER not in caplog.text
    assert _TEXT not in caplog.text


def test_extract_stream_deadline_ends_with_one_timeout_event(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    classifier = _Classifier(block_on_call=2)
    monkeypatch.setattr(openmed, "extract_pii", classifier)
    app = create_app()
    timeout_seconds = 30.0

    async def scenario() -> httpx.Response:
        async with app.router.lifespan_context(app):
            app.state.runtime.config.timeout = timeout_seconds
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
            async with httpx.AsyncClient(
                transport=transport, base_url=LOOPBACK_BASE_URL
            ) as client:
                try:
                    request = asyncio.create_task(
                        client.post(EXTRACT_PATH, json=_extract_body())
                    )
                    assert await asyncio.to_thread(
                        classifier.started.wait, _SETTLE_SECONDS
                    )
                    # The second window is blocked; jump the loop clock past
                    # the one request deadline instead of sleeping through it.
                    _advance_loop_clock(monkeypatch, timeout_seconds)
                    response = await asyncio.wait_for(request, _SETTLE_SECONDS)

                    # Negative control: the window still runs in its worker
                    # thread, so its model reference and permit stay held and
                    # the slot is quarantined as orphaned work.
                    assert not classifier.release.is_set()
                    assert _active_requests(app) == 1
                    assert _orphaned_work(app) == 1
                    assert app.state.inflight == 1
                    await _assert_admission_busy(client)
                finally:
                    classifier.release.set()
                    await _wait_for_baseline(app)
                await _assert_admission_free(client)
                return response

    response = asyncio.run(scenario())

    assert response.status_code == 200
    events = _ndjson_lines(response.text)
    assert [event["type"] for event in events] == ["emit", "error"]
    assert events[-1] == {
        "type": "error",
        "error": {
            "code": "timeout",
            "message": "Request exceeded configured timeout of 30 seconds",
            "details": {"timeout_seconds": timeout_seconds},
        },
    }
    assert classifier.calls == 3  # two stream windows, then one admission probe
    _assert_logs_are_value_free(caplog)


def test_extract_stream_deadline_is_checked_before_a_window_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classifier = _Classifier()
    monkeypatch.setattr(openmed, "extract_pii", classifier)
    runtime = service_runtime.ServiceRuntime.from_env()
    timeout_seconds = 30.0

    async def scenario() -> None:
        events = runtime.stream_pii_extract(
            text=_TEXT,
            model_name=runtime.served_model_ids()[0],
            confidence_threshold=0.5,
            use_smart_merging=True,
            lang="en",
            normalize_accents=None,
            keep_alive=None,
            chunk_size=len(_CHUNK),
            window_chars=64,
            tokenizer_context_chars=16,
            max_entity_chars=32,
            timeout_seconds=timeout_seconds,
        )
        first = await events.__anext__()
        assert first.type == "emit"
        assert classifier.calls == 1

        # The deadline passes between windows: no further window may start.
        _advance_loop_clock(monkeypatch, timeout_seconds)
        with pytest.raises(service_runtime.StreamDeadlineExceeded) as raised:
            await events.__anext__()
        assert raised.value.timeout_seconds == timeout_seconds
        assert _MARKER not in str(raised.value)

    asyncio.run(scenario())

    assert classifier.calls == 1
    models = runtime.loaded_models()["models"]
    assert all(state["active_requests"] == 0 for state in models.values())


def test_extract_stream_disconnect_mid_window_releases_after_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classifier = _Classifier(block_on_call=2)
    monkeypatch.setattr(openmed, "extract_pii", classifier)
    app = create_app()

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            app.state.runtime.config.timeout = 0
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
            async with httpx.AsyncClient(
                transport=transport, base_url=LOOPBACK_BASE_URL
            ) as client:
                try:
                    request = asyncio.create_task(
                        client.post(EXTRACT_PATH, json=_extract_body())
                    )
                    assert await asyncio.to_thread(
                        classifier.started.wait, _SETTLE_SECONDS
                    )
                    request.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await request

                    assert _active_requests(app) == 1
                    assert _orphaned_work(app) == 1
                    await _assert_admission_busy(client)
                finally:
                    classifier.release.set()
                    await _wait_for_baseline(app)
                await _assert_admission_free(client)

    asyncio.run(scenario())


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
@pytest.mark.parametrize("path", [EXTRACT_PATH, DEIDENTIFY_PATH])
def test_stalled_consumer_is_disconnected_after_write_idle_bound(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    path: str,
    spec_version: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    classifier = _Classifier()
    core = _CoreStream(_STALL_CHUNK_COUNT)
    monkeypatch.setattr(openmed, "extract_pii", classifier)
    monkeypatch.setattr(service_streaming, "deidentify_stream", core)
    monkeypatch.setattr(service_streaming, "STREAM_TERMINAL_GRACE_SECONDS", 0.01)
    app = create_app()
    idle_seconds = 30.0
    text = _CHUNK * _STALL_CHUNK_COUNT
    payload = _extract_body(text) if path == EXTRACT_PATH else _deidentify_body(text)
    consumer = _StallingConsumer(json.dumps(payload).encode("utf-8"), read_lines=1)

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            runtime = app.state.runtime
            runtime.config.timeout = 0
            runtime.stream_write_idle_seconds = idle_seconds
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
            async with httpx.AsyncClient(
                transport=transport, base_url=LOOPBACK_BASE_URL
            ) as client:
                stream = asyncio.create_task(
                    app(
                        _http_scope(path, consumer.body, spec_version),
                        consumer.receive,
                        consumer.send,
                    )
                )
                await asyncio.wait_for(consumer.stalled.wait(), _SETTLE_SECONDS)

                # Negative control: while the consumer is stalled (and before
                # the bound), the stream keeps its model reference and permit.
                assert not stream.done()
                assert _active_requests(app) == 1
                assert app.state.inflight == 1
                await _assert_admission_busy(client)

                _advance_loop_clock(monkeypatch, idle_seconds)
                await asyncio.wait_for(stream, _SETTLE_SECONDS)
                await _wait_for_baseline(app)
                await _assert_admission_free(client)

    asyncio.run(scenario())

    lines = consumer.lines()
    assert len(lines) == 1
    assert lines[0]["type"] in {"emit", "chunk"}
    # The consumer never read again, so the response is left incomplete and
    # the server closes the connection; the producer stopped early.
    assert not consumer.completed()
    if path == EXTRACT_PATH:
        # Admission probes add one classifier call each.
        assert classifier.calls - 2 < _STALL_CHUNK_COUNT
    else:
        assert core.produced < _STALL_CHUNK_COUNT
        assert core.closed
    assert service_streaming.STREAM_WRITE_IDLE_CODE in caplog.text
    _assert_logs_are_value_free(caplog)


@pytest.mark.parametrize("path", [EXTRACT_PATH, DEIDENTIFY_PATH])
def test_slow_consumer_gets_one_terminal_write_idle_event(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    monkeypatch.setattr(openmed, "extract_pii", _Classifier())
    monkeypatch.setattr(
        service_streaming, "deidentify_stream", _CoreStream(_STALL_CHUNK_COUNT)
    )
    app = create_app()
    text = _CHUNK * _STALL_CHUNK_COUNT
    payload = _extract_body(text) if path == EXTRACT_PATH else _deidentify_body(text)
    consumer = _StallingConsumer(
        json.dumps(payload).encode("utf-8"), read_lines=2, resume=True
    )

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            runtime = app.state.runtime
            runtime.config.timeout = 0
            # The stalled write never completes on its own, so any bound fires.
            runtime.stream_write_idle_seconds = 0.05
            await asyncio.wait_for(
                app(
                    _http_scope(path, consumer.body, "2.4"),
                    consumer.receive,
                    consumer.send,
                ),
                _SETTLE_SECONDS,
            )
            await _wait_for_baseline(app)

    asyncio.run(scenario())

    lines = consumer.lines()
    assert len(lines) == 3
    assert [line["type"] for line in lines[:2]] in (
        ["emit", "emit"],
        ["chunk", "chunk"],
    )
    assert lines[-1] == {
        "type": "error",
        "error": {
            "code": "stream_write_idle",
            "message": "Stream consumer did not read within the write-idle bound",
            "details": {"write_idle_seconds": 0.05},
        },
    }
    assert consumer.completed()
    assert consumer.messages[-1] == {
        "type": "http.response.body",
        "body": b"",
        "more_body": False,
    }


@pytest.mark.parametrize("path", [EXTRACT_PATH, DEIDENTIFY_PATH])
def test_event_cap_ends_stream_with_one_value_free_event(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    path: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    core = _CoreStream(_CHUNK_COUNT)
    monkeypatch.setattr(openmed, "extract_pii", _Classifier())
    monkeypatch.setattr(service_streaming, "deidentify_stream", core)
    payload = _extract_body() if path == EXTRACT_PATH else _deidentify_body()

    with TestClient(app := create_app(), base_url=LOOPBACK_BASE_URL) as client:
        client.app.state.runtime.stream_max_events = 2
        with client.stream("POST", path, json=payload) as response:
            assert response.status_code == 200
            events = [json.loads(line) for line in response.iter_lines() if line]
        assert _at_baseline(app)

    assert len(events) == 3
    assert events[-1] == {
        "type": "error",
        "error": {
            "code": "stream_event_limit",
            "message": "Stream exceeded the configured event limit",
            "details": {"max_events": 2},
        },
    }
    assert all(event["type"] != "final" for event in events)
    assert _MARKER not in json.dumps(events[-1])
    if path == DEIDENTIFY_PATH:
        assert core.closed
        assert core.produced == 3
    _assert_logs_are_value_free(caplog)


@pytest.mark.parametrize(
    ("path", "event_count"),
    [(EXTRACT_PATH, _CHUNK_COUNT + 1), (DEIDENTIFY_PATH, _CHUNK_COUNT + 1)],
)
def test_stream_that_fits_the_event_cap_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    event_count: int,
) -> None:
    monkeypatch.setattr(openmed, "extract_pii", _Classifier())
    monkeypatch.setattr(
        service_streaming, "deidentify_stream", _CoreStream(_CHUNK_COUNT)
    )
    payload = _extract_body() if path == EXTRACT_PATH else _deidentify_body()

    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        client.app.state.runtime.stream_max_events = event_count
        with client.stream("POST", path, json=payload) as response:
            events = [json.loads(line) for line in response.iter_lines() if line]

    assert len(events) == event_count
    assert events[-1]["type"] == "final"
    assert all(event["type"] != "error" for event in events)


def test_extract_stream_model_error_ends_with_value_free_event(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    def failing(text: str, *args: Any, **kwargs: Any) -> PredictionResult:
        raise RuntimeError(f"synthetic backend failure for {text}")

    monkeypatch.setattr(openmed, "extract_pii", failing)

    with TestClient(app := create_app(), base_url=LOOPBACK_BASE_URL) as client:
        with client.stream("POST", EXTRACT_PATH, json=_extract_body()) as response:
            events = [json.loads(line) for line in response.iter_lines() if line]
        assert _at_baseline(app)

    assert events == [
        {
            "type": "error",
            "error": {
                "code": "internal_error",
                "message": "Internal server error",
                "details": None,
            },
        }
    ]
    _assert_logs_are_value_free(caplog)


def test_stalled_end_of_body_write_is_retried_without_an_error_line() -> None:
    async def scenario() -> tuple[service_streaming.StreamWriteGuard, list[Any]]:
        sent: list[dict[str, Any]] = []
        stalled = False

        async def send(message: dict[str, Any]) -> None:
            nonlocal stalled
            end_of_body = message["type"] == "http.response.body" and not message.get(
                "more_body"
            )
            if end_of_body and not stalled:
                stalled = True
                await asyncio.Event().wait()
            sent.append(message)

        guard = service_streaming.StreamWriteGuard(0.01)
        bounded = guard.wrap(send)
        await bounded({"type": "http.response.start", "status": 200, "headers": []})
        final = b'{"type":"final"}\n'
        await bounded({"type": "http.response.body", "body": final, "more_body": True})
        await bounded({"type": "http.response.body", "body": b"", "more_body": False})
        await bounded(
            {"type": "http.response.body", "body": b"late", "more_body": True}
        )
        return guard, sent

    guard, sent = asyncio.run(scenario())

    assert guard.expired
    # Every event was already handed off, so no error line follows "final".
    assert [message.get("body") for message in sent] == [
        None,
        b'{"type":"final"}\n',
        b"",
    ]


def test_request_work_hold_keeps_permit_until_released() -> None:
    class _Metrics:
        def __init__(self) -> None:
            self.orphaned = 0

        def orphaned_work_started(self) -> None:
            self.orphaned += 1

        def orphaned_work_finished(self) -> None:
            self.orphaned -= 1

    async def scenario() -> None:
        gate = ConcurrencyGate(max_concurrency=1, wait_seconds=0)
        metrics = _Metrics()
        work = RequestWork(await gate.acquire(), gate, "global", "global", metrics)
        finished: list[bool] = []
        work.on_finished(lambda: finished.append(True))
        release = work.hold()

        work.close()  # Headers sent: the HTTP wait detaches.
        assert await gate.acquire() is None
        assert finished == []
        assert metrics.orphaned == 0

        abandoned = asyncio.get_running_loop().create_future()
        work.track(abandoned)  # Work left running after the wait detached.
        assert metrics.orphaned == 1
        release()
        release()  # Idempotent.
        assert await gate.acquire() is None
        assert finished == []

        abandoned.set_result(None)
        await asyncio.sleep(0)
        assert finished == [True]
        assert metrics.orphaned == 0
        assert await gate.acquire() is not None

    asyncio.run(scenario())


def test_stream_bounds_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    default = service_runtime.ServiceRuntime.from_env()
    assert default.stream_write_idle_seconds == 30.0
    assert default.stream_max_events == 100_000

    monkeypatch.setenv("OPENMED_SERVICE_STREAM_WRITE_IDLE_SECONDS", "2.5")
    monkeypatch.setenv("OPENMED_SERVICE_STREAM_MAX_EVENTS", "250")
    configured = service_runtime.ServiceRuntime.from_env()

    assert configured.stream_write_idle_seconds == 2.5
    assert configured.stream_max_events == 250


@pytest.mark.parametrize("raw", ["0", "-1", "nan", "inf", "soon"])
def test_write_idle_bound_cannot_be_disabled_or_malformed(raw: str) -> None:
    with pytest.raises(ValueError, match="OPENMED_SERVICE_STREAM_WRITE_IDLE_SECONDS"):
        service_runtime.parse_stream_write_idle_seconds(raw)


@pytest.mark.parametrize("raw", ["0", "-5", "10000001", "1.5", "many"])
def test_event_cap_cannot_be_disabled_or_malformed(raw: str) -> None:
    with pytest.raises(ValueError, match="OPENMED_SERVICE_STREAM_MAX_EVENTS"):
        service_runtime.parse_stream_max_events(raw)
