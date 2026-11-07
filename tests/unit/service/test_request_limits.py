"""Global request-body size and receive-time limits run before route parsing."""

from __future__ import annotations

import asyncio
import gc
import json
import weakref

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse
from starlette.types import Message, Scope

from openmed.guard.operational_limits import OperationalLimits
from openmed.service.request_limits import (
    BODY_TIMEOUT_DEADLINE,
    BODY_TIMEOUT_MIN_RATE,
    REQUEST_BODY_GRACE_ENV_VAR,
    REQUEST_BODY_MIN_RATE_ENV_VAR,
    REQUEST_BODY_TIMEOUT,
    REQUEST_BODY_TIMEOUT_ENV_VAR,
    BoundedRequestBodyMiddleware,
    RequestBodyReadLimits,
    request_body_read_limits_from_env,
)


def _client(*, maximum: int) -> TestClient:
    app = FastAPI()

    @app.post("/parse")
    async def parse(request: Request) -> dict[str, int]:
        payload = await request.body()
        return {"byte_count": len(payload)}

    app.add_middleware(
        BoundedRequestBodyMiddleware,
        limits=OperationalLimits(max_request_bytes=maximum),
    )
    return TestClient(app, raise_server_exceptions=False)


def test_declared_oversize_is_rejected_before_receive_or_route() -> None:
    with _client(maximum=8) as client:
        response = client.post(
            "/parse",
            headers={"content-length": "9"},
            content=b"ignored",
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_size_exceeded"


def test_observed_body_limit_and_valid_body_are_deterministic() -> None:
    with _client(maximum=8) as client:
        rejected = client.post("/parse", content=b"123456789")
        accepted = client.post("/parse", content=b"12345678")

    assert rejected.status_code == 413
    assert rejected.json()["error"]["code"] == "request_size_exceeded"
    assert accepted.status_code == 200
    assert accepted.json() == {"byte_count": 8}


def test_invalid_content_length_is_content_free() -> None:
    with _client(maximum=8) as client:
        response = client.post(
            "/parse",
            headers={"content-length": "not-a-count"},
            content=b"synthetic-sensitive-canary",
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "request_size_invalid"
    assert "synthetic-sensitive" not in response.text


def test_fragmented_body_replays_in_original_order() -> None:
    chunks: list[Message] = [
        {"type": "http.request", "body": b"abc", "more_body": True},
        {"type": "http.request", "body": b"def", "more_body": False},
    ]
    observed: list[Message] = []

    async def app(scope, receive, send) -> None:
        observed.append(await receive())
        observed.append(await receive())

    async def receive() -> Message:
        return chunks.pop(0)

    async def send(message: Message) -> None:
        raise AssertionError(f"unexpected ASGI response: {message['type']}")

    scope: Scope = {"type": "http", "method": "POST", "headers": []}
    middleware = BoundedRequestBodyMiddleware(
        app, limits=OperationalLimits(max_request_bytes=6)
    )
    asyncio.run(middleware(scope, receive, send))

    assert [item["body"] for item in observed] == [b"abc", b"def"]


# --- Receive-time bounds -----------------------------------------------------

CANARY = b"synthetic-body-canary-0042"


class _TrackedMessage(dict):
    """ASGI message whose lifetime a test can observe through a weak reference."""


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _TimeoutRecorder:
    def __init__(self) -> None:
        self.reasons: list[str] = []

    def record_request_body_timeout(self, *, reason: str) -> None:
        self.reasons.append(reason)


def _chunk(body: bytes, *, more_body: bool = True) -> _TrackedMessage:
    return _TrackedMessage(type="http.request", body=body, more_body=more_body)


def _drive(
    middleware: BoundedRequestBodyMiddleware,
    receive,
    *,
    on_start=None,
    path: str = "/parse",
) -> list[Message]:
    sent: list[Message] = []

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start" and on_start is not None:
            on_start()
        sent.append(message)

    scope: Scope = {"type": "http", "method": "POST", "path": path, "headers": []}
    asyncio.run(middleware(scope, receive, send))
    return sent


def _error_code(sent: list[Message]) -> str:
    return json.loads(sent[1]["body"])["error"]["code"]


def _headers(sent: list[Message]) -> dict[bytes, bytes]:
    return dict(sent[0]["headers"])


def test_trickling_body_is_rejected_and_its_buffered_chunks_released() -> None:
    clock = _FakeClock()
    recorder = _TimeoutRecorder()
    pending = [_chunk(CANARY[index : index + 1]) for index in range(len(CANARY))]
    delivered: list[weakref.ref[_TrackedMessage]] = []
    reached_app = []

    async def app(scope, receive, send) -> None:
        reached_app.append(True)

    async def receive() -> Message:
        # One byte per second is far below the 1 KiB/s minimum rate.
        clock.now += 1.0
        message = pending.pop(0)
        delivered.append(weakref.ref(message))
        return message

    def assert_released() -> None:
        gc.collect()
        assert delivered
        assert all(ref() is None for ref in delivered)

    middleware = BoundedRequestBodyMiddleware(
        app,
        limits=OperationalLimits(max_request_bytes=1024),
        read_limits=RequestBodyReadLimits(
            timeout_seconds=60, grace_seconds=2, min_bytes_per_second=1024
        ),
        metrics=recorder,
        clock=clock,
    )
    sent = _drive(middleware, receive, on_start=assert_released)

    assert not reached_app
    assert sent[0]["status"] == 408
    assert _error_code(sent) == REQUEST_BODY_TIMEOUT
    assert _headers(sent)[b"connection"] == b"close"
    assert recorder.reasons == [BODY_TIMEOUT_MIN_RATE]
    # Reading stops as soon as the client falls behind the minimum rate.
    assert len(delivered) == 3
    rendered = b"".join(message.get("body", b"") for message in sent)
    assert b"synthetic" not in rendered


def test_stalled_body_is_rejected_within_the_deadline() -> None:
    reached_app = []
    cancelled = []
    first = [_chunk(b"{")]

    async def app(scope, receive, send) -> None:
        reached_app.append(True)

    async def receive() -> Message:
        if first:
            return first.pop()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        raise AssertionError("a stalled client never sends more data")

    recorder = _TimeoutRecorder()
    middleware = BoundedRequestBodyMiddleware(
        app,
        read_limits=RequestBodyReadLimits(
            timeout_seconds=0.05, grace_seconds=0.02, min_bytes_per_second=1024
        ),
        metrics=recorder,
    )
    sent = _drive(middleware, receive)

    assert not reached_app
    assert cancelled == [True]
    assert sent[0]["status"] == 408
    assert _error_code(sent) == REQUEST_BODY_TIMEOUT
    # The grace allowance, not the total deadline, ran out first.
    assert recorder.reasons == [BODY_TIMEOUT_MIN_RATE]


def test_total_deadline_bounds_a_fast_but_endless_upload() -> None:
    clock = _FakeClock()
    recorder = _TimeoutRecorder()

    async def app(scope, receive, send) -> None:
        raise AssertionError("the body never completed")

    async def receive() -> Message:
        clock.now += 1.0
        return _chunk(b"x" * 4096)

    middleware = BoundedRequestBodyMiddleware(
        app,
        limits=OperationalLimits(max_request_bytes=1024 * 1024),
        read_limits=RequestBodyReadLimits(
            timeout_seconds=5, grace_seconds=1, min_bytes_per_second=1024
        ),
        metrics=recorder,
        clock=clock,
    )
    sent = _drive(middleware, receive)

    assert sent[0]["status"] == 408
    assert recorder.reasons == [BODY_TIMEOUT_DEADLINE]


def test_workflow_boundary_rejects_trickling_bodies_before_its_buffer() -> None:
    from openmed.service.workflow_routes import WorkflowBoundaryMiddleware

    clock = _FakeClock()
    recorder = _TimeoutRecorder()
    reached_inner = []

    async def inner(scope, receive, send) -> None:
        reached_inner.append(True)

    async def receive() -> Message:
        clock.now += 1.0
        return _chunk(b"x")

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = WorkflowBoundaryMiddleware(
        inner,
        read_limits=RequestBodyReadLimits(
            timeout_seconds=60, grace_seconds=2, min_bytes_per_second=1024
        ),
        metrics=recorder,
        clock=clock,
    )
    scope: Scope = {"type": "http", "path": "/v1/workflows/status", "headers": []}
    asyncio.run(middleware(scope, receive, send))

    headers = _headers(sent)
    assert not reached_inner
    assert sent[0]["status"] == 408
    assert _error_code(sent) == REQUEST_BODY_TIMEOUT
    assert headers[b"connection"] == b"close"
    assert headers[b"cache-control"] == b"no-store"
    assert headers[b"x-request-id"]
    assert recorder.reasons == [BODY_TIMEOUT_MIN_RATE]


def test_steady_upload_up_to_the_size_limit_is_replayed_unchanged() -> None:
    clock = _FakeClock()
    chunk_bytes = 2048
    chunks: list[Message] = [
        {
            "type": "http.request",
            "body": bytes([65 + index]) * chunk_bytes,
            "more_body": index < 7,
        }
        for index in range(8)
    ]
    expected = b"".join(chunk["body"] for chunk in chunks)
    observed: list[bytes] = []

    async def app(scope, receive, send) -> None:
        while True:
            message = await receive()
            observed.append(message["body"])
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive() -> Message:
        # Two KiB per second keeps ahead of the 1 KiB/s minimum rate.
        clock.now += 1.0
        return chunks.pop(0)

    recorder = _TimeoutRecorder()
    middleware = BoundedRequestBodyMiddleware(
        app,
        limits=OperationalLimits(max_request_bytes=len(expected)),
        read_limits=RequestBodyReadLimits(
            timeout_seconds=60, grace_seconds=1, min_bytes_per_second=1024
        ),
        metrics=recorder,
        clock=clock,
    )
    sent = _drive(middleware, receive)

    assert sent[0]["status"] == 200
    assert b"".join(observed) == expected
    assert recorder.reasons == []


def test_zero_minimum_rate_leaves_only_the_total_deadline() -> None:
    clock = _FakeClock()
    chunks = [_chunk(b"x") for _ in range(4)] + [_chunk(b"y", more_body=False)]

    async def app(scope, receive, send) -> None:
        while (await receive()).get("more_body", False):
            pass
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive() -> Message:
        clock.now += 1.0
        return chunks.pop(0)

    middleware = BoundedRequestBodyMiddleware(
        app,
        read_limits=RequestBodyReadLimits(
            timeout_seconds=30, grace_seconds=1, min_bytes_per_second=0
        ),
        clock=clock,
    )

    assert _drive(middleware, receive)[0]["status"] == 204


def test_path_scoped_instance_uses_its_own_budget_and_envelope() -> None:
    passthrough = []

    async def app(scope, receive, send) -> None:
        passthrough.append(scope["path"])
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    def responder(code: str) -> JSONResponse:
        return JSONResponse(status_code=499, content={"scoped": code})

    middleware = BoundedRequestBodyMiddleware(
        app,
        max_bytes=2,
        limited_paths=["/limited"],
        rejection_response=responder,
    )

    async def oversized() -> Message:
        return {"type": "http.request", "body": b"abc", "more_body": False}

    rejected = _drive(middleware, oversized, path="/limited")
    unrelated = _drive(middleware, oversized, path="/other")

    assert rejected[0]["status"] == 499
    assert json.loads(rejected[1]["body"]) == {"scoped": "request_size_exceeded"}
    assert unrelated[0]["status"] == 204
    assert passthrough == ["/other"]


def test_invalid_byte_budget_is_rejected() -> None:
    async def app(scope, receive, send) -> None:
        return None

    with pytest.raises(ValueError, match="positive"):
        BoundedRequestBodyMiddleware(app, max_bytes=0)


def test_read_limits_load_from_environment_without_echoing_values() -> None:
    assert request_body_read_limits_from_env({}) == RequestBodyReadLimits()
    assert request_body_read_limits_from_env(
        {
            REQUEST_BODY_TIMEOUT_ENV_VAR: "120",
            REQUEST_BODY_GRACE_ENV_VAR: "2.5",
            REQUEST_BODY_MIN_RATE_ENV_VAR: "0",
        }
    ) == RequestBodyReadLimits(
        timeout_seconds=120, grace_seconds=2.5, min_bytes_per_second=0
    )

    for name, raw in (
        (REQUEST_BODY_TIMEOUT_ENV_VAR, "synthetic-canary"),
        (REQUEST_BODY_TIMEOUT_ENV_VAR, "0"),
        (REQUEST_BODY_GRACE_ENV_VAR, "inf"),
        (REQUEST_BODY_MIN_RATE_ENV_VAR, "synthetic-canary"),
        (REQUEST_BODY_MIN_RATE_ENV_VAR, "-1"),
        (REQUEST_BODY_MIN_RATE_ENV_VAR, "1.5"),
    ):
        with pytest.raises(ValueError) as caught:
            request_body_read_limits_from_env({name: raw})
        assert str(caught.value).startswith(f"{name} must be ")
        assert "synthetic" not in str(caught.value)


def test_read_limits_reject_values_outside_the_safe_range() -> None:
    for kwargs in (
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
        {"grace_seconds": 3601},
        {"min_bytes_per_second": -1},
        {"min_bytes_per_second": True},
    ):
        with pytest.raises(ValueError, match="safe range"):
            RequestBodyReadLimits(**kwargs)


def _service_scope(path: str) -> Scope:
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [
            (b"host", b"127.0.0.1"),
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8080),
        "scheme": "http",
        "http_version": "1.1",
        "root_path": "",
    }


@pytest.mark.parametrize("path", ["/analyze", "/graphql", "/v1/workflows/status"])
def test_service_app_rejects_stalled_bodies_and_counts_them(monkeypatch, path) -> None:
    from openmed.service.app import create_app
    from openmed.service.metrics import REQUEST_BODY_TIMEOUT_NAME

    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    monkeypatch.setenv(REQUEST_BODY_TIMEOUT_ENV_VAR, "0.05")
    monkeypatch.setenv(REQUEST_BODY_GRACE_ENV_VAR, "0.02")
    app = create_app()
    first = [{"type": "http.request", "body": CANARY, "more_body": True}]

    async def receive() -> Message:
        if first:
            return first.pop()
        await asyncio.Event().wait()
        raise AssertionError("a stalled client never sends more data")

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    asyncio.run(app(_service_scope(path), receive, send))

    assert sent[0]["status"] == 408
    assert _error_code(sent) == REQUEST_BODY_TIMEOUT
    assert _headers(sent)[b"connection"] == b"close"
    rendered = b"".join(message.get("body", b"") for message in sent)
    assert b"synthetic" not in rendered
    assert b"127.0.0.1" not in rendered

    rendered_metrics = app.state.metrics.render()
    timeouts = [
        line
        for line in rendered_metrics.splitlines()
        if line.startswith(REQUEST_BODY_TIMEOUT_NAME + "{")
    ]
    assert sum(int(line.rsplit(" ", 1)[1]) for line in timeouts) == 1
    assert "synthetic" not in rendered_metrics
    assert "127.0.0.1" not in rendered_metrics
