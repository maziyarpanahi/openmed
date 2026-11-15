"""Resource budget, deadline and permit-release tests for the gRPC service.

Servers run in-process on loopback with model work replaced by spies. Budgets
use a manually advanced clock wherever possible, so outcomes never depend on
how fast the machine is.
"""

from __future__ import annotations

import gc
import threading
import time
from collections.abc import Iterator
from typing import Any, Callable, Optional

import grpc
import pytest

from openmed.core.streaming import StreamingDeidentificationEvent
from openmed.service import grpc_policy, grpc_server
from openmed.service import runtime as service_runtime
from openmed.service.grpc_policy import (
    DEFAULT_GRPC_STREAM_BUFFER_CHARS,
    REASON_CONCURRENCY_LIMITED,
    REASON_DEADLINE_EXCEEDED,
    REASON_INVALID_REQUEST,
    REASON_RATE_LIMITED,
    REASON_STREAM_BUFFER_TOO_LARGE,
    REASON_STREAM_CHUNK_TOO_LARGE,
    REASON_STREAM_TOO_MANY_CHUNKS,
    REASON_TEXT_TOO_LONG,
    GrpcLimits,
    GrpcPolicyError,
    GrpcSecurityInterceptor,
    GrpcServicePolicy,
)
from openmed.service.proto.generated import openmed_pb2
from openmed.service.runtime import ServiceThrottleConfig
from openmed.service.throttle import BlockingConcurrencyGate, TokenBucketRateLimiter
from tests.unit.service.test_grpc_auth import (
    RPCS,
    SYNTHETIC_TEXT,
    FakeClock,
    FakeContext,
    ModelSpy,
    Served,
    call_rpc,
    failure_of,
    make_runtime,
    outcome,
    reason_of,
    serving,
)

WAIT_SECONDS = 10.0


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> service_runtime.ServiceRuntime:
    return make_runtime(monkeypatch)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> ModelSpy:
    return ModelSpy().install(monkeypatch)


def limited(**limits: Any) -> GrpcServicePolicy:
    return GrpcServicePolicy(limits=GrpcLimits(**limits))


def throttled(**throttle: Any) -> GrpcServicePolicy:
    return GrpcServicePolicy(throttle=ServiceThrottleConfig(**throttle))


def eventually(predicate: Callable[[], bool]) -> bool:
    """Wait for background RPC cleanup; never asserts on elapsed time."""
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def balanced(served: Served) -> bool:
    counters = served.interceptor.concurrency_gate.counters()
    return counters["in_flight"] == 0 and counters["acquired"] == counters["released"]


class ModelSpans:
    """Record where runtime model permits are taken and returned."""

    def __init__(self, runtime: service_runtime.ServiceRuntime, monkeypatch) -> None:
        self.begins: list[int] = []
        self.finishes: list[int] = []
        begin = runtime.begin_model_request
        finish = runtime.finish_model_request

        def recording_begin(model_name: str) -> str:
            self.begins.append(threading.get_ident())
            return begin(model_name)

        def recording_finish(model_key: str, keep_alive: Any, error=None) -> None:
            self.finishes.append(threading.get_ident())
            finish(model_key, keep_alive, error)

        monkeypatch.setattr(runtime, "begin_model_request", recording_begin)
        monkeypatch.setattr(runtime, "finish_model_request", recording_finish)


# --- Limits object ----------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_receive_message_bytes": 0},
        {"max_send_message_bytes": 2**31},
        {"max_text_chars": True},
        {"max_stream_chunks": -1},
        {"max_stream_chunk_bytes": 1.5},
        {"default_stream_buffer_chars": 10, "max_stream_buffer_chars": 5},
        {"max_rpc_seconds": 0},
        {"max_rpc_seconds": float("inf")},
        {"max_rpc_seconds": True},
    ],
)
def test_limits_reject_unsafe_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        GrpcLimits(**overrides)


def test_limits_from_env_share_rest_budgets(
    monkeypatch: pytest.MonkeyPatch, runtime: service_runtime.ServiceRuntime
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_MAX_REQUEST_BYTES", "4096")
    monkeypatch.setenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", "123")

    limits = GrpcLimits.from_env(runtime=runtime)

    assert limits.max_receive_message_bytes == 4096
    assert limits.max_text_chars == 123
    assert limits.max_rpc_seconds == float(runtime.config.timeout)
    assert GrpcLimits.from_env().max_rpc_seconds == (
        grpc_policy.DEFAULT_GRPC_MAX_RPC_SECONDS
    )
    assert dict(limits.server_options()) == {
        "grpc.max_receive_message_length": 4096,
        "grpc.max_send_message_length": limits.max_send_message_bytes,
    }


def test_create_grpc_server_applies_message_size_options(
    monkeypatch: pytest.MonkeyPatch, runtime: service_runtime.ServiceRuntime
) -> None:
    captured: dict[str, Any] = {}
    real_server = grpc.server

    def recording_server(*args: Any, **kwargs: Any) -> grpc.Server:
        captured.update(kwargs)
        return real_server(*args, **kwargs)

    monkeypatch.setattr(grpc_server.grpc, "server", recording_server)
    policy = limited(max_receive_message_bytes=2048, max_send_message_bytes=4096)
    server = grpc_server.create_grpc_server(runtime=runtime, policy=policy)
    server.stop(None)

    assert dict(captured["options"]) == {
        "grpc.max_receive_message_length": 2048,
        "grpc.max_send_message_length": 4096,
    }
    assert isinstance(captured["interceptors"][0], GrpcSecurityInterceptor)


def test_stream_chunks_split_unset_chunk_size_at_a_byte_safe_size() -> None:
    limits = GrpcLimits(max_stream_chunk_bytes=16)
    request = openmed_pb2.DeidentifyStreamRequest(
        request=openmed_pb2.DeidentifyRequest(text="abcdefghij")
    )

    assert limits.default_stream_chunk_chars == 4
    assert limits.stream_chunks(request) == ("abcd", "efgh", "ij")
    short = openmed_pb2.DeidentifyStreamRequest(
        request=openmed_pb2.DeidentifyRequest(text="abc")
    )
    assert limits.stream_chunks(short) == ("abc",)


# --- Message and text budgets over the wire ---------------------------------


@pytest.mark.parametrize("rpc", RPCS)
def test_oversized_messages_fail_at_the_transport(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    rpc: str,
) -> None:
    policy = GrpcServicePolicy(
        limits=GrpcLimits(max_receive_message_bytes=1024),
        throttle=ServiceThrottleConfig(max_concurrency=1),
    )
    with serving(monkeypatch, runtime, policy) as served:
        code, _ = outcome(served.stub(), rpc, text="s" * 4096)
        counters = served.interceptor.concurrency_gate.counters()

    assert code == grpc.StatusCode.RESOURCE_EXHAUSTED
    assert spy.calls == []
    assert counters == {"acquired": 0, "released": 0, "in_flight": 0}


def test_oversized_responses_fail_at_the_transport(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
) -> None:
    def large(payload: Any, runtime: Any) -> dict[str, Any]:
        return {"deidentified_text": "r" * 4096, "method": "mask", "timestamp": "t"}

    monkeypatch.setattr(grpc_server, "_pii_deidentify_payload", large)
    policy = GrpcServicePolicy(
        limits=GrpcLimits(max_send_message_bytes=1024),
        throttle=ServiceThrottleConfig(max_concurrency=1),
    )
    with serving(monkeypatch, runtime, policy) as served:
        code, _ = outcome(served.stub(), "Deidentify")
        assert eventually(lambda: balanced(served))

    assert code == grpc.StatusCode.RESOURCE_EXHAUSTED


@pytest.mark.parametrize("rpc", RPCS)
def test_text_budget_applies_to_every_rpc(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    rpc: str,
) -> None:
    with serving(monkeypatch, runtime, limited(max_text_chars=32)) as served:
        stub = served.stub()
        at_limit = outcome(stub, rpc, text="s" * 32)
        over_limit = outcome(stub, rpc, text="s" * 33)

    assert at_limit == (grpc.StatusCode.OK, None)
    assert over_limit == failure_of(REASON_TEXT_TOO_LONG)
    assert spy.calls == [rpc]


STREAM_BUDGETS = {
    "too_many_chunks": (
        {"max_stream_chunks": 2},
        {"chunks": ["a", "b", "c"]},
        REASON_STREAM_TOO_MANY_CHUNKS,
    ),
    "too_many_split_chunks": (
        {"max_stream_chunks": 4},
        {"chunk_size": 1},
        REASON_STREAM_TOO_MANY_CHUNKS,
    ),
    "chunk_characters": (
        {"max_stream_chunk_bytes": 8},
        {"chunks": ["x" * 9]},
        REASON_STREAM_CHUNK_TOO_LARGE,
    ),
    "chunk_utf8_bytes": (
        {"max_stream_chunk_bytes": 8},
        {"chunks": ["é" * 5]},
        REASON_STREAM_CHUNK_TOO_LARGE,
    ),
    "oversized_chunk_size": (
        {"max_stream_chunk_bytes": 8},
        {"chunk_size": 1024},
        REASON_STREAM_CHUNK_TOO_LARGE,
    ),
    "aggregate_chunks": (
        {"max_text_chars": 8},
        {"chunks": ["abcd", "efgh", "i"]},
        REASON_TEXT_TOO_LONG,
    ),
    "buffer_above_cap": (
        {"max_stream_buffer_chars": 8192},
        {"max_buffer": 8193},
        REASON_STREAM_BUFFER_TOO_LARGE,
    ),
    "negative_buffer": ({}, {"max_buffer": -1}, REASON_INVALID_REQUEST),
    "negative_chunk_size": ({}, {"chunk_size": -1}, REASON_INVALID_REQUEST),
}


@pytest.mark.parametrize("case", sorted(STREAM_BUDGETS))
def test_stream_budgets_fail_before_buffering(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    case: str,
) -> None:
    limits, fields, reason = STREAM_BUDGETS[case]
    spans = ModelSpans(runtime, monkeypatch)
    with serving(monkeypatch, runtime, limited(**limits)) as served:
        result = outcome(served.stub(), "StreamDeidentify", text="0123456789", **fields)

    assert result == failure_of(reason)
    assert spy.calls == []
    assert spans.begins == []


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(0, DEFAULT_GRPC_STREAM_BUFFER_CHARS), (128, 128), (8192, 8192)],
)
def test_stream_buffer_is_server_capped(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    requested: int,
    expected: int,
) -> None:
    with serving(monkeypatch, runtime, limited(max_stream_buffer_chars=8192)) as served:
        result = outcome(served.stub(), "StreamDeidentify", max_buffer=requested)

    assert result == (grpc.StatusCode.OK, None)
    assert spy.stream_kwargs[0]["max_buffer"] == expected


def test_stream_receives_bounded_explicit_chunks(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
) -> None:
    with serving(monkeypatch, runtime, limited(max_stream_chunks=3)) as served:
        result = outcome(served.stub(), "StreamDeidentify", chunks=["ab", "cd", "e"])

    assert result == (grpc.StatusCode.OK, None)
    assert spy.stream_chunks == [("ab", "cd", "e")]


# --- Rate and concurrency ---------------------------------------------------


def test_token_bucket_try_consume_is_deterministic() -> None:
    clock = FakeClock()
    limiter = TokenBucketRateLimiter(requests_per_second=2.0, burst_size=2, clock=clock)

    decisions = [limiter.try_consume("peer-a") for _ in range(3)]
    other_key = limiter.try_consume("peer-b")
    clock.advance(0.5)
    refilled = limiter.try_consume("peer-a")

    assert [decision.allowed for decision in decisions] == [True, True, False]
    assert decisions[2].retry_after_seconds == pytest.approx(0.5)
    assert other_key.allowed is True
    assert refilled.allowed is True


def test_blocking_gate_releases_each_permit_once() -> None:
    gate = BlockingConcurrencyGate(max_concurrency=1, wait_seconds=0.0)

    first = gate.acquire("a")
    blocked = gate.acquire("a")
    other_key = gate.acquire("b")
    assert first is not None and other_key is not None
    assert blocked is None
    assert first.release() is True
    assert first.release() is False
    second = gate.acquire("a")
    assert second is not None
    second.release()
    other_key.release()

    assert gate.counters() == {"acquired": 3, "released": 3, "in_flight": 0}
    assert BlockingConcurrencyGate(max_concurrency=0, wait_seconds=0).acquire() is None


def test_blocking_gate_waits_for_a_released_slot() -> None:
    gate = BlockingConcurrencyGate(max_concurrency=1, wait_seconds=WAIT_SECONDS)
    held = gate.acquire()
    assert held is not None
    acquired: list[Any] = []
    waiter = threading.Thread(target=lambda: acquired.append(gate.acquire()))
    waiter.start()
    assert eventually(lambda: waiter.is_alive())
    held.release()
    waiter.join(WAIT_SECONDS)

    assert acquired and acquired[0] is not None
    acquired[0].release()
    assert gate.in_flight() == 0
    assert gate.acquire(timeout=0) is not None


@pytest.mark.parametrize("rpc", RPCS)
def test_rate_limit_is_deterministic_for_every_rpc(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    rpc: str,
) -> None:
    clock = FakeClock()
    policy = throttled(rate_limit_rps=1.0, rate_limit_burst=2, key_by="peer")
    with serving(monkeypatch, runtime, policy, clock=clock) as served:
        stub = served.stub()
        results = [outcome(stub, rpc) for _ in range(3)]
        clock.advance(1.0)
        recovered = outcome(stub, rpc)

    assert results == [
        (grpc.StatusCode.OK, None),
        (grpc.StatusCode.OK, None),
        failure_of(REASON_RATE_LIMITED),
    ]
    assert recovered == (grpc.StatusCode.OK, None)
    assert spy.calls == [rpc, rpc, rpc]


class BlockingDeidentify:
    """Unary model double that holds its worker until the test releases it."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.calls = 0

    def __call__(self, payload: Any, runtime: Any) -> dict[str, Any]:
        self.calls += 1
        self.entered.set()
        try:
            assert self.release.wait(WAIT_SECONDS)
            return {"deidentified_text": "[REDACTED]", "method": "mask"}
        finally:
            self.finished.set()


def test_concurrency_limit_rejects_then_releases(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
) -> None:
    blocking = BlockingDeidentify()
    monkeypatch.setattr(grpc_server, "_pii_deidentify_payload", blocking)
    policy = throttled(max_concurrency=1, concurrency_wait_seconds=0.0)
    first: list[Any] = []
    with serving(monkeypatch, runtime, policy) as served:
        stub = served.stub()
        worker = threading.Thread(
            target=lambda: first.append(outcome(stub, "Deidentify"))
        )
        worker.start()
        assert blocking.entered.wait(WAIT_SECONDS)
        rejected = [outcome(stub, rpc) for rpc in RPCS]
        blocking.release.set()
        worker.join(WAIT_SECONDS)
        assert eventually(lambda: balanced(served))
        after = outcome(stub, "Deidentify")
        counters = served.interceptor.concurrency_gate.counters()

    assert first == [(grpc.StatusCode.OK, None)]
    assert rejected == [failure_of(REASON_CONCURRENCY_LIMITED)] * len(RPCS)
    assert after == (grpc.StatusCode.OK, None)
    assert counters == {"acquired": 2, "released": 2, "in_flight": 0}
    assert spy.calls == []


@pytest.mark.parametrize("rpc", RPCS)
def test_permits_are_released_after_handler_errors(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    rpc: str,
) -> None:
    def failing(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("synthetic failure with SYNTH-MRN-000417")

    def failing_stream(chunks: Any, **kwargs: Any) -> Iterator[Any]:
        yield StreamingDeidentificationEvent(redacted_text="[A] ")
        raise ValueError("synthetic failure with SYNTH-MRN-000417")

    for name in ("_analyze_payload", "_pii_extract_payload", "_pii_deidentify_payload"):
        monkeypatch.setattr(grpc_server, name, failing)
    monkeypatch.setattr(grpc_server, "deidentify_stream", failing_stream)
    policy = throttled(max_concurrency=1, concurrency_wait_seconds=0.0)
    with serving(monkeypatch, runtime, policy) as served:
        stub = served.stub()
        try:
            call_rpc(stub, rpc)
        except grpc.RpcError as exc:
            failed = (exc.code(), reason_of(exc), exc.details())
        else:  # pragma: no cover - the double always fails
            raise AssertionError("the RPC must fail")
        assert eventually(lambda: balanced(served))
        too_long = outcome(stub, rpc, text="s" * (served_text_limit(served) + 1))
        assert eventually(lambda: balanced(served))
        counters = served.interceptor.concurrency_gate.counters()

    assert failed == (
        grpc.StatusCode.INVALID_ARGUMENT,
        "invalid_request",
        "Invalid request",
    )
    assert too_long == failure_of(REASON_TEXT_TOO_LONG)
    assert counters["acquired"] == 2


def served_text_limit(served: Served) -> int:
    return served.interceptor.policy.limits.max_text_chars


# --- Deadlines --------------------------------------------------------------


def test_effective_deadline_is_the_smaller_budget() -> None:
    clock = FakeClock(start=100.0)
    interceptor = GrpcSecurityInterceptor(
        GrpcServicePolicy(limits=GrpcLimits(max_rpc_seconds=60.0)), clock=clock
    )

    assert interceptor._deadline(FakeContext(time_remaining=5.0)) == 105.0
    assert interceptor._deadline(FakeContext(time_remaining=1e18)) == 160.0
    assert interceptor._deadline(FakeContext(time_remaining=float("inf"))) == 160.0
    assert interceptor._deadline(FakeContext(time_remaining=0.0)) == 100.0


@pytest.mark.parametrize("rpc", ["Analyze", "Extract", "Deidentify"])
def test_server_budget_bounds_unary_rpcs(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    rpc: str,
) -> None:
    clock = FakeClock()

    def slow(payload: Any, runtime: Any) -> dict[str, Any]:
        clock.advance(31.0)
        return {"text": "", "deidentified_text": "", "method": "mask"}

    for name in ("_analyze_payload", "_pii_extract_payload", "_pii_deidentify_payload"):
        monkeypatch.setattr(grpc_server, name, slow)
    policy = GrpcServicePolicy(
        limits=GrpcLimits(max_rpc_seconds=30.0),
        throttle=ServiceThrottleConfig(max_concurrency=1),
    )
    with serving(monkeypatch, runtime, policy, clock=clock) as served:
        result = outcome(served.stub(), rpc, timeout=60)
        assert eventually(lambda: balanced(served))

    assert result == failure_of(REASON_DEADLINE_EXCEEDED)


def test_server_budget_bounds_streams(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
) -> None:
    clock = FakeClock()
    closed = threading.Event()

    def slow_stream(chunks: Any, **kwargs: Any) -> Iterator[Any]:
        try:
            yield StreamingDeidentificationEvent(redacted_text="[A] ")
            clock.advance(31.0)
            yield StreamingDeidentificationEvent(redacted_text="[B] ")
            yield StreamingDeidentificationEvent(redacted_text="[C] ")
        finally:
            closed.set()

    monkeypatch.setattr(grpc_server, "deidentify_stream", slow_stream)
    spans = ModelSpans(runtime, monkeypatch)
    policy = GrpcServicePolicy(
        limits=GrpcLimits(max_rpc_seconds=30.0),
        throttle=ServiceThrottleConfig(max_concurrency=1),
    )
    received: list[str] = []
    with serving(monkeypatch, runtime, policy, clock=clock) as served:
        request = openmed_pb2.DeidentifyStreamRequest(
            request=openmed_pb2.DeidentifyRequest(text=SYNTHETIC_TEXT, method="mask")
        )
        with pytest.raises(grpc.RpcError) as excinfo:
            for event in served.stub().StreamDeidentify(request, timeout=60):
                received.append(event.redacted_text)
        assert closed.wait(WAIT_SECONDS)
        assert eventually(lambda: balanced(served))

    assert (excinfo.value.code(), reason_of(excinfo.value)) == failure_of(
        REASON_DEADLINE_EXCEEDED
    )
    assert received == ["[A] "]
    assert spans.begins and spans.finishes == spans.begins


def test_client_deadline_releases_unary_permits_after_work_settles(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
) -> None:
    blocking = BlockingDeidentify()
    monkeypatch.setattr(grpc_server, "_pii_deidentify_payload", blocking)
    policy = throttled(max_concurrency=1, concurrency_wait_seconds=0.0)
    with serving(monkeypatch, runtime, policy) as served:
        stub = served.stub()
        expired = outcome(stub, "Deidentify", timeout=0.5)
        assert served.interceptor.concurrency_gate.in_flight() == 1
        blocking.release.set()
        assert blocking.finished.wait(WAIT_SECONDS)
        assert eventually(lambda: balanced(served))
        after = outcome(stub, "Deidentify")

    assert expired == (grpc.StatusCode.DEADLINE_EXCEEDED, None)
    assert after == (grpc.StatusCode.OK, None)
    assert served.interceptor.concurrency_gate.counters()["released"] == 2


# --- Cancellation -----------------------------------------------------------


class BlockingStream:
    """Stream double that blocks after its first fragment."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.closed = threading.Event()
        self.fragments = 0

    def __call__(self, chunks: Any, **kwargs: Any) -> Iterator[Any]:
        try:
            self.fragments += 1
            yield StreamingDeidentificationEvent(redacted_text="[A] ")
            assert self.release.wait(WAIT_SECONDS)
            self.fragments += 1
            yield StreamingDeidentificationEvent(redacted_text="[B] ")
            yield StreamingDeidentificationEvent(redacted_text="", final=True)
        finally:
            self.closed.set()


@pytest.mark.parametrize("trigger", ["cancel", "client_deadline"])
def test_stream_cancellation_releases_permits_once(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    trigger: str,
) -> None:
    blocking = BlockingStream()
    monkeypatch.setattr(grpc_server, "deidentify_stream", blocking)
    terminated = threading.Event()
    original_terminate = grpc_policy._GuardedStream.terminate

    def recording_terminate(self: Any) -> None:
        original_terminate(self)
        terminated.set()

    monkeypatch.setattr(grpc_policy._GuardedStream, "terminate", recording_terminate)
    spans = ModelSpans(runtime, monkeypatch)
    policy = throttled(max_concurrency=1, concurrency_wait_seconds=0.0)
    with serving(monkeypatch, runtime, policy) as served:
        stub = served.stub()
        request = openmed_pb2.DeidentifyStreamRequest(
            request=openmed_pb2.DeidentifyRequest(text=SYNTHETIC_TEXT, method="mask")
        )
        call = stub.StreamDeidentify(
            request, timeout=0.5 if trigger == "client_deadline" else WAIT_SECONDS
        )
        assert next(call).redacted_text == "[A] "
        if trigger == "cancel":
            call.cancel()
        with pytest.raises(grpc.RpcError) as excinfo:
            next(call)
        assert terminated.wait(WAIT_SECONDS)
        assert served.interceptor.concurrency_gate.in_flight() == 1
        blocking.release.set()
        assert blocking.closed.wait(WAIT_SECONDS)
        assert eventually(lambda: balanced(served))
        busy_free = outcome(stub, "Deidentify")
        counters = served.interceptor.concurrency_gate.counters()

    expected_code = (
        grpc.StatusCode.CANCELLED
        if trigger == "cancel"
        else grpc.StatusCode.DEADLINE_EXCEEDED
    )
    assert excinfo.value.code() == expected_code
    assert blocking.fragments == 2
    assert busy_free == (grpc.StatusCode.OK, None)
    assert spy.calls == ["Deidentify"]
    assert counters == {"acquired": 2, "released": 2, "in_flight": 0}
    assert len(spans.finishes) == 1
    assert spans.finishes == spans.begins


class _Admission:
    def __init__(self, *, expired: bool = False) -> None:
        self.releases = 0
        self._expired = expired

    def expired(self) -> bool:
        return self._expired

    def release(self) -> None:
        self.releases += 1


class _StreamContext(FakeContext):
    def __init__(self, *, accept_callbacks: bool = True) -> None:
        super().__init__()
        self.callbacks: list[Callable[[], None]] = []
        self.trailers: Any = None
        self.aborted: Optional[tuple[Any, str]] = None
        self._accept = accept_callbacks

    def add_callback(self, callback: Callable[[], None]) -> bool:
        if self._accept:
            self.callbacks.append(callback)
        return self._accept

    def set_trailing_metadata(self, metadata: Any) -> None:
        self.trailers = tuple(metadata)

    def abort(self, code: Any, details: str) -> None:
        self.aborted = (code, details)
        raise RuntimeError("aborted")


def _fragments(closed: list[bool]) -> Iterator[str]:
    try:
        yield "a"
        yield "b"
    finally:
        closed.append(True)


def _guarded(
    *, expired: bool = False, accept_callbacks: bool = True
) -> tuple[Any, _Admission, _StreamContext, list[bool]]:
    closed: list[bool] = []
    admission = _Admission(expired=expired)
    stream = grpc_policy._GuardedStream(_fragments(closed), admission)  # type: ignore[arg-type]
    context = _StreamContext(accept_callbacks=accept_callbacks)
    stream.attach(context)  # type: ignore[arg-type]
    return stream, admission, context, closed


def test_guarded_stream_closes_unstarted_streams_on_termination() -> None:
    stream, admission, context, closed = _guarded()

    context.callbacks[0]()
    context.callbacks[0]()

    assert admission.releases == 1
    with pytest.raises(StopIteration):
        next(stream)
    assert admission.releases == 1
    rejected, rejected_admission, _, _ = _guarded(accept_callbacks=False)
    assert rejected_admission.releases == 1
    with pytest.raises(StopIteration):
        next(rejected)


def test_guarded_stream_defers_started_streams_to_their_worker() -> None:
    stream, admission, context, closed = _guarded()
    assert next(stream) == "a"

    context.callbacks[0]()
    assert admission.releases == 0
    assert closed == []

    del stream
    gc.collect()
    assert admission.releases == 1
    assert closed == [True]
    context.callbacks[0]()
    assert admission.releases == 1


def test_guarded_stream_stops_after_a_step_that_outlived_the_rpc() -> None:
    stepping = threading.Event()
    proceed = threading.Event()
    closed: list[int] = []

    def fragments() -> Iterator[str]:
        try:
            yield "a"
            stepping.set()
            assert proceed.wait(WAIT_SECONDS)
            yield "b"
        finally:
            closed.append(threading.get_ident())

    admission = _Admission()
    stream = grpc_policy._GuardedStream(fragments(), admission)  # type: ignore[arg-type]
    context = _StreamContext()
    stream.attach(context)  # type: ignore[arg-type]
    outcomes: list[Any] = []

    def worker() -> None:
        outcomes.append(next(stream))
        try:
            outcomes.append(next(stream))
        except StopIteration:
            outcomes.append("stopped")
        outcomes.append(threading.get_ident())

    thread = threading.Thread(target=worker)
    thread.start()
    assert stepping.wait(WAIT_SECONDS)
    context.callbacks[0]()
    assert admission.releases == 0
    proceed.set()
    thread.join(WAIT_SECONDS)

    assert outcomes[:2] == ["a", "stopped"]
    assert admission.releases == 1
    assert closed == [outcomes[2]]


def test_guarded_stream_releases_once_on_exhaustion() -> None:
    stream, admission, context, closed = _guarded()

    assert list(stream) == ["a", "b"]
    context.callbacks[0]()
    stream.close()

    assert admission.releases == 1
    assert closed == [True]


def test_guarded_stream_aborts_after_its_deadline() -> None:
    stream, admission, context, closed = _guarded(expired=True)

    with pytest.raises(RuntimeError, match="aborted"):
        next(stream)

    assert context.aborted == (
        grpc.StatusCode.DEADLINE_EXCEEDED,
        "Request exceeded its deadline",
    )
    assert context.trailers == (("openmed-reason", REASON_DEADLINE_EXCEEDED),)
    assert admission.releases == 1


def test_stream_chunk_errors_are_value_free() -> None:
    limits = GrpcLimits(max_stream_chunk_bytes=8)
    request = openmed_pb2.DeidentifyStreamRequest(chunks=["SYNTH-MRN-000417"])

    with pytest.raises(GrpcPolicyError) as excinfo:
        limits.stream_chunks(request)

    assert str(excinfo.value) == REASON_STREAM_CHUNK_TOO_LARGE
    assert "SYNTH" not in repr(excinfo.value)
