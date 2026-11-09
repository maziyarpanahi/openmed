"""Offline lifecycle tests for the coordinated service drain.

Every workload here is synthetic. Tests prove ordering, typed refusals, and the
absence of lost or duplicate work by observing outcomes and counts; they never
assert wall-clock durations. Small drain budgets only keep the suite fast, and
any wait on a condition has a generous safety bound.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import grpc
import httpx
import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.core import pii as pii_core
from openmed.core.pii import DeidentificationResult
from openmed.core.streaming import StreamingDeidentificationEvent
from openmed.processing.outputs import PredictionResult
from openmed.service import jobs
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.batcher import BatcherClosedError, DynamicBatcher
from openmed.service.drain import (
    SHUTTING_DOWN,
    AcceleratorDrainParticipant,
    CallbackDrainParticipant,
    DrainBudget,
    DrainCoordinator,
    DrainOutcome,
    DrainParticipant,
    DrainPhase,
    DrainRegistrationError,
    DrainState,
    JobQueueDrainParticipant,
    PhaseResult,
    ServiceShuttingDownError,
    WarmPoolDrainParticipant,
    register_drain_participant,
    shutting_down_ndjson_line,
)
from openmed.service.grpc_server import (
    GRPC_RETRY_PUSHBACK_METADATA_KEY,
    GrpcDrainParticipant,
    create_grpc_server,
)
from openmed.service.metrics import PrometheusMetricsRegistry
from openmed.service.proto.generated import openmed_pb2, openmed_pb2_grpc
from openmed.service.schemas import (
    DeidentifyJobDocument,
    DeidentifyJobRequest,
    JobWebhookRequest,
)
from openmed.service.warm_pool import WarmPool, WarmPoolClosedError
from openmed.service.webhooks import WebhookDeliveryResult

LOOPBACK_BASE_URL = "http://127.0.0.1"
_SYNTHETIC_PHI = ("Maria Garcia", "555-0100", "MRN-0042")
_SERVICE_ENV_VARS = (
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_KEEP_ALIVE",
    "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
    "OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES",
    "OPENMED_SERVICE_MAX_TEXT_LENGTH",
    "OPENMED_SERVICE_CORS_ORIGINS",
    "OPENMED_SERVICE_TRUSTED_HOSTS",
    "OPENMED_SERVICE_BATCHING_ENABLED",
    "OPENMED_SERVICE_BATCH_MAX_SIZE",
    "OPENMED_SERVICE_BATCH_MAX_WAIT_MS",
    "OPENMED_SERVICE_BATCH_MAX_QUEUE_SIZE",
    "OPENMED_SERVICE_COALESCING_ENABLED",
    "OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS",
    "OPENMED_SERVICE_SHUTDOWN_RELEASE_SECONDS",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
    "OPENMED_SERVICE_THROTTLE_KEY",
    "OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS",
    "OPENMED_SERVICE_JOBS_WORKERS",
    "OPENMED_SERVICE_METRICS_ENABLED",
)


class FakeLoader:
    """Loader double; models are opaque objects and nothing is downloaded."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        return object()

    def loaded_models(self) -> dict[str, Any]:
        return {}

    def unload_model(self, model_name: str) -> dict[str, Any]:
        return {"model_name": model_name, "models": 0, "tokenizers": 0, "pipelines": 0}

    def unload_all_models(self) -> dict[str, int]:
        return {"models": 0, "tokenizers": 0, "pipelines": 0}


@pytest.fixture(autouse=True)
def service_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for name in _SERVICE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_STORE_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)


def _deid_result(text: str) -> DeidentificationResult:
    return DeidentificationResult(
        original_text=text,
        deidentified_text="[REDACTED]",
        pii_entities=[],
        method="mask",
        timestamp=datetime(2026, 1, 1),
        metadata={},
    )


async def _until(predicate: Callable[[], bool], *, bound: float = 10.0) -> None:
    """Wait for a condition; ``bound`` is a hang guard, not a timing assertion."""
    loop = asyncio.get_running_loop()
    limit = loop.time() + bound
    while not predicate():
        if loop.time() > limit:
            raise AssertionError("condition was not reached")
        await asyncio.sleep(0.005)


def _wait_sync(predicate: Callable[[], bool], *, bound: float = 10.0) -> None:
    done = threading.Event()
    for _ in range(int(bound / 0.005)):
        if predicate():
            return
        done.wait(0.005)
    raise AssertionError("condition was not reached")


def _signature(report: Any) -> list[tuple[Any, ...]]:
    return [
        (
            event.phase.value,
            event.participant,
            event.outcome.value,
            event.completed,
            event.cancelled,
            event.unfinished,
            event.reason_code,
        )
        for event in report.events
    ]


def _assert_value_free(document: Any) -> None:
    dumped = json.dumps(document, sort_keys=True)
    for value in _SYNTHETIC_PHI:
        assert value not in dumped


# ---------------------------------------------------------------------------
# Coordinator contract
# ---------------------------------------------------------------------------


class _Recorder(DrainParticipant):
    def __init__(self, category: str, priority: int, log: list[tuple[str, str]]):
        super().__init__(category, priority=priority)
        self.log = log

    def stop_admission(self, ctx):
        self.log.append(("stop_admission", self.category))
        return PhaseResult()

    async def finish_in_flight(self, ctx):
        self.log.append(("finish_in_flight", self.category))
        return PhaseResult.from_counts(completed=1)

    def flush(self, ctx):
        self.log.append(("flush", self.category))
        return None

    def release(self, ctx):
        self.log.append(("release", self.category))
        return PhaseResult()


def test_phases_run_in_order_after_readiness_drops_and_drain_runs_once():
    log: list[tuple[str, str]] = []
    coordinator = DrainCoordinator(DrainBudget(drain_seconds=1, release_seconds=1))
    coordinator.on_readiness_lost(lambda: log.append(("readiness_lost", "")))
    coordinator.register(_Recorder("second", 20, log))
    first = coordinator.register(_Recorder("first", 10, log))
    coordinator.register(first)  # duplicate registration is a no-op
    assert coordinator.ready and coordinator.live

    async def scenario():
        one, two = await asyncio.gather(coordinator.drain(), coordinator.drain())
        return one, two, await coordinator.drain()

    one, two, three = asyncio.run(scenario())

    assert one is two is three
    assert log[0] == ("readiness_lost", "")
    assert log[1:] == [
        ("stop_admission", "first"),
        ("stop_admission", "second"),
        ("finish_in_flight", "first"),
        ("finish_in_flight", "second"),
        ("flush", "first"),
        ("flush", "second"),
        ("release", "first"),
        ("release", "second"),
    ]
    assert coordinator.state is DrainState.DRAINED
    assert not coordinator.ready and coordinator.live
    assert one.outcome is DrainOutcome.COMPLETED
    assert {event.outcome for event in one.events if event.phase.value == "flush"} == {
        DrainOutcome.SKIPPED
    }
    with pytest.raises(DrainRegistrationError):
        coordinator.register(_Recorder("late", 1, log))


def test_wedged_and_failing_participants_are_bounded_with_value_free_evidence():
    class Wedged(DrainParticipant):
        category = "wedged"

        async def finish_in_flight(self, ctx):
            await asyncio.Event().wait()  # never finishes on its own

        def outstanding(self) -> int:
            return 2

    class Broken(DrainParticipant):
        category = "broken"

        def release(self, ctx):
            raise RuntimeError(f"Patient {_SYNTHETIC_PHI[0]} {_SYNTHETIC_PHI[2]}")

    class Quick(DrainParticipant):
        category = "quick"

        async def finish_in_flight(self, ctx):
            return PhaseResult.from_counts(completed=3)

    signatures = []
    for _ in range(3):
        coordinator = DrainCoordinator(
            DrainBudget(drain_seconds=0.05, release_seconds=0.05)
        )
        for participant in (Wedged(), Broken(), Quick()):
            coordinator.register(participant)
        report = asyncio.run(coordinator.drain())
        signatures.append(_signature(report))

        wedged = report.events_for("wedged")[0]
        assert wedged.outcome is DrainOutcome.DEADLINE_EXCEEDED
        assert wedged.unfinished == 2
        assert wedged.reason_code == "drain_deadline_exceeded"
        assert wedged.budget_seconds <= report.budget.drain_seconds
        broken = report.events_for("broken")[0]
        assert broken.outcome is DrainOutcome.FAILED
        assert broken.reason_code == "participant_failed"
        assert broken.budget_seconds <= report.budget.release_seconds
        assert report.events_for("quick")[0].completed == 3
        assert report.outcome is DrainOutcome.FAILED
        assert report.unfinished_participants == ("wedged",)
        document = report.to_dict()
        _assert_value_free(document)
        assert set(document["events"][0]) == {
            "phase",
            "participant",
            "outcome",
            "completed",
            "cancelled",
            "checkpointed",
            "unfinished",
            "budget_seconds",
            "duration_seconds",
            "reason_code",
        }

    # Deadline handling is deterministic: the same work yields the same evidence.
    assert signatures[0] == signatures[1] == signatures[2]


def test_participant_labels_and_budgets_are_controlled(subtests):
    for label in ("Maria Garcia", "UPPER", "", "x" * 64, "has-dash", 7):
        with subtests.test(label=str(label)):
            with pytest.raises(ValueError):
                DrainParticipant(label)  # type: ignore[arg-type]
    for bad in (-1, float("inf"), float("nan")):
        with subtests.test(budget=str(bad)):
            with pytest.raises(ValueError):
                DrainBudget(drain_seconds=bad)
    with pytest.raises(TypeError):
        DrainCoordinator().register(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PhaseResult(reason_code="free text")
    budget = DrainBudget(drain_seconds=2, release_seconds=1)
    assert budget.settle_seconds == 0.5
    assert budget.total_seconds == 3


def test_release_window_is_configurable_and_bounded(monkeypatch):
    assert service_runtime.parse_shutdown_release_seconds(None) == 5.0
    assert service_runtime.parse_shutdown_release_seconds("0.5") == 0.5
    for raw in ("-1", "inf", "nan", "soon"):
        with pytest.raises(ValueError):
            service_runtime.parse_shutdown_release_seconds(raw)
    with pytest.raises(ValueError):
        service_runtime.parse_shutdown_drain_seconds("inf")
    monkeypatch.setenv("OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS", "3")
    monkeypatch.setenv("OPENMED_SERVICE_SHUTDOWN_RELEASE_SECONDS", "2")
    runtime = service_runtime.ServiceRuntime.from_env()
    coordinator = runtime.get_drain_coordinator()
    assert coordinator is runtime.get_drain_coordinator()
    assert coordinator.budget.drain_seconds == 3
    assert coordinator.budget.release_seconds == 2
    categories = {item.category for item in coordinator.participants()}
    assert {"warm_pool", "accelerator"} <= categories


def test_drain_metrics_use_controlled_labels_only():
    registry = PrometheusMetricsRegistry()
    coordinator = DrainCoordinator(
        DrainBudget(drain_seconds=0.05, release_seconds=0.05), metrics=registry
    )
    coordinator.register(
        CallbackDrainParticipant(
            "quick",
            finish_in_flight=lambda ctx: PhaseResult.from_counts(completed=1),
        )
    )
    asyncio.run(coordinator.drain())
    registry.record_drain_event(
        SimpleNamespace(
            phase="release", participant=_SYNTHETIC_PHI[0], outcome="completed"
        )
    )
    rendered = registry.render()

    assert "openmed_service_draining 0" in rendered
    assert (
        'openmed_service_drain_participant_total{phase="finish_in_flight",'
        'participant="quick",outcome="completed"} 1'
    ) in rendered
    assert 'openmed_service_drain_unfinished{participant="quick"} 0' in rendered
    assert _SYNTHETIC_PHI[0] not in rendered


# ---------------------------------------------------------------------------
# REST and GraphQL
# ---------------------------------------------------------------------------


def test_rest_graphql_and_jobs_refuse_new_work_while_accepted_request_commits_once(
    monkeypatch,
):
    monkeypatch.setenv("OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS", "10")
    entered, release = threading.Event(), threading.Event()
    calls: list[int] = []

    def fake_deidentify(text: str, **_: Any) -> DeidentificationResult:
        calls.append(len(text))
        entered.set()
        release.wait(timeout=10)
        return _deid_result(text)

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)
    app = create_app()

    async def scenario():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url=LOOPBACK_BASE_URL
        ) as client:
            lifespan = app.router.lifespan_context(app)
            await lifespan.__aenter__()
            accepted = asyncio.create_task(
                client.post(
                    "/pii/deidentify",
                    json={"text": f"Patient {_SYNTHETIC_PHI[0]}", "method": "mask"},
                )
            )
            await _until(entered.is_set)
            drain = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await _until(lambda: bool(getattr(app.state, "shutting_down", False)))
            probes = {
                "livez": await client.get("/livez"),
                "readyz": await client.get("/readyz"),
                "rest": await client.post("/analyze", json={"text": "synthetic"}),
                "graphql": await client.post(
                    "/graphql", json={"query": "{ entityTypes { label } }"}
                ),
                "jobs": await client.post(
                    "/jobs", json={"documents": [{"text": "synthetic"}]}
                ),
            }
            still_running = not accepted.done()
            release.set()
            response = await accepted
            await drain
            return probes, still_running, response

    probes, still_running, response = asyncio.run(scenario())

    assert probes["livez"].status_code == 200
    for name in ("readyz", "rest", "graphql", "jobs"):
        refused = probes[name]
        assert refused.status_code == 503, name
        error = refused.json()["error"]
        assert error["code"] == "not_ready"
        assert error["details"]["reason"] == SHUTTING_DOWN
        assert error["details"]["retryable"] is True
        assert refused.headers["Retry-After"] == "1"
    assert still_running
    assert response.status_code == 200
    assert response.json()["deidentified_text"] == "[REDACTED]"
    assert len(calls) == 1  # committed once, never retried by the drain

    report = app.state.drain_report
    http = report.events_for("http_requests")
    assert http[-1].phase is DrainPhase.FINISH_IN_FLIGHT
    assert (http[-1].completed, http[-1].unfinished) == (1, 0)
    assert report.outcome is DrainOutcome.COMPLETED
    assert app.state.inflight == 0
    pool = app.state.runtime.get_loader()
    assert pool.closed and pool.active_request_count() == 0
    _assert_value_free(report.to_dict())


def test_batched_requests_queued_at_drain_dispatch_exactly_once(monkeypatch):
    monkeypatch.setenv("OPENMED_SERVICE_BATCHING_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_SIZE", "2")
    monkeypatch.setenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", "1")
    monkeypatch.setenv("OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS", "10")
    entered, release = threading.Event(), threading.Event()
    dispatched: list[str] = []
    lock = threading.Lock()

    def fake_extract_batch(texts, **_: Any):
        batch = list(texts)
        with lock:
            dispatched.extend(batch)
        entered.set()
        release.wait(timeout=10)
        return [
            PredictionResult(
                text=text,
                entities=[],
                model_name="synthetic",
                timestamp=datetime(2026, 1, 1).isoformat(),
            )
            for text in batch
        ]

    monkeypatch.setattr(pii_core, "_extract_pii_batch", fake_extract_batch)
    app = create_app()
    texts = ["alpha", "bravo", "charlie", "delta"]

    async def scenario():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url=LOOPBACK_BASE_URL
        ) as client:
            lifespan = app.router.lifespan_context(app)
            await lifespan.__aenter__()
            requests = [
                asyncio.create_task(client.post("/pii/extract", json={"text": text}))
                for text in texts
            ]
            await _until(entered.is_set)
            # Every request is admitted before the drain starts.
            await _until(lambda: app.state.inflight == len(texts))
            drain = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await _until(lambda: bool(getattr(app.state, "shutting_down", False)))
            release.set()
            responses = await asyncio.gather(*requests)
            await drain
            return responses

    responses = asyncio.run(scenario())

    assert [response.status_code for response in responses] == [200] * len(texts)
    assert [response.json()["text"] for response in responses] == texts
    assert sorted(dispatched) == sorted(texts)  # no loss, no duplicate dispatch
    batcher = app.state.pii_extract_batcher
    assert batcher.outstanding_count() == 0
    with pytest.raises(BatcherClosedError):
        asyncio.run(batcher.submit(object()))


def test_batcher_drain_fails_leftovers_with_a_typed_retryable_result():
    async def scenario():
        gate = asyncio.Event()
        dispatched: list[int] = []

        async def dispatch(items):
            dispatched.extend(items)
            await gate.wait()
            return [item * 10 for item in items]

        within = DynamicBatcher(
            dispatch,
            max_batch_size=2,
            max_wait_ms=60_000,
            max_concurrent_batches=1,
            max_queue_wait_ms=60_000,
        )
        tasks = [asyncio.create_task(within.submit(index)) for index in range(5)]
        # One full batch runs (concurrency 1) while three jobs stay queued.
        await _until(lambda: dispatched == [0, 1] and within.outstanding_count() == 4)
        draining = asyncio.create_task(within.drain(timeout=10))
        await _until(lambda: within._draining)
        with pytest.raises(BatcherClosedError):
            await within.submit(99)
        gate.set()
        stats = await draining
        results = await asyncio.gather(*tasks)
        snapshot = await within.admission_snapshot()

        gate.clear()
        dispatched.clear()
        bounded = DynamicBatcher(
            dispatch,
            max_batch_size=2,
            max_wait_ms=60_000,
            max_concurrent_batches=1,
            max_queue_wait_ms=60_000,
        )
        late = [asyncio.create_task(bounded.submit(index)) for index in range(5)]
        await _until(lambda: dispatched == [0, 1] and bounded.outstanding_count() == 4)
        bounded_stats = await bounded.drain(timeout=0.05)
        gate.set()
        settled = await asyncio.gather(*late, return_exceptions=True)
        return stats, results, snapshot, bounded_stats, settled, list(dispatched)

    stats, results, snapshot, bounded_stats, settled, late_dispatch = asyncio.run(
        scenario()
    )

    assert results == [0, 10, 20, 30, 40]
    assert stats == {"dispatched": 3, "failed_queued": 0, "running": 0}
    assert snapshot.depth == 0
    assert bounded_stats == {"dispatched": 0, "failed_queued": 3, "running": 1}
    assert settled[:2] == [0, 10]
    for outcome in settled[2:]:
        assert isinstance(outcome, BatcherClosedError)
        assert isinstance(outcome, ServiceShuttingDownError)
    assert late_dispatch == [0, 1]  # leftovers were never dispatched late


# ---------------------------------------------------------------------------
# NDJSON streams
# ---------------------------------------------------------------------------


def _stalling_stream(stalled: threading.Event, release: threading.Event):
    def stream(*_: Any, **__: Any):
        yield StreamingDeidentificationEvent(redacted_text="Patient [NAME] ")
        stalled.set()
        release.wait(timeout=10)
        # Stand-in for an unflushed buffer: it must never reach the client.
        yield StreamingDeidentificationEvent(redacted_text=f"TAIL {_SYNTHETIC_PHI[0]}")
        yield StreamingDeidentificationEvent(
            redacted_text="",
            final=True,
            audit_record={"span_count": 0, "spans": []},
        )

    return stream


def _run_stream_drain(app, stalled, release, *, release_before_settle: bool):
    async def scenario():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url=LOOPBACK_BASE_URL
        ) as client:
            lifespan = app.router.lifespan_context(app)
            await lifespan.__aenter__()
            streaming = asyncio.create_task(
                client.post(
                    "/pii/deidentify/stream",
                    json={"text": f"Patient {_SYNTHETIC_PHI[0]}", "chunk_size": 4},
                )
            )
            await _until(stalled.is_set)
            drain = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await _until(lambda: bool(getattr(app.state, "shutting_down", False)))
            if release_before_settle:
                release.set()
            response = await streaming
            release.set()
            await drain
            streams = app.state.drain_streams
            pool = app.state.runtime.get_loader()
            await _until(lambda: streams.active == 0)
            await _until(lambda: pool.active_request_count() == 0)
            return response

    return asyncio.run(scenario())


def test_running_stream_ends_with_typed_terminal_event_and_no_unsafe_tail(
    monkeypatch,
):
    monkeypatch.setenv("OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS", "1")
    stalled, release = threading.Event(), threading.Event()
    monkeypatch.setattr(
        "openmed.service.streaming.deidentify_stream",
        _stalling_stream(stalled, release),
    )
    app = create_app()

    response = _run_stream_drain(app, stalled, release, release_before_settle=False)

    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.splitlines()]
    assert lines == [
        {"type": "chunk", "index": 0, "redacted_text": "Patient [NAME] "},
        json.loads(shutting_down_ndjson_line(1.0)),
    ]
    assert lines[-1]["error"]["code"] == SHUTTING_DOWN
    assert "TAIL" not in response.text
    _assert_value_free(response.text)
    streams = app.state.drain_report.events_for("ndjson_streams")[0]
    assert streams.cancelled == 1
    assert app.state.runtime.get_loader().active_request_count() == 0


def test_stream_finishing_inside_the_budget_keeps_its_final_event(monkeypatch):
    monkeypatch.setenv("OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS", "10")
    stalled, release = threading.Event(), threading.Event()
    monkeypatch.setattr(
        "openmed.service.streaming.deidentify_stream",
        _stalling_stream(stalled, release),
    )
    app = create_app()

    response = _run_stream_drain(app, stalled, release, release_before_settle=True)

    types = [json.loads(line)["type"] for line in response.text.splitlines()]
    assert types == ["chunk", "chunk", "final"]
    streams = app.state.drain_report.events_for("ndjson_streams")[0]
    assert (streams.completed, streams.cancelled, streams.unfinished) == (1, 0, 0)


def test_stream_opened_at_the_drain_boundary_returns_only_the_terminal_event():
    app = create_app()
    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        streams = app.state.drain_streams
        streams.terminate()
        response = client.post("/pii/extract/stream", json={"text": "synthetic note"})
        assert response.text == shutting_down_ndjson_line(1.0)
        _wait_sync(lambda: streams.active == 0)


# ---------------------------------------------------------------------------
# gRPC
# ---------------------------------------------------------------------------


def _grpc_runtime(monkeypatch, drain_seconds: str):
    monkeypatch.setenv("OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS", drain_seconds)
    monkeypatch.setenv("OPENMED_SERVICE_SHUTDOWN_RELEASE_SECONDS", "2")
    runtime = service_runtime.ServiceRuntime.from_env()
    server = create_grpc_server(runtime=runtime, max_workers=4)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    participant = next(
        item
        for item in runtime.get_drain_coordinator().participants()
        if isinstance(item, GrpcDrainParticipant)
    )
    return (
        runtime,
        server,
        channel,
        openmed_pb2_grpc.OpenMedServiceStub(channel),
        (participant),
    )


def test_grpc_refuses_new_calls_with_retry_hint_and_finishes_in_flight_unary(
    monkeypatch,
):
    entered, release = threading.Event(), threading.Event()
    calls: list[int] = []

    def fake_deidentify(text: str, **_: Any) -> DeidentificationResult:
        calls.append(len(text))
        entered.set()
        release.wait(timeout=10)
        return _deid_result(text)

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)
    runtime, server, channel, stub, participant = _grpc_runtime(monkeypatch, "10")
    coordinator = runtime.get_drain_coordinator()

    async def scenario():
        accepted = stub.Deidentify.future(
            openmed_pb2.DeidentifyRequest(text="synthetic note", method="mask")
        )
        await _until(entered.is_set)
        drain = asyncio.create_task(coordinator.drain())
        await _until(lambda: participant.admission.closed)
        with pytest.raises(grpc.RpcError) as refused:
            await asyncio.to_thread(
                stub.Analyze, openmed_pb2.AnalyzeRequest(text="synthetic")
            )
        release.set()
        response = await asyncio.to_thread(accepted.result, 10)
        return refused.value, response, await drain

    try:
        refused, response, report = asyncio.run(scenario())
    finally:
        release.set()
        channel.close()
        server.stop(0).wait()

    assert refused.code() is grpc.StatusCode.UNAVAILABLE
    assert refused.details() == SHUTTING_DOWN
    metadata = dict(refused.trailing_metadata())
    assert metadata[GRPC_RETRY_PUSHBACK_METADATA_KEY] == "1000"
    assert metadata["openmed-retryable"] == "true"
    assert response.deidentified_text == "[REDACTED]"
    assert len(calls) == 1
    events = {event.phase: event for event in report.events_for("grpc")}
    assert events[DrainPhase.FINISH_IN_FLIGHT].completed == 1
    assert events[DrainPhase.FINISH_IN_FLIGHT].unfinished == 0
    assert events[DrainPhase.FLUSH].completed == 1  # server stopped in budget
    assert participant.admission.active == 0


def test_grpc_stream_ends_with_unavailable_and_never_flushes_its_tail(monkeypatch):
    stalled, release = threading.Event(), threading.Event()

    def stream(chunks, **_: Any):
        del chunks
        yield StreamingDeidentificationEvent(redacted_text="Patient [NAME] ")
        stalled.set()
        release.wait(timeout=10)
        yield StreamingDeidentificationEvent(redacted_text=f"TAIL {_SYNTHETIC_PHI[0]}")
        yield StreamingDeidentificationEvent(redacted_text="", final=True)

    monkeypatch.setattr("openmed.service.grpc_server.deidentify_stream", stream)
    runtime, server, channel, stub, participant = _grpc_runtime(monkeypatch, "0.4")
    coordinator = runtime.get_drain_coordinator()
    received: list[Any] = []

    def consume() -> grpc.RpcError:
        try:
            for item in stub.StreamDeidentify(
                openmed_pb2.DeidentifyStreamRequest(
                    request=openmed_pb2.DeidentifyRequest(method="mask"),
                    chunks=["Patient ", "note"],
                )
            ):
                received.append(item)
        except grpc.RpcError as exc:
            return exc
        raise AssertionError("stream ended without a terminal status")

    async def scenario():
        consumer = asyncio.create_task(asyncio.to_thread(consume))
        await _until(stalled.is_set)
        drain = asyncio.create_task(coordinator.drain())
        await _until(lambda: participant.admission.streams_terminating)
        release.set()
        return await consumer, await drain

    try:
        error, report = asyncio.run(scenario())
    finally:
        release.set()
        channel.close()
        server.stop(0).wait()

    assert error.code() is grpc.StatusCode.UNAVAILABLE
    assert error.details() == SHUTTING_DOWN
    assert [item.redacted_text for item in received] == ["Patient [NAME] "]
    assert not any(item.final for item in received)
    finish = [
        event
        for event in report.events_for("grpc")
        if event.phase is DrainPhase.FINISH_IN_FLIGHT
    ][0]
    assert finish.cancelled == 1
    assert runtime.get_loader().active_request_count() == 0


# ---------------------------------------------------------------------------
# Public job queue: every lifecycle boundary
# ---------------------------------------------------------------------------


class _GatedStore(jobs.LocalJobStore):
    """Store that can pause one transition and logs every status commit."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.pause: Callable[[dict[str, Any]], Optional[threading.Event]] = (
            lambda changes: None
        )
        self.transitions: list[tuple[str, str]] = []
        self._log_lock = threading.Lock()

    def update_if(self, job_id, *, expected_statuses, **changes):
        gate = self.pause(changes)
        if gate is not None:
            gate.wait(timeout=10)
        updated = super().update_if(
            job_id, expected_statuses=expected_statuses, **changes
        )
        if updated is not None and "status" in changes:
            with self._log_lock:
                self.transitions.append((job_id, changes["status"]))
        return updated


def _job_payload(documents: int, *, webhook: bool = False) -> DeidentifyJobRequest:
    return DeidentifyJobRequest(
        documents=[
            DeidentifyJobDocument(id=f"synthetic-{index}", text="synthetic note")
            for index in range(documents)
        ],
        webhook=(
            JobWebhookRequest(
                url="https://callbacks.example.test/openmed",
                secret="synthetic-secret",
                max_attempts=3,
                backoff_seconds=0.5,
            )
            if webhook
            else None
        ),
    )


def _job_harness(tmp_path: Path, name: str):
    store = _GatedStore(tmp_path / f"{name}.json")
    webhooks: list[dict[str, Any]] = []

    def webhook_sender(url, payload, *, secret, max_attempts, backoff_seconds, **rest):
        webhooks.append(
            {
                "status": payload["status"],
                "max_attempts": max_attempts,
                "backoff_seconds": backoff_seconds,
                **rest,
            }
        )
        return WebhookDeliveryResult(success=True, attempts=1, status_code=204)

    queue = jobs.DeidentifyJobQueue(
        SimpleNamespace(validate_served_model=lambda model_name: model_name),
        store=store,
        max_workers=1,
        webhook_sender=webhook_sender,
    )
    coordinator = DrainCoordinator(DrainBudget(drain_seconds=0.2, release_seconds=0.2))
    coordinator.register(JobQueueDrainParticipant(lambda: queue))
    return store, queue, coordinator, webhooks


def _assert_single_terminal(store: _GatedStore, job_id: str) -> str:
    terminal = [
        status
        for recorded_id, status in store.transitions
        if recorded_id == job_id and status in jobs.TERMINAL_STATUSES
    ]
    assert len(terminal) == 1, terminal
    record = store.get(job_id)
    assert record is not None and record["status"] == terminal[0]
    return terminal[0]


def test_drain_at_every_job_boundary_never_loses_or_duplicates_work(tmp_path, subtests):
    with subtests.test(boundary="admission"):
        store, queue, coordinator, webhooks = _job_harness(tmp_path, "admission")
        asyncio.run(coordinator.drain())
        with pytest.raises(jobs.JobQueueClosedError):
            queue.submit(_job_payload(1))
        assert store._records == {}

    with subtests.test(boundary="queue_and_dispatch"):
        store, queue, coordinator, webhooks = _job_harness(tmp_path, "dispatch")
        hold = threading.Event()
        processed: list[str] = []

        def blocking(_payload, document):
            processed.append(document.id or "")
            hold.wait(timeout=10)
            return SimpleNamespace(pii_entities=[])

        queue._deidentify_document = blocking  # type: ignore[method-assign]
        running = queue.submit(_job_payload(1, webhook=True))
        queued = queue.submit(_job_payload(1, webhook=True))
        _wait_sync(lambda: bool(processed))
        report = asyncio.run(coordinator.drain())
        hold.set()  # the stale worker resumes after the handoff
        _wait_sync(lambda: queue.outstanding_count() == 0)

        assert _assert_single_terminal(store, running["id"]) == "cancelled"
        assert _assert_single_terminal(store, queued["id"]) == "cancelled"
        assert store.get(queued["id"])["started_at"] is None
        for job_id in (running["id"], queued["id"]):
            error = store.get(job_id)["error"]
            assert error["code"] == SHUTTING_DOWN and error["retryable"] is True
        assert processed == ["synthetic-0"]  # the queued job never ran
        assert webhooks == []
        event = report.events_for("job_queue")[1]
        assert event.cancelled == 2

    with subtests.test(boundary="checkpoint"):
        store, queue, coordinator, webhooks = _job_harness(tmp_path, "checkpoint")
        processed = []

        def quick(_payload, document):
            processed.append(document.id or "")
            return SimpleNamespace(pii_entities=[])

        queue._deidentify_document = quick  # type: ignore[method-assign]
        # Pause the first progress commit until the drain asks for a stop.
        store.pause = lambda changes: (
            queue._cancel_requested
            if changes.get("processed_count") == 1 and "status" not in changes
            else None
        )
        job = queue.submit(_job_payload(3))
        _wait_sync(lambda: bool(processed))
        asyncio.run(coordinator.drain())
        _wait_sync(lambda: queue.outstanding_count() == 0)

        assert _assert_single_terminal(store, job["id"]) == "cancelled"
        record = store.get(job["id"])
        assert record["processed_count"] == 1  # last committed checkpoint
        assert processed == ["synthetic-0"]

    with subtests.test(boundary="commit"):
        store, queue, coordinator, webhooks = _job_harness(tmp_path, "commit")
        hold = threading.Event()
        queue._deidentify_document = (  # type: ignore[method-assign]
            lambda _payload, _document: SimpleNamespace(pii_entities=[])
        )
        store.pause = lambda changes: (
            hold if changes.get("status") in {"done", "failed"} else None
        )
        job = queue.submit(_job_payload(1, webhook=True))
        _wait_sync(lambda: store.get(job["id"])["processed_count"] == 1)
        asyncio.run(coordinator.drain())
        hold.set()  # a stale terminal commit arrives after the fence
        _wait_sync(lambda: queue.outstanding_count() == 0)

        assert _assert_single_terminal(store, job["id"]) == "cancelled"
        assert webhooks == []

    with subtests.test(boundary="callback"):
        store, queue, coordinator, webhooks = _job_harness(tmp_path, "callback")
        drain_started = threading.Event()
        # Runs right after the queue closes admission (job_queue priority is 50).
        coordinator.register(
            CallbackDrainParticipant(
                "drain_probe",
                priority=60,
                stop_admission=lambda ctx: drain_started.set(),
            )
        )

        def after_drain_starts(_payload, _document):
            drain_started.wait(timeout=10)
            return SimpleNamespace(pii_entities=[])

        queue._deidentify_document = after_drain_starts  # type: ignore[method-assign]
        job = queue.submit(_job_payload(1, webhook=True))
        report = asyncio.run(coordinator.drain())

        assert _assert_single_terminal(store, job["id"]) == "done"
        assert len(webhooks) == 1
        assert webhooks[0]["status"] == "done"
        # A drain bounds callbacks to one attempt with no retry backoff.
        assert webhooks[0]["max_attempts"] == 1
        assert webhooks[0]["backoff_seconds"] == 0.0
        assert webhooks[0]["timeout_seconds"] == pytest.approx(0.1)
        assert store.get(job["id"])["webhook_delivery"]["success"] is True
        event = report.events_for("job_queue")[1]
        assert (event.completed, event.cancelled, event.unfinished) == (1, 0, 0)

    persisted = (tmp_path / "callback.json").read_text(encoding="utf-8")
    _assert_value_free(persisted)
    assert "synthetic-secret" not in persisted
    assert "callbacks.example.test" not in persisted


# ---------------------------------------------------------------------------
# Warm pool and accelerators
# ---------------------------------------------------------------------------


class _ResidencyLoader:
    def __init__(self) -> None:
        self.resident: set[str] = set()

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        self.resident.add(model_name)
        return object()

    def loaded_models(self) -> dict[str, Any]:
        return {
            name: {"models": 1, "tokenizers": 0, "pipelines": 1}
            for name in self.resident
        }

    def unload_model(self, model_name: str) -> dict[str, Any]:
        removed = int(model_name in self.resident)
        self.resident.discard(model_name)
        return {
            "model_name": model_name,
            "models": removed,
            "tokenizers": 0,
            "pipelines": removed,
        }


class _FakeAccelerator:
    blocking = True

    def __init__(self, kind: str, *, baseline: int, wedge: bool = False) -> None:
        self.kind = kind
        self.units = baseline
        self.cached = 0
        self.wedge = threading.Event() if wedge else None

    def allocated(self) -> int:
        return self.units + self.cached

    def synchronize(self) -> None:
        if self.wedge is not None:
            self.wedge.wait(timeout=10)

    def release(self) -> None:
        self.cached = 0


def test_warm_pool_and_accelerators_return_to_their_declared_baseline():
    loader = _ResidencyLoader()
    pool = WarmPool(lambda: loader, default_keep_alive_seconds=300)
    active = pool.begin_request("synthetic/active")
    pool.create_pipeline("synthetic/active")
    idle = pool.begin_request("synthetic/idle")
    pool.create_pipeline("synthetic/idle")
    pool.finish_request(idle, None)
    assert loader.resident == {"synthetic/active", "synthetic/idle"}

    device = _FakeAccelerator("fake_gpu", baseline=64)
    wedged = _FakeAccelerator("wedged_gpu", baseline=0, wedge=True)
    accelerators = AcceleratorDrainParticipant([device, wedged])
    device.cached = 4096  # cache grown while serving; released on drain
    coordinator = DrainCoordinator(DrainBudget(drain_seconds=0.05, release_seconds=0.2))
    coordinator.register(WarmPoolDrainParticipant(lambda: pool))
    coordinator.register(accelerators)

    try:
        report = asyncio.run(coordinator.drain())
    finally:
        assert wedged.wedge is not None
        wedged.wedge.set()

    pool_event = report.events_for("warm_pool")[0]
    assert pool_event.completed == 1  # the idle model was released
    assert pool_event.unfinished == 1  # the active request keeps its model
    assert loader.resident == {"synthetic/active"}
    with pytest.raises(WarmPoolClosedError):
        pool.begin_request("synthetic/idle")
    with pytest.raises(WarmPoolClosedError):
        pool.create_pipeline("synthetic/cold")

    pool.finish_request(active, None)  # last reference of the drained generation
    assert pool.active_request_count() == 0
    assert loader.resident == set()

    accelerator = report.events_for("accelerator")[0]
    assert (accelerator.completed, accelerator.unfinished) == (1, 1)
    assert accelerator.outcome is DrainOutcome.DEADLINE_EXCEEDED
    assert accelerators.retained() == {"fake_gpu": 0, "wedged_gpu": 0}
    assert device.allocated() == 64


# ---------------------------------------------------------------------------
# Restart cycles and process signals
# ---------------------------------------------------------------------------


def test_repeated_start_drain_restart_cycles_return_resources_to_zero(monkeypatch):
    monkeypatch.setenv("OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS", "0.3")
    monkeypatch.setenv("OPENMED_SERVICE_SHUTDOWN_RELEASE_SECONDS", "0.4")
    monkeypatch.setenv("OPENMED_SERVICE_JOBS_WORKERS", "1")
    hold = threading.Event()
    cycle_calls: list[str] = []

    def fake_deidentify(text: str, **_: Any) -> DeidentificationResult:
        if text == "slow synthetic note":
            hold.wait(timeout=10)
        return _deid_result(text)

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify)
    app = create_app()
    register_drain_participant(
        app,
        CallbackDrainParticipant(
            "cycle_probe",
            release=lambda ctx: cycle_calls.append("release"),
        ),
    )
    coordinators = []
    reports = []
    interrupted_job = None
    for cycle in range(3):
        with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
            assert client.get("/readyz").status_code == 200
            response = client.post(
                "/pii/deidentify", json={"text": "synthetic note", "method": "mask"}
            )
            assert response.status_code == 200
            if cycle == 0:
                created = client.post(
                    "/jobs", json={"documents": [{"text": "slow synthetic note"}]}
                )
                assert created.status_code == 202
                interrupted_job = created.json()["id"]
            if cycle == 1:
                record = client.get(f"/jobs/{interrupted_job}").json()
                assert record["status"] == "cancelled"
                assert record["error"]["code"] == SHUTTING_DOWN
                assert record["error"]["retryable"] is True
            coordinators.append(app.state.drain)
        hold.set()  # release the cycle-0 straggler only after its drain fenced it
        reports.append(app.state.drain_report)
        assert app.state.inflight == 0
        assert app.state.drain_streams.active == 0
        pool = app.state.runtime.get_loader()
        _wait_sync(lambda: pool.active_request_count() == 0)

    assert len({id(item) for item in coordinators}) == 3
    assert cycle_calls == ["release"] * 3
    assert reports[0].events_for("job_queue")[1].cancelled == 1
    assert [report.outcome for report in reports[1:]] == [DrainOutcome.COMPLETED] * 2
    _wait_sync(
        lambda: (
            not any(
                thread.name.startswith("openmed-job") and thread.is_alive()
                for thread in threading.enumerate()
            )
        )
    )


_SIGNAL_PROBE = textwrap.dedent(
    """
    import asyncio
    import json

    import uvicorn

    from openmed.service import runtime as service_runtime
    from openmed.service.app import create_app
    from openmed.service.drain import CallbackDrainParticipant, register_drain_participant


    class FakeLoader:
        def __init__(self, config):
            self.config = config

        def resolve_model_name(self, model_name):
            return model_name

        def loaded_models(self):
            return {}


    service_runtime.ModelLoader = FakeLoader
    app = create_app()


    def record(phase):
        def hook(ctx):
            print(json.dumps({"phase": phase}), flush=True)

        return hook


    register_drain_participant(
        app,
        CallbackDrainParticipant(
            "signal_probe",
            stop_admission=record("stop_admission"),
            finish_in_flight=record("finish_in_flight"),
            flush=record("flush"),
            release=record("release"),
        ),
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )


    async def main():
        serving = asyncio.ensure_future(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        print(json.dumps({"ready": True}), flush=True)
        await serving


    asyncio.run(main())
    """
)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_sigterm_runs_the_ordered_drain_before_the_process_exits(tmp_path):
    script = tmp_path / "signal_probe.py"
    script.write_text(_SIGNAL_PROBE, encoding="utf-8")
    env = dict(os.environ)
    env.update(
        {
            "OPENMED_PROFILE": "test",
            "OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS": "1",
            "OPENMED_SERVICE_SHUTDOWN_RELEASE_SECONDS": "1",
            "OPENMED_SERVICE_JOBS_STORE_PATH": str(tmp_path / "jobs.json"),
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        }
    )
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    try:
        assert process.stdout is not None
        assert json.loads(process.stdout.readline()) == {"ready": True}
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    phases = [json.loads(line)["phase"] for line in output.splitlines() if line]
    assert phases == ["stop_admission", "finish_in_flight", "flush", "release"]
    assert process.returncode in {0, -signal.SIGTERM}
