"""Actual ASGI completion and service permit ownership endurance controls."""

import asyncio
import json

import pytest

from openmed.eval.resource_soak import ResourceSnapshot, SoakProfile, run_resource_soak
from openmed.service.resilience import ResilienceManager, ServiceResilienceConfig
from openmed.service.throttle import ConcurrencyGate, RequestWork
from openmed.service.warm_pool import WarmPool


class SyntheticWorkerApp:
    """Inject a synthetic asynchronous worker through actual RequestWork ownership."""

    def __init__(self, *, leak=0, stuck=False):
        self.gate = ConcurrencyGate(max_concurrency=4, wait_seconds=0.01)
        self.active = self.handles = self.memory = self.callbacks = 0
        self.leak = leak
        self.stuck = stuck
        self.draining = False
        self.action_counts = {}
        self.retry_attempts = 0
        self.current_action = "load"
        self.resilience = ResilienceManager(
            ServiceResilienceConfig(
                max_attempts=2, backoff_initial_seconds=0, backoff_jitter_seconds=0
            )
        )

    def probe(self):
        return ResourceSnapshot(
            permits=self.active, handles=self.handles, memory_bytes=self.memory
        )

    async def lifecycle(self, action, phase):
        self.action_counts[(action, phase)] = (
            self.action_counts.get((action, phase), 0) + 1
        )
        self.draining = action == "drain" and phase == "before"
        self.current_action = action
        if phase == "after" and action == "model_swap":
            self.memory += self.leak

    async def __call__(self, scope, receive, send):
        await receive()
        if self.draining:
            await send({"type": "http.response.start", "status": 503})
            await send({"type": "http.response.body", "body": b""})
            return
        permit = await self.gate.acquire()
        assert permit is not None
        work = RequestWork(permit, self.gate, "global", "global", None)
        self.active += 1
        self.handles += 1

        def finished():
            self.active -= 1
            self.handles -= 1
            self.callbacks += 1

        work.on_finished(finished)
        pending = asyncio.create_task(
            asyncio.Event().wait() if self.stuck else asyncio.sleep(0.001)
        )
        work.track(pending)
        try:
            await pending
            if self.current_action == "retry":
                attempts = 0

                def operation():
                    nonlocal attempts
                    attempts += 1
                    self.retry_attempts += 1
                    if attempts == 1:
                        raise TimeoutError("synthetic transient worker timeout")
                    return None

                self.resilience.execute("synthetic_worker", operation)
            await send({"type": "http.response.start", "status": 200})
            await send(
                {"type": "http.response.body", "body": b"one", "more_body": True}
            )
            await send({"type": "http.response.body", "body": b"two"})
        finally:
            work.close()


def execute(app, profile=None):
    return asyncio.run(
        run_resource_soak(
            app,
            profile=profile or SoakProfile(),
            probe=app.probe,
            lifecycle=app.lifecycle,
        )
    )


def test_repeated_real_request_ownership_cancellation_drain_and_stream_cleanup():
    app = SyntheticWorkerApp()
    report = execute(app)
    assert report["outcome"] == "pass"
    assert report["cycles_finished"] == 10
    assert report["attempted_requests"] == 80
    assert report["cancelled_requests"] == 4
    assert report["expected_drain_rejections"] == 16
    assert app.callbacks == 64
    assert app.retry_attempts == 32
    assert app.active == app.handles == 0
    assert report["cleanup_runs"] == 10
    assert set(report["lifecycle_counts"].values()) == {2}

    # Verify actual admission slots can all be reacquired after every response.
    async def reacquire():
        permits = [await app.gate.acquire() for _ in range(4)]
        assert all(permit is not None for permit in permits)
        assert await app.gate.acquire() is None
        for permit in permits:
            permit.release()

    asyncio.run(reacquire())


def test_memory_leak_fails_independently_of_successful_throughput():
    report = execute(SyntheticWorkerApp(leak=64))
    assert report["outcome"] == "fail"
    assert report["reasons"] == ["unrecovered_memory_bytes"]
    assert report["completed_requests_per_second"] > 0


def test_stuck_worker_times_out_and_releases_actual_permits():
    app = SyntheticWorkerApp(stuck=True)
    report = execute(app, SoakProfile(request_deadline_seconds=0.01))
    assert report["outcome"] == "fail"
    assert "worker_timeout" in report["reasons"]
    assert app.active == app.handles == 0


def test_declared_memory_tolerance_is_pinned():
    profile = SoakProfile(tolerances=ResourceSnapshot(memory_bytes=128))
    report = execute(SyntheticWorkerApp(leak=64), profile)
    assert report["outcome"] == "pass"
    assert report["post_cycle_resource_peaks"]["memory_bytes"] == 128
    assert report["profile"]["tolerances"]["memory_bytes"] == 128


@pytest.mark.parametrize("resource", ["permits", "leases", "handles", "queued_jobs"])
def test_all_resource_residue_dimensions_fail(resource):
    app = SyntheticWorkerApp()
    calls = 0

    def probe():
        nonlocal calls
        calls += 1
        return ResourceSnapshot(**{resource: int(calls > 1)})

    report = asyncio.run(
        run_resource_soak(
            app, profile=SoakProfile(), probe=probe, lifecycle=app.lifecycle
        )
    )
    assert report["reasons"] == ["unrecovered_" + resource]


def test_payload_path_and_callback_errors_never_enter_report():
    app = SyntheticWorkerApp()

    async def broken(action, phase):
        raise RuntimeError("PRIVATE PAYLOAD SENTINEL")

    report = asyncio.run(
        run_resource_soak(
            app,
            profile=SoakProfile(),
            probe=app.probe,
            lifecycle=broken,
            payload={"text": "PRIVATE PAYLOAD SENTINEL"},
            path="/private/path",
        )
    )
    assert report["reasons"] == ["cleanup_error", "lifecycle_error"]
    assert "PRIVATE" not in json.dumps(report)
    assert "/private/path" not in json.dumps(report)


def test_initial_probe_failure_has_controlled_value_free_context():
    app = SyntheticWorkerApp()

    def broken():
        raise RuntimeError("PRIVATE SENTINEL")

    with pytest.raises(ValueError, match="^resource_soak_probe_unavailable$") as caught:
        asyncio.run(
            run_resource_soak(
                app, profile=SoakProfile(), probe=broken, lifecycle=app.lifecycle
            )
        )
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "values",
    [
        {"cycles": True},
        {"cycles": 10001},
        {"requests_per_cycle": 10001},
        {"concurrency": 65},
        {"request_deadline_seconds": float("nan")},
        {"evidence": "guaranteed_production"},
    ],
)
def test_profile_limits(values):
    with pytest.raises(ValueError):
        SoakProfile(**values)


@pytest.mark.integration
def test_actual_warm_pool_model_swap_and_lease_recovery():
    class Loader:
        def __init__(self):
            self.pipelines = {}
            self.releases = 0

        def resolve_model_name(self, name):
            return name

        def create_pipeline(self, name, **kwargs):
            return self.pipelines.setdefault(name, object())

        def loaded_models(self):
            return {
                name: {"models": 0, "tokenizers": 0, "pipelines": 1}
                for name in self.pipelines
            }

        def unload_model(self, name):
            self.releases += 1
            self.pipelines.pop(name, None)
            return {"models": 0, "tokenizers": 0, "pipelines": 1}

        def unload_all_models(self):
            count = len(self.pipelines)
            self.pipelines.clear()
            self.releases += count
            return {"models": 0, "tokenizers": 0, "pipelines": count}

    loader = Loader()
    pool = WarmPool(
        lambda: loader, max_resident_models=2, default_model_footprint_bytes=1
    )

    class ModelApp(SyntheticWorkerApp):
        def __init__(self):
            super().__init__()
            self.model = "synthetic_a"

        async def lifecycle(self, action, phase):
            await super().lifecycle(action, phase)
            if action == "model_swap" and phase == "before":
                self.model = (
                    "synthetic_b" if self.model == "synthetic_a" else "synthetic_a"
                )
            if phase == "after":
                pool.unload_all_models()

        async def __call__(self, scope, receive, send):
            model = pool.begin_request(self.model)
            try:
                pool.create_pipeline(model)
                await super().__call__(scope, receive, send)
            finally:
                pool.finish_request(model, "0s")

        def probe(self):
            models = pool.loaded_models()["models"]
            return ResourceSnapshot(
                permits=self.active,
                leases=sum(m["active_requests"] for m in models.values()),
                handles=sum(m["pipelines"] for m in models.values()),
            )

    report = execute(ModelApp())
    assert report["outcome"] == "pass"
    assert report["post_cycle_resource_peaks"]["leases"] == 0
    assert report["post_cycle_resource_peaks"]["handles"] == 0
    assert loader.releases > 0


def test_unplanned_cancellation_cannot_qualify_zero_throughput():
    app = SyntheticWorkerApp()

    async def cancelled(scope, receive, send):
        raise asyncio.CancelledError()

    report = asyncio.run(
        run_resource_soak(
            cancelled, profile=SoakProfile(), probe=app.probe, lifecycle=app.lifecycle
        )
    )
    assert report["outcome"] == "fail"
    assert report["reasons"] == ["unexpected_cancellation"]
