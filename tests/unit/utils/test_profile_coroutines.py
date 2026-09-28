"""Regression tests for timing coroutine execution rather than creation."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from openmed.utils import profiling


@pytest.fixture
def clock_and_profiler(monkeypatch):
    clock = SimpleNamespace(now=10.0)
    profiler = profiling.Profiler()
    monkeypatch.setattr(profiling, "_global_profiler", profiler)
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: clock.now)
    return clock, profiler


def test_creation_is_not_timed_and_await_includes_suspension(clock_and_profiler):
    clock, profiler = clock_and_profiler

    @profiling.profile()
    async def operation(value, *, increment):
        clock.now += 1.0
        await asyncio.sleep(0)
        clock.now += 1.5
        return value + increment

    coroutine = operation(40, increment=2)
    try:
        assert profiler.report().timings == []
        assert asyncio.run(coroutine) == 42
    finally:
        coroutine.close()
    timing = profiler.report().get_timing("operation")
    assert timing is not None
    assert timing.duration == pytest.approx(2.5)
    assert profiler.report().timing_count == 1


def test_coroutine_identity_and_metadata_are_preserved():
    async def operation(value: int, *, increment: int = 2) -> int:
        """Synthetic operation."""
        return value + increment

    decorated = profiling.profile("custom")(operation)
    assert inspect.iscoroutinefunction(decorated)
    assert decorated.__wrapped__ is operation
    assert decorated.__name__ == operation.__name__
    assert decorated.__doc__ == operation.__doc__
    assert inspect.signature(decorated) == inspect.signature(operation)


def test_unstarted_coroutine_can_be_closed_without_a_timing(clock_and_profiler):
    _, profiler = clock_and_profiler

    @profiling.profile()
    async def operation():
        return 42

    coroutine = operation()
    coroutine.close()
    assert profiler.report().timings == []


def test_async_failure_propagates_and_records_elapsed_time(clock_and_profiler):
    clock, profiler = clock_and_profiler
    failure = ValueError("synthetic failure")

    @profiling.profile("failing")
    async def operation():
        await asyncio.sleep(0)
        clock.now += 0.75
        raise failure

    with pytest.raises(ValueError) as caught:
        asyncio.run(operation())
    assert caught.value is failure
    assert profiler.report().timing_count == 1
    assert profiler.report().get_timing("failing").duration == pytest.approx(0.75)


def test_cancellation_propagates_and_records_elapsed_time(clock_and_profiler):
    clock, profiler = clock_and_profiler

    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()

        @profiling.profile("cancelled")
        async def operation():
            started.set()
            await release.wait()

        task = asyncio.create_task(operation())
        await started.wait()
        clock.now += 2.0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

    asyncio.run(scenario())
    assert profiler.report().timing_count == 1
    assert profiler.report().get_timing("cancelled").duration == pytest.approx(2.0)


def test_disabled_profiler_still_awaits_the_operation(clock_and_profiler):
    clock, profiler = clock_and_profiler
    profiler.enabled = False

    @profiling.profile()
    async def operation():
        await asyncio.sleep(0)
        clock.now += 1.0
        return 42

    assert asyncio.run(operation()) == 42
    assert profiler.report().timings == []


def test_concurrent_calls_have_independent_timing_contexts(clock_and_profiler):
    clock, profiler = clock_and_profiler

    @profiling.profile("concurrent")
    async def operation(value):
        clock.now += 0.5
        await asyncio.sleep(0)
        return value

    async def scenario():
        return await asyncio.gather(*(operation(value) for value in range(3)))

    assert asyncio.run(scenario()) == [0, 1, 2]
    timings = profiler.report().timings
    assert len(timings) == 3
    assert all(timing.name == "concurrent" for timing in timings)
    assert sorted(timing.duration for timing in timings) == [0.5, 1.0, 1.5]


@pytest.mark.parametrize("fail", [False, True])
def test_synchronous_behavior_is_unchanged(clock_and_profiler, fail):
    clock, profiler = clock_and_profiler
    failure = ValueError("synthetic sync failure")

    @profiling.profile("sync")
    def operation(value):
        clock.now += 1.25
        if fail:
            raise failure
        return value

    assert not inspect.iscoroutinefunction(operation)
    if fail:
        with pytest.raises(ValueError) as caught:
            operation(42)
        assert caught.value is failure
    else:
        assert operation(42) == 42
    assert profiler.report().timing_count == 1
    assert profiler.report().get_timing("sync").duration == pytest.approx(1.25)


def test_async_instance_method_receives_its_arguments(clock_and_profiler):
    clock, profiler = clock_and_profiler

    class Worker:
        @profiling.profile()
        async def operation(self, value):
            clock.now += 1.0
            return self, value

    worker = Worker()
    assert asyncio.run(worker.operation(42)) == (worker, 42)
    assert profiler.report().get_timing("operation").duration == pytest.approx(1.0)
