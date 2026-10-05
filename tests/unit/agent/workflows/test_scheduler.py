"""Offline scheduler safety controls with injected executors and clocks."""

import asyncio
import json
from dataclasses import replace

import pytest

from openmed.agent.action_graph import ActionNode
from openmed.agent.correlation import RunId
from openmed.agent.workflows.scheduler import (
    ActionScheduler,
    ActionState,
    CancellationSignal,
    DispatchOutcome,
    ScheduledAction,
    SchedulerCheckpoint,
    SchedulerError,
    SchedulerLimits,
    StopReason,
)

RUN = RunId.parse("run_" + "1" * 32)
TOOL = "tool:openmed.agent/synthetic"
PRIVATE = "SYNTHETIC PATIENT MRN 8675309 Bearer abc.def /private/notes"


def action(key, *parents, read_only=True, units=1):
    return ScheduledAction(ActionNode(key, TOOL, parents), read_only, units)


def limits(**changes):
    return replace(SchedulerLimits(2, 20, 4, 40, 1_000, True), **changes)


class Executor:
    def __init__(self):
        self.started = []
        self.active = 0
        self.peak = 0
        self.releases = {}
        self.outcomes = {}

    async def dispatch(self, run_id, node, cancellation):
        assert run_id == RUN
        assert not cancellation.cancelled
        self.started.append(node.action_id)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if node.action_id in self.releases:
                await self.releases[node.action_id].wait()
            outcome = self.outcomes.get(node.action_id, DispatchOutcome.COMMITTED)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        finally:
            self.active -= 1


async def until(predicate):
    # Yield to injected coroutines; no wall-clock timing assertion is needed.
    for _ in range(10000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("synthetic executor did not progress")


def scheduler(actions, executor=None, snapshots=None, budget=None, clock=lambda: 0):
    return ActionScheduler(
        actions,
        budget or limits(),
        executor or Executor(),
        (snapshots if snapshots is not None else []).append,
        clock=clock,
    )


def test_independent_progress_dependency_commit_and_concurrency():
    async def run():
        executor, saved = Executor(), []
        executor.releases = {key: asyncio.Event() for key in ("a", "b")}
        worker = scheduler(
            [action("join", "a", "b"), action("c"), action("b"), action("a")],
            executor,
            saved,
        )
        task = asyncio.create_task(worker.run(RUN))
        await until(lambda: len(executor.started) == 2)
        assert executor.started == ["a", "b"]
        assert executor.peak == 2
        executor.releases["b"].set()
        await until(lambda: "c" in executor.started)
        assert "join" not in executor.started
        executor.releases["a"].set()
        result = await task
        assert executor.started == ["a", "b", "c", "join"]
        assert result.stop_reason is StopReason.COMPLETED
        assert all(s is ActionState.COMMITTED for _, s in result.states)
        before_join = next(
            cp for cp in saved if dict(cp.states)["join"] is ActionState.RUNNING
        )
        assert dict(before_join.states)["a"] is ActionState.COMMITTED
        assert dict(before_join.states)["b"] is ActionState.COMMITTED
        assert result.admitted_actions == 4

    asyncio.run(run())


@pytest.mark.parametrize("parallel", [False, True])
def test_explicit_read_policy_and_exclusive_effect_barrier(parallel):
    async def run():
        executor = Executor()
        executor.releases["a"] = asyncio.Event()
        worker = scheduler(
            [action("a"), action("b", read_only=False), action("c")],
            executor,
            budget=limits(parallel_reads=parallel),
        )
        task = asyncio.create_task(worker.run(RUN))
        await until(lambda: executor.started == ["a"])
        for _ in range(10):
            await asyncio.sleep(0)
        assert executor.started == ["a"]
        executor.releases["a"].set()
        result = await task
        assert result.stop_reason is StopReason.COMPLETED
        assert executor.started == ["a", "b", "c"]
        assert executor.peak == 1

    asyncio.run(run())


def test_default_action_classification_is_exclusive():
    assert ScheduledAction(ActionNode("a", TOOL)).read_only is False


def test_inflight_resource_reservations_admit_fitting_independent_work():
    async def run():
        executor = Executor()
        executor.releases["a"] = asyncio.Event()
        task = asyncio.create_task(
            scheduler(
                [action("a", units=2), action("b", units=2), action("c", units=1)],
                executor,
                budget=limits(max_concurrency=3, max_inflight_units=3),
            ).run(RUN)
        )
        await until(lambda: "c" in executor.started)
        assert executor.started == ["a", "c"]
        assert executor.peak == 2
        executor.releases["a"].set()
        assert (await task).stop_reason is StopReason.COMPLETED

    asyncio.run(run())


@pytest.mark.parametrize("budget", [limits(max_actions=1), limits(max_total_units=1)])
def test_cumulative_budget_stops_additional_dispatch(budget):
    executor = Executor()
    result = asyncio.run(
        scheduler([action("a"), action("b")], executor, budget=budget).run(RUN)
    )
    assert executor.started == ["a"]
    assert result.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert result.admitted_actions == result.reserved_units == 1
    assert dict(result.states)["a"] is ActionState.COMMITTED
    assert dict(result.states)["b"] is ActionState.PENDING


def test_unadmittable_resource_estimate_never_dispatches():
    executor = Executor()
    result = asyncio.run(scheduler([action("a", units=5)], executor).run(RUN))
    assert executor.started == []
    assert result.stop_reason is StopReason.BUDGET_EXHAUSTED


@pytest.mark.parametrize(
    "verdict,reason",
    [
        (DispatchOutcome.REVIEW_REQUIRED, StopReason.REVIEW_REQUIRED),
        (DispatchOutcome.FAILED, StopReason.DISPATCH_FAILED),
        (RuntimeError(PRIVATE), StopReason.REVIEW_REQUIRED),
        (PRIVATE, StopReason.REVIEW_REQUIRED),
    ],
)
def test_failure_and_review_stop_descendants_and_independent_pending(verdict, reason):
    executor = Executor()
    executor.outcomes["a"] = verdict
    result = asyncio.run(
        scheduler(
            [action("a"), action("b", "a"), action("c")],
            executor,
            budget=limits(max_concurrency=1),
        ).run(RUN)
    )
    assert executor.started == ["a"]
    assert result.stop_reason is reason
    assert dict(result.states)["b"] is ActionState.PENDING
    assert dict(result.states)["c"] is ActionState.PENDING
    assert PRIVATE not in result.to_json()


def test_review_peer_prevents_successful_peer_descendant_admission():
    async def run():
        executor = Executor()
        executor.outcomes["b"] = DispatchOutcome.REVIEW_REQUIRED
        result = await scheduler(
            [action("a"), action("b"), action("c", "a")], executor
        ).run(RUN)
        assert executor.started == ["a", "b"]
        assert result.stop_reason is StopReason.REVIEW_REQUIRED
        assert dict(result.states)["a"] is ActionState.COMMITTED
        assert dict(result.states)["c"] is ActionState.PENDING

    asyncio.run(run())


def test_token_cancellation_interrupts_running_and_stops_pending():
    async def run():
        executor, signal = Executor(), CancellationSignal()
        executor.releases["a"] = asyncio.Event()
        task = asyncio.create_task(
            scheduler(
                [action("a"), action("b", "a")],
                executor,
            ).run(RUN, cancellation=signal)
        )
        await until(lambda: executor.started == ["a"])
        signal.cancel()
        result = await task
        assert result.stop_reason is StopReason.CANCELLED
        assert executor.started == ["a"]
        assert dict(result.states)["a"] is ActionState.REVIEW_REQUIRED
        assert dict(result.states)["b"] is ActionState.PENDING
        assert executor.active == 0

    asyncio.run(run())


def test_pre_cancelled_run_never_invokes_executor():
    signal, executor = CancellationSignal(), Executor()
    signal.cancel()
    result = asyncio.run(
        scheduler([action("a")], executor).run(RUN, cancellation=signal)
    )
    assert result.stop_reason is StopReason.CANCELLED
    assert executor.started == []


def test_asyncio_cancellation_drains_and_checkpoints_before_propagating():
    async def run():
        executor, saved = Executor(), []
        executor.releases["a"] = asyncio.Event()
        task = asyncio.create_task(scheduler([action("a")], executor, saved).run(RUN))
        await until(lambda: executor.started == ["a"])
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert executor.active == 0
        assert saved[-1].stop_reason is StopReason.CANCELLED
        assert dict(saved[-1].states)["a"] is ActionState.REVIEW_REQUIRED

    asyncio.run(run())


def test_elapsed_budget_uses_injected_clock_and_stops_descendants():
    now = [100]

    class TimedExecutor(Executor):
        async def dispatch(self, *args):
            result = await super().dispatch(*args)
            now[0] += 1000
            return result

    executor = TimedExecutor()
    result = asyncio.run(
        scheduler(
            [action("a"), action("b", "a")],
            executor,
            clock=lambda: now[0],
        ).run(RUN)
    )
    assert result.elapsed_ns == 1000
    assert executor.started == ["a"]
    assert result.stop_reason is StopReason.BUDGET_EXHAUSTED


@pytest.mark.parametrize("bad_clock", [-1, True, PRIVATE])
def test_invalid_clock_fails_closed_without_values(bad_clock):
    executor = Executor()
    result = asyncio.run(
        scheduler([action("a")], executor, clock=lambda: bad_clock).run(RUN)
    )
    assert result.stop_reason is StopReason.CLOCK_FAILED
    assert executor.started == []
    assert PRIVATE not in result.to_json()


def test_regressing_clock_stops_before_dispatch():
    times = iter([10, 9])
    executor = Executor()
    result = asyncio.run(
        scheduler([action("a")], executor, clock=lambda: next(times)).run(RUN)
    )
    assert result.stop_reason is StopReason.CLOCK_FAILED
    assert executor.started == []


@pytest.mark.parametrize("boundary", [1, 2])
def test_checkpoint_failure_prevents_dispatch_or_descendants(boundary):
    executor, saved = Executor(), []

    def save(cp):
        saved.append(cp)
        if len(saved) == boundary:
            raise RuntimeError(PRIVATE)

    worker = ActionScheduler(
        [action("a"), action("b", "a")], limits(), executor, save, clock=lambda: 0
    )
    result = asyncio.run(worker.run(RUN))
    assert executor.started == ([] if boundary == 1 else ["a"])
    assert result.stop_reason is StopReason.CHECKPOINT_FAILED
    assert dict(result.states)["b"] is ActionState.PENDING
    assert PRIVATE not in result.to_json()


@pytest.mark.parametrize(
    "field",
    [
        "max_actions",
        "max_concurrency",
        "max_inflight_units",
        "max_total_units",
        "max_elapsed_ns",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, PRIVATE])
def test_limits_fail_closed(field, value):
    with pytest.raises(SchedulerError) as exc:
        limits(**{field: value})
    assert str(exc.value) == "invalid_integer"


def test_invalid_graph_is_never_scheduled():
    for nodes in (
        [action("a", "b")],
        [action("a"), action("a")],
        [action("a", "b"), action("b", "a")],
    ):
        with pytest.raises(SchedulerError, match="invalid_graph"):
            scheduler(nodes)


def test_checkpoint_roundtrip_exact_schema_and_private_negative_controls():
    cp = asyncio.run(scheduler([action("a")]).run(RUN))
    assert SchedulerCheckpoint.from_dict(json.loads(cp.to_json())) == cp
    for changes in (
        {"extra": PRIVATE},
        {"states": [[PRIVATE, "committed"]]},
        {"run_id": PRIVATE},
        {"stop_reason": PRIVATE},
    ):
        with pytest.raises(SchedulerError) as exc:
            SchedulerCheckpoint.from_dict({**cp.to_dict(), **changes})
        assert str(exc.value) == "invalid_checkpoint"
        assert exc.value.__context__ is None


def test_resume_completed_and_terminal_failure_does_not_repeat():
    for outcome in (DispatchOutcome.COMMITTED, DispatchOutcome.FAILED):
        executor = Executor()
        executor.outcomes["a"] = outcome
        worker = scheduler([action("a")], executor)
        cp = asyncio.run(worker.run(RUN))
        result = asyncio.run(worker.run(RUN, checkpoint=cp))
        assert result == cp
        assert executor.started == ["a"]


def test_resume_rejects_changed_graph_policy_run_and_counters():
    cp = asyncio.run(scheduler([action("a")]).run(RUN))
    workers = [
        scheduler([action("a", read_only=False)]),
        scheduler([action("a")], budget=limits(max_concurrency=3)),
    ]
    for worker in workers:
        with pytest.raises(SchedulerError, match="identity_mismatch"):
            asyncio.run(worker.run(RUN, checkpoint=cp))
    with pytest.raises(SchedulerError, match="identity_mismatch"):
        asyncio.run(
            scheduler([action("a")]).run(RunId.parse("run_" + "2" * 32), checkpoint=cp)
        )
    with pytest.raises(SchedulerError, match="budget_mismatch"):
        asyncio.run(
            scheduler([action("a")]).run(RUN, checkpoint=replace(cp, reserved_units=0))
        )


def test_same_scheduler_rejects_simultaneous_runs():
    async def run():
        executor = Executor()
        executor.releases["a"] = asyncio.Event()
        worker = scheduler([action("a")], executor)
        task = asyncio.create_task(worker.run(RUN))
        await until(lambda: executor.started == ["a"])
        with pytest.raises(SchedulerError, match="already_running"):
            await worker.run(RUN)
        executor.releases["a"].set()
        await task

    asyncio.run(run())


def test_cancellation_at_write_ahead_boundary_prevents_invocation():
    executor, signal = Executor(), CancellationSignal()

    def save(cp):
        if dict(cp.states)["a"] is ActionState.RUNNING:
            signal.cancel()

    worker = ActionScheduler([action("a")], limits(), executor, save, clock=lambda: 0)
    result = asyncio.run(worker.run(RUN, cancellation=signal))
    assert executor.started == []
    assert result.stop_reason is StopReason.CANCELLED
    assert result.admitted_actions == 1


def test_late_commit_during_cancellation_is_retained_without_descendants():
    async def run():
        entered, signal = asyncio.Event(), CancellationSignal()

        class LateExecutor:
            async def dispatch(self, run_id, node, cancellation):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    # Synthetic sink committed while cancellation was requested.
                    return DispatchOutcome.COMMITTED

        task = asyncio.create_task(
            scheduler(
                [action("a"), action("b", "a")],
                LateExecutor(),
            ).run(RUN, cancellation=signal)
        )
        await entered.wait()
        signal.cancel()
        result = await task
        assert result.stop_reason is StopReason.CANCELLED
        assert dict(result.states) == {
            "a": ActionState.COMMITTED,
            "b": ActionState.PENDING,
        }

    asyncio.run(run())


def test_parallel_reads_disabled_never_overlaps_read_only_actions():
    async def run():
        executor = Executor()
        executor.releases["a"] = asyncio.Event()
        task = asyncio.create_task(
            scheduler(
                [action("a"), action("b")],
                executor,
                budget=limits(parallel_reads=False),
            ).run(RUN)
        )
        await until(lambda: executor.started == ["a"])
        for _ in range(10):
            await asyncio.sleep(0)
        assert executor.started == ["a"]
        executor.releases["a"].set()
        assert (await task).stop_reason is StopReason.COMPLETED
        assert executor.peak == 1

    asyncio.run(run())


def test_all_admission_snapshots_respect_concurrency_and_resource_caps():
    saved = []
    actions = [action(f"a{i:02}", units=i % 3 + 1) for i in range(12)]
    budget = limits(max_concurrency=3, max_inflight_units=4)
    result = asyncio.run(scheduler(actions, snapshots=saved, budget=budget).run(RUN))
    costs = {a.node.action_id: a.resource_units for a in actions}
    assert result.stop_reason is StopReason.COMPLETED
    for cp in saved:
        running = [key for key, state in cp.states if state is ActionState.RUNNING]
        assert len(running) <= budget.max_concurrency
        assert sum(costs[key] for key in running) <= budget.max_inflight_units
        assert cp.admitted_actions <= budget.max_actions
        assert cp.reserved_units <= budget.max_total_units


def test_plan_fingerprint_and_initial_admission_ignore_input_and_dependency_order():
    nodes = [action("z", "b", "a"), action("b"), action("a")]
    saved = []
    result = asyncio.run(scheduler(nodes, snapshots=saved).run(RUN))
    reversed_nodes = [action("a"), action("z", "a", "b"), action("b")]
    other = asyncio.run(scheduler(reversed_nodes).run(RUN))
    assert result.plan_digest == other.plan_digest
    assert [key for key, state in saved[0].states if state is ActionState.RUNNING] == [
        "a"
    ]


def test_elapsed_time_after_save_blocks_admission():
    now = [0]
    executor = Executor()

    def save(cp):
        now[0] = 1000

    worker = ActionScheduler(
        [action("a")], limits(), executor, save, clock=lambda: now[0]
    )
    result = asyncio.run(worker.run(RUN))
    assert result.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert executor.started == []


def test_empty_graph_completes_without_invocation():
    executor = Executor()
    result = asyncio.run(scheduler([], executor).run(RUN))
    assert result.stop_reason is StopReason.COMPLETED
    assert result.states == ()
    assert executor.started == []


@pytest.mark.parametrize(
    "verdict",
    [DispatchOutcome.REVIEW_REQUIRED, DispatchOutcome.FAILED, RuntimeError(PRIVATE)],
)
def test_immediate_stop_verdict_prevents_not_yet_invoked_peer(verdict):
    executor = Executor()
    executor.outcomes["a"] = verdict
    result = asyncio.run(scheduler([action("a"), action("b")], executor).run(RUN))
    assert executor.started == ["a"]
    assert result.stop_reason in (
        StopReason.REVIEW_REQUIRED,
        StopReason.DISPATCH_FAILED,
    )
    assert dict(result.states)["b"] is ActionState.REVIEW_REQUIRED


def test_dispatcher_self_cancellation_stops_peer_before_invocation():
    class InterruptedExecutor(Executor):
        async def dispatch(self, run_id, node, cancellation):
            self.started.append(node.action_id)
            raise asyncio.CancelledError

    executor = InterruptedExecutor()
    result = asyncio.run(scheduler([action("a"), action("b")], executor).run(RUN))
    assert executor.started == ["a"]
    assert result.stop_reason is StopReason.REVIEW_REQUIRED
    assert all(state is ActionState.REVIEW_REQUIRED for _, state in result.states)


def test_terminal_failure_snapshot_includes_elapsed_time():
    now = [50]

    class FailingExecutor:
        async def dispatch(self, run_id, node, cancellation):
            now[0] += 20
            return DispatchOutcome.FAILED

    result = asyncio.run(
        scheduler([action("a")], FailingExecutor(), clock=lambda: now[0]).run(RUN)
    )
    assert result.stop_reason is StopReason.DISPATCH_FAILED
    assert result.elapsed_ns == 20


def test_elapsed_budget_crossing_during_completion_save_stops_descendant():
    now, executor = [0], Executor()

    def save(cp):
        if dict(cp.states)["a"] is ActionState.COMMITTED:
            now[0] = 1000

    worker = ActionScheduler(
        [action("a"), action("b", "a")],
        limits(),
        executor,
        save,
        clock=lambda: now[0],
    )
    result = asyncio.run(worker.run(RUN))
    assert result.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert result.elapsed_ns == 1000
    assert executor.started == ["a"]


def test_action_iterable_exception_is_categorical_without_private_context():
    def broken():
        yield action("a")
        raise RuntimeError(PRIVATE)

    with pytest.raises(SchedulerError) as exc:
        scheduler(broken())
    assert str(exc.value) == "invalid_actions"
    assert exc.value.__context__ is None
