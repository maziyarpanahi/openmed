"""Synthetic durable scheduler restart boundaries; no tools or PHI."""

import asyncio
import json
from dataclasses import replace

import pytest

from openmed.agent.action_graph import ActionNode
from openmed.agent.correlation import RunId
from openmed.agent.workflows.scheduler import (
    ActionScheduler,
    ActionState,
    DispatchOutcome,
    ScheduledAction,
    SchedulerCheckpoint,
    SchedulerError,
    SchedulerLimits,
    StopReason,
)

pytestmark = pytest.mark.integration
RUN = RunId.parse("run_" + "3" * 32)
TOOL = "tool:openmed.agent/synthetic"


def test_restart_at_every_snapshot_preserves_identities_and_stops_uncertain_work(
    tmp_path,
):
    nodes = [
        ScheduledAction(ActionNode("a", TOOL), True),
        ScheduledAction(ActionNode("b", TOOL, ("a",)), True),
    ]
    limits = SchedulerLimits(1, 2, 1, 2, 100, True)
    calls, saved = [], []
    path = tmp_path / "checkpoint.json"

    def save(cp):
        # Test-only synthetic store. Production supplies an atomic trusted store.
        path.write_text(cp.to_json())
        saved.append(SchedulerCheckpoint.from_dict(json.loads(path.read_text())))

    class Dispatcher:
        async def dispatch(self, run_id, node, cancellation):
            # Write-ahead snapshot is durable before executor invocation.
            cp = SchedulerCheckpoint.from_dict(json.loads(path.read_text()))
            assert dict(cp.states)[node.action_id] is ActionState.RUNNING
            calls.append((run_id, node.action_id))
            return DispatchOutcome.COMMITTED

    worker = ActionScheduler(nodes, limits, Dispatcher(), save, clock=lambda: 0)
    assert asyncio.run(worker.run(RUN)).stop_reason is StopReason.COMPLETED
    history = list(saved)
    for checkpoint in history:
        calls.clear()
        restored = ActionScheduler(
            list(reversed(nodes)), limits, Dispatcher(), save, clock=lambda: 50
        )
        result = asyncio.run(restored.run(RUN, checkpoint=checkpoint))
        uncertain = any(s is ActionState.RUNNING for _, s in checkpoint.states)
        if uncertain:
            assert result.stop_reason is StopReason.REVIEW_REQUIRED
            assert calls == []
        else:
            assert result.stop_reason is StopReason.COMPLETED
            expected = [
                (RUN, k) for k, s in checkpoint.states if s is ActionState.PENDING
            ]
            assert calls == expected
        assert tuple(k for k, _ in result.states) == ("a", "b")
        assert result.admitted_actions <= limits.max_actions
        assert result.reserved_units <= limits.max_total_units


def test_restart_retains_elapsed_budget_and_rejects_impossible_dependency_state():
    actions = [
        ScheduledAction(ActionNode("a", TOOL), True),
        ScheduledAction(ActionNode("b", TOOL, ("a",)), True),
    ]
    limits = SchedulerLimits(1, 2, 1, 2, 100, True)
    saved, calls = [], []

    class Dispatcher:
        async def dispatch(self, run_id, node, cancellation):
            calls.append(node.action_id)
            return DispatchOutcome.COMMITTED

    worker = ActionScheduler(
        actions, limits, Dispatcher(), saved.append, clock=lambda: 0
    )
    asyncio.run(worker.run(RUN))
    partial = next(
        cp
        for cp in saved
        if dict(cp.states) == {"a": ActionState.COMMITTED, "b": ActionState.PENDING}
    )
    calls.clear()
    expired = replace(partial, elapsed_ns=100)
    assert (
        asyncio.run(worker.run(RUN, checkpoint=expired)).stop_reason
        is StopReason.BUDGET_EXHAUSTED
    )
    assert calls == []
    impossible = replace(
        partial, states=(("a", ActionState.PENDING), ("b", ActionState.COMMITTED))
    )
    with pytest.raises(SchedulerError, match="dependency_mismatch"):
        asyncio.run(worker.run(RUN, checkpoint=impossible))
