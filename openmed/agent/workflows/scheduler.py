"""Bounded, process-local scheduling over identifier-only action graphs.

Authority checks, payload storage, tool execution and effect reconciliation
belong to the injected guarded dispatcher, never to this scheduler.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol

from openmed.agent.action_graph import (
    MAX_ACTION_GRAPH_NODES,
    ActionNode,
    validate_action_graph,
)
from openmed.agent.correlation import RunId


class SchedulerError(ValueError):
    """Controlled validation error that never retains rejected values."""


class ActionState(str, Enum):
    """Scheduling knowledge; RUNNING after restart requires reconciliation."""

    PENDING = "pending"
    RUNNING = "running"
    COMMITTED = "committed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"


class StopReason(str, Enum):
    """Closed, content-free reasons for halting admission."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REVIEW_REQUIRED = "review_required"
    DISPATCH_FAILED = "dispatch_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CLOCK_FAILED = "clock_failed"
    CHECKPOINT_FAILED = "checkpoint_failed"


class DispatchOutcome(str, Enum):
    """Guarded-dispatch verdict; COMMITTED includes protected input storage."""

    COMMITTED = "committed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"


class CancellationSignal:
    """Cooperative run stop signal, confined to one asyncio event loop."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def cancelled(self) -> bool:
        """Return whether admission and protected dispatch must stop."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Request cancellation; this cannot undo an already committed effect."""
        self._event.set()

    async def wait(self) -> None:
        """Wait until cancellation has been requested."""
        await self._event.wait()


class GuardedDispatcher(Protocol):
    """Application-owned executor with authority and durable commit guards.

    Implementations must check cancellation immediately before protected work,
    enforce approvals, bind run/action identities to their exact arguments, and
    return COMMITTED only after inputs for descendants are available. Exceptions
    and interrupted calls are uncertain, never proof that effects did not occur.
    """

    async def dispatch(
        self, run_id: RunId, node: ActionNode, cancellation: CancellationSignal
    ) -> DispatchOutcome:
        """Execute one guarded action using application-held payloads."""
        ...


def _integer(value: Any, *, positive: bool = False) -> None:
    if type(value) is not int or value < (1 if positive else 0):
        raise SchedulerError("invalid_integer")


@dataclass(frozen=True, slots=True)
class ScheduledAction:
    """Trusted scheduling metadata, separate from authority or tool payloads.

    Args:
        node: Existing identifier-only action-graph node.
        read_only: Trusted policy classification; effects are the default.
        resource_units: Positive worst-case reservation in host-defined units.
    """

    node: ActionNode
    read_only: bool = False
    resource_units: int = 1

    def __post_init__(self) -> None:
        if type(self.node) is not ActionNode or type(self.read_only) is not bool:
            raise SchedulerError("invalid_action")
        _integer(self.resource_units, positive=True)


@dataclass(frozen=True, slots=True)
class SchedulerLimits:
    """Finite admission limits; reservations count even on failure or restart.

    Args:
        max_concurrency: Maximum simultaneously dispatched actions.
        max_actions: Maximum dispatch admissions across this run's checkpoints.
        max_inflight_units: Maximum overlapping worst-case resource reservations.
        max_total_units: Maximum cumulative reservations, never refunded.
        max_elapsed_ns: Cooperative elapsed-time admission limit across restarts.
        parallel_reads: Explicit policy permitting overlapping read-only actions.

    Resource units are scheduling estimates, not measurement of model memory,
    artifact storage or tool calls. The guarded runtime owns those limits.
    """

    max_concurrency: int
    max_actions: int
    max_inflight_units: int
    max_total_units: int
    max_elapsed_ns: int
    parallel_reads: bool = False

    def __post_init__(self) -> None:
        for value in (
            self.max_concurrency,
            self.max_actions,
            self.max_inflight_units,
            self.max_total_units,
            self.max_elapsed_ns,
        ):
            _integer(value, positive=True)
        if type(self.parallel_reads) is not bool:
            raise SchedulerError("invalid_policy")


@dataclass(frozen=True, slots=True)
class SchedulerCheckpoint:
    """Metadata-only snapshot to persist in a host-owned trusted atomic store.

    The plan digest binds identities, dependencies, reservations and policy.
    This snapshot is not a signed authority or an effect-recovery proof. The host
    must protect freshness and integrity, and reconcile uncertain work through
    existing recovery contracts before supplying a resumable snapshot.
    """

    run_id: RunId
    plan_digest: str
    states: tuple[tuple[str, ActionState], ...]
    admitted_actions: int
    reserved_units: int
    elapsed_ns: int
    stop_reason: StopReason | None = None

    def __post_init__(self) -> None:
        if type(self.run_id) is not RunId:
            raise SchedulerError("invalid_run_id")
        if (
            type(self.plan_digest) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.plan_digest) is None
        ):
            raise SchedulerError("invalid_plan_digest")
        if type(self.states) is not tuple or len(self.states) > MAX_ACTION_GRAPH_NODES:
            raise SchedulerError("invalid_states")
        ids = []
        for entry in self.states:
            if type(entry) is not tuple or len(entry) != 2:
                raise SchedulerError("invalid_state")
            action_id, state = entry
            if (
                type(action_id) is not str
                or re.fullmatch(
                    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?", action_id
                )
                is None
                or type(state) is not ActionState
            ):
                raise SchedulerError("invalid_state")
            ids.append(action_id)
        if ids != sorted(set(ids)):
            raise SchedulerError("invalid_state_order")
        for value in (self.admitted_actions, self.reserved_units, self.elapsed_ns):
            _integer(value)
        if self.stop_reason is not None and type(self.stop_reason) is not StopReason:
            raise SchedulerError("invalid_stop_reason")

    def to_dict(self) -> dict[str, Any]:
        """Return controlled metadata; no absolute clock or payload fields."""
        return {
            "schema_version": "openmed.agent.scheduler.v1",
            "run_id": self.run_id.serialize(),
            "plan_digest": self.plan_digest,
            "states": [[key, value.value] for key, value in self.states],
            "admitted_actions": self.admitted_actions,
            "reserved_units": self.reserved_units,
            "elapsed_ns": self.elapsed_ns,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
        }

    def to_json(self) -> str:
        """Return stable compact JSON for host checkpoint storage."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SchedulerCheckpoint:
        """Restore an exact metadata snapshot, without retaining rejected data."""
        try:
            if (
                set(payload)
                != {
                    "schema_version",
                    "run_id",
                    "plan_digest",
                    "states",
                    "admitted_actions",
                    "reserved_units",
                    "elapsed_ns",
                    "stop_reason",
                }
                or payload["schema_version"] != "openmed.agent.scheduler.v1"
            ):
                raise SchedulerError("invalid_checkpoint")
            if (
                type(payload["states"]) is not list
                or len(payload["states"]) > MAX_ACTION_GRAPH_NODES
            ):
                raise SchedulerError("invalid_checkpoint")
            states = []
            for entry in payload["states"]:
                if type(entry) is not list or len(entry) != 2:
                    raise SchedulerError("invalid_checkpoint")
                states.append((entry[0], ActionState(entry[1])))
            return cls(
                RunId.parse(payload["run_id"]),
                payload["plan_digest"],
                tuple(states),
                payload["admitted_actions"],
                payload["reserved_units"],
                payload["elapsed_ns"],
                StopReason(payload["stop_reason"])
                if payload["stop_reason"] is not None
                else None,
            )
        except Exception:
            pass
        raise SchedulerError("invalid_checkpoint")


class ActionScheduler:
    """Schedule a validated DAG in one event loop through guarded dispatch.

    Args:
        actions: Bounded synthetic or trusted identifier-only graph metadata.
        limits: Explicit scheduling budgets and read concurrency policy.
        dispatcher: Injected executor owning authority, tool execution and commits.
        save_checkpoint: Atomic durable host callback, called before each dispatch
            and after completions, before any descendants can be admitted.
        clock: Injected non-negative monotonic nanosecond clock.

    No automatic retries occur. A stopped checkpoint cannot restart admission;
    in-flight restart state requires human review and effect reconciliation.
    """

    def __init__(
        self,
        actions: Iterable[ScheduledAction],
        limits: SchedulerLimits,
        dispatcher: GuardedDispatcher,
        save_checkpoint: Callable[[SchedulerCheckpoint], None],
        *,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if type(limits) is not SchedulerLimits:
            raise SchedulerError("invalid_limits")
        collected: list[ScheduledAction] = []
        invalid_code = None
        try:
            for action in actions:
                if type(action) is not ScheduledAction:
                    invalid_code = "invalid_action"
                    break
                if len(collected) == MAX_ACTION_GRAPH_NODES:
                    invalid_code = "too_many_actions"
                    break
                collected.append(action)
        except Exception:
            invalid_code = "invalid_actions"
        if invalid_code is not None:
            # A hostile or failing iterable must not leak its exception payload.
            raise SchedulerError(invalid_code)
        graph = validate_action_graph(action.node for action in collected)
        if not graph.is_valid:
            raise SchedulerError("invalid_graph")
        self._actions = {a.node.action_id: a for a in collected}
        self._limits = limits
        self._dispatcher = dispatcher
        self._save = save_checkpoint
        self._clock = clock
        plan = {
            "limits": asdict(limits),
            "actions": [
                {
                    "node": {
                        **a.node.to_dict(),
                        "depends_on": sorted(a.node.depends_on),
                    },
                    "read_only": a.read_only,
                    "resource_units": a.resource_units,
                }
                for _, a in sorted(self._actions.items())
            ],
        }
        self._digest = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        self._active = False

    async def run(
        self,
        run_id: RunId,
        *,
        cancellation: CancellationSignal | None = None,
        checkpoint: SchedulerCheckpoint | None = None,
    ) -> SchedulerCheckpoint:
        """Run or resume trusted state and return the final metadata checkpoint.

        Token cancellation returns a stopped checkpoint. Cancelling the asyncio
        task drains dispatch tasks, saves uncertain states, then propagates
        CancelledError. Non-cooperative dispatchers can delay draining; no hard
        deadline or rollback of external effects is claimed.

        Args:
            run_id: Stable application-owned run identity, preserved on restart.
            cancellation: Optional cooperative stop signal for this event loop.
            checkpoint: Fresh, integrity-protected host snapshot of the same plan.

        Returns:
            Final checkpoint with action states, charged budgets and a stop code.

        Raises:
            SchedulerError: If identities, policy or checkpoint invariants fail.
            asyncio.CancelledError: After caller task cancellation and cleanup.
        """
        if self._active:
            raise SchedulerError("scheduler_already_running")
        if type(run_id) is not RunId:
            raise SchedulerError("invalid_run_id")
        if cancellation is not None and type(cancellation) is not CancellationSignal:
            raise SchedulerError("invalid_cancellation")
        signal = cancellation if cancellation is not None else CancellationSignal()
        if checkpoint is not None:
            self._validate_restore(checkpoint, run_id)
        state = _Run(self, run_id, signal, checkpoint)
        self._active = True
        try:
            return await state.execute()
        finally:
            self._active = False

    def _validate_restore(self, checkpoint: SchedulerCheckpoint, run_id: RunId) -> None:
        if (
            type(checkpoint) is not SchedulerCheckpoint
            or checkpoint.run_id != run_id
            or checkpoint.plan_digest != self._digest
            or set(dict(checkpoint.states)) != set(self._actions)
        ):
            raise SchedulerError("checkpoint_identity_mismatch")
        states = dict(checkpoint.states)
        admitted = [
            self._actions[k] for k, v in states.items() if v != ActionState.PENDING
        ]
        if (
            checkpoint.admitted_actions != len(admitted)
            or checkpoint.reserved_units != sum(a.resource_units for a in admitted)
            or checkpoint.admitted_actions > self._limits.max_actions
            or checkpoint.reserved_units > self._limits.max_total_units
        ):
            raise SchedulerError("checkpoint_budget_mismatch")
        for key, status in states.items():
            if status != ActionState.PENDING and any(
                states[p] != ActionState.COMMITTED
                for p in self._actions[key].node.depends_on
            ):
                raise SchedulerError("checkpoint_dependency_mismatch")
        if checkpoint.stop_reason is StopReason.COMPLETED and any(
            status is not ActionState.COMMITTED for status in states.values()
        ):
            raise SchedulerError("checkpoint_completion_mismatch")


class _Run:
    def __init__(
        self,
        owner: ActionScheduler,
        run_id: RunId,
        signal: CancellationSignal,
        checkpoint: SchedulerCheckpoint | None,
    ) -> None:
        self.owner = owner
        self.run_id = run_id
        self.signal = signal
        self.states = (
            dict(checkpoint.states)
            if checkpoint
            else {key: ActionState.PENDING for key in owner._actions}
        )
        self.admitted = checkpoint.admitted_actions if checkpoint else 0
        self.reserved = checkpoint.reserved_units if checkpoint else 0
        self.elapsed = checkpoint.elapsed_ns if checkpoint else 0
        self.base_elapsed = self.elapsed
        self.start: int | None = None
        self.last_clock: int | None = None
        self.reason = checkpoint.stop_reason if checkpoint else None
        if self.reason is None and any(
            state not in (ActionState.PENDING, ActionState.COMMITTED)
            for state in self.states.values()
        ):
            self.reason = StopReason.REVIEW_REQUIRED
        self.tasks: dict[str, asyncio.Task[DispatchOutcome]] = {}

    def snapshot(self) -> SchedulerCheckpoint:
        return SchedulerCheckpoint(
            self.run_id,
            self.owner._digest,
            tuple(sorted(self.states.items())),
            self.admitted,
            self.reserved,
            self.elapsed,
            self.reason,
        )

    def save(self) -> None:
        try:
            self.owner._save(self.snapshot())
        except Exception:
            self.reason = StopReason.CHECKPOINT_FAILED

    def update_elapsed(self) -> None:
        try:
            now = self.owner._clock()
            _integer(now)
            if self.last_clock is not None and now < self.last_clock:
                raise SchedulerError("clock_regressed")
            if self.start is None:
                self.start = now
            self.last_clock = now
            self.elapsed = self.base_elapsed + now - self.start
        except Exception:
            if self.reason in (None, StopReason.COMPLETED):
                self.reason = StopReason.CLOCK_FAILED
        if (
            self.reason is StopReason.COMPLETED
            and self.elapsed >= self.owner._limits.max_elapsed_ns
        ):
            self.reason = StopReason.BUDGET_EXHAUSTED

    def check(self) -> None:
        if self.reason is not None:
            return
        if self.signal.cancelled:
            self.reason = StopReason.CANCELLED
            return
        self.update_elapsed()
        if self.reason is None and self.elapsed >= self.owner._limits.max_elapsed_ns:
            self.reason = StopReason.BUDGET_EXHAUSTED

    async def dispatch(self, action: ScheduledAction) -> DispatchOutcome:
        if self.signal.cancelled:
            return DispatchOutcome.REVIEW_REQUIRED
        try:
            result = await self.owner._dispatcher.dispatch(
                self.run_id, action.node, self.signal
            )
        except (Exception, asyncio.CancelledError):
            result = DispatchOutcome.REVIEW_REQUIRED
        if type(result) is not DispatchOutcome:
            result = DispatchOutcome.REVIEW_REQUIRED
        if result is not DispatchOutcome.COMMITTED:
            # Signal before yielding back to the scheduler: other admitted
            # coroutines may not yet have reached their invocation boundary.
            if self.reason is None:
                self.reason = (
                    StopReason.DISPATCH_FAILED
                    if result is DispatchOutcome.FAILED
                    else StopReason.REVIEW_REQUIRED
                )
            self.signal.cancel()
        return result

    def finish(self, key: str, result: object) -> None:
        if type(result) is DispatchOutcome:
            self.states[key] = ActionState(result.value)
            if self.reason is None:
                if result is DispatchOutcome.FAILED:
                    self.reason = StopReason.DISPATCH_FAILED
                elif result is DispatchOutcome.REVIEW_REQUIRED:
                    self.reason = StopReason.REVIEW_REQUIRED
        else:
            self.states[key] = ActionState.REVIEW_REQUIRED
            if self.reason is None:
                self.reason = StopReason.REVIEW_REQUIRED

    async def drain(self) -> None:
        self.signal.cancel()
        keys = sorted(self.tasks)
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        results = await asyncio.gather(
            *(self.tasks[key] for key in keys), return_exceptions=True
        )
        for key, result in zip(keys, results):
            self.finish(key, result)
        self.tasks.clear()
        self.update_elapsed()
        self.save()

    async def execute(self) -> SchedulerCheckpoint:
        try:
            while self.reason is None:
                self.check()
                # Collect every finished peer before admitting any descendant.
                for key in sorted(tuple(self.tasks)):
                    task = self.tasks[key]
                    if task.done():
                        try:
                            result = task.result()
                        except (Exception, asyncio.CancelledError):
                            result = None
                        self.finish(key, result)
                        del self.tasks[key]
                        self.save()
                        self.check()
                if self.reason is not None:
                    break
                if all(s is ActionState.COMMITTED for s in self.states.values()):
                    self.reason = StopReason.COMPLETED
                    break
                self.admit()
                if self.reason is None and self.tasks:
                    await asyncio.wait(
                        self.tasks.values(),
                        timeout=0.01,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
            if self.tasks:
                await self.drain()
            else:
                self.update_elapsed()
                self.save()
            return self.snapshot()
        except asyncio.CancelledError:
            self.reason = StopReason.CANCELLED
            await self.drain()
            raise

    def admit(self) -> None:
        limits = self.owner._limits
        for key, action in sorted(self.owner._actions.items()):
            self.check()
            if self.reason is not None:
                return
            if self.states[key] is not ActionState.PENDING or any(
                self.states[p] is not ActionState.COMMITTED
                for p in action.node.depends_on
            ):
                continue
            active = [self.owner._actions[k] for k in self.tasks]
            cap = limits.max_concurrency if limits.parallel_reads else 1
            if len(active) >= cap:
                return
            # Effects are exclusive against reads and other effects. A ready
            # effect is a barrier so a stream of later reads cannot starve it.
            if active and (
                not action.read_only or any(not a.read_only for a in active)
            ):
                return
            units = sum(a.resource_units for a in active)
            if units + action.resource_units > limits.max_inflight_units:
                if not active:
                    self.reason = StopReason.BUDGET_EXHAUSTED
                    return
                continue
            if (
                self.admitted >= limits.max_actions
                or self.reserved + action.resource_units > limits.max_total_units
            ):
                if active:
                    return
                self.reason = StopReason.BUDGET_EXHAUSTED
                return
            self.states[key] = ActionState.RUNNING
            self.admitted += 1
            self.reserved += action.resource_units
            # Write-ahead admission: persistence failure never invokes dispatch.
            self.save()
            self.check()
            if self.reason is not None:
                return
            self.tasks[key] = asyncio.create_task(self.dispatch(action))


__all__ = [
    "ActionScheduler",
    "ActionState",
    "CancellationSignal",
    "DispatchOutcome",
    "GuardedDispatcher",
    "ScheduledAction",
    "SchedulerCheckpoint",
    "SchedulerError",
    "SchedulerLimits",
    "StopReason",
]
