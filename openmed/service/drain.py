"""Coordinated, bounded drain for the OpenMed service lifecycle.

One :class:`DrainCoordinator` owns the shutdown of a service process. Lifecycle
participants register with it and take part in four ordered phases, each bound
by the configured budget:

1. ``stop_admission``: readiness drops first, then every surface stops admitting
   new work (REST, GraphQL, gRPC, jobs, callbacks).
2. ``finish_in_flight``: accepted work finishes or reaches a declared terminal
   state. Participants run concurrently against one deadline. Streams and jobs
   still running near the end of the window are told to stop at their next safe
   boundary (stream event or job document) so they end with a typed, retryable
   result instead of being dropped.
3. ``flush``: queues, callbacks, and exporters flush what remains.
4. ``release``: model references, accelerator memory, servers, and executors are
   returned to zero or to a declared retained baseline.

Hooks never receive or report request bodies, stream fragments, job payloads,
model content, or identifiers. Evidence is limited to controlled participant
categories, phases, outcomes, aggregate counts, durations, and reason codes.
Liveness stays true while the drain runs; only readiness changes.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import re
import sys
import threading
import weakref
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

SHUTTING_DOWN = "shutting_down"
DRAIN_DEADLINE_EXCEEDED = "drain_deadline_exceeded"
PARTICIPANT_FAILED = "participant_failed"
DRAIN_REASON_CODES = frozenset(
    {SHUTTING_DOWN, DRAIN_DEADLINE_EXCEEDED, PARTICIPANT_FAILED}
)

DEFAULT_DRAIN_SECONDS = 30.0
DEFAULT_RELEASE_SECONDS = 5.0
DEFAULT_SETTLE_SECONDS = 1.0
DEFAULT_RETRY_AFTER_SECONDS = 1.0
SHUTTING_DOWN_MESSAGE = "Service is shutting down; retry the request"

_SETTLE_FRACTION = 0.25
_HOOK_GRACE_SECONDS = 0.05
_BLOCKING_FLOOR_SECONDS = 0.01
_POLL_SECONDS = 0.01
_MAX_COUNT = 1_000_000_000
_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
_LOGGER = logging.getLogger(__name__)


class DrainPhase(str, Enum):
    """Ordered drain phases."""

    STOP_ADMISSION = "stop_admission"
    FINISH_IN_FLIGHT = "finish_in_flight"
    FLUSH = "flush"
    RELEASE = "release"


DRAIN_PHASES: tuple[DrainPhase, ...] = tuple(DrainPhase)


class DrainOutcome(str, Enum):
    """Controlled outcome for one participant phase or a whole drain."""

    COMPLETED = "completed"
    CHECKPOINTED = "checkpointed"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    FAILED = "failed"
    SKIPPED = "skipped"


_OUTCOME_SEVERITY = {
    DrainOutcome.SKIPPED: 0,
    DrainOutcome.COMPLETED: 1,
    DrainOutcome.CHECKPOINTED: 2,
    DrainOutcome.CANCELLED: 3,
    DrainOutcome.DEADLINE_EXCEEDED: 4,
    DrainOutcome.FAILED: 5,
}


class DrainState(str, Enum):
    """Coarse lifecycle state exposed by a coordinator."""

    SERVING = "serving"
    DRAINING = "draining"
    DRAINED = "drained"


class ServiceShuttingDownError(RuntimeError):
    """Typed, retryable refusal raised once a surface stops admitting work.

    Args:
        message: Fixed, value-free message.
        retry_after_seconds: Hint for when a client may retry elsewhere.
    """

    reason_code = SHUTTING_DOWN

    def __init__(
        self,
        message: str = SHUTTING_DOWN_MESSAGE,
        *,
        retry_after_seconds: float = DEFAULT_RETRY_AFTER_SECONDS,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = _non_negative_seconds(retry_after_seconds)

    def to_details(self) -> dict[str, Any]:
        """Return the value-free retry contract shared by every surface."""
        return shutting_down_details(self.retry_after_seconds)


def shutting_down_details(retry_after_seconds: float) -> dict[str, Any]:
    """Return the value-free details attached to a shutdown refusal.

    Args:
        retry_after_seconds: Retry hint in seconds.

    Returns:
        Mapping with the reason code, retryability, and retry hint.
    """
    return {
        "reason": SHUTTING_DOWN,
        "retryable": True,
        "retry_after_seconds": _non_negative_seconds(retry_after_seconds),
    }


def shutting_down_ndjson_line(
    retry_after_seconds: float = DEFAULT_RETRY_AFTER_SECONDS,
) -> str:
    """Return the typed terminal NDJSON event for a stream ended by a drain.

    Args:
        retry_after_seconds: Retry hint in seconds.

    Returns:
        One newline-terminated JSON object with a ``shutting_down`` error code.
    """
    payload = {
        "type": "error",
        "error": {
            "code": SHUTTING_DOWN,
            "message": SHUTTING_DOWN_MESSAGE,
            "details": {
                "retryable": True,
                "retry_after_seconds": _non_negative_seconds(retry_after_seconds),
            },
        },
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"


@dataclass(frozen=True)
class DrainBudget:
    """Bounded time budget for one drain.

    Attributes:
        drain_seconds: Window for stopping admission and finishing in-flight work.
        release_seconds: Window, after in-flight work ends, for flush and release.
        settle_seconds: Tail of the in-flight window reserved for streams and jobs
            to reach a declared terminal state after they are asked to stop.
            Defaults to ``min(1.0, drain_seconds / 4)``.
    """

    drain_seconds: float = DEFAULT_DRAIN_SECONDS
    release_seconds: float = DEFAULT_RELEASE_SECONDS
    settle_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        drain = _non_negative_seconds(self.drain_seconds)
        release = _non_negative_seconds(self.release_seconds)
        settle = (
            min(DEFAULT_SETTLE_SECONDS, drain * _SETTLE_FRACTION)
            if self.settle_seconds is None
            else min(_non_negative_seconds(self.settle_seconds), drain)
        )
        object.__setattr__(self, "drain_seconds", drain)
        object.__setattr__(self, "release_seconds", release)
        object.__setattr__(self, "settle_seconds", settle)

    @property
    def total_seconds(self) -> float:
        """Return the hard bound across every phase, excluding hook grace."""
        return self.drain_seconds + self.release_seconds

    def to_dict(self) -> dict[str, float]:
        """Return the budget as plain numbers."""
        return {
            "drain_seconds": self.drain_seconds,
            "release_seconds": self.release_seconds,
            "settle_seconds": float(self.settle_seconds or 0.0),
        }


@dataclass(frozen=True)
class PhaseResult:
    """Value-free result returned by one participant hook.

    Attributes:
        outcome: Controlled outcome for this participant and phase.
        completed: Units of work that committed or ended normally.
        cancelled: Units explicitly cancelled with a retryable result.
        checkpointed: Units durably checkpointed for resume.
        unfinished: Units still outstanding when the hook returned.
        reason_code: Optional controlled reason code.
    """

    outcome: DrainOutcome = DrainOutcome.COMPLETED
    completed: int = 0
    cancelled: int = 0
    checkpointed: int = 0
    unfinished: int = 0
    reason_code: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", DrainOutcome(self.outcome))
        for name in ("completed", "cancelled", "checkpointed", "unfinished"):
            object.__setattr__(self, name, _bounded_count(getattr(self, name)))
        if self.reason_code is not None and self.reason_code not in (
            DRAIN_REASON_CODES
        ):
            raise ValueError("drain reason code is not in the controlled vocabulary")

    @classmethod
    def from_counts(
        cls,
        *,
        completed: int = 0,
        cancelled: int = 0,
        checkpointed: int = 0,
        unfinished: int = 0,
    ) -> "PhaseResult":
        """Derive the outcome and reason code from aggregate counts.

        Args:
            completed: Units that ended normally.
            cancelled: Units cancelled with a retryable result.
            checkpointed: Units checkpointed for resume.
            unfinished: Units still outstanding.

        Returns:
            A result whose outcome is the most severe one the counts imply.
        """
        if unfinished > 0:
            outcome, reason = DrainOutcome.DEADLINE_EXCEEDED, DRAIN_DEADLINE_EXCEEDED
        elif cancelled > 0:
            outcome, reason = DrainOutcome.CANCELLED, SHUTTING_DOWN
        elif checkpointed > 0:
            outcome, reason = DrainOutcome.CHECKPOINTED, SHUTTING_DOWN
        else:
            outcome, reason = DrainOutcome.COMPLETED, None
        return cls(
            outcome=outcome,
            completed=completed,
            cancelled=cancelled,
            checkpointed=checkpointed,
            unfinished=unfinished,
            reason_code=reason,
        )


@dataclass(frozen=True)
class DrainEvent:
    """One value-free progress record for a participant phase."""

    phase: DrainPhase
    participant: str
    outcome: DrainOutcome
    completed: int
    cancelled: int
    checkpointed: int
    unfinished: int
    budget_seconds: float
    duration_seconds: float
    reason_code: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic representation containing no work values."""
        return {
            "phase": self.phase.value,
            "participant": self.participant,
            "outcome": self.outcome.value,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "checkpointed": self.checkpointed,
            "unfinished": self.unfinished,
            "budget_seconds": round(self.budget_seconds, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class DrainReport:
    """Aggregate, value-free evidence for one completed drain."""

    outcome: DrainOutcome
    budget: DrainBudget
    duration_seconds: float
    events: tuple[DrainEvent, ...]

    @property
    def unfinished_participants(self) -> tuple[str, ...]:
        """Return categories that still had outstanding work at their deadline."""
        return tuple(
            sorted({event.participant for event in self.events if event.unfinished})
        )

    def events_for(self, participant: str) -> tuple[DrainEvent, ...]:
        """Return the events recorded for one participant category."""
        return tuple(event for event in self.events if event.participant == participant)

    def totals(self) -> dict[str, int]:
        """Return aggregate counts across every participant."""
        return {
            name: sum(getattr(event, name) for event in self.events)
            for name in ("completed", "cancelled", "checkpointed", "unfinished")
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, value-free evidence document."""
        return {
            "outcome": self.outcome.value,
            "budget": self.budget.to_dict(),
            "duration_seconds": round(self.duration_seconds, 3),
            "totals": self.totals(),
            "unfinished_participants": list(self.unfinished_participants),
            "events": [event.to_dict() for event in self.events],
        }


@dataclass
class DrainContext:
    """Deadline information handed to one participant hook.

    Attributes:
        phase: Phase being executed.
        deadline: Event-loop time by which the hook must return.
        settle_deadline: Point after which streams and jobs are asked to stop.
            Equal to ``deadline`` outside the in-flight phase.
        retry_after_seconds: Retry hint attached to shutdown refusals.
        budget: Budget of the drain this hook belongs to.
    """

    phase: DrainPhase
    deadline: float
    settle_deadline: float
    retry_after_seconds: float = DEFAULT_RETRY_AFTER_SECONDS
    budget: DrainBudget = field(default_factory=DrainBudget)

    def remaining(self) -> float:
        """Return seconds left before this hook's deadline."""
        return max(self.deadline - asyncio.get_running_loop().time(), 0.0)

    async def wait_until(
        self,
        predicate: Callable[[], bool],
        *,
        deadline: Optional[float] = None,
    ) -> bool:
        """Poll ``predicate`` until it is true or the deadline passes.

        Args:
            predicate: Cheap, non-blocking condition.
            deadline: Optional earlier deadline in event-loop time.

        Returns:
            Whether the predicate held before the deadline.
        """
        limit = self.deadline if deadline is None else min(deadline, self.deadline)
        loop = asyncio.get_running_loop()
        while True:
            if predicate():
                return True
            remaining = limit - loop.time()
            if remaining <= 0:
                return bool(predicate())
            await asyncio.sleep(min(_POLL_SECONDS, remaining))

    async def run_blocking(self, operation: Callable[[], Any]) -> Any:
        """Run blocking cleanup in a daemon thread bounded by this deadline.

        The thread is a daemon so a wedged device or server cannot keep the
        process alive after the hard deadline.

        Args:
            operation: Blocking callable with no arguments.

        Returns:
            The callable's result.

        Raises:
            asyncio.TimeoutError: When the deadline passes first.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        def resolve(error: Optional[BaseException], value: Any) -> None:
            if future.done():
                return
            if error is None:
                future.set_result(value)
            else:
                future.set_exception(error)

        def target() -> None:
            try:
                value = operation()
            except BaseException as exc:  # noqa: BLE001 - re-raised on the loop
                outcome: tuple[Optional[BaseException], Any] = (exc, None)
            else:
                outcome = (None, value)
            try:
                loop.call_soon_threadsafe(resolve, *outcome)
            except RuntimeError:
                pass

        threading.Thread(target=target, name="openmed-drain", daemon=True).start()
        # A tiny floor lets an already-finished operation report success even
        # when the budget is zero; the coordinator's hook bound still applies.
        timeout = max(self.remaining(), _BLOCKING_FLOOR_SECONDS)
        return await asyncio.wait_for(asyncio.shield(future), timeout)


class DrainParticipant:
    """Base class for lifecycle participants.

    Subclasses override the phase hooks they take part in. A hook may be sync or
    async, receives a :class:`DrainContext`, and returns a :class:`PhaseResult`
    or ``None`` (recorded as skipped). Hooks must respect ``ctx.deadline``; the
    coordinator also cancels an async hook shortly after it.

    Args:
        category: Controlled lowercase label such as ``"job_queue"``. It is used
            as a metric label and must never carry request or model values.
        priority: Ordering within sequential phases; lower runs first.
    """

    category: str = "participant"
    priority: int = 100

    def __init__(
        self,
        category: Optional[str] = None,
        *,
        priority: Optional[int] = None,
    ) -> None:
        if category is not None:
            self.category = category
        if priority is not None:
            self.priority = int(priority)
        validate_drain_category(self.category)

    def handles(self, phase: DrainPhase) -> bool:
        """Return whether this participant takes part in ``phase``."""
        return getattr(type(self), phase.value) is not getattr(
            DrainParticipant, phase.value
        )

    def stop_admission(self, ctx: DrainContext) -> Any:
        """Stop admitting new work. Must not block."""
        return None

    def finish_in_flight(self, ctx: DrainContext) -> Any:
        """Finish, checkpoint, or explicitly cancel accepted work."""
        return None

    def flush(self, ctx: DrainContext) -> Any:
        """Flush queues, callbacks, or exporters."""
        return None

    def release(self, ctx: DrainContext) -> Any:
        """Release references, devices, servers, and executors."""
        return None

    def outstanding(self) -> int:
        """Return outstanding work units for deadline evidence."""
        return 0


class CallbackDrainParticipant(DrainParticipant):
    """Participant assembled from optional per-phase callbacks.

    Each callback receives the :class:`DrainContext` and may return a
    :class:`PhaseResult`, ``None``, or an awaitable of either.

    Args:
        category: Controlled participant label.
        priority: Ordering within sequential phases.
        stop_admission: Optional admission hook.
        finish_in_flight: Optional in-flight hook.
        flush: Optional flush hook.
        release: Optional release hook.
        outstanding: Optional zero-argument count of outstanding work.
    """

    def __init__(
        self,
        category: str,
        *,
        priority: int = 100,
        stop_admission: Optional[Callable[[DrainContext], Any]] = None,
        finish_in_flight: Optional[Callable[[DrainContext], Any]] = None,
        flush: Optional[Callable[[DrainContext], Any]] = None,
        release: Optional[Callable[[DrainContext], Any]] = None,
        outstanding: Optional[Callable[[], int]] = None,
    ) -> None:
        super().__init__(category, priority=priority)
        self._hooks = {
            phase: hook
            for phase, hook in (
                (DrainPhase.STOP_ADMISSION, stop_admission),
                (DrainPhase.FINISH_IN_FLIGHT, finish_in_flight),
                (DrainPhase.FLUSH, flush),
                (DrainPhase.RELEASE, release),
            )
            if hook is not None
        }
        self._outstanding = outstanding

    def handles(self, phase: DrainPhase) -> bool:
        """Return whether a callback was supplied for ``phase``."""
        return phase in self._hooks

    async def _call(self, phase: DrainPhase, ctx: DrainContext) -> Any:
        result = self._hooks[phase](ctx)
        if inspect.isawaitable(result):
            result = await result
        return result if isinstance(result, PhaseResult) else PhaseResult()

    async def stop_admission(self, ctx: DrainContext) -> Any:
        """Run the supplied admission callback."""
        return await self._call(DrainPhase.STOP_ADMISSION, ctx)

    async def finish_in_flight(self, ctx: DrainContext) -> Any:
        """Run the supplied in-flight callback."""
        return await self._call(DrainPhase.FINISH_IN_FLIGHT, ctx)

    async def flush(self, ctx: DrainContext) -> Any:
        """Run the supplied flush callback."""
        return await self._call(DrainPhase.FLUSH, ctx)

    async def release(self, ctx: DrainContext) -> Any:
        """Run the supplied release callback."""
        return await self._call(DrainPhase.RELEASE, ctx)

    def outstanding(self) -> int:
        """Return the supplied outstanding count, if any."""
        return 0 if self._outstanding is None else int(self._outstanding())


class DrainRegistrationError(RuntimeError):
    """Raised when a participant is registered after a drain started."""


class DrainCoordinator:
    """Run one bounded, ordered drain across registered participants.

    The coordinator is single-shot: concurrent or repeated :meth:`drain` calls
    share one run and return the same report, so a repeated signal never drains
    work twice.

    Args:
        budget: Phase budget; defaults to :class:`DrainBudget`.
        metrics: Optional registry exposing ``record_drain_event`` and
            ``set_draining``.
        retry_after_seconds: Retry hint attached to shutdown refusals.
    """

    def __init__(
        self,
        budget: Optional[DrainBudget] = None,
        *,
        metrics: Optional[Any] = None,
        retry_after_seconds: float = DEFAULT_RETRY_AFTER_SECONDS,
    ) -> None:
        self.budget = budget or DrainBudget()
        self.metrics = metrics
        self.retry_after_seconds = _non_negative_seconds(retry_after_seconds)
        self._participants: list[DrainParticipant] = []
        self._readiness_callbacks: list[Callable[[], Any]] = []
        self._listeners: list[Callable[[DrainEvent], Any]] = []
        self._events: list[DrainEvent] = []
        self._state = DrainState.SERVING
        self._task: Optional[asyncio.Future[DrainReport]] = None
        self._report: Optional[DrainReport] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> DrainState:
        """Return the coarse lifecycle state."""
        return self._state

    @property
    def ready(self) -> bool:
        """Return whether the coordinator still advertises readiness."""
        return self._state is DrainState.SERVING

    @property
    def live(self) -> bool:
        """Return liveness, which stays true through a bounded drain."""
        return True

    @property
    def report(self) -> Optional[DrainReport]:
        """Return the finished report, if the drain completed."""
        return self._report

    @property
    def events(self) -> tuple[DrainEvent, ...]:
        """Return progress events recorded so far."""
        return tuple(self._events)

    def participants(self) -> tuple[DrainParticipant, ...]:
        """Return registered participants in priority order."""
        return tuple(sorted(self._participants, key=lambda item: item.priority))

    def register(self, participant: DrainParticipant) -> DrainParticipant:
        """Register a participant; registering the same object twice is a no-op.

        Args:
            participant: Participant to add.

        Returns:
            The registered participant.

        Raises:
            DrainRegistrationError: When the drain already started.
            TypeError: When ``participant`` is not a :class:`DrainParticipant`.
        """
        if not isinstance(participant, DrainParticipant):
            raise TypeError("drain participants must subclass DrainParticipant")
        validate_drain_category(participant.category)
        with self._lock:
            if self._state is not DrainState.SERVING:
                raise DrainRegistrationError("drain already started")
            if not any(item is participant for item in self._participants):
                self._participants.append(participant)
        return participant

    def on_readiness_lost(self, callback: Callable[[], Any]) -> None:
        """Call ``callback`` first when a drain starts, before admission closes."""
        self._readiness_callbacks.append(callback)

    def add_listener(self, listener: Callable[[DrainEvent], Any]) -> None:
        """Receive every value-free progress event as it is recorded."""
        self._listeners.append(listener)

    def snapshot(self) -> dict[str, Any]:
        """Return the current state and outstanding counts per category."""
        outstanding: dict[str, int] = {}
        for participant in self.participants():
            outstanding[participant.category] = outstanding.get(
                participant.category, 0
            ) + _safe_outstanding(participant)
        return {"state": self._state.value, "outstanding": outstanding}

    async def drain(self) -> DrainReport:
        """Run the drain once and return its report.

        Returns:
            The value-free report for this coordinator's single drain.
        """
        if self._report is not None:
            return self._report
        if self._task is None:
            self._task = asyncio.ensure_future(self._run())
        return await asyncio.shield(self._task)

    async def _run(self) -> DrainReport:
        loop = asyncio.get_running_loop()
        started = loop.time()
        with self._lock:
            self._state = DrainState.DRAINING
        self._metric("set_draining", True)
        for callback in tuple(self._readiness_callbacks):
            try:
                result = callback()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                _LOGGER.warning("service drain readiness hook failed")

        budget = self.budget
        in_flight_deadline = started + budget.drain_seconds
        settle_deadline = in_flight_deadline - float(budget.settle_seconds or 0.0)
        await self._run_sequential(
            DrainPhase.STOP_ADMISSION, in_flight_deadline, in_flight_deadline
        )
        await self._run_concurrent(
            DrainPhase.FINISH_IN_FLIGHT, in_flight_deadline, settle_deadline
        )
        released_at = loop.time()
        flush_deadline = released_at + budget.release_seconds / 2
        release_deadline = released_at + budget.release_seconds
        await self._run_sequential(DrainPhase.FLUSH, flush_deadline, flush_deadline)
        await self._run_sequential(
            DrainPhase.RELEASE, release_deadline, release_deadline
        )

        events = tuple(self._events)
        outcome = max(
            (event.outcome for event in events),
            key=_OUTCOME_SEVERITY.__getitem__,
            default=DrainOutcome.COMPLETED,
        )
        if outcome is DrainOutcome.SKIPPED:
            outcome = DrainOutcome.COMPLETED
        report = DrainReport(
            outcome=outcome,
            budget=budget,
            duration_seconds=max(loop.time() - started, 0.0),
            events=events,
        )
        self._report = report
        with self._lock:
            self._state = DrainState.DRAINED
        self._metric("set_draining", False)
        _LOGGER.info(
            "service drain finished outcome=%s unfinished=%d",
            report.outcome.value,
            report.totals()["unfinished"],
        )
        return report

    async def _run_sequential(
        self, phase: DrainPhase, deadline: float, settle_deadline: float
    ) -> None:
        for participant in self.participants():
            if participant.handles(phase):
                await self._invoke(participant, phase, deadline, settle_deadline)

    async def _run_concurrent(
        self, phase: DrainPhase, deadline: float, settle_deadline: float
    ) -> None:
        invocations = [
            self._invoke(participant, phase, deadline, settle_deadline)
            for participant in self.participants()
            if participant.handles(phase)
        ]
        if invocations:
            await asyncio.gather(*invocations)

    async def _invoke(
        self,
        participant: DrainParticipant,
        phase: DrainPhase,
        deadline: float,
        settle_deadline: float,
    ) -> None:
        loop = asyncio.get_running_loop()
        started = loop.time()
        budget_seconds = max(deadline - started, 0.0)
        ctx = DrainContext(
            phase=phase,
            deadline=deadline,
            settle_deadline=min(settle_deadline, deadline),
            retry_after_seconds=self.retry_after_seconds,
            budget=self.budget,
        )
        result: Any
        try:
            result = getattr(participant, phase.value)(ctx)
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(
                    result, timeout=budget_seconds + _HOOK_GRACE_SECONDS
                )
        except asyncio.TimeoutError:
            result = PhaseResult(
                outcome=DrainOutcome.DEADLINE_EXCEEDED,
                unfinished=max(_safe_outstanding(participant), 1),
                reason_code=DRAIN_DEADLINE_EXCEEDED,
            )
        except Exception:
            # Exception text can carry request values; record only the code.
            result = PhaseResult(
                outcome=DrainOutcome.FAILED,
                unfinished=_safe_outstanding(participant),
                reason_code=PARTICIPANT_FAILED,
            )
        if not isinstance(result, PhaseResult):
            result = PhaseResult(outcome=DrainOutcome.SKIPPED)
        self._record(
            DrainEvent(
                phase=phase,
                participant=participant.category,
                outcome=result.outcome,
                completed=result.completed,
                cancelled=result.cancelled,
                checkpointed=result.checkpointed,
                unfinished=result.unfinished,
                budget_seconds=budget_seconds,
                duration_seconds=max(loop.time() - started, 0.0),
                reason_code=result.reason_code,
            )
        )

    def _record(self, event: DrainEvent) -> None:
        self._events.append(event)
        self._metric("record_drain_event", event)
        for listener in tuple(self._listeners):
            try:
                listener(event)
            except Exception:
                _LOGGER.warning("service drain listener failed")

    def _metric(self, name: str, *args: Any) -> None:
        recorder = getattr(self.metrics, name, None)
        if callable(recorder):
            try:
                recorder(*args)
            except Exception:
                _LOGGER.warning("service drain metric update failed")


def resolve_drain_coordinator(target: Any) -> DrainCoordinator:
    """Return the coordinator owned by ``target``.

    Args:
        target: A :class:`DrainCoordinator`, an object exposing
            ``get_drain_coordinator()`` (such as ``ServiceRuntime``), or an
            application whose ``state.drain`` holds the active coordinator.

    Returns:
        The resolved coordinator.

    Raises:
        TypeError: When no coordinator can be resolved.
    """
    if isinstance(target, DrainCoordinator):
        return target
    getter = getattr(target, "get_drain_coordinator", None)
    if callable(getter):
        coordinator = getter()
        if isinstance(coordinator, DrainCoordinator):
            return coordinator
    state = getattr(target, "state", None)
    coordinator = getattr(state, "drain", None)
    if isinstance(coordinator, DrainCoordinator):
        return coordinator
    raise TypeError("target does not own a drain coordinator")


def register_drain_participant(
    target: Any,
    participant: DrainParticipant,
) -> DrainParticipant:
    """Register a lifecycle participant for a service drain.

    For an application object (anything with ``state``), the participant is kept
    in ``state.drain_participants`` and joins every lifespan's coordinator,
    including the one currently serving when it has not started draining. For a
    runtime or a coordinator, the participant joins that coordinator directly.

    Args:
        target: Application, ``ServiceRuntime``, or :class:`DrainCoordinator`.
        participant: Participant to register.

    Returns:
        The registered participant.
    """
    if not isinstance(participant, DrainParticipant):
        raise TypeError("drain participants must subclass DrainParticipant")
    validate_drain_category(participant.category)
    state = getattr(target, "state", None)
    if state is not None and not isinstance(target, DrainCoordinator):
        registered = getattr(state, "drain_participants", None)
        if not isinstance(registered, list):
            registered = []
            state.drain_participants = registered
        if not any(item is participant for item in registered):
            registered.append(participant)
        active = getattr(state, "drain", None)
        if isinstance(active, DrainCoordinator) and active.state is DrainState.SERVING:
            active.register(participant)
        return participant
    return resolve_drain_coordinator(target).register(participant)


def validate_drain_category(category: Any) -> str:
    """Reject participant labels that are not controlled lowercase tokens.

    Args:
        category: Candidate label.

    Returns:
        The validated label.

    Raises:
        ValueError: When the label could carry free text.
    """
    if type(category) is not str or not _CATEGORY_PATTERN.fullmatch(category):
        raise ValueError("drain participant category must be a controlled label")
    return category


class HTTPRequestDrainParticipant(DrainParticipant):
    """Drain REST and GraphQL requests tracked by the readiness middleware.

    Reuses the existing ``state.ready``, ``state.shutting_down`` and
    ``state.inflight`` counters.

    Args:
        state: Application state object.
    """

    category = "http_requests"
    priority = 10

    def __init__(self, state: Any) -> None:
        super().__init__()
        self._state = state
        self._initial = 0

    def stop_admission(self, ctx: DrainContext) -> PhaseResult:
        """Drop readiness, then reject new model-backed requests."""
        self._state.ready = False
        self._state.shutting_down = True
        self._initial = self.outstanding()
        return PhaseResult()

    async def finish_in_flight(self, ctx: DrainContext) -> PhaseResult:
        """Wait for admitted requests until the in-flight deadline."""
        await ctx.wait_until(lambda: self.outstanding() == 0)
        unfinished = self.outstanding()
        return PhaseResult.from_counts(
            completed=max(self._initial - unfinished, 0), unfinished=unfinished
        )

    def outstanding(self) -> int:
        """Return admitted requests that have not settled."""
        return max(int(getattr(self._state, "inflight", 0) or 0), 0)


class _StreamLease:
    """Ownership of one stream slot; release is idempotent and thread-safe."""

    __slots__ = ("_owner", "_released", "started")

    def __init__(self, owner: "NDJSONStreamDrainParticipant") -> None:
        self._owner = owner
        self._released = False
        self.started = False

    def release(self) -> None:
        with self._owner._lock:
            if self._released:
                return
            self._released = True
            self._owner._active -= 1

    def release_if_unstarted(self) -> None:
        if not self.started:
            self.release()


class NDJSONStreamDrainParticipant(DrainParticipant):
    """Bound NDJSON stream tails and end them with a typed terminal event.

    Streams keep running during the in-flight window. Streams still open at the
    settle deadline stop at their next event boundary and end with one
    ``shutting_down`` error line. Buffered, unflushed text is discarded rather
    than emitted, and a running window is left to finish before the source is
    closed, so model references are released only once the work settles.

    Args:
        retry_after_seconds: Retry hint written into the terminal event.
    """

    category = "ndjson_streams"
    priority = 20

    def __init__(
        self, *, retry_after_seconds: float = DEFAULT_RETRY_AFTER_SECONDS
    ) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._active = 0
        self._terminating = False
        self._signal: Optional[asyncio.Event] = None
        self._cleanup: set[asyncio.Task[None]] = set()
        self.terminal_line = shutting_down_ndjson_line(retry_after_seconds)

    @property
    def active(self) -> int:
        """Return open stream slots, including ones still settling."""
        with self._lock:
            return self._active

    @property
    def terminating(self) -> bool:
        """Return whether open streams were told to stop."""
        return self._terminating

    def wrap(self, source: AsyncIterator[str]) -> AsyncIterator[str]:
        """Return a drain-aware view of an NDJSON line iterator.

        Args:
            source: Async iterator of newline-terminated JSON lines.

        Returns:
            An async iterator that ends with the terminal event when drained.
        """
        with self._lock:
            self._active += 1
        lease = _StreamLease(self)
        iterator = self._iterate(source, lease)
        # A body that is never iterated still frees its slot when discarded.
        weakref.finalize(iterator, lease.release_if_unstarted)
        return iterator

    def terminate(self) -> int:
        """Ask every open stream to stop at its next boundary.

        Returns:
            Streams open when the signal was raised.
        """
        self._terminating = True
        if self._signal is not None:
            self._signal.set()
        return self.active

    async def finish_in_flight(self, ctx: DrainContext) -> PhaseResult:
        """Let streams finish until the settle deadline, then end the rest."""
        initial = self.active
        await ctx.wait_until(lambda: self.active == 0, deadline=ctx.settle_deadline)
        terminated = self.terminate() if self.active else 0
        await ctx.wait_until(lambda: self.active == 0)
        return PhaseResult.from_counts(
            completed=max(initial - terminated, 0),
            cancelled=terminated,
            unfinished=self.active,
        )

    def outstanding(self) -> int:
        """Return open stream slots."""
        return self.active

    def _event(self) -> asyncio.Event:
        if self._signal is None:
            self._signal = asyncio.Event()
            if self._terminating:
                self._signal.set()
        return self._signal

    async def _iterate(
        self, source: AsyncIterator[str], lease: _StreamLease
    ) -> AsyncIterator[str]:
        lease.started = True
        pending: Optional[asyncio.Future[str]] = None
        exhausted = False
        try:
            while True:
                if self._terminating:
                    yield self.terminal_line
                    return
                pending = asyncio.ensure_future(source.__anext__())
                if not await self._race(pending):
                    yield self.terminal_line
                    return
                step, pending = pending, None
                try:
                    line = step.result()
                except StopAsyncIteration:
                    exhausted = True
                    return
                yield line
        finally:
            self._settle(source, pending, lease, exhausted)

    async def _race(self, pending: asyncio.Future[str]) -> bool:
        if pending.done():
            return True
        waiter = asyncio.ensure_future(self._event().wait())
        try:
            done, _ = await asyncio.wait(
                {pending, waiter}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            if not waiter.done():
                waiter.cancel()
        return pending in done

    def _settle(
        self,
        source: AsyncIterator[str],
        pending: Optional[asyncio.Future[str]],
        lease: _StreamLease,
        exhausted: bool,
    ) -> None:
        if pending is None and exhausted:
            lease.release()
            return

        async def cleanup() -> None:
            try:
                if pending is not None:
                    await asyncio.wait({pending})
                    if not pending.cancelled():
                        pending.exception()
                aclose = getattr(source, "aclose", None)
                if callable(aclose):
                    await aclose()
            except Exception:
                # Never log or surface stream content during cleanup.
                pass
            finally:
                lease.release()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            lease.release()
            return
        # Run outside the response task: its cancellation scope must not cut
        # the wait for a running window short.
        task = loop.create_task(cleanup())
        self._cleanup.add(task)
        task.add_done_callback(self._cleanup.discard)


class CallAdmission:
    """Thread-safe admission gate and in-flight counter for RPC surfaces.

    Args:
        retry_after_seconds: Retry hint attached to refusals.
    """

    def __init__(
        self, *, retry_after_seconds: float = DEFAULT_RETRY_AFTER_SECONDS
    ) -> None:
        self.retry_after_seconds = _non_negative_seconds(retry_after_seconds)
        self._lock = threading.Lock()
        self._closed = False
        self._terminating = False
        self._unary = 0
        self._streams = 0
        self._terminated = 0

    @property
    def closed(self) -> bool:
        """Return whether new calls are refused."""
        return self._closed

    @property
    def streams_terminating(self) -> bool:
        """Return whether open streams must stop at their next boundary."""
        return self._terminating

    @property
    def active(self) -> int:
        """Return in-flight unary calls plus open streams."""
        with self._lock:
            return self._unary + self._streams

    @property
    def active_streams(self) -> int:
        """Return open streaming calls."""
        with self._lock:
            return self._streams

    def try_enter(self, *, streaming: bool = False) -> bool:
        """Admit one call unless admission is closed.

        Args:
            streaming: Whether the call is a server stream.

        Returns:
            ``True`` when admitted; the caller must later call :meth:`leave`.
        """
        with self._lock:
            if self._closed:
                return False
            if streaming:
                self._streams += 1
            else:
                self._unary += 1
            return True

    def leave(self, *, streaming: bool = False) -> None:
        """Release one admitted call."""
        with self._lock:
            if streaming:
                self._streams = max(self._streams - 1, 0)
            else:
                self._unary = max(self._unary - 1, 0)

    def record_terminated_stream(self) -> None:
        """Count one stream ended with a typed shutdown status."""
        with self._lock:
            self._terminated += 1

    @property
    def terminated_streams(self) -> int:
        """Return streams ended by a drain."""
        with self._lock:
            return self._terminated

    def close(self) -> None:
        """Refuse new calls."""
        with self._lock:
            self._closed = True

    def terminate_streams(self) -> int:
        """Ask open streams to stop and return how many were open."""
        with self._lock:
            self._terminating = True
            return self._streams


class BatcherDrainParticipant(DrainParticipant):
    """Flush micro-batchers: dispatch queued work, then fail the remainder.

    Args:
        resolver: Returns the batchers to drain; ``None`` entries are ignored.
    """

    category = "batcher"
    priority = 40

    def __init__(self, resolver: Callable[[], Iterable[Any]]) -> None:
        super().__init__()
        self._resolver = resolver

    def _batchers(self) -> list[Any]:
        return [item for item in self._resolver() or () if item is not None]

    async def flush(self, ctx: DrainContext) -> Optional[PhaseResult]:
        """Dispatch queued jobs and wait for running batches until the deadline."""
        batchers = self._batchers()
        if not batchers:
            return None
        cancelled = unfinished = completed = 0
        for batcher in batchers:
            stats = await batcher.drain(timeout=ctx.remaining())
            completed += int(stats.get("dispatched", 0))
            cancelled += int(stats.get("failed_queued", 0))
            unfinished += int(stats.get("running", 0))
        return PhaseResult.from_counts(
            completed=completed, cancelled=cancelled, unfinished=unfinished
        )

    def outstanding(self) -> int:
        """Return queued plus running batch work across batchers."""
        return sum(int(item.outstanding_count()) for item in self._batchers())


class JobQueueDrainParticipant(DrainParticipant):
    """Drain the in-process public job queue without silent loss.

    Running jobs may finish until the settle deadline. Afterwards they stop at
    their next document boundary; at the deadline every job still queued or
    running is fenced as ``cancelled`` with a retryable ``shutting_down`` code,
    so a stale worker can never commit after the handoff.

    Args:
        resolver: Returns the job queue, or ``None`` when absent.
    """

    category = "job_queue"
    priority = 50

    def __init__(self, resolver: Callable[[], Any]) -> None:
        super().__init__()
        self._resolver = resolver

    def stop_admission(self, ctx: DrainContext) -> Optional[PhaseResult]:
        """Refuse new submissions and bound terminal callbacks."""
        queue = self._resolver()
        if queue is None:
            return None
        queue.close_admission(
            callback_timeout_seconds=max(float(ctx.budget.settle_seconds or 0.0), 0.1)
        )
        return PhaseResult()

    async def finish_in_flight(self, ctx: DrainContext) -> Optional[PhaseResult]:
        """Finish, cooperatively cancel, then fence unfinished jobs."""
        queue = self._resolver()
        if queue is None:
            return None
        idle = lambda: queue.outstanding_count() == 0  # noqa: E731
        if not await ctx.wait_until(idle, deadline=ctx.settle_deadline):
            queue.request_cancel()
            if not await ctx.wait_until(idle):
                queue.cancel_outstanding()
        stats = queue.drain_stats()
        return PhaseResult.from_counts(
            completed=int(stats.get("completed", 0)),
            cancelled=int(stats.get("cancelled", 0)),
            unfinished=queue.outstanding_count(),
        )

    def release(self, ctx: DrainContext) -> Optional[PhaseResult]:
        """Shut the worker pool down without waiting on stale workers."""
        queue = self._resolver()
        if queue is None:
            return None
        queue.shutdown()
        return PhaseResult.from_counts(unfinished=queue.outstanding_count())

    def outstanding(self) -> int:
        """Return queued, running, or callback-phase jobs."""
        queue = self._resolver()
        return 0 if queue is None else int(queue.outstanding_count())


class BackgroundJobsDrainParticipant(DrainParticipant):
    """Let SMART or bulk background jobs finish, then cancel them.

    Cancelled ingestions keep their own committed checkpoints, so a resubmitted
    job resumes from the last committed resource.

    Args:
        resolver: Returns a manager exposing ``active_count()`` and
            ``cancel_all()``, or ``None``.
        category: Controlled participant label.
    """

    priority = 60

    def __init__(self, resolver: Callable[[], Any], *, category: str) -> None:
        super().__init__(category)
        self._resolver = resolver

    async def finish_in_flight(self, ctx: DrainContext) -> Optional[PhaseResult]:
        """Wait until the settle deadline, then cancel what is still running."""
        manager = self._resolver()
        if manager is None:
            return None
        initial = int(manager.active_count())
        idle = lambda: manager.active_count() == 0  # noqa: E731
        cancelled = 0
        if not await ctx.wait_until(idle, deadline=ctx.settle_deadline):
            cancelled = int(manager.active_count())
            await asyncio.wait_for(manager.cancel_all(), timeout=ctx.remaining())
        unfinished = int(manager.active_count())
        return PhaseResult.from_counts(
            completed=max(initial - cancelled, 0),
            cancelled=max(cancelled - unfinished, 0),
            unfinished=unfinished,
        )

    def outstanding(self) -> int:
        """Return running background jobs."""
        manager = self._resolver()
        return 0 if manager is None else int(manager.active_count())


class WarmPoolDrainParticipant(DrainParticipant):
    """Fence model activation and release warm-pool references.

    Active requests keep the model they hold until they finish; once the pool is
    closed their last reference unloads the model instead of scheduling an idle
    timer.

    Args:
        resolver: Returns the warm pool, or ``None`` when it was never created.
    """

    category = "warm_pool"
    priority = 80

    def __init__(self, resolver: Callable[[], Any]) -> None:
        super().__init__()
        self._resolver = resolver

    async def release(self, ctx: DrainContext) -> Optional[PhaseResult]:
        """Close the pool, unload inactive models, and wait for stragglers."""
        pool = self._resolver()
        if pool is None:
            return None
        summary = pool.close()
        loop = asyncio.get_running_loop()
        settle = loop.time() + ctx.remaining() / 2
        await ctx.wait_until(lambda: pool.active_request_count() == 0, deadline=settle)
        return PhaseResult.from_counts(
            completed=int(summary.get("released_models", 0)),
            unfinished=int(pool.active_request_count()),
        )

    def outstanding(self) -> int:
        """Return active model references."""
        pool = self._resolver()
        return 0 if pool is None else int(pool.active_request_count())


@runtime_checkable
class AcceleratorDevice(Protocol):
    """Minimal accelerator surface used to bound device release.

    Attributes:
        kind: Controlled device label such as ``"cpu"`` or ``"cuda"``.
    """

    kind: str

    def allocated(self) -> int:
        """Return currently held units (bytes or allocations)."""

    def synchronize(self) -> None:
        """Block until queued device work completes."""

    def release(self) -> None:
        """Return cached device memory to the driver."""


class CPUAcceleratorDevice:
    """CPU placeholder: nothing to synchronize and nothing retained."""

    kind = "cpu"
    blocking = False

    def allocated(self) -> int:
        """Return zero; host memory is reclaimed with the process."""
        return 0

    def synchronize(self) -> None:
        """Do nothing."""

    def release(self) -> None:
        """Do nothing."""


class TorchAcceleratorDevice:
    """Torch CUDA or MPS device, only used when torch is already imported.

    Args:
        torch_module: The already-imported ``torch`` module.
        kind: ``"cuda"`` or ``"mps"``.
    """

    def __init__(self, torch_module: Any, kind: str) -> None:
        if kind not in {"cuda", "mps"}:
            raise ValueError("unsupported torch accelerator kind")
        self.kind = kind
        self._backend = getattr(torch_module, kind)

    def allocated(self) -> int:
        """Return bytes currently allocated by tensors."""
        name = "memory_allocated" if self.kind == "cuda" else "current_allocated_memory"
        reader = getattr(self._backend, name, None)
        return int(reader()) if callable(reader) else 0

    def synchronize(self) -> None:
        """Wait for queued kernels."""
        sync = getattr(self._backend, "synchronize", None)
        if callable(sync):
            sync()

    def release(self) -> None:
        """Return cached allocator blocks."""
        empty = getattr(self._backend, "empty_cache", None)
        if callable(empty):
            empty()


class MLXAcceleratorDevice:
    """MLX Metal device, only used when ``mlx.core`` is already imported.

    Args:
        mx: The already-imported ``mlx.core`` module.
    """

    kind = "mlx"

    def __init__(self, mx: Any) -> None:
        self._mx = mx

    def allocated(self) -> int:
        """Return active MLX memory in bytes."""
        reader = getattr(self._mx, "get_active_memory", None)
        if not callable(reader):
            reader = getattr(
                getattr(self._mx, "metal", None), "get_active_memory", None
            )
        return int(reader()) if callable(reader) else 0

    def synchronize(self) -> None:
        """Wait for queued MLX work."""
        sync = getattr(self._mx, "synchronize", None)
        if callable(sync):
            sync()

    def release(self) -> None:
        """Clear the MLX buffer cache."""
        clear = getattr(self._mx, "clear_cache", None)
        if not callable(clear):
            clear = getattr(getattr(self._mx, "metal", None), "clear_cache", None)
        if callable(clear):
            clear()


def detect_accelerator_devices() -> tuple[Any, ...]:
    """Return devices to release without importing any accelerator runtime.

    Returns:
        A CPU placeholder plus torch CUDA/MPS and MLX devices whose runtimes are
        already imported and report themselves available.
    """
    devices: list[Any] = [CPUAcceleratorDevice()]
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        for kind, probe in (("cuda", "is_initialized"), ("mps", "is_available")):
            # CUDA is only touched once this process already created a context.
            check = getattr(getattr(torch_module, kind, None), probe, None)
            try:
                if callable(check) and check():
                    devices.append(TorchAcceleratorDevice(torch_module, kind))
            except Exception:
                continue
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        devices.append(MLXAcceleratorDevice(mx))
    return tuple(devices)


class AcceleratorDrainParticipant(DrainParticipant):
    """Synchronize and release accelerators back to their start-up baseline.

    The baseline is captured when the participant is created, so memory held
    before the service started serving is declared retained, not leaked.

    Args:
        devices: Devices implementing :class:`AcceleratorDevice`.
    """

    category = "accelerator"
    priority = 90

    def __init__(self, devices: Sequence[Any]) -> None:
        super().__init__()
        self._devices = tuple(devices)
        self._baseline = {id(device): _safe_allocated(device) for device in devices}

    async def release(self, ctx: DrainContext) -> Optional[PhaseResult]:
        """Bound synchronization, release caches, and compare to the baseline."""
        if not self._devices:
            return None
        completed = unfinished = 0
        for device in self._devices:
            try:
                if getattr(device, "blocking", True):
                    await ctx.run_blocking(device.synchronize)
                else:
                    device.synchronize()
            except asyncio.TimeoutError:
                unfinished += 1
                continue
            device.release()
            if _safe_allocated(device) > self._baseline.get(id(device), 0):
                unfinished += 1
            else:
                completed += 1
        return PhaseResult.from_counts(completed=completed, unfinished=unfinished)

    def retained(self) -> dict[str, int]:
        """Return units above baseline per device kind."""
        result: dict[str, int] = {}
        for device in self._devices:
            excess = max(_safe_allocated(device) - self._baseline.get(id(device), 0), 0)
            kind = str(getattr(device, "kind", "device"))
            result[kind] = result.get(kind, 0) + excess
        return result

    def outstanding(self) -> int:
        """Return devices still above their baseline."""
        return sum(1 for value in self.retained().values() if value > 0)


def _safe_allocated(device: Any) -> int:
    try:
        return max(int(device.allocated()), 0)
    except Exception:
        return 0


def _safe_outstanding(participant: DrainParticipant) -> int:
    try:
        return _bounded_count(participant.outstanding())
    except Exception:
        return 0


def _bounded_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("drain counts must be integers")
    return min(max(value, 0), _MAX_COUNT)


def _non_negative_seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("drain durations must be numbers")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("drain durations must be finite and non-negative")
    return seconds


__all__ = [
    "DEFAULT_DRAIN_SECONDS",
    "DEFAULT_RELEASE_SECONDS",
    "DEFAULT_RETRY_AFTER_SECONDS",
    "DRAIN_PHASES",
    "DRAIN_REASON_CODES",
    "SHUTTING_DOWN",
    "AcceleratorDevice",
    "AcceleratorDrainParticipant",
    "BackgroundJobsDrainParticipant",
    "BatcherDrainParticipant",
    "CPUAcceleratorDevice",
    "CallAdmission",
    "CallbackDrainParticipant",
    "DrainBudget",
    "DrainContext",
    "DrainCoordinator",
    "DrainEvent",
    "DrainOutcome",
    "DrainParticipant",
    "DrainPhase",
    "DrainRegistrationError",
    "DrainReport",
    "DrainState",
    "HTTPRequestDrainParticipant",
    "JobQueueDrainParticipant",
    "MLXAcceleratorDevice",
    "NDJSONStreamDrainParticipant",
    "PhaseResult",
    "ServiceShuttingDownError",
    "TorchAcceleratorDevice",
    "WarmPoolDrainParticipant",
    "detect_accelerator_devices",
    "register_drain_participant",
    "resolve_drain_coordinator",
    "shutting_down_details",
    "shutting_down_ndjson_line",
    "validate_drain_category",
]
