"""Cooperative, content-free resource limits for one local agent run.

The caller reserves estimated work at safe checkpoints before scheduling it.
This module never executes or interrupts an action and never inspects payloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from threading import RLock

_MAX_VALUE = (1 << 63) - 1


class ResourceBudgetError(ValueError):
    """A value-free validation or halted-run error with a stable code."""

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        super().__init__(code if field_name is None else f"{field_name}: {code}")


def _integer(value: object, field_name: str, *, positive: bool = False) -> int:
    if (
        type(value) is not int
        or not (0 < value if positive else 0 <= value)
        or value > _MAX_VALUE
    ):
        raise ResourceBudgetError("invalid_integer", field_name)
    return value


class BudgetStopReason(str, Enum):
    """Closed reasons for stopping a run at a safe checkpoint."""

    WALL_TIME = "wall_time"
    STEPS = "steps"
    TOOL_CALLS = "tool_calls"
    MEMORY_ESTIMATE = "memory_estimate_bytes"
    ARTIFACT_STORAGE = "artifact_bytes"
    INVALID_CLOCK = "invalid_clock"


@dataclass(frozen=True, slots=True)
class ResourceBudgetLimits:
    """Declared maximums for one run; all five limits are required.

    Args:
        max_steps: Maximum scheduled steps.
        max_tool_calls: Maximum scheduled tool invocations.
        max_wall_time_ns: Maximum elapsed monotonic nanoseconds.
        max_memory_estimate_bytes: Maximum estimated peak memory in bytes.
        max_artifact_bytes: Maximum stored redacted artifact bytes.
    """

    max_steps: int
    max_tool_calls: int
    max_wall_time_ns: int
    max_memory_estimate_bytes: int
    max_artifact_bytes: int

    def __post_init__(self) -> None:
        for field_name in ("max_steps", "max_tool_calls", "max_wall_time_ns"):
            _integer(getattr(self, field_name), field_name, positive=True)
        for field_name in ("max_memory_estimate_bytes", "max_artifact_bytes"):
            _integer(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, int]:
        """Return numeric limits without caller-supplied content."""

        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ResourceBudgetReport:
    """Immutable utilization at the last checkpoint, without run content."""

    limits: ResourceBudgetLimits
    steps: int
    tool_calls: int
    elapsed_ns: int
    peak_memory_estimate_bytes: int
    artifact_bytes: int
    stop_reason: BudgetStopReason | None

    def __post_init__(self) -> None:
        if type(self.limits) is not ResourceBudgetLimits:
            raise ResourceBudgetError("invalid_limits", "limits")
        for field_name in (
            "steps",
            "tool_calls",
            "elapsed_ns",
            "peak_memory_estimate_bytes",
            "artifact_bytes",
        ):
            _integer(getattr(self, field_name), field_name)
        if (
            self.stop_reason is not None
            and type(self.stop_reason) is not BudgetStopReason
        ):
            raise ResourceBudgetError("invalid_stop_reason", "stop_reason")

    @property
    def stopped(self) -> bool:
        """Return whether the run must stop scheduling work."""

        return self.stop_reason is not None

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic report containing only counts and codes."""

        return {
            "limits": self.limits.to_dict(),
            "usage": {
                "steps": self.steps,
                "tool_calls": self.tool_calls,
                "elapsed_ns": self.elapsed_ns,
                "peak_memory_estimate_bytes": self.peak_memory_estimate_bytes,
                "artifact_bytes": self.artifact_bytes,
            },
            "stopped": self.stopped,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
        }

    def to_json(self) -> str:
        """Serialize the content-free report in canonical key order."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


class RunResourceBudget:
    """Reserve bounded work at checkpoints for exactly one local run.

    Args:
        limits: Declared per-run resource ceilings.
        started_ns: Monotonic clock value captured at run start.

    An exceeded reservation is rejected without charging its proposed work.
    The meter then remains stopped. No automatic reset or retry is provided.
    """

    def __init__(self, limits: ResourceBudgetLimits, *, started_ns: int) -> None:
        if type(limits) is not ResourceBudgetLimits:
            raise ResourceBudgetError("invalid_limits", "limits")
        self._started_ns = _integer(started_ns, "started_ns")
        self._last_ns = started_ns
        self._limits = limits
        self._steps = 0
        self._tool_calls = 0
        self._peak_memory = 0
        self._artifact_bytes = 0
        self._stop_reason: BudgetStopReason | None = None
        self._lock = RLock()

    @property
    def stopped(self) -> bool:
        """Return whether further work must be refused."""

        with self._lock:
            return self._stop_reason is not None

    def report(self) -> ResourceBudgetReport:
        """Return utilization at the most recent checkpoint."""

        with self._lock:
            return ResourceBudgetReport(
                limits=self._limits,
                steps=self._steps,
                tool_calls=self._tool_calls,
                elapsed_ns=self._last_ns - self._started_ns,
                peak_memory_estimate_bytes=self._peak_memory,
                artifact_bytes=self._artifact_bytes,
                stop_reason=self._stop_reason,
            )

    def reserve(
        self,
        *,
        now_ns: int,
        steps: int = 0,
        tool_calls: int = 0,
        memory_estimate_bytes: int = 0,
        artifact_bytes: int = 0,
    ) -> ResourceBudgetReport:
        """Check the clock and reserve proposed work before an action.

        ``memory_estimate_bytes`` is the estimated peak for the next action;
        all other work amounts are increments. A zero-work call checks time.
        The caller must stop scheduling when the returned report is stopped.
        """

        with self._lock:
            if self.stopped:
                raise ResourceBudgetError("run_stopped")
            now_ns = _integer(now_ns, "now_ns")
            steps = _integer(steps, "steps")
            tool_calls = _integer(tool_calls, "tool_calls")
            memory_estimate_bytes = _integer(
                memory_estimate_bytes, "memory_estimate_bytes"
            )
            artifact_bytes = _integer(artifact_bytes, "artifact_bytes")

            if now_ns < self._last_ns:
                self._stop_reason = BudgetStopReason.INVALID_CLOCK
                return self.report()

            self._last_ns = now_ns
            checks = (
                (
                    now_ns - self._started_ns,
                    self._limits.max_wall_time_ns,
                    BudgetStopReason.WALL_TIME,
                ),
                (self._steps + steps, self._limits.max_steps, BudgetStopReason.STEPS),
                (
                    self._tool_calls + tool_calls,
                    self._limits.max_tool_calls,
                    BudgetStopReason.TOOL_CALLS,
                ),
                (
                    max(self._peak_memory, memory_estimate_bytes),
                    self._limits.max_memory_estimate_bytes,
                    BudgetStopReason.MEMORY_ESTIMATE,
                ),
                (
                    self._artifact_bytes + artifact_bytes,
                    self._limits.max_artifact_bytes,
                    BudgetStopReason.ARTIFACT_STORAGE,
                ),
            )
            for proposed, limit, reason in checks:
                if proposed > limit:
                    self._stop_reason = reason
                    return self.report()

            self._steps += steps
            self._tool_calls += tool_calls
            self._peak_memory = max(self._peak_memory, memory_estimate_bytes)
            self._artifact_bytes += artifact_bytes
            return self.report()
