"""Deterministic monotonic timing records for agent runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


class TimingValidationError(ValueError):
    """Raised when timing metadata fails closed validation."""


def _validate_ns(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TimingValidationError(f"{field_name} must be an integer")
    if value < 0:
        raise TimingValidationError(f"{field_name} must be non-negative")
    return value


def _validate_correlation_id(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TimingValidationError(f"{field_name} must be a string")
    return value


@dataclass(frozen=True)
class RunTiming:
    """Immutable monotonic nanosecond boundaries for one agent run."""

    start_ns: int
    end_ns: int
    max_duration_ns: int | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        _validate_interval(
            self.start_ns,
            self.end_ns,
            "run",
            max_duration_ns=self.max_duration_ns,
        )
        _validate_correlation_id(self.correlation_id, "run.correlation_id")

    @property
    def duration_ns(self) -> int:
        """Return the exact integer duration."""

        return self.end_ns - self.start_ns

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-safe relative timing record."""

        payload: dict[str, int | str] = {
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.duration_ns,
        }
        if self.correlation_id is not None:
            payload["correlation_id"] = self.correlation_id
        return payload


@dataclass(frozen=True)
class ActionTiming:
    """Immutable monotonic nanosecond boundaries for one agent action."""

    action_id: str
    start_ns: int
    end_ns: int
    parent_action_id: str | None = None
    max_duration_ns: int | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id:
            raise TimingValidationError("action_id must be a non-empty string")
        _validate_correlation_id(self.parent_action_id, "parent_action_id")
        _validate_correlation_id(self.correlation_id, "action.correlation_id")
        _validate_interval(
            self.start_ns,
            self.end_ns,
            "action",
            max_duration_ns=self.max_duration_ns,
        )

    @property
    def duration_ns(self) -> int:
        """Return the exact integer duration."""

        return self.end_ns - self.start_ns

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-safe relative timing record."""

        payload: dict[str, int | str] = {
            "action_id": self.action_id,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.duration_ns,
        }
        if self.parent_action_id is not None:
            payload["parent_action_id"] = self.parent_action_id
        if self.correlation_id is not None:
            payload["correlation_id"] = self.correlation_id
        return payload


@dataclass(frozen=True)
class AgentRunTiming:
    """Validated monotonic timing metadata for an agent run and its actions."""

    run: RunTiming
    actions: Sequence[ActionTiming] = field(default_factory=tuple)
    allow_action_overlaps: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunTiming):
            raise TimingValidationError("run must be a RunTiming record")
        if not isinstance(self.allow_action_overlaps, bool):
            raise TimingValidationError("allow_action_overlaps must be a boolean")
        actions = tuple(self.actions)
        for action in actions:
            if not isinstance(action, ActionTiming):
                raise TimingValidationError("actions must contain ActionTiming records")
        _validate_actions(actions, self.run, self.allow_action_overlaps)
        object.__setattr__(self, "actions", actions)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe run record without wall-clock or event payloads."""

        return {
            "run": self.run.to_dict(),
            "actions": [action.to_dict() for action in self.actions],
        }


def _validate_interval(
    start_ns: Any,
    end_ns: Any,
    prefix: str,
    *,
    max_duration_ns: Any,
) -> None:
    start = _validate_ns(start_ns, f"{prefix}.start_ns")
    end = _validate_ns(end_ns, f"{prefix}.end_ns")
    if end < start:
        raise TimingValidationError(f"{prefix}.end_ns must be greater than start_ns")
    if max_duration_ns is not None:
        maximum = _validate_ns(max_duration_ns, f"{prefix}.max_duration_ns")
        if end - start > maximum:
            raise TimingValidationError(
                f"{prefix}.duration_ns must not exceed max_duration_ns"
            )


def _validate_actions(
    actions: tuple[ActionTiming, ...],
    run: RunTiming,
    allow_action_overlaps: bool,
) -> None:
    by_id: dict[str, ActionTiming] = {}
    for action in actions:
        if action.action_id in by_id:
            raise TimingValidationError("action_id must be unique")
        by_id[action.action_id] = action
        if action.start_ns < run.start_ns:
            raise TimingValidationError("action.start_ns must be within run")
        if action.end_ns > run.end_ns:
            raise TimingValidationError("action.end_ns must be within run")

    for action in actions:
        if action.parent_action_id is None:
            continue
        parent = by_id.get(action.parent_action_id)
        if parent is None:
            raise TimingValidationError(
                "parent_action_id must reference an existing action_id"
            )
        if parent.action_id == action.action_id:
            raise TimingValidationError("parent_action_id must not reference action_id")
        if action.start_ns < parent.start_ns or action.end_ns > parent.end_ns:
            raise TimingValidationError(
                "action interval must be within parent_action_id"
            )

    if not allow_action_overlaps:
        ordered = sorted(actions, key=lambda item: (item.start_ns, item.end_ns))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_ns < previous.end_ns:
                raise TimingValidationError(
                    "actions must not overlap when allow_action_overlaps is false"
                )


__all__ = [
    "ActionTiming",
    "AgentRunTiming",
    "RunTiming",
    "TimingValidationError",
]
