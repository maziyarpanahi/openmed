"""Deterministic, metadata-only execution phases for agent actions."""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Final, Mapping


class ActionPhase(str, Enum):
    """Closed vocabulary for an action's execution lifecycle."""

    QUEUED = "queued"
    PREFLIGHT = "preflight"
    READY = "ready"
    RUNNING = "running"
    WAITING_REVIEW = "waiting-review"
    COMPLETED = "completed"
    ABORTED = "aborted"


# Each target maps to whether completed external review is required.
ACTION_PHASE_TRANSITIONS: Final[Mapping[ActionPhase, Mapping[ActionPhase, bool]]] = (
    MappingProxyType(
        {
            ActionPhase.QUEUED: MappingProxyType(
                {ActionPhase.PREFLIGHT: False, ActionPhase.ABORTED: False}
            ),
            ActionPhase.PREFLIGHT: MappingProxyType(
                {
                    ActionPhase.READY: False,
                    ActionPhase.WAITING_REVIEW: False,
                    ActionPhase.ABORTED: False,
                }
            ),
            ActionPhase.READY: MappingProxyType(
                {ActionPhase.RUNNING: False, ActionPhase.ABORTED: False}
            ),
            ActionPhase.RUNNING: MappingProxyType(
                {
                    ActionPhase.WAITING_REVIEW: False,
                    ActionPhase.COMPLETED: False,
                    ActionPhase.ABORTED: False,
                }
            ),
            ActionPhase.WAITING_REVIEW: MappingProxyType(
                {ActionPhase.PREFLIGHT: True, ActionPhase.ABORTED: False}
            ),
            ActionPhase.COMPLETED: MappingProxyType({}),
            ActionPhase.ABORTED: MappingProxyType({}),
        }
    )
)


class ActionPhaseError(ValueError):
    """Raised when phase validation fails closed.

    Args:
        code: Stable machine-readable validation code.
        field_name: Optional public field associated with the failure.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


def _parse_phase(value: ActionPhase | str, field_name: str) -> ActionPhase:
    if isinstance(value, ActionPhase):
        return value
    if type(value) is str:
        try:
            return ActionPhase(value)
        except ValueError:
            pass
    # Raise outside the handler so rejected text is not retained in context.
    raise ActionPhaseError("unknown_phase", field_name)


def validate_action_transition(
    current: ActionPhase | str,
    target: ActionPhase | str,
    *,
    reviewed: bool = False,
) -> None:
    """Validate one lifecycle edge without executing or changing an action.

    Args:
        current: Current phase enum or exact canonical string.
        target: Requested phase enum or exact canonical string.
        reviewed: Whether the caller has completed external review permitting
            resume. Only the waiting-review to preflight edge requires this.
            This declaration is not an approval decision or credential.

    Raises:
        ActionPhaseError: If either phase is unknown, reviewed is not a bool,
            the edge is forbidden, or required review has not been declared.
    """

    source = _parse_phase(current, "current")
    destination = _parse_phase(target, "target")
    if type(reviewed) is not bool:
        raise ActionPhaseError("invalid_reviewed", "reviewed")
    targets = ACTION_PHASE_TRANSITIONS[source]
    if destination not in targets:
        raise ActionPhaseError("invalid_transition")
    if targets[destination] and not reviewed:
        raise ActionPhaseError("review_required", "reviewed")


def is_terminal_phase(phase: ActionPhase | str) -> bool:
    """Return whether a known phase has no outgoing transitions.

    Args:
        phase: Phase enum or exact canonical string.

    Raises:
        ActionPhaseError: If the phase is unknown.
    """

    return not ACTION_PHASE_TRANSITIONS[_parse_phase(phase, "phase")]


def is_resumable_phase(phase: ActionPhase | str) -> bool:
    """Return whether a known phase has an explicit reviewed resume edge.

    Args:
        phase: Phase enum or exact canonical string.

    Raises:
        ActionPhaseError: If the phase is unknown.
    """

    return any(ACTION_PHASE_TRANSITIONS[_parse_phase(phase, "phase")].values())


__all__ = [
    "ACTION_PHASE_TRANSITIONS",
    "ActionPhase",
    "ActionPhaseError",
    "is_resumable_phase",
    "is_terminal_phase",
    "validate_action_transition",
]
