"""One lifecycle vocabulary for runtime and streaming speech sessions.

``MultimodalSession`` (``openmed.multimodal.session.v1``), the streaming ASR
session and the OpenMedKit speech sessions each name their states
differently. This module fixes one shared vocabulary, its allowed transitions
and its terminal states. Each session keeps its own enum and serialized
reports unchanged and exposes a ``lifecycle_state`` projection onto this
vocabulary.

The vocabulary is described by ``session_lifecycle.schema.json`` and pinned
for every runtime by ``tests/fixtures/parity/session_lifecycle_v1.json``.
Reason codes are controlled strings; they never carry session content.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Final

SESSION_LIFECYCLE_VERSION: Final[str] = "openmed.session.lifecycle.v1"


class LifecycleState(str, Enum):
    """Shared session lifecycle states.

    Values:
        OPEN: The session accepts work.
        CANCELLING: Cancellation or close was requested and active work or
            owned cleanup is still unwinding.
        FINALIZED: The session completed its stream and accepts no input.
        CANCELLED: The session was cancelled and accepts no input.
        FAILED: The session stopped on an error and accepts no input.
        CLOSED: The session was closed and released its resources.
    """

    OPEN = "open"
    CANCELLING = "cancelling"
    FINALIZED = "finalized"
    CANCELLED = "cancelled"
    FAILED = "failed"
    CLOSED = "closed"


LIFECYCLE_STATES: Final[tuple[str, ...]] = tuple(
    state.value for state in LifecycleState
)

TERMINAL_STATES: Final[frozenset[LifecycleState]] = frozenset(
    {
        LifecycleState.FINALIZED,
        LifecycleState.CANCELLED,
        LifecycleState.FAILED,
        LifecycleState.CLOSED,
    }
)

#: Source state -> the states it may move to. Terminal states have none.
ALLOWED_TRANSITIONS: Final[Mapping[LifecycleState, frozenset[LifecycleState]]] = (
    MappingProxyType(
        {
            LifecycleState.OPEN: frozenset(
                {
                    LifecycleState.CANCELLING,
                    LifecycleState.FINALIZED,
                    LifecycleState.CANCELLED,
                    LifecycleState.FAILED,
                    LifecycleState.CLOSED,
                }
            ),
            LifecycleState.CANCELLING: frozenset(
                {
                    LifecycleState.CANCELLED,
                    LifecycleState.FAILED,
                    LifecycleState.CLOSED,
                }
            ),
            LifecycleState.FINALIZED: frozenset(),
            LifecycleState.CANCELLED: frozenset(),
            LifecycleState.FAILED: frozenset(),
            LifecycleState.CLOSED: frozenset(),
        }
    )
)

REASON_SESSION_CANCELLED: Final[str] = "session_cancelled"
REASON_SESSION_CLOSED: Final[str] = "session_closed"
REASON_SESSION_FAILED: Final[str] = "session_failed"
REASON_TRANSITION_NOT_ALLOWED: Final[str] = "transition_not_allowed"
REASON_UNKNOWN_STATE: Final[str] = "unknown_state"

#: The closed set of rejection reason codes. ``session_closed`` and
#: ``session_cancelled`` are the codes the streaming ASR session already uses.
LIFECYCLE_REASON_CODES: Final[tuple[str, ...]] = (
    REASON_SESSION_CANCELLED,
    REASON_SESSION_CLOSED,
    REASON_SESSION_FAILED,
    REASON_TRANSITION_NOT_ALLOWED,
    REASON_UNKNOWN_STATE,
)

_TERMINAL_REASONS: Final[Mapping[LifecycleState, str]] = MappingProxyType(
    {
        LifecycleState.FINALIZED: REASON_SESSION_CLOSED,
        LifecycleState.CANCELLED: REASON_SESSION_CANCELLED,
        LifecycleState.FAILED: REASON_SESSION_FAILED,
        LifecycleState.CLOSED: REASON_SESSION_CLOSED,
    }
)


def parse_lifecycle_state(value: object) -> LifecycleState | None:
    """Return the lifecycle state for a value, or ``None`` when unknown.

    Args:
        value: A :class:`LifecycleState` or its string value.
    """

    if isinstance(value, LifecycleState):
        return value
    if isinstance(value, str):
        try:
            return LifecycleState(value)
        except ValueError:
            return None
    return None


def transition_reason(source: object, target: object) -> str | None:
    """Return why a transition is rejected, or ``None`` when it is allowed.

    Evaluation order: an unknown source or target gives ``unknown_state``; a
    terminal source gives its terminal code (``session_closed`` for
    ``finalized`` and ``closed``, ``session_cancelled`` for ``cancelled``,
    ``session_failed`` for ``failed``); any other pair outside
    :data:`ALLOWED_TRANSITIONS`, including staying in the same state, gives
    ``transition_not_allowed``.

    Args:
        source: Current state.
        target: Requested state.

    Returns:
        A member of :data:`LIFECYCLE_REASON_CODES`, or ``None``.
    """

    start = parse_lifecycle_state(source)
    end = parse_lifecycle_state(target)
    if start is None or end is None:
        return REASON_UNKNOWN_STATE
    if start in TERMINAL_STATES:
        return _TERMINAL_REASONS[start]
    if end not in ALLOWED_TRANSITIONS[start]:
        return REASON_TRANSITION_NOT_ALLOWED
    return None


def is_terminal(state: object) -> bool:
    """Return whether a lifecycle state is terminal.

    Args:
        state: A :class:`LifecycleState` or its string value.
    """

    parsed = parse_lifecycle_state(state)
    return parsed is not None and parsed in TERMINAL_STATES


__all__ = [
    "ALLOWED_TRANSITIONS",
    "LIFECYCLE_REASON_CODES",
    "LIFECYCLE_STATES",
    "REASON_SESSION_CANCELLED",
    "REASON_SESSION_CLOSED",
    "REASON_SESSION_FAILED",
    "REASON_TRANSITION_NOT_ALLOWED",
    "REASON_UNKNOWN_STATE",
    "SESSION_LIFECYCLE_VERSION",
    "TERMINAL_STATES",
    "LifecycleState",
    "is_terminal",
    "parse_lifecycle_state",
    "transition_reason",
]
