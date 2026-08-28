"""Deterministic lifecycle states for federated training rounds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Mapping

FEDERATED_ROUND_SCHEMA_VERSION = "openmed.training.federated_round.v1"


class FederatedRoundState(str, Enum):
    """A privacy-safe lifecycle state shared by federated components."""

    PLANNED = "planned"
    PREFLIGHT = "preflight"
    COLLECTING = "collecting"
    AGGREGATING = "aggregating"
    EVALUATING = "evaluating"
    HELD = "held"
    PROMOTED = "promoted"
    ABORTED = "aborted"

    def __str__(self) -> str:
        return self.value


FEDERATED_ROUND_STATES: Final[tuple[FederatedRoundState, ...]] = tuple(
    FederatedRoundState
)
FEDERATED_ROUND_TERMINAL_STATES: Final[frozenset[FederatedRoundState]] = frozenset(
    {FederatedRoundState.PROMOTED, FederatedRoundState.ABORTED}
)

# A held round returns to evaluation after review; it cannot promote directly.
FEDERATED_ROUND_TRANSITIONS: Final[
    Mapping[FederatedRoundState, frozenset[FederatedRoundState]]
] = MappingProxyType(
    {
        FederatedRoundState.PLANNED: frozenset(
            {FederatedRoundState.PREFLIGHT, FederatedRoundState.ABORTED}
        ),
        FederatedRoundState.PREFLIGHT: frozenset(
            {FederatedRoundState.COLLECTING, FederatedRoundState.ABORTED}
        ),
        FederatedRoundState.COLLECTING: frozenset(
            {FederatedRoundState.AGGREGATING, FederatedRoundState.ABORTED}
        ),
        FederatedRoundState.AGGREGATING: frozenset(
            {FederatedRoundState.EVALUATING, FederatedRoundState.ABORTED}
        ),
        FederatedRoundState.EVALUATING: frozenset(
            {
                FederatedRoundState.HELD,
                FederatedRoundState.PROMOTED,
                FederatedRoundState.ABORTED,
            }
        ),
        FederatedRoundState.HELD: frozenset(
            {FederatedRoundState.EVALUATING, FederatedRoundState.ABORTED}
        ),
        FederatedRoundState.PROMOTED: frozenset(),
        FederatedRoundState.ABORTED: frozenset(),
    }
)


class FederatedRoundStateError(ValueError):
    """Raised when serialized lifecycle state is malformed or unsupported."""


class FederatedRoundTransitionError(ValueError):
    """Raised when a lifecycle transition is skipped, backward, or terminal."""


def allowed_round_transitions(
    state: FederatedRoundState,
) -> tuple[FederatedRoundState, ...]:
    """Return allowed targets in canonical lifecycle order."""

    _require_state(state)
    allowed = FEDERATED_ROUND_TRANSITIONS[state]
    return tuple(
        candidate for candidate in FEDERATED_ROUND_STATES if candidate in allowed
    )


def can_transition_round(
    current: FederatedRoundState,
    target: FederatedRoundState,
) -> bool:
    """Return whether one explicit state transition is allowed."""

    _require_state(current)
    _require_state(target)
    return target in FEDERATED_ROUND_TRANSITIONS[current]


def validate_round_transition(
    current: FederatedRoundState,
    target: FederatedRoundState,
) -> None:
    """Raise a typed error unless ``current -> target`` is allowed."""

    if not can_transition_round(current, target):
        raise FederatedRoundTransitionError(
            f"invalid federated round transition: {current.value} -> {target.value}"
        )


@dataclass(frozen=True)
class FederatedRoundLifecycle:
    """Immutable, metadata-only state for one federated training round."""

    state: FederatedRoundState = FederatedRoundState.PLANNED
    schema_version: str = FEDERATED_ROUND_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_state(self.state)
        if self.schema_version != FEDERATED_ROUND_SCHEMA_VERSION:
            raise FederatedRoundStateError(
                "unsupported federated round lifecycle schema"
            )

    @property
    def is_terminal(self) -> bool:
        """Return whether no further transitions are permitted."""

        return self.state in FEDERATED_ROUND_TERMINAL_STATES

    def allowed_transitions(self) -> tuple[FederatedRoundState, ...]:
        """Return the legal next states in canonical order."""

        return allowed_round_transitions(self.state)

    def can_transition_to(self, target: FederatedRoundState) -> bool:
        """Return whether ``target`` is a legal next state."""

        return can_transition_round(self.state, target)

    def transition_to(self, target: FederatedRoundState) -> FederatedRoundLifecycle:
        """Validate and return a new lifecycle in ``target`` state."""

        validate_round_transition(self.state, target)
        return FederatedRoundLifecycle(state=target)

    def to_dict(self) -> dict[str, str]:
        """Return the versioned metadata-only representation."""

        return {
            "schema_version": self.schema_version,
            "state": self.state.value,
        }

    def to_json(self) -> str:
        """Return deterministic JSON with stable field ordering."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FederatedRoundLifecycle:
        """Parse a strict lifecycle payload without retaining extra metadata."""

        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "state",
        }:
            raise FederatedRoundStateError("invalid federated round lifecycle payload")
        schema_version = payload["schema_version"]
        state_value = payload["state"]
        if (
            not isinstance(schema_version, str)
            or schema_version != FEDERATED_ROUND_SCHEMA_VERSION
        ):
            raise FederatedRoundStateError(
                "unsupported federated round lifecycle schema"
            )
        if not isinstance(state_value, str):
            raise FederatedRoundStateError("unknown federated round state")
        try:
            state = FederatedRoundState(state_value)
        except ValueError:
            raise FederatedRoundStateError("unknown federated round state") from None
        return cls(state=state, schema_version=schema_version)

    @classmethod
    def from_json(cls, payload: str) -> FederatedRoundLifecycle:
        """Parse lifecycle JSON and replace parser details with a safe error."""

        try:
            decoded = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            raise FederatedRoundStateError(
                "invalid federated round lifecycle JSON"
            ) from None
        if not isinstance(decoded, Mapping):
            raise FederatedRoundStateError("invalid federated round lifecycle payload")
        return cls.from_dict(decoded)


def _require_state(state: object) -> None:
    if not isinstance(state, FederatedRoundState):
        raise FederatedRoundStateError("unknown federated round state")


__all__ = [
    "FEDERATED_ROUND_SCHEMA_VERSION",
    "FEDERATED_ROUND_STATES",
    "FEDERATED_ROUND_TERMINAL_STATES",
    "FEDERATED_ROUND_TRANSITIONS",
    "FederatedRoundLifecycle",
    "FederatedRoundState",
    "FederatedRoundStateError",
    "FederatedRoundTransitionError",
    "allowed_round_transitions",
    "can_transition_round",
    "validate_round_transition",
]
