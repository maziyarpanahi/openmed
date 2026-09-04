"""Deterministic, PHI-safe lifecycle records for privacy waivers.

The ledger stores only a controlled lifecycle event, an opaque waiver
identifier, and an opaque policy identifier.  It intentionally has no fields
for identities, finding text, reasons, or timestamps.  This keeps the record
surface safe for local audit output while leaving the meaning and approval
authority of a waiver with the caller's governance process.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

WAIVER_LEDGER_SCHEMA_VERSION: Final[int] = 1

_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,255}$"
)


class WaiverEventType(str, Enum):
    """Controlled events in a privacy-waiver lifecycle."""

    CREATE = "create"
    APPROVE = "approve"
    SUPERSEDE = "supersede"
    REVOKE = "revoke"
    EXPIRE = "expire"


class WaiverState(str, Enum):
    """Current state derived by replaying a waiver's lifecycle events."""

    PENDING = "pending"
    # These aliases keep the event vocabulary and the state vocabulary
    # readable to callers without creating a second serialized state.
    CREATED = "pending"
    ACTIVE = "active"
    APPROVED = "active"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"
    EXPIRED = "expired"


_STATE_ORDER: Final[tuple[WaiverState, ...]] = (
    WaiverState.PENDING,
    WaiverState.ACTIVE,
    WaiverState.SUPERSEDED,
    WaiverState.REVOKED,
    WaiverState.EXPIRED,
)
_TARGET_STATE: Final[Mapping[WaiverEventType, WaiverState]] = {
    WaiverEventType.CREATE: WaiverState.PENDING,
    WaiverEventType.APPROVE: WaiverState.ACTIVE,
    WaiverEventType.SUPERSEDE: WaiverState.SUPERSEDED,
    WaiverEventType.REVOKE: WaiverState.REVOKED,
    WaiverEventType.EXPIRE: WaiverState.EXPIRED,
}


class WaiverLedgerError(ValueError):
    """Base error for malformed or invalid waiver ledger operations."""


class InvalidWaiverTransitionError(WaiverLedgerError):
    """Raised when an event is not valid for the waiver's current state."""


class InvalidWaiverIdentifierError(WaiverLedgerError):
    """Raised when a waiver or policy reference is not an opaque token."""


class UnknownWaiverError(InvalidWaiverTransitionError):
    """Raised when an event references a waiver that was not created."""


# Short aliases make the error intent easy to discover while preserving one
# implementation and one safe error-message policy.
InvalidTransitionError = InvalidWaiverTransitionError
WaiverTransitionError = InvalidWaiverTransitionError


def _coerce_event_type(value: WaiverEventType | str) -> WaiverEventType:
    if isinstance(value, WaiverEventType):
        return value
    if not isinstance(value, str) or not value.strip():
        raise WaiverLedgerError("event_type must be a supported lifecycle event")
    canonical = value.strip().lower().replace("_", "-").replace(" ", "-")
    try:
        return WaiverEventType(canonical)
    except ValueError as exc:
        raise WaiverLedgerError(
            "event_type must be a supported lifecycle event"
        ) from exc


def _coerce_state(value: WaiverState | str) -> WaiverState:
    if isinstance(value, WaiverState):
        return value
    if not isinstance(value, str) or not value.strip():
        raise WaiverLedgerError("state must be a supported waiver state")
    canonical = value.strip().lower().replace("_", "-").replace(" ", "-")
    try:
        return WaiverState(canonical)
    except ValueError as exc:
        raise WaiverLedgerError("state must be a supported waiver state") from exc


def _identifier(value: Any, field_name: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise InvalidWaiverIdentifierError(
            f"{field_name} must be an opaque identifier token"
        )
    return value


def _sequence(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WaiverLedgerError("event sequence must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class WaiverLifecycleEvent:
    """One immutable, PHI-safe transition in a waiver lifecycle."""

    sequence: int
    event_type: WaiverEventType
    waiver_id: str
    policy_id: str
    state: WaiverState
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", _sequence(self.sequence))
        event_type = _coerce_event_type(self.event_type)
        state = _coerce_state(self.state)
        waiver_id = _identifier(self.waiver_id, "waiver_id")
        policy_id = _identifier(self.policy_id, "policy_id")
        superseded_by = _identifier(
            self.superseded_by,
            "superseded_by",
            required=False,
        )

        if state is not _TARGET_STATE[event_type]:
            raise WaiverLedgerError("event state does not match its lifecycle event")
        if event_type is not WaiverEventType.SUPERSEDE and superseded_by is not None:
            raise WaiverLedgerError("superseded_by is only valid for a supersede event")
        if superseded_by == waiver_id:
            raise WaiverLedgerError("superseded_by must reference a different waiver")

        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "waiver_id", waiver_id)
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "superseded_by", superseded_by)

    @property
    def event(self) -> WaiverEventType:
        """Alias for callers that use ``event`` for the event kind."""

        return self.event_type

    @property
    def kind(self) -> WaiverEventType:
        """Alias for callers that use ``kind`` for the event type."""

        return self.event_type

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "WaiverLifecycleEvent":
        """Load one strict JSON-compatible event record."""

        if not isinstance(payload, Mapping):
            raise WaiverLedgerError("waiver event must be an object")
        allowed = {
            "event_type",
            "policy_id",
            "sequence",
            "state",
            "superseded_by",
            "waiver_id",
        }
        if set(payload) - allowed:
            raise WaiverLedgerError("waiver event contains unsupported fields")
        required = allowed - {"superseded_by"}
        if not required.issubset(payload):
            raise WaiverLedgerError("waiver event is missing required fields")
        return cls(
            sequence=payload["sequence"],
            event_type=payload["event_type"],
            waiver_id=payload["waiver_id"],
            policy_id=payload["policy_id"],
            state=payload["state"],
            superseded_by=payload.get("superseded_by"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic, PHI-safe event representation."""

        payload: dict[str, Any] = {
            "event_type": self.event_type.value,
            "policy_id": self.policy_id,
            "sequence": self.sequence,
            "state": self.state.value,
            "waiver_id": self.waiver_id,
        }
        if self.superseded_by is not None:
            payload["superseded_by"] = self.superseded_by
        return payload


# Common names for the immutable record, kept as aliases rather than separate
# classes so equality and serialization remain unambiguous.
WaiverEvent = WaiverLifecycleEvent
WaiverRecord = WaiverLifecycleEvent
WaiverLifecycleEventType = WaiverEventType
WaiverStatus = WaiverState


class WaiverLedger:
    """An append-only, local-only privacy-waiver lifecycle ledger.

    The ledger derives state by replaying its immutable event records.  It
    accepts no free-form metadata, and all transition methods return the new
    event while keeping the read-only ``events`` view available for audit
    serialization.
    """

    def __init__(
        self,
        events: Iterable[WaiverLifecycleEvent | Mapping[str, Any]] = (),
    ) -> None:
        self._events: tuple[WaiverLifecycleEvent, ...] = ()
        self._states: dict[str, WaiverState] = {}
        self._policies: dict[str, str] = {}
        for event in events:
            resolved = (
                event
                if isinstance(event, WaiverLifecycleEvent)
                else WaiverLifecycleEvent.from_mapping(event)
            )
            self._append_event(resolved)

    @property
    def events(self) -> tuple[WaiverLifecycleEvent, ...]:
        """Return immutable lifecycle records in append order."""

        return self._events

    @property
    def records(self) -> tuple[WaiverLifecycleEvent, ...]:
        """Alias for :attr:`events`."""

        return self._events

    def __len__(self) -> int:
        return len(self._events)

    def _append_event(self, event: WaiverLifecycleEvent) -> None:
        if event.sequence != len(self._events):
            raise WaiverLedgerError("event sequence must be contiguous")

        current = self._states.get(event.waiver_id)
        if event.event_type is WaiverEventType.CREATE:
            if current is not None:
                raise InvalidWaiverTransitionError(
                    "create is only valid for an unknown waiver"
                )
        else:
            if current is None:
                raise UnknownWaiverError(
                    "lifecycle events require a previously created waiver"
                )
            expected_policy = self._policies[event.waiver_id]
            if event.policy_id != expected_policy:
                raise InvalidWaiverTransitionError(
                    "policy_id must remain stable across a waiver lifecycle"
                )
            if (
                current is not WaiverState.PENDING
                and event.event_type is WaiverEventType.APPROVE
            ):
                raise InvalidWaiverTransitionError(
                    "approve is only valid for a pending waiver"
                )
            if current is not WaiverState.ACTIVE and event.event_type in {
                WaiverEventType.SUPERSEDE,
                WaiverEventType.REVOKE,
                WaiverEventType.EXPIRE,
            }:
                raise InvalidWaiverTransitionError(
                    "terminal waiver events are only valid for an active waiver"
                )

        self._events = (*self._events, event)
        self._states[event.waiver_id] = event.state
        self._policies.setdefault(event.waiver_id, event.policy_id)

    def record(
        self,
        event_type: WaiverEventType | str,
        waiver_id: str,
        policy_id: str | None = None,
        *,
        superseded_by: str | None = None,
    ) -> WaiverLifecycleEvent:
        """Record one validated lifecycle event.

        ``policy_id`` is required for ``create``.  For later events it may be
        omitted to reuse the policy reference recorded at creation; when
        supplied, it must match that reference.
        """

        resolved_event_type = _coerce_event_type(event_type)
        resolved_waiver_id = _identifier(waiver_id, "waiver_id")
        resolved_policy_id = _identifier(
            policy_id,
            "policy_id",
            required=resolved_event_type is WaiverEventType.CREATE,
        )
        resolved_superseded_by = _identifier(
            superseded_by,
            "superseded_by",
            required=False,
        )

        if resolved_event_type is not WaiverEventType.SUPERSEDE:
            if resolved_superseded_by is not None:
                raise InvalidWaiverTransitionError(
                    "superseded_by is only valid for a supersede event"
                )
        elif resolved_superseded_by == resolved_waiver_id:
            raise InvalidWaiverTransitionError(
                "superseded_by must reference a different waiver"
            )

        if resolved_policy_id is None:
            if resolved_waiver_id not in self._policies:
                raise UnknownWaiverError(
                    "lifecycle events require a previously created waiver"
                )
            resolved_policy_id = self._policies[resolved_waiver_id]

        event = WaiverLifecycleEvent(
            sequence=len(self._events),
            event_type=resolved_event_type,
            waiver_id=resolved_waiver_id,
            policy_id=resolved_policy_id,
            state=_TARGET_STATE[resolved_event_type],
            superseded_by=resolved_superseded_by,
        )
        self._append_event(event)
        return event

    def append(
        self,
        event_type: WaiverEventType | str,
        waiver_id: str,
        policy_id: str | None = None,
        *,
        superseded_by: str | None = None,
    ) -> WaiverLifecycleEvent:
        """Alias for :meth:`record`."""

        return self.record(
            event_type,
            waiver_id,
            policy_id,
            superseded_by=superseded_by,
        )

    def create(self, waiver_id: str, policy_id: str) -> WaiverLifecycleEvent:
        """Record creation of a pending waiver."""

        return self.record(WaiverEventType.CREATE, waiver_id, policy_id)

    def approve(
        self, waiver_id: str, policy_id: str | None = None
    ) -> WaiverLifecycleEvent:
        """Approve a pending waiver, making it active."""

        return self.record(WaiverEventType.APPROVE, waiver_id, policy_id)

    def supersede(
        self,
        waiver_id: str,
        policy_id: str | None = None,
        replacement_waiver_id: str | None = None,
        *,
        superseded_by: str | None = None,
    ) -> WaiverLifecycleEvent:
        """Supersede an active waiver, optionally naming its replacement."""

        if replacement_waiver_id is not None and superseded_by is not None:
            raise InvalidWaiverTransitionError(
                "provide only one superseding waiver reference"
            )
        replacement = (
            replacement_waiver_id
            if replacement_waiver_id is not None
            else superseded_by
        )
        return self.record(
            WaiverEventType.SUPERSEDE,
            waiver_id,
            policy_id,
            superseded_by=replacement,
        )

    def revoke(
        self, waiver_id: str, policy_id: str | None = None
    ) -> WaiverLifecycleEvent:
        """Revoke an active waiver."""

        return self.record(WaiverEventType.REVOKE, waiver_id, policy_id)

    def expire(
        self, waiver_id: str, policy_id: str | None = None
    ) -> WaiverLifecycleEvent:
        """Record explicit expiration of an active waiver."""

        return self.record(WaiverEventType.EXPIRE, waiver_id, policy_id)

    # Verbose method names make event-oriented integrations self-documenting.
    record_event = record
    record_create = create
    record_approve = approve
    record_supersede = supersede
    record_revoke = revoke
    record_expire = expire

    def current_state(self, waiver_id: str) -> WaiverState:
        """Return the current state for one known waiver."""

        resolved_waiver_id = _identifier(waiver_id, "waiver_id")
        try:
            return self._states[resolved_waiver_id]
        except KeyError as exc:
            raise UnknownWaiverError("waiver has no lifecycle record") from exc

    def state_counts(self) -> dict[str, int]:
        """Return counts for every state in a stable order."""

        counts = {state.value: 0 for state in _STATE_ORDER}
        for state in self._states.values():
            counts[state.value] += 1
        return counts

    def active_state_counts(self) -> dict[str, int]:
        """Return deterministic counts of current waiver states."""

        return self.state_counts()

    @property
    def active_count(self) -> int:
        """Return the number of currently active waivers."""

        return self._count_state(WaiverState.ACTIVE)

    @property
    def waiver_count(self) -> int:
        """Return the number of distinct waivers represented in the ledger."""

        return len(self._states)

    @property
    def counts_by_state(self) -> dict[str, int]:
        """Property alias for :meth:`state_counts`."""

        return self.state_counts()

    def _count_state(self, state: WaiverState) -> int:
        return sum(current is state for current in self._states.values())

    def render_active_state_counts(self) -> str:
        """Render only aggregate state counts as deterministic compact JSON."""

        return json.dumps(
            self.state_counts(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, identifier-only ledger representation."""

        return {
            "events": [event.to_dict() for event in self._events],
            "schema_version": WAIVER_LEDGER_SCHEMA_VERSION,
            "state_counts": self.state_counts(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the ledger deterministically without free-form content."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int | None = 2) -> Path:
        """Write deterministic ledger JSON to a local path."""

        output_path = Path(path)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "WaiverLedger":
        """Load and validate a strict JSON-compatible ledger mapping."""

        if not isinstance(payload, Mapping):
            raise WaiverLedgerError("waiver ledger must be an object")
        allowed = {"events", "schema_version", "state_counts"}
        if set(payload) - allowed:
            raise WaiverLedgerError("waiver ledger contains unsupported fields")
        if payload.get("schema_version") != WAIVER_LEDGER_SCHEMA_VERSION:
            raise WaiverLedgerError("unsupported waiver ledger schema version")
        events = payload.get("events")
        if not isinstance(events, list):
            raise WaiverLedgerError("waiver ledger events must be a list")
        ledger = cls(WaiverLifecycleEvent.from_mapping(event) for event in events)
        if (
            "state_counts" in payload
            and payload["state_counts"] != ledger.state_counts()
        ):
            raise WaiverLedgerError("waiver ledger state counts do not match events")
        return ledger

    @classmethod
    def from_json(cls, payload: str) -> "WaiverLedger":
        """Load a ledger from deterministic JSON text."""

        try:
            decoded = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise WaiverLedgerError("waiver ledger JSON is invalid") from exc
        return cls.from_mapping(decoded)


PrivacyWaiverLedger = WaiverLedger


def render_active_state_counts(ledger: WaiverLedger) -> str:
    """Render aggregate current-state counts for a waiver ledger."""

    if not isinstance(ledger, WaiverLedger):
        raise TypeError("ledger must be a WaiverLedger")
    return ledger.render_active_state_counts()


__all__ = [
    "InvalidTransitionError",
    "InvalidWaiverIdentifierError",
    "InvalidWaiverTransitionError",
    "PrivacyWaiverLedger",
    "UnknownWaiverError",
    "WaiverEvent",
    "WaiverEventType",
    "WaiverLedger",
    "WaiverLedgerError",
    "WaiverLifecycleEvent",
    "WaiverLifecycleEventType",
    "WaiverRecord",
    "WaiverState",
    "WaiverStatus",
    "WaiverTransitionError",
    "WAIVER_LEDGER_SCHEMA_VERSION",
    "render_active_state_counts",
]
