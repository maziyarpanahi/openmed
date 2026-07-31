"""Deterministic shadow -> canary -> stable rollout state machine.

A challenger model runs in ``SHADOW``, is promoted to ``CANARY`` only when its
release gate is ``RELEASABLE``, and reaches ``STABLE`` only after passing again
on the canary set. The machine is keyed by ``(family, tier, format)`` so every
model surface rolls out independently, and its whole state is reconstructable
from the committed ``gates/rollout_state.json`` document plus the gate reports
passed to :meth:`RolloutStateMachine.advance`.

The machine is pure and offline: it never calls a live model registry or the
Hugging Face Hub. It stores, per key, the current phase, the timestamp the
phase was entered, and two opaque *rollout target* pointers -- the challenger
version being rolled out (``target``) and the last version that reached
``STABLE`` (``last_green``).

Rollout-target boundary
-----------------------
A rollout target is treated as an opaque version pointer -- a short string such
as ``"v3"`` -- supplied by the caller. The machine deliberately does not import
or depend on any registry / lineage record schema (for example the SemVer and
pointer-store work tracked separately under OM-047): any pointer store that can
name the version under rollout as a string satisfies this contract. Keeping the
public surface the finite-state machine, not a registry schema, lets the
eventual registry own the pointer format without reshaping this module.

Rollback here is the *transition*, not the *decision*: the machine exposes a
guarded ``ROLLED_BACK`` transition that a caller may invoke, but the policy that
decides when to roll back lives elsewhere.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from openmed.core.baseline import baseline_key
from openmed.eval.release_gates import RELEASABLE, GateReport

ROLLOUT_STATE_SCHEMA_VERSION = 1
ROLLOUT_STATE_PATH = (
    Path(__file__).resolve().parents[2] / "gates" / "rollout_state.json"
)

PHASE_SHADOW = "shadow"
PHASE_CANARY = "canary"
PHASE_STABLE = "stable"
PHASE_ROLLED_BACK = "rolled_back"

#: Every phase the machine can occupy, in forward-promotion order.
PHASES: tuple[str, ...] = (PHASE_SHADOW, PHASE_CANARY, PHASE_STABLE, PHASE_ROLLED_BACK)

#: Adjacent forward promotion for each phase (missing => no forward step).
_FORWARD_PHASE: Mapping[str, str] = {
    PHASE_SHADOW: PHASE_CANARY,
    PHASE_CANARY: PHASE_STABLE,
}

#: The complete set of legal ``(from_phase, to_phase)`` transitions. Any pair
#: outside this set -- a skip such as ``shadow -> stable`` or a rollback from a
#: phase with nothing live -- is rejected before the state is touched.
LEGAL_TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {
        (PHASE_SHADOW, PHASE_CANARY),
        (PHASE_CANARY, PHASE_STABLE),
        (PHASE_CANARY, PHASE_ROLLED_BACK),
        (PHASE_STABLE, PHASE_ROLLED_BACK),
    }
)

#: Phases whose entry is guarded by a ``RELEASABLE`` gate report.
_GATE_GUARDED_PHASES: frozenset[str] = frozenset({PHASE_CANARY, PHASE_STABLE})

#: Maximum dwell window per phase, in seconds. ``None`` means the phase never
#: expires (a shipped ``STABLE`` surface and a ``ROLLED_BACK`` surface both
#: rest until a new challenger is seeded).
DEFAULT_DWELL_WINDOWS: Mapping[str, int | None] = {
    PHASE_SHADOW: 7 * 24 * 60 * 60,
    PHASE_CANARY: 3 * 24 * 60 * 60,
    PHASE_STABLE: None,
    PHASE_ROLLED_BACK: None,
}


class RolloutError(Exception):
    """Base class for every rollout state-machine error."""


class IllegalTransitionError(RolloutError):
    """Raised when a requested phase transition is not legal."""


class GateNotReleasableError(RolloutError):
    """Raised when a guarded promotion lacks a ``RELEASABLE`` gate report."""


class RolloutStateError(RolloutError):
    """Raised when persisted state or a gate report is malformed."""


Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _format_ts(moment: datetime) -> str:
    return _coerce_utc(moment).isoformat()


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _coerce_utc(value)
    try:
        return _coerce_utc(datetime.fromisoformat(str(value)))
    except (TypeError, ValueError) as exc:
        raise RolloutStateError(f"invalid rollout timestamp: {value!r}") from exc


@dataclass(frozen=True)
class PhaseState:
    """Immutable rollout state for one ``(family, tier, format)`` key."""

    family: str
    tier: str
    format: str
    phase: str
    entered_at: datetime
    target: str | None = None
    last_green: str | None = None

    @property
    def key(self) -> str:
        """Return the canonical ``family::tier::format`` store key."""

        return baseline_key(self.family, self.tier, self.format)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible payload for this entry."""

        return {
            "key": self.key,
            "family": self.family,
            "tier": self.tier,
            "format": self.format,
            "phase": self.phase,
            "entered_at": _format_ts(self.entered_at),
            "target": self.target,
            "last_green": self.last_green,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseState":
        """Restore a phase entry from its JSON-compatible payload."""

        phase = str(data.get("phase", ""))
        if phase not in PHASES:
            raise RolloutStateError(f"unknown rollout phase: {phase!r}")
        return cls(
            family=str(data.get("family", "")),
            tier=str(data.get("tier", "")),
            format=str(data.get("format", "")),
            phase=phase,
            entered_at=_parse_ts(data.get("entered_at")),
            target=(None if data.get("target") is None else str(data["target"])),
            last_green=(
                None if data.get("last_green") is None else str(data["last_green"])
            ),
        )


@dataclass
class RolloutStateMachine:
    """Guarded shadow/canary/stable state machine over rollout keys."""

    entries: dict[str, PhaseState] = field(default_factory=dict)
    dwell_windows: Mapping[str, int | None] = field(
        default_factory=lambda: dict(DEFAULT_DWELL_WINDOWS)
    )
    clock: Clock = _utcnow

    def __post_init__(self) -> None:
        self.dwell_windows = dict(self.dwell_windows)

    # -- queries ---------------------------------------------------------

    def phase_state(self, family: str, tier: str, format: str) -> PhaseState | None:
        """Return the recorded :class:`PhaseState` for a key, or ``None``."""

        return self.entries.get(baseline_key(family, tier, format))

    def current_phase(self, family: str, tier: str, format: str) -> str:
        """Return the current phase, defaulting to ``SHADOW`` when unseeded."""

        state = self.phase_state(family, tier, format)
        return state.phase if state is not None else PHASE_SHADOW

    def dwell_expired(
        self,
        family: str,
        tier: str,
        format: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Return whether the current phase has outlived its dwell window.

        Phases with no configured window (``STABLE``, ``ROLLED_BACK``) and
        unseeded keys never report as expired.
        """

        state = self.phase_state(family, tier, format)
        if state is None:
            return False
        window = self.dwell_windows.get(state.phase)
        if window is None:
            return False
        moment = _coerce_utc(now) if now is not None else self.clock()
        elapsed = (moment - _coerce_utc(state.entered_at)).total_seconds()
        return elapsed > float(window)

    # -- mutations -------------------------------------------------------

    def seed(
        self,
        family: str,
        tier: str,
        format: str,
        *,
        target: str | None = None,
        now: datetime | None = None,
    ) -> PhaseState:
        """Register a challenger in ``SHADOW`` and return its state."""

        moment = _coerce_utc(now) if now is not None else self.clock()
        state = PhaseState(
            family=family,
            tier=tier,
            format=format,
            phase=PHASE_SHADOW,
            entered_at=moment,
            target=target,
            last_green=None,
        )
        self.entries[state.key] = state
        return state

    def advance(
        self,
        family: str,
        tier: str,
        format: str,
        gate_report: GateReport,
        *,
        target: str | None = None,
        now: datetime | None = None,
    ) -> PhaseState:
        """Promote a key to the next phase, guarded by ``gate_report``.

        The next phase is the adjacent forward step (``SHADOW -> CANARY`` or
        ``CANARY -> STABLE``). The promotion is refused unless
        ``gate_report.decision`` is ``RELEASABLE`` and the report's coordinates
        match the key. Advancing a key that has no forward step (``STABLE`` or
        ``ROLLED_BACK``) raises :class:`IllegalTransitionError`.
        """

        from_phase = self.current_phase(family, tier, format)
        to_phase = _FORWARD_PHASE.get(from_phase)
        if to_phase is None:
            raise IllegalTransitionError(
                f"cannot advance {baseline_key(family, tier, format)} from "
                f"terminal phase {from_phase!r}"
            )
        return self.transition(
            family,
            tier,
            format,
            to_phase,
            gate_report=gate_report,
            target=target,
            now=now,
        )

    def rollback(
        self,
        family: str,
        tier: str,
        format: str,
        *,
        now: datetime | None = None,
    ) -> PhaseState:
        """Roll a live ``CANARY`` or ``STABLE`` key back to its last green target.

        Rollback is unguarded by a gate report -- it is the failure path -- but
        it is still only legal from a phase with something live. The rolled-back
        state's ``target`` reverts to ``last_green``.
        """

        return self.transition(
            family, tier, format, PHASE_ROLLED_BACK, gate_report=None, now=now
        )

    def transition(
        self,
        family: str,
        tier: str,
        format: str,
        to_phase: str,
        *,
        gate_report: GateReport | None = None,
        target: str | None = None,
        now: datetime | None = None,
    ) -> PhaseState:
        """Apply a single validated transition and return the new state.

        This is the core all mutations funnel through. It validates the
        ``(from_phase, to_phase)`` pair against :data:`LEGAL_TRANSITIONS`, then
        the gate guard, and only then replaces the stored entry -- so a rejected
        transition leaves the persisted state unchanged.
        """

        if to_phase not in PHASES:
            raise RolloutStateError(f"unknown target phase: {to_phase!r}")

        key = baseline_key(family, tier, format)
        current = self.entries.get(key)
        from_phase = current.phase if current is not None else PHASE_SHADOW

        if (from_phase, to_phase) not in LEGAL_TRANSITIONS:
            raise IllegalTransitionError(
                f"illegal transition {from_phase!r} -> {to_phase!r} for {key}"
            )

        if to_phase in _GATE_GUARDED_PHASES:
            self._require_releasable(family, tier, format, gate_report)

        moment = _coerce_utc(now) if now is not None else self.clock()
        carried_target = (
            target
            if target is not None
            else (current.target if current is not None else None)
        )
        last_green = current.last_green if current is not None else None

        if to_phase == PHASE_STABLE:
            last_green = carried_target
        elif to_phase == PHASE_ROLLED_BACK:
            carried_target = last_green

        new_state = PhaseState(
            family=family,
            tier=tier,
            format=format,
            phase=to_phase,
            entered_at=moment,
            target=carried_target,
            last_green=last_green,
        )
        self.entries[key] = new_state
        return new_state

    def _require_releasable(
        self,
        family: str,
        tier: str,
        format: str,
        gate_report: GateReport | None,
    ) -> None:
        if gate_report is None:
            raise GateNotReleasableError(
                "a gate report is required to promote "
                f"{baseline_key(family, tier, format)}"
            )
        report_key = baseline_key(
            gate_report.family, gate_report.tier, gate_report.format
        )
        if report_key != baseline_key(family, tier, format):
            raise RolloutStateError(
                "gate report coordinates "
                f"{report_key} do not match rollout key "
                f"{baseline_key(family, tier, format)}"
            )
        if gate_report.decision != RELEASABLE:
            raise GateNotReleasableError(
                f"gate decision is {gate_report.decision!r}, expected {RELEASABLE!r}"
            )

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible document for the whole machine."""

        return {
            "schema_version": ROLLOUT_STATE_SCHEMA_VERSION,
            "entries": {
                key: state.to_dict() for key, state in sorted(self.entries.items())
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        dwell_windows: Mapping[str, int | None] | None = None,
        clock: Clock | None = None,
    ) -> "RolloutStateMachine":
        """Restore a machine from its JSON-compatible document."""

        version = data.get("schema_version")
        if version != ROLLOUT_STATE_SCHEMA_VERSION:
            raise RolloutStateError(
                f"unsupported rollout state schema_version: {version!r}"
            )
        raw_entries = data.get("entries")
        if not isinstance(raw_entries, Mapping):
            raise RolloutStateError("rollout state 'entries' must be an object")
        entries: dict[str, PhaseState] = {}
        for stored_key, payload in raw_entries.items():
            if not isinstance(payload, Mapping):
                raise RolloutStateError(
                    f"rollout entry {stored_key!r} must be an object"
                )
            state = PhaseState.from_dict(payload)
            entries[state.key] = state
        machine = cls(
            entries=entries,
            dwell_windows=dict(dwell_windows or DEFAULT_DWELL_WINDOWS),
            clock=clock or _utcnow,
        )
        return machine

    def to_json(self) -> str:
        """Return the machine's canonical, sorted JSON document."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def load(
        cls,
        path: str | Path = ROLLOUT_STATE_PATH,
        *,
        dwell_windows: Mapping[str, int | None] | None = None,
        clock: Clock | None = None,
    ) -> "RolloutStateMachine":
        """Load and validate a committed rollout state document."""

        state_path = Path(path)
        with state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise RolloutStateError(f"{state_path} must contain a JSON object")
        return cls.from_dict(payload, dwell_windows=dwell_windows, clock=clock)

    def save(self, path: str | Path = ROLLOUT_STATE_PATH) -> Path:
        """Write the machine to ``path`` with deterministic formatting."""

        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(self.to_json(), encoding="utf-8")
        return state_path

    def copy(self) -> "RolloutStateMachine":
        """Return a deep copy of the machine, sharing the clock callable."""

        return RolloutStateMachine(
            entries={key: replace(state) for key, state in self.entries.items()},
            dwell_windows=copy.deepcopy(dict(self.dwell_windows)),
            clock=self.clock,
        )


__all__ = [
    "DEFAULT_DWELL_WINDOWS",
    "LEGAL_TRANSITIONS",
    "PHASES",
    "PHASE_CANARY",
    "PHASE_ROLLED_BACK",
    "PHASE_SHADOW",
    "PHASE_STABLE",
    "ROLLOUT_STATE_PATH",
    "ROLLOUT_STATE_SCHEMA_VERSION",
    "GateNotReleasableError",
    "IllegalTransitionError",
    "PhaseState",
    "RolloutError",
    "RolloutStateError",
    "RolloutStateMachine",
]
