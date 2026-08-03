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
``STABLE`` (``last_green``). During ``CANARY``, ``target`` is the distinct
canary pointer while ``last_green`` remains the ``latest`` pointer.

Rollout-target boundary
-----------------------
A rollout target is treated as an opaque version pointer -- a short string such
as ``"v3"`` -- supplied by the caller. The machine deliberately does not import
or depend on any registry / lineage record schema (for example the SemVer and
pointer-store work tracked separately under OM-047): any pointer store that can
name the version under rollout as a string satisfies this contract. Keeping the
public surface the finite-state machine, not a registry schema, lets the
eventual registry own the pointer format without reshaping this module.

Rollback here is the *transition*, not the regression-scoring decision: the
coordinator consumes an already-scored :class:`~openmed.eval.release_gates.GateReport`,
invokes a caller-supplied local pointer flip, and only then records
``ROLLED_BACK``. The policy that calculates whether a metric crossed its gate
tolerance remains in the release-gate layer.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openmed.core.audit import manifest_hash, stable_hash
from openmed.core.baseline import baseline_key, get_baseline, load_baseline_store
from openmed.eval import harness
from openmed.eval.history import BenchmarkHistoryDiff, diff_against_baseline
from openmed.eval.release_gates import QUARANTINED, RELEASABLE, GateReport
from openmed.eval.report import BenchmarkReport

ROLLOUT_STATE_SCHEMA_VERSION = 1
ROLLOUT_AUDIT_SCHEMA_VERSION = 1
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
_REQUIRED_PHASE_GATES: tuple[str, ...] = (
    "G1a",
    "G1b",
    "G2",
    "G3",
    "G4",
    "G5",
    "G6",
    "G7",
    "G8",
)

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
PointerAction = Callable[["PhaseState", GateReport], str]


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

    @property
    def canary_pointer(self) -> str | None:
        """Return the canary pointer while this key is in ``CANARY``."""

        return self.target if self.phase == PHASE_CANARY else None

    @property
    def latest_pointer(self) -> str | None:
        """Return the shipping pointer without consulting a live registry."""

        if self.phase == PHASE_STABLE:
            return self.target
        return self.last_green

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
            "pointers": {
                "canary": self.canary_pointer,
                "latest": self.latest_pointer,
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseState":
        """Restore a phase entry from its JSON-compatible payload."""

        phase = str(data.get("phase", ""))
        if phase not in PHASES:
            raise RolloutStateError(f"unknown rollout phase: {phase!r}")
        state = cls(
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
        pointers = data.get("pointers")
        if pointers is not None:
            if not isinstance(pointers, Mapping):
                raise RolloutStateError("rollout entry 'pointers' must be an object")
            expected = {
                "canary": state.canary_pointer,
                "latest": state.latest_pointer,
            }
            if dict(pointers) != expected:
                raise RolloutStateError(
                    f"rollout pointers do not match phase targets for {state.key}"
                )
        return state


@dataclass(frozen=True)
class RolloutAuditRecord:
    """PHI-free evidence for one gate outcome or phase transition."""

    family: str
    tier: str
    format: str
    from_phase: str
    to_phase: str
    recorded_at: datetime
    gate_status: str
    outcome: str
    pointer_name: str | None = None
    pointer_target: str | None = None
    gate_report_hash: str | None = None
    baseline_key: str | None = None
    rollback_latency_seconds: float | None = None
    schema_version: int = ROLLOUT_AUDIT_SCHEMA_VERSION

    @property
    def key(self) -> str:
        """Return the canonical rollout key."""

        return baseline_key(self.family, self.tier, self.format)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-compatible audit record."""

        return {
            "baseline_key": self.baseline_key,
            "family": self.family,
            "format": self.format,
            "from_phase": self.from_phase,
            "gate_report_hash": self.gate_report_hash,
            "gate_status": self.gate_status,
            "key": self.key,
            "outcome": self.outcome,
            "pointer_name": self.pointer_name,
            "pointer_target": self.pointer_target,
            "recorded_at": _format_ts(self.recorded_at),
            "rollback_latency_seconds": self.rollback_latency_seconds,
            "schema_version": self.schema_version,
            "tier": self.tier,
            "to_phase": self.to_phase,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RolloutAuditRecord":
        """Restore and validate one persisted audit record."""

        version = int(data.get("schema_version", ROLLOUT_AUDIT_SCHEMA_VERSION))
        if version != ROLLOUT_AUDIT_SCHEMA_VERSION:
            raise RolloutStateError(
                f"unsupported rollout audit schema_version: {version!r}"
            )
        from_phase = str(data.get("from_phase", ""))
        to_phase = str(data.get("to_phase", ""))
        for phase in (from_phase, to_phase):
            if phase not in PHASES:
                raise RolloutStateError(f"unknown rollout audit phase: {phase!r}")
        gate_status = str(data.get("gate_status", ""))
        if gate_status not in {"green", "red"}:
            raise RolloutStateError(
                f"unknown rollout audit gate status: {gate_status!r}"
            )
        outcome = str(data.get("outcome", ""))
        if not outcome:
            raise RolloutStateError("rollout audit outcome must be non-empty")
        pointer_name = data.get("pointer_name")
        if pointer_name not in {None, "canary", "latest"}:
            raise RolloutStateError(f"unknown rollout audit pointer: {pointer_name!r}")
        latency = data.get("rollback_latency_seconds")
        parsed_latency = None if latency is None else float(latency)
        if parsed_latency is not None and (
            not math.isfinite(parsed_latency) or parsed_latency < 0.0
        ):
            raise RolloutStateError(
                "rollout audit latency must be finite and non-negative"
            )
        record = cls(
            family=str(data.get("family", "")),
            tier=str(data.get("tier", "")),
            format=str(data.get("format", "")),
            from_phase=from_phase,
            to_phase=to_phase,
            recorded_at=_parse_ts(data.get("recorded_at")),
            gate_status=gate_status,
            outcome=outcome,
            pointer_name=(None if pointer_name is None else str(pointer_name)),
            pointer_target=(
                None
                if data.get("pointer_target") is None
                else str(data["pointer_target"])
            ),
            gate_report_hash=(
                None
                if data.get("gate_report_hash") is None
                else str(data["gate_report_hash"])
            ),
            baseline_key=(
                None if data.get("baseline_key") is None else str(data["baseline_key"])
            ),
            rollback_latency_seconds=parsed_latency,
            schema_version=version,
        )
        stored_key = data.get("key")
        if stored_key is not None and str(stored_key) != record.key:
            raise RolloutStateError(
                f"rollout audit key mismatch: {stored_key!r} != {record.key!r}"
            )
        if record.baseline_key is not None and record.baseline_key != record.key:
            raise RolloutStateError(
                f"rollout audit baseline key mismatch for {record.key}"
            )
        return record


@dataclass(frozen=True)
class ShadowSuiteComparison:
    """Offline champion/challenger evidence over one shared fixture suite."""

    suite: str
    fixture_path: Path
    fixture_hash: str
    champion_report: BenchmarkReport
    challenger_report: BenchmarkReport
    champion_diff: BenchmarkHistoryDiff
    challenger_diff: BenchmarkHistoryDiff

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic audit payload without fixture plaintext."""

        return {
            "challenger_diff": self.challenger_diff.to_dict(),
            "challenger_report_hash": stable_hash(self.challenger_report.to_dict()),
            "champion_diff": self.champion_diff.to_dict(),
            "champion_report_hash": stable_hash(self.champion_report.to_dict()),
            "fixture_hash": self.fixture_hash,
            "suite": self.suite,
        }


@dataclass(frozen=True)
class GateApplication:
    """Result of applying one scored phase gate to a rollout key."""

    action: str
    state: PhaseState
    audit_record: RolloutAuditRecord


def run_shadow_comparison(
    fixture_paths: Mapping[str, str | Path],
    *,
    champion_model: str,
    challenger_model: str,
    family: str,
    tier: str,
    format: str,
    baseline_path: str | Path,
    runner: harness.ModelRunner | None = None,
    device: str = "cpu",
    generated_at: str | None = None,
) -> tuple[ShadowSuiteComparison, ...]:
    """Run champion and challenger on identical golden and SHIELD fixtures.

    Both models are evaluated by :func:`openmed.eval.harness.run_suite` using
    the same path, runner, device, and generated timestamp for each suite. Each
    metric report is then diffed against the committed last-green baseline.
    The returned payload contains only metrics and hashes, never fixture text.
    """

    required_suites = {"golden", "shield"}
    normalized_paths = {
        str(suite).strip().lower(): Path(path) for suite, path in fixture_paths.items()
    }
    missing = required_suites - set(normalized_paths)
    if missing:
        raise RolloutStateError(
            "shadow comparison requires fixture paths for: "
            + ", ".join(sorted(missing))
        )

    metadata = {"family": family, "tier": tier, "format": format}
    baseline_store = load_baseline_store(baseline_path)
    baseline_entry = get_baseline(family, tier, format, store=baseline_store)
    if baseline_entry is None:
        raise RolloutStateError(
            f"no committed baseline for {baseline_key(family, tier, format)}"
        )
    comparisons: list[ShadowSuiteComparison] = []
    for suite in ("golden", "shield"):
        fixture_path = normalized_paths[suite]
        if not fixture_path.is_file():
            raise RolloutStateError(f"shadow fixture does not exist: {fixture_path}")
        champion_report = harness.run_suite(
            fixture_path,
            suite=suite,
            model_name=champion_model,
            device=device,
            runner=runner,
            generated_at=generated_at,
            metadata=metadata,
        )
        challenger_report = harness.run_suite(
            fixture_path,
            suite=suite,
            model_name=challenger_model,
            device=device,
            runner=runner,
            generated_at=generated_at,
            metadata=metadata,
        )
        if not isinstance(champion_report, BenchmarkReport) or not isinstance(
            challenger_report, BenchmarkReport
        ):
            raise RolloutStateError("shadow rollout only supports benchmark suites")
        suite_baseline = baseline_entry
        baseline_metrics = baseline_entry.get("metrics")
        if suite == "shield" and isinstance(baseline_metrics, Mapping):
            shield_metrics = baseline_metrics.get("public_shield")
            if isinstance(shield_metrics, Mapping):
                suite_baseline = {**baseline_entry, "metrics": shield_metrics}
        comparisons.append(
            ShadowSuiteComparison(
                suite=suite,
                fixture_path=fixture_path,
                fixture_hash=manifest_hash(fixture_path),
                champion_report=champion_report,
                challenger_report=challenger_report,
                champion_diff=diff_against_baseline(
                    champion_report,
                    suite_baseline,
                    rank_limit=None,
                ),
                challenger_diff=diff_against_baseline(
                    challenger_report,
                    suite_baseline,
                    rank_limit=None,
                ),
            )
        )
    return tuple(comparisons)


@dataclass
class RolloutStateMachine:
    """Guarded shadow/canary/stable state machine over rollout keys."""

    entries: dict[str, PhaseState] = field(default_factory=dict)
    dwell_windows: Mapping[str, int | None] = field(
        default_factory=lambda: dict(DEFAULT_DWELL_WINDOWS)
    )
    clock: Clock = _utcnow
    audit_records: list[RolloutAuditRecord] = field(default_factory=list)

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
        moment = _coerce_utc(now) if now is not None else _coerce_utc(self.clock())
        elapsed = (moment - _coerce_utc(state.entered_at)).total_seconds()
        return elapsed > float(window)

    def mean_rollback_latency_seconds(self) -> float | None:
        """Return the mean recorded detection-to-pointer-flip latency."""

        latencies = [
            record.rollback_latency_seconds
            for record in self.audit_records
            if record.rollback_latency_seconds is not None
        ]
        if not latencies:
            return None
        return sum(latencies) / len(latencies)

    def record_audit(self, record: RolloutAuditRecord) -> RolloutAuditRecord:
        """Append a validated PHI-free audit record."""

        if record.key not in self.entries:
            raise RolloutStateError(
                f"cannot audit an unseeded rollout key: {record.key}"
            )
        self.audit_records.append(record)
        return record

    # -- mutations -------------------------------------------------------

    def seed(
        self,
        family: str,
        tier: str,
        format: str,
        *,
        target: str | None = None,
        last_green: str | None = None,
        now: datetime | None = None,
    ) -> PhaseState:
        """Register a challenger in ``SHADOW`` and return its state.

        ``last_green`` is the shipping ``latest`` pointer while the challenger
        remains shadowed. When omitted, an existing key's prior stable target
        is preserved so seeding a new challenger cannot erase rollback state.
        """

        moment = _coerce_utc(now) if now is not None else _coerce_utc(self.clock())
        previous = self.phase_state(family, tier, format)
        preserved_green = last_green
        if preserved_green is None and previous is not None:
            preserved_green = previous.last_green
            if preserved_green is None and previous.phase == PHASE_STABLE:
                preserved_green = previous.target
        state = PhaseState(
            family=family,
            tier=tier,
            format=format,
            phase=PHASE_SHADOW,
            entered_at=moment,
            target=target,
            last_green=preserved_green,
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
        gate_report: GateReport | None = None,
        detected_at: datetime | None = None,
        pointer_flipped_at: datetime | None = None,
        outcome: str = "rolled_back",
        now: datetime | None = None,
    ) -> PhaseState:
        """Roll a live ``CANARY`` or ``STABLE`` key back to its last green target.

        Rollback is unguarded by a gate report -- it is the failure path -- but
        it is still only legal from a phase with something live. The rolled-back
        state's ``target`` reverts to ``last_green``.
        """

        return self.transition(
            family,
            tier,
            format,
            PHASE_ROLLED_BACK,
            gate_report=gate_report,
            detected_at=detected_at,
            pointer_flipped_at=pointer_flipped_at,
            outcome=outcome,
            now=now,
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
        detected_at: datetime | None = None,
        pointer_flipped_at: datetime | None = None,
        outcome: str | None = None,
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

        moment = _coerce_utc(now) if now is not None else _coerce_utc(self.clock())
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
        pointer_name = "canary" if to_phase == PHASE_CANARY else "latest"
        pointer_target = (
            new_state.canary_pointer
            if pointer_name == "canary"
            else new_state.latest_pointer
        )
        latency: float | None = None
        if detected_at is not None and pointer_flipped_at is not None:
            latency = max(
                0.0,
                (
                    _coerce_utc(pointer_flipped_at) - _coerce_utc(detected_at)
                ).total_seconds(),
            )
        gate_status = (
            "green"
            if gate_report is not None and gate_report.decision == RELEASABLE
            else "red"
        )
        self.audit_records.append(
            RolloutAuditRecord(
                family=family,
                tier=tier,
                format=format,
                from_phase=from_phase,
                to_phase=to_phase,
                recorded_at=moment,
                gate_status=gate_status,
                outcome=outcome or f"{from_phase}_to_{to_phase}",
                pointer_name=pointer_name,
                pointer_target=pointer_target,
                gate_report_hash=(
                    gate_report.repro_hash if gate_report is not None else None
                ),
                baseline_key=key,
                rollback_latency_seconds=latency,
            )
        )
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
            "audit": [record.to_dict() for record in self.audit_records],
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
            if str(stored_key) != state.key:
                raise RolloutStateError(
                    f"rollout entry key mismatch: {stored_key!r} != {state.key!r}"
                )
            entries[state.key] = state
        raw_audit = data.get("audit", [])
        if not isinstance(raw_audit, Sequence) or isinstance(
            raw_audit, (str, bytes, bytearray)
        ):
            raise RolloutStateError("rollout state 'audit' must be an array")
        audit_records = []
        for payload in raw_audit:
            if not isinstance(payload, Mapping):
                raise RolloutStateError("rollout audit entries must be objects")
            record = RolloutAuditRecord.from_dict(payload)
            if record.key not in entries:
                raise RolloutStateError(
                    f"rollout audit references missing key: {record.key}"
                )
            audit_records.append(record)
        machine = cls(
            entries=entries,
            audit_records=audit_records,
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
            audit_records=[replace(record) for record in self.audit_records],
            dwell_windows=copy.deepcopy(dict(self.dwell_windows)),
            clock=self.clock,
        )

    def audit_to_dict(self) -> dict[str, Any]:
        """Return the reconstructable rollout audit document."""

        return {
            "mean_rollback_latency_seconds": self.mean_rollback_latency_seconds(),
            "records": [record.to_dict() for record in self.audit_records],
            "schema_version": ROLLOUT_AUDIT_SCHEMA_VERSION,
            "state_hash": stable_hash(self.to_dict()),
        }

    def save_audit(self, path: str | Path) -> Path:
        """Write rollout audit evidence with deterministic formatting."""

        audit_path = Path(path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(self.audit_to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return audit_path


@dataclass
class RolloutCoordinator:
    """Apply already-scored phase gates and execute local pointer actions.

    The coordinator does not calculate regression tolerances. It trusts only a
    self-consistent :class:`GateReport` produced by ``ReleaseGate`` and treats
    every non-``RELEASABLE`` canary result as fail-closed. Pointer mutations are
    injected callbacks so tests and read-only replay remain fully offline.
    """

    machine: RolloutStateMachine
    canary_action: PointerAction | None = None
    stable_action: PointerAction | None = None
    rollback_action: PointerAction | None = None

    def apply_gate(
        self,
        family: str,
        tier: str,
        format: str,
        gate_report: GateReport,
        *,
        now: datetime | None = None,
    ) -> GateApplication:
        """Apply one phase gate, promoting or rolling back atomically."""

        state = self.machine.phase_state(family, tier, format)
        if state is None:
            raise RolloutStateError(
                f"rollout key must be seeded before applying a gate: "
                f"{baseline_key(family, tier, format)}"
            )
        self._require_report(state, gate_report)
        detected_at = (
            _coerce_utc(now) if now is not None else _coerce_utc(self.machine.clock())
        )
        expired = self.machine.dwell_expired(
            family,
            tier,
            format,
            now=detected_at,
        )

        if state.phase == PHASE_SHADOW:
            if gate_report.decision != RELEASABLE or expired:
                outcome = (
                    "shadow_dwell_expired" if expired else "shadow_gate_not_releasable"
                )
                record = RolloutAuditRecord(
                    family=family,
                    tier=tier,
                    format=format,
                    from_phase=PHASE_SHADOW,
                    to_phase=PHASE_SHADOW,
                    recorded_at=detected_at,
                    gate_status="red",
                    outcome=outcome,
                    pointer_name="latest",
                    pointer_target=state.latest_pointer,
                    gate_report_hash=gate_report.repro_hash,
                    baseline_key=state.key,
                )
                self.machine.record_audit(record)
                return GateApplication(action="hold", state=state, audit_record=record)

            self._require_rollout_targets(state)
            if self.canary_action is not None:
                pointer_target = self.canary_action(state, gate_report)
                self._require_pointer_target(
                    pointer_target,
                    state.target,
                    pointer_name="canary",
                )
            advanced = self.machine.advance(
                family,
                tier,
                format,
                gate_report,
                now=detected_at,
            )
            return GateApplication(
                action="canary",
                state=advanced,
                audit_record=self.machine.audit_records[-1],
            )

        if state.phase == PHASE_CANARY:
            if gate_report.decision != RELEASABLE or expired:
                return self._rollback(
                    state,
                    gate_report,
                    detected_at=detected_at,
                    expired=expired,
                )

            self._require_rollout_targets(state)
            if self.stable_action is None:
                raise RolloutStateError(
                    "a stable pointer action is required before CANARY -> STABLE"
                )
            pointer_target = self.stable_action(state, gate_report)
            self._require_pointer_target(
                pointer_target,
                state.target,
                pointer_name="latest",
            )
            advanced = self.machine.advance(
                family,
                tier,
                format,
                gate_report,
                now=detected_at,
            )
            return GateApplication(
                action="stable",
                state=advanced,
                audit_record=self.machine.audit_records[-1],
            )

        raise IllegalTransitionError(
            f"cannot apply a phase gate in terminal phase {state.phase!r}"
        )

    def _rollback(
        self,
        state: PhaseState,
        gate_report: GateReport,
        *,
        detected_at: datetime,
        expired: bool,
    ) -> GateApplication:
        if self.rollback_action is None:
            raise RolloutStateError(
                "a rollback pointer action is required for a failed canary gate"
            )
        self._require_rollout_targets(state)
        pointer_target = self.rollback_action(state, gate_report)
        self._require_pointer_target(
            pointer_target,
            state.last_green,
            pointer_name="latest",
        )
        pointer_flipped_at = _coerce_utc(self.machine.clock())
        window = self.machine.dwell_windows.get(PHASE_CANARY)
        latency = max(
            0.0,
            (pointer_flipped_at - detected_at).total_seconds(),
        )
        within_window = window is None or latency <= float(window)
        if expired:
            outcome = "canary_dwell_expired_rollback"
        elif within_window:
            outcome = "canary_gate_regression_rollback"
        else:
            outcome = "canary_gate_regression_rollback_late"
        rolled_back = self.machine.rollback(
            state.family,
            state.tier,
            state.format,
            gate_report=gate_report,
            detected_at=detected_at,
            pointer_flipped_at=pointer_flipped_at,
            outcome=outcome,
            now=pointer_flipped_at,
        )
        return GateApplication(
            action="rolled_back" if within_window else "rolled_back_late",
            state=rolled_back,
            audit_record=self.machine.audit_records[-1],
        )

    @staticmethod
    def _require_report(state: PhaseState, gate_report: GateReport) -> None:
        report_key = baseline_key(
            gate_report.family,
            gate_report.tier,
            gate_report.format,
        )
        if report_key != state.key:
            raise RolloutStateError(
                f"gate report coordinates {report_key} do not match {state.key}"
            )
        if gate_report.decision not in {RELEASABLE, QUARANTINED}:
            raise RolloutStateError(f"unknown gate decision: {gate_report.decision!r}")
        if gate_report.recompute_repro_hash() != gate_report.repro_hash:
            raise RolloutStateError("gate report reproducibility hash is invalid")
        if gate_report.decision == RELEASABLE:
            phase_gates = {
                check.gate: check.passed for check in gate_report.gate_results
            }
            missing_or_failed = [
                gate
                for gate in _REQUIRED_PHASE_GATES
                if phase_gates.get(gate) is not True
            ]
            if missing_or_failed:
                raise RolloutStateError(
                    "releasable phase report lacks passing checks for: "
                    + ", ".join(missing_or_failed)
                )

    @staticmethod
    def _require_rollout_targets(state: PhaseState) -> None:
        if not state.target:
            raise RolloutStateError(f"rollout target is missing for {state.key}")
        if not state.last_green:
            raise RolloutStateError(f"last-green target is missing for {state.key}")

    @staticmethod
    def _require_pointer_target(
        actual: str,
        expected: str | None,
        *,
        pointer_name: str,
    ) -> None:
        if not actual or actual != expected:
            raise RolloutStateError(
                f"{pointer_name} pointer action returned {actual!r}, "
                f"expected {expected!r}"
            )


__all__ = [
    "DEFAULT_DWELL_WINDOWS",
    "LEGAL_TRANSITIONS",
    "PHASES",
    "PHASE_CANARY",
    "PHASE_ROLLED_BACK",
    "PHASE_SHADOW",
    "PHASE_STABLE",
    "ROLLOUT_AUDIT_SCHEMA_VERSION",
    "ROLLOUT_STATE_PATH",
    "ROLLOUT_STATE_SCHEMA_VERSION",
    "GateNotReleasableError",
    "GateApplication",
    "IllegalTransitionError",
    "PhaseState",
    "RolloutAuditRecord",
    "RolloutCoordinator",
    "RolloutError",
    "RolloutStateError",
    "RolloutStateMachine",
    "ShadowSuiteComparison",
    "run_shadow_comparison",
]
