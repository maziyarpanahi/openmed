"""Tests for the shadow/canary/stable rollout state machine (#1802)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from openmed.eval.release_gates import QUARANTINED, RELEASABLE, GateReport
from openmed.eval.rollout import (
    PHASE_CANARY,
    PHASE_ROLLED_BACK,
    PHASE_SHADOW,
    PHASE_STABLE,
    GateNotReleasableError,
    IllegalTransitionError,
    RolloutStateError,
    RolloutStateMachine,
)

GATE_KEY = "test-release-gate-key"
_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _gate_report(
    *,
    family: str = "pii",
    tier: str = "tiny",
    fmt: str = "mlx-fp",
    decision: str = RELEASABLE,
) -> GateReport:
    """Build a signed synthetic gate report for the given coordinates."""

    return GateReport(
        repo_id=f"OpenMed/{family}-{tier}-{fmt}",
        family=family,
        tier=tier,
        param_count=1_000_000,
        format=fmt,
        per_label_recall={"PERSON": 0.999},
        per_label_precision={"PERSON": 0.999},
        critical_leakage_count=0 if decision == RELEASABLE else 3,
        residual_leakage_rate=0.001 if decision == RELEASABLE else 0.05,
        quant_recall_delta=None,
        p50_ms=10.0,
        p95_ms=20.0,
        ram_mb=128.0,
        eval_set_hash="a" * 8,
        leakage_fixture_hash="b" * 8,
        decision=decision,
    ).sign(GATE_KEY)


def _machine(clock_value: datetime = _EPOCH) -> RolloutStateMachine:
    return RolloutStateMachine(clock=lambda: clock_value)


def test_full_happy_path_promotion() -> None:
    """A challenger walks shadow -> canary -> stable on releasable gates."""

    machine = _machine()
    machine.seed("pii", "tiny", "mlx-fp", target="v3")
    assert machine.current_phase("pii", "tiny", "mlx-fp") == PHASE_SHADOW

    canary = machine.advance("pii", "tiny", "mlx-fp", _gate_report())
    assert canary.phase == PHASE_CANARY
    assert machine.current_phase("pii", "tiny", "mlx-fp") == PHASE_CANARY

    stable = machine.advance("pii", "tiny", "mlx-fp", _gate_report())
    assert stable.phase == PHASE_STABLE
    # Reaching STABLE records the rolled-out version as the last green target.
    assert stable.target == "v3"
    assert stable.last_green == "v3"


def test_advance_refuses_non_releasable_gate() -> None:
    """A quarantined gate blocks promotion and leaves state unchanged."""

    machine = _machine()
    machine.seed("pii", "tiny", "mlx-fp", target="v3")

    with pytest.raises(GateNotReleasableError):
        machine.advance("pii", "tiny", "mlx-fp", _gate_report(decision=QUARANTINED))

    assert machine.current_phase("pii", "tiny", "mlx-fp") == PHASE_SHADOW


def test_advance_rejects_mismatched_gate_coordinates() -> None:
    """A gate report for a different key cannot promote this key."""

    machine = _machine()
    machine.seed("pii", "tiny", "mlx-fp", target="v3")

    with pytest.raises(RolloutStateError):
        machine.advance("pii", "tiny", "mlx-fp", _gate_report(family="clinical"))

    assert machine.current_phase("pii", "tiny", "mlx-fp") == PHASE_SHADOW


def test_illegal_skip_transition_raises_and_preserves_state() -> None:
    """A shadow -> stable skip is a typed error that does not mutate state."""

    machine = _machine()
    seeded = machine.seed("pii", "tiny", "mlx-fp", target="v3")

    with pytest.raises(IllegalTransitionError):
        machine.transition(
            "pii", "tiny", "mlx-fp", PHASE_STABLE, gate_report=_gate_report()
        )

    # The persisted entry is byte-for-byte the pre-attempt shadow state.
    assert machine.phase_state("pii", "tiny", "mlx-fp") == seeded
    assert machine.current_phase("pii", "tiny", "mlx-fp") == PHASE_SHADOW


def test_advance_from_terminal_phase_is_illegal() -> None:
    """Advancing a stable key past the forward chain raises a typed error."""

    machine = _machine()
    machine.seed("pii", "tiny", "mlx-fp", target="v3")
    machine.advance("pii", "tiny", "mlx-fp", _gate_report())
    machine.advance("pii", "tiny", "mlx-fp", _gate_report())

    with pytest.raises(IllegalTransitionError):
        machine.advance("pii", "tiny", "mlx-fp", _gate_report())


def test_rollback_reverts_to_last_green() -> None:
    """Rollback from canary/stable moves to rolled_back and restores the target."""

    machine = _machine()
    machine.seed("pii", "tiny", "mlx-fp", target="v3")
    machine.advance("pii", "tiny", "mlx-fp", _gate_report())
    stable = machine.advance("pii", "tiny", "mlx-fp", _gate_report(), target="v4")
    assert stable.phase == PHASE_STABLE
    assert stable.last_green == "v4"

    rolled = machine.rollback("pii", "tiny", "mlx-fp")
    assert rolled.phase == PHASE_ROLLED_BACK
    assert rolled.target == "v4"

    # Rollback is unguarded but still needs something live to roll back.
    with pytest.raises(IllegalTransitionError):
        machine.rollback("pii", "tiny", "mlx-fp")


def test_rollback_from_shadow_is_illegal() -> None:
    """Nothing is live in shadow, so a rollback there is a typed error."""

    machine = _machine()
    machine.seed("pii", "tiny", "mlx-fp", target="v3")

    with pytest.raises(IllegalTransitionError):
        machine.rollback("pii", "tiny", "mlx-fp")


def test_keys_advance_independently() -> None:
    """Each (family, tier, format) key carries its own phase."""

    machine = _machine()
    machine.seed("pii", "tiny", "mlx-fp", target="v3")
    machine.seed("clinical", "base", "onnx", target="v9")

    # Promote only the first key to canary.
    machine.advance("pii", "tiny", "mlx-fp", _gate_report())

    assert machine.current_phase("pii", "tiny", "mlx-fp") == PHASE_CANARY
    assert machine.current_phase("clinical", "base", "onnx") == PHASE_SHADOW

    # Promoting the second key does not disturb the first.
    machine.advance(
        "clinical",
        "base",
        "onnx",
        _gate_report(family="clinical", tier="base", fmt="onnx"),
    )
    assert machine.current_phase("pii", "tiny", "mlx-fp") == PHASE_CANARY
    assert machine.current_phase("clinical", "base", "onnx") == PHASE_CANARY


def test_dwell_expired_uses_controlled_clock() -> None:
    """dwell_expired flips true only past the configured phase window."""

    machine = RolloutStateMachine(
        dwell_windows={
            PHASE_SHADOW: 3600,
            PHASE_CANARY: 3600,
            PHASE_STABLE: None,
            PHASE_ROLLED_BACK: None,
        },
        clock=lambda: _EPOCH,
    )
    machine.seed("pii", "tiny", "mlx-fp", target="v3", now=_EPOCH)

    # Inside the window: not expired.
    assert not machine.dwell_expired(
        "pii", "tiny", "mlx-fp", now=_EPOCH + timedelta(seconds=3600)
    )
    # Past the window: expired.
    assert machine.dwell_expired(
        "pii", "tiny", "mlx-fp", now=_EPOCH + timedelta(seconds=3601)
    )

    # A phase without a configured window never expires.
    machine.advance("pii", "tiny", "mlx-fp", _gate_report(), now=_EPOCH)
    machine.advance("pii", "tiny", "mlx-fp", _gate_report(), now=_EPOCH)
    assert machine.current_phase("pii", "tiny", "mlx-fp") == PHASE_STABLE
    assert not machine.dwell_expired(
        "pii", "tiny", "mlx-fp", now=_EPOCH + timedelta(days=3650)
    )


def test_unseeded_key_defaults_to_shadow_without_dwell() -> None:
    """An unknown key reads as shadow and never reports a dwell expiry."""

    machine = _machine()
    assert machine.current_phase("pii", "tiny", "mlx-fp") == PHASE_SHADOW
    assert not machine.dwell_expired(
        "pii", "tiny", "mlx-fp", now=_EPOCH + timedelta(days=365)
    )


def test_state_reproducible_from_committed_document(tmp_path) -> None:
    """State round-trips through JSON and replays deterministically, offline."""

    machine = _machine()
    machine.seed("pii", "tiny", "mlx-fp", target="v3", now=_EPOCH)
    machine.advance("pii", "tiny", "mlx-fp", _gate_report(), now=_EPOCH)

    state_path = tmp_path / "rollout_state.json"
    machine.save(state_path)

    # Reload from the committed document alone -- no live call.
    reloaded = RolloutStateMachine.load(state_path, clock=lambda: _EPOCH)
    assert reloaded.to_dict() == machine.to_dict()

    # Replaying the same gate report from the reloaded state is deterministic.
    reloaded.advance("pii", "tiny", "mlx-fp", _gate_report(), now=_EPOCH)

    fresh = _machine()
    fresh.seed("pii", "tiny", "mlx-fp", target="v3", now=_EPOCH)
    fresh.advance("pii", "tiny", "mlx-fp", _gate_report(), now=_EPOCH)
    fresh.advance("pii", "tiny", "mlx-fp", _gate_report(), now=_EPOCH)

    assert reloaded.to_dict() == fresh.to_dict()


def test_load_rejects_unknown_schema_version(tmp_path) -> None:
    """A state document with an unexpected schema version is rejected."""

    state_path = tmp_path / "rollout_state.json"
    state_path.write_text('{"schema_version": 999, "entries": {}}', encoding="utf-8")

    with pytest.raises(RolloutStateError):
        RolloutStateMachine.load(state_path)


def test_committed_rollout_state_is_loadable() -> None:
    """The committed gates/rollout_state.json seed loads cleanly."""

    machine = RolloutStateMachine.load()
    assert machine.to_dict()["schema_version"] == 1
