"""Tests for the shadow/canary/stable rollout state machine."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from openmed.core.baseline import BASELINE_SCHEMA_VERSION, write_baseline_store
from openmed.core.manifest import load_manifest_rows, write_manifest_rows
from openmed.eval.evidence_bundle import bundle_gate_evidence
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.release_gates import (
    QUARANTINED,
    RELEASABLE,
    GateCheck,
    GateReport,
)
from openmed.eval.rollout import (
    DEFAULT_DWELL_WINDOWS,
    PHASE_CANARY,
    PHASE_ROLLED_BACK,
    PHASE_SHADOW,
    PHASE_STABLE,
    GateNotReleasableError,
    IllegalTransitionError,
    RolloutCoordinator,
    RolloutStateError,
    RolloutStateMachine,
    run_shadow_comparison,
)

GATE_KEY = "test-release-gate-key"
_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _gate_report(
    *,
    family: str = "pii",
    tier: str = "tiny",
    fmt: str = "mlx-fp",
    decision: str = RELEASABLE,
    g7_regression: bool = False,
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
        gate_results=tuple(
            GateCheck(
                gate,
                not (gate == "G7" and g7_regression),
                reason=(
                    "baseline regression gate failed"
                    if gate == "G7" and g7_regression
                    else "ok"
                ),
                details=(
                    {
                        "baseline_key": "pii::tiny::mlx-fp",
                        "violations": (
                            {
                                "recall_drop": {
                                    "PERSON": {
                                        "baseline": 0.999,
                                        "candidate": 0.990,
                                        "drop": 0.009,
                                        "limit": 0.002,
                                    }
                                }
                            }
                            if g7_regression
                            else {}
                        ),
                    }
                    if gate == "G7"
                    else {}
                ),
            )
            for gate in ("G1a", "G1b", "G2", "G3", "G4", "G5", "G6", "G7", "G8")
        ),
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


def test_unseeded_key_cannot_transition_without_persisted_state() -> None:
    """Mutation requires an explicit seed and preserves the empty document."""

    machine = _machine()
    before = machine.to_json()

    with pytest.raises(RolloutStateError, match="must be seeded"):
        machine.advance("pii", "tiny", "mlx-fp", _gate_report())
    assert machine.to_json() == before

    with pytest.raises(RolloutStateError, match="must be seeded"):
        machine.transition(
            "pii", "tiny", "mlx-fp", PHASE_CANARY, gate_report=_gate_report()
        )
    assert machine.to_json() == before


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


def test_shadow_runs_both_models_on_shared_golden_and_shield_fixtures(
    tmp_path: Path,
) -> None:
    """Shadow evidence uses identical synthetic fixture paths for both models."""

    fixture_payload = (
        '{"id":"synthetic-1","text":"Avery","gold_spans":'
        '[{"start":0,"end":5,"label":"PERSON"}]}\n'
    )
    golden = tmp_path / "golden.jsonl"
    shield = tmp_path / "shield.jsonl"
    golden.write_text(fixture_payload, encoding="utf-8")
    shield.write_text(fixture_payload, encoding="utf-8")

    def runner(fixture: BenchmarkFixture, model_name: str, _device: str):
        return fixture.gold_spans if model_name == "champion-v1" else ()

    from openmed.eval.harness import run_suite

    champion = run_suite(
        golden,
        suite="golden",
        model_name="champion-v1",
        runner=runner,
        metadata={"family": "PII", "tier": "Small", "format": "mlx-fp"},
        generated_at="2026-01-01T00:00:00+00:00",
    )
    baseline = tmp_path / "baseline.json"
    write_baseline_store(
        {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "entries": {
                "pii::small::mlx-fp": {
                    "key": "pii::small::mlx-fp",
                    "family": "PII",
                    "tier": "Small",
                    "format": "mlx-fp",
                    "metrics": champion.metrics,
                    "reproducibility_hash": "sha256:" + "a" * 64,
                }
            },
        },
        baseline,
    )

    comparisons = run_shadow_comparison(
        {"golden": golden, "shield": shield},
        champion_model="champion-v1",
        challenger_model="challenger-v2",
        family="PII",
        tier="Small",
        format="mlx-fp",
        baseline_path=baseline,
        runner=runner,
        generated_at="2026-01-01T00:00:00+00:00",
    )

    assert [comparison.suite for comparison in comparisons] == ["golden", "shield"]
    assert all(
        comparison.champion_report.fixture_count
        == comparison.challenger_report.fixture_count
        == 1
        for comparison in comparisons
    )
    assert all(
        delta.verdict == "unchanged"
        for comparison in comparisons
        for metric, delta in comparison.champion_diff.metrics.items()
        if ".by_label." in metric
    )
    assert any(
        ".by_label." in delta.metric
        for comparison in comparisons
        for delta in comparison.challenger_diff.largest_regressions
    )
    serialized = str([comparison.to_dict() for comparison in comparisons])
    assert "Avery" not in serialized


def test_canary_g7_regression_auto_rolls_back_within_dwell_and_audits(
    tmp_path: Path,
) -> None:
    """A failed canary G7 gate flips latest to last-green and never stabilizes."""

    pointer_flip_at = _EPOCH + timedelta(seconds=12)
    machine = RolloutStateMachine(clock=lambda: pointer_flip_at)
    machine.seed(
        "pii",
        "tiny",
        "mlx-fp",
        target="OpenMed/pii-v2",
        last_green="OpenMed/pii-v1",
        now=_EPOCH,
    )
    calls: list[tuple[str, str]] = []

    def canary_action(state, _report):
        calls.append(("canary", state.target))
        return state.target

    def stable_action(state, _report):
        calls.append(("stable", state.target))
        return state.target

    def rollback_action(state, _report):
        calls.append(("latest", state.last_green))
        return state.last_green

    coordinator = RolloutCoordinator(
        machine,
        canary_action=canary_action,
        stable_action=stable_action,
        rollback_action=rollback_action,
    )
    entered = coordinator.apply_gate(
        "pii",
        "tiny",
        "mlx-fp",
        _gate_report(),
        now=_EPOCH,
    )
    assert entered.action == "canary"
    assert entered.state.canary_pointer == "OpenMed/pii-v2"
    assert entered.state.latest_pointer == "OpenMed/pii-v1"

    rolled_back = coordinator.apply_gate(
        "pii",
        "tiny",
        "mlx-fp",
        _gate_report(decision=QUARANTINED, g7_regression=True),
        now=_EPOCH + timedelta(seconds=10),
    )

    assert rolled_back.action == "rolled_back"
    assert rolled_back.state.phase == PHASE_ROLLED_BACK
    assert rolled_back.state.phase != PHASE_STABLE
    assert rolled_back.state.target == "OpenMed/pii-v1"
    assert calls == [
        ("canary", "OpenMed/pii-v2"),
        ("latest", "OpenMed/pii-v1"),
    ]
    assert [record.gate_status for record in machine.audit_records] == [
        "green",
        "red",
    ]
    assert machine.audit_records[-1].gate_report_hash
    assert machine.audit_records[-1].pointer_target == "OpenMed/pii-v1"
    assert machine.mean_rollback_latency_seconds() == 2.0
    assert machine.mean_rollback_latency_seconds() < DEFAULT_DWELL_WINDOWS[PHASE_CANARY]

    state_path = tmp_path / "rollout_state.json"
    audit_path = tmp_path / "rollout_audit.json"
    machine.save(state_path)
    machine.save_audit(audit_path)
    reloaded = RolloutStateMachine.load(state_path, clock=lambda: pointer_flip_at)
    assert reloaded.to_dict() == machine.to_dict()

    bundle = bundle_gate_evidence(
        _gate_report(decision=QUARANTINED, g7_regression=True),
        tmp_path / "evidence",
        extra_artifacts={
            "rollout_audit": audit_path,
            "rollout_state": state_path,
        },
    )
    artifact_ids = {item["artifact_id"] for item in bundle.manifest["artifacts"]}
    assert {"rollout_audit", "rollout_state"} <= artifact_ids
    rollout_artifacts = {
        item["artifact_id"]: item for item in bundle.manifest["artifacts"]
    }
    assert rollout_artifacts["rollout_audit"]["gates"] == ["G7"]
    assert "Synthetic Patient" not in audit_path.read_text(encoding="utf-8")


def test_canary_reaches_stable_only_after_pointer_action_and_full_green_gate() -> None:
    """Stable promotion executes the latest flip before recording the phase."""

    machine = _machine()
    machine.seed(
        "pii",
        "tiny",
        "mlx-fp",
        target="OpenMed/pii-v2",
        last_green="OpenMed/pii-v1",
    )
    pointer_targets: list[str] = []

    def latest_action(state, _report):
        pointer_targets.append(state.target)
        return state.target

    coordinator = RolloutCoordinator(
        machine,
        canary_action=lambda state, _report: state.target,
        stable_action=latest_action,
    )
    coordinator.apply_gate("pii", "tiny", "mlx-fp", _gate_report())
    promoted = coordinator.apply_gate("pii", "tiny", "mlx-fp", _gate_report())

    assert promoted.action == "stable"
    assert promoted.state.phase == PHASE_STABLE
    assert promoted.state.latest_pointer == "OpenMed/pii-v2"
    assert pointer_targets == ["OpenMed/pii-v2"]
    assert [record.gate_status for record in machine.audit_records] == [
        "green",
        "green",
    ]


def test_non_releasable_shadow_gate_holds_without_pointer_flip() -> None:
    """A red shadow gate is recorded but cannot enter canary."""

    machine = _machine()
    machine.seed(
        "pii",
        "tiny",
        "mlx-fp",
        target="OpenMed/pii-v2",
        last_green="OpenMed/pii-v1",
    )
    coordinator = RolloutCoordinator(
        machine,
        canary_action=lambda _state, _report: pytest.fail("pointer must not flip"),
    )

    result = coordinator.apply_gate(
        "pii",
        "tiny",
        "mlx-fp",
        _gate_report(decision=QUARANTINED),
    )

    assert result.action == "hold"
    assert result.state.phase == PHASE_SHADOW
    assert result.audit_record.gate_status == "red"


def test_releasable_phase_report_requires_every_g1a_through_g8_check() -> None:
    """A decision string cannot bypass a missing phase gate check."""

    machine = _machine()
    machine.seed(
        "pii",
        "tiny",
        "mlx-fp",
        target="OpenMed/pii-v2",
        last_green="OpenMed/pii-v1",
    )
    incomplete = _gate_report()
    incomplete.gate_results = (GateCheck("G7", True),)
    incomplete.sign(GATE_KEY)

    with pytest.raises(RolloutStateError, match="G1a"):
        RolloutCoordinator(machine).apply_gate(
            "pii",
            "tiny",
            "mlx-fp",
            incomplete,
        )

    assert machine.current_phase("pii", "tiny", "mlx-fp") == PHASE_SHADOW


def test_apply_gate_cli_rolls_back_manifest_and_regenerates_card(
    tmp_path: Path,
) -> None:
    """The canary CLI flips the local manifest before persisting rollback state."""

    def row(repo_id: str, released: str, digest: str) -> dict[str, object]:
        return {
            "repo_id": repo_id,
            "family": "PII",
            "task": "token-classification",
            "languages": ["en"],
            "tier": "Small",
            "param_count": 1_000_000,
            "architecture": "synthetic",
            "base_model": repo_id,
            "formats": ["mlx-fp"],
            "canonical_labels": ["PERSON"],
            "benchmark": {"dataset": "synthetic", "recall": 0.99},
            "arxiv": None,
            "license": "apache-2.0",
            "reproducibility_hash": "sha256:" + digest * 64,
            "released": released,
        }

    v2 = "OpenMed/pii-v2"
    v1 = "OpenMed/pii-v1"
    manifest = tmp_path / "models.jsonl"
    staged_manifest = tmp_path / "staged-models.jsonl"
    rows = [row(v2, "2026-01-02", "b"), row(v1, "2026-01-01", "a")]
    write_manifest_rows(rows, manifest)
    write_manifest_rows(rows, staged_manifest)
    baseline = tmp_path / "baseline.json"
    write_baseline_store(
        {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "entries": {
                "pii::small::mlx-fp": {
                    "key": "pii::small::mlx-fp",
                    "family": "PII",
                    "tier": "Small",
                    "format": "mlx-fp",
                    "metrics": {"per_label_recall": {"PERSON": 0.999}},
                    "reproducibility_hash": "sha256:" + "a" * 64,
                    "repo_id": v1,
                    "source_model_id": v1,
                    "released": "2026-01-01",
                }
            },
        },
        baseline,
    )
    machine = _machine()
    machine.seed(
        "PII",
        "Small",
        "mlx-fp",
        target=v2,
        last_green=v1,
        now=_EPOCH,
    )
    machine.advance(
        "PII",
        "Small",
        "mlx-fp",
        _gate_report(family="PII", tier="Small"),
        now=_EPOCH,
    )
    state_path = tmp_path / "rollout-state.json"
    machine.save(state_path)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        _gate_report(
            family="PII",
            tier="Small",
            decision=QUARANTINED,
            g7_regression=True,
        ).to_json(),
        encoding="utf-8",
    )
    audit_path = tmp_path / "rollout-audit.json"
    result_path = tmp_path / "result.json"
    card_dir = tmp_path / "cards"
    status_path = tmp_path / "release-status.json"
    tracking_log = tmp_path / "rollback-log.jsonl"
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join(
        value for value in (str(repo_root), os.environ.get("PYTHONPATH")) if value
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/release/rollout_state.py"),
            "--state",
            str(state_path),
            "apply-gate",
            "--family",
            "PII",
            "--tier",
            "Small",
            "--format",
            "mlx-fp",
            "--gate-report",
            str(gate_path),
            "--manifest",
            str(manifest),
            "--staged-manifest",
            str(staged_manifest),
            "--baseline",
            str(baseline),
            "--card-dir",
            str(card_dir),
            "--status-path",
            str(status_path),
            "--tracking-log",
            str(tracking_log),
            "--audit-output",
            str(audit_path),
            "--result-output",
            str(result_path),
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert (
        RolloutStateMachine.load(state_path).current_phase("PII", "Small", "mlx-fp")
        == PHASE_ROLLED_BACK
    )
    assert load_manifest_rows(manifest)[0]["repo_id"] == v1
    assert any(card_dir.glob("*.md"))
    assert status_path.is_file()
    assert tracking_log.is_file()
    assert '"gate_status": "red"' in audit_path.read_text(encoding="utf-8")
    assert '"action": "rolled_back"' in result_path.read_text(encoding="utf-8")


def test_model_rollout_workflow_is_fail_closed_and_tracks_rollbacks() -> None:
    """The workflow drives both phases and opens tracking only after rollback."""

    workflow = (
        Path(__file__).resolve().parents[3] / ".github/workflows/model-rollout.yml"
    ).read_text(encoding="utf-8")

    assert "rollout_state.py shadow" in workflow
    assert "apply-gate" in workflow
    assert "gh issue create" in workflow
    assert "rolled_back" in workflow
    assert "exit 1" in workflow
