"""Tests for the champion-challenger promotion framework (#1274).

All fixtures are synthetic and generated algorithmically: no real PHI, model
outputs, or network access is involved. Every verdict is derived purely from two
committed gate reports.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openmed.eval.champion_challenger import (
    HOLD,
    PROMOTE,
    REJECT,
    ChampionPointer,
    ChampionPointerEntry,
    PromotionVerdict,
    apply_promotion,
    build_promotion_issue_draft,
    compare_champion_challenger,
    write_comparison_record,
)
from openmed.eval.release_gates import QUARANTINED, RELEASABLE, GateCheck, GateReport

_WORKTREE_ROOT = Path(__file__).resolve().parents[3]


def _report(
    *,
    decision: str = RELEASABLE,
    per_label_recall: dict[str, float] | None = None,
    residual_leakage_rate: float = 0.001,
    critical_leakage_count: int = 0,
    quant_recall_delta: float | None = None,
    p50_ms: float | None = 10.0,
    p95_ms: float | None = 20.0,
    ram_mb: float | None = 128.0,
    gate_results: tuple[GateCheck, ...] = (),
    repo_id: str = "OpenMed/pii-tiny-mlx-fp-v3",
    family: str = "pii",
    tier: str = "tiny",
    fmt: str = "mlx-fp",
) -> GateReport:
    """Build a synthetic gate report for a champion or challenger."""

    recall = per_label_recall or {"SSN": 0.9999, "API_KEY": 0.9995}
    return GateReport(
        repo_id=repo_id,
        family=family,
        tier=tier,
        param_count=1_000_000,
        format=fmt,
        per_label_recall=recall,
        per_label_precision={label: 0.999 for label in recall},
        critical_leakage_count=critical_leakage_count,
        residual_leakage_rate=residual_leakage_rate,
        quant_recall_delta=quant_recall_delta,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        ram_mb=ram_mb,
        eval_set_hash="a" * 8,
        leakage_fixture_hash="b" * 8,
        decision=decision,
        gate_results=gate_results,
    )


def _payload_with_f1(report: GateReport, micro_f1: float) -> dict:
    """Return the report payload with an aggregate F1 metric attached."""

    payload = report.to_dict()
    payload["metrics"] = {"micro_f1": micro_f1}
    return payload


# -- acceptance criterion 1: gate criterion (recall over F1) ------------------


def test_reject_on_higher_f1_but_lower_critical_recall() -> None:
    """Higher F1 cannot rescue a critical-label recall regression: REJECT."""

    champion = _payload_with_f1(
        _report(per_label_recall={"SSN": 0.9999, "API_KEY": 0.9995}),
        micro_f1=0.900,
    )
    # Challenger has strictly higher aggregate F1 but drops SSN recall by 0.0049,
    # well past the G7 tolerance of 0.002.
    challenger = _payload_with_f1(
        _report(per_label_recall={"SSN": 0.9950, "API_KEY": 0.9999}),
        micro_f1=0.980,
    )

    verdict = compare_champion_challenger(champion, challenger)

    assert verdict.verdict == REJECT
    assert "per_label_recall.SSN" in verdict.regressions
    # Aggregate F1 was captured but never used as a promotion reason.
    assert verdict.ignored_f1["challenger"]["micro_f1"] == 0.980
    assert all("f1" not in reason.lower() for reason in verdict.reasons)


def test_aggregate_f1_never_changes_verdict() -> None:
    """Two challengers identical but for F1 produce identical verdicts + ledgers."""

    champion = _report()
    low_f1 = _payload_with_f1(_report(residual_leakage_rate=0.0005), micro_f1=0.10)
    high_f1 = _payload_with_f1(_report(residual_leakage_rate=0.0005), micro_f1=0.99)

    low = compare_champion_challenger(champion, low_f1)
    high = compare_champion_challenger(champion, high_f1)

    assert low.verdict == high.verdict == PROMOTE
    assert [e.to_dict() for e in low.ledger] == [e.to_dict() for e in high.ledger]


def test_hold_when_only_f1_improves() -> None:
    """A challenger that only improves aggregate F1 is held, not promoted."""

    champion = _report()
    challenger = _payload_with_f1(_report(), micro_f1=0.99)

    verdict = compare_champion_challenger(champion, challenger)

    assert verdict.verdict == HOLD
    assert verdict.wins == ()


def test_promote_on_recall_and_leakage_win() -> None:
    """A releasable challenger that wins on recall/leakage is promoted."""

    champion = _report(
        per_label_recall={"SSN": 0.9999, "API_KEY": 0.9950},
        residual_leakage_rate=0.0020,
    )
    challenger = _report(
        per_label_recall={"SSN": 0.9999, "API_KEY": 0.9999},
        residual_leakage_rate=0.0010,
    )

    verdict = compare_champion_challenger(champion, challenger)

    assert verdict.verdict == PROMOTE
    assert "per_label_recall.API_KEY" in verdict.wins
    assert "residual_leakage_rate" in verdict.wins
    assert verdict.regressions == ()


def test_leakage_regression_downgrades_win_to_hold() -> None:
    """A recall win offset by a leakage regression is held, not promoted."""

    champion = _report(
        per_label_recall={"SSN": 0.9999, "API_KEY": 0.9950},
        residual_leakage_rate=0.0010,
    )
    challenger = _report(
        per_label_recall={"SSN": 0.9999, "API_KEY": 0.9999},
        residual_leakage_rate=0.0020,
    )

    verdict = compare_champion_challenger(champion, challenger)

    assert verdict.verdict == HOLD
    assert "per_label_recall.API_KEY" in verdict.wins
    assert "residual_leakage_rate" in verdict.regressions


def test_within_tolerance_recall_drop_is_a_tie() -> None:
    """A critical-recall drop within the G7 tolerance is a tie, not a regression."""

    # SSN drops 0.0014 (within the 0.002 tolerance) -> tie; API_KEY rises well
    # past tolerance -> a clear win.
    champion = _report(per_label_recall={"SSN": 0.9999, "API_KEY": 0.9950})
    challenger = _report(per_label_recall={"SSN": 0.9985, "API_KEY": 0.9999})

    verdict = compare_champion_challenger(champion, challenger)

    assert "per_label_recall.SSN" in verdict.ties
    assert verdict.verdict == PROMOTE  # API_KEY win, no blocking regression


def test_not_releasable_challenger_is_rejected() -> None:
    """A challenger failing its own floors (QUARANTINED) is rejected."""

    champion = _report()
    challenger = _report(decision=QUARANTINED)

    verdict = compare_champion_challenger(champion, challenger)

    assert verdict.verdict == REJECT
    assert any("RELEASABLE" in reason for reason in verdict.reasons)


def test_gate_fix_counts_as_promotion_win() -> None:
    """A gate the champion failed that the challenger now passes is a win."""

    champion = _report(gate_results=(GateCheck("G11", False, reason="miss"),))
    challenger = _report(gate_results=(GateCheck("G11", True),))

    verdict = compare_champion_challenger(champion, challenger)

    assert verdict.verdict == PROMOTE
    assert "gate.G11" in verdict.wins


def test_mismatched_coordinates_are_rejected() -> None:
    """A champion and challenger from different (family, tier, format) cannot compare."""

    champion = _report(family="clinical", tier="large", fmt="onnx")
    challenger = _report(family="pii", tier="tiny", fmt="mlx-fp")

    with pytest.raises(ValueError, match="different rollout coordinates"):
        compare_champion_challenger(champion, challenger)


def test_bootstrap_without_champion_promotes_releasable() -> None:
    """With no incumbent, a releasable challenger bootstraps the pointer."""

    assert compare_champion_challenger(None, _report()).verdict == PROMOTE
    assert (
        compare_champion_challenger(None, _report(decision=QUARANTINED)).verdict
        == REJECT
    )


# -- acceptance criterion 2: byte-identical reproducibility -------------------


def test_ledger_reproduces_byte_identically(tmp_path: Path) -> None:
    """The comparison record reproduces byte-for-byte from the two reports."""

    champion = _report(residual_leakage_rate=0.0020)
    challenger = _report(residual_leakage_rate=0.0010)

    first = compare_champion_challenger(champion, challenger)
    second = compare_champion_challenger(champion, challenger)
    assert first.to_json() == second.to_json()

    path_a = write_comparison_record(first, tmp_path / "a.json")
    path_b = write_comparison_record(second, tmp_path / "b.json")
    assert path_a.read_bytes() == path_b.read_bytes()

    # The embedded content hash is a self-consistent signature.
    assert first.verify()
    # Round-tripping through the persisted record preserves the exact payload.
    reloaded = PromotionVerdict.from_dict(json.loads(path_a.read_text()))
    assert reloaded.to_json() == first.to_json()
    assert reloaded.record_hash == first.record_hash


def test_record_reproduces_across_process(tmp_path: Path) -> None:
    """A record written by the CLI matches the in-process record byte-for-byte."""

    champion_path = tmp_path / "champion.json"
    challenger_path = tmp_path / "challenger.json"
    champion = _report(residual_leakage_rate=0.0020)
    challenger = _report(residual_leakage_rate=0.0010)
    champion_path.write_text(json.dumps(champion.to_dict()), encoding="utf-8")
    challenger_path.write_text(json.dumps(challenger.to_dict()), encoding="utf-8")

    in_process = write_comparison_record(
        compare_champion_challenger(champion, challenger),
        tmp_path / "in_process.json",
    )

    cli_record = tmp_path / "cli.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_WORKTREE_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openmed.eval.champion_challenger",
            "--champion",
            str(champion_path),
            "--challenger",
            str(challenger_path),
            "--record",
            str(cli_record),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert cli_record.read_bytes() == in_process.read_bytes()


# -- acceptance criterion 3: rollback safety ----------------------------------


def test_promote_writes_pointer_and_record(tmp_path: Path) -> None:
    """PROMOTE advances the pointer and writes the comparison record."""

    pointer_path = tmp_path / "champion.json"
    ChampionPointer().save(pointer_path)
    record_path = tmp_path / "record.json"

    champion = _report(residual_leakage_rate=0.0020)
    challenger = _report(
        residual_leakage_rate=0.0010, repo_id="OpenMed/pii-tiny-mlx-fp-v4"
    )
    verdict = compare_champion_challenger(champion, challenger)
    assert verdict.verdict == PROMOTE

    result = apply_promotion(
        verdict,
        pointer_path=pointer_path,
        record_path=record_path,
        challenger_version="v4",
    )

    assert result.promoted is True
    assert record_path.is_file()
    pointer = ChampionPointer.load(pointer_path)
    entry = pointer.get("pii", "tiny", "mlx-fp")
    assert entry is not None
    assert entry.champion == "v4"
    assert entry.record_hash == verdict.record_hash


@pytest.mark.parametrize(
    "challenger_kwargs",
    [
        {"per_label_recall": {"SSN": 0.9950, "API_KEY": 0.9999}},  # REJECT
        {},  # HOLD (all ties)
    ],
)
def test_hold_or_reject_never_advances_pointer(
    tmp_path: Path,
    challenger_kwargs: dict,
) -> None:
    """A HOLD/REJECT verdict leaves the committed champion pointer untouched."""

    pointer_path = tmp_path / "champion.json"
    seeded = ChampionPointer(
        entries={
            "pii::tiny::mlx-fp": ChampionPointerEntry(
                family="pii",
                tier="tiny",
                format="mlx-fp",
                champion="v3",
                record_hash="sha256:" + "0" * 64,
            )
        }
    )
    seeded.save(pointer_path)
    before = pointer_path.read_bytes()

    champion = _report()
    challenger = _report(**challenger_kwargs)
    verdict = compare_champion_challenger(champion, challenger)
    assert verdict.verdict in {HOLD, REJECT}

    record_path = tmp_path / "record.json"
    result = apply_promotion(
        verdict,
        pointer_path=pointer_path,
        record_path=record_path,
    )

    assert result.promoted is False
    # The pointer file is byte-for-byte unchanged and the incumbent survives.
    assert pointer_path.read_bytes() == before
    assert (
        ChampionPointer.load(pointer_path).get("pii", "tiny", "mlx-fp").champion == "v3"
    )
    # No pointer/record is advanced; an offline issue draft is emitted instead.
    assert not record_path.exists()
    assert result.issue_draft is not None
    assert result.issue_draft["verdict"] == verdict.verdict


def test_issue_draft_is_offline_and_deterministic() -> None:
    """The issue draft is a deterministic, network-free artifact."""

    champion = _report()
    challenger = _report(decision=QUARANTINED)
    verdict = compare_champion_challenger(champion, challenger)

    first = build_promotion_issue_draft(verdict)
    second = build_promotion_issue_draft(verdict)
    assert first == second
    assert first["labels"][-1] == "reject"


# -- acceptance criterion 4: derivable with no model call ---------------------


def test_verdict_derivable_from_committed_reports(tmp_path: Path) -> None:
    """The verdict is derived from two committed report files, no model call."""

    champion_path = tmp_path / "champion.json"
    challenger_path = tmp_path / "challenger.json"
    champion_path.write_text(
        json.dumps(_report(residual_leakage_rate=0.0020).to_dict()),
        encoding="utf-8",
    )
    challenger_path.write_text(
        json.dumps(_report(residual_leakage_rate=0.0010).to_dict()),
        encoding="utf-8",
    )

    verdict = compare_champion_challenger(champion_path, challenger_path)

    assert verdict.verdict == PROMOTE
    assert verdict.challenger_repro_hash
    assert verdict.champion_repro_hash


def test_committed_champion_pointer_is_loadable() -> None:
    """The committed gates/champion.json seed loads cleanly."""

    pointer = ChampionPointer.load()
    assert pointer.to_dict()["schema_version"] == 1
