"""Tests for enforceable edge install-size and peak-RSS budgets."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.eval.footprint_gate import (
    FootprintBudget,
    evaluate_footprint,
    load_footprint_budgets,
    main,
)

MIB = 1024 * 1024


def _budget() -> FootprintBudget:
    return FootprintBudget(
        profile="jetson-nano",
        device="NVIDIA Jetson Nano 4GB",
        tier="Tiny",
        install_size_bytes_max=300 * MIB,
        peak_rss_bytes_max=350 * MIB,
    )


def _report(*, install_size: int, peak_rss: int | None) -> dict[str, object]:
    return {
        "benchmark": "edge_sbc",
        "schema_version": 1,
        "offline": True,
        "network_guard": "socket-blocked",
        "profile": "jetson-nano",
        "install_size_bytes": install_size,
        "peak_rss_bytes": peak_rss,
    }


def test_committed_profiles_match_device_tier_budgets() -> None:
    budgets = load_footprint_budgets()

    assert set(budgets) == {"jetson-nano", "raspberry-pi-5"}
    assert budgets["jetson-nano"].tier == "Tiny"
    assert budgets["jetson-nano"].install_size_bytes_max == 300 * MIB
    assert budgets["jetson-nano"].peak_rss_bytes_max == 350 * MIB
    assert budgets["raspberry-pi-5"].tier == "Base"
    assert budgets["raspberry-pi-5"].peak_rss_bytes_max == 900 * MIB


def test_gate_passes_at_inclusive_install_and_ram_boundaries() -> None:
    budget = _budget()
    result = evaluate_footprint(
        _report(
            install_size=budget.install_size_bytes_max,
            peak_rss=budget.peak_rss_bytes_max,
        ),
        budget,
    )

    assert result.passed is True
    assert all(check.passed for check in result.checks.values())


def test_gate_fails_when_install_size_exceeds_budget() -> None:
    budget = _budget()
    result = evaluate_footprint(
        _report(
            install_size=budget.install_size_bytes_max + 1,
            peak_rss=budget.peak_rss_bytes_max,
        ),
        budget,
    )

    assert result.passed is False
    assert result.checks["install_size_bytes"].passed is False


def test_gate_fails_when_peak_ram_exceeds_budget() -> None:
    budget = _budget()
    result = evaluate_footprint(
        _report(
            install_size=budget.install_size_bytes_max,
            peak_rss=budget.peak_rss_bytes_max + 1,
        ),
        budget,
    )

    assert result.passed is False
    assert result.checks["peak_rss_bytes"].passed is False


def test_gate_fails_closed_on_missing_ram_or_profile_mismatch() -> None:
    budget = _budget()
    missing = evaluate_footprint(
        _report(install_size=1, peak_rss=None),
        budget,
    )
    mismatch = evaluate_footprint(
        {
            **_report(install_size=1, peak_rss=1),
            "profile": "raspberry-pi-5",
        },
        budget,
    )

    assert missing.passed is False
    assert "missing a valid peak_rss_bytes" in " ".join(missing.errors)
    assert mismatch.passed is False
    assert "does not match" in " ".join(mismatch.errors)


def test_gate_fails_closed_without_offline_benchmark_provenance() -> None:
    payload = _report(install_size=1, peak_rss=1)
    payload["offline"] = False
    payload["network_guard"] = "not-recorded"

    result = evaluate_footprint(payload, _budget())

    assert result.passed is False
    assert "offline execution" in " ".join(result.errors)
    assert "socket-blocked" in " ".join(result.errors)


def test_gate_cli_returns_failure_and_writes_evidence(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "gate.json"
    report_path.write_text(
        json.dumps(_report(install_size=301 * MIB, peak_rss=100 * MIB)),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--report",
            str(report_path),
            "--profile",
            "jetson-nano",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert json.loads(output_path.read_text(encoding="utf-8"))["passed"] is False
