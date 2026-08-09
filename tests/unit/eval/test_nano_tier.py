"""Tests for the Nano sub-tier budget and report-level fit check."""

from __future__ import annotations

from openmed.eval.report import BenchmarkReport
from openmed.eval.tiers import (
    NANO_SUB_TIER,
    check_tier_fit,
    evaluate_tier_fit,
    tier_fit,
)

MIB = 1024 * 1024


def _report(
    *,
    param_count: int = 20_000_000,
    ram_mb: float = 128.0,
    p50_ms: float = 20.0,
    p95_ms: float = 50.0,
) -> BenchmarkReport:
    return BenchmarkReport(
        suite="synthetic-nano",
        model_name="synthetic-nano-model",
        device="cpu",
        fixture_count=2,
        generated_at="2026-08-09T00:00:00+00:00",
        metadata={"tier": "Nano", "param_count": param_count},
        metrics={
            "latency": {"p50_ms": p50_ms, "p95_ms": p95_ms},
            "resources": {
                "model_size_bytes": 24 * MIB,
                "peak_rss_bytes": int(ram_mb * MIB),
            },
        },
    )


def test_nano_budget_is_a_tiny_distillation_floor() -> None:
    assert NANO_SUB_TIER == {
        "parent_tier": "Tiny",
        "param_count_min": 10_000_000,
        "param_count_max": 30_000_000,
        "ram_mb_max": 150,
        "p50_ms_max": 25,
        "p95_ms_max": 60,
        "default_format": "INT8",
    }


def test_within_budget_report_passes() -> None:
    result = evaluate_tier_fit(_report())

    assert result.passed is True
    assert bool(result) is True
    assert result.tier == "Nano"
    assert result.failing_dimension is None
    assert result.observed["size_bytes"] == 24 * MIB
    assert check_tier_fit(_report()) is True
    assert tier_fit(_report()) is True


def test_over_ram_budget_fails() -> None:
    result = evaluate_tier_fit(_report(ram_mb=150.1))

    assert result.passed is False
    assert result.failing_dimension == "peak_rss_mb"
    assert result.violations["peak_rss_mb"]["limit"] == 150
    assert check_tier_fit(_report(ram_mb=150.1)) is False


def test_over_latency_budget_fails() -> None:
    result = evaluate_tier_fit(_report(p50_ms=25.1))

    assert result.passed is False
    assert result.failing_dimension == "p50_ms"
    assert result.violations["p50_ms"]["limit"] == 25


def test_over_p95_budget_fails() -> None:
    result = evaluate_tier_fit(_report(p95_ms=60.1))

    assert result.passed is False
    assert result.failing_dimension == "p95_ms"
    assert result.violations["p95_ms"]["limit"] == 60


def test_compact_report_fields_are_supported() -> None:
    report = {
        "tier": "Nano",
        "size_bytes": 24 * MIB,
        "peak_rss": 128.0,
        "p50": 20.0,
        "p95": 50.0,
    }

    assert check_tier_fit("Nano", report) is True


def test_parameter_count_above_nano_ceiling_fails_when_reported() -> None:
    result = evaluate_tier_fit(_report(param_count=30_000_001))

    assert result.passed is False
    assert result.failing_dimension == "param_count"
    assert result.violations["param_count"]["maximum"] == 30_000_000


def test_missing_tier_declaration_fails_closed() -> None:
    report = _report().to_dict()
    report["metadata"] = {}

    result = evaluate_tier_fit(report)

    assert result.passed is False
    assert result.failing_dimension == "tier"
