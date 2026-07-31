from __future__ import annotations

from openmed.eval.nano_cert import certify
from openmed.eval.report import BenchmarkReport


def _report(
    *,
    param_count: int = 20_000_000,
    ram_mb: float = 128.0,
    p50_ms: float = 20.0,
    p95_ms: float = 50.0,
    tier: str = "Nano",
) -> BenchmarkReport:
    return BenchmarkReport(
        suite="golden",
        model_name="unit-nano",
        device="cpu",
        fixture_count=1,
        generated_at="2026-06-27T00:00:00+00:00",
        metadata={
            "tier": tier,
            "param_count": param_count,
            "format": "mlx-8bit",
        },
        metrics={
            "latency": {"p50_ms": p50_ms, "p95_ms": p95_ms},
            "resources": {"peak_rss_mib": ram_mb},
        },
    )


def test_within_nano_budget_certifies() -> None:
    result = certify(_report())

    assert result.passed is True
    assert bool(result) is True
    assert result.failing_dimension is None
    assert result.parent_tier == "Tiny"
    assert result.to_dict()["budget"]["default_format"] == "INT8"


def test_over_ram_budget_fails_with_dimension() -> None:
    result = certify(_report(ram_mb=151.0))

    assert result.passed is False
    assert bool(result) is False
    assert result.failing_dimension == "ram_mb"
    assert result.failures["ram_mb"]["limit"] == 150
    assert result.failures["ram_mb"]["reason"] == "exceeds_limit"


def test_over_p95_budget_fails_with_dimension() -> None:
    result = certify(_report(p95_ms=60.1))

    assert result.passed is False
    assert result.failing_dimension == "p95_ms"
    assert result.failures["p95_ms"]["limit"] == 60


def test_tiny_but_not_nano_artifact_is_not_certified_as_nano() -> None:
    result = certify(_report(param_count=44_000_000, tier="Tiny"))

    assert result.passed is False
    assert result.failing_dimension == "param_count"
    assert result.failures["param_count"]["maximum"] == 30_000_000
