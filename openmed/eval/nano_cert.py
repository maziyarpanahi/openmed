"""Nano sub-tier certification for measured benchmark reports."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final, Mapping

from openmed.eval.report import BenchmarkReport
from openmed.eval.tiers import NANO_SUB_TIER

NANO_BUDGET: Final[Mapping[str, int | str]] = NANO_SUB_TIER


@dataclass(frozen=True)
class NanoCertification:
    """Structured Nano budget certification evidence."""

    passed: bool
    failing_dimension: str | None
    observed: Mapping[str, float | int | None]
    budget: Mapping[str, int | str] = field(default_factory=lambda: dict(NANO_BUDGET))
    failures: Mapping[str, Mapping[str, float | int | None | str]] = field(
        default_factory=dict
    )
    sub_tier: str = "Nano"
    parent_tier: str = "Tiny"

    def __bool__(self) -> bool:
        """Return the pass/fail status for simple boolean checks."""
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable certification evidence."""
        return {
            "passed": self.passed,
            "failing_dimension": self.failing_dimension,
            "sub_tier": self.sub_tier,
            "parent_tier": self.parent_tier,
            "observed": dict(self.observed),
            "budget": dict(self.budget),
            "failures": {
                dimension: dict(details) for dimension, details in self.failures.items()
            },
        }


def certify(report: BenchmarkReport) -> NanoCertification:
    """Certify a measured BenchmarkReport against the Nano sub-tier budget."""

    metrics = report.metrics or {}
    metadata = report.metadata or {}
    return certify_measurements(
        param_count=_param_count(metrics, metadata),
        ram_mb=_ram_mb(metrics, metadata),
        p50_ms=_latency_ms(metrics, metadata, "p50_ms"),
        p95_ms=_latency_ms(metrics, metadata, "p95_ms"),
    )


def certify_measurements(
    *,
    param_count: Any,
    ram_mb: Any,
    p50_ms: Any,
    p95_ms: Any,
) -> NanoCertification:
    """Certify measured Nano dimensions without re-running a generic tier gate."""

    observed = {
        "param_count": _optional_int(param_count),
        "ram_mb": _optional_float(ram_mb),
        "p50_ms": _optional_float(p50_ms),
        "p95_ms": _optional_float(p95_ms),
    }
    failures: dict[str, Mapping[str, float | int | None | str]] = {}

    param_count = observed["param_count"]
    param_min = int(NANO_BUDGET["param_count_min"])
    param_max = int(NANO_BUDGET["param_count_max"])
    if not isinstance(param_count, int):
        failures["param_count"] = {
            "observed": param_count,
            "minimum": param_min,
            "maximum": param_max,
            "reason": "missing",
        }
    elif param_count < param_min or param_count > param_max:
        failures["param_count"] = {
            "observed": param_count,
            "minimum": param_min,
            "maximum": param_max,
            "reason": "outside_range",
        }

    _add_limit_failure(
        failures,
        "ram_mb",
        observed["ram_mb"],
        int(NANO_BUDGET["ram_mb_max"]),
    )
    _add_limit_failure(
        failures,
        "p50_ms",
        observed["p50_ms"],
        int(NANO_BUDGET["p50_ms_max"]),
    )
    _add_limit_failure(
        failures,
        "p95_ms",
        observed["p95_ms"],
        int(NANO_BUDGET["p95_ms_max"]),
    )

    failing_dimension = next(iter(failures), None)
    return NanoCertification(
        passed=not failures,
        failing_dimension=failing_dimension,
        observed=observed,
        failures=failures,
    )


def _add_limit_failure(
    failures: dict[str, Mapping[str, float | int | None | str]],
    dimension: str,
    observed: float | int | None,
    limit: int,
) -> None:
    if observed is None:
        failures[dimension] = {
            "observed": observed,
            "limit": limit,
            "reason": "missing",
        }
        return

    value = float(observed)
    if not math.isfinite(value) or value < 0.0:
        failures[dimension] = {
            "observed": observed,
            "limit": limit,
            "reason": "invalid",
        }
        return

    if value > float(limit):
        failures[dimension] = {
            "observed": observed,
            "limit": limit,
            "reason": "exceeds_limit",
        }


def _param_count(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> int | None:
    return _optional_int(
        _first_value(
            metadata.get("param_count"),
            metrics.get("param_count"),
            _nested(metrics, "model", "param_count"),
            _nested(metrics, "resources", "param_count"),
        )
    )


def _latency_ms(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    key: str,
) -> float | None:
    return _optional_float(
        _first_value(
            metadata.get(key),
            _nested(metrics, "latency", key),
        )
    )


def _ram_mb(metrics: Mapping[str, Any], metadata: Mapping[str, Any]) -> float | None:
    value = _first_value(
        metadata.get("ram_mb"),
        metadata.get("peak_rss_mib"),
        _nested(metrics, "resources", "peak_rss_mib"),
        _nested(metrics, "resources", "ram_mb"),
    )
    parsed = _optional_float(value)
    if parsed is not None:
        return parsed

    bytes_value = _first_value(
        metadata.get("peak_rss_bytes"),
        _nested(metrics, "resources", "peak_rss_bytes"),
    )
    bytes_parsed = _optional_float(bytes_value)
    if bytes_parsed is None:
        return None
    return bytes_parsed / (1024 * 1024)


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not parsed.is_integer():
        return None
    return int(parsed)


__all__ = ["NANO_BUDGET", "NanoCertification", "certify", "certify_measurements"]
