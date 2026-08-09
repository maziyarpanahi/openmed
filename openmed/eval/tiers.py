"""Canonical device-tier SLO budgets for evaluation gates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final, Mapping

NANO_SUB_TIER: Final[dict[str, int | str]] = {
    "parent_tier": "Tiny",
    "param_count_min": 10_000_000,
    "param_count_max": 30_000_000,
    "ram_mb_max": 150,
    "p50_ms_max": 25,
    "p95_ms_max": 60,
    "default_format": "INT8",
}

TIERS: Final[dict[str, dict[str, int | str | dict[str, dict[str, int | str]]]]] = {
    "Tiny": {
        "ram_mb_max": 350,
        "p50_ms_max": 60,
        "p95_ms_max": 150,
        "default_format": "INT8 (MLX-8bit / CoreML)",
        "sub_tiers": {"Nano": NANO_SUB_TIER},
    },
    "Base": {
        "ram_mb_max": 900,
        "p50_ms_max": 150,
        "p95_ms_max": 400,
        "default_format": "INT8 (FP fallback)",
    },
    "Large": {
        "ram_mb_max": 4096,
        "p50_ms_max": 250,
        "p95_ms_max": 800,
        "default_format": "FP16 (INT8 if recall holds)",
    },
    "Accurate-XLarge": {
        "ram_mb_max": 8192,
        "p50_ms_max": 400,
        "p95_ms_max": 1200,
        "default_format": "FP16 (INT8 if recall holds)",
    },
}


@dataclass(frozen=True)
class TierFitResult:
    """Structured result from comparing a report with a device-tier SLO."""

    passed: bool
    tier: str | None
    observed: Mapping[str, float | int | None]
    budget: Mapping[str, int | str]
    violations: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def failing_dimension(self) -> str | None:
        """Return the first failing dimension, if any."""
        return next(iter(self.violations), None)

    def __bool__(self) -> bool:
        """Return the pass/fail status for boolean-style checks."""
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable tier-fit evidence."""
        return {
            "budget": dict(self.budget),
            "failing_dimension": self.failing_dimension,
            "observed": dict(self.observed),
            "passed": self.passed,
            "tier": self.tier,
            "violations": {
                dimension: dict(details)
                for dimension, details in self.violations.items()
            },
        }


def evaluate_tier_fit(
    report_or_tier: Any = None,
    report: Any | None = None,
    *,
    declared_tier: str | None = None,
    tier: str | None = None,
) -> TierFitResult:
    """Evaluate a benchmark report against its declared device-tier budget.

    Reports may be ``BenchmarkReport`` instances, other report objects with a
    ``to_dict`` method, or mappings. The OM-018 nested paths are supported as
    well as compact ``p50``/``p95``/``peak_rss`` report fields. ``Nano`` uses
    the tighter sub-tier budget; an optional ``param_count`` is checked against
    its 10–30M range when present.

    ``report`` is the preferred first argument. For compatibility with callers
    that put the tier first, ``evaluate_tier_fit("Nano", report)`` and
    ``evaluate_tier_fit(report, "Nano")`` are also accepted.
    """

    report_payload, explicit_tier = _split_report_and_tier(
        report_or_tier,
        report,
        declared_tier=declared_tier or tier,
    )
    payload = _report_mapping(report_payload)
    metadata = _mapping(payload.get("metadata"))
    report_tier_value = _first_value(
        metadata.get("sub_tier"),
        metadata.get("tier"),
        payload.get("sub_tier"),
        payload.get("tier"),
    )
    selected_tier_value = explicit_tier or report_tier_value
    selected_tier = _canonical_tier(selected_tier_value)

    violations: dict[str, Mapping[str, Any]] = {}
    if selected_tier is None:
        violations["tier"] = {
            "observed": selected_tier_value,
            "reason": "missing_or_unknown",
        }
        return TierFitResult(
            passed=False,
            tier=None,
            observed={},
            budget={},
            violations=violations,
        )

    report_tier = _canonical_tier(report_tier_value)
    if (
        explicit_tier is not None
        and report_tier is not None
        and report_tier != selected_tier
    ):
        violations["tier"] = {
            "observed": report_tier_value,
            "expected": selected_tier,
            "reason": "declaration_mismatch",
        }

    budget = _budget_for(selected_tier)
    observed = {
        "size_bytes": _optional_int(
            _first_path_value(
                payload,
                ("size_bytes",),
                ("model_size_bytes",),
                ("artifact_size_bytes",),
                ("metadata", "size_bytes"),
                ("metadata", "model_size_bytes"),
                ("metrics", "resources", "size_bytes"),
                ("metrics", "resources", "model_size_bytes"),
                ("resources", "size_bytes"),
                ("resources", "model_size_bytes"),
            )
        ),
        "peak_rss_mb": _peak_rss_mb(payload),
        "p50_ms": _optional_float(_latency_value(payload, "p50")),
        "p95_ms": _optional_float(_latency_value(payload, "p95")),
    }

    raw_param_count = _first_path_value(
        payload,
        ("param_count",),
        ("metadata", "param_count"),
        ("metrics", "param_count"),
        ("metrics", "model", "param_count"),
        ("checkpoint", "param_count"),
    )
    if raw_param_count is not None:
        observed["param_count"] = _optional_int(raw_param_count)

    _add_limit_violation(
        violations,
        "peak_rss_mb",
        observed["peak_rss_mb"],
        int(budget["ram_mb_max"]),
    )
    _add_limit_violation(
        violations,
        "p50_ms",
        observed["p50_ms"],
        int(budget["p50_ms_max"]),
    )
    _add_limit_violation(
        violations,
        "p95_ms",
        observed["p95_ms"],
        int(budget["p95_ms_max"]),
    )

    if selected_tier == "Nano" and raw_param_count is not None:
        _add_param_count_violation(violations, observed.get("param_count"))

    return TierFitResult(
        passed=not violations,
        tier=selected_tier,
        observed=observed,
        budget=budget,
        violations=violations,
    )


def check_tier_fit(
    report_or_tier: Any = None,
    report: Any | None = None,
    *,
    declared_tier: str | None = None,
    tier: str | None = None,
) -> bool:
    """Return whether a benchmark report fits its declared tier budget."""
    return evaluate_tier_fit(
        report_or_tier,
        report,
        declared_tier=declared_tier,
        tier=tier,
    ).passed


def tier_fit(
    report_or_tier: Any = None,
    report: Any | None = None,
    *,
    declared_tier: str | None = None,
    tier: str | None = None,
) -> bool:
    """Backward-compatible boolean alias for :func:`check_tier_fit`."""
    return check_tier_fit(
        report_or_tier,
        report,
        declared_tier=declared_tier,
        tier=tier,
    )


def _split_report_and_tier(
    first: Any,
    second: Any | None,
    *,
    declared_tier: str | None,
) -> tuple[Any, str | None]:
    if isinstance(first, str) and second is not None:
        return second, declared_tier or first
    if isinstance(second, str):
        return first, declared_tier or second
    if first is None and second is not None:
        return second, declared_tier
    return first, declared_tier


def _report_mapping(report: Any) -> Mapping[str, Any]:
    if isinstance(report, Mapping):
        return report
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    return {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _canonical_tier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "nano": "Nano",
        "tiny": "Tiny",
        "base": "Base",
        "large": "Large",
        "accurate": "Accurate-XLarge",
        "accurate-xlarge": "Accurate-XLarge",
        "xlarge": "Accurate-XLarge",
    }
    return aliases.get(normalized)


def _budget_for(tier: str) -> dict[str, int | str]:
    values = NANO_SUB_TIER if tier == "Nano" else TIERS[tier]
    return {
        key: value for key, value in values.items() if isinstance(value, (int, str))
    }


def _first_path_value(payload: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = payload
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        if value is not None:
            return value
    return None


def _latency_value(payload: Mapping[str, Any], metric: str) -> Any:
    return _first_path_value(
        payload,
        (f"{metric}_ms",),
        (metric,),
        ("metadata", f"{metric}_ms"),
        ("metadata", metric),
        ("latency", f"{metric}_ms"),
        ("latency", metric),
        ("metrics", f"{metric}_ms"),
        ("metrics", metric),
        ("metrics", "latency", f"{metric}_ms"),
        ("metrics", "latency", metric),
    )


def _peak_rss_mb(payload: Mapping[str, Any]) -> float | None:
    bytes_value = _first_path_value(
        payload,
        ("peak_rss_bytes",),
        ("resources", "peak_rss_bytes"),
        ("metrics", "resources", "peak_rss_bytes"),
        ("metadata", "peak_rss_bytes"),
    )
    parsed_bytes = _optional_float(bytes_value)
    if parsed_bytes is not None:
        return parsed_bytes / (1024 * 1024)

    megabytes_value = _first_path_value(
        payload,
        ("peak_rss_mib",),
        ("peak_rss_mb",),
        ("ram_mb",),
        ("resources", "peak_rss_mib"),
        ("resources", "peak_rss_mb"),
        ("resources", "ram_mb"),
        ("metrics", "resources", "peak_rss_mib"),
        ("metrics", "resources", "peak_rss_mb"),
        ("metrics", "resources", "ram_mb"),
        ("metadata", "peak_rss_mib"),
        ("metadata", "peak_rss_mb"),
        ("metadata", "ram_mb"),
    )
    parsed_megabytes = _optional_float(megabytes_value)
    if parsed_megabytes is not None:
        return parsed_megabytes

    generic_value = _first_path_value(
        payload,
        ("peak_rss",),
        ("resources", "peak_rss"),
        ("metrics", "resources", "peak_rss"),
        ("metadata", "peak_rss"),
    )
    parsed_generic = _optional_float(generic_value)
    if parsed_generic is None:
        return None
    # ``peak_rss`` is bytes in the OM-018 harness; compact fixtures often use
    # MB. Values above 4 KiB are unambiguously a byte count for this budget.
    if abs(parsed_generic) > 4096:
        return parsed_generic / (1024 * 1024)
    return parsed_generic


def _add_limit_violation(
    violations: dict[str, Mapping[str, Any]],
    dimension: str,
    observed: float | int | None,
    limit: int,
) -> None:
    if observed is None:
        violations[dimension] = {"observed": None, "limit": limit, "reason": "missing"}
        return
    value = float(observed)
    if not math.isfinite(value) or value < 0.0:
        violations[dimension] = {
            "observed": observed,
            "limit": limit,
            "reason": "invalid",
        }
    elif value > float(limit):
        violations[dimension] = {
            "observed": observed,
            "limit": limit,
            "reason": "exceeds_limit",
        }


def _add_param_count_violation(
    violations: dict[str, Mapping[str, Any]],
    observed: float | int | None,
) -> None:
    minimum = int(NANO_SUB_TIER["param_count_min"])
    maximum = int(NANO_SUB_TIER["param_count_max"])
    if not isinstance(observed, int):
        violations["param_count"] = {
            "observed": observed,
            "minimum": minimum,
            "maximum": maximum,
            "reason": "invalid",
        }
    elif observed < minimum or observed > maximum:
        violations["param_count"] = {
            "observed": observed,
            "minimum": minimum,
            "maximum": maximum,
            "reason": "outside_range",
        }


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


__all__ = [
    "NANO_SUB_TIER",
    "TIERS",
    "TierFitResult",
    "check_tier_fit",
    "evaluate_tier_fit",
    "tier_fit",
]
