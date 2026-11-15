"""Aggregate-only fidelity evidence from locally evaluated synthetic records."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")
_METRICS = ("constraint_validity", "fact_consistency", "task_utility")


@dataclass(frozen=True)
class FidelityCounts:
    """Precomputed successes/trials, never reference records or task payloads."""

    successes: int
    trials: int

    def __post_init__(self) -> None:
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (self.successes, self.trials)
            )
            or not 0 <= self.successes <= self.trials <= 1_000_000_000
        ):
            raise ValueError("fidelity counts must satisfy 0 <= successes <= trials")


@dataclass(frozen=True)
class FidelityEvidence:
    """Aggregate outcomes and controlled synthetic distribution categories."""

    metrics: Mapping[str, FidelityCounts]
    distribution: Mapping[str, int]
    unique_record_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.metrics, Mapping)
            or set(self.metrics) != set(_METRICS)
            or not all(
                isinstance(value, FidelityCounts) for value in self.metrics.values()
            )
        ):
            raise ValueError("provide exactly the three validated fidelity metrics")
        if not isinstance(self.distribution, Mapping) or len(self.distribution) > 256:
            raise ValueError("distribution must be a bounded controlled mapping")
        for code, count in self.distribution.items():
            if not isinstance(code, str) or not _CODE.fullmatch(code):
                raise ValueError("invalid controlled distribution category")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 0 <= count <= 1_000_000_000
            ):
                raise ValueError(
                    "distribution counts must be bounded non-negative integers"
                )
        total = sum(self.distribution.values())
        if (
            isinstance(self.unique_record_count, bool)
            or not isinstance(self.unique_record_count, int)
            or not 0 <= self.unique_record_count <= total
        ):
            raise ValueError("unique synthetic record count exceeds distribution total")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(
            self, "distribution", MappingProxyType(dict(self.distribution))
        )


def wilson_interval(counts: FidelityCounts) -> tuple[float, float] | None:
    """Return a descriptive two-sided 95% Wilson interval, or None without trials."""
    if not isinstance(counts, FidelityCounts):
        raise ValueError("expected validated counts")
    if not counts.trials:
        return None
    n = counts.trials
    p = counts.successes / n
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def synthetic_fidelity_report(
    evidence: FidelityEvidence,
    *,
    minimum_rates: Mapping[str, float],
    minimum_trials: int = 5,
    reference: FidelityEvidence | None = None,
) -> dict[str, object]:
    """Score independent aggregate fidelity axes without inspecting real records.

    Thresholds apply separately, never a weighted average. Insufficient samples
    return inconclusive with suppressed rates. Reference comparisons require
    already-approved local aggregates; their records never enter this API.
    This reports supplied evaluations, not scientific or clinical validation.
    """
    if not isinstance(evidence, FidelityEvidence) or (
        reference is not None and not isinstance(reference, FidelityEvidence)
    ):
        raise ValueError("expected validated aggregate fidelity evidence")
    if (
        isinstance(minimum_trials, bool)
        or not isinstance(minimum_trials, int)
        or not 1 <= minimum_trials <= 1_000_000_000
    ):
        raise ValueError("minimum trials must be a bounded positive integer")
    if not isinstance(minimum_rates, Mapping) or set(minimum_rates) != set(_METRICS):
        raise ValueError("provide a pinned minimum rate for every metric")
    for rate in minimum_rates.values():
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or not 0 <= rate <= 1
        ):
            raise ValueError("minimum rates must be finite within [0, 1]")
    metrics = {}
    failed = []
    incomplete = []
    for metric in _METRICS:
        counts = evidence.metrics[metric]
        enough = counts.trials >= minimum_trials
        rate = counts.successes / counts.trials if enough else None
        passed = None if rate is None else rate >= minimum_rates[metric]
        if passed is False:
            failed.append(metric)
        if passed is None:
            incomplete.append(metric)
        result = {
            "trials": counts.trials,
            "rate": rate,
            "wilson_95_interval": wilson_interval(counts) if enough else None,
            "minimum_rate": minimum_rates[metric],
            "passed": passed,
        }
        if reference is not None:
            reference_counts = reference.metrics[metric]
            reference_rate = (
                reference_counts.successes / reference_counts.trials
                if reference_counts.trials >= minimum_trials
                else None
            )
            result["reference_rate"] = reference_rate
            result["rate_delta"] = (
                None
                if rate is None or reference_rate is None
                else rate - reference_rate
            )
            result["reference_wilson_95_interval"] = (
                wilson_interval(reference_counts)
                if reference_rate is not None
                else None
            )
        metrics[metric] = result
    total = sum(evidence.distribution.values())
    diversity = (
        evidence.unique_record_count / total if total >= minimum_trials else None
    )
    comparison = None
    if reference is not None:
        reference_total = sum(reference.distribution.values())
        if total >= minimum_trials and reference_total >= minimum_trials:
            categories = set(evidence.distribution) | set(reference.distribution)
            comparison = 0.5 * math.fsum(
                abs(
                    evidence.distribution.get(code, 0) / total
                    - reference.distribution.get(code, 0) / reference_total
                )
                for code in categories
            )
    return {
        "schema_version": 1,
        "verdict": "fail"
        if failed
        else "inconclusive"
        if incomplete
        else "meets_declared_limits",
        "failed_metrics": failed,
        "incomplete_metrics": incomplete,
        "minimum_trials": minimum_trials,
        "metrics": metrics,
        "distribution_total": total,
        "unique_record_fraction": diversity,
        "reference_total_variation": comparison,
    }
