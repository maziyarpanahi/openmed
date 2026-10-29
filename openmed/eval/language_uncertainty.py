"""Stratified aggregate uncertainty without ranking underpowered language slices."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from types import MappingProxyType

from openmed.core.clinical_language import normalize_clinical_language

_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")


@dataclass(frozen=True)
class StratumCounts:
    """Bounded aggregate Bernoulli outcomes for one declared evaluation stratum."""

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
            raise ValueError("invalid stratum aggregate counts")


@dataclass(frozen=True)
class LanguageSlice:
    """A registered language, controlled label and immutable aggregate strata."""

    language: str
    label: str
    strata: Mapping[str, StratumCounts]

    def __post_init__(self) -> None:
        language, _locale = normalize_clinical_language(self.language)
        if language == "auto":
            raise ValueError("language slices require explicit language codes")
        if not isinstance(self.label, str) or not _CODE.fullmatch(self.label):
            raise ValueError("invalid controlled metric label")
        if not isinstance(self.strata, Mapping) or not 1 <= len(self.strata) <= 128:
            raise ValueError("provide 1 to 128 aggregate strata")
        for name, counts in self.strata.items():
            if (
                not isinstance(name, str)
                or not _CODE.fullmatch(name)
                or not isinstance(counts, StratumCounts)
            ):
                raise ValueError("invalid controlled stratum")
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "strata", MappingProxyType(dict(self.strata)))


def language_uncertainty_report(
    slices: Sequence[LanguageSlice],
    *,
    confidence: float = 0.95,
    power: float = 0.8,
    desired_effect: float = 0.1,
    minimum_stratum_trials: int = 20,
) -> dict[str, object]:
    """Return stratified Wilson bounds and conservative two-arm MDE summaries.

    Weights are observed stratum proportions. Bonferroni-adjusted Wilson bounds
    are combined by those weights, rather than pooling away small strata.
    Normal-approximation MDE uses worst-case Bernoulli variance and equal-size
    independent comparison arms. Underpowered or empty strata cannot be ranked.
    """
    for value, name, low, high in (
        (confidence, "confidence", 0.5, 0.9999999),
        (power, "power", 0.5, 1),
        (desired_effect, "desired effect", 0, 1),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low < value < high
        ):
            raise ValueError(f"invalid {name}")
    if (
        isinstance(minimum_stratum_trials, bool)
        or not isinstance(minimum_stratum_trials, int)
        or not 1 <= minimum_stratum_trials <= 1_000_000_000
    ):
        raise ValueError("invalid minimum stratum trial count")
    if len(slices) > 10_000 or not all(
        isinstance(item, LanguageSlice) for item in slices
    ):
        raise ValueError("provide bounded validated language slices")
    if len({(item.language, item.label) for item in slices}) != len(slices):
        raise ValueError("duplicate language and label slice")
    normal = NormalDist()
    z_effect = normal.inv_cdf((1 + confidence) / 2) + normal.inv_cdf(power)
    rows = []
    for item in sorted(slices, key=lambda row: (row.language, row.label)):
        total = sum(counts.trials for counts in item.strata.values())
        z = normal.inv_cdf(1 - (1 - confidence) / (2 * len(item.strata)))
        strata = []
        lower = upper = estimate = 0.0
        insufficient = False
        for name, counts in sorted(item.strata.items()):
            n = counts.trials
            mde = z_effect * math.sqrt(0.5 / n) if n else None
            enough = (
                n >= minimum_stratum_trials
                and mde is not None
                and mde <= desired_effect
            )
            insufficient = insufficient or not enough
            interval = None
            rate = counts.successes / n if n else None
            if n:
                denominator = 1 + z * z / n
                center = (rate + z * z / (2 * n)) / denominator
                radius = (
                    z
                    * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
                    / denominator
                )
                interval = (max(0.0, center - radius), min(1.0, center + radius))
                weight = n / total
                lower += weight * interval[0]
                upper += weight * interval[1]
                estimate += weight * rate
            strata.append(
                {
                    "stratum": name,
                    "trials": n,
                    "estimate": rate,
                    "confidence_interval": interval,
                    "minimum_detectable_effect": mde,
                    "evidence": "sufficient" if enough else "insufficient_evidence",
                }
            )
        rows.append(
            {
                "language": item.language,
                "label": item.label,
                "trials": total,
                "estimate": estimate if total else None,
                "stratified_confidence_interval": (lower, upper) if total else None,
                "evidence": "insufficient_evidence" if insufficient else "sufficient",
                "ranking_allowed": not insufficient,
                "strata": strata,
            }
        )
    return {
        "schema_version": 1,
        "confidence": confidence,
        "power": power,
        "desired_effect": desired_effect,
        "minimum_stratum_trials": minimum_stratum_trials,
        "method": "bonferroni_stratified_wilson_and_equal_arm_normal_mde",
        "slices": rows,
    }
