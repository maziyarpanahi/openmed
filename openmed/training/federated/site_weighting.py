"""Transparent bounded weighting over anonymous coarse site declarations."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import islice
from types import MappingProxyType

_VOLUME = {"tiny": 1, "small": 2, "medium": 4, "large": 8, "very_large": 16}
_STRATEGIES = frozenset({"uniform", "volume", "inverse_volume"})
_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


def _bounded(values: Iterable, maximum: int) -> tuple:
    if isinstance(values, (str, bytes)):
        raise ValueError("weighting requires a bounded collection")
    try:
        result = tuple(islice(values, maximum + 1))
    except Exception:
        raise ValueError("invalid weighting collection") from None
    if len(result) > maximum:
        raise ValueError("weighting collection limit exceeded")
    return result


def _fraction(value: object) -> Fraction:
    if type(value) is int:
        result = Fraction(value)
    elif type(value) is float and math.isfinite(value):
        result = Fraction(str(value))
    elif isinstance(value, Fraction):
        result = value
    else:
        raise ValueError("invalid weighting fraction")
    if not 0 <= result <= 1 or result.denominator > 10**9:
        raise ValueError("weighting fraction out of range or precision")
    return result


def _code(value: object) -> str:
    if not isinstance(value, str) or not _CODE.fullmatch(value):
        raise ValueError("invalid protected weighting code")
    return value


@dataclass(frozen=True, slots=True)
class CoarseSiteMetadata:
    """An anonymous position's volume class and controlled protected-slice codes.

    No site identifier or raw record count is accepted. Volume factors 1,2,4,8,16
    are declared relative coarse classes, not measured patient counts.
    """

    volume_class: str
    protected_slices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.volume_class, str) or self.volume_class not in _VOLUME:
            raise ValueError("invalid coarse weighting volume class")
        codes = tuple(_code(value) for value in _bounded(self.protected_slices, 16))
        if len(set(codes)) != len(codes):
            raise ValueError("duplicate protected weighting code")
        object.__setattr__(self, "protected_slices", tuple(sorted(codes)))


@dataclass(frozen=True, slots=True)
class SiteWeightingPolicy:
    """Declared strategy, per-position caps/floors and protected minimum coverage.

    Constraints are enforced rather than silently relaxed. No automatic strategy
    or clinical fairness decision is made; callers choose the policy explicitly.
    """

    strategy: str
    floors: tuple[Fraction, ...] | None = None
    caps: tuple[Fraction, ...] | None = None
    protected_minimums: Mapping[str, Fraction] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, str) or self.strategy not in _STRATEGIES:
            raise ValueError("invalid weighting strategy")
        for name in ("floors", "caps"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, tuple(_fraction(item) for item in _bounded(value, 128))
                )
        if (
            not isinstance(self.protected_minimums, Mapping)
            or len(self.protected_minimums) > 16
        ):
            raise ValueError("invalid protected weighting constraints")
        minima = {
            _code(code): _fraction(value)
            for code, value in self.protected_minimums.items()
        }
        object.__setattr__(self, "protected_minimums", MappingProxyType(minima))


def _groups(groups: Iterable[CoarseSiteMetadata]) -> tuple[CoarseSiteMetadata, ...]:
    groups = _bounded(groups, 128)
    if not groups or any(not isinstance(group, CoarseSiteMetadata) for group in groups):
        raise ValueError("invalid coarse weighting groups")
    return groups


def _candidate(
    groups: tuple[CoarseSiteMetadata, ...], policy: SiteWeightingPolicy
) -> tuple[Fraction, ...]:
    if not isinstance(policy, SiteWeightingPolicy):
        raise ValueError("invalid weighting policy")
    count = len(groups)
    floors = (Fraction(0),) * count if policy.floors is None else policy.floors
    caps = (Fraction(1),) * count if policy.caps is None else policy.caps
    if (
        len(floors) != count
        or len(caps) != count
        or any(floor > cap for floor, cap in zip(floors, caps))
    ):
        raise ValueError("weighting bounds do not match anonymous positions")
    if sum(floors) > 1 or sum(caps) < 1:
        raise ValueError("weighting bounds cannot normalize to one")
    scores = tuple(
        Fraction(1)
        if policy.strategy == "uniform"
        else Fraction(_VOLUME[group.volume_class])
        if policy.strategy == "volume"
        else Fraction(1, _VOLUME[group.volume_class])
        for group in groups
    )
    weights = list(floors)
    remaining = Fraction(1) - sum(weights)
    active = {index for index in range(count) if caps[index] > weights[index]}
    # At least one position saturates per capped iteration; otherwise finish.
    while remaining:
        total_score = sum(scores[index] for index in active)
        proposed = {index: remaining * scores[index] / total_score for index in active}
        saturated = {
            index for index in active if proposed[index] >= caps[index] - weights[index]
        }
        if not saturated:
            for index, amount in proposed.items():
                weights[index] += amount
            remaining = Fraction(0)
        else:
            for index in sorted(saturated):
                available = caps[index] - weights[index]
                weights[index] += available
                remaining -= available
            active -= saturated
    return tuple(weights)


def _coverage(
    groups: tuple[CoarseSiteMetadata, ...],
    weights: tuple[Fraction, ...],
    policy: SiteWeightingPolicy,
) -> dict[str, Fraction]:
    return {
        code: sum(
            (
                weight
                for group, weight in zip(groups, weights)
                if code in group.protected_slices
            ),
            Fraction(0),
        )
        for code in sorted(policy.protected_minimums)
    }


def compute_site_weights(
    groups: Iterable[CoarseSiteMetadata], policy: SiteWeightingPolicy
) -> tuple[Fraction, ...]:
    """Return normalized exact weights only when every declared constraint holds.

    Args:
        groups: One to 128 anonymous coarse metadata positions.
        policy: Explicit strategy, per-position bounds and protected minima.

    Returns:
        Exact rational weights in input position order, summing to one.

    Raises:
        ValueError: Invalid bounds or selected policy violates protected minima.
            This is not a proof that all possible weighting strategies are infeasible.
    """
    groups = _groups(groups)
    weights = _candidate(groups, policy)
    coverage = _coverage(groups, weights, policy)
    if any(
        coverage[code] < minimum for code, minimum in policy.protected_minimums.items()
    ):
        raise ValueError("selected weighting policy violates protected minimum")
    return weights


def _ratio(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def compare_site_weighting(
    groups: Iterable[CoarseSiteMetadata], policy: SiteWeightingPolicy
) -> dict[str, object]:
    """Compare selected constrained weights against unconstrained baselines.

    A protected-constraint failure suppresses selected export weights and deltas;
    uniform and coarse-volume vectors remain labeled counterfactual baselines.
    No site IDs, raw counts, local metrics or source examples enter the report.
    """
    groups = _groups(groups)
    candidate = _candidate(groups, policy)
    coverage = _coverage(groups, candidate, policy)
    met = all(
        coverage[code] >= minimum for code, minimum in policy.protected_minimums.items()
    )
    baselines = {
        strategy: _candidate(groups, SiteWeightingPolicy(strategy))
        for strategy in ("uniform", "volume")
    }
    return {
        "schema_version": 1,
        "strategy": policy.strategy,
        "outcome": "accept" if met else "hold",
        "reasons": [] if met else ["protected_minimum_unsatisfied"],
        "bounds": {
            "floors": [
                _ratio(value)
                for value in (
                    policy.floors
                    if policy.floors is not None
                    else (Fraction(0),) * len(groups)
                )
            ],
            "caps": [
                _ratio(value)
                for value in (
                    policy.caps
                    if policy.caps is not None
                    else (Fraction(1),) * len(groups)
                )
            ],
        },
        "selected_weights": [_ratio(weight) for weight in candidate] if met else None,
        "counterfactual_baselines": {
            name: [_ratio(weight) for weight in weights]
            for name, weights in baselines.items()
        },
        "selected_minus_baseline": {
            name: [_ratio(a - b) for a, b in zip(candidate, weights)] if met else None
            for name, weights in baselines.items()
        },
        "protected_coverage": {
            code: {
                "minimum": _ratio(policy.protected_minimums[code]),
                "selected": _ratio(coverage[code]),
                "counterfactual_baselines": {
                    name: _ratio(_coverage(groups, weights, policy)[code])
                    for name, weights in baselines.items()
                },
            }
            for code in sorted(coverage)
        },
    }


def render_site_weighting_comparison(
    groups: Iterable[CoarseSiteMetadata], policy: SiteWeightingPolicy
) -> str:
    """Render a fresh validated comparison as deterministic content-free JSON."""
    return (
        json.dumps(
            compare_site_weighting(groups, policy),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
