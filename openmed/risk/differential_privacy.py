"""Differential-privacy mechanisms for already-aggregated values.

The functions in this module accept aggregate values, not source records. They
apply calibrated Laplace or Gaussian noise locally and keep an optional,
numeric-only privacy ledger for one dataset. The ledger uses basic sequential
composition: query epsilon and delta values are added across releases.

The Gaussian scale is calibrated against the exact Gaussian privacy profile
(Balle and Wang, 2018, Theorem 8), rather than extending the classical bound
beyond its proven epsilon range. Sensitivity is the caller's
responsibility: it must match the adjacency relation and any clipping or
bounded contribution assumptions for the aggregate.

The ``*_discrete_*`` helpers and :func:`release_snapped` draw noise with the
exact integer samplers in :mod:`openmed.risk.discrete_sampling`, so their
output distributions do not depend on floating-point rounding. The original
float helpers remain available for compatibility; every spend record names the
sampler that produced it.

This module does not accept or persist raw rows, identifiers, or query values
in its accounting records. A noisy aggregate is not a clinical decision,
compliance certification, or guarantee of zero disclosure risk.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral
from typing import Any, Literal, Protocol, TypeVar

from .discrete_sampling import (
    DP_SAMPLERS,
    DPRandomSource,
    DPSampler,
    _discrete_gaussian,
    _discrete_laplace,
    _dp_source,
    _positive_rational,
)

__all__ = [
    "AggregateKind",
    "BudgetDecision",
    "BudgetExceeded",
    "DPMechanism",
    "DifferentialPrivacy",
    "PrivacyBudget",
    "PrivacyBudgetExceeded",
    "PrivacyBudgetStatus",
    "PrivacySpend",
    "UtilityPoint",
    "UtilityReport",
    "discrete_gaussian_sigma_squared",
    "gaussian_mechanism",
    "gaussian_noise",
    "gaussian_privacy_delta",
    "gaussian_scale",
    "gaussian_stddev",
    "laplace_mechanism",
    "laplace_noise",
    "laplace_scale",
    "release_aggregate",
    "release_count",
    "release_discrete_count",
    "release_discrete_histogram",
    "release_histogram",
    "release_mean",
    "release_snapped",
    "release_sum",
    "snapped_epsilon",
    "utility_report",
    "utility_vs_epsilon",
]

DPMechanism = Literal["laplace", "gaussian"]
AggregateKind = Literal["count", "sum", "mean", "histogram", "aggregate"]

_HistogramKey = TypeVar("_HistogramKey")

# Snapped outputs are integer multiples of a power-of-two grid. Keeping the
# multiple within 2**52 makes every output exactly representable as a float.
_MAX_GRID_STEPS = 2**52
# Relative safety margin and dyadic rounding for discrete Gaussian calibration.
_ZCDP_MARGIN = 2.0**-30
_SIGMA_SQUARED_GRID = 2**32
_GAUSSIAN_SAMPLERS = frozenset({"float_gaussian", "discrete_gaussian"})
_LAPLACE_SAMPLERS = frozenset(
    {"float_laplace", "discrete_laplace", "grid_discrete_laplace"}
)


class _RandomSource(Protocol):
    def random(self) -> float:
        """Return a value in the half-open interval ``[0, 1)``."""

    def gauss(self, mu: float, sigma: float) -> float:
        """Return a Gaussian sample."""


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise TypeError(f"{field_name} must be a real number") from None
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _epsilon(value: Any) -> float:
    result = _finite_float(value, field_name="epsilon")
    if result <= 0.0:
        raise ValueError("epsilon must be greater than zero")
    return result


def _delta(value: Any) -> float:
    result = _finite_float(value, field_name="delta")
    if result < 0.0 or result >= 1.0:
        raise ValueError("delta must be in the interval [0, 1)")
    return result


def _sensitivity(value: Any) -> float:
    result = _finite_float(value, field_name="sensitivity")
    if result < 0.0:
        raise ValueError("sensitivity must be non-negative")
    return result


def _scalar(value: Any, *, field_name: str = "value") -> float:
    result = _finite_float(value, field_name=field_name)
    return result


def _mechanism(value: Any) -> DPMechanism:
    if not isinstance(value, str):
        raise TypeError("mechanism must be 'laplace' or 'gaussian'")
    normalized = value.strip().lower()
    if normalized not in {"laplace", "gaussian"}:
        raise ValueError("mechanism must be 'laplace' or 'gaussian'")
    return normalized  # type: ignore[return-value]


def _aggregate_kind(value: Any) -> AggregateKind:
    if not isinstance(value, str):
        raise TypeError("aggregate kind must be a string")
    normalized = value.strip().lower()
    allowed = {"count", "sum", "mean", "histogram", "aggregate"}
    if normalized not in allowed:
        raise ValueError(
            "aggregate kind must be count, sum, mean, histogram, or aggregate"
        )
    return normalized  # type: ignore[return-value]


def _positive_sensitivity(value: Any) -> float:
    result = _sensitivity(value)
    if result <= 0.0:
        raise ValueError("sensitivity must be greater than zero")
    return result


def _integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    return int(value)


def _sampler(value: Any, mechanism: DPMechanism) -> DPSampler:
    if not isinstance(value, str):
        raise TypeError("sampler must be a string")
    if value not in DP_SAMPLERS:
        raise ValueError("sampler is not a supported noise sampler")
    if (value in _GAUSSIAN_SAMPLERS and mechanism != "gaussian") or (
        value in _LAPLACE_SAMPLERS and mechanism != "laplace"
    ):
        raise ValueError("sampler does not match the mechanism")
    return value  # type: ignore[return-value]


def _calibration_method(mechanism: DPMechanism, sampler: DPSampler) -> str:
    if sampler == "discrete_gaussian":
        return "zcdp_discrete_gaussian"
    if sampler == "grid_discrete_laplace":
        return "grid_snapped_laplace"
    return "analytic_gaussian" if mechanism == "gaussian" else "laplace"


def _random_source(rng: _RandomSource | None) -> _RandomSource:
    if rng is None:
        return random.SystemRandom()
    if not any(
        callable(getattr(rng, method, None))
        for method in ("random", "gauss", "normalvariate")
    ):
        raise TypeError("rng must provide random(), gauss(), or normalvariate()")
    return rng


def laplace_scale(sensitivity: float, epsilon: float) -> float:
    """Return the Laplace scale ``sensitivity / epsilon``."""

    sensitivity_value = _sensitivity(sensitivity)
    return sensitivity_value / _epsilon(epsilon)


def _scaled_erfc(value: float) -> float:
    # exp(x*x) * erfc(x), without overflowing in the positive far tail.
    if value <= 26.0:
        return math.exp(value * value) * math.erfc(value)
    inverse_square = (1.0 / value) ** 2
    term = total = 1.0
    for index in range(1, 12):
        term *= -(2 * index - 1) * inverse_square / 2.0
        total += term
    return total / math.sqrt(math.pi) / value


def gaussian_privacy_delta(sigma: float, sensitivity: float, epsilon: float) -> float:
    """Return a conservative Gaussian privacy profile at the declared L2 sensitivity.

    Uses Theorem 8 of https://proceedings.mlr.press/v80/balle18a.html.
    A scaled complementary error function avoids ``exp(epsilon)`` overflow.
    Zero sensitivity is a constant query and has zero privacy loss.
    Near catastrophic subtraction cancellation, return the first Gaussian
    tail alone: a conservative upper bound, never a false zero privacy loss.
    """

    sigma_value = _sensitivity(sigma)
    sensitivity_value = _sensitivity(sensitivity)
    epsilon_value = _epsilon(epsilon)
    if sensitivity_value == 0.0:
        return 0.0
    if sigma_value == 0.0:
        return 1.0
    ratio = sigma_value / sensitivity_value
    if ratio == 0.0:
        return 1.0
    x = 0.5 / ratio
    y = epsilon_value * ratio
    a = (y - x) / math.sqrt(2.0)
    b = (y + x) / math.sqrt(2.0)
    first = 0.5 * math.erfc(a)
    # b*b - a*a == epsilon, so this equals exp(epsilon)*Phi(-x-y).
    second = 0.5 * math.exp(-a * a) * _scaled_erfc(b)
    if first - second <= 1e-8 * first:
        return first
    # CDF subtraction can understate a small positive profile by a few ulps.
    # Include the rounding allowance before calibration compares to delta.
    allowance = 4.0 * (math.ulp(first) + math.ulp(second))
    return max(0.0, min(1.0, first - second + allowance))


def gaussian_scale(sensitivity: float, epsilon: float, delta: float) -> float:
    """Return an analytically calibrated Gaussian standard deviation.

    ``delta`` must be positive because a Gaussian mechanism cannot use this
    calibration at ``delta=0``. Bisection maintains a privacy-safe upper bound
    and adds a small floating-point margin before checking the final scale.
    Unrepresentable scales fail closed instead of returning zero or infinity.
    """

    sensitivity_value = _sensitivity(sensitivity)
    epsilon_value = _epsilon(epsilon)
    delta_value = _delta(delta)
    if delta_value <= 0.0:
        raise ValueError("Gaussian mechanism requires delta greater than zero")
    if sensitivity_value == 0.0:
        return 0.0
    lower, upper = 0.0, 1.0
    while gaussian_privacy_delta(upper, 1.0, epsilon_value) > delta_value:
        upper *= 2.0
        if not math.isfinite(upper):
            raise ValueError("Gaussian calibration is not representable")
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if gaussian_privacy_delta(midpoint, 1.0, epsilon_value) > delta_value:
            lower = midpoint
        else:
            upper = midpoint
    scale = sensitivity_value * upper * (1.0 + 1e-12)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Gaussian calibration is not representable")
    if gaussian_privacy_delta(scale, sensitivity_value, epsilon_value) > delta_value:
        raise ValueError("Gaussian calibration failed its privacy check")
    return scale


gaussian_stddev = gaussian_scale


def discrete_gaussian_sigma_squared(
    sensitivity: float,
    epsilon: float,
    delta: float,
) -> Fraction:
    """Return an exact discrete Gaussian variance parameter for ``(epsilon, delta)``.

    The discrete Gaussian with variance parameter ``sigma_squared`` added to
    an integer query with L2 sensitivity ``sensitivity`` satisfies
    ``rho = sensitivity**2 / (2 * sigma_squared)``-zCDP (Canonne, Kamath and
    Steinke 2020, Theorems 4 and 14). ``rho``-zCDP implies
    ``(rho + 2 * sqrt(rho * log(1 / delta)), delta)``-DP (Bun and Steinke
    2016, Proposition 1.3). This helper solves that conversion for ``rho``,
    applies a small relative safety margin, rounds ``sigma_squared`` up to a
    multiple of ``2**-32``, and re-checks the conversion. The conversion is
    conservative, so the variance is larger than the analytic continuous
    Gaussian calibration in :func:`gaussian_scale` for the same budget.

    Args:
        sensitivity: Positive L2 sensitivity of the integer-valued query.
        epsilon: Positive finite epsilon.
        delta: Delta in ``(0, 1)``.

    Returns:
        The variance parameter as an exact ``Fraction``.

    Raises:
        TypeError: If a parameter is not a real number.
        ValueError: If a parameter is out of range or the calibration is not
            representable.
    """

    sensitivity_value = _positive_sensitivity(sensitivity)
    epsilon_value = _epsilon(epsilon)
    delta_value = _delta(delta)
    if delta_value <= 0.0:
        raise ValueError("Gaussian mechanism requires delta greater than zero")
    log_term = -math.log(delta_value)
    # Stable form of (sqrt(log_term + epsilon) - sqrt(log_term)) ** 2.
    root_sum = math.sqrt(log_term + epsilon_value) + math.sqrt(log_term)
    rho = (epsilon_value / root_sum) ** 2 * (1.0 - _ZCDP_MARGIN)
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("Gaussian calibration is not representable")
    exact = Fraction(sensitivity_value) ** 2 / (2 * Fraction(rho))
    sigma_squared = Fraction(
        math.ceil(exact * _SIGMA_SQUARED_GRID), _SIGMA_SQUARED_GRID
    )
    achieved = float(Fraction(sensitivity_value) ** 2 / (2 * sigma_squared))
    if achieved + 2.0 * math.sqrt(achieved * log_term) > epsilon_value:
        raise ValueError("Gaussian calibration failed its privacy check")
    return sigma_squared


def laplace_noise(
    sensitivity: float,
    epsilon: float,
    *,
    rng: _RandomSource | None = None,
) -> float:
    """Draw calibrated Laplace noise for one scalar query.

    Passing a seeded ``random.Random`` instance is useful for synthetic tests.
    Production calls default to ``SystemRandom``.
    """

    scale = laplace_scale(sensitivity, epsilon)
    if scale == 0.0:
        return 0.0
    source = _random_source(rng)
    unit = float(source.random())
    if not 0.0 <= unit < 1.0:
        raise ValueError("rng.random() must return a value in [0, 1)")
    if unit == 0.0:
        unit = math.nextafter(0.0, 1.0)
    centered = unit - 0.5
    return -scale * math.copysign(1.0, centered) * math.log1p(-2.0 * abs(centered))


def gaussian_noise(
    sensitivity: float,
    epsilon: float,
    delta: float,
    *,
    rng: _RandomSource | None = None,
) -> float:
    """Draw calibrated Gaussian noise for one scalar query."""

    scale = gaussian_scale(sensitivity, epsilon, delta)
    if scale == 0.0:
        return 0.0
    source = _random_source(rng)
    gauss = getattr(source, "gauss", None)
    if not callable(gauss):
        gauss = getattr(source, "normalvariate", None)
    if not callable(gauss):
        raise TypeError("rng must provide gauss() or normalvariate()")
    return float(gauss(0.0, scale))


@dataclass(frozen=True)
class PrivacySpend:
    """Numeric metadata for one aggregate release.

    No aggregate value, row, identifier, noise draw, or caller-provided label
    is stored. ``sampler`` names the noise sampler: ``float_laplace`` and
    ``float_gaussian`` for the floating-point helpers, ``discrete_laplace``,
    ``discrete_gaussian`` and ``grid_discrete_laplace`` for the exact
    samplers, and ``unspecified`` for spends recorded directly.
    """

    sequence: int
    aggregate: AggregateKind
    mechanism: DPMechanism
    epsilon: float
    delta: float
    sensitivity: float
    noise_scale: float
    sampler: DPSampler = "unspecified"

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("sequence must be an integer")
        if self.sequence <= 0:
            raise ValueError("sequence must be greater than zero")
        object.__setattr__(self, "aggregate", _aggregate_kind(self.aggregate))
        object.__setattr__(self, "mechanism", _mechanism(self.mechanism))
        object.__setattr__(self, "epsilon", _epsilon(self.epsilon))
        object.__setattr__(self, "delta", _delta(self.delta))
        object.__setattr__(self, "sensitivity", _sensitivity(self.sensitivity))
        scale = _finite_float(self.noise_scale, field_name="noise_scale")
        if scale < 0.0:
            raise ValueError("noise_scale must be non-negative")
        object.__setattr__(self, "noise_scale", scale)
        object.__setattr__(self, "sampler", _sampler(self.sampler, self.mechanism))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, PHI-free spend record."""

        return {
            "sequence": self.sequence,
            "aggregate": self.aggregate,
            "mechanism": self.mechanism,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sensitivity": self.sensitivity,
            "noise_scale": self.noise_scale,
            "calibration_method": _calibration_method(self.mechanism, self.sampler),
            "sampler": self.sampler,
        }


@dataclass(frozen=True)
class PrivacyBudgetStatus:
    """Cumulative basic-composition status for one aggregate dataset."""

    total_epsilon: float
    total_delta: float
    spent_epsilon: float
    spent_delta: float
    remaining_epsilon: float
    remaining_delta: float
    query_count: int

    @property
    def epsilon(self) -> float:
        """Return cumulative epsilon spent so far."""

        return self.spent_epsilon

    @property
    def delta(self) -> float:
        """Return cumulative delta spent so far."""

        return self.spent_delta

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible status payload."""

        return {
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "remaining_epsilon": self.remaining_epsilon,
            "remaining_delta": self.remaining_delta,
            "query_count": self.query_count,
        }


@dataclass(frozen=True)
class BudgetDecision:
    """Non-mutating result of checking one proposed privacy spend."""

    allowed: bool
    requested_epsilon: float
    requested_delta: float
    projected_epsilon: float
    projected_delta: float
    total_epsilon: float
    total_delta: float
    remaining_epsilon: float
    remaining_delta: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible decision payload."""

        return {
            "allowed": self.allowed,
            "requested_epsilon": self.requested_epsilon,
            "requested_delta": self.requested_delta,
            "projected_epsilon": self.projected_epsilon,
            "projected_delta": self.projected_delta,
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "remaining_epsilon": self.remaining_epsilon,
            "remaining_delta": self.remaining_delta,
            "reason": self.reason,
        }


class PrivacyBudgetExceeded(ValueError):
    """Raised when an aggregate query would exceed its configured budget."""

    def __init__(self, decision: BudgetDecision) -> None:
        self.decision = decision
        super().__init__(
            "aggregate differential-privacy budget exceeded: "
            f"epsilon={decision.projected_epsilon:.6g}/"
            f"{decision.total_epsilon:.6g}, "
            f"delta={decision.projected_delta:.6g}/{decision.total_delta:.6g}"
        )


BudgetExceeded = PrivacyBudgetExceeded


@dataclass(init=False)
class PrivacyBudget:
    """Basic sequential accountant for queries against one dataset.

    Construct one instance per dataset and pass it to each release function.
    Only the configured budget and aggregate query metadata are retained; the
    class has no access to source rows and does not identify the dataset.

    ``epsilon``/``delta`` are the preferred constructor names. The
    ``total_*`` and ``max_*`` keyword aliases make the budget's ceiling
    explicit when it is used in a larger release pipeline.
    """

    epsilon: float
    delta: float
    _spends: list[PrivacySpend] = field(default_factory=list, repr=False)

    def __init__(
        self,
        epsilon: float | None = None,
        delta: float | None = None,
        *,
        total_epsilon: float | None = None,
        total_delta: float | None = None,
        max_epsilon: float | None = None,
        max_delta: float | None = None,
    ) -> None:
        epsilon_value = _one_budget_value(
            epsilon,
            total_epsilon,
            max_epsilon,
            field_name="epsilon",
        )
        delta_value = _one_budget_value(
            delta,
            total_delta,
            max_delta,
            field_name="delta",
            default=0.0,
        )
        self.epsilon = _epsilon(epsilon_value)
        self.delta = _delta(delta_value)
        self._spends = []

    @property
    def total_epsilon(self) -> float:
        """Return the configured epsilon ceiling."""

        return self.epsilon

    @property
    def total_delta(self) -> float:
        """Return the configured delta ceiling."""

        return self.delta

    @property
    def spends(self) -> tuple[PrivacySpend, ...]:
        """Return immutable spend records in query order."""

        return tuple(self._spends)

    @property
    def ledger(self) -> tuple[PrivacySpend, ...]:
        """Alias for :attr:`spends`."""

        return self.spends

    @property
    def spent_epsilon(self) -> float:
        """Return cumulative epsilon under basic composition."""

        return math.fsum(spend.epsilon for spend in self._spends)

    @property
    def spent_delta(self) -> float:
        """Return cumulative delta under basic composition."""

        return math.fsum(spend.delta for spend in self._spends)

    @property
    def epsilon_spent(self) -> float:
        """Alias for :attr:`spent_epsilon`."""

        return self.spent_epsilon

    @property
    def delta_spent(self) -> float:
        """Alias for :attr:`spent_delta`."""

        return self.spent_delta

    @property
    def remaining_epsilon(self) -> float:
        """Return epsilon headroom remaining."""

        return max(0.0, self.epsilon - self.spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        """Return delta headroom remaining."""

        return max(0.0, self.delta - self.spent_delta)

    @property
    def query_count(self) -> int:
        """Return the number of accepted releases."""

        return len(self._spends)

    def status(self) -> PrivacyBudgetStatus:
        """Return current cumulative spend and remaining headroom."""

        return PrivacyBudgetStatus(
            total_epsilon=self.epsilon,
            total_delta=self.delta,
            spent_epsilon=self.spent_epsilon,
            spent_delta=self.spent_delta,
            remaining_epsilon=self.remaining_epsilon,
            remaining_delta=self.remaining_delta,
            query_count=self.query_count,
        )

    def composition(self) -> PrivacyBudgetStatus:
        """Return the basic sequential composition status."""

        return self.status()

    def check(self, epsilon: float, delta: float = 0.0) -> BudgetDecision:
        """Check a proposed spend without changing the ledger."""

        requested_epsilon = _epsilon(epsilon)
        requested_delta = _delta(delta)
        projected_epsilon = self.spent_epsilon + requested_epsilon
        projected_delta = self.spent_delta + requested_delta
        epsilon_ok = _within_budget(projected_epsilon, self.epsilon)
        delta_ok = _within_budget(projected_delta, self.delta)
        allowed = epsilon_ok and delta_ok
        if allowed:
            reason = "within budget"
        elif not epsilon_ok and not delta_ok:
            reason = "epsilon and delta exceed budget"
        elif not epsilon_ok:
            reason = "epsilon exceeds budget"
        else:
            reason = "delta exceeds budget"
        return BudgetDecision(
            allowed=allowed,
            requested_epsilon=requested_epsilon,
            requested_delta=requested_delta,
            projected_epsilon=projected_epsilon,
            projected_delta=projected_delta,
            total_epsilon=self.epsilon,
            total_delta=self.delta,
            remaining_epsilon=max(0.0, self.epsilon - projected_epsilon),
            remaining_delta=max(0.0, self.delta - projected_delta),
            reason=reason,
        )

    def can_spend(self, epsilon: float, delta: float = 0.0) -> bool:
        """Return whether a proposed spend fits without recording it."""

        try:
            return self.check(epsilon, delta).allowed
        except (TypeError, ValueError):
            return False

    def spend(
        self,
        epsilon: float,
        delta: float = 0.0,
        *,
        aggregate: AggregateKind = "aggregate",
        mechanism: DPMechanism = "laplace",
        sensitivity: float = 1.0,
        noise_scale: float = 0.0,
        sampler: DPSampler = "unspecified",
    ) -> PrivacySpend:
        """Record one spend, or raise before mutating the ledger."""

        decision = self.check(epsilon, delta)
        if not decision.allowed:
            raise PrivacyBudgetExceeded(decision)
        spend = PrivacySpend(
            sequence=len(self._spends) + 1,
            aggregate=aggregate,
            mechanism=mechanism,
            epsilon=decision.requested_epsilon,
            delta=decision.requested_delta,
            sensitivity=sensitivity,
            noise_scale=noise_scale,
            sampler=sampler,
        )
        self._spends.append(spend)
        return spend

    def charge(self, *args: Any, **kwargs: Any) -> PrivacySpend:
        """Alias for :meth:`spend` used by release pipelines."""

        return self.spend(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, PHI-free accounting payload."""

        return {
            "total_epsilon": self.epsilon,
            "total_delta": self.delta,
            "status": self.status().to_dict(),
            "spends": [spend.to_dict() for spend in self._spends],
        }


def _one_budget_value(
    primary: float | None,
    first_alias: float | None,
    second_alias: float | None,
    *,
    field_name: str,
    default: float | None = None,
) -> float:
    values = [
        value for value in (primary, first_alias, second_alias) if value is not None
    ]
    if not values:
        if default is None:
            raise TypeError(f"{field_name} is required")
        return default
    normalized = [_finite_float(value, field_name=field_name) for value in values]
    if any(value != normalized[0] for value in normalized[1:]):
        raise TypeError(f"conflicting values supplied for {field_name}")
    return normalized[0]


def _within_budget(value: float, ceiling: float) -> bool:
    return value <= ceiling


def _validate_budget(budget: PrivacyBudget | None) -> None:
    if budget is not None and not isinstance(budget, PrivacyBudget):
        raise TypeError("budget must be a PrivacyBudget instance")


def _noise(
    value: float,
    *,
    sensitivity: float,
    epsilon: float,
    delta: float,
    mechanism: DPMechanism,
    rng: _RandomSource | None,
) -> tuple[float, float]:
    if mechanism == "laplace":
        scale = laplace_scale(sensitivity, epsilon)
        return value + laplace_noise(sensitivity, epsilon, rng=rng), scale
    scale = gaussian_scale(sensitivity, epsilon, delta)
    return value + gaussian_noise(sensitivity, epsilon, delta, rng=rng), scale


def _prepare_query(
    *,
    budget: PrivacyBudget | None,
    epsilon: float,
    delta: float,
) -> tuple[float, float]:
    _validate_budget(budget)
    epsilon_value = _epsilon(epsilon)
    delta_value = _delta(delta)
    if budget is not None:
        decision = budget.check(epsilon_value, delta_value)
        if not decision.allowed:
            raise PrivacyBudgetExceeded(decision)
    return epsilon_value, delta_value


def _commit_query(
    budget: PrivacyBudget | None,
    *,
    epsilon: float,
    delta: float,
    aggregate: AggregateKind,
    mechanism: DPMechanism,
    sensitivity: float,
    noise_scale: float,
    sampler: DPSampler,
) -> None:
    if budget is not None:
        budget.spend(
            epsilon,
            delta,
            aggregate=aggregate,
            mechanism=mechanism,
            sensitivity=sensitivity,
            noise_scale=noise_scale,
            sampler=sampler,
        )


def laplace_mechanism(
    value: float,
    *,
    epsilon: float,
    sensitivity: float = 1.0,
    delta: float = 0.0,
    budget: PrivacyBudget | None = None,
    rng: _RandomSource | None = None,
    aggregate: AggregateKind = "aggregate",
) -> float:
    """Release one scalar with Laplace noise and optionally charge a budget."""

    aggregate_value = _scalar(value)
    sensitivity_value = _sensitivity(sensitivity)
    aggregate_kind = _aggregate_kind(aggregate)
    epsilon_value, delta_value = _prepare_query(
        budget=budget,
        epsilon=epsilon,
        delta=delta,
    )
    released, scale = _noise(
        aggregate_value,
        sensitivity=sensitivity_value,
        epsilon=epsilon_value,
        delta=delta_value,
        mechanism="laplace",
        rng=rng,
    )
    _commit_query(
        budget,
        epsilon=epsilon_value,
        delta=delta_value,
        aggregate=aggregate_kind,
        mechanism="laplace",
        sensitivity=sensitivity_value,
        noise_scale=scale,
        sampler="float_laplace",
    )
    return released


def _gaussian_release(
    value: float,
    *,
    epsilon: float,
    sensitivity: float,
    delta: float,
    budget: PrivacyBudget | None,
    rng: _RandomSource | None,
    aggregate: AggregateKind,
) -> float:
    aggregate_value = _scalar(value)
    sensitivity_value = _sensitivity(sensitivity)
    aggregate_kind = _aggregate_kind(aggregate)
    epsilon_value, delta_value = _prepare_query(
        budget=budget,
        epsilon=epsilon,
        delta=delta,
    )
    released, scale = _noise(
        aggregate_value,
        sensitivity=sensitivity_value,
        epsilon=epsilon_value,
        delta=delta_value,
        mechanism="gaussian",
        rng=rng,
    )
    _commit_query(
        budget,
        epsilon=epsilon_value,
        delta=delta_value,
        aggregate=aggregate_kind,
        mechanism="gaussian",
        sensitivity=sensitivity_value,
        noise_scale=scale,
        sampler="float_gaussian",
    )
    return released


def gaussian_mechanism(
    value: float,
    *,
    epsilon: float,
    sensitivity: float = 1.0,
    delta: float,
    budget: PrivacyBudget | None = None,
    rng: _RandomSource | None = None,
    aggregate: AggregateKind = "aggregate",
) -> float:
    """Release one scalar with Gaussian noise and optionally charge a budget."""

    return _gaussian_release(
        value,
        epsilon=epsilon,
        sensitivity=sensitivity,
        delta=delta,
        budget=budget,
        rng=rng,
        aggregate=aggregate,
    )


def release_count(
    count: float,
    *,
    epsilon: float,
    delta: float = 0.0,
    mechanism: DPMechanism = "laplace",
    budget: PrivacyBudget | None = None,
    rng: _RandomSource | None = None,
) -> float:
    """Release a count with unit add/remove sensitivity."""

    selected = _mechanism(mechanism)
    if selected == "laplace":
        return laplace_mechanism(
            count,
            epsilon=epsilon,
            sensitivity=1.0,
            delta=delta,
            budget=budget,
            rng=rng,
            aggregate="count",
        )
    return _gaussian_release(
        count,
        epsilon=epsilon,
        sensitivity=1.0,
        delta=delta,
        budget=budget,
        rng=rng,
        aggregate="count",
    )


def release_sum(
    total: float,
    *,
    epsilon: float,
    sensitivity: float = 1.0,
    delta: float = 0.0,
    mechanism: DPMechanism = "laplace",
    budget: PrivacyBudget | None = None,
    rng: _RandomSource | None = None,
) -> float:
    """Release a bounded sum with caller-supplied contribution sensitivity."""

    selected = _mechanism(mechanism)
    if selected == "laplace":
        return laplace_mechanism(
            total,
            epsilon=epsilon,
            sensitivity=sensitivity,
            delta=delta,
            budget=budget,
            rng=rng,
            aggregate="sum",
        )
    return _gaussian_release(
        total,
        epsilon=epsilon,
        sensitivity=sensitivity,
        delta=delta,
        budget=budget,
        rng=rng,
        aggregate="sum",
    )


def release_mean(
    mean: float,
    *,
    epsilon: float,
    sensitivity: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    count: int | None = None,
    delta: float = 0.0,
    mechanism: DPMechanism = "laplace",
    budget: PrivacyBudget | None = None,
    rng: _RandomSource | None = None,
) -> float:
    """Release a mean using explicit or bounded ``(upper-lower) / count`` sensitivity."""

    if sensitivity is None:
        if lower is None or upper is None or count is None:
            raise ValueError(
                "mean sensitivity or lower, upper, and count must be provided"
            )
        lower_value = _scalar(lower, field_name="lower")
        upper_value = _scalar(upper, field_name="upper")
        if upper_value < lower_value:
            raise ValueError("upper must be greater than or equal to lower")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        sensitivity = (upper_value - lower_value) / count
    selected = _mechanism(mechanism)
    if selected == "laplace":
        return laplace_mechanism(
            mean,
            epsilon=epsilon,
            sensitivity=sensitivity,
            delta=delta,
            budget=budget,
            rng=rng,
            aggregate="mean",
        )
    return _gaussian_release(
        mean,
        epsilon=epsilon,
        sensitivity=sensitivity,
        delta=delta,
        budget=budget,
        rng=rng,
        aggregate="mean",
    )


def release_histogram(
    histogram: Mapping[_HistogramKey, float] | Sequence[float],
    *,
    epsilon: float,
    sensitivity: float = 1.0,
    delta: float = 0.0,
    mechanism: DPMechanism = "laplace",
    budget: PrivacyBudget | None = None,
    rng: _RandomSource | None = None,
) -> dict[_HistogramKey, float] | list[float]:
    """Release all bins as one vector query.

    For a histogram built from add/remove records, the default vector
    sensitivity is one. The full histogram consumes one budget query rather
    than one budget charge per bin.
    """

    selected = _mechanism(mechanism)
    sensitivity_value = _sensitivity(sensitivity)
    epsilon_value, delta_value = _prepare_query(
        budget=budget,
        epsilon=epsilon,
        delta=delta,
    )
    source = _random_source(rng)
    if isinstance(histogram, Mapping):
        items = list(histogram.items())
        values = [_scalar(value, field_name="histogram bin") for _, value in items]
        output_kind = "mapping"
    elif isinstance(histogram, Sequence) and not isinstance(
        histogram, (str, bytes, bytearray)
    ):
        items = []
        values = [_scalar(value, field_name="histogram bin") for value in histogram]
        output_kind = "sequence"
    else:
        raise TypeError("histogram must be a mapping or a numeric sequence")

    scale = (
        laplace_scale(sensitivity_value, epsilon_value)
        if selected == "laplace"
        else gaussian_scale(sensitivity_value, epsilon_value, delta_value)
    )
    if selected == "laplace":
        released_values = [
            value + laplace_noise(sensitivity_value, epsilon_value, rng=source)
            for value in values
        ]
    else:
        released_values = [
            value
            + gaussian_noise(
                sensitivity_value,
                epsilon_value,
                delta_value,
                rng=source,
            )
            for value in values
        ]
    _commit_query(
        budget,
        epsilon=epsilon_value,
        delta=delta_value,
        aggregate="histogram",
        mechanism=selected,
        sensitivity=sensitivity_value,
        noise_scale=scale,
        sampler="float_laplace" if selected == "laplace" else "float_gaussian",
    )
    if output_kind == "mapping":
        return {key: noisy for (key, _), noisy in zip(items, released_values)}
    return released_values


def release_aggregate(
    aggregate: float | Mapping[_HistogramKey, float] | Sequence[float],
    *,
    kind: AggregateKind,
    epsilon: float,
    sensitivity: float = 1.0,
    delta: float = 0.0,
    mechanism: DPMechanism = "laplace",
    budget: PrivacyBudget | None = None,
    rng: _RandomSource | None = None,
) -> float | dict[_HistogramKey, float] | list[float]:
    """Dispatch a scalar or histogram release by aggregate kind."""

    selected_kind = _aggregate_kind(kind)
    if selected_kind == "count":
        return release_count(
            _scalar(aggregate, field_name="count"),
            epsilon=epsilon,
            delta=delta,
            mechanism=mechanism,
            budget=budget,
            rng=rng,
        )
    if selected_kind == "sum":
        return release_sum(
            _scalar(aggregate, field_name="sum"),
            epsilon=epsilon,
            sensitivity=sensitivity,
            delta=delta,
            mechanism=mechanism,
            budget=budget,
            rng=rng,
        )
    if selected_kind == "mean":
        return release_mean(
            _scalar(aggregate, field_name="mean"),
            epsilon=epsilon,
            sensitivity=sensitivity,
            delta=delta,
            mechanism=mechanism,
            budget=budget,
            rng=rng,
        )
    if selected_kind != "histogram":
        raise ValueError("aggregate kind 'aggregate' requires a mechanism helper")
    return release_histogram(
        aggregate,
        epsilon=epsilon,
        sensitivity=sensitivity,
        delta=delta,
        mechanism=mechanism,
        budget=budget,
        rng=rng,
    )


def _integer_values(
    histogram: Any,
) -> tuple[list[Any], list[int], bool]:
    if isinstance(histogram, Mapping):
        items = list(histogram.items())
        values = [_integer(value, field_name="histogram bin") for _, value in items]
        return [key for key, _ in items], values, True
    if isinstance(histogram, Sequence) and not isinstance(
        histogram, (str, bytes, bytearray)
    ):
        values = [_integer(value, field_name="histogram bin") for value in histogram]
        return [], values, False
    raise TypeError("histogram must be a mapping or an integer sequence")


def _discrete_release(
    values: list[int],
    *,
    aggregate: AggregateKind,
    epsilon: float,
    sensitivity: float,
    delta: float,
    mechanism: DPMechanism,
    budget: PrivacyBudget | None,
    rng: DPRandomSource | None,
) -> list[int]:
    selected = _mechanism(mechanism)
    sensitivity_value = _positive_sensitivity(sensitivity)
    source = _dp_source(rng)
    epsilon_value, delta_value = _prepare_query(
        budget=budget,
        epsilon=epsilon,
        delta=delta,
    )
    if selected == "laplace":
        scale = Fraction(sensitivity_value) / Fraction(epsilon_value)
        released = [value + _discrete_laplace(scale, source) for value in values]
        noise_scale = float(scale)
        sampler: DPSampler = "discrete_laplace"
    else:
        sigma_squared = discrete_gaussian_sigma_squared(
            sensitivity_value,
            epsilon_value,
            delta_value,
        )
        released = [
            value + _discrete_gaussian(sigma_squared, source) for value in values
        ]
        noise_scale = math.sqrt(sigma_squared)
        sampler = "discrete_gaussian"
    _commit_query(
        budget,
        epsilon=epsilon_value,
        delta=delta_value,
        aggregate=aggregate,
        mechanism=selected,
        sensitivity=sensitivity_value,
        noise_scale=noise_scale,
        sampler=sampler,
    )
    return released


def release_discrete_count(
    count: int,
    *,
    epsilon: float,
    sensitivity: float = 1.0,
    delta: float = 0.0,
    mechanism: DPMechanism = "laplace",
    budget: PrivacyBudget | None = None,
    rng: DPRandomSource | None = None,
) -> int:
    """Release an integer count with exact discrete noise.

    ``mechanism="laplace"`` adds discrete Laplace noise with scale
    ``sensitivity / epsilon``, which is pure ``epsilon``-DP for integer
    queries whose L1 sensitivity is at most ``sensitivity``.
    ``mechanism="gaussian"`` adds discrete Gaussian noise calibrated by
    :func:`discrete_gaussian_sigma_squared` and requires ``delta > 0``.
    The budget is checked before any noise is drawn and charged once after
    sampling succeeds. The spend record names the sampler but never the
    count, the noise draw, or the released value.

    Args:
        count: Integer count to release.
        epsilon: Positive finite epsilon charged for this query.
        sensitivity: Positive sensitivity bound; one for add/remove counts.
        delta: Delta charged for this query.
        mechanism: ``"laplace"`` or ``"gaussian"``.
        budget: Optional budget charged for this query.
        rng: Optional :class:`DPRandomSource`; system randomness when omitted.

    Returns:
        The noisy count as an ``int``. It may be negative; clamping is left
        to the caller as post-processing.

    Raises:
        TypeError: If ``count`` is not an integer or ``rng`` is not a
            :class:`DPRandomSource`.
        ValueError: If a privacy parameter is invalid.
        PrivacyBudgetExceeded: If the query would exceed ``budget``.
        DiscreteSamplerError: If a sampler reaches its rejection bound.
    """

    values = [_integer(count, field_name="count")]
    return _discrete_release(
        values,
        aggregate="count",
        epsilon=epsilon,
        sensitivity=sensitivity,
        delta=delta,
        mechanism=mechanism,
        budget=budget,
        rng=rng,
    )[0]


def release_discrete_histogram(
    histogram: Mapping[_HistogramKey, int] | Sequence[int],
    *,
    epsilon: float,
    sensitivity: float = 1.0,
    delta: float = 0.0,
    mechanism: DPMechanism = "laplace",
    budget: PrivacyBudget | None = None,
    rng: DPRandomSource | None = None,
) -> dict[_HistogramKey, int] | list[int]:
    """Release integer histogram bins as one exact discrete vector query.

    Each bin receives independent discrete noise. ``sensitivity`` is the L1
    sensitivity for ``"laplace"`` and the L2 sensitivity for ``"gaussian"``;
    the default of one matches add/remove histograms. The full histogram
    consumes one budget charge.

    Args:
        histogram: Mapping or sequence of integer bin counts.
        epsilon: Positive finite epsilon charged for this query.
        sensitivity: Positive vector sensitivity bound.
        delta: Delta charged for this query.
        mechanism: ``"laplace"`` or ``"gaussian"``.
        budget: Optional budget charged for this query.
        rng: Optional :class:`DPRandomSource`; system randomness when omitted.

    Returns:
        Noisy integer bins with the same keys or order as the input.

    Raises:
        TypeError: If a bin is not an integer or ``rng`` is not a
            :class:`DPRandomSource`.
        ValueError: If a privacy parameter is invalid.
        PrivacyBudgetExceeded: If the query would exceed ``budget``.
        DiscreteSamplerError: If a sampler reaches its rejection bound.
    """

    keys, values, is_mapping = _integer_values(histogram)
    released = _discrete_release(
        values,
        aggregate="histogram",
        epsilon=epsilon,
        sensitivity=sensitivity,
        delta=delta,
        mechanism=mechanism,
        budget=budget,
        rng=rng,
    )
    if is_mapping:
        return dict(zip(keys, released))
    return released


def _snapping_grid(value: Any) -> Fraction:
    grid = _positive_rational(value, field_name="grid")
    numerator, denominator = grid.numerator, grid.denominator
    if not (
        (numerator == 1 and denominator & (denominator - 1) == 0)
        or (denominator == 1 and numerator & (numerator - 1) == 0)
    ):
        raise ValueError("grid must be a power of two")
    if grid < Fraction(2) ** -1022 or grid > Fraction(2) ** 1023:
        raise ValueError("grid must be a normal floating-point power of two")
    return grid


def _snapping_bound(value: Any, grid: Fraction) -> Fraction:
    bound = _positive_rational(value, field_name="bound")
    steps = bound / grid
    if steps.denominator != 1:
        raise ValueError("bound must be an integer multiple of grid")
    if steps > _MAX_GRID_STEPS:
        raise ValueError("bound must be at most 2**52 grid steps")
    return bound


def _float_at_least(value: Fraction) -> float:
    result = float(value)
    if Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise ValueError("adjusted epsilon is not representable")
    return result


def _snapped_epsilon_exact(
    epsilon: float,
    sensitivity: float,
    grid: Fraction,
) -> Fraction:
    exact_sensitivity = Fraction(sensitivity)
    steps = math.ceil(exact_sensitivity / grid)
    return steps * grid * Fraction(epsilon) / exact_sensitivity


def snapped_epsilon(epsilon: float, *, sensitivity: float, grid: float) -> float:
    """Return the epsilon charged by :func:`release_snapped`.

    Rounding the clamped query to the grid can move neighboring inputs by up
    to ``ceil(sensitivity / grid)`` grid steps, while the noise scale stays
    ``sensitivity / epsilon``. The exact privacy loss is therefore
    ``ceil(sensitivity / grid) * grid * epsilon / sensitivity``, which equals
    ``epsilon`` when ``sensitivity / grid`` is an integer and is at most
    ``epsilon * (1 + grid / sensitivity)`` otherwise. The returned float is
    the smallest float not below that exact value.

    Args:
        epsilon: Requested positive finite epsilon for the noise scale.
        sensitivity: Positive L1 sensitivity of the real-valued query.
        grid: Positive power-of-two output grid.

    Returns:
        The adjusted epsilon as a float rounded up.

    Raises:
        TypeError: If a parameter is not a real number.
        ValueError: If a parameter is out of range.
    """

    epsilon_value = _epsilon(epsilon)
    sensitivity_value = _positive_sensitivity(sensitivity)
    grid_value = _snapping_grid(grid)
    return _float_at_least(
        _snapped_epsilon_exact(epsilon_value, sensitivity_value, grid_value)
    )


def release_snapped(
    value: float,
    *,
    epsilon: float,
    sensitivity: float,
    bound: float,
    grid: float,
    delta: float = 0.0,
    budget: PrivacyBudget | None = None,
    rng: DPRandomSource | None = None,
    aggregate: AggregateKind = "aggregate",
) -> float:
    """Release a real value on a declared power-of-two grid with exact noise.

    The value is clamped to ``[-bound, bound]``, rounded half up to the
    nearest multiple of ``grid`` with exact rational arithmetic, perturbed by
    exact discrete Laplace noise in grid units with scale
    ``sensitivity / (epsilon * grid)``, and clamped again to
    ``[-bound, bound]``. Every output is an integer multiple of ``grid`` and
    exactly representable as a float. The budget is charged the adjusted
    epsilon from :func:`snapped_epsilon`; the query is pure DP at that value.

    ``bound`` must be an integer multiple of ``grid`` and at most ``2**52``
    grid steps, and ``grid`` must be a normal floating-point power of two.
    Like Mironov's snapping mechanism (CCS 2012) the output lies on a
    power-of-two grid inside a clamp, but no floating-point Laplace draw or
    logarithm is involved, so no correctly rounded math library is assumed.

    Args:
        value: Finite real aggregate to release.
        epsilon: Requested positive finite epsilon for the noise scale.
        sensitivity: Positive L1 sensitivity of the real-valued query.
        bound: Positive clamp bound, an integer multiple of ``grid``.
        grid: Positive power-of-two output grid.
        delta: Delta charged for this query.
        budget: Optional budget charged the adjusted epsilon.
        rng: Optional :class:`DPRandomSource`; system randomness when omitted.
        aggregate: Aggregate kind recorded in the spend.

    Returns:
        The noisy value, an exact multiple of ``grid`` in ``[-bound, bound]``.

    Raises:
        TypeError: If a parameter has an unsupported type or ``rng`` is not a
            :class:`DPRandomSource`.
        ValueError: If a parameter is out of range.
        PrivacyBudgetExceeded: If the adjusted epsilon would exceed ``budget``.
        DiscreteSamplerError: If the sampler reaches its rejection bound.
    """

    aggregate_value = _scalar(value)
    sensitivity_value = _positive_sensitivity(sensitivity)
    aggregate_kind = _aggregate_kind(aggregate)
    grid_value = _snapping_grid(grid)
    bound_value = _snapping_bound(bound, grid_value)
    source = _dp_source(rng)
    requested_epsilon = _epsilon(epsilon)
    charged_epsilon = _float_at_least(
        _snapped_epsilon_exact(requested_epsilon, sensitivity_value, grid_value)
    )
    epsilon_value, delta_value = _prepare_query(
        budget=budget,
        epsilon=charged_epsilon,
        delta=delta,
    )
    scale = Fraction(sensitivity_value) / (Fraction(requested_epsilon) * grid_value)
    limit = int(bound_value / grid_value)
    clamped = min(max(Fraction(aggregate_value), -bound_value), bound_value)
    center = math.floor(clamped / grid_value + Fraction(1, 2))
    steps = min(max(center + _discrete_laplace(scale, source), -limit), limit)
    released = float(steps * grid_value)
    _commit_query(
        budget,
        epsilon=epsilon_value,
        delta=delta_value,
        aggregate=aggregate_kind,
        mechanism="laplace",
        sensitivity=sensitivity_value,
        noise_scale=laplace_scale(sensitivity_value, requested_epsilon),
        sampler="grid_discrete_laplace",
    )
    return released


@dataclass(frozen=True)
class UtilityPoint:
    """Expected error at one epsilon value for a chosen mechanism."""

    epsilon: float
    delta: float
    mechanism: DPMechanism
    sensitivity: float
    noise_scale: float
    expected_absolute_error: float
    root_mean_square_error: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible utility point."""

        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "mechanism": self.mechanism,
            "sensitivity": self.sensitivity,
            "noise_scale": self.noise_scale,
            "expected_absolute_error": self.expected_absolute_error,
            "root_mean_square_error": self.root_mean_square_error,
        }


@dataclass(frozen=True)
class UtilityReport:
    """PHI-free utility-versus-epsilon report."""

    mechanism: DPMechanism
    delta: float
    sensitivity: float
    points: tuple[UtilityPoint, ...]

    @property
    def values(self) -> tuple[UtilityPoint, ...]:
        """Alias for report points."""

        return self.points

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report without source aggregates."""

        return {
            "mechanism": self.mechanism,
            "delta": self.delta,
            "sensitivity": self.sensitivity,
            "points": [point.to_dict() for point in self.points],
        }


def utility_report(
    epsilons: Sequence[float],
    *,
    sensitivity: float = 1.0,
    mechanism: DPMechanism = "laplace",
    delta: float = 0.0,
) -> UtilityReport:
    """Build expected-error points for candidate privacy budgets.

    Laplace points use ``E|noise| = scale`` and ``RMSE = sqrt(2) * scale``.
    Gaussian points use ``E|noise| = scale * sqrt(2 / pi)`` and ``RMSE =
    scale``. The report contains no released value or source record.
    """

    if isinstance(epsilons, (str, bytes, bytearray)):
        raise TypeError("epsilons must be a numeric sequence")
    selected = _mechanism(mechanism)
    sensitivity_value = _sensitivity(sensitivity)
    delta_value = _delta(delta)
    points: list[UtilityPoint] = []
    for candidate in epsilons:
        epsilon_value = _epsilon(candidate)
        if selected == "laplace":
            scale = laplace_scale(sensitivity_value, epsilon_value)
            expected_absolute_error = scale
            root_mean_square_error = math.sqrt(2.0) * scale
        else:
            scale = gaussian_scale(
                sensitivity_value,
                epsilon_value,
                delta_value,
            )
            expected_absolute_error = scale * math.sqrt(2.0 / math.pi)
            root_mean_square_error = scale
        points.append(
            UtilityPoint(
                epsilon=epsilon_value,
                delta=delta_value,
                mechanism=selected,
                sensitivity=sensitivity_value,
                noise_scale=scale,
                expected_absolute_error=expected_absolute_error,
                root_mean_square_error=root_mean_square_error,
            )
        )
    return UtilityReport(
        mechanism=selected,
        delta=delta_value,
        sensitivity=sensitivity_value,
        points=tuple(points),
    )


utility_vs_epsilon = utility_report


class DifferentialPrivacy:
    """Convenience facade bundling a budget and default release settings."""

    def __init__(
        self,
        *,
        epsilon: float | None = None,
        delta: float = 0.0,
        budget: PrivacyBudget | None = None,
        rng: _RandomSource | None = None,
    ) -> None:
        if budget is None:
            if epsilon is None:
                raise TypeError("epsilon is required when budget is not supplied")
            budget = PrivacyBudget(epsilon=epsilon, delta=delta)
        _validate_budget(budget)
        self.budget = budget
        self.rng = rng

    def count(
        self,
        count: float,
        *,
        epsilon: float | None = None,
        delta: float = 0.0,
        mechanism: DPMechanism = "laplace",
    ) -> float:
        """Release a count against this facade's shared budget."""

        selected_epsilon = self.budget.remaining_epsilon if epsilon is None else epsilon
        return release_count(
            count,
            epsilon=selected_epsilon,
            delta=delta,
            mechanism=mechanism,
            budget=self.budget,
            rng=self.rng,
        )

    def sum(
        self,
        total: float,
        *,
        sensitivity: float = 1.0,
        epsilon: float | None = None,
        delta: float = 0.0,
        mechanism: DPMechanism = "laplace",
    ) -> float:
        """Release a bounded sum against this facade's shared budget."""

        selected_epsilon = self.budget.remaining_epsilon if epsilon is None else epsilon
        return release_sum(
            total,
            epsilon=selected_epsilon,
            sensitivity=sensitivity,
            delta=delta,
            mechanism=mechanism,
            budget=self.budget,
            rng=self.rng,
        )

    def histogram(
        self,
        histogram: Mapping[_HistogramKey, float] | Sequence[float],
        *,
        sensitivity: float = 1.0,
        epsilon: float | None = None,
        delta: float = 0.0,
        mechanism: DPMechanism = "laplace",
    ) -> dict[_HistogramKey, float] | list[float]:
        """Release a histogram vector against this facade's shared budget."""

        selected_epsilon = self.budget.remaining_epsilon if epsilon is None else epsilon
        return release_histogram(
            histogram,
            epsilon=selected_epsilon,
            sensitivity=sensitivity,
            delta=delta,
            mechanism=mechanism,
            budget=self.budget,
            rng=self.rng,
        )
