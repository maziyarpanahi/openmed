"""Differential-privacy mechanisms for already-aggregated values.

The functions in this module accept aggregate values, not source records. They
apply calibrated Laplace or Gaussian noise locally and keep an optional,
numeric-only privacy ledger for one dataset. The ledger uses basic sequential
composition: query epsilon and delta values are added across releases.

The Gaussian scale uses the common conservative bound
``sensitivity * sqrt(2 * log(1.25 / delta)) / epsilon``. The bound is intended
for the usual approximate-DP setting and is exposed so callers can document
the exact calibration used by their release. Sensitivity is the caller's
responsibility: it must match the adjacency relation and any clipping or
bounded contribution assumptions for the aggregate.

This module does not accept or persist raw rows, identifiers, or query values
in its accounting records. A noisy aggregate is not a clinical decision,
compliance certification, or guarantee of zero disclosure risk.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

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
    "gaussian_mechanism",
    "gaussian_noise",
    "gaussian_scale",
    "gaussian_stddev",
    "laplace_mechanism",
    "laplace_noise",
    "laplace_scale",
    "release_aggregate",
    "release_count",
    "release_histogram",
    "release_mean",
    "release_sum",
    "utility_report",
    "utility_vs_epsilon",
]

DPMechanism = Literal["laplace", "gaussian"]
AggregateKind = Literal["count", "sum", "mean", "histogram", "aggregate"]

_HistogramKey = TypeVar("_HistogramKey")


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
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field_name} must be a real number") from exc
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


def gaussian_scale(sensitivity: float, epsilon: float, delta: float) -> float:
    """Return the conservative Gaussian standard deviation for ``(epsilon, delta)``.

    ``delta`` must be positive because a Gaussian mechanism cannot use this
    calibration at ``delta=0``.
    """

    sensitivity_value = _sensitivity(sensitivity)
    epsilon_value = _epsilon(epsilon)
    delta_value = _delta(delta)
    if delta_value <= 0.0:
        raise ValueError("Gaussian mechanism requires delta greater than zero")
    return (
        sensitivity_value
        * math.sqrt(2.0 * math.log(1.25 / delta_value))
        / epsilon_value
    )


gaussian_stddev = gaussian_scale


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

    No aggregate value, row, identifier, or caller-provided label is stored.
    """

    sequence: int
    aggregate: AggregateKind
    mechanism: DPMechanism
    epsilon: float
    delta: float
    sensitivity: float
    noise_scale: float

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
) -> None:
    if budget is not None:
        budget.spend(
            epsilon,
            delta,
            aggregate=aggregate,
            mechanism=mechanism,
            sensitivity=sensitivity,
            noise_scale=noise_scale,
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
