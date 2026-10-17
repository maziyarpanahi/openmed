"""Offline differential-privacy primitives for aggregate releases.

This module is deliberately separate from row-level de-identification. The
ledger accounts only for named aggregate queries, and the Laplace mechanism
rejects row-shaped inputs. A budget entry is never evidence that individual
rows were anonymized.

:func:`release_aggregate` keeps its floating-point Laplace sampler for
compatibility. :func:`release_discrete_aggregate` releases integer counts with
the exact discrete Laplace sampler from :mod:`openmed.risk.discrete_sampling`
and should be preferred for counts.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral
from typing import Any

from openmed.core.audit import stable_hash

from .discrete_sampling import (
    DP_SAMPLERS,
    DPRandomSource,
    _discrete_laplace,
    _dp_source,
)

__all__ = [
    "AggregateDPBudgetLedger",
    "AggregateDPRelease",
    "DPBudgetComposition",
    "DPBudgetExhausted",
    "DPBudgetLedger",
    "DPBudgetSpend",
    "DPAggregateBudgetExceeded",
    "laplace_aggregate",
    "release_aggregate",
    "release_discrete_aggregate",
]


class DPAggregateBudgetExceeded(ValueError):
    """Raised when an aggregate query would exhaust the declared budget."""


DPBudgetExhausted = DPAggregateBudgetExceeded


@dataclass(frozen=True)
class DPBudgetSpend:
    """One aggregate-query privacy spend.

    ``sampler`` names the noise sampler when a release helper recorded the
    spend. Spends recorded directly through
    :meth:`AggregateDPBudgetLedger.spend` keep ``None`` and serialize without
    a ``sampler`` key, so their ledger digests are unchanged.
    """

    sequence: int
    label: str
    mechanism: str
    epsilon: float
    delta: float
    scope: str = "aggregate"
    sampler: str | None = None

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("sequence must be an integer >= 1")
        _validate_label(self.label)
        _validate_label(self.mechanism, field_name="mechanism")
        _positive_finite(self.epsilon, field_name="epsilon")
        _delta(self.delta)
        if self.scope != "aggregate":
            raise ValueError("the differential-privacy ledger is aggregate-only")
        if self.sampler is not None and (
            not isinstance(self.sampler, str) or self.sampler not in DP_SAMPLERS
        ):
            raise ValueError("sampler is not a supported noise sampler")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe accounting metadata."""

        record: dict[str, Any] = {
            "sequence": self.sequence,
            "label": self.label,
            "mechanism": self.mechanism,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "scope": self.scope,
        }
        if self.sampler is not None:
            record["sampler"] = self.sampler
        return record


@dataclass(frozen=True)
class DPBudgetComposition:
    """Deterministic basic composition totals for one aggregate ledger."""

    epsilon: float
    delta: float
    max_epsilon: float
    max_delta: float
    remaining_epsilon: float
    remaining_delta: float

    @property
    def exhausted(self) -> bool:
        """Return whether either configured budget has no remaining slack."""

        return self.remaining_epsilon <= 0.0 or self.remaining_delta < 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic composition evidence."""

        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "max_epsilon": self.max_epsilon,
            "max_delta": self.max_delta,
            "remaining_epsilon": self.remaining_epsilon,
            "remaining_delta": self.remaining_delta,
            "exhausted": self.exhausted,
        }


@dataclass
class AggregateDPBudgetLedger:
    """Mutable, deterministic epsilon/delta ledger for aggregate queries."""

    max_epsilon: float
    max_delta: float
    _spends: list[DPBudgetSpend] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.max_epsilon = _positive_finite(
            self.max_epsilon,
            field_name="max_epsilon",
        )
        self.max_delta = _delta(self.max_delta)

    @property
    def spends(self) -> tuple[DPBudgetSpend, ...]:
        """Return an immutable view of committed aggregate spends."""

        return tuple(self._spends)

    @property
    def spent_epsilon(self) -> float:
        """Return the deterministically composed epsilon spend."""

        return math.fsum(spend.epsilon for spend in self._spends)

    @property
    def spent_delta(self) -> float:
        """Return the deterministically composed delta spend."""

        return math.fsum(spend.delta for spend in self._spends)

    @property
    def remaining_epsilon(self) -> float:
        """Return remaining epsilon budget."""

        return self.max_epsilon - self.spent_epsilon

    @property
    def remaining_delta(self) -> float:
        """Return remaining delta budget."""

        return self.max_delta - self.spent_delta

    def spend(
        self,
        epsilon: float,
        delta: float = 0.0,
        *,
        label: str = "aggregate_query",
        mechanism: str = "laplace",
        scope: str = "aggregate",
        sampler: str | None = None,
    ) -> DPBudgetSpend:
        """Commit one aggregate spend or raise without mutating the ledger."""

        if scope != "aggregate":
            raise ValueError("row-level differential privacy is not supported")
        spend = DPBudgetSpend(
            sequence=len(self._spends) + 1,
            label=label,
            mechanism=mechanism,
            epsilon=epsilon,
            delta=delta,
            scope=scope,
            sampler=sampler,
        )
        projected_epsilon = math.fsum(
            [self.spent_epsilon, spend.epsilon],
        )
        projected_delta = math.fsum([self.spent_delta, spend.delta])
        if projected_epsilon > self.max_epsilon or projected_delta > self.max_delta:
            raise DPAggregateBudgetExceeded(
                "aggregate differential-privacy budget exhausted"
            )
        self._spends.append(spend)
        return spend

    def compose(self) -> DPBudgetComposition:
        """Return deterministic basic composition for committed spends."""

        epsilon = self.spent_epsilon
        delta = self.spent_delta
        return DPBudgetComposition(
            epsilon=epsilon,
            delta=delta,
            max_epsilon=self.max_epsilon,
            max_delta=self.max_delta,
            remaining_epsilon=self.max_epsilon - epsilon,
            remaining_delta=self.max_delta - delta,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return an aggregate-only budget ledger."""

        return {
            "schema_version": 1,
            "scope": "aggregate_only",
            "row_level_anonymization": False,
            "max_epsilon": self.max_epsilon,
            "max_delta": self.max_delta,
            "spends": [spend.to_dict() for spend in self._spends],
            "composition": self.compose().to_dict(),
            "ledger_digest": stable_hash(
                {
                    "max_epsilon": self.max_epsilon,
                    "max_delta": self.max_delta,
                    "spends": [spend.to_dict() for spend in self._spends],
                }
            ),
        }


DPBudgetLedger = AggregateDPBudgetLedger


@dataclass(frozen=True)
class AggregateDPRelease:
    """One aggregate-only Laplace release and its accounting evidence."""

    value: float | int | dict[str, float] | dict[str, int]
    spend: DPBudgetSpend
    composition: DPBudgetComposition
    mechanism: str = "laplace"
    scope: str = "aggregate_only"
    test_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the released aggregate and safe accounting metadata."""

        return {
            "schema_version": 2,
            "scope": self.scope,
            "row_level_anonymization": False,
            "mechanism": self.mechanism,
            "value": self.value,
            "spend": self.spend.to_dict(),
            "composition": self.composition.to_dict(),
            "test_only": self.test_only,
            "limitations": [
                "This mechanism releases aggregates only.",
                "It does not anonymize or authorize row-level release.",
            ],
        }


def release_aggregate(
    value: float | Mapping[str, float],
    *,
    ledger: AggregateDPBudgetLedger,
    epsilon: float,
    sensitivity: float = 1.0,
    delta: float = 0.0,
    label: str = "aggregate_query",
    seed: int | str | bytes | None = None,
) -> AggregateDPRelease:
    """Release a scalar or named mapping of aggregates with Laplace noise.

    The input is deliberately restricted to a scalar or a mapping of named
    numeric aggregates. Sequences of rows, row mappings, and nested values are
    rejected so callers cannot mistake this operation for row-level release.
    A supplied seed makes synthetic tests reproducible and marks the result
    ``test_only``. Seeded output provides no production privacy guarantee.
    Neither the noise draw nor seed information is retained in the result.
    """

    if not isinstance(ledger, AggregateDPBudgetLedger):
        raise TypeError("ledger must be an AggregateDPBudgetLedger")
    scale = _positive_finite(sensitivity, field_name="sensitivity") / _positive_finite(
        epsilon,
        field_name="epsilon",
    )
    if delta != 0.0:
        raise ValueError("the Laplace mechanism uses delta=0")
    normalized = _aggregate_value(value)
    _validate_label(label)
    rng = _random_source(seed, label)
    spend = ledger.spend(
        epsilon,
        delta,
        label=label,
        mechanism="laplace",
        sampler="float_laplace",
    )
    if isinstance(normalized, dict):
        noise = {name: _laplace_noise(rng, scale) for name in normalized}
        released = {name: normalized[name] + noise[name] for name in normalized}
    else:
        noise = _laplace_noise(rng, scale)
        released = normalized + noise
    return AggregateDPRelease(
        value=released,
        spend=spend,
        composition=ledger.compose(),
        test_only=seed is not None,
    )


def laplace_aggregate(
    value: float | Mapping[str, float],
    *,
    ledger: AggregateDPBudgetLedger,
    epsilon: float,
    sensitivity: float = 1.0,
    delta: float = 0.0,
    label: str = "aggregate_query",
    seed: int | str | bytes | None = None,
) -> AggregateDPRelease:
    """Alias for :func:`release_aggregate`."""

    return release_aggregate(
        value,
        ledger=ledger,
        epsilon=epsilon,
        sensitivity=sensitivity,
        delta=delta,
        label=label,
        seed=seed,
    )


def release_discrete_aggregate(
    value: int | Mapping[str, int],
    *,
    ledger: AggregateDPBudgetLedger,
    epsilon: float,
    sensitivity: float = 1.0,
    delta: float = 0.0,
    label: str = "aggregate_query",
    rng: DPRandomSource | None = None,
) -> AggregateDPRelease:
    """Release an integer count or named integer counts with exact noise.

    Each count receives independent discrete Laplace noise with scale
    ``sensitivity / epsilon``, drawn with integer arithmetic only. The query
    is pure ``epsilon``-DP when ``sensitivity`` bounds the L1 change of the
    integer counts. The spend records ``sampler="discrete_laplace"`` and is
    committed before noise is drawn, as in :func:`release_aggregate`. Pass
    ``DPRandomSource.for_testing(seed)`` for reproducible synthetic tests;
    such results are marked ``test_only``.

    Args:
        value: Integer count or non-empty mapping of names to integer counts.
        ledger: Aggregate budget ledger charged for this query.
        epsilon: Positive finite epsilon.
        sensitivity: Positive L1 sensitivity bound.
        delta: Must be zero for the discrete Laplace mechanism.
        label: Short accounting label.
        rng: Optional :class:`DPRandomSource`; system randomness when omitted.

    Returns:
        The release with integer values and accounting evidence.

    Raises:
        TypeError: If an input has an unsupported type or shape.
        ValueError: If a parameter is out of range.
        DPAggregateBudgetExceeded: If the query would exceed the budget.
    """

    if not isinstance(ledger, AggregateDPBudgetLedger):
        raise TypeError("ledger must be an AggregateDPBudgetLedger")
    scale = Fraction(_positive_finite(sensitivity, field_name="sensitivity")) / (
        Fraction(_positive_finite(epsilon, field_name="epsilon"))
    )
    if delta != 0.0:
        raise ValueError("the Laplace mechanism uses delta=0")
    normalized = _integer_aggregate_value(value)
    _validate_label(label)
    source = _dp_source(rng)
    spend = ledger.spend(
        epsilon,
        delta,
        label=label,
        mechanism="laplace",
        sampler="discrete_laplace",
    )
    released: int | dict[str, int]
    if isinstance(normalized, dict):
        released = {
            name: count + _discrete_laplace(scale, source)
            for name, count in normalized.items()
        }
    else:
        released = normalized + _discrete_laplace(scale, source)
    return AggregateDPRelease(
        value=released,
        spend=spend,
        composition=ledger.compose(),
        test_only=source.test_mode,
    )


def _integer_aggregate_value(value: Any) -> int | dict[str, int]:
    if isinstance(value, Mapping):
        if not value:
            raise ValueError("aggregate mapping must not be empty")
        result: dict[str, int] = {}
        for name, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if not isinstance(name, str) or not name:
                raise ValueError("aggregate names must be non-empty strings")
            result[name] = _integer_count(item)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        raise TypeError("row-level release is not supported by aggregate DP")
    return _integer_count(value)


def _integer_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("aggregate counts must be integers")
    return int(value)


def _aggregate_value(value: Any) -> float | dict[str, float]:
    if isinstance(value, Mapping):
        if not value:
            raise ValueError("aggregate mapping must not be empty")
        result: dict[str, float] = {}
        for name, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if not isinstance(name, str) or not name:
                raise ValueError("aggregate names must be non-empty strings")
            result[name] = _finite_number(item, field_name="aggregate")
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        raise TypeError("row-level release is not supported by aggregate DP")
    return _finite_number(value, field_name="aggregate")


def _finite_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} values must be finite numbers")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{field_name} values must be finite numbers") from None
    if not math.isfinite(number):
        raise ValueError(f"{field_name} values must be finite numbers")
    return number


def _positive_finite(value: Any, *, field_name: str) -> float:
    number = _finite_number(value, field_name=field_name)
    if number <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero")
    return number


def _delta(value: Any) -> float:
    number = _finite_number(value, field_name="delta")
    if not 0.0 <= number < 1.0:
        raise ValueError("delta must be in [0, 1)")
    return number


def _validate_label(value: Any, *, field_name: str = "label") -> None:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(f"{field_name} must be a non-empty short string")


def _random_source(seed: Any, label: str) -> random.Random | random.SystemRandom:
    if seed is None:
        return random.SystemRandom()
    seed_bytes = hashlib.sha256(
        _seed_text(seed).encode("utf-8") + b"\0" + label.encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(seed_bytes, "big"))


def _seed_text(seed: Any) -> str:
    if isinstance(seed, bytes):
        return seed.hex()
    if isinstance(seed, (str, int)) and not isinstance(seed, bool):
        return str(seed)
    raise TypeError("seed must be a string, integer, bytes, or None")


def _laplace_noise(
    rng: random.Random | random.SystemRandom,
    scale: float,
) -> float:
    unit = rng.random() - 0.5
    magnitude = -scale * math.log1p(-2.0 * abs(unit))
    if unit == 0.0:
        return 0.0
    return math.copysign(magnitude, unit)
