"""Offline differential-privacy primitives for aggregate releases.

This module is deliberately separate from row-level de-identification. The
ledger accounts only for named aggregate queries, and the Laplace mechanism
rejects row-shaped inputs. A budget entry is never evidence that individual
rows were anonymized.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from openmed.core.audit import stable_hash

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
]


class DPAggregateBudgetExceeded(ValueError):
    """Raised when an aggregate query would exhaust the declared budget."""


DPBudgetExhausted = DPAggregateBudgetExceeded


@dataclass(frozen=True)
class DPBudgetSpend:
    """One aggregate-query privacy spend."""

    sequence: int
    label: str
    mechanism: str
    epsilon: float
    delta: float
    scope: str = "aggregate"

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("sequence must be an integer >= 1")
        _validate_label(self.label)
        _validate_label(self.mechanism, field_name="mechanism")
        _positive_finite(self.epsilon, field_name="epsilon")
        _delta(self.delta)
        if self.scope != "aggregate":
            raise ValueError("the differential-privacy ledger is aggregate-only")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe accounting metadata."""

        return {
            "sequence": self.sequence,
            "label": self.label,
            "mechanism": self.mechanism,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "scope": self.scope,
        }


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

    value: float | dict[str, float]
    noise: float | dict[str, float]
    spend: DPBudgetSpend
    composition: DPBudgetComposition
    mechanism: str = "laplace"
    scope: str = "aggregate_only"
    seed_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the released aggregate and safe accounting metadata."""

        return {
            "schema_version": 1,
            "scope": self.scope,
            "row_level_anonymization": False,
            "mechanism": self.mechanism,
            "value": self.value,
            "noise": self.noise,
            "spend": self.spend.to_dict(),
            "composition": self.composition.to_dict(),
            "seed_digest": self.seed_digest,
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
    A supplied seed makes synthetic/offline tests reproducible; production
    callers should omit it and use system randomness.
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
    spend = ledger.spend(
        epsilon,
        delta,
        label=label,
        mechanism="laplace",
    )
    rng = _random_source(seed, label)
    if isinstance(normalized, dict):
        noise = {name: _laplace_noise(rng, scale) for name in normalized}
        released = {name: normalized[name] + noise[name] for name in normalized}
    else:
        noise = _laplace_noise(rng, scale)
        released = normalized + noise
    return AggregateDPRelease(
        value=released,
        noise=noise,
        spend=spend,
        composition=ledger.compose(),
        seed_digest=(
            stable_hash({"seed": _seed_text(seed), "label": label})
            if seed is not None
            else None
        ),
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
