"""Round-aware Renyi accounting for subsampled federated training rounds.

The module turns a declared round schedule into a composed ``(epsilon, delta)``
guarantee and gates a candidate round before it runs. Accounting is offline and
deterministic: it reads numeric round parameters only (round index, sampling
rate, clipping norm, noise multiplier) and never touches client updates,
payloads, identifiers, or record counts.

Privacy analysis
----------------

Each round is treated as a clipped Gaussian mechanism whose noise has standard
deviation ``noise_multiplier * clipping_norm``, so the sensitivity-to-noise ratio
is ``1 / noise_multiplier`` and the Renyi divergence of order ``alpha`` is

    eps_gaussian(alpha) = alpha / (2 * noise_multiplier ** 2)

A round samples its participants with rate ``q``, so the mechanism that touches
any single client's contribution is a two-point mixture: with probability
``1 - q`` the client is not sampled and the output distribution is unchanged,
and with probability ``q`` the sampled mechanism applies. Joint convexity of
Renyi divergence over that mixture gives the conservative per-round bound

    eps_round(alpha) = log(1 - q + q * exp((alpha - 1) * eps_gaussian)) / (alpha - 1)

which is valid for every order ``alpha > 1`` and reduces to ``eps_gaussian``
exactly when ``q == 1``. The analytic sampled-Gaussian accountant is tighter but
only has closed forms for integer orders; the convexity bound is used here so the
declared order grid, including fractional orders, stays sound.

Renyi divergences compose additively across the rounds one client participates
in, and the standard conversion

    eps = min over alpha of (eps_total(alpha) + log(1 / delta) / (alpha - 1))

turns the composed curve into a ``(epsilon, delta)`` guarantee. A schedule that
enumerates fewer rounds than a client participates in understates that client's
loss, so callers must list every round of the worst-case participation set.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

RDP_ACCOUNTANT_SCHEMA_VERSION: Final[str] = (
    "openmed.training.federated.rdp_accountant.v1"
)
DEFAULT_RENYI_ORDERS: Final[tuple[int, ...]] = (2, 4, 8, 16, 32, 64)
RDP_ACCOUNTING_REASON_CODES: Final[tuple[str, ...]] = (
    "within_policy",
    "epsilon_exceeded",
    "delta_exceeded",
    "round_limit_exceeded",
)
MAX_RENYI_ORDER: Final[int] = 4096
MAX_RENYI_ORDERS: Final[int] = 64
MAX_ACCOUNTED_ROUNDS: Final[int] = 1_000_000
MAX_CLIPPING_NORM: Final[float] = 1e12
MAX_NOISE_MULTIPLIER: Final[float] = 1e6

_DIGEST_PREFIX: Final[str] = RDP_ACCOUNTANT_SCHEMA_VERSION + "\0"
_REPORT_FIELDS: Final[tuple[str, ...]] = (
    "round_count",
    "orders",
    "epsilons",
    "delta",
    "epsilon",
)


class RdpAccountantError(ValueError):
    """Value-free accounting failure with a stable machine-readable category.

    The message names the rejected field and its contract but never echoes the
    submitted value, so invalid round parameters can be logged safely.
    """

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


def _require_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise RdpAccountantError(f"{name}_invalid", f"{name} must be an integer")
    number = int(value)
    if number < minimum or number > maximum:
        raise RdpAccountantError(
            f"{name}_invalid",
            f"{name} must be between {minimum} and {maximum}",
        )
    return number


def _require_number(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = False,
    maximum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RdpAccountantError(
            f"{name}_invalid",
            f"{name} must be a number",
        )
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise RdpAccountantError(
            f"{name}_invalid",
            f"{name} must be finite",
        )
    above_minimum = number >= minimum if minimum_inclusive else number > minimum
    below_maximum = number <= maximum if maximum_inclusive else number < maximum
    if not above_minimum or not below_maximum:
        raise RdpAccountantError(
            f"{name}_invalid",
            f"{name} is outside its supported range",
        )
    return number


def _require_order(value: object, *, name: str = "order") -> int:
    order = _require_int(value, name=name, minimum=2, maximum=MAX_RENYI_ORDER)
    return order


def _require_orders(value: object, *, name: str = "orders") -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RdpAccountantError(f"{name}_invalid", f"{name} must be a sequence")
    if not value:
        raise RdpAccountantError(f"{name}_invalid", f"{name} must not be empty")
    if len(value) > MAX_RENYI_ORDERS:
        raise RdpAccountantError(
            f"{name}_invalid",
            f"{name} must contain at most {MAX_RENYI_ORDERS} entries",
        )
    orders: list[int] = []
    for entry in value:
        orders.append(_require_order(entry, name=f"{name}_entry"))
    if orders != sorted(set(orders)):
        raise RdpAccountantError(
            f"{name}_invalid",
            f"{name} must be strictly increasing and unique",
        )
    return tuple(orders)


def _require_delta(value: object, *, name: str = "delta") -> float:
    return _require_number(
        value,
        name=name,
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )


@dataclass(frozen=True, slots=True)
class FederatedRenyiRound:
    """One subsampled federated round charged to a client's privacy budget.

    The record carries round mechanics only: the round index, the sampling rate,
    the clipping norm applied to client updates, and the noise multiplier. No
    client identifier, site label, update value, or record count is accepted.
    """

    round_index: int
    sampling_rate: float
    clipping_norm: float
    noise_multiplier: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "round_index",
            _require_int(
                self.round_index,
                name="round_index",
                minimum=0,
                maximum=MAX_ACCOUNTED_ROUNDS,
            ),
        )
        object.__setattr__(
            self,
            "sampling_rate",
            _require_number(
                self.sampling_rate,
                name="sampling_rate",
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "clipping_norm",
            _require_number(
                self.clipping_norm,
                name="clipping_norm",
                minimum=0.0,
                maximum=MAX_CLIPPING_NORM,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "noise_multiplier",
            _require_number(
                self.noise_multiplier,
                name="noise_multiplier",
                minimum=0.0,
                maximum=MAX_NOISE_MULTIPLIER,
                minimum_inclusive=False,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible round payload."""

        return {
            "round_index": self.round_index,
            "sampling_rate": self.sampling_rate,
            "clipping_norm": self.clipping_norm,
            "noise_multiplier": self.noise_multiplier,
        }


@dataclass(frozen=True, slots=True)
class RenyiFederationPolicy:
    """Declared federation ceiling for a composed Renyi accounting result."""

    max_epsilon: float
    max_delta: float
    orders: tuple[int, ...] = DEFAULT_RENYI_ORDERS
    max_rounds: int = MAX_ACCOUNTED_ROUNDS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_epsilon",
            _require_number(
                self.max_epsilon,
                name="max_epsilon",
                minimum=0.0,
                maximum=MAX_CLIPPING_NORM,
                minimum_inclusive=False,
            ),
        )
        object.__setattr__(
            self,
            "max_delta",
            _require_delta(self.max_delta, name="max_delta"),
        )
        object.__setattr__(self, "orders", _require_orders(self.orders))
        object.__setattr__(
            self,
            "max_rounds",
            _require_int(
                self.max_rounds,
                name="max_rounds",
                minimum=1,
                maximum=MAX_ACCOUNTED_ROUNDS,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible policy payload."""

        return {
            "max_epsilon": self.max_epsilon,
            "max_delta": self.max_delta,
            "orders": list(self.orders),
            "max_rounds": self.max_rounds,
        }


@dataclass(frozen=True, slots=True)
class RenyiAccountingReport:
    """Composed Renyi curve and its ``(epsilon, delta)`` conversion."""

    round_count: int
    orders: tuple[int, ...]
    epsilons: tuple[float, ...]
    delta: float
    epsilon: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "round_count",
            _require_int(
                self.round_count,
                name="round_count",
                minimum=0,
                maximum=MAX_ACCOUNTED_ROUNDS,
            ),
        )
        orders = _require_orders(self.orders)
        object.__setattr__(self, "orders", orders)
        epsilons = tuple(self.epsilons)
        if len(epsilons) != len(orders):
            raise RdpAccountantError(
                "epsilons_invalid",
                "epsilons must have one entry per order",
            )
        for value in epsilons:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RdpAccountantError(
                    "epsilons_invalid",
                    "epsilons must be numbers",
                )
            if math.isnan(float(value)) or float(value) < 0.0:
                raise RdpAccountantError(
                    "epsilons_invalid",
                    "epsilons must be non-negative finite numbers",
                )
        object.__setattr__(self, "epsilons", tuple(float(v) for v in epsilons))
        object.__setattr__(self, "delta", _require_delta(self.delta))
        epsilon = _require_number(
            self.epsilon,
            name="epsilon",
            minimum=-1.0,
            maximum=MAX_CLIPPING_NORM,
            minimum_inclusive=True,
        )
        if epsilon < 0.0:
            raise RdpAccountantError(
                "epsilon_invalid",
                "epsilon must be non-negative",
            )
        object.__setattr__(self, "epsilon", epsilon)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible report payload."""

        payload: dict[str, Any] = {
            "round_count": self.round_count,
            "orders": list(self.orders),
            "epsilons": list(self.epsilons),
            "delta": self.delta,
            "epsilon": self.epsilon,
        }
        return {field: payload[field] for field in _REPORT_FIELDS}

    def to_json(self) -> str:
        """Return the canonical compact JSON encoding of this report."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class RenyiRoundDecision:
    """Gate verdict for one candidate federated round."""

    allowed: bool
    reason_code: str
    report: RenyiAccountingReport

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise RdpAccountantError("allowed_invalid", "allowed must be a boolean")
        if self.reason_code not in RDP_ACCOUNTING_REASON_CODES:
            raise RdpAccountantError(
                "reason_code_invalid",
                "reason_code must be a declared accounting reason code",
            )
        if not isinstance(self.report, RenyiAccountingReport):
            raise RdpAccountantError(
                "report_invalid",
                "report must be a RenyiAccountingReport",
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible decision payload."""

        return {
            "allowed": self.allowed,
            "reason_code": self.reason_code,
            "report": self.report.to_dict(),
        }

    def to_json(self) -> str:
        """Return the canonical compact JSON encoding of this decision."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _validated_rounds(value: object) -> tuple[FederatedRenyiRound, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RdpAccountantError("rounds_invalid", "rounds must be a sequence")
    if len(value) > MAX_ACCOUNTED_ROUNDS:
        raise RdpAccountantError(
            "rounds_invalid",
            f"rounds must contain at most {MAX_ACCOUNTED_ROUNDS} entries",
        )
    rounds: list[FederatedRenyiRound] = []
    for entry in value:
        if not isinstance(entry, FederatedRenyiRound):
            raise RdpAccountantError(
                "rounds_invalid",
                "rounds must contain FederatedRenyiRound entries",
            )
        rounds.append(entry)
    ordered = tuple(
        sorted(
            rounds,
            key=lambda item: (
                item.round_index,
                item.sampling_rate,
                item.clipping_norm,
                item.noise_multiplier,
            ),
        )
    )
    indices = [round_.round_index for round_ in ordered]
    if len(indices) != len(set(indices)):
        raise RdpAccountantError(
            "duplicate_round_index",
            "round_index values must be unique within a schedule",
        )
    return ordered


def round_renyi_epsilon(round: FederatedRenyiRound, order: int) -> float:
    """Return the per-round Renyi divergence of ``order`` for ``round``.

    The returned value is the conservative subsampling bound documented at module
    level: it equals ``order / (2 * noise_multiplier ** 2)`` when the round
    samples every participant and is strictly smaller otherwise.
    """

    if not isinstance(round, FederatedRenyiRound):
        raise RdpAccountantError(
            "round_invalid",
            "round must be a FederatedRenyiRound",
        )
    alpha = _require_order(order)
    gaussian = alpha / (2.0 * round.noise_multiplier * round.noise_multiplier)
    exponent = (alpha - 1) * gaussian
    if round.sampling_rate == 1.0:
        return gaussian
    if exponent > 1.0:
        # log(1 - q + q * exp(exponent)) evaluated without overflowing exp.
        correction = math.log1p((1.0 - round.sampling_rate) * math.expm1(-exponent))
        return (exponent + correction) / (alpha - 1)
    return math.log1p(round.sampling_rate * math.expm1(exponent)) / (alpha - 1)


def compose_renyi_curve(
    rounds: Sequence[FederatedRenyiRound],
    orders: Sequence[int] = DEFAULT_RENYI_ORDERS,
) -> tuple[float, ...]:
    """Return the composed Renyi curve over the rounds one client participates in."""

    ordered = _validated_rounds(rounds)
    order_grid = _require_orders(orders)
    return tuple(
        sum(round_renyi_epsilon(round_, order) for round_ in ordered)
        for order in order_grid
    )


def renyi_to_dp(
    epsilons: Sequence[float],
    orders: Sequence[int],
    delta: float,
) -> float:
    """Convert a Renyi curve into an ``(epsilon, delta)`` epsilon ceiling."""

    order_grid = _require_orders(orders)
    conversion_delta = _require_delta(delta)
    curve = tuple(epsilons)
    if len(curve) != len(order_grid):
        raise RdpAccountantError(
            "epsilons_invalid",
            "epsilons must have one entry per order",
        )
    slack = -math.log(conversion_delta)
    converted: list[float] = []
    for order, epsilon in zip(order_grid, curve):
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise RdpAccountantError(
                "epsilons_invalid",
                "epsilons must be numbers",
            )
        value = float(epsilon)
        if math.isnan(value) or math.isinf(value) or value < 0.0:
            raise RdpAccountantError(
                "epsilons_invalid",
                "epsilons must be non-negative finite numbers",
            )
        converted.append(value + slack / (order - 1))
    return min(converted)


def account_federated_rounds(
    rounds: Sequence[FederatedRenyiRound],
    *,
    orders: Sequence[int] = DEFAULT_RENYI_ORDERS,
    delta: float,
) -> RenyiAccountingReport:
    """Compose ``rounds`` and convert the curve with ``delta``.

    ``rounds`` must list every round of the worst-case client participation set.
    A schedule with no rounds is accounted as ``epsilon = 0`` rather than as the
    vacuous conversion of a zero curve.
    """

    ordered = _validated_rounds(rounds)
    order_grid = _require_orders(orders)
    conversion_delta = _require_delta(delta)
    epsilons = compose_renyi_curve(ordered, order_grid)
    epsilon = (
        0.0 if not ordered else renyi_to_dp(epsilons, order_grid, conversion_delta)
    )
    return RenyiAccountingReport(
        round_count=len(ordered),
        orders=order_grid,
        epsilons=epsilons,
        delta=conversion_delta,
        epsilon=epsilon,
    )


def evaluate_federation_round(
    policy: RenyiFederationPolicy,
    schedule: Sequence[FederatedRenyiRound],
    candidate: FederatedRenyiRound,
    *,
    target_delta: float | None = None,
) -> RenyiRoundDecision:
    """Gate ``candidate`` against ``policy`` using the composed schedule.

    The candidate is charged on top of ``schedule``, which lists the rounds the
    worst-case client already participates in.
    """

    if not isinstance(policy, RenyiFederationPolicy):
        raise RdpAccountantError(
            "policy_invalid",
            "policy must be a RenyiFederationPolicy",
        )
    if not isinstance(candidate, FederatedRenyiRound):
        raise RdpAccountantError(
            "candidate_invalid",
            "candidate must be a FederatedRenyiRound",
        )
    charged = _validated_rounds(schedule)
    if len(charged) + 1 > policy.max_rounds:
        report = account_federated_rounds(
            charged + (candidate,),
            orders=policy.orders,
            delta=policy.max_delta,
        )
        return RenyiRoundDecision(
            allowed=False,
            reason_code="round_limit_exceeded",
            report=report,
        )
    delta = (
        policy.max_delta
        if target_delta is None
        else _require_delta(target_delta, name="target_delta")
    )
    if delta > policy.max_delta:
        report = account_federated_rounds(
            charged + (candidate,),
            orders=policy.orders,
            delta=policy.max_delta,
        )
        return RenyiRoundDecision(
            allowed=False,
            reason_code="delta_exceeded",
            report=report,
        )
    report = account_federated_rounds(
        charged + (candidate,),
        orders=policy.orders,
        delta=delta,
    )
    if report.epsilon > policy.max_epsilon:
        return RenyiRoundDecision(
            allowed=False,
            reason_code="epsilon_exceeded",
            report=report,
        )
    return RenyiRoundDecision(
        allowed=True,
        reason_code="within_policy",
        report=report,
    )


def fingerprint_accounting_report(report: RenyiAccountingReport) -> str:
    """Return a domain-separated digest over a report's canonical payload."""

    if not isinstance(report, RenyiAccountingReport):
        raise RdpAccountantError(
            "report_invalid",
            "report must be a RenyiAccountingReport",
        )
    payload = json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(_DIGEST_PREFIX.encode("utf-8"))
    digest.update(payload.encode("utf-8"))
    return "sha256:" + digest.hexdigest()


__all__ = [
    "DEFAULT_RENYI_ORDERS",
    "MAX_ACCOUNTED_ROUNDS",
    "MAX_CLIPPING_NORM",
    "MAX_NOISE_MULTIPLIER",
    "MAX_RENYI_ORDERS",
    "MAX_RENYI_ORDER",
    "RDP_ACCOUNTANT_SCHEMA_VERSION",
    "RDP_ACCOUNTING_REASON_CODES",
    "FederatedRenyiRound",
    "RdpAccountantError",
    "RenyiAccountingReport",
    "RenyiFederationPolicy",
    "RenyiRoundDecision",
    "account_federated_rounds",
    "compose_renyi_curve",
    "evaluate_federation_round",
    "fingerprint_accounting_report",
    "renyi_to_dp",
    "round_renyi_epsilon",
]
