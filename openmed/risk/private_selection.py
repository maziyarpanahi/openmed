"""Privacy accounting for private hyperparameter and checkpoint selection.

Training several differentially private candidates and keeping the best one
by a private validation result leaks more than the selected run alone. A
:class:`TuningLedger` records every candidate (including abandoned ones), the
mechanism parameters handed to an injected per-trial accountant, an opaque
lineage digest, a banded outcome, and every selection event on private
metrics (best-of-k, early stopping, checkpoint or round selection).

By default the ledger charges the basic composition of every trial plus every
private metric evaluation used for selection. A caller may instead declare a
random-stopping scheme whose number of trials is drawn from a published
distribution with an injected cryptographically secure generator. The
supported bounds are:

* Papernot and Steinke, "Hyperparameter Tuning with Renyi Differential
  Privacy", ICLR 2022: Definition 1 (truncated negative binomial), Theorem 2
  (equation 5, RDP to RDP), Corollary 3 (pure DP, ``(2 + eta) * epsilon``),
  Corollary 4 (zCDP) and Theorem 6 (Poisson).
* Liu and Talwar, "Private Selection from Private Candidates", STOC 2019,
  Theorem 3.2: geometric stopping of an ``epsilon``-DP candidate is
  ``3 * epsilon``-DP. This is Corollary 3 with ``eta = 1``.

RDP guarantees are converted to approximate DP with Mironov, "Renyi
Differential Privacy", CSF 2017, Proposition 3.

Records contain controlled codes, numeric mechanism parameters, banded
outcomes and digests only: no metric values, seeds, generator state, site
identities or free-form text. Errors carry a fixed reason code and at most a
field name.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Any, Final

from openmed.risk.budget import DEFAULT_RDP_ORDERS
from openmed.risk.privacy_budget import (
    MAX_PRIVACY_BUDGET_EPSILON,
    PrivacyBudgetDecision,
    PrivacyBudgetLedger,
    PrivacyBudgetLedgerExceeded,
)

PRIVATE_SELECTION_SCHEMA_VERSION: Final = "openmed.private_selection.v1"
MAX_TUNING_TRIALS: Final = 1_000
MAX_TRIAL_EVALUATIONS: Final = 10_000
MAX_SCHEME_TAIL_MASS: Final = 1e-6
MAX_RDP_CURVE_ORDERS: Final = 64

SELECTION_KINDS: Final = (
    "best_of_k",
    "early_stopping",
    "checkpoint_selection",
    "round_selection",
)
IN_TRIAL_SELECTION_KINDS: Final = SELECTION_KINDS[1:]
METRIC_SOURCES: Final = ("private", "public")
TRIAL_STATUSES: Final = ("completed", "abandoned")
OUTCOME_BANDS: Final = ("not_evaluated", "low", "moderate", "high", "very_high")
STOPPING_DISTRIBUTIONS: Final = (
    "truncated_negative_binomial",
    "geometric",
    "logarithmic",
    "poisson",
)
ACCOUNTING_METHODS: Final = (
    "basic_composition",
    "tnb_pure_dp",
    "tnb_rdp",
    "poisson_rdp",
)
EVIDENCE_COVERAGE: Final = "all_trials"

PRIVATE_SELECTION_REASONS: Final = frozenset(
    {
        "accountant_failed",
        "accountant_invalid",
        "account_overflow",
        "already_charged",
        "already_selected",
        "bound_not_applicable",
        "budget_context_unknown",
        "budget_exceeded",
        "evidence_account_understated",
        "evidence_coverage_incomplete",
        "evidence_invalid",
        "evidence_ledger_mismatch",
        "evidence_selected_trial_only",
        "fixed_trial_count",
        "invalid_field",
        "ledger_closed",
        "lineage_digest_invalid",
        "lineage_digest_reused",
        "no_selected_trial",
        "no_trials",
        "not_charged",
        "outcome_not_banded",
        "rng_not_csprng",
        "scheme_not_declared",
        "scheme_tail_too_heavy",
        "scoring_cost_missing",
        "selected_trial_not_eligible",
        "selection_kind_invalid",
        "trial_count_already_drawn",
        "trial_count_exceeded",
        "trial_count_incomplete",
        "trial_count_limit",
        "trial_count_not_drawn",
        "trial_exceeds_declared_cost",
        "trial_limit",
        "unknown_trial",
        "unsupported_distribution",
    }
)

_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_FIXED_COUNT_KEYS: Final = frozenset(
    {"trial_count", "trials", "fixed_trial_count", "num_trials", "k"}
)
_POINT_MASS_DISTRIBUTIONS: Final = frozenset({"fixed", "point_mass", "constant"})
_SCHEME_FIELDS: Final = frozenset(
    {"distribution", "eta", "gamma", "mu", "target_delta", "per_trial"}
)
_COST_FIELDS: Final = frozenset({"epsilon", "delta", "rdp"})
_MECHANISM_FIELDS: Final = (
    "noise_multiplier",
    "clipping_norm",
    "sampling_rate",
    "round_count",
    "learning_rate",
)
_EVIDENCE_FIELDS: Final = frozenset(
    {
        "schema_version",
        "ledger_digest",
        "accounting",
        "method",
        "epsilon",
        "delta",
        "coverage",
        "selected_lineage_digest",
        "trial_count",
    }
)
_MAX_ROUNDS: Final = 10_000_000
_DEFAULT_BAND_EDGES: Final = (0.25, 0.5, 0.75)


class PrivateSelectionError(ValueError):
    """Fail-closed refusal with a stable reason code.

    Args:
        reason: A code from :data:`PRIVATE_SELECTION_REASONS`.
        field_name: Optional name of the offending field. Values are never
            included in the message.
    """

    def __init__(self, reason: str, field_name: str | None = None) -> None:
        if reason not in PRIVATE_SELECTION_REASONS:
            reason = "invalid_field"
        self.reason = reason
        self.field_name = field_name
        message = f"private selection refused: {reason}"
        if field_name:
            message = f"{message} ({field_name})"
        super().__init__(message)


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrivateSelectionError("invalid_field", field_name)
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PrivateSelectionError("invalid_field", field_name)
    return parsed


def _non_negative(
    value: Any, field_name: str, maximum: float = MAX_PRIVACY_BUDGET_EPSILON
) -> float:
    parsed = _number(value, field_name)
    if parsed < 0.0 or parsed > maximum:
        raise PrivateSelectionError("invalid_field", field_name)
    return parsed


def _positive(value: Any, field_name: str) -> float:
    parsed = _number(value, field_name)
    if parsed <= 0.0:
        raise PrivateSelectionError("invalid_field", field_name)
    return parsed


def _delta(value: Any, field_name: str) -> float:
    parsed = _number(value, field_name)
    if parsed < 0.0 or parsed >= 1.0:
        raise PrivateSelectionError("invalid_field", field_name)
    return parsed


def _open_unit(value: Any, field_name: str) -> float:
    parsed = _number(value, field_name)
    if parsed <= 0.0 or parsed >= 1.0:
        raise PrivateSelectionError("invalid_field", field_name)
    return parsed


def _eta(value: Any) -> float:
    parsed = _number(value, "eta")
    if parsed <= -1.0 or parsed > 1_000.0:
        raise PrivateSelectionError("invalid_field", "eta")
    return parsed


def _order(value: Any, field_name: str = "order") -> float:
    parsed = _number(value, field_name)
    if parsed <= 1.0:
        raise PrivateSelectionError("invalid_field", field_name)
    return parsed


def _bounded_int(value: Any, field_name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise PrivateSelectionError("invalid_field", field_name)
    return value


def _mapping_items(payload: Any, field_name: str, limit: int) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PrivateSelectionError("invalid_field", field_name)
    try:
        items = list(payload.items())
    except Exception:  # noqa: BLE001 - mappings are caller-controlled protocols.
        raise PrivateSelectionError("invalid_field", field_name) from None
    if len(items) > limit:
        raise PrivateSelectionError("invalid_field", field_name)
    result: dict[str, Any] = {}
    for key, value in items:
        if type(key) is not str:
            raise PrivateSelectionError("invalid_field", field_name)
        result[key] = value
    return result


def _rdp_curve(value: Any) -> tuple[tuple[float, float], ...]:
    if isinstance(value, Mapping):
        pairs: Iterable[Any] = list(value.items())
    elif isinstance(value, (list, tuple)):
        pairs = value
    else:
        raise PrivateSelectionError("invalid_field", "rdp")
    curve: dict[float, float] = {}
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise PrivateSelectionError("invalid_field", "rdp")
        order = _order(pair[0], "rdp")
        if order in curve or len(curve) >= MAX_RDP_CURVE_ORDERS:
            raise PrivateSelectionError("invalid_field", "rdp")
        curve[order] = _non_negative(pair[1], "rdp")
    return tuple(sorted(curve.items()))


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256(payload: Any) -> str:
    encoded = _canonical_json(payload).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Privacy costs and mechanisms.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialPrivacyCost:
    """Privacy guarantee of one candidate run or one private metric evaluation.

    Attributes:
        epsilon: Approximate-DP epsilon of the run.
        delta: Approximate-DP delta of the run. Zero means pure DP.
        rdp: Optional Renyi DP curve as ``(order, epsilon)`` pairs with each
            order greater than one.
    """

    epsilon: float
    delta: float = 0.0
    rdp: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "epsilon", _non_negative(self.epsilon, "epsilon"))
        object.__setattr__(self, "delta", _delta(self.delta, "delta"))
        object.__setattr__(self, "rdp", _rdp_curve(self.rdp))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TrialPrivacyCost:
        """Build a cost from a closed JSON-style mapping.

        Args:
            payload: Mapping with ``epsilon`` and optional ``delta`` and
                ``rdp`` keys.

        Returns:
            The validated cost.

        Raises:
            PrivateSelectionError: If the mapping has unknown or invalid
                fields.
        """

        values = _mapping_items(payload, "per_trial", len(_COST_FIELDS))
        if set(values) - _COST_FIELDS or "epsilon" not in values:
            raise PrivateSelectionError("invalid_field", "per_trial")
        return cls(
            epsilon=values["epsilon"],
            delta=values.get("delta", 0.0),
            rdp=values.get("rdp", ()),
        )

    def rdp_at(self, order: float) -> float | None:
        """Return the best known Renyi DP epsilon at ``order``.

        A pure ``epsilon``-DP guarantee implies ``(order, epsilon)``-RDP by
        monotonicity and ``(order, epsilon**2 * order / 2)``-RDP (Papernot and
        Steinke, Lemma 13). An explicit curve value is used when present.

        Args:
            order: Renyi order greater than one.

        Returns:
            The RDP epsilon, or ``None`` when no bound is known at that order.
        """

        candidates = [value for curve_order, value in self.rdp if curve_order == order]
        if self.delta == 0.0:
            candidates.append(min(self.epsilon, 0.5 * order * self.epsilon**2))
        return min(candidates) if candidates else None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "rdp": [[order, value] for order, value in self.rdp],
        }


def _compose(costs: Sequence[tuple[TrialPrivacyCost, int]]) -> TrialPrivacyCost:
    """Return the basic sequential composition of repeated costs."""

    active = [(cost, count) for cost, count in costs if count > 0]
    epsilon = math.fsum(cost.epsilon * count for cost, count in active)
    delta = math.fsum(cost.delta * count for cost, count in active)
    if epsilon > MAX_PRIVACY_BUDGET_EPSILON or delta >= 1.0:
        raise PrivateSelectionError("account_overflow")
    orders = sorted({order for cost, _ in active for order, _ in cost.rdp})
    curve = []
    for order in orders:
        values = [(cost.rdp_at(order), count) for cost, count in active]
        if any(value is None for value, _ in values):
            continue
        total = math.fsum(float(value) * count for value, count in values)
        if total <= MAX_PRIVACY_BUDGET_EPSILON:
            curve.append((order, total))
    return TrialPrivacyCost(epsilon=epsilon, delta=delta, rdp=tuple(curve))


def _dominated(cost: TrialPrivacyCost, declared: TrialPrivacyCost) -> bool:
    if cost.epsilon > declared.epsilon or cost.delta > declared.delta:
        return False
    for order, value in declared.rdp:
        actual = cost.rdp_at(order)
        if actual is None or actual > value:
            return False
    return True


@dataclass(frozen=True)
class TrialMechanism:
    """Numeric mechanism parameters of one candidate, passed to the accountant.

    Attributes:
        noise_multiplier: Gaussian noise multiplier, greater than zero.
        clipping_norm: Per-example or per-update clipping norm.
        sampling_rate: Participant or example sampling rate in ``(0, 1]``.
        round_count: Planned number of rounds or steps.
        learning_rate: Optional learning rate recorded for lineage.
    """

    noise_multiplier: float
    clipping_norm: float
    sampling_rate: float
    round_count: int
    learning_rate: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "noise_multiplier",
            _positive(self.noise_multiplier, "noise_multiplier"),
        )
        object.__setattr__(
            self, "clipping_norm", _positive(self.clipping_norm, "clipping_norm")
        )
        rate = _positive(self.sampling_rate, "sampling_rate")
        if rate > 1.0:
            raise PrivateSelectionError("invalid_field", "sampling_rate")
        object.__setattr__(self, "sampling_rate", rate)
        _bounded_int(self.round_count, "round_count", 1, _MAX_ROUNDS)
        if self.learning_rate is not None:
            object.__setattr__(
                self, "learning_rate", _positive(self.learning_rate, "learning_rate")
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TrialMechanism:
        """Build mechanism parameters from a closed mapping.

        Args:
            payload: Mapping restricted to the declared mechanism fields.
                Seeds, generator state and site fields are refused.

        Returns:
            The validated mechanism parameters.

        Raises:
            PrivateSelectionError: If unknown or invalid fields are present.
        """

        values = _mapping_items(payload, "mechanism", len(_MECHANISM_FIELDS))
        if set(values) - set(_MECHANISM_FIELDS):
            raise PrivateSelectionError("invalid_field", "mechanism")
        try:
            return cls(**values)
        except TypeError:
            raise PrivateSelectionError("invalid_field", "mechanism") from None

    def to_dict(self) -> dict[str, Any]:
        """Return the numeric mechanism parameters."""

        payload: dict[str, Any] = {
            "noise_multiplier": self.noise_multiplier,
            "clipping_norm": self.clipping_norm,
            "sampling_rate": self.sampling_rate,
            "round_count": self.round_count,
        }
        if self.learning_rate is not None:
            payload["learning_rate"] = self.learning_rate
        return payload


TrialAccountant = Callable[[TrialMechanism], TrialPrivacyCost]


def band_outcome(value: float, edges: Sequence[float] = _DEFAULT_BAND_EDGES) -> str:
    """Map a metric value in ``[0, 1]`` to a controlled outcome band.

    The value itself is not retained; only the returned band code may be
    recorded in a :class:`TuningLedger`.

    Args:
        value: Metric value, typically in ``[0, 1]``.
        edges: Three strictly increasing band edges.

    Returns:
        One of ``low``, ``moderate``, ``high`` or ``very_high``.

    Raises:
        PrivateSelectionError: If the value or edges are invalid.
    """

    metric = _number(value, "value")
    bounds = tuple(_number(edge, "edges") for edge in edges)
    if len(bounds) != 3 or not bounds[0] < bounds[1] < bounds[2]:
        raise PrivateSelectionError("invalid_field", "edges")
    for index, edge in enumerate(bounds):
        if metric < edge:
            return OUTCOME_BANDS[index + 1]
    return OUTCOME_BANDS[-1]


# ---------------------------------------------------------------------------
# Published private-selection bounds.
# ---------------------------------------------------------------------------


def truncated_negative_binomial_mean(eta: float, gamma: float) -> float:
    """Return ``E[K]`` for the truncated negative binomial ``D_{eta,gamma}``.

    Papernot and Steinke (ICLR 2022), Definition 1: ``eta * (1 - gamma) /
    (gamma * (1 - gamma**eta))`` for ``eta != 0`` and ``(1/gamma - 1) /
    log(1/gamma)`` for the logarithmic case ``eta == 0``. The geometric case
    ``eta == 1`` gives ``1 / gamma`` (Liu and Talwar, Theorem 1.3(b)).

    Args:
        eta: Shape parameter in ``(-1, inf)``.
        gamma: Stopping parameter in ``(0, 1)``.

    Returns:
        The expected number of trials.
    """

    eta = _eta(eta)
    gamma = _open_unit(gamma, "gamma")
    if eta == 0.0:
        return (1.0 / gamma - 1.0) / math.log(1.0 / gamma)
    return eta * (1.0 - gamma) / (gamma * -math.expm1(eta * math.log(gamma)))


def random_stopping_pure_dp_epsilon(epsilon: float, eta: float) -> float:
    """Return the pure-DP epsilon of best-of-``K`` with ``K ~ D_{eta,gamma}``.

    Papernot and Steinke (ICLR 2022), Corollary 3: an ``(epsilon, 0)``-DP
    candidate repeated a truncated negative binomial number of times and
    reduced to its best output is ``((2 + eta) * epsilon, 0)``-DP for any
    ``gamma``. With ``eta = 1`` (geometric stopping) this is the
    ``3 * epsilon`` bound of Liu and Talwar (STOC 2019), Theorem 3.2.

    Args:
        epsilon: Pure-DP epsilon of one candidate run.
        eta: Shape parameter in ``(-1, inf)``.

    Returns:
        The epsilon of the selected output.
    """

    return (2.0 + _eta(eta)) * _non_negative(epsilon, "epsilon")


def truncated_negative_binomial_rdp(
    order: float,
    epsilon: float,
    order_hat: float,
    epsilon_hat: float,
    *,
    eta: float,
    gamma: float,
) -> float:
    """Return Theorem 2 of Papernot and Steinke (ICLR 2022), equation 5.

    For a candidate that is ``(order, epsilon)``-RDP and
    ``(order_hat, epsilon_hat)``-RDP, best-of-``K`` with ``K ~ D_{eta,gamma}``
    is ``(order, epsilon')``-RDP with ``epsilon' = epsilon + (1 + eta) * (1 -
    1/order_hat) * epsilon_hat + (1 + eta) * log(1/gamma) / order_hat +
    log(E[K]) / (order - 1)``.

    Args:
        order: Renyi order ``lambda`` greater than one.
        epsilon: RDP epsilon of one candidate at ``order``.
        order_hat: Auxiliary order ``lambda_hat`` of at least one.
        epsilon_hat: RDP epsilon of one candidate at ``order_hat``.
        eta: Shape parameter in ``(-1, inf)``.
        gamma: Stopping parameter in ``(0, 1)``.

    Returns:
        The RDP epsilon of the selected output at ``order``.
    """

    order = _order(order)
    order_hat = _number(order_hat, "order_hat")
    if order_hat < 1.0:
        raise PrivateSelectionError("invalid_field", "order_hat")
    epsilon = _non_negative(epsilon, "epsilon")
    epsilon_hat = _non_negative(epsilon_hat, "epsilon_hat")
    mean = truncated_negative_binomial_mean(eta, gamma)
    eta = float(eta)
    return (
        epsilon
        + (1.0 + eta) * (1.0 - 1.0 / order_hat) * epsilon_hat
        + (1.0 + eta) * math.log(1.0 / gamma) / order_hat
        + math.log(mean) / (order - 1.0)
    )


def truncated_negative_binomial_zcdp_rdp(
    order: float, rho: float, *, eta: float, gamma: float
) -> float:
    """Return Corollary 4 of Papernot and Steinke (ICLR 2022).

    For a ``rho``-zCDP candidate with ``rho <= log(1/gamma)``, best-of-``K``
    with ``K ~ D_{eta,gamma}`` is ``(order, epsilon')``-RDP where
    ``epsilon' = rho * (order - 1) + log(E[K]) / (order - 1) + 2 * (1 + eta)
    * sqrt(rho * log(1/gamma)) - eta * rho`` above the order ``1 +
    sqrt(log(E[K]) / rho)`` and ``2 * sqrt(rho * log(E[K])) + 2 * (1 + eta) *
    sqrt(rho * log(1/gamma)) - eta * rho`` at or below it.

    Args:
        order: Renyi order greater than one.
        rho: zCDP parameter of one candidate, greater than zero.
        eta: Shape parameter in ``(-1, inf)``.
        gamma: Stopping parameter in ``(0, 1)``.

    Returns:
        The RDP epsilon of the selected output at ``order``.

    Raises:
        PrivateSelectionError: If ``rho > log(1/gamma)``.
    """

    order = _order(order)
    rho = _positive(rho, "rho")
    eta = _eta(eta)
    gamma = _open_unit(gamma, "gamma")
    if rho > math.log(1.0 / gamma):
        raise PrivateSelectionError("bound_not_applicable", "rho")
    log_mean = math.log(truncated_negative_binomial_mean(eta, gamma))
    tail = 2.0 * (1.0 + eta) * math.sqrt(rho * math.log(1.0 / gamma)) - eta * rho
    if order > 1.0 + math.sqrt(log_mean / rho):
        return rho * (order - 1.0) + log_mean / (order - 1.0) + tail
    return 2.0 * math.sqrt(rho * log_mean) + tail


def poisson_rdp(
    order: float,
    epsilon: float,
    epsilon_hat: float,
    delta_hat: float,
    *,
    mu: float,
) -> float:
    """Return Theorem 6 of Papernot and Steinke (ICLR 2022).

    For a candidate that is ``(order, epsilon)``-RDP and
    ``(epsilon_hat, delta_hat)``-DP with ``exp(epsilon_hat) <= 1 + 1 / (order
    - 1)``, best-of-``K`` with ``K ~ Poisson(mu)`` is ``(order, epsilon +
    mu * delta_hat + log(mu) / (order - 1))``-RDP.

    Args:
        order: Renyi order greater than one.
        epsilon: RDP epsilon of one candidate at ``order``.
        epsilon_hat: Approximate-DP epsilon of one candidate.
        delta_hat: Approximate-DP delta of one candidate.
        mu: Poisson mean, greater than zero.

    Returns:
        The RDP epsilon of the selected output at ``order``.

    Raises:
        PrivateSelectionError: If the theorem's order condition fails.
    """

    order = _order(order)
    epsilon = _non_negative(epsilon, "epsilon")
    epsilon_hat = _non_negative(epsilon_hat, "epsilon_hat")
    delta_hat = _delta(delta_hat, "delta_hat")
    mu = _positive(mu, "mu")
    if epsilon_hat > math.log1p(1.0 / (order - 1.0)):
        raise PrivateSelectionError("bound_not_applicable", "order")
    return epsilon + mu * delta_hat + math.log(mu) / (order - 1.0)


def rdp_to_dp_epsilon(curve: Iterable[tuple[float, float]], delta: float) -> float:
    """Convert an RDP curve to approximate-DP epsilon at ``delta``.

    Mironov (CSF 2017), Proposition 3: ``(order, epsilon)``-RDP implies
    ``(epsilon + log(1/delta) / (order - 1), delta)``-DP. The minimum over the
    curve is returned, floored at zero.

    Args:
        curve: ``(order, epsilon)`` pairs.
        delta: Target delta in ``(0, 1)``.

    Returns:
        The approximate-DP epsilon.
    """

    delta = _open_unit(delta, "delta")
    pairs = []
    for pair in curve:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise PrivateSelectionError("invalid_field", "rdp")
        pairs.append((_order(pair[0], "rdp"), _number(pair[1], "rdp")))
    if not pairs:
        raise PrivateSelectionError("bound_not_applicable", "rdp")
    log_term = math.log(1.0 / delta)
    return max(0.0, min(value + log_term / (order - 1.0) for order, value in pairs))


# ---------------------------------------------------------------------------
# Random-stopping schemes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionBound:
    """Approximate-DP charge for a declared random-stopping scheme.

    Attributes:
        epsilon: Charged epsilon.
        delta: Charged delta.
        method: Accounting method code from :data:`ACCOUNTING_METHODS`.
    """

    epsilon: float
    delta: float
    method: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {"epsilon": self.epsilon, "delta": self.delta, "method": self.method}


def _check_csprng(rng: Any) -> random.SystemRandom:
    if rng is None:
        return random.SystemRandom()
    if not isinstance(rng, random.SystemRandom):
        raise PrivateSelectionError("rng_not_csprng")
    return rng


@dataclass(frozen=True)
class RandomStoppingScheme:
    """Declared distribution of the number of trials and per-trial guarantee.

    The bound applies only when the number of trials is drawn from the
    declared distribution, every trial is an independent run of the same
    candidate-generating procedure whose guarantee is dominated by
    ``per_trial`` (training plus every private metric evaluation in that
    trial), and only the selected output is released: the realized trial
    count and the outputs of other trials stay inside the ledger.

    Attributes:
        distribution: ``truncated_negative_binomial``, ``geometric``
            (``eta = 1``), ``logarithmic`` (``eta = 0``) or ``poisson``.
        per_trial: Declared guarantee of one trial.
        eta: Truncated negative binomial shape parameter.
        gamma: Truncated negative binomial stopping parameter.
        mu: Poisson mean.
        target_delta: Delta used to convert RDP bounds to approximate DP.
    """

    distribution: str
    per_trial: TrialPrivacyCost
    eta: float | None = None
    gamma: float | None = None
    mu: float | None = None
    target_delta: float | None = None
    _bound: SelectionBound = field(init=False, repr=False, compare=False)
    _pmf: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        distribution = self.distribution
        if type(distribution) is not str:
            raise PrivateSelectionError("unsupported_distribution")
        if distribution in _POINT_MASS_DISTRIBUTIONS:
            raise PrivateSelectionError("fixed_trial_count")
        if distribution not in STOPPING_DISTRIBUTIONS:
            raise PrivateSelectionError("unsupported_distribution")
        if type(self.per_trial) is not TrialPrivacyCost:
            raise PrivateSelectionError("invalid_field", "per_trial")
        if self.target_delta is not None:
            object.__setattr__(
                self, "target_delta", _open_unit(self.target_delta, "target_delta")
            )
        if distribution == "poisson":
            if self.eta is not None or self.gamma is not None:
                raise PrivateSelectionError("invalid_field", "eta")
            mu = _positive(self.mu, "mu")
            if mu > MAX_TUNING_TRIALS:
                raise PrivateSelectionError("scheme_tail_too_heavy")
            object.__setattr__(self, "mu", mu)
            pmf = _poisson_pmf(mu)
        else:
            if self.mu is not None:
                raise PrivateSelectionError("invalid_field", "mu")
            implied = {"geometric": 1.0, "logarithmic": 0.0}.get(distribution)
            if implied is not None:
                if self.eta is not None and _number(self.eta, "eta") != implied:
                    raise PrivateSelectionError("invalid_field", "eta")
                eta = implied
            else:
                if self.eta is None:
                    raise PrivateSelectionError("invalid_field", "eta")
                eta = _eta(self.eta)
            object.__setattr__(self, "eta", eta)
            object.__setattr__(self, "gamma", _open_unit(self.gamma, "gamma"))
            pmf = _tnb_pmf(eta, float(self.gamma))
        if 1.0 - math.fsum(pmf) > MAX_SCHEME_TAIL_MASS:
            raise PrivateSelectionError("scheme_tail_too_heavy")
        object.__setattr__(self, "_pmf", pmf)
        object.__setattr__(self, "_bound", self._compute_bound())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RandomStoppingScheme:
        """Build a scheme from a closed mapping.

        Args:
            payload: Mapping with ``distribution``, ``per_trial`` and the
                distribution parameters. A fixed trial count is refused.

        Returns:
            The validated scheme.

        Raises:
            PrivateSelectionError: With ``fixed_trial_count`` when a fixed
                trial count is declared, or ``invalid_field`` for unknown keys.
        """

        values = _mapping_items(payload, "scheme", 16)
        if set(values) & _FIXED_COUNT_KEYS:
            raise PrivateSelectionError("fixed_trial_count")
        if set(values) - _SCHEME_FIELDS or "per_trial" not in values:
            raise PrivateSelectionError("invalid_field", "scheme")
        per_trial = values["per_trial"]
        if not isinstance(per_trial, TrialPrivacyCost):
            per_trial = TrialPrivacyCost.from_mapping(per_trial)
        return cls(
            distribution=values.get("distribution"),  # type: ignore[arg-type]
            per_trial=per_trial,
            eta=values.get("eta"),
            gamma=values.get("gamma"),
            mu=values.get("mu"),
            target_delta=values.get("target_delta"),
        )

    @property
    def expected_trial_count(self) -> float:
        """Return the expected number of trials of the declared distribution."""

        if self.distribution == "poisson":
            return float(self.mu)  # type: ignore[arg-type]
        return truncated_negative_binomial_mean(
            float(self.eta),  # type: ignore[arg-type]
            float(self.gamma),  # type: ignore[arg-type]
        )

    def bound(self) -> SelectionBound:
        """Return the approximate-DP charge of the selected output.

        The charge depends only on the declared scheme, never on the realized
        trial count.
        """

        return self._bound

    def draw_trial_count(self, rng: random.SystemRandom | None = None) -> int:
        """Draw the number of trials by inverse CDF with a CSPRNG.

        Args:
            rng: Injected ``random.SystemRandom`` instance. ``None`` uses a
                fresh operating-system generator. Seedable generators are
                refused.

        Returns:
            The number of trials to run.

        Raises:
            PrivateSelectionError: ``rng_not_csprng`` for other generators and
                ``trial_count_limit`` if the draw exceeds the trial limit.
        """

        generator = _check_csprng(rng)
        uniform = generator.random()
        if isinstance(uniform, bool) or not isinstance(uniform, float):
            raise PrivateSelectionError("rng_not_csprng")
        if not 0.0 <= uniform < 1.0:
            raise PrivateSelectionError("rng_not_csprng")
        offset = 0 if self.distribution == "poisson" else 1
        cumulative = 0.0
        for index, probability in enumerate(self._pmf):
            cumulative += probability
            if uniform < cumulative:
                return index + offset
        raise PrivateSelectionError("trial_count_limit")

    def to_dict(self) -> dict[str, Any]:
        """Return the declared scheme and its charge."""

        return {
            "distribution": self.distribution,
            "eta": self.eta,
            "gamma": self.gamma,
            "mu": self.mu,
            "target_delta": self.target_delta,
            "expected_trial_count": self.expected_trial_count,
            "per_trial": self.per_trial.to_dict(),
            "bound": self._bound.to_dict(),
        }

    def _compute_bound(self) -> SelectionBound:
        cost = self.per_trial
        if self.distribution == "poisson":
            if self.target_delta is None:
                raise PrivateSelectionError("bound_not_applicable", "target_delta")
            orders = [order for order, _ in cost.rdp] or list(DEFAULT_RDP_ORDERS)
            curve = []
            for order in orders:
                value = cost.rdp_at(order)
                if value is None or cost.epsilon > math.log1p(1.0 / (order - 1.0)):
                    continue
                curve.append(
                    (
                        order,
                        poisson_rdp(
                            order,
                            value,
                            cost.epsilon,
                            cost.delta,
                            mu=float(self.mu),  # type: ignore[arg-type]
                        ),
                    )
                )
            if not curve:
                raise PrivateSelectionError("bound_not_applicable", "per_trial")
            epsilon = rdp_to_dp_epsilon(curve, self.target_delta)
            return SelectionBound(epsilon, self.target_delta, "poisson_rdp")
        eta = float(self.eta)  # type: ignore[arg-type]
        gamma = float(self.gamma)  # type: ignore[arg-type]
        if cost.rdp:
            if self.target_delta is None:
                raise PrivateSelectionError("bound_not_applicable", "target_delta")
            orders = [order for order, _ in cost.rdp]
            curve = []
            for order in orders:
                value = float(cost.rdp_at(order))  # type: ignore[arg-type]
                best = min(
                    truncated_negative_binomial_rdp(
                        order,
                        value,
                        hat,
                        float(cost.rdp_at(hat)),  # type: ignore[arg-type]
                        eta=eta,
                        gamma=gamma,
                    )
                    for hat in orders
                )
                curve.append((order, best))
            epsilon = rdp_to_dp_epsilon(curve, self.target_delta)
            return SelectionBound(epsilon, self.target_delta, "tnb_rdp")
        if cost.delta == 0.0:
            epsilon = random_stopping_pure_dp_epsilon(cost.epsilon, eta)
            return SelectionBound(epsilon, 0.0, "tnb_pure_dp")
        raise PrivateSelectionError("bound_not_applicable", "per_trial")


def _tnb_pmf(eta: float, gamma: float) -> tuple[float, ...]:
    """Return ``P[K = k]`` for ``k = 1..MAX_TUNING_TRIALS`` (Definition 1)."""

    if eta == 0.0:
        probability = (1.0 - gamma) / math.log(1.0 / gamma)
    else:
        probability = (1.0 - gamma) * eta / math.expm1(-eta * math.log(gamma))
    values = [probability]
    for k in range(1, MAX_TUNING_TRIALS):
        probability = probability * (1.0 - gamma) * (k + eta) / (k + 1)
        values.append(probability)
    return tuple(values)


def _poisson_pmf(mu: float) -> tuple[float, ...]:
    """Return ``P[K = k]`` for ``k = 0..MAX_TUNING_TRIALS``."""

    probability = math.exp(-mu)
    values = [probability]
    for k in range(1, MAX_TUNING_TRIALS + 1):
        probability = probability * mu / k
        values.append(probability)
    return tuple(values)


# ---------------------------------------------------------------------------
# Tuning ledger.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuningTrialRecord:
    """One candidate run in a tuning program.

    Attributes:
        trial_index: One-based trial position.
        mechanism: Numeric mechanism parameters.
        lineage_digest: Opaque ``sha256:`` lineage reference.
        status: ``completed`` or ``abandoned``.
        outcome_band: Controlled outcome band; never a metric value.
        training_cost: Guarantee returned by the injected accountant.
        selection_evaluations: In-trial private selection evaluations by kind.
    """

    trial_index: int
    mechanism: TrialMechanism
    lineage_digest: str
    status: str
    outcome_band: str
    training_cost: TrialPrivacyCost
    selection_evaluations: tuple[tuple[str, int], ...]

    @property
    def evaluation_count(self) -> int:
        """Return the number of in-trial selection evaluations."""

        return sum(count for _, count in self.selection_evaluations)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record with codes, numbers and digests."""

        return {
            "trial_index": self.trial_index,
            "lineage_digest": self.lineage_digest,
            "status": self.status,
            "outcome_band": self.outcome_band,
            "mechanism": self.mechanism.to_dict(),
            "training_cost": self.training_cost.to_dict(),
            "selection_evaluations": [
                [kind, count] for kind, count in self.selection_evaluations
            ],
        }


@dataclass(frozen=True)
class TuningAccount:
    """Composed privacy account of a tuning program.

    Under a random-stopping scheme the counts are ``None``: the realized
    number of trials must not be released alongside the selected output.

    Attributes:
        accounting: ``composition`` or ``random_stopping``.
        method: Accounting method code.
        epsilon: Charged epsilon.
        delta: Charged delta.
        trial_count: Number of trials (composition only).
        evaluated_candidate_count: Candidates with a recorded outcome band.
        private_evaluation_count: Private metric evaluations charged.
        selection_counts: Selection events by kind (composition only).
    """

    accounting: str
    method: str
    epsilon: float
    delta: float
    trial_count: int | None = field(default=None, repr=False)
    evaluated_candidate_count: int | None = field(default=None, repr=False)
    private_evaluation_count: int | None = field(default=None, repr=False)
    selection_counts: tuple[tuple[str, int], ...] | None = field(
        default=None, repr=False
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible account, omitting unavailable counts."""

        payload: dict[str, Any] = {
            "accounting": self.accounting,
            "method": self.method,
            "epsilon": self.epsilon,
            "delta": self.delta,
        }
        if self.trial_count is not None:
            payload["trial_count"] = self.trial_count
            payload["evaluated_candidate_count"] = self.evaluated_candidate_count
            payload["private_evaluation_count"] = self.private_evaluation_count
            payload["selection_counts"] = [
                [kind, count] for kind, count in self.selection_counts or ()
            ]
        return payload


@dataclass(frozen=True)
class TuningPromotionEvidence:
    """Promotion evidence citing the composed account of every trial.

    Attributes:
        ledger_digest: Digest committing to every trial record.
        accounting: ``composition`` or ``random_stopping``.
        method: Accounting method code.
        epsilon: Charged epsilon.
        delta: Charged delta.
        coverage: Always ``all_trials``.
        selected_lineage_digest: Lineage digest of the selected trial.
        trial_count: Number of trials (composition only).
        schema_version: Evidence schema version.
    """

    ledger_digest: str
    accounting: str
    method: str
    epsilon: float
    delta: float
    coverage: str
    selected_lineage_digest: str
    trial_count: int | None = None
    schema_version: str = PRIVATE_SELECTION_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TuningPromotionEvidence:
        """Parse evidence from a closed mapping.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The parsed evidence.

        Raises:
            PrivateSelectionError: If the mapping does not cite a tuning
                ledger (``evidence_selected_trial_only``) or is malformed
                (``evidence_invalid``).
        """

        values = _mapping_items(payload, "evidence", len(_EVIDENCE_FIELDS))
        if "ledger_digest" not in values and "coverage" not in values:
            raise PrivateSelectionError("evidence_selected_trial_only")
        if set(values) - _EVIDENCE_FIELDS:
            raise PrivateSelectionError("evidence_invalid")
        required = _EVIDENCE_FIELDS - {"trial_count", "schema_version"}
        if required - set(values):
            raise PrivateSelectionError("evidence_invalid")
        try:
            epsilon = _non_negative(values["epsilon"], "epsilon")
            delta = _delta(values["delta"], "delta")
            trial_count = values.get("trial_count")
            if trial_count is not None:
                trial_count = _bounded_int(trial_count, "trial_count", 0, 1 << 31)
        except PrivateSelectionError:
            raise PrivateSelectionError("evidence_invalid") from None
        texts = ("ledger_digest", "accounting", "method", "coverage")
        if any(type(values[key]) is not str for key in texts):
            raise PrivateSelectionError("evidence_invalid")
        if type(values["selected_lineage_digest"]) is not str:
            raise PrivateSelectionError("evidence_invalid")
        version = values.get("schema_version", PRIVATE_SELECTION_SCHEMA_VERSION)
        if version != PRIVATE_SELECTION_SCHEMA_VERSION:
            raise PrivateSelectionError("evidence_invalid")
        return cls(
            ledger_digest=values["ledger_digest"],
            accounting=values["accounting"],
            method=values["method"],
            epsilon=epsilon,
            delta=delta,
            coverage=values["coverage"],
            selected_lineage_digest=values["selected_lineage_digest"],
            trial_count=trial_count,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible evidence payload."""

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "ledger_digest": self.ledger_digest,
            "accounting": self.accounting,
            "method": self.method,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "coverage": self.coverage,
            "selected_lineage_digest": self.selected_lineage_digest,
        }
        if self.trial_count is not None:
            payload["trial_count"] = self.trial_count
        return payload


class TuningLedger:
    """Ledger of candidate runs and private selection in one tuning program.

    Args:
        accountant: Injected per-trial accountant mapping
            :class:`TrialMechanism` to :class:`TrialPrivacyCost`.
        metric_source: ``private`` when selection uses private validation
            metrics, ``public`` when it uses public data only.
        scoring_cost: Guarantee of one private metric evaluation. Required and
            positive for ``private``; must be omitted for ``public``.
        scheme: Optional declared random-stopping scheme. Without one, the
            basic composition of every trial and evaluation is charged.
    """

    def __init__(
        self,
        accountant: TrialAccountant,
        *,
        metric_source: str = "private",
        scoring_cost: TrialPrivacyCost | None = None,
        scheme: RandomStoppingScheme | None = None,
    ) -> None:
        if not callable(accountant):
            raise PrivateSelectionError("invalid_field", "accountant")
        if metric_source not in METRIC_SOURCES:
            raise PrivateSelectionError("invalid_field", "metric_source")
        if metric_source == "private":
            if type(scoring_cost) is not TrialPrivacyCost or scoring_cost.epsilon <= 0:
                raise PrivateSelectionError("scoring_cost_missing")
        elif scoring_cost is not None:
            raise PrivateSelectionError("invalid_field", "scoring_cost")
        if scheme is not None and type(scheme) is not RandomStoppingScheme:
            raise PrivateSelectionError("invalid_field", "scheme")
        self._accountant = accountant
        self._metric_source = metric_source
        self._scoring_cost = scoring_cost
        self._scheme = scheme
        self._lock = RLock()
        self._records: list[TuningTrialRecord] = []
        self._drawn_trial_count: int | None = None
        self._selected_index: int | None = None
        self._charged: TuningAccount | None = None

    def __repr__(self) -> str:
        accounting = "composition" if self._scheme is None else "random_stopping"
        return (
            f"TuningLedger(accounting={accounting}, "
            f"metric_source={self._metric_source}, "
            f"charged={self._charged is not None})"
        )

    @property
    def records(self) -> tuple[TuningTrialRecord, ...]:
        """Return in-process trial records in trial order.

        Under a random-stopping scheme these records reveal the realized trial
        count and must not be released with the selected output.
        """

        with self._lock:
            return tuple(self._records)

    @property
    def scheme(self) -> RandomStoppingScheme | None:
        """Return the declared random-stopping scheme, if any."""

        return self._scheme

    @property
    def digest(self) -> str:
        """Return a digest committing to the configuration and every record."""

        with self._lock:
            return _sha256(self._digest_payload())

    def draw_trial_count(self, rng: random.SystemRandom | None = None) -> int:
        """Draw the number of trials once from the declared scheme.

        Args:
            rng: Injected ``random.SystemRandom`` instance; ``None`` uses the
                operating-system generator. The generator is not retained.

        Returns:
            The number of trials the program must run.

        Raises:
            PrivateSelectionError: If no scheme is declared, the count was
                already drawn, or the generator is not a CSPRNG.
        """

        if self._scheme is None:
            raise PrivateSelectionError("scheme_not_declared")
        with self._lock:
            if self._drawn_trial_count is not None:
                raise PrivateSelectionError("trial_count_already_drawn")
            count = self._scheme.draw_trial_count(rng)
            self._drawn_trial_count = count
            return count

    def record_trial(
        self,
        mechanism: TrialMechanism | Mapping[str, Any],
        lineage_digest: str,
        *,
        status: str = "completed",
        outcome_band: str = "not_evaluated",
    ) -> TuningTrialRecord:
        """Record and account one candidate run, completed or abandoned.

        Args:
            mechanism: Mechanism parameters passed to the accountant.
            lineage_digest: Opaque ``sha256:`` lineage reference.
            status: ``completed`` or ``abandoned``.
            outcome_band: Code from :data:`OUTCOME_BANDS`. Any band other
                than ``not_evaluated`` records one candidate evaluation.

        Returns:
            The appended record.

        Raises:
            PrivateSelectionError: If validation, the accountant, the drawn
                trial count or the declared per-trial guarantee refuses it.
        """

        if not isinstance(mechanism, TrialMechanism):
            mechanism = TrialMechanism.from_mapping(mechanism)
        if type(lineage_digest) is not str or not _DIGEST_RE.match(lineage_digest):
            raise PrivateSelectionError("lineage_digest_invalid")
        if status not in TRIAL_STATUSES:
            raise PrivateSelectionError("invalid_field", "status")
        if type(outcome_band) is not str or outcome_band not in OUTCOME_BANDS:
            raise PrivateSelectionError("outcome_not_banded")
        with self._lock:
            self._require_open()
            if self._scheme is not None:
                if self._drawn_trial_count is None:
                    raise PrivateSelectionError("trial_count_not_drawn")
                if len(self._records) >= self._drawn_trial_count:
                    raise PrivateSelectionError("trial_count_exceeded")
            elif len(self._records) >= MAX_TUNING_TRIALS:
                raise PrivateSelectionError("trial_limit")
            if any(r.lineage_digest == lineage_digest for r in self._records):
                raise PrivateSelectionError("lineage_digest_reused")
            try:
                cost = self._accountant(mechanism)
            except Exception:  # noqa: BLE001 - the accountant is injected.
                raise PrivateSelectionError("accountant_failed") from None
            if type(cost) is not TrialPrivacyCost:
                raise PrivateSelectionError("accountant_invalid")
            record = TuningTrialRecord(
                trial_index=len(self._records) + 1,
                mechanism=mechanism,
                lineage_digest=lineage_digest,
                status=status,
                outcome_band=outcome_band,
                training_cost=cost,
                selection_evaluations=tuple(
                    (kind, 0) for kind in IN_TRIAL_SELECTION_KINDS
                ),
            )
            self._check_declared(record)
            self._records.append(record)
            return record

    def record_selection(
        self, trial_index: int, kind: str, *, evaluation_count: int = 1
    ) -> TuningTrialRecord:
        """Record early stopping, checkpoint or round selection in one trial.

        Each evaluation on a private metric is charged at ``scoring_cost``.

        Args:
            trial_index: One-based trial index.
            kind: ``early_stopping``, ``checkpoint_selection`` or
                ``round_selection``. Best-of-k is recorded by :meth:`select`.
            evaluation_count: Number of metric evaluations used.

        Returns:
            The updated record.
        """

        if kind not in IN_TRIAL_SELECTION_KINDS:
            raise PrivateSelectionError("selection_kind_invalid")
        count = _bounded_int(
            evaluation_count, "evaluation_count", 1, MAX_TRIAL_EVALUATIONS
        )
        with self._lock:
            self._require_open()
            position = self._position(trial_index)
            record = self._records[position]
            if record.evaluation_count + count > MAX_TRIAL_EVALUATIONS:
                raise PrivateSelectionError("invalid_field", "evaluation_count")
            updated = replace(
                record,
                selection_evaluations=tuple(
                    (name, value + count if name == kind else value)
                    for name, value in record.selection_evaluations
                ),
            )
            self._check_declared(updated)
            self._records[position] = updated
            return updated

    def select(self, trial_index: int) -> TuningTrialRecord:
        """Record best-of-k selection of one completed, evaluated trial.

        Args:
            trial_index: One-based index of the selected trial.

        Returns:
            The selected record.
        """

        with self._lock:
            self._require_open()
            if self._selected_index is not None:
                raise PrivateSelectionError("already_selected")
            position = self._position(trial_index)
            record = self._records[position]
            if record.status != "completed" or record.outcome_band == "not_evaluated":
                raise PrivateSelectionError("selected_trial_not_eligible")
            self._require_complete()
            self._selected_index = record.trial_index
            return record

    def account(self) -> TuningAccount:
        """Return the composed account of every trial plus selection.

        Returns:
            The random-stopping bound when a scheme is declared, otherwise the
            basic composition of every trial and private evaluation.
        """

        with self._lock:
            if self._charged is not None:
                return self._charged
            return self._account_unlocked()

    def charge(
        self, budget_ledger: PrivacyBudgetLedger, context: str
    ) -> PrivacyBudgetDecision:
        """Charge the composed account once to a shared privacy-budget ledger.

        After charging, no trial, selection event or selection may be added.

        Args:
            budget_ledger: Shared :class:`PrivacyBudgetLedger`.
            context: Registered release context to charge.

        Returns:
            The accepted budget decision.

        Raises:
            PrivateSelectionError: ``budget_exceeded`` without mutating either
                ledger when the account does not fit.
        """

        if not isinstance(budget_ledger, PrivacyBudgetLedger):
            raise PrivateSelectionError("invalid_field", "budget_ledger")
        with self._lock:
            if self._charged is not None:
                raise PrivateSelectionError("already_charged")
            account = self._account_unlocked()
            try:
                decision = budget_ledger.record_release(
                    context, account.epsilon, account.delta
                )
            except PrivacyBudgetLedgerExceeded:
                raise PrivateSelectionError("budget_exceeded") from None
            except KeyError:
                raise PrivateSelectionError("budget_context_unknown") from None
            except (TypeError, ValueError, OverflowError):
                raise PrivateSelectionError("invalid_field", "context") from None
            self._charged = account
            return decision

    def promotion_evidence(self) -> TuningPromotionEvidence:
        """Return promotion evidence citing the charged account of all trials.

        Raises:
            PrivateSelectionError: ``not_charged`` before :meth:`charge` and
                ``no_selected_trial`` without a selection.
        """

        with self._lock:
            if self._charged is None:
                raise PrivateSelectionError("not_charged")
            if self._selected_index is None:
                raise PrivateSelectionError("no_selected_trial")
            account = self._charged
            selected = self._records[self._selected_index - 1]
            return TuningPromotionEvidence(
                ledger_digest=_sha256(self._digest_payload()),
                accounting=account.accounting,
                method=account.method,
                epsilon=account.epsilon,
                delta=account.delta,
                coverage=EVIDENCE_COVERAGE,
                selected_lineage_digest=selected.lineage_digest,
                trial_count=account.trial_count,
            )

    def render(self) -> dict[str, Any]:
        """Return exportable evidence without per-trial records.

        Under a random-stopping scheme no trial or evaluation count is
        included.
        """

        with self._lock:
            payload: dict[str, Any] = {
                "schema_version": PRIVATE_SELECTION_SCHEMA_VERSION,
                "accounting": (
                    "composition" if self._scheme is None else "random_stopping"
                ),
                "metric_source": self._metric_source,
                "scheme": None if self._scheme is None else self._scheme.to_dict(),
                "charged": self._charged is not None,
                "selected": self._selected_index is not None,
                "ledger_digest": _sha256(self._digest_payload()),
                "account": (None if self._charged is None else self._charged.to_dict()),
            }
            if self._scheme is None:
                payload["trial_count"] = len(self._records)
                payload["selection_counts"] = [
                    [kind, count] for kind, count in self._selection_counts()
                ]
            return payload

    def to_dict(self) -> dict[str, Any]:
        """Return :meth:`render`."""

        return self.render()

    # -- internals ---------------------------------------------------------

    def _require_open(self) -> None:
        if self._charged is not None:
            raise PrivateSelectionError("ledger_closed")

    def _require_complete(self) -> None:
        if self._scheme is None:
            if not self._records:
                raise PrivateSelectionError("no_trials")
            return
        if self._drawn_trial_count is None:
            raise PrivateSelectionError("trial_count_not_drawn")
        if len(self._records) != self._drawn_trial_count:
            raise PrivateSelectionError("trial_count_incomplete")

    def _position(self, trial_index: Any) -> int:
        if (
            type(trial_index) is not int
            or trial_index < 1
            or trial_index > len(self._records)
        ):
            raise PrivateSelectionError("unknown_trial")
        return trial_index - 1

    def _private_evaluations(self, record: TuningTrialRecord) -> int:
        if self._metric_source != "private":
            return 0
        scored = 0 if record.outcome_band == "not_evaluated" else 1
        return record.evaluation_count + scored

    def _trial_cost(self, record: TuningTrialRecord) -> TrialPrivacyCost:
        costs = [(record.training_cost, 1)]
        if self._scoring_cost is not None:
            costs.append((self._scoring_cost, self._private_evaluations(record)))
        return _compose(costs)

    def _check_declared(self, record: TuningTrialRecord) -> None:
        if self._scheme is None:
            return
        if not _dominated(self._trial_cost(record), self._scheme.per_trial):
            raise PrivateSelectionError("trial_exceeds_declared_cost")

    def _selection_counts(self) -> tuple[tuple[str, int], ...]:
        counts = {kind: 0 for kind in SELECTION_KINDS}
        counts["best_of_k"] = 0 if self._selected_index is None else 1
        for record in self._records:
            for kind, count in record.selection_evaluations:
                counts[kind] += count
        return tuple(sorted(counts.items()))

    def _account_unlocked(self) -> TuningAccount:
        self._require_complete()
        if self._scheme is not None:
            bound = self._scheme.bound()
            return TuningAccount(
                accounting="random_stopping",
                method=bound.method,
                epsilon=bound.epsilon,
                delta=bound.delta,
            )
        evaluations = sum(self._private_evaluations(r) for r in self._records)
        costs = [(record.training_cost, 1) for record in self._records]
        if self._scoring_cost is not None:
            costs.append((self._scoring_cost, evaluations))
        composed = _compose(costs)
        return TuningAccount(
            accounting="composition",
            method="basic_composition",
            epsilon=composed.epsilon,
            delta=composed.delta,
            trial_count=len(self._records),
            evaluated_candidate_count=sum(
                1 for r in self._records if r.outcome_band != "not_evaluated"
            ),
            private_evaluation_count=evaluations,
            selection_counts=self._selection_counts(),
        )

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PRIVATE_SELECTION_SCHEMA_VERSION,
            "metric_source": self._metric_source,
            "scoring_cost": (
                None if self._scoring_cost is None else self._scoring_cost.to_dict()
            ),
            "scheme": None if self._scheme is None else self._scheme.to_dict(),
            "records": [record.to_dict() for record in self._records],
            "selected_index": self._selected_index,
        }


def verify_promotion_evidence(
    evidence: TuningPromotionEvidence | Mapping[str, Any], ledger: TuningLedger
) -> TuningAccount:
    """Check that promotion evidence cites the composed account of all trials.

    Args:
        evidence: Evidence object or mapping offered to a promotion gate.
        ledger: Charged tuning ledger the evidence claims to summarize.

    Returns:
        The charged account of the ledger.

    Raises:
        PrivateSelectionError: ``evidence_selected_trial_only`` when the
            evidence cites only the selected run, ``evidence_coverage_incomplete``
            when it covers fewer trials, ``evidence_ledger_mismatch`` when it
            does not match the ledger, ``evidence_account_understated`` when it
            reports less privacy loss and ``evidence_invalid`` when malformed.
    """

    if not isinstance(ledger, TuningLedger):
        raise PrivateSelectionError("invalid_field", "ledger")
    if not isinstance(evidence, TuningPromotionEvidence):
        evidence = TuningPromotionEvidence.from_mapping(evidence)
    expected = ledger.promotion_evidence()
    account = ledger.account()
    if evidence.coverage != EVIDENCE_COVERAGE:
        if evidence.coverage == "selected_trial":
            raise PrivateSelectionError("evidence_selected_trial_only")
        raise PrivateSelectionError("evidence_coverage_incomplete")
    if expected.trial_count is not None:
        if evidence.trial_count is None:
            raise PrivateSelectionError("evidence_coverage_incomplete")
        if evidence.trial_count < expected.trial_count:
            if evidence.trial_count <= 1:
                raise PrivateSelectionError("evidence_selected_trial_only")
            raise PrivateSelectionError("evidence_coverage_incomplete")
    if (
        evidence.ledger_digest != expected.ledger_digest
        or evidence.selected_lineage_digest != expected.selected_lineage_digest
        or evidence.accounting != expected.accounting
        or evidence.method != expected.method
        or evidence.trial_count != expected.trial_count
        or evidence.schema_version != expected.schema_version
    ):
        raise PrivateSelectionError("evidence_ledger_mismatch")
    if evidence.epsilon < account.epsilon or evidence.delta < account.delta:
        raise PrivateSelectionError("evidence_account_understated")
    return account
