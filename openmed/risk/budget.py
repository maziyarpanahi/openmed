"""Per-document residual-risk, DP surrogate, and generation budget evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Literal, cast

from openmed.core.policy import CANONICAL_POLICY_NAMES, canonical_policy_name

_DIRECT_ID_COUNT_KEYS = (
    "surviving_direct_ids",
    "surviving_direct_id_count",
    "surviving_direct_ids_count",
    "surviving_direct_identifiers",
    "surviving_direct_identifier_count",
    "direct_identifier_count",
    "direct_identifiers",
)

DEFAULT_QI_WEIGHTS: Mapping[str, float] = {
    "age": 1.0,
    "date": 1.0,
    "geography": 1.5,
    "provider_institution": 2.0,
    "rare_condition": 3.0,
    "*": 1.0,
}

SurrogateDrawKind = Literal["categorical", "numeric", "date_offset"]

CompositionRule = Literal["basic", "advanced"]

EPSILON_POLICY_CONFIG_RESOURCE = "dp_epsilon_policies.json"
CURRENT_EPSILON_POLICY_SCHEMA_VERSION = 1

DEFAULT_RDP_ORDERS: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
DEFAULT_DP_SURROGATE_SENSITIVITIES: Mapping[
    tuple[str, SurrogateDrawKind],
    tuple[float, float],
] = {
    ("AGE", "numeric"): (1.0, 1.0),
    ("DATE", "date_offset"): (1.0, 1.0),
    ("EMAIL_ADDRESS", "categorical"): (1.0, 1.0),
    ("ID", "categorical"): (1.0, 1.0),
    ("LOCATION", "categorical"): (1.0, 1.0),
    ("MEDICAL_RECORD_NUMBER", "categorical"): (1.0, 1.0),
    ("NAME", "categorical"): (1.0, 1.0),
    ("ORGANIZATION", "categorical"): (1.0, 1.0),
    ("PERSON", "categorical"): (1.0, 1.0),
    ("PHONE_NUMBER", "categorical"): (1.0, 1.0),
}


@dataclass(frozen=True)
class DPSurrogateSensitivity:
    """Sensitivity metadata for one canonical surrogate draw label."""

    label: str
    draw_kind: SurrogateDrawKind
    l1: float = 1.0
    l2: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _canonical_dp_label(self.label))
        object.__setattr__(self, "draw_kind", _draw_kind(self.draw_kind))
        object.__setattr__(self, "l1", _positive_float(self.l1, field_name="l1"))
        object.__setattr__(self, "l2", _positive_float(self.l2, field_name="l2"))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible sensitivity payload."""

        return {
            "label": self.label,
            "draw_kind": self.draw_kind,
            "l1": self.l1,
            "l2": self.l2,
        }


@dataclass(frozen=True)
class DPSurrogateSensitivityRegistry:
    """Lookup table for per-label DP surrogate draw sensitivities."""

    sensitivities: Mapping[
        tuple[str, SurrogateDrawKind],
        DPSurrogateSensitivity,
    ] = field(default_factory=dict)
    default_l1: float = 1.0
    default_l2: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sensitivities",
            {
                (_canonical_dp_label(label), _draw_kind(draw_kind)): (
                    sensitivity
                    if isinstance(sensitivity, DPSurrogateSensitivity)
                    else DPSurrogateSensitivity(
                        label=label,
                        draw_kind=draw_kind,
                        l1=sensitivity[0],
                        l2=sensitivity[1],
                    )
                )
                for (label, draw_kind), sensitivity in self.sensitivities.items()
            },
        )
        object.__setattr__(
            self,
            "default_l1",
            _positive_float(self.default_l1, field_name="default_l1"),
        )
        object.__setattr__(
            self,
            "default_l2",
            _positive_float(self.default_l2, field_name="default_l2"),
        )

    @classmethod
    def defaults(cls) -> DPSurrogateSensitivityRegistry:
        """Return the bundled registry for common PHI surrogate labels."""

        return cls(
            {
                key: DPSurrogateSensitivity(
                    label=key[0],
                    draw_kind=key[1],
                    l1=values[0],
                    l2=values[1],
                )
                for key, values in DEFAULT_DP_SURROGATE_SENSITIVITIES.items()
            }
        )

    def with_entry(
        self,
        label: str,
        draw_kind: SurrogateDrawKind,
        *,
        l1: float = 1.0,
        l2: float = 1.0,
    ) -> DPSurrogateSensitivityRegistry:
        """Return a copy with one sensitivity entry added or replaced."""

        sensitivity = DPSurrogateSensitivity(
            label=label,
            draw_kind=draw_kind,
            l1=l1,
            l2=l2,
        )
        entries = dict(self.sensitivities)
        entries[(sensitivity.label, sensitivity.draw_kind)] = sensitivity
        return DPSurrogateSensitivityRegistry(
            entries,
            default_l1=self.default_l1,
            default_l2=self.default_l2,
        )

    def for_label(
        self,
        label: str,
        draw_kind: SurrogateDrawKind,
    ) -> DPSurrogateSensitivity:
        """Return sensitivity metadata for ``label`` and ``draw_kind``."""

        canonical_label = _canonical_dp_label(label)
        canonical_kind = _draw_kind(draw_kind)
        sensitivity = self.sensitivities.get((canonical_label, canonical_kind))
        if sensitivity is not None:
            return sensitivity
        return DPSurrogateSensitivity(
            label=canonical_label,
            draw_kind=canonical_kind,
            l1=self.default_l1,
            l2=self.default_l2,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible registry payload."""

        return {
            "default_l1": self.default_l1,
            "default_l2": self.default_l2,
            "sensitivities": [
                self.sensitivities[key].to_dict() for key in sorted(self.sensitivities)
            ],
        }


@dataclass(frozen=True)
class DPSurrogateSpend:
    """One privacy spend made by a surrogate draw."""

    sequence: int
    label: str
    draw_kind: SurrogateDrawKind
    mechanism: str
    epsilon: float
    delta: float = 0.0
    l1_sensitivity: float = 1.0
    l2_sensitivity: float = 1.0
    rho: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence",
            _positive_int(self.sequence, field_name="sequence"),
        )
        object.__setattr__(self, "label", _canonical_dp_label(self.label))
        object.__setattr__(self, "draw_kind", _draw_kind(self.draw_kind))
        object.__setattr__(self, "mechanism", _non_empty_string(self.mechanism))
        object.__setattr__(
            self,
            "epsilon",
            _non_negative_float(self.epsilon, field_name="epsilon"),
        )
        object.__setattr__(
            self,
            "delta",
            _delta_float(self.delta, field_name="delta"),
        )
        object.__setattr__(
            self,
            "l1_sensitivity",
            _positive_float(self.l1_sensitivity, field_name="l1_sensitivity"),
        )
        object.__setattr__(
            self,
            "l2_sensitivity",
            _positive_float(self.l2_sensitivity, field_name="l2_sensitivity"),
        )
        rho = self.rho
        if rho is None:
            rho = _rho_from_epsilon(self.epsilon)
        object.__setattr__(self, "rho", _non_negative_float(rho, field_name="rho"))

    def spend_hash(self, *, salt: str = "") -> str:
        """Return a deterministic hash over non-PHI accounting metadata."""

        payload = {
            "delta": self.delta,
            "draw_kind": self.draw_kind,
            "epsilon": self.epsilon,
            "l1_sensitivity": self.l1_sensitivity,
            "l2_sensitivity": self.l2_sensitivity,
            "label": self.label,
            "mechanism": self.mechanism,
            "rho": self.rho,
            "salt": salt,
            "sequence": self.sequence,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self, *, salt: str = "") -> dict[str, Any]:
        """Return a deterministic JSON-compatible spend payload."""

        return {
            "sequence": self.sequence,
            "label": self.label,
            "draw_kind": self.draw_kind,
            "mechanism": self.mechanism,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "l1_sensitivity": self.l1_sensitivity,
            "l2_sensitivity": self.l2_sensitivity,
            "rho": self.rho,
            "spend_hash": self.spend_hash(salt=salt),
        }


@dataclass(frozen=True)
class DPSurrogateComposition:
    """Conservative composition totals for a DP surrogate run."""

    query_count: int
    basic_epsilon: float
    basic_delta: float
    zcdp_rho: float
    zcdp_epsilon: float
    rdp_epsilon: float
    reported_epsilon: float
    reported_delta: float
    target_epsilon: float
    target_delta: float
    remaining_epsilon: float
    remaining_delta: float

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible composition payload."""

        return {
            "query_count": self.query_count,
            "basic_epsilon": self.basic_epsilon,
            "basic_delta": self.basic_delta,
            "zcdp_rho": self.zcdp_rho,
            "zcdp_epsilon": self.zcdp_epsilon,
            "rdp_epsilon": self.rdp_epsilon,
            "reported_epsilon": self.reported_epsilon,
            "reported_delta": self.reported_delta,
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "remaining_epsilon": self.remaining_epsilon,
            "remaining_delta": self.remaining_delta,
        }


class DPSurrogateBudgetExceeded(ValueError):
    """Raised when a surrogate draw would exceed the configured DP budget."""

    def __init__(
        self,
        attempted_spend: DPSurrogateSpend,
        attempted_composition: DPSurrogateComposition,
    ) -> None:
        self.attempted_spend = attempted_spend
        self.attempted_composition = attempted_composition
        super().__init__(
            "DP surrogate budget exceeded: "
            f"epsilon={attempted_composition.reported_epsilon:.6g}/"
            f"{attempted_composition.target_epsilon:.6g}, "
            f"delta={attempted_composition.basic_delta:.6g}/"
            f"{attempted_composition.target_delta:.6g}"
        )


@dataclass
class DPSurrogateBudget:
    """Corpus-level DP accountant for surrogate generation."""

    target_epsilon: float
    target_delta: float
    sensitivity_registry: DPSurrogateSensitivityRegistry = field(
        default_factory=DPSurrogateSensitivityRegistry.defaults
    )
    rdp_orders: Sequence[float] = DEFAULT_RDP_ORDERS
    _spends: list[DPSurrogateSpend] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.target_epsilon = _positive_float(
            self.target_epsilon,
            field_name="target_epsilon",
        )
        self.target_delta = _delta_float(
            self.target_delta,
            field_name="target_delta",
        )
        if not isinstance(
            self.sensitivity_registry,
            DPSurrogateSensitivityRegistry,
        ):
            self.sensitivity_registry = DPSurrogateSensitivityRegistry(
                self.sensitivity_registry
            )
        self.rdp_orders = tuple(_rdp_order(order) for order in self.rdp_orders)
        if not self.rdp_orders:
            raise ValueError("rdp_orders must not be empty")
        self._spends = list(self._spends)

    @property
    def spends(self) -> tuple[DPSurrogateSpend, ...]:
        """Return spend records in draw order."""

        return tuple(self._spends)

    def spend(
        self,
        *,
        label: str,
        draw_kind: SurrogateDrawKind,
        mechanism: str,
        epsilon: float,
        delta: float = 0.0,
        rho: float | None = None,
    ) -> DPSurrogateSpend:
        """Record one surrogate draw spend or raise before mutating state."""

        sensitivity = self.sensitivity_registry.for_label(label, draw_kind)
        spend = DPSurrogateSpend(
            sequence=len(self._spends) + 1,
            label=sensitivity.label,
            draw_kind=sensitivity.draw_kind,
            mechanism=mechanism,
            epsilon=epsilon,
            delta=delta,
            l1_sensitivity=sensitivity.l1,
            l2_sensitivity=sensitivity.l2,
            rho=rho,
        )
        attempted = self.composition(spends=(*self._spends, spend))
        if _exceeds_budget(attempted):
            raise DPSurrogateBudgetExceeded(spend, attempted)
        self._spends.append(spend)
        return spend

    def can_spend(
        self,
        *,
        label: str,
        draw_kind: SurrogateDrawKind,
        mechanism: str,
        epsilon: float,
        delta: float = 0.0,
        rho: float | None = None,
    ) -> bool:
        """Return whether a spend would fit without recording it."""

        try:
            sensitivity = self.sensitivity_registry.for_label(label, draw_kind)
            spend = DPSurrogateSpend(
                sequence=len(self._spends) + 1,
                label=sensitivity.label,
                draw_kind=sensitivity.draw_kind,
                mechanism=mechanism,
                epsilon=epsilon,
                delta=delta,
                l1_sensitivity=sensitivity.l1,
                l2_sensitivity=sensitivity.l2,
                rho=rho,
            )
            return not _exceeds_budget(self.composition(spends=(*self._spends, spend)))
        except (TypeError, ValueError):
            return False

    def composition(
        self,
        *,
        spends: Sequence[DPSurrogateSpend] | None = None,
    ) -> DPSurrogateComposition:
        """Return conservative basic, zCDP, and RDP composition totals."""

        selected_spends = tuple(spends if spends is not None else self._spends)
        basic_epsilon = math.fsum(spend.epsilon for spend in selected_spends)
        basic_delta = math.fsum(spend.delta for spend in selected_spends)
        rho = math.fsum(spend.rho or 0.0 for spend in selected_spends)
        conversion_delta = self.target_delta - basic_delta
        zcdp_epsilon = _zcdp_epsilon(rho, conversion_delta)
        rdp_epsilon = _rdp_epsilon(rho, conversion_delta, self.rdp_orders)
        reported_epsilon = max(basic_epsilon, zcdp_epsilon, rdp_epsilon)
        reported_delta = (
            self.target_delta
            if rho > 0.0 and conversion_delta > 0.0
            else max(0.0, basic_delta)
        )
        return DPSurrogateComposition(
            query_count=len(selected_spends),
            basic_epsilon=basic_epsilon,
            basic_delta=basic_delta,
            zcdp_rho=rho,
            zcdp_epsilon=zcdp_epsilon,
            rdp_epsilon=rdp_epsilon,
            reported_epsilon=reported_epsilon,
            reported_delta=reported_delta,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            remaining_epsilon=self.target_epsilon - reported_epsilon,
            remaining_delta=self.target_delta - basic_delta,
        )

    def to_dict(self, *, salt: str = "") -> dict[str, Any]:
        """Return a deterministic JSON-compatible accountant payload."""

        return {
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "composition": self.composition().to_dict(),
            "sensitivity_registry": self.sensitivity_registry.to_dict(),
            "spends": [spend.to_dict(salt=salt) for spend in self._spends],
        }


@dataclass(frozen=True)
class EpsilonPolicy:
    """Declared privacy budget ceiling for one synthetic-generation scope.

    A policy bounds the cumulative ``(epsilon, delta)`` that may be spent under a
    release scope and fixes the composition rule used to accumulate spend. It is
    loaded from committed config and carries no source data.
    """

    scope: str
    max_epsilon: float
    max_delta: float
    composition: CompositionRule = "basic"
    delta_prime: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _non_empty_string(self.scope))
        object.__setattr__(
            self,
            "max_epsilon",
            _positive_float(self.max_epsilon, field_name="max_epsilon"),
        )
        object.__setattr__(
            self,
            "max_delta",
            _delta_float(self.max_delta, field_name="max_delta"),
        )
        object.__setattr__(
            self,
            "composition",
            _composition_rule(self.composition),
        )
        delta_prime = _non_negative_float(self.delta_prime, field_name="delta_prime")
        if self.composition == "advanced":
            if delta_prime <= 0.0:
                raise ValueError(
                    "advanced composition requires a positive delta_prime slack"
                )
            if delta_prime >= self.max_delta:
                raise ValueError("delta_prime must be smaller than max_delta")
        else:
            delta_prime = 0.0
        object.__setattr__(self, "delta_prime", delta_prime)

    @classmethod
    def from_mapping(cls, scope: str, payload: Mapping[str, Any]) -> EpsilonPolicy:
        """Build a policy from a committed-config mapping for ``scope``."""

        if not isinstance(payload, Mapping):
            raise TypeError("policy payload must be a mapping")
        return cls(
            scope=scope,
            max_epsilon=payload["max_epsilon"],
            max_delta=payload["max_delta"],
            composition=payload.get("composition", "basic"),
            delta_prime=payload.get("delta_prime", 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible policy payload."""

        return {
            "scope": self.scope,
            "max_epsilon": self.max_epsilon,
            "max_delta": self.max_delta,
            "composition": self.composition,
            "delta_prime": self.delta_prime,
        }


@dataclass(frozen=True)
class GenerationSpend:
    """One privacy spend charged by a synthetic-generation request.

    Records only numeric accounting aggregates and scope/family identifiers; no
    raw source rows, column values, or PHI ever enter a spend record.
    """

    sequence: int
    scope: str
    family: str
    epsilon: float
    delta: float
    mechanism: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence",
            _positive_int(self.sequence, field_name="sequence"),
        )
        object.__setattr__(self, "scope", _identifier(self.scope, field_name="scope"))
        object.__setattr__(
            self, "family", _identifier(self.family, field_name="family")
        )
        object.__setattr__(
            self,
            "epsilon",
            _non_negative_float(self.epsilon, field_name="epsilon"),
        )
        object.__setattr__(
            self,
            "delta",
            _delta_float(self.delta, field_name="delta"),
        )
        object.__setattr__(
            self,
            "mechanism",
            _identifier(self.mechanism, field_name="mechanism"),
        )

    def spend_hash(self, *, salt: str = "") -> str:
        """Return a deterministic hash over non-PHI accounting metadata."""

        payload = {
            "delta": self.delta,
            "epsilon": self.epsilon,
            "family": self.family,
            "mechanism": self.mechanism,
            "salt": salt,
            "scope": self.scope,
            "sequence": self.sequence,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self, *, salt: str = "") -> dict[str, Any]:
        """Return a deterministic JSON-compatible spend payload."""

        return {
            "sequence": self.sequence,
            "scope": self.scope,
            "family": self.family,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "mechanism": self.mechanism,
            "spend_hash": self.spend_hash(salt=salt),
        }


@dataclass(frozen=True)
class BudgetComposition:
    """Cumulative composition totals for one scope under its policy rule."""

    scope: str
    composition: CompositionRule
    query_count: int
    epsilon: float
    delta: float
    max_epsilon: float
    max_delta: float
    remaining_epsilon: float
    remaining_delta: float

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible composition payload."""

        return {
            "scope": self.scope,
            "composition": self.composition,
            "query_count": self.query_count,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "max_epsilon": self.max_epsilon,
            "max_delta": self.max_delta,
            "remaining_epsilon": self.remaining_epsilon,
            "remaining_delta": self.remaining_delta,
        }


@dataclass(frozen=True)
class BudgetDecision:
    """Outcome of gating a generation request against a scope's policy."""

    allowed: bool
    scope: str
    requested_epsilon: float
    requested_delta: float
    composition: CompositionRule
    projected_epsilon: float
    projected_delta: float
    max_epsilon: float
    max_delta: float
    remaining_epsilon: float
    remaining_delta: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible decision payload."""

        return {
            "allowed": self.allowed,
            "scope": self.scope,
            "requested_epsilon": self.requested_epsilon,
            "requested_delta": self.requested_delta,
            "composition": self.composition,
            "projected_epsilon": self.projected_epsilon,
            "projected_delta": self.projected_delta,
            "max_epsilon": self.max_epsilon,
            "max_delta": self.max_delta,
            "remaining_epsilon": self.remaining_epsilon,
            "remaining_delta": self.remaining_delta,
            "reason": self.reason,
        }


class BudgetExceeded(ValueError):
    """Raised when a generation request would exceed the scope's epsilon policy."""

    def __init__(self, decision: BudgetDecision) -> None:
        self.decision = decision
        super().__init__(
            "DP generation budget exceeded for scope "
            f"{decision.scope!r}: epsilon={decision.projected_epsilon:.6g}/"
            f"{decision.max_epsilon:.6g} (remaining {decision.remaining_epsilon:.6g}), "
            f"delta={decision.projected_delta:.6g}/{decision.max_delta:.6g} "
            f"(remaining {decision.remaining_delta:.6g})"
        )


@dataclass
class DPGenerationBudgetAccountant:
    """Accountant that gates synthetic-data generation against epsilon policies.

    Cumulative ``(epsilon, delta)`` spend is tracked per release scope, composed
    under the scope policy's rule (``basic`` sequential composition or ``advanced``
    Dwork-Rothblum-Vadhan composition), and persisted as a numeric-only ledger.
    A synthesizer bridge (sibling OM-772 work) calls :meth:`guard_generation`
    before generation; this class never generates data itself.
    """

    policies: Mapping[str, EpsilonPolicy]
    _ledger: list[GenerationSpend] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        normalized: dict[str, EpsilonPolicy] = {}
        for scope, policy in dict(self.policies).items():
            key = _non_empty_string(scope)
            if not isinstance(policy, EpsilonPolicy):
                policy = EpsilonPolicy.from_mapping(key, policy)
            if policy.scope != key:
                raise ValueError(
                    f"policy scope {policy.scope!r} does not match key {key!r}"
                )
            normalized[key] = policy
        if not normalized:
            raise ValueError("at least one epsilon policy is required")
        self.policies = normalized
        self._ledger = list(self._ledger)

    @classmethod
    def from_config(cls) -> DPGenerationBudgetAccountant:
        """Build an accountant from the committed epsilon-policy config."""

        return cls(load_epsilon_policies())

    @property
    def ledger(self) -> tuple[GenerationSpend, ...]:
        """Return recorded spends in charge order."""

        return tuple(self._ledger)

    def policy_for(self, scope: str) -> EpsilonPolicy:
        """Return the epsilon policy governing ``scope`` or raise ``KeyError``."""

        key = _non_empty_string(scope)
        if key not in self.policies:
            raise KeyError(f"no epsilon policy for scope {key!r}")
        return self.policies[key]

    def compose(
        self,
        scope: str,
        *,
        family: str | None = None,
        extra: Sequence[GenerationSpend] = (),
    ) -> BudgetComposition:
        """Return cumulative composition for ``scope`` plus any ``extra`` spend."""

        policy = self.policy_for(scope)
        recorded = [spend for spend in self._ledger if spend.scope == policy.scope]
        if family is not None:
            wanted = _non_empty_string(family)
            recorded = [spend for spend in recorded if spend.family == wanted]
        selected = (*recorded, *extra)
        epsilons = [spend.epsilon for spend in selected]
        deltas = [spend.delta for spend in selected]
        if policy.composition == "advanced":
            epsilon = _advanced_epsilon(epsilons, policy.delta_prime)
            delta = _advanced_delta(deltas, policy.delta_prime)
        else:
            epsilon = math.fsum(epsilons)
            delta = math.fsum(deltas)
        return BudgetComposition(
            scope=policy.scope,
            composition=policy.composition,
            query_count=len(selected),
            epsilon=epsilon,
            delta=delta,
            max_epsilon=policy.max_epsilon,
            max_delta=policy.max_delta,
            remaining_epsilon=policy.max_epsilon - epsilon,
            remaining_delta=policy.max_delta - delta,
        )

    def check_budget(
        self,
        requested_epsilon: float,
        requested_delta: float,
        scope: str,
        *,
        family: str = "default",
        mechanism: str = "gaussian",
    ) -> BudgetDecision:
        """Return whether a request fits the scope policy without recording it.

        The ledger is never mutated. The returned decision reports the projected
        cumulative spend, the composition rule used, and the remaining headroom.
        """

        policy = self.policy_for(scope)
        candidate = GenerationSpend(
            sequence=len(self._ledger) + 1,
            scope=policy.scope,
            family=family,
            epsilon=requested_epsilon,
            delta=requested_delta,
            mechanism=mechanism,
        )
        committed = self.compose(policy.scope)
        projected = self.compose(policy.scope, extra=(candidate,))
        epsilon_ok = projected.epsilon <= policy.max_epsilon
        delta_ok = projected.delta <= policy.max_delta
        allowed = epsilon_ok and delta_ok
        if allowed:
            reason = "within budget"
        elif not epsilon_ok and not delta_ok:
            reason = "epsilon and delta exceed policy"
        elif not epsilon_ok:
            reason = "epsilon exceeds policy"
        else:
            reason = "delta exceeds policy"
        return BudgetDecision(
            allowed=allowed,
            scope=policy.scope,
            requested_epsilon=candidate.epsilon,
            requested_delta=candidate.delta,
            composition=policy.composition,
            projected_epsilon=projected.epsilon,
            projected_delta=projected.delta,
            max_epsilon=policy.max_epsilon,
            max_delta=policy.max_delta,
            remaining_epsilon=committed.remaining_epsilon,
            remaining_delta=committed.remaining_delta,
            reason=reason,
        )

    def guard_generation(
        self,
        requested_epsilon: float,
        requested_delta: float,
        scope: str,
        *,
        family: str = "default",
        mechanism: str = "gaussian",
    ) -> BudgetDecision:
        """Gate one generation request; record the spend or raise ``BudgetExceeded``.

        This is the hook a synthesizer bridge calls before generating rows. On a
        blocked request the ledger is left unchanged and ``BudgetExceeded`` is
        raised carrying the remaining budget.
        """

        decision = self.check_budget(
            requested_epsilon,
            requested_delta,
            scope,
            family=family,
            mechanism=mechanism,
        )
        if not decision.allowed:
            raise BudgetExceeded(decision)
        self._ledger.append(
            GenerationSpend(
                sequence=len(self._ledger) + 1,
                scope=decision.scope,
                family=family,
                epsilon=requested_epsilon,
                delta=requested_delta,
                mechanism=mechanism,
            )
        )
        return decision

    def to_dict(self, *, salt: str = "") -> dict[str, Any]:
        """Return a deterministic JSON-compatible accountant payload."""

        return {
            "policies": {
                scope: self.policies[scope].to_dict() for scope in sorted(self.policies)
            },
            "compositions": {
                scope: self.compose(scope).to_dict() for scope in sorted(self.policies)
            },
            "ledger": [spend.to_dict(salt=salt) for spend in self._ledger],
        }


def load_epsilon_policies() -> dict[str, EpsilonPolicy]:
    """Load the committed epsilon-policy config as a scope-to-policy mapping."""

    resource = resources.files("openmed.risk").joinpath(
        "data",
        EPSILON_POLICY_CONFIG_RESOURCE,
    )
    with resource.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    schema_version = _positive_int(
        payload.get("schema_version"),
        field_name="schema_version",
    )
    if schema_version != CURRENT_EPSILON_POLICY_SCHEMA_VERSION:
        raise ValueError(
            "epsilon policy schema_version "
            f"{schema_version} is not supported; expected "
            f"{CURRENT_EPSILON_POLICY_SCHEMA_VERSION}"
        )
    policies = payload.get("policies")
    if not isinstance(policies, Mapping) or not policies:
        raise ValueError("epsilon policy config must define a non-empty 'policies' map")
    return {
        str(scope): EpsilonPolicy.from_mapping(str(scope), entry)
        for scope, entry in policies.items()
    }


def epsilon_policy_for(scope: str) -> EpsilonPolicy:
    """Return the committed epsilon policy governing ``scope``."""

    policies = load_epsilon_policies()
    key = _non_empty_string(scope)
    if key not in policies:
        raise KeyError(f"no epsilon policy for scope {key!r}")
    return policies[key]


@dataclass(frozen=True)
class RiskBudget:
    """Limits for a single document's residual disclosure risk."""

    name: str = "custom"
    max_residual_qi_weight: float | None = None
    max_surviving_direct_ids: int | None = 0
    min_k: int | None = None
    max_singleton_records: int | None = None
    qi_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_QI_WEIGHTS)
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible budget payload."""

        return {
            "name": self.name,
            "max_residual_qi_weight": self.max_residual_qi_weight,
            "max_surviving_direct_ids": self.max_surviving_direct_ids,
            "min_k": self.min_k,
            "max_singleton_records": self.max_singleton_records,
            "qi_weights": {
                str(category): float(weight)
                for category, weight in sorted(self.qi_weights.items())
            },
        }


@dataclass(frozen=True)
class RiskBudgetViolation:
    """One budget limit exceeded by a document."""

    metric: str
    consumed: int | float
    limit: int | float
    comparison: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible violation payload."""

        return {
            "metric": self.metric,
            "consumed": self.consumed,
            "limit": self.limit,
            "comparison": self.comparison,
        }


@dataclass(frozen=True)
class RiskBudgetVerdict:
    """Risk-budget result suitable for embedding in reports."""

    within_budget: bool
    budget: Mapping[str, Any]
    breakdown: Mapping[str, Mapping[str, Any]]
    violations: tuple[RiskBudgetViolation, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible verdict payload."""

        return {
            "within_budget": self.within_budget,
            "budget": copy.deepcopy(dict(self.budget)),
            "breakdown": {
                metric: dict(values) for metric, values in self.breakdown.items()
            },
            "violations": [violation.to_dict() for violation in self.violations],
        }


class RiskBudgetExceeded(ValueError):
    """Raised when strict risk-budget evaluation rejects a document."""

    def __init__(self, verdict: RiskBudgetVerdict) -> None:
        self.verdict = verdict
        metrics = (
            ", ".join(violation.metric for violation in verdict.violations)
            or "risk_budget"
        )
        super().__init__(f"Risk budget exceeded: {metrics}")


DEFAULT_RISK_BUDGET = RiskBudget(
    name="balanced",
    max_residual_qi_weight=6.0,
    max_surviving_direct_ids=0,
)

DEFAULT_POLICY_BUDGETS: Mapping[str, RiskBudget] = {
    "hipaa_safe_harbor": RiskBudget(
        name="hipaa_safe_harbor",
        max_residual_qi_weight=1.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "hipaa_expert_review_assist": RiskBudget(
        name="hipaa_expert_review_assist",
        max_residual_qi_weight=4.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "gdpr_pseudonymization": RiskBudget(
        name="gdpr_pseudonymization",
        max_residual_qi_weight=3.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "gdpr_art9_health": RiskBudget(
        name="gdpr_art9_health",
        max_residual_qi_weight=2.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "research_limited_dataset": RiskBudget(
        name="research_limited_dataset",
        max_residual_qi_weight=8.0,
        max_surviving_direct_ids=0,
    ),
    "strict_no_leak": RiskBudget(
        name="strict_no_leak",
        max_residual_qi_weight=0.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "clinical_minimal_redaction": RiskBudget(
        name="clinical_minimal_redaction",
        max_residual_qi_weight=10.0,
        max_surviving_direct_ids=0,
    ),
    "canada_pipeda": RiskBudget(
        name="canada_pipeda",
        max_residual_qi_weight=3.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "uk_ico_anonymisation": RiskBudget(
        name="uk_ico_anonymisation",
        max_residual_qi_weight=3.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "australia_privacy_act": RiskBudget(
        name="australia_privacy_act",
        max_residual_qi_weight=3.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "china_pipl": RiskBudget(
        name="china_pipl",
        max_residual_qi_weight=2.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "india_dpdp_act": RiskBudget(
        name="india_dpdp_act",
        max_residual_qi_weight=3.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "africa_malabo_baseline": RiskBudget(
        name="africa_malabo_baseline",
        max_residual_qi_weight=0.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "za_popia": RiskBudget(
        name="za_popia",
        max_residual_qi_weight=0.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "ng_ndpa": RiskBudget(
        name="ng_ndpa",
        max_residual_qi_weight=0.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "ke_dpa": RiskBudget(
        name="ke_dpa",
        max_residual_qi_weight=0.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "india_health_id": RiskBudget(
        name="india_health_id",
        max_residual_qi_weight=0.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "eg_pdpl": RiskBudget(
        name="eg_pdpl",
        max_residual_qi_weight=0.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
    "ma_law_09_08": RiskBudget(
        name="ma_law_09_08",
        max_residual_qi_weight=0.0,
        max_surviving_direct_ids=0,
        max_singleton_records=0,
    ),
}

if set(DEFAULT_POLICY_BUDGETS) != set(CANONICAL_POLICY_NAMES):
    missing = sorted(set(CANONICAL_POLICY_NAMES) - set(DEFAULT_POLICY_BUDGETS))
    extra = sorted(set(DEFAULT_POLICY_BUDGETS) - set(CANONICAL_POLICY_NAMES))
    raise RuntimeError(f"policy budget mismatch: missing={missing}, extra={extra}")


def budget_for_policy(policy: Any) -> RiskBudget:
    """Return the bundled default budget for a policy profile or policy name."""

    name = canonical_policy_name(getattr(policy, "name", policy))
    return DEFAULT_POLICY_BUDGETS[name]


def evaluate_budget(
    risk: Any,
    budget: RiskBudget | None = None,
    *,
    strict: bool = False,
) -> RiskBudgetVerdict:
    """Evaluate a residual-risk report against a per-document budget.

    Args:
        risk: A ``risk_report`` mapping, a residual-risk mapping containing a
            nested ``risk_report``, or an object exposing ``to_dict()``. Surviving
            direct identifiers should be provided as counts or countable records,
            not raw identifier strings.
        budget: Budget limits to enforce. Defaults to ``DEFAULT_RISK_BUDGET``.
        strict: When true, raise ``RiskBudgetExceeded`` if any limit is violated.

    Returns:
        A deterministic verdict with consumed-vs-limit breakdowns and violations.
    """

    selected_budget = budget or DEFAULT_RISK_BUDGET
    payload = _mapping(risk)
    report = _risk_report(payload)

    consumed = {
        "residual_qi_weight": _residual_qi_weight(
            report.get("quasi_identifiers", ()),
            selected_budget.qi_weights,
        ),
        "surviving_direct_ids": _direct_identifier_count(report, payload),
        "k_min": _non_negative_int(report.get("k_min", 0), field_name="k_min"),
        "singleton_records": _count(report.get("singleton_records", ())),
    }

    breakdown = _breakdown(consumed, selected_budget)
    violations = tuple(_violations(breakdown))
    verdict = RiskBudgetVerdict(
        within_budget=not violations,
        budget=selected_budget.to_dict(),
        breakdown=breakdown,
        violations=violations,
    )
    if strict and violations:
        raise RiskBudgetExceeded(verdict)
    return verdict


def _canonical_dp_label(value: Any) -> str:
    label = _non_empty_string(value)
    return label.upper().replace("-", "_").replace(" ", "_")


def _composition_rule(value: Any) -> CompositionRule:
    rule = _non_empty_string(value)
    if rule not in {"basic", "advanced"}:
        raise ValueError("composition must be one of basic or advanced")
    return cast(CompositionRule, rule)


def _advanced_epsilon(epsilons: Sequence[float], delta_prime: float) -> float:
    """Return the Dwork-Rothblum-Vadhan advanced composition epsilon.

    For a heterogeneous sequence of pure/approximate mechanisms this uses the
    conservative bound ``sqrt(2 * ln(1/delta') * sum(eps_i**2)) +
    sum(eps_i * (exp(eps_i) - 1))``, which reduces to the classic homogeneous
    advanced composition theorem when every ``eps_i`` is equal.
    """

    if not epsilons:
        return 0.0
    if delta_prime <= 0.0:
        return math.inf
    sum_squares = math.fsum(epsilon * epsilon for epsilon in epsilons)
    tail = math.fsum(epsilon * math.expm1(epsilon) for epsilon in epsilons)
    return math.sqrt(2.0 * math.log(1.0 / delta_prime) * sum_squares) + tail


def _advanced_delta(deltas: Sequence[float], delta_prime: float) -> float:
    """Return advanced-composition delta: summed per-call delta plus the slack."""

    return math.fsum(deltas) + delta_prime


def _draw_kind(value: Any) -> SurrogateDrawKind:
    kind = _non_empty_string(value)
    if kind not in {"categorical", "numeric", "date_offset"}:
        raise ValueError(
            "draw_kind must be one of categorical, numeric, or date_offset"
        )
    return cast(SurrogateDrawKind, kind)


def _non_empty_string(value: Any) -> str:
    parsed = str(value).strip()
    if not parsed:
        raise ValueError("value must not be empty")
    return parsed


def _identifier(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string identifier, not "
            f"{type(value).__name__}; raw rows must never enter the ledger"
        )
    parsed = value.strip()
    if not parsed:
        raise ValueError(f"{field_name} must not be empty")
    return parsed


def _positive_int(value: Any, *, field_name: str) -> int:
    parsed = _non_negative_int(value, field_name=field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _positive_float(value: Any, *, field_name: str) -> float:
    parsed = _finite_float(value, field_name=field_name)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _non_negative_float(value: Any, *, field_name: str) -> float:
    parsed = _finite_float(value, field_name=field_name)
    if parsed < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _delta_float(value: Any, *, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name=field_name)
    if parsed >= 1.0:
        raise ValueError(f"{field_name} must be less than 1")
    return parsed


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must not be a boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _rdp_order(value: Any) -> float:
    parsed = _finite_float(value, field_name="rdp_order")
    if parsed <= 1.0:
        raise ValueError("rdp_order must be greater than 1")
    return parsed


def _rho_from_epsilon(epsilon: float) -> float:
    return (epsilon**2) / 2.0


def _zcdp_epsilon(rho: float, delta: float) -> float:
    if rho == 0.0:
        return 0.0
    if delta <= 0.0:
        return math.inf
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def _rdp_epsilon(rho: float, delta: float, orders: Sequence[float]) -> float:
    if rho == 0.0:
        return 0.0
    if delta <= 0.0:
        return math.inf
    log_delta = math.log(1.0 / delta)
    return min(order * rho + (log_delta / (order - 1.0)) for order in orders)


def _exceeds_budget(composition: DPSurrogateComposition) -> bool:
    return (
        composition.reported_epsilon > composition.target_epsilon
        or composition.basic_delta > composition.target_delta
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("risk must be a mapping or expose to_dict()")
    return value


def _risk_report(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("risk_report")
    if isinstance(nested, Mapping):
        return nested
    residual = payload.get("residual_risk")
    if isinstance(residual, Mapping) and isinstance(
        residual.get("risk_report"), Mapping
    ):
        return residual["risk_report"]
    return payload


def _residual_qi_weight(value: Any, weights: Mapping[str, float]) -> float:
    weight = 0.0
    for item in _items(value):
        if isinstance(item, Mapping):
            category = str(item.get("category", "*"))
        else:
            category = "*"
        weight += float(weights.get(category, weights.get("*", 1.0)))
    return weight + 0.0


def _direct_identifier_count(
    report: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> int:
    for source in (report, payload):
        count = _first_count(source, _DIRECT_ID_COUNT_KEYS)
        if count is not None:
            return count

    residual = payload.get("residual_risk")
    if isinstance(residual, Mapping):
        count = _first_count(residual, _DIRECT_ID_COUNT_KEYS)
        if count is not None:
            return count
    return 0


def _first_count(payload: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        if key not in payload:
            continue
        return _count(payload[key])
    return None


def _count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError("count values must not be booleans")
    if isinstance(value, int):
        return _non_negative_int(value, field_name="count")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("count values must be whole numbers")
        return _non_negative_int(int(value), field_name="count")
    if isinstance(value, Mapping):
        if "count" in value:
            return _count(value["count"])
        if "total" in value:
            return _count(value["total"])
        return len(value)
    if _is_sequence(value):
        return len(value)
    raise ValueError(f"cannot derive count from {type(value).__name__}")


def _non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must not be a boolean")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if _is_sequence(value):
        return list(value)
    raise ValueError(
        f"expected a sequence of quasi-identifiers, got {type(value).__name__}"
    )


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _breakdown(
    consumed: Mapping[str, int | float],
    budget: RiskBudget,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if budget.max_residual_qi_weight is not None:
        rows["residual_qi_weight"] = {
            "consumed": float(consumed["residual_qi_weight"]),
            "limit": float(budget.max_residual_qi_weight),
            "comparison": "max",
        }
    if budget.max_surviving_direct_ids is not None:
        rows["surviving_direct_ids"] = {
            "consumed": int(consumed["surviving_direct_ids"]),
            "limit": int(budget.max_surviving_direct_ids),
            "comparison": "max",
        }
    if budget.min_k is not None:
        rows["k_min"] = {
            "consumed": int(consumed["k_min"]),
            "limit": int(budget.min_k),
            "comparison": "min",
        }
    if budget.max_singleton_records is not None:
        rows["singleton_records"] = {
            "consumed": int(consumed["singleton_records"]),
            "limit": int(budget.max_singleton_records),
            "comparison": "max",
        }
    return rows


def _violations(
    breakdown: Mapping[str, Mapping[str, Any]],
) -> list[RiskBudgetViolation]:
    violations: list[RiskBudgetViolation] = []
    for metric, values in breakdown.items():
        consumed = values["consumed"]
        limit = values["limit"]
        comparison = str(values["comparison"])
        if comparison == "max" and consumed > limit:
            violations.append(RiskBudgetViolation(metric, consumed, limit, comparison))
        elif comparison == "min" and consumed < limit:
            violations.append(RiskBudgetViolation(metric, consumed, limit, comparison))
    return violations


__all__ = [
    "CURRENT_EPSILON_POLICY_SCHEMA_VERSION",
    "CompositionRule",
    "DEFAULT_DP_SURROGATE_SENSITIVITIES",
    "DEFAULT_POLICY_BUDGETS",
    "DEFAULT_QI_WEIGHTS",
    "DEFAULT_RDP_ORDERS",
    "DEFAULT_RISK_BUDGET",
    "EPSILON_POLICY_CONFIG_RESOURCE",
    "BudgetComposition",
    "BudgetDecision",
    "BudgetExceeded",
    "DPGenerationBudgetAccountant",
    "DPSurrogateBudget",
    "DPSurrogateBudgetExceeded",
    "DPSurrogateComposition",
    "DPSurrogateSensitivity",
    "DPSurrogateSensitivityRegistry",
    "DPSurrogateSpend",
    "EpsilonPolicy",
    "GenerationSpend",
    "RiskBudget",
    "RiskBudgetExceeded",
    "RiskBudgetVerdict",
    "RiskBudgetViolation",
    "SurrogateDrawKind",
    "budget_for_policy",
    "epsilon_policy_for",
    "evaluate_budget",
    "load_epsilon_policies",
]
