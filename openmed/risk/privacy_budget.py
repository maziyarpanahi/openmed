"""Deterministic privacy-budget accounting for aggregate releases.

The ledger in this module records only release-context identifiers, numeric
epsilon/delta values, and aggregate counts. It never accepts a source payload
or makes a network call. Callers should invoke :meth:`PrivacyBudgetLedger.check`
when they need a non-mutating decision and :meth:`PrivacyBudgetLedger.record_release`
immediately before an aggregate release.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

PRIVACY_BUDGET_LEDGER_SCHEMA_VERSION = "openmed.privacy_budget_ledger.v1"

_SAFE_CONTEXT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_PHI_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{3}[-.)]\d{3}[-.]\d{4}\b"),
)


@dataclass(frozen=True)
class PrivacyBudget:
    """Cumulative epsilon and delta ceiling for one release context."""

    epsilon: float
    delta: float

    def __post_init__(self) -> None:
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

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> PrivacyBudget:
        """Build a budget from a JSON-style mapping.

        ``max_epsilon`` and ``max_delta`` are accepted as aliases so a budget
        can be moved from an existing policy document without carrying any
        unapproved fields into the ledger.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("privacy budget must be a mapping")
        allowed = {"epsilon", "delta", "max_epsilon", "max_delta"}
        if set(payload) - allowed:
            raise ValueError("privacy budget contains unsupported fields")
        epsilon = payload.get("epsilon", payload.get("max_epsilon"))
        delta = payload.get("delta", payload.get("max_delta"))
        if epsilon is None or delta is None:
            raise ValueError("privacy budget requires epsilon and delta")
        return cls(epsilon=epsilon, delta=delta)

    @property
    def max_epsilon(self) -> float:
        """Return the epsilon ceiling using policy-compatible terminology."""

        return self.epsilon

    @property
    def max_delta(self) -> float:
        """Return the delta ceiling using policy-compatible terminology."""

        return self.delta

    def to_dict(self) -> dict[str, float]:
        """Return the numeric budget ceiling."""

        return {"epsilon": self.epsilon, "delta": self.delta}


@dataclass(frozen=True)
class PrivacyBudgetSpend:
    """One accepted aggregate-release charge.

    A spend contains no row, cell, document, recipient, or free-form request
    data. ``context`` is restricted to a short safe identifier.
    """

    sequence: int
    context: str
    epsilon: float
    delta: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence",
            _positive_int(self.sequence, field_name="sequence"),
        )
        object.__setattr__(
            self,
            "context",
            _safe_identifier(self.context, field_name="context"),
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

    def to_dict(self) -> dict[str, Any]:
        """Return the safe, numeric-only spend representation."""

        return {
            "sequence": self.sequence,
            "context": self.context,
            "epsilon": self.epsilon,
            "delta": self.delta,
        }


@dataclass(frozen=True)
class PrivacyBudgetDecision:
    """Non-mutating result of checking one proposed aggregate release."""

    allowed: bool
    context: str
    requested_epsilon: float
    requested_delta: float
    projected_epsilon: float
    projected_delta: float
    max_epsilon: float
    max_delta: float
    remaining_epsilon: float
    remaining_delta: float
    reason: str

    @property
    def budget_epsilon(self) -> float:
        """Return the configured epsilon ceiling."""

        return self.max_epsilon

    @property
    def budget_delta(self) -> float:
        """Return the configured delta ceiling."""

        return self.max_delta

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, payload-free decision representation."""

        return {
            "allowed": self.allowed,
            "context": self.context,
            "requested_epsilon": self.requested_epsilon,
            "requested_delta": self.requested_delta,
            "projected_epsilon": self.projected_epsilon,
            "projected_delta": self.projected_delta,
            "max_epsilon": self.max_epsilon,
            "max_delta": self.max_delta,
            "remaining_epsilon": self.remaining_epsilon,
            "remaining_delta": self.remaining_delta,
            "reason": self.reason,
        }


class PrivacyBudgetExceeded(ValueError):
    """Raised when an aggregate release would exceed its context budget."""

    def __init__(self, decision: PrivacyBudgetDecision) -> None:
        self.decision = decision
        super().__init__(
            "privacy budget exceeded: "
            f"epsilon={decision.projected_epsilon:.6g}/{decision.max_epsilon:.6g}, "
            f"delta={decision.projected_delta:.6g}/{decision.max_delta:.6g}"
        )


@dataclass
class PrivacyBudgetLedger:
    """Local ledger that gates cumulative aggregate-release privacy spend.

    Budgets are keyed by named release context and composed with conservative
    sequential addition. A rejected request is never appended to ``spends``;
    its count is retained separately as aggregate evidence. The ledger has no
    persistence or network behavior, so callers can choose their own local
    storage boundary.
    """

    budgets: Mapping[str, PrivacyBudget | Mapping[str, Any]] = field(
        default_factory=dict
    )
    _spends: list[PrivacyBudgetSpend] = field(default_factory=list, repr=False)
    _rejected: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.budgets, Mapping):
            raise TypeError("budgets must be a mapping")

        normalized: dict[str, PrivacyBudget] = {}
        for context, budget in self.budgets.items():
            key = _safe_identifier(context, field_name="context")
            if key in normalized:
                raise ValueError("privacy budget contexts must be unique")
            normalized[key] = (
                budget
                if isinstance(budget, PrivacyBudget)
                else PrivacyBudget.from_mapping(budget)
            )
        self.budgets = normalized
        self._spends = list(self._spends)

        rejected: dict[str, int] = {}
        for context, count in self._rejected.items():
            key = _safe_identifier(context, field_name="context")
            rejected[key] = _non_negative_int(count, field_name="rejected")
        self._rejected = rejected

    @classmethod
    def from_budgets(
        cls,
        budgets: Mapping[str, PrivacyBudget | Mapping[str, Any]],
    ) -> PrivacyBudgetLedger:
        """Construct a ledger from named epsilon/delta budgets."""

        return cls(budgets=budgets)

    @property
    def contexts(self) -> tuple[str, ...]:
        """Return registered release contexts in deterministic order."""

        return tuple(sorted(self.budgets))

    @property
    def spends(self) -> tuple[PrivacyBudgetSpend, ...]:
        """Return accepted spends in charge order."""

        return tuple(self._spends)

    @property
    def ledger(self) -> tuple[PrivacyBudgetSpend, ...]:
        """Return accepted spends using the conventional ledger name."""

        return self.spends

    @property
    def rejected_count(self) -> int:
        """Return the aggregate count of refused requests."""

        return sum(self._rejected.values())

    def register_context(
        self,
        context: str,
        epsilon: float,
        delta: float,
    ) -> PrivacyBudget:
        """Register or replace one named release-context budget.

        Replacing a budget below already accepted spend is refused so an
        administrator cannot retroactively make the ledger inconsistent.
        """

        key = _safe_identifier(context, field_name="context")
        budget = PrivacyBudget(epsilon=epsilon, delta=delta)
        current_epsilon, current_delta = self._spent(key)
        if budget.epsilon < current_epsilon or budget.delta < current_delta:
            raise ValueError("privacy budget cannot be below recorded spend")
        self.budgets = {**self.budgets, key: budget}
        self._rejected.setdefault(key, 0)
        return budget

    def set_budget(
        self,
        context: str,
        epsilon: float,
        delta: float,
    ) -> PrivacyBudget:
        """Alias for :meth:`register_context`."""

        return self.register_context(context, epsilon, delta)

    def budget_for(self, context: str) -> PrivacyBudget:
        """Return the budget for a safe context or raise ``KeyError``."""

        key = _safe_identifier(context, field_name="context")
        try:
            return self.budgets[key]
        except KeyError as exc:
            raise KeyError("no privacy budget registered for release context") from exc

    def check(
        self,
        context: str,
        epsilon: float,
        delta: float,
    ) -> PrivacyBudgetDecision:
        """Check a proposed spend without changing the ledger."""

        key = _safe_identifier(context, field_name="context")
        budget = self.budget_for(key)
        requested_epsilon = _non_negative_float(epsilon, field_name="epsilon")
        requested_delta = _delta_float(delta, field_name="delta")
        current_epsilon, current_delta = self._spent(key)
        projected_epsilon = math.fsum((current_epsilon, requested_epsilon))
        projected_delta = math.fsum((current_delta, requested_delta))
        epsilon_ok = projected_epsilon <= budget.epsilon
        delta_ok = projected_delta <= budget.delta
        allowed = epsilon_ok and delta_ok
        if allowed:
            reason = "within budget"
        elif not epsilon_ok and not delta_ok:
            reason = "epsilon and delta exceed budget"
        elif not epsilon_ok:
            reason = "epsilon exceeds budget"
        else:
            reason = "delta exceeds budget"
        return PrivacyBudgetDecision(
            allowed=allowed,
            context=key,
            requested_epsilon=requested_epsilon,
            requested_delta=requested_delta,
            projected_epsilon=projected_epsilon,
            projected_delta=projected_delta,
            max_epsilon=budget.epsilon,
            max_delta=budget.delta,
            remaining_epsilon=max(0.0, budget.epsilon - current_epsilon),
            remaining_delta=max(0.0, budget.delta - current_delta),
            reason=reason,
        )

    def check_budget(
        self,
        requested_epsilon: float,
        requested_delta: float,
        context: str,
    ) -> PrivacyBudgetDecision:
        """Check a spend using the existing risk-budget argument order."""

        return self.check(context, requested_epsilon, requested_delta)

    def can_spend(self, context: str, epsilon: float, delta: float) -> bool:
        """Return whether a request fits without recording it."""

        try:
            return self.check(context, epsilon, delta).allowed
        except (KeyError, TypeError, ValueError):
            return False

    def record_release(
        self,
        context: str,
        epsilon: float,
        delta: float,
    ) -> PrivacyBudgetDecision:
        """Record an accepted release or raise before adding an over-budget spend."""

        decision = self.check(context, epsilon, delta)
        if not decision.allowed:
            self._rejected[decision.context] = (
                self._rejected.get(decision.context, 0) + 1
            )
            raise PrivacyBudgetExceeded(decision)
        self._spends.append(
            PrivacyBudgetSpend(
                sequence=len(self._spends) + 1,
                context=decision.context,
                epsilon=decision.requested_epsilon,
                delta=decision.requested_delta,
            )
        )
        return decision

    def spend(
        self,
        context: str,
        epsilon: float,
        delta: float,
    ) -> PrivacyBudgetDecision:
        """Alias for :meth:`record_release`."""

        return self.record_release(context, epsilon, delta)

    def consume(
        self,
        context: str,
        epsilon: float,
        delta: float,
    ) -> PrivacyBudgetDecision:
        """Alias for :meth:`record_release`."""

        return self.record_release(context, epsilon, delta)

    def render_counts_only(self) -> dict[str, Any]:
        """Render deterministic aggregate evidence without release payloads.

        The result includes counts and numeric epsilon/delta totals per safe
        context. It intentionally omits individual spend records and all
        caller payloads.
        """

        contexts: dict[str, dict[str, Any]] = {}
        total_epsilon = 0.0
        total_delta = 0.0
        total_releases = 0
        for context in self.contexts:
            budget = self.budgets[context]
            spends = tuple(spend for spend in self._spends if spend.context == context)
            spent_epsilon = math.fsum(spend.epsilon for spend in spends)
            spent_delta = math.fsum(spend.delta for spend in spends)
            release_count = len(spends)
            rejected_count = self._rejected.get(context, 0)
            contexts[context] = {
                "attempt_count": release_count + rejected_count,
                "release_count": release_count,
                "rejected_count": rejected_count,
                "spent_epsilon": spent_epsilon,
                "spent_delta": spent_delta,
                "budget_epsilon": budget.epsilon,
                "budget_delta": budget.delta,
                "remaining_epsilon": max(0.0, budget.epsilon - spent_epsilon),
                "remaining_delta": max(0.0, budget.delta - spent_delta),
            }
            total_epsilon = math.fsum((total_epsilon, spent_epsilon))
            total_delta = math.fsum((total_delta, spent_delta))
            total_releases += release_count

        return {
            "schema_version": PRIVACY_BUDGET_LEDGER_SCHEMA_VERSION,
            "context_count": len(contexts),
            "attempt_count": total_releases + self.rejected_count,
            "release_count": total_releases,
            "rejected_count": self.rejected_count,
            "spent_epsilon": total_epsilon,
            "spent_delta": total_delta,
            "contexts": contexts,
        }

    def render(self) -> dict[str, Any]:
        """Return counts-only evidence for callers that prefer a render verb."""

        return self.render_counts_only()

    def to_dict(self) -> dict[str, Any]:
        """Return the counts-only ledger evidence."""

        return self.render_counts_only()

    def to_json(self) -> str:
        """Return canonical JSON for the counts-only ledger evidence."""

        return json.dumps(
            self.render_counts_only(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def _spent(self, context: str) -> tuple[float, float]:
        spends = (spend for spend in self._spends if spend.context == context)
        return (
            math.fsum(spend.epsilon for spend in spends),
            math.fsum(spend.delta for spend in spends),
        )


def _safe_identifier(value: Any, *, field_name: str) -> str:
    """Validate a non-sensitive identifier without echoing its value."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a safe identifier")
    parsed = value.strip()
    if not _SAFE_CONTEXT_RE.fullmatch(parsed):
        raise ValueError(f"{field_name} must be a safe identifier")
    if any(pattern.search(parsed) for pattern in _PHI_PATTERNS):
        raise ValueError(f"{field_name} must not contain PHI-shaped data")
    return parsed


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return parsed


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be a finite number")
    return 0.0 if parsed == 0.0 else parsed


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


__all__ = [
    "PRIVACY_BUDGET_LEDGER_SCHEMA_VERSION",
    "PrivacyBudget",
    "PrivacyBudgetDecision",
    "PrivacyBudgetExceeded",
    "PrivacyBudgetLedger",
    "PrivacyBudgetSpend",
]
