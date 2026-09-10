"""Deterministic privacy-budget accounting for aggregate releases.

The ledger records only bounded release-context identifiers, numeric
epsilon/delta values, and aggregate counts. It never accepts a source payload
or makes a network call. Charges are atomic so concurrent release workers
cannot collectively pass the same remaining budget.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from threading import RLock
from types import MappingProxyType
from typing import Any, Final

PRIVACY_BUDGET_LEDGER_SCHEMA_VERSION: Final = "openmed.privacy_budget_ledger.v1"
MAX_PRIVACY_BUDGET_CONTEXTS: Final = 512
MAX_PRIVACY_BUDGET_SPENDS: Final = 10_000
MAX_PRIVACY_BUDGET_EPSILON: Final = 1_000_000.0

_MAX_COUNT: Final = (1 << 63) - 1
_SAFE_CONTEXT_RE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,63}$")
_PHI_PATTERNS: Final = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{3}[-.)]\d{3}[-.]\d{4}\b"),
)
_DECISION_REASONS: Final = (
    "within budget",
    "epsilon and delta exceed budget",
    "epsilon exceeds budget",
    "delta exceeds budget",
)
_BUDGET_FIELDS: Final = frozenset({"epsilon", "delta", "max_epsilon", "max_delta"})


def _snapshot_mapping(
    value: Mapping[Any, Any], *, field_name: str, max_items: int
) -> dict[str, Any]:
    try:
        items = list(itertools.islice(value.items(), max_items + 1))
    except Exception:  # noqa: BLE001 - mappings are caller-controlled protocols.
        raise ValueError(f"{field_name} could not be read") from None
    if len(items) > max_items:
        raise ValueError(f"{field_name} exceeds the supported item limit")

    result: dict[str, Any] = {}
    for item in items:
        if type(item) not in {list, tuple} or len(item) != 2:
            raise ValueError(f"{field_name} contains an invalid entry")
        key, field_value = item
        if type(key) is not str:
            raise ValueError(f"{field_name} keys must be strings")
        if key in result:
            raise ValueError(f"{field_name} keys must be unique")
        result[key] = field_value
    return result


@dataclass(frozen=True)
class ReleaseContextPrivacyBudget:
    """Cumulative epsilon and delta ceiling for one release context."""

    epsilon: float
    delta: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "epsilon",
            _epsilon_float(self.epsilon, field_name="epsilon"),
        )
        object.__setattr__(
            self,
            "delta",
            _delta_float(self.delta, field_name="delta"),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ReleaseContextPrivacyBudget:
        """Build a ceiling from a closed JSON-style mapping."""

        if not isinstance(payload, Mapping):
            raise TypeError("privacy budget must be a mapping")
        values = _snapshot_mapping(
            payload,
            field_name="privacy budget",
            max_items=len(_BUDGET_FIELDS),
        )
        if set(values) - _BUDGET_FIELDS:
            raise ValueError("privacy budget contains unsupported fields")
        epsilon = _one_alias(values, "epsilon", "max_epsilon", field_name="epsilon")
        delta = _one_alias(values, "delta", "max_delta", field_name="delta")
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
    """One accepted, aggregate-only release charge."""

    sequence: int
    context: str
    epsilon: float
    delta: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence",
            _positive_int(
                self.sequence,
                field_name="sequence",
                maximum=MAX_PRIVACY_BUDGET_SPENDS,
            ),
        )
        object.__setattr__(
            self,
            "context",
            _safe_identifier(self.context, field_name="context"),
        )
        object.__setattr__(
            self,
            "epsilon",
            _epsilon_float(self.epsilon, field_name="epsilon"),
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

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise ValueError("allowed must be a boolean")
        if type(self.reason) is not str:
            raise ValueError("reason must be a supported decision reason")
        context = _safe_identifier(self.context, field_name="context")
        requested_epsilon = _epsilon_float(
            self.requested_epsilon, field_name="requested_epsilon"
        )
        requested_delta = _delta_float(
            self.requested_delta, field_name="requested_delta"
        )
        projected_epsilon = _non_negative_float(
            self.projected_epsilon,
            field_name="projected_epsilon",
            maximum=MAX_PRIVACY_BUDGET_EPSILON * 2,
        )
        projected_delta = _non_negative_float(
            self.projected_delta,
            field_name="projected_delta",
            maximum=2.0,
        )
        max_epsilon = _epsilon_float(self.max_epsilon, field_name="max_epsilon")
        max_delta = _delta_float(self.max_delta, field_name="max_delta")
        if projected_epsilon < requested_epsilon or projected_delta < requested_delta:
            raise ValueError("privacy budget decision totals are inconsistent")

        epsilon_ok = projected_epsilon <= max_epsilon
        delta_ok = projected_delta <= max_delta
        expected_allowed = epsilon_ok and delta_ok
        expected_reason = _decision_reason(epsilon_ok, delta_ok)
        if self.allowed is not expected_allowed or self.reason != expected_reason:
            raise ValueError("privacy budget decision outcome is inconsistent")

        object.__setattr__(self, "context", context)
        object.__setattr__(self, "requested_epsilon", requested_epsilon)
        object.__setattr__(self, "requested_delta", requested_delta)
        object.__setattr__(self, "projected_epsilon", projected_epsilon)
        object.__setattr__(self, "projected_delta", projected_delta)
        object.__setattr__(self, "max_epsilon", max_epsilon)
        object.__setattr__(self, "max_delta", max_delta)
        object.__setattr__(
            self,
            "remaining_epsilon",
            float(max(Decimal(0), _decimal(max_epsilon) - _decimal(projected_epsilon))),
        )
        object.__setattr__(
            self,
            "remaining_delta",
            float(max(Decimal(0), _decimal(max_delta) - _decimal(projected_delta))),
        )

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


class PrivacyBudgetLedgerExceeded(ValueError):
    """Raised when a release would exceed its named context budget."""

    def __init__(self, decision: PrivacyBudgetDecision) -> None:
        if type(decision) is not PrivacyBudgetDecision:
            raise TypeError("decision must be a PrivacyBudgetDecision")
        self.decision = decision
        super().__init__(
            "privacy budget ledger exceeded: "
            f"epsilon={decision.projected_epsilon:.6g}/{decision.max_epsilon:.6g}, "
            f"delta={decision.projected_delta:.6g}/{decision.max_delta:.6g}"
        )


class PrivacyBudgetLedger:
    """Thread-safe ledger for named aggregate-release privacy budgets."""

    def __init__(
        self,
        budgets: Mapping[str, ReleaseContextPrivacyBudget | Mapping[str, Any]]
        | None = None,
    ) -> None:
        self._lock = RLock()
        self._budgets: dict[str, ReleaseContextPrivacyBudget] = {}
        self._spends: list[PrivacyBudgetSpend] = []
        self._totals: dict[str, tuple[Decimal, Decimal]] = {}
        self._release_counts: dict[str, int] = {}
        self._rejected: dict[str, int] = {}

        if budgets is None:
            return
        if not isinstance(budgets, Mapping):
            raise TypeError("budgets must be a mapping")
        values = _snapshot_mapping(
            budgets,
            field_name="privacy budget contexts",
            max_items=MAX_PRIVACY_BUDGET_CONTEXTS,
        )
        for context, budget in values.items():
            key = _safe_identifier(context, field_name="context")
            self._budgets[key] = (
                budget
                if type(budget) is ReleaseContextPrivacyBudget
                else ReleaseContextPrivacyBudget.from_mapping(budget)
            )
            self._totals[key] = (Decimal(0), Decimal(0))
            self._release_counts[key] = 0
            self._rejected[key] = 0

    @classmethod
    def from_budgets(
        cls,
        budgets: Mapping[str, ReleaseContextPrivacyBudget | Mapping[str, Any]],
    ) -> PrivacyBudgetLedger:
        """Construct a ledger from named epsilon/delta budgets."""

        return cls(budgets=budgets)

    @property
    def budgets(self) -> Mapping[str, ReleaseContextPrivacyBudget]:
        """Return an immutable snapshot of configured context budgets."""

        with self._lock:
            return MappingProxyType(dict(self._budgets))

    @property
    def contexts(self) -> tuple[str, ...]:
        """Return registered release contexts in deterministic order."""

        with self._lock:
            return tuple(sorted(self._budgets))

    @property
    def spends(self) -> tuple[PrivacyBudgetSpend, ...]:
        """Return accepted spends in charge order."""

        with self._lock:
            return tuple(self._spends)

    @property
    def ledger(self) -> tuple[PrivacyBudgetSpend, ...]:
        """Return accepted spends using the conventional ledger name."""

        return self.spends

    @property
    def rejected_count(self) -> int:
        """Return the aggregate count of refused requests."""

        with self._lock:
            return sum(self._rejected.values())

    def register_context(
        self,
        context: str,
        epsilon: float,
        delta: float,
    ) -> ReleaseContextPrivacyBudget:
        """Register or safely replace one context budget."""

        key = _safe_identifier(context, field_name="context")
        budget = ReleaseContextPrivacyBudget(epsilon=epsilon, delta=delta)
        with self._lock:
            if (
                key not in self._budgets
                and len(self._budgets) >= MAX_PRIVACY_BUDGET_CONTEXTS
            ):
                raise ValueError("privacy budget context limit reached")
            current_epsilon, current_delta = self._totals.get(
                key, (Decimal(0), Decimal(0))
            )
            if (
                _decimal(budget.epsilon) < current_epsilon
                or _decimal(budget.delta) < current_delta
            ):
                raise ValueError("privacy budget cannot be below recorded spend")
            self._budgets[key] = budget
            self._totals.setdefault(key, (Decimal(0), Decimal(0)))
            self._release_counts.setdefault(key, 0)
            self._rejected.setdefault(key, 0)
        return budget

    def set_budget(
        self,
        context: str,
        epsilon: float,
        delta: float,
    ) -> ReleaseContextPrivacyBudget:
        """Alias for :meth:`register_context`."""

        return self.register_context(context, epsilon, delta)

    def budget_for(self, context: str) -> ReleaseContextPrivacyBudget:
        """Return the budget for a safe context or raise ``KeyError``."""

        key = _safe_identifier(context, field_name="context")
        with self._lock:
            try:
                return self._budgets[key]
            except KeyError:
                raise KeyError(
                    "no privacy budget registered for release context"
                ) from None

    def check(
        self,
        context: str,
        epsilon: float,
        delta: float,
    ) -> PrivacyBudgetDecision:
        """Check a proposed spend without changing the ledger."""

        key = _safe_identifier(context, field_name="context")
        requested_epsilon = _epsilon_float(epsilon, field_name="epsilon")
        requested_delta = _delta_float(delta, field_name="delta")
        with self._lock:
            return self._check_unlocked(key, requested_epsilon, requested_delta)

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
        """Atomically record an accepted release or raise before mutation."""

        key = _safe_identifier(context, field_name="context")
        requested_epsilon = _epsilon_float(epsilon, field_name="epsilon")
        requested_delta = _delta_float(delta, field_name="delta")
        with self._lock:
            decision = self._check_unlocked(key, requested_epsilon, requested_delta)
            if not decision.allowed:
                rejected = self._rejected.get(key, 0)
                if rejected >= _MAX_COUNT:
                    raise OverflowError("privacy budget rejection count limit reached")
                self._rejected[key] = rejected + 1
                raise PrivacyBudgetLedgerExceeded(decision)
            if len(self._spends) >= MAX_PRIVACY_BUDGET_SPENDS:
                raise OverflowError("privacy budget spend limit reached")

            sequence = len(self._spends) + 1
            self._spends.append(
                PrivacyBudgetSpend(
                    sequence=sequence,
                    context=key,
                    epsilon=requested_epsilon,
                    delta=requested_delta,
                )
            )
            current_epsilon, current_delta = self._totals[key]
            with localcontext() as context_decimal:
                context_decimal.prec = 64
                self._totals[key] = (
                    current_epsilon + _decimal(requested_epsilon),
                    current_delta + _decimal(requested_delta),
                )
            self._release_counts[key] += 1
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
        """Render deterministic aggregate evidence without release payloads."""

        with self._lock:
            contexts: dict[str, dict[str, Any]] = {}
            total_epsilon = 0.0
            total_delta = 0.0
            total_releases = 0
            total_rejected = 0
            for context in sorted(self._budgets):
                budget = self._budgets[context]
                spent_epsilon_decimal, spent_delta_decimal = self._totals[context]
                spent_epsilon = float(spent_epsilon_decimal)
                spent_delta = float(spent_delta_decimal)
                release_count = self._release_counts[context]
                rejected_count = self._rejected[context]
                contexts[context] = {
                    "attempt_count": release_count + rejected_count,
                    "release_count": release_count,
                    "rejected_count": rejected_count,
                    "spent_epsilon": spent_epsilon,
                    "spent_delta": spent_delta,
                    "budget_epsilon": budget.epsilon,
                    "budget_delta": budget.delta,
                    "remaining_epsilon": float(
                        max(
                            Decimal(0),
                            _decimal(budget.epsilon) - spent_epsilon_decimal,
                        )
                    ),
                    "remaining_delta": float(
                        max(
                            Decimal(0),
                            _decimal(budget.delta) - spent_delta_decimal,
                        )
                    ),
                }
                total_epsilon = math.fsum((total_epsilon, spent_epsilon))
                total_delta = math.fsum((total_delta, spent_delta))
                total_releases += release_count
                total_rejected += rejected_count

            return {
                "schema_version": PRIVACY_BUDGET_LEDGER_SCHEMA_VERSION,
                "context_count": len(contexts),
                "attempt_count": total_releases + total_rejected,
                "release_count": total_releases,
                "rejected_count": total_rejected,
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

    def _check_unlocked(
        self, context: str, epsilon: float, delta: float
    ) -> PrivacyBudgetDecision:
        try:
            budget = self._budgets[context]
        except KeyError:
            raise KeyError("no privacy budget registered for release context") from None
        current_epsilon, current_delta = self._totals[context]
        with localcontext() as context_decimal:
            context_decimal.prec = 64
            projected_epsilon_decimal = current_epsilon + _decimal(epsilon)
            projected_delta_decimal = current_delta + _decimal(delta)
        epsilon_ok = projected_epsilon_decimal <= _decimal(budget.epsilon)
        delta_ok = projected_delta_decimal <= _decimal(budget.delta)
        projected_epsilon = float(projected_epsilon_decimal)
        projected_delta = float(projected_delta_decimal)
        allowed = epsilon_ok and delta_ok
        return PrivacyBudgetDecision(
            allowed=allowed,
            context=context,
            requested_epsilon=epsilon,
            requested_delta=delta,
            projected_epsilon=projected_epsilon,
            projected_delta=projected_delta,
            max_epsilon=budget.epsilon,
            max_delta=budget.delta,
            remaining_epsilon=float(
                max(Decimal(0), _decimal(budget.epsilon) - projected_epsilon_decimal)
            ),
            remaining_delta=float(
                max(Decimal(0), _decimal(budget.delta) - projected_delta_decimal)
            ),
            reason=_decision_reason(epsilon_ok, delta_ok),
        )


def _one_alias(
    values: Mapping[str, Any], primary: str, alias: str, *, field_name: str
) -> Any:
    matches = [name for name in (primary, alias) if name in values]
    if not matches:
        raise ValueError(f"privacy budget requires {field_name}")
    if len(matches) > 1:
        raise ValueError("privacy budget must not use duplicate aliases")
    return values[matches[0]]


def _safe_identifier(value: Any, *, field_name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a safe identifier")
    if any(pattern.search(value) for pattern in _PHI_PATTERNS):
        raise ValueError(f"{field_name} must not contain PHI-shaped data")
    if _SAFE_CONTEXT_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a safe identifier")
    return value


def _positive_int(value: Any, *, field_name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{field_name} must be a bounded positive integer")
    return value


def _non_negative_float(value: Any, *, field_name: str, maximum: float) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{field_name} must be a finite number")
    try:
        parsed = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{field_name} must be a finite number") from None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= maximum:
        raise ValueError(f"{field_name} must be a bounded non-negative number")
    return 0.0 if parsed == 0.0 else parsed


def _epsilon_float(value: Any, *, field_name: str) -> float:
    return _non_negative_float(
        value, field_name=field_name, maximum=MAX_PRIVACY_BUDGET_EPSILON
    )


def _delta_float(value: Any, *, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name=field_name, maximum=1.0)
    if parsed >= 1.0:
        raise ValueError(f"{field_name} must be less than 1")
    return parsed


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _decision_reason(epsilon_ok: bool, delta_ok: bool) -> str:
    if epsilon_ok and delta_ok:
        return _DECISION_REASONS[0]
    if not epsilon_ok and not delta_ok:
        return _DECISION_REASONS[1]
    if not epsilon_ok:
        return _DECISION_REASONS[2]
    return _DECISION_REASONS[3]


__all__ = [
    "MAX_PRIVACY_BUDGET_CONTEXTS",
    "MAX_PRIVACY_BUDGET_EPSILON",
    "MAX_PRIVACY_BUDGET_SPENDS",
    "PRIVACY_BUDGET_LEDGER_SCHEMA_VERSION",
    "PrivacyBudgetDecision",
    "PrivacyBudgetLedger",
    "PrivacyBudgetLedgerExceeded",
    "PrivacyBudgetSpend",
    "ReleaseContextPrivacyBudget",
]
