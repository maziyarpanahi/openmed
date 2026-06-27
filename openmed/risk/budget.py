"""Per-document residual-risk budget evaluation."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

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
    "DEFAULT_POLICY_BUDGETS",
    "DEFAULT_QI_WEIGHTS",
    "DEFAULT_RISK_BUDGET",
    "RiskBudget",
    "RiskBudgetExceeded",
    "RiskBudgetVerdict",
    "RiskBudgetViolation",
    "budget_for_policy",
    "evaluate_budget",
]
