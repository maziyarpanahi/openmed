"""Severity-weighted, non-compensable omission gates for summaries.

The gate consumes value-free evidence coordinates. Evidence and importance
classes are identified by opaque SHA-256 identifiers so reports and errors do
not reproduce clinical facts or caller-defined labels.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

SUMMARY_OMISSION_BUDGET_SCHEMA_VERSION: Final[int] = 1
MANDATORY_OMISSION_REFUSAL: Final[str] = "mandatory_class_omitted"
CLASS_BUDGET_REFUSAL: Final[str] = "class_omission_budget_exceeded"

_OPAQUE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class SummaryOmissionBudgetError(ValueError):
    """Raised when an omission policy or evidence record is invalid."""


@dataclass(frozen=True, slots=True)
class ImportanceClassPolicy:
    """One caller-configured importance class.

    Args:
        class_id: Opaque digest identifying the policy class.
        severity_weight: Positive integer weight for each omitted fact.
        omission_limit: Maximum omissions tolerated in this class.
        mandatory: Whether every fact in this class must be represented.
    """

    class_id: str
    severity_weight: int
    omission_limit: int = 0
    mandatory: bool = False

    def __post_init__(self) -> None:
        _validate_opaque_id(self.class_id, "importance class")
        if type(self.severity_weight) is not int or self.severity_weight <= 0:
            raise SummaryOmissionBudgetError(
                "severity weight must be a positive integer"
            )
        if type(self.omission_limit) is not int or self.omission_limit < 0:
            raise SummaryOmissionBudgetError(
                "omission limit must be a non-negative integer"
            )
        if type(self.mandatory) is not bool:
            raise SummaryOmissionBudgetError("mandatory must be a boolean")
        if self.mandatory and self.omission_limit != 0:
            raise SummaryOmissionBudgetError(
                "mandatory importance classes require a zero omission limit"
            )


@dataclass(frozen=True, slots=True)
class SummaryEvidenceCoverage:
    """Value-free coverage status for one source fact."""

    evidence_id: str
    importance_class_id: str
    represented: bool

    def __post_init__(self) -> None:
        _validate_opaque_id(self.evidence_id, "evidence")
        _validate_opaque_id(self.importance_class_id, "importance class")
        if type(self.represented) is not bool:
            raise SummaryOmissionBudgetError("represented must be a boolean")


@dataclass(frozen=True, slots=True)
class ClassOmissionResult:
    """Counts-only result for one importance class."""

    class_id: str
    fact_count: int
    omitted_count: int
    severity_weight: int
    omission_limit: int
    mandatory: bool
    passed: bool

    @property
    def weighted_omissions(self) -> int:
        """Return the severity-weighted omitted-fact count."""

        return self.omitted_count * self.severity_weight

    @property
    def weighted_budget(self) -> int:
        """Return this class's severity-weighted omission allowance."""

        return self.omission_limit * self.severity_weight

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic value-free representation."""

        return {
            "class_id": self.class_id,
            "fact_count": self.fact_count,
            "mandatory": self.mandatory,
            "omission_limit": self.omission_limit,
            "omitted_count": self.omitted_count,
            "passed": self.passed,
            "severity_weight": self.severity_weight,
            "weighted_budget": self.weighted_budget,
            "weighted_omissions": self.weighted_omissions,
        }


@dataclass(frozen=True, slots=True)
class SummaryOmissionBudget:
    """Deterministic audit result for a summary omission gate."""

    classes: tuple[ClassOmissionResult, ...]
    fact_count: int
    omitted_count: int
    weighted_budget: int
    weighted_omissions: int
    passed: bool
    refusal_code: str | None
    schema_version: int = SUMMARY_OMISSION_BUDGET_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready report containing no evidence values."""

        return {
            "classes": [result.to_dict() for result in self.classes],
            "fact_count": self.fact_count,
            "omitted_count": self.omitted_count,
            "passed": self.passed,
            "refusal_code": self.refusal_code,
            "schema_version": self.schema_version,
            "weighted_budget": self.weighted_budget,
            "weighted_omissions": self.weighted_omissions,
        }

    def to_json(self) -> str:
        """Serialize the report with stable ordering and separators."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


def evaluate_summary_omission_budget(
    evidence: Iterable[SummaryEvidenceCoverage],
    importance_classes: Iterable[ImportanceClassPolicy],
) -> SummaryOmissionBudget:
    """Evaluate evidence coverage against independent per-class budgets.

    Every class is evaluated independently: surplus coverage in one class can
    never compensate for an over-budget class. Any omission in a mandatory
    class fails the gate even when aggregate weighted omissions appear low.

    Args:
        evidence: Value-free fact coverage records.
        importance_classes: Caller-configured class policies.

    Returns:
        A counts-only :class:`SummaryOmissionBudget` audit artifact.

    Raises:
        SummaryOmissionBudgetError: If the collection or its policy references
            are invalid. Errors never include caller-supplied identifiers.
    """

    policies = _materialize_policies(importance_classes)
    records = _materialize_evidence(evidence)
    policy_by_id = {policy.class_id: policy for policy in policies}

    if any(record.importance_class_id not in policy_by_id for record in records):
        raise SummaryOmissionBudgetError(
            "evidence references an unconfigured importance class"
        )

    seen_evidence: set[str] = set()
    totals = dict.fromkeys(policy_by_id, 0)
    omissions = dict.fromkeys(policy_by_id, 0)
    for record in records:
        if record.evidence_id in seen_evidence:
            raise SummaryOmissionBudgetError("duplicate evidence identifier")
        seen_evidence.add(record.evidence_id)
        totals[record.importance_class_id] += 1
        if not record.represented:
            omissions[record.importance_class_id] += 1

    class_results = tuple(
        ClassOmissionResult(
            class_id=policy.class_id,
            fact_count=totals[policy.class_id],
            omitted_count=omissions[policy.class_id],
            severity_weight=policy.severity_weight,
            omission_limit=policy.omission_limit,
            mandatory=policy.mandatory,
            passed=omissions[policy.class_id] <= policy.omission_limit,
        )
        for policy in sorted(policies, key=lambda item: item.class_id)
    )
    mandatory_omitted = any(
        result.mandatory and result.omitted_count > 0 for result in class_results
    )
    over_budget = any(not result.passed for result in class_results)
    refusal_code = (
        MANDATORY_OMISSION_REFUSAL
        if mandatory_omitted
        else CLASS_BUDGET_REFUSAL
        if over_budget
        else None
    )
    return SummaryOmissionBudget(
        classes=class_results,
        fact_count=len(records),
        omitted_count=sum(result.omitted_count for result in class_results),
        weighted_budget=sum(result.weighted_budget for result in class_results),
        weighted_omissions=sum(result.weighted_omissions for result in class_results),
        passed=refusal_code is None,
        refusal_code=refusal_code,
    )


def _materialize_policies(
    importance_classes: Iterable[ImportanceClassPolicy],
) -> tuple[ImportanceClassPolicy, ...]:
    if isinstance(importance_classes, (str, bytes, bytearray)):
        raise SummaryOmissionBudgetError("invalid importance class collection")
    try:
        policies = tuple(importance_classes)
    except Exception:
        raise SummaryOmissionBudgetError(
            "invalid importance class collection"
        ) from None
    if not policies or any(
        not isinstance(policy, ImportanceClassPolicy) for policy in policies
    ):
        raise SummaryOmissionBudgetError("invalid importance class collection")
    if len({policy.class_id for policy in policies}) != len(policies):
        raise SummaryOmissionBudgetError("duplicate importance class identifier")
    return policies


def _materialize_evidence(
    evidence: Iterable[SummaryEvidenceCoverage],
) -> tuple[SummaryEvidenceCoverage, ...]:
    if isinstance(evidence, (str, bytes, bytearray)):
        raise SummaryOmissionBudgetError("invalid evidence collection")
    try:
        records = tuple(evidence)
    except Exception:
        raise SummaryOmissionBudgetError("invalid evidence collection") from None
    if any(not isinstance(record, SummaryEvidenceCoverage) for record in records):
        raise SummaryOmissionBudgetError("invalid evidence collection")
    return records


def _validate_opaque_id(value: object, kind: str) -> None:
    if type(value) is not str or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise SummaryOmissionBudgetError(f"invalid {kind} identifier")


__all__ = [
    "CLASS_BUDGET_REFUSAL",
    "MANDATORY_OMISSION_REFUSAL",
    "SUMMARY_OMISSION_BUDGET_SCHEMA_VERSION",
    "ClassOmissionResult",
    "ImportanceClassPolicy",
    "SummaryEvidenceCoverage",
    "SummaryOmissionBudget",
    "SummaryOmissionBudgetError",
    "evaluate_summary_omission_budget",
]
