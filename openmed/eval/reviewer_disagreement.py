"""Privacy-safe aggregate metrics for reviewer disagreement.

The public report intentionally excludes reviewer, case, and decision values.
Cells are published only when both sides of a binary rate meet the configured
minimum size (zero-valued sides are safe). This complementary suppression
prevents a small cell from being reconstructed from a published total.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

REVIEWER_DISAGREEMENT_SCHEMA_VERSION = "openmed.reviewer_disagreement.v1"
DEFAULT_MINIMUM_CELL_SIZE = 5


class DisagreementReason(str, Enum):
    """Bounded reason codes for a case-level reviewer disagreement."""

    CLINICAL_INTERPRETATION = "clinical_interpretation"
    EVIDENCE_QUALITY = "evidence_quality"
    GUIDELINE_AMBIGUITY = "guideline_ambiguity"
    LABEL_DEFINITION = "label_definition"
    REVIEWER_ERROR = "reviewer_error"
    OTHER = "other"


@dataclass(frozen=True, slots=True, repr=False)
class ReviewerDecision:
    """One pseudonymous review decision.

    ``case_id`` and ``reviewer_id`` must already be pseudonymous. They are used
    only for in-memory grouping and are never copied into reports or exception
    messages. A disagreement reason and adjudication flag are case-level facts,
    repeated on each decision for the case so inconsistent inputs fail closed.
    """

    case_id: str
    reviewer_id: str
    decision: str
    reason: DisagreementReason | None = None
    adjudicated: bool = False

    def __post_init__(self) -> None:
        """Validate the input shape without reflecting values in errors."""

        _require_nonempty_string(self.case_id, "case identifier")
        _require_nonempty_string(self.reviewer_id, "reviewer identifier")
        _require_nonempty_string(self.decision, "decision code")
        if self.reason is not None and not isinstance(self.reason, DisagreementReason):
            raise TypeError("reason must be a DisagreementReason or None")
        if type(self.adjudicated) is not bool:
            raise TypeError("adjudicated must be a boolean")

    def __repr__(self) -> str:
        """Return a value-free representation suitable for diagnostics."""

        return "ReviewerDecision(<redacted>)"


@dataclass(frozen=True, slots=True)
class PublishedRate:
    """One binary aggregate after minimum-cell suppression."""

    rate: float | None
    numerator: int | None
    denominator: int | None
    suppressed: bool

    def to_dict(self) -> dict[str, bool | float | int | None]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "denominator": self.denominator,
            "numerator": self.numerator,
            "rate": self.rate,
            "suppressed": self.suppressed,
        }


@dataclass(frozen=True, slots=True)
class ReasonAdjudicationRate:
    """Publishable adjudication rate for one typed disagreement reason."""

    reason: DisagreementReason
    rate: PublishedRate

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {"reason": self.reason.value, **self.rate.to_dict()}


@dataclass(frozen=True, slots=True)
class ReviewerDisagreementReport:
    """Counts-only reviewer agreement and adjudication report."""

    minimum_cell_size: int
    agreement: PublishedRate
    adjudication: PublishedRate
    reason_adjudication: tuple[ReasonAdjudicationRate, ...]
    suppressed_reason_cells: int
    schema_version: str = REVIEWER_DISAGREEMENT_SCHEMA_VERSION

    @property
    def has_suppressed_cells(self) -> bool:
        """Return whether any overall or reason cell was suppressed."""

        return (
            self.agreement.suppressed
            or self.adjudication.suppressed
            or self.suppressed_reason_cells > 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report containing no input identifiers."""

        return {
            "adjudication": self.adjudication.to_dict(),
            "agreement": self.agreement.to_dict(),
            "has_suppressed_cells": self.has_suppressed_cells,
            "minimum_cell_size": self.minimum_cell_size,
            "reason_adjudication": [
                item.to_dict() for item in self.reason_adjudication
            ],
            "schema_version": self.schema_version,
            "suppressed_reason_cells": self.suppressed_reason_cells,
        }


@dataclass(frozen=True, slots=True)
class _CaseResult:
    agreed: bool
    adjudicated: bool
    reason: DisagreementReason | None


def _require_nonempty_string(value: Any, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _validate_minimum_cell_size(value: int) -> int:
    if type(value) is not int or value < 2:
        raise ValueError("minimum_cell_size must be an integer >= 2")
    return value


def _group_cases(decisions: Iterable[ReviewerDecision]) -> tuple[_CaseResult, ...]:
    grouped: dict[str, list[ReviewerDecision]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    found = False

    for item in decisions:
        found = True
        if not isinstance(item, ReviewerDecision):
            raise TypeError("decisions must contain ReviewerDecision records")
        key = (item.case_id, item.reviewer_id)
        if key in seen:
            raise ValueError("each reviewer may decide a case only once")
        seen.add(key)
        grouped[item.case_id].append(item)

    if not found:
        raise ValueError("at least one reviewer decision is required")

    cases: list[_CaseResult] = []
    for rows in grouped.values():
        if len(rows) < 2:
            raise ValueError("each case requires at least two reviewer decisions")

        adjudication_flags = {row.adjudicated for row in rows}
        if len(adjudication_flags) != 1:
            raise ValueError("adjudication status must be consistent within a case")

        agreed = len({row.decision for row in rows}) == 1
        reasons = {row.reason for row in rows}
        adjudicated = rows[0].adjudicated
        if agreed:
            if reasons != {None} or adjudicated:
                raise ValueError(
                    "agreed cases cannot have a disagreement reason or adjudication"
                )
            reason = None
        else:
            if None in reasons or len(reasons) != 1:
                raise ValueError(
                    "disagreement cases require one consistent typed reason"
                )
            reason = next(iter(reasons))

        cases.append(
            _CaseResult(
                agreed=agreed,
                adjudicated=adjudicated,
                reason=reason,
            )
        )
    return tuple(cases)


def _publish_rate(
    numerator: int,
    denominator: int,
    *,
    minimum_cell_size: int,
) -> PublishedRate:
    if denominator == 0:
        return PublishedRate(
            rate=None,
            numerator=0,
            denominator=0,
            suppressed=False,
        )

    complement = denominator - numerator
    has_small_cell = any(
        0 < count < minimum_cell_size for count in (numerator, complement)
    )
    if denominator < minimum_cell_size or has_small_cell:
        return PublishedRate(
            rate=None,
            numerator=None,
            denominator=None,
            suppressed=True,
        )
    return PublishedRate(
        rate=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
        suppressed=False,
    )


def reviewer_disagreement_report(
    decisions: Iterable[ReviewerDecision],
    *,
    minimum_cell_size: int = DEFAULT_MINIMUM_CELL_SIZE,
) -> ReviewerDisagreementReport:
    """Aggregate pseudonymous reviewer decisions into privacy-safe rates.

    Agreement is the fraction of cases with unanimous reviewer decisions.
    Adjudication is the fraction of disagreement cases marked adjudicated.
    Reason cells use the latter definition within each typed reason.

    A binary rate is suppressed when its denominator is smaller than
    ``minimum_cell_size`` or either nonzero side of the binary split is smaller
    than that threshold. This complementary rule prevents reconstruction by
    subtraction. Zero-valued cells are safe and an empty adjudication
    denominator is represented as ``0 / 0`` with a ``None`` rate.

    Args:
        decisions: Pseudonymous reviewer decisions. Every case must have at
            least two distinct reviewers. Disagreement cases must carry one
            consistent typed reason and adjudication status.
        minimum_cell_size: Smallest publishable nonzero aggregate cell. Must be
            an integer of at least two.

    Returns:
        A deterministic report that omits reviewer, case, and decision values.

    Raises:
        TypeError: If a decision record or one of its fields has the wrong type.
        ValueError: If the input is empty, internally inconsistent, duplicated,
            or uses an invalid minimum cell size.
    """

    threshold = _validate_minimum_cell_size(minimum_cell_size)
    cases = _group_cases(decisions)

    agreed_count = sum(case.agreed for case in cases)
    disagreement_cases = tuple(case for case in cases if not case.agreed)
    adjudicated_count = sum(case.adjudicated for case in disagreement_cases)

    agreement = _publish_rate(
        agreed_count,
        len(cases),
        minimum_cell_size=threshold,
    )
    adjudication = _publish_rate(
        adjudicated_count,
        len(disagreement_cases),
        minimum_cell_size=threshold,
    )

    by_reason: dict[DisagreementReason, list[_CaseResult]] = defaultdict(list)
    for case in disagreement_cases:
        if case.reason is not None:
            by_reason[case.reason].append(case)

    reason_rates: list[ReasonAdjudicationRate] = []
    suppressed_reason_cells = 0
    for reason in sorted(by_reason, key=lambda item: item.value):
        reason_cases = by_reason[reason]
        rate = _publish_rate(
            sum(case.adjudicated for case in reason_cases),
            len(reason_cases),
            minimum_cell_size=threshold,
        )
        if rate.suppressed:
            suppressed_reason_cells += 1
        else:
            reason_rates.append(ReasonAdjudicationRate(reason=reason, rate=rate))

    return ReviewerDisagreementReport(
        minimum_cell_size=threshold,
        agreement=agreement,
        adjudication=adjudication,
        reason_adjudication=tuple(reason_rates),
        suppressed_reason_cells=suppressed_reason_cells,
    )


__all__ = [
    "DEFAULT_MINIMUM_CELL_SIZE",
    "REVIEWER_DISAGREEMENT_SCHEMA_VERSION",
    "DisagreementReason",
    "PublishedRate",
    "ReasonAdjudicationRate",
    "ReviewerDecision",
    "ReviewerDisagreementReport",
    "reviewer_disagreement_report",
]
