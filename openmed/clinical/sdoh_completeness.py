"""Counts-only SDOH category processing completeness audits."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

SDOH_COMPLETENESS_SCHEMA_VERSION: Final = 1
SDOH_COMPLETENESS_ADVISORY: Final = (
    "A processed category with zero findings is unmentioned, not a negative "
    "social-determinant finding."
)
MISSING_PROCESSING_RESULT: Final = "missing_processing_result"

_SAFE_CODE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


class SDOHCategoryState(str, Enum):
    """Processing states for every configured SDOH category."""

    PROCESSED = "processed"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SDOHCategoryResult:
    """Value-free processing result for one configured category."""

    category: str
    state: SDOHCategoryState
    finding_count: int = 0
    reason_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", _code(self.category))
        if not isinstance(self.state, SDOHCategoryState):
            raise TypeError("state must be an SDOHCategoryState")
        if isinstance(self.finding_count, bool) or not isinstance(
            self.finding_count, int
        ):
            raise TypeError("finding_count must be an integer")
        if self.finding_count < 0:
            raise ValueError("finding_count must not be negative")
        if self.state is not SDOHCategoryState.PROCESSED and self.finding_count:
            raise ValueError("unprocessed categories cannot contain findings")
        if self.state is SDOHCategoryState.PROCESSED and self.reason_code is not None:
            raise ValueError("processed categories must not have a reason code")
        if self.state is not SDOHCategoryState.PROCESSED:
            if self.reason_code is None:
                raise ValueError("unprocessed categories require a reason code")
            object.__setattr__(self, "reason_code", _code(self.reason_code))

    @property
    def absence_interpretation(self) -> str | None:
        """Explain zero findings without manufacturing a negative finding."""

        if self.state is SDOHCategoryState.PROCESSED and self.finding_count == 0:
            return "unmentioned_not_negative"
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic category audit record."""

        return {
            "category": self.category,
            "state": self.state.value,
            "finding_count": self.finding_count,
            "reason_code": self.reason_code,
            "absence_interpretation": self.absence_interpretation,
        }


@dataclass(frozen=True, slots=True)
class SDOHCompletenessAudit:
    """Complete, counts-only audit for all configured categories."""

    categories: tuple[SDOHCategoryResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return category states, counts, and controlled reason codes."""

        states = Counter(item.state.value for item in self.categories)
        reasons = Counter(
            item.reason_code for item in self.categories if item.reason_code is not None
        )
        return {
            "schema_version": SDOH_COMPLETENESS_SCHEMA_VERSION,
            "category_count": len(self.categories),
            "state_counts": {
                state.value: states[state.value] for state in SDOHCategoryState
            },
            "reason_counts": dict(sorted(reasons.items())),
            "categories": [item.to_dict() for item in self.categories],
            "advisory": SDOH_COMPLETENESS_ADVISORY,
        }


def audit_sdoh_completeness(
    configured_categories: Iterable[str],
    results: Iterable[SDOHCategoryResult],
) -> SDOHCompletenessAudit:
    """Audit every configured category and fail missing reports explicitly."""

    configured = tuple(sorted({_code(category) for category in configured_categories}))
    by_category: dict[str, SDOHCategoryResult] = {}
    for result in results:
        if not isinstance(result, SDOHCategoryResult):
            raise TypeError("results must contain SDOHCategoryResult values")
        if result.category in by_category:
            raise ValueError("each category may have only one processing result")
        by_category[result.category] = result
    if not set(by_category).issubset(configured):
        raise ValueError("processing results contain an unconfigured category")
    completed = tuple(
        by_category.get(category)
        or SDOHCategoryResult(
            category=category,
            state=SDOHCategoryState.FAILED,
            reason_code=MISSING_PROCESSING_RESULT,
        )
        for category in configured
    )
    return SDOHCompletenessAudit(categories=completed)


def _code(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("category and reason codes must be strings")
    if _SAFE_CODE_RE.fullmatch(value) is None:
        raise ValueError(
            "category and reason codes must use the controlled-code format"
        )
    return value


__all__ = [
    "MISSING_PROCESSING_RESULT",
    "SDOH_COMPLETENESS_ADVISORY",
    "SDOH_COMPLETENESS_SCHEMA_VERSION",
    "SDOHCategoryResult",
    "SDOHCategoryState",
    "SDOHCompletenessAudit",
    "audit_sdoh_completeness",
]
