"""Human-review routing for uncertain, conflicting, or refused SDOH evidence."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .sdoh_deduplicate import SDOHSourceReference

SDOH_REVIEW_SCHEMA_VERSION: Final = 1
SDOH_REVIEW_ADVISORY: Final = (
    "SDOH review routes must never drive automatic eligibility, diagnosis, "
    "care denial, or underwriting decisions."
)

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")


class SDOHOutcome(str, Enum):
    """Controlled evidence outcomes, including non-answers."""

    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNKNOWN = "unknown"
    DECLINED = "declined"
    REFUSED = "refused"


class SDOHReviewState(str, Enum):
    """Typed reasons that require human review."""

    LOW_CONFIDENCE = "low_confidence"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"
    DECLINED = "declined"
    REFUSED = "refused"


@dataclass(frozen=True, slots=True)
class SDOHReviewEvidence:
    """Value-free SDOH evidence supplied to the review router."""

    evidence_id: str
    category: str
    outcome: SDOHOutcome
    confidence: float
    source_references: tuple[SDOHSourceReference, ...]
    conflicting: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id))
        object.__setattr__(self, "category", _identifier(self.category))
        if not isinstance(self.outcome, SDOHOutcome):
            raise TypeError("outcome must be an SDOHOutcome")
        object.__setattr__(self, "confidence", _probability(self.confidence))
        references = tuple(self.source_references)
        if not references:
            raise ValueError("source_references must not be empty")
        if any(not isinstance(item, SDOHSourceReference) for item in references):
            raise TypeError("source_references must contain SDOHSourceReference values")
        if type(self.conflicting) is not bool:
            raise TypeError("conflicting must be a boolean")
        object.__setattr__(self, "source_references", references)


@dataclass(frozen=True, slots=True)
class SDOHReviewItem:
    """A typed, source-linked review queue item."""

    evidence_id: str
    category: str
    outcome: SDOHOutcome
    states: tuple[SDOHReviewState, ...]
    source_references: tuple[SDOHSourceReference, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free queue item."""

        return {
            "evidence_id": self.evidence_id,
            "category": self.category,
            "outcome": self.outcome.value,
            "review_states": [state.value for state in self.states],
            "source_references": [
                reference.to_dict() for reference in self.source_references
            ],
            "automated_decision_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class SDOHReviewQueueSummary:
    """Counts-only review queue summary with no item identifiers or sources."""

    total_count: int
    state_counts: tuple[tuple[str, int], ...]
    category_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return counts only."""

        return {
            "total_count": self.total_count,
            "state_counts": dict(self.state_counts),
            "category_counts": dict(self.category_counts),
        }


@dataclass(frozen=True, slots=True)
class SDOHReviewQueue:
    """Deterministically ordered SDOH review items."""

    items: tuple[SDOHReviewItem, ...]

    def summary(self) -> SDOHReviewQueueSummary:
        """Return aggregate counts without evidence or source identifiers."""

        states = Counter(state.value for item in self.items for state in item.states)
        categories = Counter(item.category for item in self.items)
        return SDOHReviewQueueSummary(
            total_count=len(self.items),
            state_counts=tuple(sorted(states.items())),
            category_counts=tuple(sorted(categories.items())),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the full value-free review queue."""

        return {
            "schema_version": SDOH_REVIEW_SCHEMA_VERSION,
            "items": [item.to_dict() for item in self.items],
            "summary": self.summary().to_dict(),
            "advisory": SDOH_REVIEW_ADVISORY,
        }


def route_sdoh_review(
    evidence: Iterable[SDOHReviewEvidence],
    *,
    confidence_threshold: float = 0.75,
) -> SDOHReviewQueue:
    """Route evidence requiring review while preserving explicit outcomes."""

    threshold = _probability(confidence_threshold)
    items: list[SDOHReviewItem] = []
    for record in evidence:
        if not isinstance(record, SDOHReviewEvidence):
            raise TypeError("evidence must contain SDOHReviewEvidence values")
        states: set[SDOHReviewState] = set()
        if record.confidence < threshold:
            states.add(SDOHReviewState.LOW_CONFIDENCE)
        if record.conflicting:
            states.add(SDOHReviewState.CONFLICTING)
        outcome_state = {
            SDOHOutcome.UNKNOWN: SDOHReviewState.UNKNOWN,
            SDOHOutcome.DECLINED: SDOHReviewState.DECLINED,
            SDOHOutcome.REFUSED: SDOHReviewState.REFUSED,
        }.get(record.outcome)
        if outcome_state is not None:
            states.add(outcome_state)
        if states:
            items.append(
                SDOHReviewItem(
                    evidence_id=record.evidence_id,
                    category=record.category,
                    outcome=record.outcome,
                    states=tuple(sorted(states, key=lambda item: item.value)),
                    source_references=record.source_references,
                )
            )
    return SDOHReviewQueue(
        items=tuple(sorted(items, key=lambda item: (item.category, item.evidence_id)))
    )


def _identifier(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("identifier must be a string")
    if _SAFE_IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError("identifier must use the safe opaque-reference format")
    return value


def _probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("confidence must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be between zero and one")
    return result


__all__ = [
    "SDOH_REVIEW_ADVISORY",
    "SDOH_REVIEW_SCHEMA_VERSION",
    "SDOHOutcome",
    "SDOHReviewEvidence",
    "SDOHReviewItem",
    "SDOHReviewQueue",
    "SDOHReviewQueueSummary",
    "SDOHReviewState",
    "route_sdoh_review",
]
