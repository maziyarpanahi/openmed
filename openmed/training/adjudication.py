"""Adjudication queue primitives for weak-labeling disagreements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class AdjudicationCandidate:
    start: int
    end: int
    label: str
    text: str = ""
    score: float = 0.0
    sources: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "end": self.end,
            "label": self.label,
            "metadata": dict(self.metadata),
            "score": self.score,
            "sources": list(self.sources),
            "start": self.start,
            "text": self.text,
        }


@dataclass(frozen=True)
class AdjudicationItem:
    text: str
    candidates: tuple[AdjudicationCandidate, ...]
    reason: str
    record_id: str = ""
    active_learning_priority: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_learning_priority": self.active_learning_priority,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": dict(self.metadata),
            "reason": self.reason,
            "record_id": self.record_id,
            "text": self.text,
        }

    def to_hard_negative_seed(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "source": "weak_labeling_adjudication",
            "spans": [candidate.to_dict() for candidate in self.candidates],
            "text": self.text,
        }


@dataclass
class AdjudicationQueue:
    items: list[AdjudicationItem] = field(default_factory=list)

    def add(self, item: AdjudicationItem) -> AdjudicationItem:
        self.items.append(item)
        return item

    def extend(self, items: Iterable[AdjudicationItem]) -> None:
        self.items.extend(items)

    def drain(self) -> tuple[AdjudicationItem, ...]:
        drained = tuple(self.items)
        self.items.clear()
        return drained


def make_adjudication_item(
    *,
    text: str,
    candidates: Iterable[AdjudicationCandidate],
    reason: str,
    record_id: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> AdjudicationItem:
    candidates_tuple = tuple(candidates)
    priority = 1.0 + max(
        (candidate.score for candidate in candidates_tuple), default=0.0
    )
    return AdjudicationItem(
        text=text,
        candidates=candidates_tuple,
        reason=reason,
        record_id=record_id,
        active_learning_priority=priority,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "AdjudicationCandidate",
    "AdjudicationItem",
    "AdjudicationQueue",
    "make_adjudication_item",
]
