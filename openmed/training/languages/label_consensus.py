"""Independent-source consensus and adjudication for local projected labels."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_LABEL = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
SOURCES = frozenset({"native_model", "translated_text", "rule", "terminology"})


@dataclass(frozen=True)
class LabelVote:
    """An opaque record span, controlled label and independent local producer."""

    record_digest: str
    start: int
    end: int
    label: str
    source: str
    producer_digest: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not _DIGEST.fullmatch(value)
            for value in (self.record_digest, self.producer_digest)
        ):
            raise ValueError("consensus references must be SHA-256 digests")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (self.start, self.end)
            )
            or not 0 <= self.start < self.end <= 10_000_000
        ):
            raise ValueError("invalid consensus span coordinates")
        if not isinstance(self.label, str) or not _LABEL.fullmatch(self.label):
            raise ValueError("invalid controlled label code")
        if not isinstance(self.source, str) or self.source not in SOURCES:
            raise ValueError("unknown consensus source category")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be finite within [0, 1]")


@dataclass(frozen=True)
class ConsensusRule:
    """Minimum independent source evidence; disagreements always require review."""

    minimum_sources: int = 2
    minimum_confidence: float = 0.75

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_sources, bool)
            or not isinstance(self.minimum_sources, int)
            or not 2 <= self.minimum_sources <= len(SOURCES)
        ):
            raise ValueError("consensus requires 2 to 4 independent source categories")
        if (
            isinstance(self.minimum_confidence, bool)
            or not isinstance(self.minimum_confidence, (int, float))
            or not math.isfinite(self.minimum_confidence)
            or not 0 <= self.minimum_confidence <= 1
        ):
            raise ValueError("confidence threshold must be finite within [0, 1]")


@dataclass(frozen=True)
class ConsensusDecision:
    """A source-aligned accepted label or explicit metadata-only review record."""

    record_digest: str
    start: int
    end: int
    label: str | None
    decision: str
    reason: str
    independent_source_count: int
    candidate_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.record_digest, str) or not _DIGEST.fullmatch(
            self.record_digest
        ):
            raise ValueError("decision record reference must be a SHA-256 digest")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (self.start, self.end)
            )
            or not 0 <= self.start < self.end <= 10_000_000
        ):
            raise ValueError("invalid decision span coordinates")
        if (
            not isinstance(self.candidate_labels, Sequence)
            or isinstance(self.candidate_labels, (str, bytes))
            or not 1 <= len(self.candidate_labels) <= 64
            or any(not isinstance(label, str) for label in self.candidate_labels)
        ):
            raise ValueError("candidate labels must be a sequence")
        labels = tuple(sorted(set(self.candidate_labels)))
        if not labels or any(
            not isinstance(label, str) or not _LABEL.fullmatch(label)
            for label in labels
        ):
            raise ValueError("invalid decision label category")
        object.__setattr__(self, "candidate_labels", labels)
        if (
            isinstance(self.independent_source_count, bool)
            or not isinstance(self.independent_source_count, int)
            or not 0 <= self.independent_source_count <= 4
        ):
            raise ValueError("invalid independent source count")
        reasons = {
            "overlapping_projection",
            "label_disagreement",
            "low_confidence",
            "insufficient_independent_sources",
            "exact_source_consensus",
        }
        if (
            not isinstance(self.reason, str)
            or not isinstance(self.decision, str)
            or self.reason not in reasons
            or self.decision
            not in {
                "accepted",
                "needs_adjudication",
            }
        ):
            raise ValueError("invalid consensus disposition")
        if self.decision == "accepted":
            if (
                self.reason != "exact_source_consensus"
                or self.independent_source_count < 2
                or len(labels) != 1
                or self.label != labels[0]
            ):
                raise ValueError("inconsistent accepted consensus metadata")
        elif self.label is not None or self.reason == "exact_source_consensus":
            raise ValueError("inconsistent adjudication metadata")

    def to_dict(self) -> dict[str, object]:
        """Serialize only controlled categories, offsets, digests and counts."""
        return {
            "record_digest": self.record_digest,
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "decision": self.decision,
            "reason": self.reason,
            "independent_source_count": self.independent_source_count,
            "candidate_labels": list(self.candidate_labels),
        }


def build_label_consensus(
    votes: Sequence[LabelVote], *, rule: ConsensusRule = ConsensusRule()
) -> tuple[ConsensusDecision, ...]:
    """Accept exact agreement from independent categories and producer identities.

    Duplicate votes never increase evidence. Conflicting labels or overlapping
    non-identical spans require adjudication, even when a majority agrees. All
    votes must clear confidence; missing or weak evidence cannot be overwritten
    by a strong teacher. No source text, model execution, or label training occurs.
    """
    if not isinstance(rule, ConsensusRule):
        raise ValueError("expected validated consensus rule")
    if len(votes) > 10_000 or not all(isinstance(vote, LabelVote) for vote in votes):
        raise ValueError("provide at most 10000 validated label votes")
    grouped = defaultdict(list)
    for vote in set(votes):
        grouped[(vote.record_digest, vote.start, vote.end)].append(vote)
    overlap = set()
    by_record = defaultdict(list)
    for key in sorted(grouped):
        by_record[key[0]].append(key)
    for spans in by_record.values():
        active = []
        for key in spans:
            active = [other for other in active if other[2] > key[1]]
            for other in active:
                overlap.add(other)
                overlap.add(key)
            active.append(key)
    decisions = []
    for key, rows in sorted(grouped.items()):
        labels = tuple(sorted({row.label for row in rows}))
        eligible = [row for row in rows if row.confidence >= rule.minimum_confidence]
        # Reusing one producer in multiple categories is not independent evidence.
        producers_by_source = {
            source: {row.producer_digest for row in eligible if row.source == source}
            for source in SOURCES
        }
        sources = sorted(
            source for source, producers in producers_by_source.items() if producers
        )
        maximum = 0

        def matching(index: int, used: set[str], count: int) -> None:
            nonlocal maximum
            if index == len(sources):
                maximum = max(maximum, count)
                return
            matching(index + 1, used, count)
            for producer in producers_by_source[sources[index]] - used:
                matching(index + 1, used | {producer}, count + 1)

        # Four source categories bound matching depth; retain only four producer
        # candidates per category, enough for a maximum four-source matching.
        producers_by_source = {
            source: set(sorted(producers)[:4])
            for source, producers in producers_by_source.items()
        }
        matching(0, set(), 0)
        reason = (
            "overlapping_projection"
            if key in overlap
            else "label_disagreement"
            if len(labels) != 1
            else "low_confidence"
            if len(eligible) != len(rows)
            else "insufficient_independent_sources"
            if maximum < rule.minimum_sources
            else "exact_source_consensus"
        )
        accepted = reason == "exact_source_consensus"
        decisions.append(
            ConsensusDecision(
                *key,
                labels[0] if accepted else None,
                "accepted" if accepted else "needs_adjudication",
                reason,
                maximum,
                labels,
            )
        )
    return tuple(decisions)
