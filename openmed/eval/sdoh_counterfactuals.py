"""Synthetic counterfactual invariance checks for local SDOH extraction.

Only aggregate category counts leave this module's evaluation report. Source
text and finding values stay in memory and are never written to an artifact.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from openmed.clinical.sdoh import SDOHFinding, extract_sdoh
from openmed.clinical.sections import detect_sections
from openmed.training.synthetic.social_history import (
    SOCIAL_HISTORY_CATEGORIES,
    generate_social_history_examples,
)

_BASELINE_CONTEXT = "Synthetic patient context: adult age 34; pronouns she/her.\n"
_COUNTERFACTUAL_CONTEXT = "Synthetic patient context: adult age 76; pronouns he/him.\n"
_SAFE_CATEGORIES = frozenset((*SOCIAL_HISTORY_CATEGORIES, "food_insecurity"))

FindingExtractor = Callable[[str], Iterable[SDOHFinding]]
FindingLabel = tuple[str, str, str | None, str | None, str | None]


@dataclass(frozen=True)
class SDOHCounterfactualPair:
    """A synthetic pair with identical social history and altered context."""

    baseline_text: str = field(repr=False)
    counterfactual_text: str = field(repr=False)
    synthetic: bool = True


@dataclass(frozen=True)
class SDOHCounterfactualReport:
    """Aggregate invariance scores without source text or finding values."""

    pair_count: int
    label_mismatch_pairs: int
    confidence_mismatch_pairs: int
    mismatch_by_category: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for value in (
            self.pair_count,
            self.label_mismatch_pairs,
            self.confidence_mismatch_pairs,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("counterfactual counts must be non-negative integers")
        if self.label_mismatch_pairs + self.confidence_mismatch_pairs > self.pair_count:
            raise ValueError("counterfactual mismatch counts exceed pair count")
        for category, count in self.mismatch_by_category:
            if category not in _SAFE_CATEGORIES | {"other"}:
                raise ValueError("counterfactual category is not recognized")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("counterfactual counts must be non-negative integers")

    @property
    def invariant_pairs(self) -> int:
        """Count pairs with equal labels and confidences."""

        return (
            self.pair_count - self.label_mismatch_pairs - self.confidence_mismatch_pairs
        )

    @property
    def invariance_rate(self) -> float:
        """Return the fraction of pairs with invariant labels and scores."""

        return self.invariant_pairs / self.pair_count if self.pair_count else 1.0

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic aggregate-only report."""

        return {
            "pair_count": self.pair_count,
            "label_mismatch_pairs": self.label_mismatch_pairs,
            "confidence_mismatch_pairs": self.confidence_mismatch_pairs,
            "invariant_pairs": self.invariant_pairs,
            "invariance_rate": self.invariance_rate,
            "mismatch_by_category": dict(self.mismatch_by_category),
            "synthetic": True,
        }


def generate_sdoh_counterfactual_pairs(
    count: int, *, seed: int = 0
) -> tuple[SDOHCounterfactualPair, ...]:
    """Build reproducible pairs from the synthetic social-history generator.

    The only intervention is an age and pronoun change outside the Social
    History section. Both inputs contain identical determinant evidence.

    Args:
        count: Number of complete synthetic social histories.
        seed: Deterministic seed for the shared social-history generator.

    Returns:
        Pairs whose baseline and counterfactual share every SDOH clause.
    """

    return tuple(
        SDOHCounterfactualPair(
            baseline_text=_BASELINE_CONTEXT + example.text,
            counterfactual_text=_COUNTERFACTUAL_CONTEXT + example.text,
        )
        for example in generate_social_history_examples(count, seed=seed)
    )


def _local_findings(text: str) -> Iterable[SDOHFinding]:
    return extract_sdoh(text, (), sections=detect_sections(text))


def _label(finding: SDOHFinding) -> FindingLabel:
    return (
        finding.category,
        finding.value,
        finding.status,
        finding.extent,
        finding.temporality,
    )


def _safe_category(category: str) -> str:
    return category if category in _SAFE_CATEGORIES else "other"


def evaluate_sdoh_counterfactuals(
    pairs: Sequence[SDOHCounterfactualPair],
    *,
    extractor: FindingExtractor = _local_findings,
) -> SDOHCounterfactualReport:
    """Score label and confidence invariance for synthetic SDOH pairs.

    An extractor failure is sanitized so its exception cannot expose source
    text. Confidence is compared only when both sides have the same labels.

    Args:
        pairs: Synthetic counterfactual pairs to evaluate.
        extractor: Local SDOH extractor receiving a text string.

    Returns:
        Pair-level mismatch counts and safe category-level mismatch counts.

    Raises:
        ValueError: If a pair is not marked synthetic or extraction fails.
    """

    label_mismatch_pairs = 0
    confidence_mismatch_pairs = 0
    mismatch_by_category: Counter[str] = Counter()
    for pair in pairs:
        if not isinstance(pair, SDOHCounterfactualPair) or pair.synthetic is not True:
            raise ValueError("counterfactual input must be synthetic")
        try:
            baseline = tuple(extractor(pair.baseline_text))
            counterfactual = tuple(extractor(pair.counterfactual_text))
            if any(
                not isinstance(item, SDOHFinding)
                for item in (*baseline, *counterfactual)
            ):
                raise TypeError("invalid finding")
        except Exception:
            raise ValueError("SDOH counterfactual extraction failed") from None

        baseline_labels = Counter(_label(item) for item in baseline)
        counterfactual_labels = Counter(_label(item) for item in counterfactual)
        if baseline_labels != counterfactual_labels:
            label_mismatch_pairs += 1
            for label in baseline_labels.keys() | counterfactual_labels.keys():
                if baseline_labels[label] != counterfactual_labels[label]:
                    mismatch_by_category[_safe_category(label[0])] += 1
            continue

        baseline_scores: dict[FindingLabel, list[float]] = defaultdict(list)
        counterfactual_scores: dict[FindingLabel, list[float]] = defaultdict(list)
        for finding in baseline:
            baseline_scores[_label(finding)].append(finding.score)
        for finding in counterfactual:
            counterfactual_scores[_label(finding)].append(finding.score)
        if any(
            sorted(baseline_scores[label]) != sorted(counterfactual_scores[label])
            for label in baseline_labels
        ):
            confidence_mismatch_pairs += 1
            for label in baseline_labels:
                if sorted(baseline_scores[label]) != sorted(
                    counterfactual_scores[label]
                ):
                    mismatch_by_category[_safe_category(label[0])] += 1

    return SDOHCounterfactualReport(
        pair_count=len(pairs),
        label_mismatch_pairs=label_mismatch_pairs,
        confidence_mismatch_pairs=confidence_mismatch_pairs,
        mismatch_by_category=tuple(sorted(mismatch_by_category.items())),
    )


def require_sdoh_counterfactual_invariance(
    pairs: Sequence[SDOHCounterfactualPair],
    *,
    extractor: FindingExtractor = _local_findings,
) -> SDOHCounterfactualReport:
    """Return aggregate results, failing closed on any non-invariant pair."""

    report = evaluate_sdoh_counterfactuals(pairs, extractor=extractor)
    if report.invariant_pairs != report.pair_count:
        raise ValueError("SDOH counterfactual invariance gate failed")
    return report


__all__ = [
    "SDOHCounterfactualPair",
    "SDOHCounterfactualReport",
    "evaluate_sdoh_counterfactuals",
    "generate_sdoh_counterfactual_pairs",
    "require_sdoh_counterfactual_invariance",
]
