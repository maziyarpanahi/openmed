"""Value-free policy bands for guarded relation review queues."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

ReviewBand = Literal["band_a", "band_b", "band_c"]
EvidenceCompleteness = Literal["complete", "partial", "missing"]
RelationConflictState = Literal[
    "none",
    "competing",
    "unresolved",
    "contradictory",
]

REVIEW_PRIORITY_ADVISORY = (
    "Review bands only order a human evidence queue under the configured "
    "policy. They do not infer diagnosis, treatment, severity, or clinical urgency."
)


@dataclass(frozen=True)
class ReviewPriorityPolicy:
    """Configured non-clinical queue weights and band thresholds."""

    relation_weights: Mapping[str, int]
    conflict_weights: Mapping[RelationConflictState, int]
    completeness_weights: Mapping[EvidenceCompleteness, int]
    band_thresholds: tuple[tuple[int, ReviewBand], ...]
    policy_id: str = "guarded-relations-v1"

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("review-priority policy_id must be non-empty")
        if not self.band_thresholds:
            raise ValueError("review-priority policy requires band thresholds")
        thresholds = tuple(sorted(self.band_thresholds, reverse=True))
        if any(type(score) is not int or score < 0 for score, _ in thresholds):
            raise ValueError("review-priority thresholds must be non-negative integers")
        object.__setattr__(
            self, "relation_weights", MappingProxyType(dict(self.relation_weights))
        )
        object.__setattr__(
            self, "conflict_weights", MappingProxyType(dict(self.conflict_weights))
        )
        object.__setattr__(
            self,
            "completeness_weights",
            MappingProxyType(dict(self.completeness_weights)),
        )
        object.__setattr__(self, "band_thresholds", thresholds)


DEFAULT_REVIEW_PRIORITY_POLICY = ReviewPriorityPolicy(
    relation_weights={
        "diagnosis_to_treatment": 2,
        "medication_change": 2,
        "laboratory_result": 1,
        "procedure_to_indication": 1,
    },
    conflict_weights={
        "none": 0,
        "competing": 1,
        "unresolved": 1,
        "contradictory": 2,
    },
    completeness_weights={
        "complete": 0,
        "partial": 1,
        "missing": 2,
    },
    band_thresholds=((3, "band_a"), (1, "band_b"), (0, "band_c")),
)


@dataclass(frozen=True)
class RelationReviewPriority:
    """One deterministic, value-free queue-band decision."""

    relation_type: str
    conflict_state: RelationConflictState
    evidence_completeness: EvidenceCompleteness
    band: ReviewBand
    policy_score: int
    reason_codes: tuple[str, ...]
    policy_id: str
    clinical_urgency_inferred: bool = False
    advisory: str = REVIEW_PRIORITY_ADVISORY

    def __post_init__(self) -> None:
        if self.clinical_urgency_inferred:
            raise ValueError("review priority cannot infer clinical urgency")

    def to_dict(self) -> dict[str, Any]:
        """Return the controlled policy result without clinical content."""

        return {
            "relation_type": self.relation_type,
            "conflict_state": self.conflict_state,
            "evidence_completeness": self.evidence_completeness,
            "band": self.band,
            "policy_score": self.policy_score,
            "reason_codes": list(self.reason_codes),
            "policy_id": self.policy_id,
            "clinical_urgency_inferred": self.clinical_urgency_inferred,
            "advisory": self.advisory,
        }


def assign_review_priority(
    relation_type: str,
    conflict_state: RelationConflictState,
    evidence_completeness: EvidenceCompleteness,
    *,
    policy: ReviewPriorityPolicy = DEFAULT_REVIEW_PRIORITY_POLICY,
) -> RelationReviewPriority:
    """Map relation metadata to a configured human-review queue band.

    The inputs are controlled metadata only. No note text, measurement value,
    diagnosis, treatment choice, or clinical urgency is inspected or inferred.
    """

    if not isinstance(relation_type, str) or not relation_type:
        raise ValueError("relation_type must be non-empty")
    if conflict_state not in policy.conflict_weights:
        raise ValueError("conflict_state is not defined by the review policy")
    if evidence_completeness not in policy.completeness_weights:
        raise ValueError("evidence_completeness is not defined by the review policy")

    relation_weight = int(policy.relation_weights.get(relation_type, 0))
    conflict_weight = int(policy.conflict_weights[conflict_state])
    completeness_weight = int(policy.completeness_weights[evidence_completeness])
    policy_score = relation_weight + conflict_weight + completeness_weight
    band = _band_for_score(policy_score, policy)
    reasons = [f"relation:{relation_type}"]
    if conflict_state != "none":
        reasons.append(f"conflict:{conflict_state}")
    if evidence_completeness != "complete":
        reasons.append(f"evidence:{evidence_completeness}")
    return RelationReviewPriority(
        relation_type=relation_type,
        conflict_state=conflict_state,
        evidence_completeness=evidence_completeness,
        band=band,
        policy_score=policy_score,
        reason_codes=tuple(reasons),
        policy_id=policy.policy_id,
    )


def _band_for_score(score: int, policy: ReviewPriorityPolicy) -> ReviewBand:
    for threshold, band in policy.band_thresholds:
        if score >= threshold:
            return band
    raise ValueError("review-priority policy does not cover the computed score")


__all__ = [
    "DEFAULT_REVIEW_PRIORITY_POLICY",
    "REVIEW_PRIORITY_ADVISORY",
    "EvidenceCompleteness",
    "RelationConflictState",
    "RelationReviewPriority",
    "ReviewBand",
    "ReviewPriorityPolicy",
    "assign_review_priority",
]
