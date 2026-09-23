"""Tests for value-free guarded relation review bands."""

import pytest

from openmed.clinical.relations.review_priority import (
    DEFAULT_REVIEW_PRIORITY_POLICY,
    ReviewPriorityPolicy,
    assign_review_priority,
)


def test_default_policy_combines_relation_conflict_and_completeness() -> None:
    priority = assign_review_priority(
        "diagnosis_to_treatment",
        "unresolved",
        "partial",
    )

    assert priority.band == "band_a"
    assert priority.policy_score == 4
    assert priority.clinical_urgency_inferred is False
    assert priority.reason_codes == (
        "relation:diagnosis_to_treatment",
        "conflict:unresolved",
        "evidence:partial",
    )


def test_policy_is_deterministic_value_free_and_configurable() -> None:
    first = assign_review_priority("unknown_relation", "none", "complete")
    second = assign_review_priority("unknown_relation", "none", "complete")
    assert first == second
    assert first.band == "band_c"
    payload = first.to_dict()
    assert not any(key in payload for key in ("diagnosis", "treatment", "urgency"))

    policy = ReviewPriorityPolicy(
        relation_weights={"unknown_relation": 5},
        conflict_weights=DEFAULT_REVIEW_PRIORITY_POLICY.conflict_weights,
        completeness_weights=DEFAULT_REVIEW_PRIORITY_POLICY.completeness_weights,
        band_thresholds=((5, "band_a"), (1, "band_b"), (0, "band_c")),
        policy_id="local-review-v1",
    )
    assert (
        assign_review_priority(
            "unknown_relation",
            "none",
            "complete",
            policy=policy,
        ).band
        == "band_a"
    )


def test_rejects_states_not_defined_by_policy() -> None:
    with pytest.raises(ValueError, match="conflict_state"):
        assign_review_priority("laboratory_result", "urgent", "complete")  # type: ignore[arg-type]
