from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from openmed.clinical.nli_hypothesis_guard import (
    HypothesisComplexityReason,
    HypothesisGuardError,
    HypothesisGuardStatus,
    guard_hypothesis,
    guard_nli_hypothesis,
    measure_hypothesis_complexity,
)


def test_single_bounded_assertion_is_ready_for_inference() -> None:
    hypothesis = "The finding is absent."

    result = guard_hypothesis(hypothesis)

    assert result.status is HypothesisGuardStatus.READY
    assert result.inference_allowed is True
    assert result.hypothesis == hypothesis
    assert result.reasons == ()
    assert result.complexity.character_count == len(hypothesis)
    assert result.complexity.clause_count == 1
    assert result.complexity.assertion_count == 1


def test_coordinated_findings_under_one_predicate_remain_one_assertion() -> None:
    result = guard_hypothesis("The record shows fever and cough.")

    assert result.inference_allowed is True
    assert result.complexity.assertion_count == 1


@pytest.mark.parametrize(
    "hypothesis",
    [
        "The finding is present and the treatment was started.",
        "The finding is present but was later denied.",
        "The finding is present. The treatment was started.",
    ],
)
def test_multi_assertion_hypotheses_require_segmentation(hypothesis: str) -> None:
    result = guard_hypothesis(hypothesis)

    assert result.status is HypothesisGuardStatus.SEGMENTATION_REQUIRED
    assert result.inference_allowed is False
    assert result.hypothesis is None
    assert HypothesisComplexityReason.ASSERTION_LIMIT_EXCEEDED in result.reasons


def test_subordinate_clause_count_is_measured_deterministically() -> None:
    hypothesis = "The finding improved because the intervention was completed."

    first = measure_hypothesis_complexity(hypothesis)
    second = measure_hypothesis_complexity(hypothesis)

    assert first == second
    assert first.clause_count == 2
    assert first.assertion_count == 1


def test_clause_ceiling_can_require_segmentation_independently() -> None:
    result = guard_hypothesis(
        "The finding improved because the intervention was completed.",
        max_clauses=1,
    )

    assert result.reasons == (HypothesisComplexityReason.CLAUSE_LIMIT_EXCEEDED,)


def test_long_hypothesis_requires_segmentation_without_retaining_text() -> None:
    hypothesis = "synthetic " * 30

    result = guard_hypothesis(hypothesis, max_characters=40)

    assert result.hypothesis is None
    assert result.reasons == (HypothesisComplexityReason.CHARACTER_LIMIT_EXCEEDED,)


def test_all_exceeded_limits_have_stable_reason_order() -> None:
    result = guard_hypothesis(
        "The finding is present. The treatment was started.",
        max_characters=10,
        max_clauses=1,
        max_assertions=1,
    )

    assert result.reasons == (
        HypothesisComplexityReason.CHARACTER_LIMIT_EXCEEDED,
        HypothesisComplexityReason.CLAUSE_LIMIT_EXCEEDED,
        HypothesisComplexityReason.ASSERTION_LIMIT_EXCEEDED,
    )


def test_decimal_point_does_not_create_an_extra_assertion() -> None:
    complexity = measure_hypothesis_complexity("The synthetic value is 5.2 units.")

    assert complexity.clause_count == 1
    assert complexity.assertion_count == 1


def test_period_after_number_and_newline_are_assertion_boundaries() -> None:
    numbered = measure_hypothesis_complexity(
        "The synthetic count is 5. The finding is absent."
    )
    multiline = measure_hypothesis_complexity(
        "The finding is present\nThe treatment was started"
    )

    assert numbered.assertion_count == 2
    assert multiline.assertion_count == 2


def test_alias_and_configured_multi_assertion_contract() -> None:
    hypothesis = "The finding is present. The treatment was started."

    assert guard_nli_hypothesis(
        hypothesis,
        max_assertions=2,
        max_clauses=2,
    ) == guard_hypothesis(
        hypothesis,
        max_assertions=2,
        max_clauses=2,
    )
    assert guard_nli_hypothesis(
        hypothesis,
        max_assertions=2,
        max_clauses=2,
    ).inference_allowed


@pytest.mark.parametrize("hypothesis", ["", "   ", None, 123])
def test_invalid_hypothesis_fails_without_echoing_values(hypothesis: object) -> None:
    with pytest.raises(HypothesisGuardError, match="invalid hypothesis text"):
        guard_hypothesis(hypothesis)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_characters": 0},
        {"max_clauses": -1},
        {"max_assertions": True},
    ],
)
def test_invalid_limits_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(
        HypothesisGuardError,
        match="invalid hypothesis complexity limit",
    ):
        guard_hypothesis("synthetic finding", **kwargs)  # type: ignore[arg-type]


def test_reports_and_repr_never_include_hypothesis_text() -> None:
    sentinel = "SENSITIVE_SENTINEL"
    result = guard_hypothesis(sentinel)

    assert result.hypothesis == sentinel
    assert sentinel not in repr(result)
    assert sentinel not in str(result.to_dict())
    assert "hypothesis" not in result.to_dict()


def test_result_is_immutable() -> None:
    result = guard_hypothesis("The finding is absent.")

    with pytest.raises(FrozenInstanceError):
        result.status = HypothesisGuardStatus.SEGMENTATION_REQUIRED  # type: ignore[misc]
