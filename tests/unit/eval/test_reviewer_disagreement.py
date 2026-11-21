"""Tests for privacy-safe reviewer disagreement metrics."""

from __future__ import annotations

import json
import socket

import pytest

from openmed.eval.reviewer_disagreement import (
    DisagreementReason,
    ReviewerDecision,
    reviewer_disagreement_report,
)


def _case(
    case_number: int,
    left: str,
    right: str,
    *,
    reason: DisagreementReason | None = None,
    adjudicated: bool = False,
) -> list[ReviewerDecision]:
    case_id = f"case-token-{case_number}"
    return [
        ReviewerDecision(
            case_id=case_id,
            reviewer_id="reviewer-token-a",
            decision=left,
            reason=reason,
            adjudicated=adjudicated,
        ),
        ReviewerDecision(
            case_id=case_id,
            reviewer_id="reviewer-token-b",
            decision=right,
            reason=reason,
            adjudicated=adjudicated,
        ),
    ]


def test_computes_agreement_and_adjudication_rates() -> None:
    decisions = [
        *_case(1, "accept", "accept"),
        *_case(2, "accept", "accept"),
        *_case(
            3,
            "accept",
            "revise",
            reason=DisagreementReason.EVIDENCE_QUALITY,
            adjudicated=True,
        ),
        *_case(
            4,
            "revise",
            "accept",
            reason=DisagreementReason.EVIDENCE_QUALITY,
            adjudicated=True,
        ),
        *_case(
            5,
            "accept",
            "revise",
            reason=DisagreementReason.EVIDENCE_QUALITY,
        ),
        *_case(
            6,
            "revise",
            "accept",
            reason=DisagreementReason.EVIDENCE_QUALITY,
        ),
    ]

    report = reviewer_disagreement_report(decisions, minimum_cell_size=2)

    assert report.agreement.to_dict() == {
        "denominator": 6,
        "numerator": 2,
        "rate": pytest.approx(1 / 3),
        "suppressed": False,
    }
    assert report.adjudication.to_dict() == {
        "denominator": 4,
        "numerator": 2,
        "rate": 0.5,
        "suppressed": False,
    }
    assert [cell.to_dict() for cell in report.reason_adjudication] == [
        {
            "denominator": 4,
            "numerator": 2,
            "rate": 0.5,
            "reason": "evidence_quality",
            "suppressed": False,
        }
    ]
    assert report.has_suppressed_cells is False


def test_suppresses_small_and_complementary_cells() -> None:
    decisions = [
        *_case(1, "accept", "accept"),
        *_case(2, "accept", "accept"),
        *_case(3, "accept", "accept"),
        *_case(4, "accept", "accept"),
        *_case(
            5,
            "accept",
            "revise",
            reason=DisagreementReason.GUIDELINE_AMBIGUITY,
            adjudicated=True,
        ),
    ]

    report = reviewer_disagreement_report(decisions, minimum_cell_size=2)

    assert report.agreement.to_dict() == {
        "denominator": None,
        "numerator": None,
        "rate": None,
        "suppressed": True,
    }
    assert report.adjudication.suppressed is True
    assert report.reason_adjudication == ()
    assert report.suppressed_reason_cells == 1
    assert report.has_suppressed_cells is True


def test_report_and_repr_never_expose_input_values() -> None:
    sensitive_case = "private-case-token"
    sensitive_reviewer = "private-reviewer-token"
    sensitive_decision = "private-decision-token"
    decision = ReviewerDecision(
        case_id=sensitive_case,
        reviewer_id=sensitive_reviewer,
        decision=sensitive_decision,
    )
    decisions = [
        decision,
        ReviewerDecision(
            case_id=sensitive_case,
            reviewer_id="second-private-reviewer-token",
            decision=sensitive_decision,
        ),
    ]

    report = reviewer_disagreement_report(decisions, minimum_cell_size=2)
    serialized = json.dumps(report.to_dict(), sort_keys=True)

    for value in (sensitive_case, sensitive_reviewer, sensitive_decision):
        assert value not in serialized
        assert value not in repr(decision)


def test_result_is_deterministic_and_requires_no_network(monkeypatch) -> None:
    decisions = [
        *_case(1, "accept", "accept"),
        *_case(2, "accept", "accept"),
        *_case(
            3,
            "accept",
            "revise",
            reason=DisagreementReason.REVIEWER_ERROR,
        ),
        *_case(
            4,
            "revise",
            "accept",
            reason=DisagreementReason.REVIEWER_ERROR,
        ),
    ]

    def reject_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "socket", reject_network)
    forward = reviewer_disagreement_report(decisions, minimum_cell_size=2)
    reverse = reviewer_disagreement_report(reversed(decisions), minimum_cell_size=2)

    assert json.dumps(forward.to_dict(), sort_keys=True) == json.dumps(
        reverse.to_dict(), sort_keys=True
    )


def test_validation_errors_are_value_free() -> None:
    sensitive_case = "private-case-token"
    duplicate = ReviewerDecision(
        case_id=sensitive_case,
        reviewer_id="private-reviewer-token",
        decision="accept",
    )

    with pytest.raises(ValueError) as caught:
        reviewer_disagreement_report([duplicate, duplicate], minimum_cell_size=2)

    assert sensitive_case not in str(caught.value)
    assert "private-reviewer-token" not in str(caught.value)


def test_rejects_untyped_or_inconsistent_disagreement_metadata() -> None:
    with pytest.raises(TypeError, match="DisagreementReason"):
        ReviewerDecision(
            case_id="case-token",
            reviewer_id="reviewer-token",
            decision="accept",
            reason="free text",  # type: ignore[arg-type]
        )

    inconsistent = [
        ReviewerDecision(
            case_id="case-token",
            reviewer_id="reviewer-a",
            decision="accept",
            reason=DisagreementReason.OTHER,
            adjudicated=True,
        ),
        ReviewerDecision(
            case_id="case-token",
            reviewer_id="reviewer-b",
            decision="revise",
            reason=DisagreementReason.LABEL_DEFINITION,
            adjudicated=True,
        ),
    ]
    with pytest.raises(ValueError, match="one consistent typed reason"):
        reviewer_disagreement_report(inconsistent, minimum_cell_size=2)


def test_minimum_cell_size_must_protect_more_than_one_case(subtests) -> None:
    for minimum_cell_size in [True, 1, 2.5]:
        with subtests.test(minimum_cell_size=minimum_cell_size):
            with pytest.raises(ValueError, match="integer >= 2"):
                reviewer_disagreement_report(
                    _case(1, "accept", "accept"),
                    minimum_cell_size=minimum_cell_size,  # type: ignore[arg-type]
                )


def test_model_review_rates_and_value_free_grouping():
    from openmed.eval.reviewer_disagreement import (
        ModelReviewDecision,
        ModelReviewOutcome,
        ReviewedModel,
        model_review_disagreement_report,
    )

    rows = []
    for model, rejects in ((ReviewedModel.ACTIVE, 2), (ReviewedModel.CANDIDATE, 4)):
        outcomes = (
            [ModelReviewOutcome.REJECT] * rejects
            + [ModelReviewOutcome.CORRECT] * 2
            + [ModelReviewOutcome.ACCEPT] * (8 - rejects)
        )
        for index, outcome in enumerate(outcomes):
            rows.append(
                ModelReviewDecision(
                    f"private-case-{index}",
                    "private-reviewer",
                    model,
                    outcome,
                    DisagreementReason.EVIDENCE_QUALITY,
                )
            )
    report = model_review_disagreement_report(rows, minimum_cell_size=2)
    assert "private" not in json.dumps(report)
    assert report == model_review_disagreement_report(
        reversed(rows), minimum_cell_size=2
    )
    active, candidate = report["models"]["active"], report["models"]["candidate"]
    assert active["rejection"]["rate"] == 0.2 and candidate["rejection"]["rate"] == 0.4
    assert candidate["by_reason"]["evidence_quality"]["correction"]["rate"] == 0.2
    with pytest.raises(ValueError, match="^duplicate model review$"):
        model_review_disagreement_report([*rows, rows[0]])


def test_model_review_suppresses_reconstructible_small_cells():
    from openmed.eval.reviewer_disagreement import (
        ModelReviewDecision,
        ModelReviewOutcome,
        ReviewedModel,
        model_review_disagreement_report,
    )

    rows = []
    for index, outcome in enumerate(
        ["accept", "correct", "correct", "reject", "reject"]
    ):
        rows.append(
            ModelReviewDecision(
                str(index),
                "private",
                ReviewedModel.ACTIVE,
                ModelReviewOutcome(outcome),
                DisagreementReason.OTHER,
            )
        )
    hidden = model_review_disagreement_report(rows, minimum_cell_size=2)["models"][
        "active"
    ]
    assert hidden["rejection"]["denominator"] is None
    assert hidden["correction"]["numerator"] is None
    rows.extend(
        ModelReviewDecision(
            str(index),
            "second",
            ReviewedModel.ACTIVE,
            ModelReviewOutcome.ACCEPT,
            DisagreementReason.LABEL_DEFINITION,
        )
        for index in range(4)
    )
    mixed = model_review_disagreement_report(rows, minimum_cell_size=2)["models"][
        "active"
    ]
    assert mixed["rejection"]["numerator"] == 2
    assert mixed["reason_breakdown_suppressed"]
    assert mixed["by_reason"] == {}
