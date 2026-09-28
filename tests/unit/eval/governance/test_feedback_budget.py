from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from openmed.eval.governance.feedback_budget import (
    FEEDBACK_COARSE,
    FEEDBACK_DETAILED,
    FEEDBACK_NONE,
    REASON_ATTEMPT_RECORDED,
    REASON_FAILURE_APPROVED,
    REASON_RERUN_BUDGET_EXHAUSTED,
    FeedbackBudgetError,
    FeedbackBudgetLedger,
    FeedbackBudgetPolicy,
)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('ascii')).hexdigest()}"


def _policy(
    epoch: str = "epoch-a", *, detail_limit: int = 1, rerun_limit: int = 1
) -> FeedbackBudgetPolicy:
    return FeedbackBudgetPolicy(
        epoch_digest=_digest(epoch),
        detailed_feedback_limit=detail_limit,
        rerun_limit=rerun_limit,
        coarse_score_boundaries=(0.5, 0.8),
    )


def test_epoch_policy_limits_score_detail_and_reruns_per_submission() -> None:
    policy = _policy()
    ledger = FeedbackBudgetLedger((policy,))
    submission = _digest("sealed-submission")

    first = ledger.record_official_attempt(policy.epoch_digest, submission, score=0.83)
    second = ledger.record_official_attempt(policy.epoch_digest, submission, score=0.75)
    denied = ledger.record_official_attempt(policy.epoch_digest, submission, score=0.91)

    assert first.accepted is True
    assert first.reason_code == REASON_ATTEMPT_RECORDED
    assert first.official_attempt_number == 1
    assert first.feedback_level == FEEDBACK_DETAILED
    assert first.detailed_score == 0.83
    assert first.coarse_band is None
    assert first.detailed_feedback_remaining == 0
    assert first.attempts_remaining == 1

    assert second.accepted is True
    assert second.official_attempt_number == 2
    assert second.feedback_level == FEEDBACK_COARSE
    assert second.detailed_score is None
    assert second.coarse_band == 1
    assert second.attempts_remaining == 0

    assert denied.accepted is False
    assert denied.reason_code == REASON_RERUN_BUDGET_EXHAUSTED
    assert denied.official_attempt_number is None
    assert denied.feedback_level == FEEDBACK_NONE
    assert denied.detailed_score is None
    assert denied.coarse_band is None
    assert denied.official_attempts == 2


def test_epoch_and_submission_digests_have_independent_budgets() -> None:
    first_epoch = _policy("epoch-a", detail_limit=0, rerun_limit=0)
    second_epoch = _policy("epoch-b", detail_limit=1, rerun_limit=1)
    ledger = FeedbackBudgetLedger((second_epoch, first_epoch))
    first_submission = _digest("submission-a")
    second_submission = _digest("submission-b")

    first_result = ledger.record_official_attempt(
        first_epoch.epoch_digest, first_submission, score=0.5
    )
    other_submission = ledger.record_official_attempt(
        first_epoch.epoch_digest, second_submission, score=0.5
    )
    other_epoch = ledger.record_official_attempt(
        second_epoch.epoch_digest, first_submission, score=0.5
    )

    assert first_result.feedback_level == FEEDBACK_COARSE
    assert first_result.coarse_band == 1
    assert other_submission.accepted is True
    assert other_submission.official_attempt_number == 1
    assert other_epoch.feedback_level == FEEDBACK_DETAILED
    assert other_epoch.detailed_score == 0.5


def test_coarse_feedback_is_deterministic_and_hides_exact_score() -> None:
    policy = _policy(detail_limit=0, rerun_limit=3)
    ledger = FeedbackBudgetLedger((policy,))
    submission = _digest("submission")

    bands = [
        ledger.record_official_attempt(
            policy.epoch_digest, submission, score=score
        ).coarse_band
        for score in (0.0, 0.5, 0.799, 0.8)
    ]
    rendered = json.dumps(
        ledger.snapshot(policy.epoch_digest, submission).to_dict(), sort_keys=True
    )

    assert bands == [0, 1, 1, 2]
    assert "0.799" not in rendered
    assert ledger.snapshot(policy.epoch_digest, submission).scored_attempts == 4


def test_failure_requires_separate_approval_for_replacement_attempt() -> None:
    policy = _policy(detail_limit=1, rerun_limit=0)
    ledger = FeedbackBudgetLedger((policy,))
    submission = _digest("submission")
    failure = _digest("runner-crash-evidence")

    failed_attempt = ledger.record_official_attempt(
        policy.epoch_digest,
        submission,
        infrastructure_failure_digest=failure,
    )
    denied = ledger.record_official_attempt(policy.epoch_digest, submission, score=0.7)

    assert failed_attempt.accepted is True
    assert failed_attempt.feedback_level == FEEDBACK_NONE
    assert failed_attempt.scored_attempts == 0
    assert failed_attempt.attempts_remaining == 0
    assert denied.reason_code == REASON_RERUN_BUDGET_EXHAUSTED

    approval = ledger.approve_infrastructure_failure(
        policy.epoch_digest,
        submission,
        failure,
        _digest("operator-approval-record"),
    )
    replacement = ledger.record_official_attempt(
        policy.epoch_digest, submission, score=0.7
    )

    assert approval.approved is True
    assert approval.reason_code == REASON_FAILURE_APPROVED
    assert approval.approved_failure_replacements == 1
    assert approval.attempts_remaining == 1
    assert replacement.accepted is True
    assert replacement.official_attempt_number == 2
    assert replacement.feedback_level == FEEDBACK_DETAILED
    assert replacement.attempts_remaining == 0


def test_failure_and_approval_records_are_one_time_only() -> None:
    policy = _policy(rerun_limit=1)
    ledger = FeedbackBudgetLedger((policy,))
    submission = _digest("submission")
    failure = _digest("failure")
    approval = _digest("approval")

    ledger.record_official_attempt(
        policy.epoch_digest,
        submission,
        infrastructure_failure_digest=failure,
    )
    with pytest.raises(FeedbackBudgetError, match="already_recorded"):
        ledger.record_official_attempt(
            policy.epoch_digest,
            submission,
            infrastructure_failure_digest=failure,
        )

    ledger.approve_infrastructure_failure(
        policy.epoch_digest, submission, failure, approval
    )
    with pytest.raises(FeedbackBudgetError, match="already_approved"):
        ledger.approve_infrastructure_failure(
            policy.epoch_digest, submission, failure, _digest("other-approval")
        )

    other_submission = _digest("other-submission")
    other_failure = _digest("other-failure")
    ledger.record_official_attempt(
        policy.epoch_digest,
        other_submission,
        infrastructure_failure_digest=other_failure,
    )
    with pytest.raises(FeedbackBudgetError, match="already_used"):
        ledger.approve_infrastructure_failure(
            policy.epoch_digest, other_submission, other_failure, approval
        )


def test_approved_replacement_does_not_replenish_score_detail() -> None:
    policy = _policy(detail_limit=1, rerun_limit=1)
    ledger = FeedbackBudgetLedger((policy,))
    submission = _digest("submission")
    failure = _digest("failure")

    detailed = ledger.record_official_attempt(
        policy.epoch_digest, submission, score=0.9
    )
    ledger.record_official_attempt(
        policy.epoch_digest,
        submission,
        infrastructure_failure_digest=failure,
    )
    ledger.approve_infrastructure_failure(
        policy.epoch_digest, submission, failure, _digest("approval")
    )
    replacement = ledger.record_official_attempt(
        policy.epoch_digest, submission, score=0.9
    )

    assert detailed.feedback_level == FEEDBACK_DETAILED
    assert replacement.feedback_level == FEEDBACK_COARSE
    assert replacement.detailed_score is None
    assert replacement.coarse_band == 2


def test_unrecorded_failure_cannot_receive_operator_approval() -> None:
    policy = _policy()
    ledger = FeedbackBudgetLedger((policy,))

    with pytest.raises(FeedbackBudgetError, match="failure_digest: not_recorded"):
        ledger.approve_infrastructure_failure(
            policy.epoch_digest,
            _digest("submission"),
            _digest("failure"),
            _digest("approval"),
        )


def test_invalid_input_and_reports_do_not_echo_sensitive_values() -> None:
    policy = _policy(detail_limit=0, rerun_limit=0)
    ledger = FeedbackBudgetLedger((policy,))
    sensitive_value = "synthetic-private-clinical-identifier"

    with pytest.raises(FeedbackBudgetError) as raised:
        ledger.record_official_attempt(policy.epoch_digest, sensitive_value, score=0.5)
    assert sensitive_value not in str(raised.value)

    with pytest.raises(FeedbackBudgetError) as raised:
        ledger.record_official_attempt(
            policy.epoch_digest,
            _digest("submission"),
            infrastructure_failure_digest=sensitive_value,
        )
    assert sensitive_value not in str(raised.value)

    result = ledger.record_official_attempt(
        policy.epoch_digest, _digest("submission"), score=0.712345
    )
    rendered = json.dumps(result.to_dict(), sort_keys=True)
    assert "0.712345" not in rendered
    assert sensitive_value not in rendered


@pytest.mark.parametrize(
    ("detail_limit", "rerun_limit", "boundaries"),
    (
        (-1, 0, (0.5,)),
        (0, -1, (0.5,)),
        (0, 0, ()),
        (0, 0, (0.8, 0.5)),
        (0, 0, (0.0, 0.5)),
        (0, 0, (0.5, float("nan"))),
    ),
)
def test_policy_rejects_invalid_limits_and_boundaries(
    detail_limit: int,
    rerun_limit: int,
    boundaries: tuple[float, ...],
) -> None:
    with pytest.raises(FeedbackBudgetError):
        FeedbackBudgetPolicy(
            epoch_digest=_digest("epoch"),
            detailed_feedback_limit=detail_limit,
            rerun_limit=rerun_limit,
            coarse_score_boundaries=boundaries,
        )


def test_policy_and_report_objects_are_immutable() -> None:
    policy = _policy()
    decision = FeedbackBudgetLedger((policy,)).record_official_attempt(
        policy.epoch_digest, _digest("submission"), score=0.9
    )

    with pytest.raises(FrozenInstanceError):
        policy.rerun_limit = 100  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        decision.accepted = False  # type: ignore[misc]
