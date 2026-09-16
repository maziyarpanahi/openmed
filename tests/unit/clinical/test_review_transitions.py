"""Focused tests for the deterministic clinical review state machine."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.review_transitions import (
    ReviewPolicyRule,
    ReviewState,
    ReviewStateMachine,
    ReviewTransitionPolicy,
    ReviewTransitionValidationError,
    compute_provenance_fingerprint,
    make_opaque_event_id,
    validate_review_history,
    validate_transition,
)


def _event(seed: int | str) -> str:
    return make_opaque_event_id({"synthetic_sequence": seed})


def _provenance(seed: int | str = "synthetic-artifact") -> str:
    return compute_provenance_fingerprint(
        {"artifact": seed, "schema": "synthetic-review-v1"}
    )


def test_default_policy_validates_complete_review_lifecycle() -> None:
    machine = ReviewStateMachine()
    transitions = (
        (ReviewState.IN_REVIEW, 1),
        (ReviewState.APPROVED, 2),
        (ReviewState.REOPENED, 3),
        (ReviewState.IN_REVIEW, 4),
        (ReviewState.REJECTED, 5),
        (ReviewState.REOPENED, 6),
        (ReviewState.IN_REVIEW, 7),
        (ReviewState.EXPIRED, 8),
        (ReviewState.REOPENED, 9),
        (ReviewState.IN_REVIEW, 10),
    )

    for state, sequence in transitions:
        record = machine.transition(state, _event(sequence), _provenance())
        assert record.sequence == sequence
        assert record.from_state == (
            ReviewState.QUEUED if sequence == 1 else transitions[sequence - 2][0]
        )

    assert machine.current_state is ReviewState.IN_REVIEW
    assert [record.to_state for record in machine.history] == [
        state for state, _ in transitions
    ]


def test_reopening_cannot_skip_new_review() -> None:
    machine = ReviewStateMachine(initial_state=ReviewState.APPROVED)

    with pytest.raises(ReviewTransitionValidationError) as caught:
        machine.transition(
            ReviewState.IN_REVIEW,
            _event("skip"),
            _provenance(),
        )

    assert caught.value.code == "transition_not_allowed"
    assert machine.current_state is ReviewState.APPROVED


def test_policy_can_require_reopen_reason_and_inject_a_rule() -> None:
    policy = ReviewTransitionPolicy(
        policy_id="synthetic-supervised-review",
        required_reason_states=(ReviewState.REOPENED,),
        rules=(
            ReviewPolicyRule(
                "reopen_reason",
                lambda request: request.reason_code == "corrected_artifact",
            ),
        ),
    )
    machine = ReviewStateMachine(initial_state=ReviewState.APPROVED, policy=policy)

    with pytest.raises(ReviewTransitionValidationError) as missing:
        machine.transition(ReviewState.REOPENED, _event(1), _provenance())
    assert missing.value.code == "reason_code_required"

    with pytest.raises(ReviewTransitionValidationError) as wrong:
        machine.transition(
            ReviewState.REOPENED,
            _event(2),
            _provenance(),
            reason_code="unrelated_rule",
        )
    assert wrong.value.code == "reopen_reason"

    record = machine.transition(
        ReviewState.REOPENED,
        _event(3),
        _provenance(),
        reason_code="corrected_artifact",
    )
    assert record.reason_code == "corrected_artifact"


def test_policy_accepts_an_unnamed_injected_callable() -> None:
    policy = ReviewTransitionPolicy(rules=(lambda request: True,))
    machine = ReviewStateMachine(policy=policy)

    record = machine.transition(
        ReviewState.IN_REVIEW,
        _event("unnamed-rule"),
        _provenance(),
    )

    assert record.to_state is ReviewState.IN_REVIEW
    assert policy.to_dict()["rule_codes"] == ["custom_rule"]


@pytest.mark.parametrize(
    ("event_id", "fingerprint", "code"),
    [
        ("reviewer-name", "sha256:" + "a" * 64, "opaque_event_id_required"),
        (_event(1), "not-a-fingerprint", "provenance_fingerprint_required"),
    ],
)
def test_transition_requires_opaque_event_and_provenance_fingerprint(
    event_id: str,
    fingerprint: str,
    code: str,
) -> None:
    machine = ReviewStateMachine()
    with pytest.raises(ReviewTransitionValidationError) as caught:
        machine.transition(ReviewState.IN_REVIEW, event_id, fingerprint)

    assert caught.value.code == code
    assert machine.history == ()


def test_duplicate_event_ids_are_rejected_without_mutating_state() -> None:
    machine = ReviewStateMachine()
    event_id = _event("duplicate")
    machine.transition(ReviewState.IN_REVIEW, event_id, _provenance())

    with pytest.raises(ReviewTransitionValidationError) as caught:
        machine.transition(ReviewState.EXPIRED, event_id, _provenance())

    assert caught.value.code == "duplicate_event_id"
    assert machine.current_state is ReviewState.IN_REVIEW
    assert len(machine.history) == 1


def test_helpers_and_reports_are_deterministic_and_do_not_copy_sensitive_inputs() -> (
    None
):
    seed = "synthetic-case-content-that-must-not-be-stored"
    event_id = make_opaque_event_id(seed)
    fingerprint = compute_provenance_fingerprint(
        {"source": seed, "schema": "synthetic-review-v1"}
    )

    first = ReviewStateMachine()
    second = ReviewStateMachine()
    for machine in (first, second):
        machine.transition(ReviewState.IN_REVIEW, event_id, fingerprint)

    first_report = first.report().to_dict()
    second_report = second.report().to_dict()
    assert first_report == second_report
    serialized = json.dumps(first_report, sort_keys=True)
    assert seed not in serialized
    assert "reviewer_id" not in first_report
    assert "case_contents" not in first_report


def test_history_validation_ignores_untrusted_extra_fields() -> None:
    machine = ReviewStateMachine()
    machine.transition(ReviewState.IN_REVIEW, _event(1), _provenance())
    payload = machine.history[0].to_dict()
    payload.update(
        {
            "reviewer_id": "synthetic-reviewer-identity",
            "case_contents": "synthetic-case-content",
        }
    )

    report = validate_review_history([payload])
    serialized = json.dumps(report.to_dict(), sort_keys=True)
    assert report.current_state is ReviewState.IN_REVIEW
    assert "synthetic-reviewer-identity" not in serialized
    assert "synthetic-case-content" not in serialized


def test_history_round_trip_and_standalone_validation() -> None:
    machine = ReviewStateMachine()
    machine.transition(ReviewState.IN_REVIEW, _event(1), _provenance())
    machine.transition(ReviewState.APPROVED, _event(2), _provenance())

    report = validate_review_history(machine.history)
    request = validate_transition(
        ReviewState.QUEUED,
        ReviewState.IN_REVIEW,
        _event("standalone"),
        _provenance(),
    )

    assert report.to_dict() == machine.report().to_dict()
    assert request.from_state is ReviewState.QUEUED
    assert request.to_state is ReviewState.IN_REVIEW


def test_invalid_transition_exception_does_not_echo_raw_input() -> None:
    raw_value = "Synthetic Patient Reviewer Case Contents"
    with pytest.raises(ReviewTransitionValidationError) as caught:
        validate_transition(
            ReviewState.QUEUED,
            ReviewState.APPROVED,
            raw_value,
            raw_value,
        )

    assert raw_value not in str(caught.value)
    assert caught.value.code == "opaque_event_id_required"
