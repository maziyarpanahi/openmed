"""Synthetic tests for digest-keyed summary retry control."""

from __future__ import annotations

import hashlib
import json

import pytest

from openmed.clinical.summary_retry_control import (
    REPEATED_FAILURE_REFUSAL,
    RETRY_CEILING_REFUSAL,
    SummaryRetryControlError,
    SummaryRetryController,
    SummaryRetryPolicy,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


INPUT = _digest("synthetic-input")
MODEL = _digest("local-model")


def test_attempt_ceiling_returns_a_sticky_auditable_refusal() -> None:
    controller = SummaryRetryController(
        SummaryRetryPolicy(max_attempts=2, repeated_failure_limit=3)
    )

    first = controller.begin_attempt(INPUT, MODEL)
    first_failure = controller.record_failure(INPUT, MODEL, "citation-check")
    second = controller.begin_attempt(INPUT, MODEL)
    refusal = controller.record_failure(INPUT, MODEL, "coverage-check")
    repeated_request = controller.begin_attempt(INPUT, MODEL)

    assert first.action == "generate"
    assert first.attempts_started == 1
    assert first_failure.action == "retry"
    assert second.action == "generate"
    assert second.attempts_started == 2
    assert refusal.refused is True
    assert refusal.refusal_code == RETRY_CEILING_REFUSAL
    assert refusal.attempts_remaining == 0
    assert repeated_request == refusal


def test_repeated_failure_class_stops_before_ceiling() -> None:
    controller = SummaryRetryController(SummaryRetryPolicy(max_attempts=5))

    controller.begin_attempt(INPUT, MODEL)
    controller.record_failure(INPUT, MODEL, "same-safety-gate")
    controller.begin_attempt(INPUT, MODEL)
    refusal = controller.record_failure(INPUT, MODEL, "same-safety-gate")

    assert refusal.refused is True
    assert refusal.refusal_code == REPEATED_FAILURE_REFUSAL
    assert refusal.attempts_started == 2
    assert controller.begin_attempt(INPUT, MODEL) == refusal


def test_failure_classes_are_counted_independently() -> None:
    controller = SummaryRetryController(SummaryRetryPolicy(max_attempts=4))

    controller.begin_attempt(INPUT, MODEL)
    assert controller.record_failure(INPUT, MODEL, "coverage").action == "retry"
    controller.begin_attempt(INPUT, MODEL)
    assert controller.record_failure(INPUT, MODEL, "leakage").action == "retry"
    third = controller.begin_attempt(INPUT, MODEL)

    assert third.action == "generate"
    assert third.attempts_started == 3


def test_input_and_model_digest_pairs_have_independent_state() -> None:
    controller = SummaryRetryController(SummaryRetryPolicy(max_attempts=1))
    other_input = _digest("other-input")
    other_model = _digest("other-model")

    controller.begin_attempt(INPUT, MODEL)
    first_refusal = controller.record_failure(INPUT, MODEL, "failure")
    other_input_attempt = controller.begin_attempt(other_input, MODEL)
    other_model_attempt = controller.begin_attempt(INPUT, other_model)

    assert first_refusal.refusal_code == RETRY_CEILING_REFUSAL
    assert other_input_attempt.action == "generate"
    assert other_model_attempt.action == "generate"
    assert (
        len(
            {
                first_refusal.generation_key,
                other_input_attempt.generation_key,
                other_model_attempt.generation_key,
            }
        )
        == 3
    )


def test_success_clears_state_for_a_future_generation_run() -> None:
    controller = SummaryRetryController(SummaryRetryPolicy(max_attempts=2))

    first = controller.begin_attempt(INPUT, MODEL)
    complete = controller.record_success(INPUT, MODEL)
    restarted = controller.begin_attempt(INPUT, MODEL)

    assert first.attempts_started == 1
    assert complete.action == "complete"
    assert restarted.attempts_started == 1


def test_audit_log_and_errors_contain_no_raw_values() -> None:
    sentinel = "SYNTHETIC_SAFETY_FAILURE_WITH_PATIENT_VALUE"
    controller = SummaryRetryController(SummaryRetryPolicy(max_attempts=1))

    controller.begin_attempt(INPUT, MODEL)
    decision = controller.record_failure(INPUT, MODEL, sentinel)
    serialized = json.dumps(controller.audit_log(), sort_keys=True)

    assert decision.refusal_code == RETRY_CEILING_REFUSAL
    assert sentinel not in serialized
    assert INPUT not in serialized
    assert MODEL not in serialized
    assert json.loads(decision.to_json()) == decision.to_dict()

    with pytest.raises(SummaryRetryControlError) as error:
        controller.begin_attempt(sentinel, MODEL)
    assert sentinel not in str(error.value)


def test_invalid_transitions_and_policy_values_fail_closed() -> None:
    controller = SummaryRetryController(SummaryRetryPolicy(max_attempts=2))

    with pytest.raises(SummaryRetryControlError, match="no active"):
        controller.record_failure(INPUT, MODEL, "failure")
    controller.begin_attempt(INPUT, MODEL)
    with pytest.raises(SummaryRetryControlError, match="already active"):
        controller.begin_attempt(INPUT, MODEL)
    controller.record_failure(INPUT, MODEL, "failure")
    with pytest.raises(SummaryRetryControlError, match="no active"):
        controller.record_success(INPUT, MODEL)

    for invalid in (0, -1, True):
        with pytest.raises(SummaryRetryControlError):
            SummaryRetryPolicy(max_attempts=invalid)  # type: ignore[arg-type]
    with pytest.raises(SummaryRetryControlError, match="at least two"):
        SummaryRetryPolicy(max_attempts=2, repeated_failure_limit=1)
