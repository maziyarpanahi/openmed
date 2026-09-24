"""Offline tests for bounded, PHI-safe escalation failure handling."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timedelta, timezone

import pytest

from openmed.agent.correlation import RunId
from openmed.agent.identifiers import WorkflowId
from openmed.agent.runtime.circuit_breaker import (
    CircuitBreakerError,
    CircuitBreakerHandoff,
    CircuitBreakerPolicy,
    EscalationCircuitBreaker,
    FailureClass,
    RecoveryAction,
)

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
RUN_ID = RunId("run_" + "1" * 32)
WORKFLOW_ID = WorkflowId("workflow:org.openmed/escalation-review@1.0.0")


def _breaker(policy: CircuitBreakerPolicy | None = None) -> EscalationCircuitBreaker:
    return EscalationCircuitBreaker(RUN_ID, WORKFLOW_ID, policy)


def _record(
    breaker: EscalationCircuitBreaker,
    failure_class: FailureClass,
    index: int,
) -> CircuitBreakerHandoff | None:
    return breaker.record_failure(failure_class, f"{index:064x}", now=NOW)


@pytest.mark.parametrize(
    ("failure_class", "limit", "reason", "decision", "actions"),
    [
        (
            FailureClass.UNCERTAINTY,
            3,
            "low_confidence",
            "review_evidence",
            ["review_evidence", "abort_run"],
        ),
        (
            FailureClass.PERMISSION_DENIAL,
            2,
            "human_gate",
            "decide_next_step",
            ["recheck_permissions", "abort_run"],
        ),
        (
            FailureClass.TOOL_FAILURE,
            3,
            "safety_review",
            "assess_safety",
            ["review_tool_failure", "abort_run"],
        ),
    ],
)
def test_each_failure_class_halts_at_its_threshold(
    failure_class: FailureClass,
    limit: int,
    reason: str,
    decision: str,
    actions: list[str],
) -> None:
    breaker = _breaker()
    for index in range(1, limit):
        assert _record(breaker, failure_class, index) is None
        assert breaker.halted is False

    handoff = _record(breaker, failure_class, limit)

    assert handoff is breaker.handoff
    assert breaker.halted is True
    assert handoff is not None
    assert handoff.failure_class is failure_class
    assert handoff.attempted_step_digests == tuple(
        f"{index:064x}" for index in range(1, limit + 1)
    )
    assert handoff.packet.reason_code == reason
    assert handoff.packet.requested_decision.value == decision
    assert handoff.packet.requires_human_review is True
    assert handoff.packet.authorizes_clinical_action is False
    assert handoff.packet.issued_at == NOW
    assert handoff.packet.expires_at == NOW + timedelta(minutes=30)
    assert [action.value for action in handoff.allowed_recovery_actions] == actions

    with pytest.raises(CircuitBreakerError, match="circuit_open"):
        _record(breaker, failure_class, limit + 1)
    assert breaker.handoff is handoff


def test_mixed_failures_halt_at_total_limit_without_reset() -> None:
    breaker = _breaker(
        CircuitBreakerPolicy(
            uncertainty=8, permission_denial=8, tool_failure=8, total=4
        )
    )
    classes = (
        FailureClass.UNCERTAINTY,
        FailureClass.PERMISSION_DENIAL,
        FailureClass.TOOL_FAILURE,
        FailureClass.UNCERTAINTY,
    )
    for index, failure_class in enumerate(classes[:-1], 1):
        assert _record(breaker, failure_class, index) is None

    handoff = _record(breaker, classes[-1], 4)
    assert handoff is not None
    assert handoff.failure_class is FailureClass.UNCERTAINTY
    assert handoff.attempted_step_digests == tuple(
        f"{index:064x}" for index in range(1, 5)
    )


def test_one_failure_limit_trips_immediately_and_serializes_deterministically() -> None:
    policy = CircuitBreakerPolicy(uncertainty=1)
    first = _record(_breaker(policy), FailureClass.UNCERTAINTY, 1)
    second = _record(_breaker(policy), FailureClass.UNCERTAINTY, 1)

    assert first is not None and second is not None
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()
    assert set(first.to_dict()) == {
        "reviewer_packet",
        "failure_class",
        "attempted_step_digests",
        "allowed_recovery_actions",
    }
    assert first.to_dict()["reviewer_packet"]["evidence_references"] == []


@pytest.mark.parametrize("value", [0, -1, 33, True, 1.5, "3"])
def test_thresholds_are_positive_bounded_integers(value: object) -> None:
    with pytest.raises(CircuitBreakerError) as caught:
        CircuitBreakerPolicy(total=value)  # type: ignore[arg-type]
    assert caught.value.code == "invalid_threshold"
    assert caught.value.field_name == "total"


def test_invalid_inputs_do_not_advance_the_circuit_or_echo_values() -> None:
    breaker = _breaker(CircuitBreakerPolicy(uncertainty=1))
    sentinel = "synthetic clinical identifier and bearer credential"

    for kwargs, code in (
        ({"failure_class": sentinel, "step_digest": "a" * 64}, "unknown_failure_class"),
        (
            {"failure_class": FailureClass.UNCERTAINTY, "step_digest": sentinel},
            "invalid_digest",
        ),
        (
            {"failure_class": FailureClass.UNCERTAINTY, "step_digest": "A" * 64},
            "invalid_digest",
        ),
    ):
        with pytest.raises(CircuitBreakerError) as caught:
            breaker.record_failure(now=NOW, **kwargs)  # type: ignore[arg-type]
        assert caught.value.code == code
        assert sentinel not in "".join(traceback.format_exception(caught.value))
        assert breaker.halted is False

    handoff = _record(breaker, FailureClass.UNCERTAINTY, 1)
    assert handoff is not None
    assert handoff.attempted_step_digests == (f"{1:064x}",)
    assert sentinel not in handoff.to_json()


def test_invalid_clock_and_overflow_do_not_record_a_failure() -> None:
    breaker = _breaker(CircuitBreakerPolicy(uncertainty=1))
    for invalid in (NOW.replace(tzinfo=None), NOW.replace(microsecond=1), "now"):
        with pytest.raises(CircuitBreakerError) as caught:
            breaker.record_failure(
                FailureClass.UNCERTAINTY,
                "a" * 64,
                now=invalid,  # type: ignore[arg-type]
            )
        assert caught.value.code == "invalid_timestamp"
        assert breaker.halted is False

    with pytest.raises(CircuitBreakerError, match="invalid_timestamp"):
        breaker.record_failure(
            FailureClass.UNCERTAINTY,
            "a" * 64,
            now=datetime.max.replace(microsecond=0, tzinfo=timezone.utc),
        )
    assert breaker.halted is False
    assert _record(breaker, FailureClass.UNCERTAINTY, 1) is not None


def test_direct_handoff_construction_cannot_smuggle_content_or_actions() -> None:
    valid = _record(
        _breaker(CircuitBreakerPolicy(uncertainty=1)), FailureClass.UNCERTAINTY, 1
    )
    assert valid is not None

    with pytest.raises(CircuitBreakerError, match="invalid_digests"):
        CircuitBreakerHandoff(
            packet=valid.packet,
            failure_class=FailureClass.UNCERTAINTY,
            attempted_step_digests=("synthetic patient text",),
            allowed_recovery_actions=valid.allowed_recovery_actions,
        )
    with pytest.raises(CircuitBreakerError, match="invalid_recovery_actions"):
        CircuitBreakerHandoff(
            packet=valid.packet,
            failure_class=FailureClass.UNCERTAINTY,
            attempted_step_digests=valid.attempted_step_digests,
            allowed_recovery_actions=(RecoveryAction.ABORT_RUN,),
        )
    with pytest.raises(CircuitBreakerError, match="invalid_recovery_actions"):
        CircuitBreakerHandoff(
            packet=valid.packet,
            failure_class=FailureClass.UNCERTAINTY,
            attempted_step_digests=valid.attempted_step_digests,
            allowed_recovery_actions=("synthetic patient text",),  # type: ignore[arg-type]
        )
    with pytest.raises(CircuitBreakerError, match="invalid_packet"):
        CircuitBreakerHandoff(
            packet=valid.packet,
            failure_class=FailureClass.TOOL_FAILURE,
            attempted_step_digests=valid.attempted_step_digests,
            allowed_recovery_actions=(
                RecoveryAction.REVIEW_TOOL_FAILURE,
                RecoveryAction.ABORT_RUN,
            ),
        )
