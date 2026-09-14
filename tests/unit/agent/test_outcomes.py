"""Tests for stable agent workflow outcome codes."""

from __future__ import annotations

import json
import traceback

import pytest

from openmed.agent.outcomes import (
    OUTCOME_SCHEMA_VERSION,
    OutcomeClass,
    OutcomeError,
    WorkflowOutcome,
    allowed_reason_codes,
)

VALID_EXAMPLES = (
    {"outcome_class": "success", "reason_code": "completed"},
    {"outcome_class": "abstained", "reason_code": "insufficient_evidence"},
    {"outcome_class": "abstained", "reason_code": "out_of_scope"},
    {"outcome_class": "abstained", "reason_code": "low_confidence"},
    {"outcome_class": "review_required", "reason_code": "conflicting_evidence"},
    {"outcome_class": "review_required", "reason_code": "safety_review"},
    {"outcome_class": "review_required", "reason_code": "human_gate"},
    {"outcome_class": "policy_denied", "reason_code": "consent_required"},
    {"outcome_class": "policy_denied", "reason_code": "purpose_mismatch"},
    {"outcome_class": "policy_denied", "reason_code": "phi_policy"},
    {"outcome_class": "failed", "reason_code": "tool_error"},
    {"outcome_class": "failed", "reason_code": "timeout"},
    {"outcome_class": "failed", "reason_code": "invalid_input"},
)


@pytest.mark.parametrize("payload", VALID_EXAMPLES)
def test_valid_outcomes_round_trip_with_stable_json(payload):
    outcome = WorkflowOutcome.from_dict(payload)

    assert WorkflowOutcome.from_json(outcome.to_json()) == outcome
    assert json.loads(outcome.to_json()) == outcome.to_dict()
    assert list(json.loads(outcome.to_json())) == sorted(outcome.to_dict())
    assert outcome.to_json() == outcome.to_json()
    assert outcome.to_json() == json.dumps(
        {
            "outcome_class": payload["outcome_class"],
            "reason_code": payload["reason_code"],
            "schema_version": OUTCOME_SCHEMA_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def test_to_dict_uses_stable_field_order():
    outcome = WorkflowOutcome.from_dict(
        {"reason_code": "completed", "outcome_class": "success"}
    )

    assert list(outcome.to_dict()) == [
        "schema_version",
        "outcome_class",
        "reason_code",
    ]


@pytest.mark.parametrize(
    ("outcome_class", "reason_code"),
    [
        ("success", "insufficient_evidence"),
        ("abstained", "completed"),
        ("review_required", "phi_policy"),
        ("policy_denied", "timeout"),
        ("failed", "human_gate"),
        ("success", "looks_good"),
        ("abstained", "Patient chart is incomplete"),
        ("not-a-real-class", "completed"),
    ],
)
def test_unknown_or_mismatched_codes_fail_without_echo(outcome_class, reason_code):
    payload = {"outcome_class": outcome_class, "reason_code": reason_code}

    with pytest.raises(OutcomeError) as exc_info:
        WorkflowOutcome.from_dict(payload)

    message = str(exc_info.value)
    known_classes = {
        "success",
        "abstained",
        "review_required",
        "policy_denied",
        "failed",
    }
    if outcome_class not in known_classes:
        assert outcome_class not in message
    assert reason_code not in message
    assert exc_info.value.code in {"unknown_class", "unknown_reason"}


def test_free_text_reason_field_fails_without_echo():
    sentinel = "Patient John Doe has AKI; bearer tok_live_sentinel"
    payload = {
        "outcome_class": "failed",
        "reason_code": "tool_error",
        "message": sentinel,
    }

    with pytest.raises(OutcomeError) as exc_info:
        WorkflowOutcome.from_dict(payload)

    assert sentinel not in str(exc_info.value)
    assert exc_info.value.code == "unknown_field"


@pytest.mark.parametrize(
    "payload",
    [
        {"outcome_class": "success"},
        {"reason_code": "completed"},
        "not-a-dict",
        ["success", "completed"],
    ],
)
def test_missing_and_non_mapping_payloads_fail_closed(payload):
    with pytest.raises(OutcomeError):
        WorkflowOutcome.from_dict(payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("outcome_class", True),
        ("outcome_class", 1),
        ("reason_code", True),
        ("reason_code", 0),
        ("schema_version", "openmed.agent.outcome.v0"),
        ("schema_version", True),
    ],
)
def test_invalid_field_types_fail_closed(field, value):
    payload = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "outcome_class": "success",
        "reason_code": "completed",
    }
    payload[field] = value

    with pytest.raises(OutcomeError):
        WorkflowOutcome.from_dict(payload)


@pytest.mark.parametrize("payload", ["{", b"\xff", 42])
def test_malformed_json_fails_closed(payload):
    with pytest.raises(OutcomeError) as exc_info:
        WorkflowOutcome.from_json(payload)  # type: ignore[arg-type]
    assert exc_info.value.code == "malformed_json"


def test_duplicate_json_fields_fail_closed_instead_of_last_value_winning():
    payload = (
        '{"outcome_class":"success","outcome_class":"failed","reason_code":"completed"}'
    )

    with pytest.raises(OutcomeError) as exc_info:
        WorkflowOutcome.from_json(payload)

    assert exc_info.value.code == "malformed_json"


def test_rejected_values_are_absent_from_full_exception_chain():
    sentinel = "Patient Jane Roe /private/chart bearer-token"

    with pytest.raises(OutcomeError) as exc_info:
        WorkflowOutcome.from_json(sentinel)

    rendered = "".join(
        traceback.format_exception(exc_info.type, exc_info.value, exc_info.tb)
    )
    assert sentinel not in rendered
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


class StringSubclass(str):
    """A string subtype that must not be retained in immutable outcomes."""


@pytest.mark.parametrize("field", ["schema_version", "reason_code"])
def test_string_subclasses_are_rejected_by_direct_constructor(field: str):
    values = {
        "outcome_class": OutcomeClass.SUCCESS,
        "reason_code": "completed",
        "schema_version": OUTCOME_SCHEMA_VERSION,
    }
    values[field] = StringSubclass(values[field])

    with pytest.raises(OutcomeError):
        WorkflowOutcome(**values)  # type: ignore[arg-type]


def test_allowed_reason_codes_are_closed_per_class():
    assert "completed" in allowed_reason_codes(OutcomeClass.SUCCESS)
    assert "insufficient_evidence" in allowed_reason_codes("abstained")
    with pytest.raises(OutcomeError) as exc_info:
        allowed_reason_codes("not-a-real-class")
    assert exc_info.value.code == "unknown_class"
    assert "not-a-real-class" not in str(exc_info.value)


def test_outcome_contract_is_available_from_public_agent_api():
    import openmed.agent as agent

    assert agent.WorkflowOutcome is WorkflowOutcome
    assert agent.OutcomeClass is OutcomeClass
    assert agent.OutcomeError is OutcomeError
    assert agent.OUTCOME_SCHEMA_VERSION == OUTCOME_SCHEMA_VERSION
    assert agent.allowed_reason_codes is allowed_reason_codes
