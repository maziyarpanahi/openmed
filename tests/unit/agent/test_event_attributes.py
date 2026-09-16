"""Synthetic unit tests for the allowlisted agent event attribute contract."""

from __future__ import annotations

import json

import pytest

from openmed.agent.event_attributes import (
    ALLOWED_ATTRIBUTES,
    EVENT_ATTRIBUTES_SCHEMA_VERSION,
    EXECUTION_STAGES,
    MAX_ATTRIBUTE_COUNT,
    MAX_ATTRIBUTE_JSON_BYTES,
    MAX_COUNT_VALUE,
    MAX_DURATION_MS,
    AttributeKind,
    EventAttributeError,
    EventAttributes,
    validate_event_attributes,
)
from openmed.agent.outcomes import OutcomeClass

RUN_ID = "run_0123456789abcdef0123456789abcdef"
ACTION_ID = "act_0123456789abcdef0123456789abcdef"
DIGEST = "sha256:" + "0" * 64

SENTINEL_PROMPT = "Summarize the note for JANE DOE, MRN 8675309"
SENTINEL_BEARER = "Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature"
SENTINEL_PATH = "/var/phi/notes/jane-doe.txt"

VALID = {
    "run_id": RUN_ID,
    "action_id": ACTION_ID,
    "parent_action_id": ACTION_ID,
    "capability_id": "capability:openmed.agent/summarize",
    "policy_id": "policy:openmed.agent/phi-guard@1.0.0",
    "purpose_id": "purpose:openmed.agent/care-coordination",
    "tool_id": "tool:openmed.agent/summarize@1.0.0",
    "workflow_id": "workflow:openmed.agent/discharge",
    "execution_stage": "tool_call",
    "outcome_class": "success",
    "outcome_reason": "completed",
    "input_digest": DIGEST,
    "output_digest": DIGEST,
    "artifact_digest": DIGEST,
    "sequence_number": 3,
    "attempt_number": 1,
    "retry_count": 0,
    "tool_call_count": 2,
    "artifact_count": 1,
    "duration_ms": 12.5,
    "retryable": False,
    "redacted": True,
}


def test_every_allowlisted_attribute_validates_together() -> None:
    attributes = EventAttributes.from_mapping(VALID)
    assert set(attributes.values) == set(ALLOWED_ATTRIBUTES)
    assert attributes.schema_version == EVENT_ATTRIBUTES_SCHEMA_VERSION
    assert attributes.get("sequence_number") == 3
    assert attributes.get("absent") is None


def test_the_allowlist_covers_every_declared_kind() -> None:
    assert set(ALLOWED_ATTRIBUTES.values()) == set(AttributeKind)
    assert set(VALID) == set(ALLOWED_ATTRIBUTES)


def test_serialization_is_byte_stable_and_key_ordered() -> None:
    attributes = EventAttributes.from_mapping(VALID)
    payload = attributes.to_dict()
    assert list(payload)[0] == "schema_version"
    assert list(payload)[1:] == sorted(ALLOWED_ATTRIBUTES)
    assert json.loads(attributes.to_json()) == payload
    assert attributes.to_json() == EventAttributes.from_mapping(VALID).to_json()


def test_values_are_immutable() -> None:
    attributes = EventAttributes.from_mapping({"retryable": True})
    with pytest.raises(TypeError):
        attributes.values["retryable"] = False  # type: ignore[index]


def test_empty_attributes_are_valid() -> None:
    attributes = EventAttributes.from_mapping({})
    assert attributes.to_dict() == {"schema_version": EVENT_ATTRIBUTES_SCHEMA_VERSION}


@pytest.mark.parametrize("stage", EXECUTION_STAGES)
def test_every_execution_stage_is_accepted(stage) -> None:
    assert validate_event_attributes({"execution_stage": stage}) == {
        "execution_stage": stage
    }


@pytest.mark.parametrize("stage", ["", "Running", "tool-call", "unknown", 1, None])
def test_unknown_execution_stages_fail_closed(stage) -> None:
    with pytest.raises(EventAttributeError) as excinfo:
        validate_event_attributes({"execution_stage": stage})
    assert (excinfo.value.code, excinfo.value.field_name) in {
        ("unknown_execution_stage", "execution_stage"),
        ("nested_value_not_allowed", "execution_stage"),
    }


@pytest.mark.parametrize("outcome", list(OutcomeClass))
def test_outcome_classes_accept_their_own_reason_codes(outcome) -> None:
    from openmed.agent.outcomes import allowed_reason_codes

    for reason in sorted(allowed_reason_codes(outcome)):
        validated = validate_event_attributes(
            {"outcome_class": outcome, "outcome_reason": reason}
        )
        assert validated == {
            "outcome_class": outcome.value,
            "outcome_reason": reason,
        }


def test_reason_codes_from_another_class_fail_closed() -> None:
    with pytest.raises(EventAttributeError) as excinfo:
        validate_event_attributes(
            {"outcome_class": "success", "outcome_reason": "timeout"}
        )
    assert (excinfo.value.code, excinfo.value.field_name) == (
        "unknown_outcome_reason",
        "outcome_reason",
    )


def test_a_reason_code_requires_its_outcome_class() -> None:
    with pytest.raises(EventAttributeError) as excinfo:
        validate_event_attributes({"outcome_reason": "completed"})
    assert excinfo.value.code == "outcome_class_required"


@pytest.mark.parametrize("value", ["", "Success", "done", 1, None])
def test_unknown_outcome_classes_fail_closed(value) -> None:
    with pytest.raises(EventAttributeError, match="outcome_class"):
        validate_event_attributes({"outcome_class": value})


@pytest.mark.parametrize("field", ["run_id", "action_id", "parent_action_id"])
@pytest.mark.parametrize("value", ["", "run_x", ACTION_ID[:-1], 7, None])
def test_invalid_correlation_identifiers_fail_closed(field, value) -> None:
    payload = {field: RUN_ID if field == "run_id" else ACTION_ID}
    payload[field] = value
    with pytest.raises(EventAttributeError) as excinfo:
        validate_event_attributes(payload)
    assert excinfo.value.field_name == field


def test_correlation_kinds_are_not_interchangeable() -> None:
    with pytest.raises(EventAttributeError, match="^run_id: invalid_identifier$"):
        validate_event_attributes({"run_id": ACTION_ID})
    with pytest.raises(EventAttributeError, match="^action_id: invalid_identifier$"):
        validate_event_attributes({"action_id": RUN_ID})


@pytest.mark.parametrize(
    "field", ["capability_id", "policy_id", "purpose_id", "workflow_id"]
)
def test_governance_identifier_kinds_are_enforced(field) -> None:
    with pytest.raises(EventAttributeError) as excinfo:
        validate_event_attributes({field: "tool:openmed.agent/summarize"})
    assert (excinfo.value.code, excinfo.value.field_name) == (
        "invalid_identifier",
        field,
    )


def test_a_tool_identifier_is_accepted_only_under_tool_id() -> None:
    assert validate_event_attributes({"tool_id": "tool:openmed.agent/summarize"}) == {
        "tool_id": "tool:openmed.agent/summarize"
    }


def test_a_serialized_schema_version_round_trips_without_becoming_an_attribute() -> (
    None
):
    payload = {"schema_version": EVENT_ATTRIBUTES_SCHEMA_VERSION, "retry_count": 1}
    assert validate_event_attributes(payload) == {"retry_count": 1}
    with pytest.raises(EventAttributeError, match="^schema_version: "):
        validate_event_attributes({"schema_version": "openmed.agent.v0"})


@pytest.mark.parametrize("field", ["input_digest", "output_digest", "artifact_digest"])
@pytest.mark.parametrize(
    "value",
    ["", "sha256:" + "0" * 63, "sha256:" + "A" * 64, "md5:" + "0" * 32, "0" * 64, 7],
)
def test_invalid_digests_fail_closed(field, value) -> None:
    with pytest.raises(EventAttributeError) as excinfo:
        validate_event_attributes({field: value})
    assert excinfo.value.field_name == field


@pytest.mark.parametrize(
    "field",
    [
        "sequence_number",
        "attempt_number",
        "retry_count",
        "tool_call_count",
        "artifact_count",
    ],
)
def test_counts_reject_booleans_and_floats(field) -> None:
    for value in (True, False, 1.0, "1", None):
        with pytest.raises(EventAttributeError) as excinfo:
            validate_event_attributes({field: value})
        assert excinfo.value.code in {"invalid_count", "nested_value_not_allowed"}
        assert excinfo.value.field_name == field


def test_counts_are_bounded() -> None:
    assert validate_event_attributes({"retry_count": MAX_COUNT_VALUE}) == {
        "retry_count": MAX_COUNT_VALUE
    }
    for value in (-1, MAX_COUNT_VALUE + 1, 2**70):
        with pytest.raises(
            EventAttributeError, match="^retry_count: count_out_of_range$"
        ):
            validate_event_attributes({"retry_count": value})


def test_durations_accept_integers_and_finite_floats() -> None:
    assert validate_event_attributes({"duration_ms": 5}) == {"duration_ms": 5.0}
    assert validate_event_attributes({"duration_ms": 0.25}) == {"duration_ms": 0.25}
    assert (
        validate_event_attributes({"duration_ms": MAX_DURATION_MS})["duration_ms"]
        == MAX_DURATION_MS
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_durations_fail_closed(value) -> None:
    with pytest.raises(EventAttributeError) as excinfo:
        validate_event_attributes({"duration_ms": value})
    assert (excinfo.value.code, excinfo.value.field_name) == (
        "non_finite_number",
        "duration_ms",
    )


@pytest.mark.parametrize("value", [-1, MAX_DURATION_MS + 1])
def test_out_of_range_durations_fail_closed(value) -> None:
    with pytest.raises(
        EventAttributeError, match="^duration_ms: duration_out_of_range$"
    ):
        validate_event_attributes({"duration_ms": value})


@pytest.mark.parametrize("value", [True, "1", None])
def test_invalid_duration_types_fail_closed(value) -> None:
    with pytest.raises(EventAttributeError) as excinfo:
        validate_event_attributes({"duration_ms": value})
    assert excinfo.value.code in {"invalid_duration", "non_finite_number"}


@pytest.mark.parametrize("field", ["retryable", "redacted"])
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_flags_require_real_booleans(field, value) -> None:
    with pytest.raises(EventAttributeError, match=f"^{field}: invalid_flag$"):
        validate_event_attributes({field: value})


@pytest.mark.parametrize(
    "key", ["stage", "count", "result", "step", "kind", "extra", "openmed"]
)
def test_unknown_keys_fail_without_echoing_the_key(key) -> None:
    with pytest.raises(EventAttributeError) as excinfo:
        validate_event_attributes({key: 1})
    assert excinfo.value.code == "unknown_attribute"
    assert excinfo.value.field_name is None
    assert key not in str(excinfo.value)


@pytest.mark.parametrize(
    "key",
    [
        "prompt",
        "prompt_text",
        "tool_arguments",
        "patient_name",
        "mrn",
        "note_text",
        "file_path",
        "callback_url",
        "authorization",
        "bearer_token",
        "api_key",
        "password",
        "session_cookie",
        "credentials",
        "email_address",
        "phone_number",
        "message_body",
        "raw_input",
    ],
)
def test_sensitive_keys_are_reported_as_a_distinct_class(key) -> None:
    with pytest.raises(EventAttributeError) as excinfo:
        validate_event_attributes({key: "x"})
    assert excinfo.value.code == "sensitive_attribute_key"
    assert excinfo.value.field_name is None
    assert key not in str(excinfo.value)


@pytest.mark.parametrize(
    "value", [{"a": 1}, [1, 2], (1, 2), {1, 2}, frozenset({1}), b"x", bytearray(b"x")]
)
def test_nested_and_binary_values_fail_closed(value) -> None:
    with pytest.raises(
        EventAttributeError, match="^retry_count: nested_value_not_allowed$"
    ):
        validate_event_attributes({"retry_count": value})


def test_non_string_keys_fail_closed() -> None:
    with pytest.raises(EventAttributeError, match="^invalid_key_type$"):
        validate_event_attributes({7: 1})


@pytest.mark.parametrize("value", ["abc", b"abc", 7, None, [("a", 1)]])
def test_non_mapping_input_fails_closed(value) -> None:
    with pytest.raises(EventAttributeError, match="^not_a_mapping$"):
        validate_event_attributes(value)


def test_oversized_mappings_fail_before_validation() -> None:
    payload = {f"k{index:03d}": 1 for index in range(MAX_ATTRIBUTE_COUNT + 1)}
    with pytest.raises(EventAttributeError, match="^too_many_attributes$"):
        validate_event_attributes(payload)


def test_json_round_trip() -> None:
    attributes = EventAttributes.from_mapping(VALID)
    assert EventAttributes.from_json(attributes.to_json()).to_json() == (
        attributes.to_json()
    )
    assert EventAttributes.from_json(b'{"retry_count": 2}').get("retry_count") == 2


def test_duplicate_json_keys_fail_closed() -> None:
    with pytest.raises(EventAttributeError, match="^duplicate_attribute$"):
        EventAttributes.from_json('{"retryable": true, "retryable": false}')


@pytest.mark.parametrize("payload", ["", "{", "[]", "null", "7", '"text"', 7, None])
def test_malformed_json_fails_closed(payload) -> None:
    with pytest.raises(EventAttributeError) as excinfo:
        EventAttributes.from_json(payload)
    assert excinfo.value.code in {"malformed_json", "not_a_mapping"}


def test_oversized_json_fails_before_parsing() -> None:
    payload = '{"note": "' + "x" * MAX_ATTRIBUTE_JSON_BYTES + '"}'
    with pytest.raises(EventAttributeError, match="^payload_too_large$"):
        EventAttributes.from_json(payload)


def test_invalid_schema_version_fails_closed() -> None:
    with pytest.raises(EventAttributeError, match="^schema_version: "):
        EventAttributes(values={}, schema_version="openmed.agent.event_attributes.v0")


@pytest.mark.parametrize("sentinel", [SENTINEL_PROMPT, SENTINEL_BEARER, SENTINEL_PATH])
@pytest.mark.parametrize("key", ["prompt", "tool_arguments", "run_id", "input_digest"])
def test_sentinels_never_reach_output_or_exceptions(sentinel, key) -> None:
    with pytest.raises(EventAttributeError) as excinfo:
        validate_event_attributes({key: sentinel})
    assert sentinel not in str(excinfo.value)
    assert sentinel not in repr(excinfo.value)
    assert excinfo.value.__cause__ is None

    attributes = EventAttributes.from_mapping({"run_id": RUN_ID})
    assert sentinel not in attributes.to_json()


def test_event_attributes_are_available_from_the_public_agent_api() -> None:
    import openmed.agent as agent

    assert agent.EventAttributes is EventAttributes
    assert agent.EventAttributeError is EventAttributeError
    assert agent.validate_event_attributes is validate_event_attributes
    exported = {
        "ALLOWED_ATTRIBUTES",
        "AttributeKind",
        "EVENT_ATTRIBUTES_SCHEMA_VERSION",
        "EXECUTION_STAGES",
        "EventAttributeError",
        "EventAttributes",
        "MAX_ATTRIBUTE_COUNT",
        "MAX_ATTRIBUTE_JSON_BYTES",
        "MAX_COUNT_VALUE",
        "MAX_DURATION_MS",
        "validate_event_attributes",
    }
    assert exported.issubset(set(agent.__all__))
