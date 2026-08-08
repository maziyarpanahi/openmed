"""Regression tests for structure-aware tool-call redaction."""

from __future__ import annotations

import json

import pytest

from openmed.traces.tool_calls import (
    ToolCallRedactionError,
    redact_tool_call,
    redact_tool_call_with_report,
    redact_tool_calls,
)

SYNTHETIC_NAME = "synthetic-name-001"
SYNTHETIC_PHONE = "synthetic-phone-001"
SYNTHETIC_NOTE = "synthetic-note-001"


def _redactor(text: str) -> str:
    return (
        text.replace(SYNTHETIC_NAME, "[NAME]")
        .replace(SYNTHETIC_PHONE, "[PHONE]")
        .replace(SYNTHETIC_NOTE, "[NOTE]")
    )


def test_redacts_string_leaves_and_preserves_tool_structure():
    payload = {
        "id": "call-001",
        "function": {
            "name": "lookup",
            "arguments": {
                "patient": SYNTHETIC_NAME,
                "nested": [{"phone": SYNTHETIC_PHONE}],
                "count": 7,
                "enabled": True,
                "optional": None,
            },
        },
        "result": {
            "note": SYNTHETIC_NOTE,
            "items": [1, False, None],
        },
        "unconfigured": SYNTHETIC_NAME,
    }

    redacted = redact_tool_call(payload, text_redactor=_redactor)

    assert redacted == {
        "id": "call-001",
        "function": {
            "name": "lookup",
            "arguments": {
                "patient": "[NAME]",
                "nested": [{"phone": "[PHONE]"}],
                "count": 7,
                "enabled": True,
                "optional": None,
            },
        },
        "result": {
            "note": "[NOTE]",
            "items": [1, False, None],
        },
        "unconfigured": SYNTHETIC_NAME,
    }
    assert payload["function"]["arguments"]["patient"] == SYNTHETIC_NAME


def test_json_encoded_arguments_are_redacted_and_serialized_deterministically():
    arguments = json.dumps(
        {"name": SYNTHETIC_NAME, "count": 3, "active": True},
        indent=2,
    )
    payload = {"function": {"arguments": arguments}}

    first = redact_tool_call(payload, text_redactor=_redactor)
    second = redact_tool_call(payload, text_redactor=_redactor)

    assert first == second
    assert isinstance(first["function"]["arguments"], str)
    assert json.loads(first["function"]["arguments"]) == {
        "active": True,
        "count": 3,
        "name": "[NAME]",
    }
    assert SYNTHETIC_NAME not in first["function"]["arguments"]


def test_wildcard_paths_reach_nested_tool_calls_without_touching_other_content():
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"arguments": {"name": SYNTHETIC_NAME}}},
                    {"function": {"arguments": {"name": SYNTHETIC_PHONE}}},
                ],
            }
        ],
        "metadata": SYNTHETIC_NAME,
    }

    redacted = redact_tool_call(
        payload,
        content_paths=("messages.*.tool_calls.*.function.arguments",),
        text_redactor=_redactor,
    )

    assert redacted["messages"][0]["tool_calls"][0]["function"]["arguments"] == {
        "name": "[NAME]"
    }
    assert redacted["messages"][0]["tool_calls"][1]["function"]["arguments"] == {
        "name": "[PHONE]"
    }
    assert redacted["metadata"] == SYNTHETIC_NAME


def test_json_pointer_and_integer_segment_paths_select_array_items():
    payload = {
        "messages": [
            {"result": {"name": SYNTHETIC_NAME}},
            {"result": {"name": SYNTHETIC_PHONE}},
        ]
    }

    pointer_redacted = redact_tool_call(
        payload,
        content_paths=("/messages/0/result",),
        text_redactor=_redactor,
    )
    segment_redacted = redact_tool_call(
        payload,
        content_paths=("messages", 1, "result"),
        text_redactor=_redactor,
    )

    assert pointer_redacted["messages"][0]["result"]["name"] == "[NAME]"
    assert pointer_redacted["messages"][1]["result"]["name"] == SYNTHETIC_PHONE
    assert segment_redacted["messages"][0]["result"]["name"] == SYNTHETIC_NAME
    assert segment_redacted["messages"][1]["result"]["name"] == "[PHONE]"


def test_malformed_json_falls_back_to_text_redaction_without_leaking_payload():
    malformed = '{"name": "synthetic-name-001",'
    result = redact_tool_call_with_report(
        {"arguments": malformed},
        text_redactor=_redactor,
    )

    assert result.value["arguments"] == '{"name": "[NAME]",'
    assert SYNTHETIC_NAME not in result.value["arguments"]
    assert result.report.malformed_payload_count == 1
    report_json = json.dumps(result.report.to_dict(), sort_keys=True)
    assert SYNTHETIC_NAME not in report_json


def test_report_is_deterministic_and_contains_only_safe_metadata():
    payload = {
        "arguments": {"name": SYNTHETIC_NAME},
        "result": {"phone": SYNTHETIC_PHONE},
    }

    first = redact_tool_call_with_report(payload, text_redactor=_redactor)
    second = redact_tool_call_with_report(payload, text_redactor=_redactor)

    assert first.report.to_dict() == second.report.to_dict()
    assert first.report.redacted_leaf_count == 2
    assert first.report.malformed_payload_count == 0
    assert SYNTHETIC_NAME not in first.report.summary()
    assert SYNTHETIC_PHONE not in json.dumps(first.report.to_dict())


def test_redactor_failure_has_a_safe_exception_message():
    def failing_redactor(text: str) -> str:
        raise RuntimeError(f"unexpected value: {text}")

    with pytest.raises(ToolCallRedactionError) as error:
        redact_tool_call(
            {"arguments": {"name": SYNTHETIC_NAME}},
            text_redactor=failing_redactor,
        )

    assert SYNTHETIC_NAME not in str(error.value)
    assert error.value.__cause__ is None


def test_redact_tool_calls_preserves_input_order():
    payloads = [
        {"arguments": {"name": SYNTHETIC_NAME}},
        {"arguments": {"name": SYNTHETIC_PHONE}},
    ]

    redacted = redact_tool_calls(payloads, text_redactor=_redactor)

    assert [item["arguments"]["name"] for item in redacted] == [
        "[NAME]",
        "[PHONE]",
    ]
