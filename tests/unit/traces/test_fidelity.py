"""Focused tests for deterministic, value-free trace fidelity checks."""

from __future__ import annotations

import copy
import json

import pytest

from openmed.traces.fidelity import (
    TraceFidelityError,
    TraceFidelityReport,
    TraceFidelityVerifier,
    assert_trace_fidelity,
    verify_trace_fidelity,
)

SYNTHETIC_CONTENT = "SYNTHETIC_CONTENT_VALUE"
SYNTHETIC_RESULT = "SYNTHETIC_RESULT_VALUE"


def _trace() -> dict[str, object]:
    return {
        "trace_id": "trace-synthetic-1",
        "messages": [
            {
                "id": "message-synthetic-1",
                "role": "user",
                "content": SYNTHETIC_CONTENT,
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "id": "message-synthetic-2",
                "role": "assistant",
                "content": SYNTHETIC_RESULT,
                "tool_calls": [
                    {
                        "id": "call-synthetic-1",
                        "function": {"name": "lookup", "arguments": {"n": 1}},
                    }
                ],
            },
            {
                "id": "message-synthetic-3",
                "role": "tool",
                "tool_call_id": "call-synthetic-1",
                "content": "SYNTHETIC_TOOL_OUTPUT",
            },
        ],
        "label": "accepted",
        "score": 1.0,
    }


def test_content_changes_are_allowed_but_structure_and_types_are_checked() -> None:
    original = _trace()
    output = copy.deepcopy(original)
    output["messages"][0]["content"] = "SYNTHETIC_REDACTED_CONTENT"
    output["messages"][2]["content"] = "SYNTHETIC_REDACTED_TOOL_OUTPUT"

    report = verify_trace_fidelity(original, output)

    assert isinstance(report, TraceFidelityReport)
    assert report.passed
    assert report.message_order_valid
    assert report.call_linkage_valid
    assert report.content_field_count == 3

    serialized = report.to_json()
    assert SYNTHETIC_CONTENT not in serialized
    assert SYNTHETIC_RESULT not in serialized
    assert json.loads(serialized) == report.to_dict()


def test_message_order_failure_reports_only_the_sequence_path() -> None:
    original = _trace()
    output = copy.deepcopy(original)
    output["messages"][0], output["messages"][1] = (
        output["messages"][1],
        output["messages"][0],
    )

    report = verify_trace_fidelity(original, output)

    assert report.has_code("message_order")
    assert report.failing_paths == ("$.messages",)
    assert SYNTHETIC_CONTENT not in report.summary()


@pytest.mark.parametrize(
    ("field", "replacement", "code"),
    [
        ("trace_id", "trace-synthetic-2", "identifier"),
        ("timestamp", "2026-01-02T00:00:00Z", "timestamp"),
        ("tool_call_id", "call-synthetic-2", "call_linkage"),
        ("label", "rejected", "training_label"),
    ],
)
def test_semantic_changes_are_classified_without_values(
    field: str,
    replacement: str,
    code: str,
) -> None:
    original = _trace()
    output = copy.deepcopy(original)
    if field == "trace_id":
        output[field] = replacement
    elif field == "timestamp":
        output["messages"][0][field] = replacement
    elif field == "tool_call_id":
        output["messages"][2][field] = replacement
    else:
        output[field] = replacement

    report = verify_trace_fidelity(original, output)

    assert report.has_code(code)
    assert replacement not in report.to_json()
    assert SYNTHETIC_CONTENT not in report.to_json()


def test_scalar_type_changes_fail_even_when_the_field_is_declared_content() -> None:
    original = _trace()
    output = copy.deepcopy(original)
    output["messages"][0]["content"] = ["SYNTHETIC_WRONG_SHAPE"]

    report = verify_trace_fidelity(
        original,
        output,
        content_paths=[("messages", "*", "content")],
    )

    assert report.has_code("scalar_type")
    assert report.scalar_types_valid is False
    assert "SYNTHETIC_WRONG_SHAPE" not in report.to_json()


def test_explicit_content_paths_can_be_strict_and_verifier_is_reusable() -> None:
    original = _trace()
    output = copy.deepcopy(original)
    output["messages"][0]["content"] = "SYNTHETIC_CHANGED_CONTENT"

    strict = verify_trace_fidelity(original, output, content_paths=[])
    configured = TraceFidelityVerifier(content_fields=["**.content"])
    first = configured.verify(original, output)
    second = configured(original, output)

    assert strict.has_code("scalar_value")
    assert first == second
    assert first.passed
    assert configured.allowed_content_fields == ("$.**.content",)


def test_assertion_error_contains_no_source_or_output_values() -> None:
    original = _trace()
    output = copy.deepcopy(original)
    output["label"] = "SYNTHETIC_CHANGED_LABEL"

    with pytest.raises(TraceFidelityError) as caught:
        assert_trace_fidelity(original, output)

    assert "SYNTHETIC_CHANGED_LABEL" not in str(caught.value)
    assert "SYNTHETIC_CONTENT_VALUE" not in str(caught.value)
    assert caught.value.report.has_code("training_label")


def test_multiple_content_aliases_are_rejected_to_keep_policy_unambiguous() -> None:
    with pytest.raises(TypeError, match="only one content"):
        verify_trace_fidelity(
            _trace(),
            _trace(),
            content_fields=["content"],
            allowed_content_fields=["content"],
        )
