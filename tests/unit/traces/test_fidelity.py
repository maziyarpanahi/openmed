"""Focused tests for deterministic, value-free trace fidelity checks."""

from __future__ import annotations

import copy
import json

import pytest

from openmed.traces.fidelity import (
    TraceFidelityError,
    TraceFidelityIssue,
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


def test_content_only_message_reordering_is_detected() -> None:
    original = {
        "messages": [
            {"role": "user", "content": "SYNTHETIC_FIRST"},
            {"role": "user", "content": "SYNTHETIC_SECOND"},
        ]
    }
    output = {
        "messages": [
            {"role": "user", "content": "SYNTHETIC_SECOND"},
            {"role": "user", "content": "SYNTHETIC_FIRST"},
        ]
    }

    report = verify_trace_fidelity(original, output)

    assert report.has_code("message_order")
    assert report.failing_paths == ("$.messages",)


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


@pytest.mark.parametrize(
    ("field", "original_value", "replacement", "code"),
    [
        ("ids", ["synthetic-id-1"], ["synthetic-id-2"], "identifier"),
        ("call_ids", ["call-1"], ["call-2"], "call_linkage"),
        ("timestamp_ms", 1_700_000_000_000, 1_700_000_000_001, "timestamp"),
        ("score", 1.0, 0.0, "training_label"),
        ("role", "user", "assistant", "structure"),
    ],
)
def test_broad_content_paths_cannot_mask_semantic_fields(
    field: str,
    original_value: object,
    replacement: object,
    code: str,
) -> None:
    original = {"content": {field: original_value}}
    output = {"content": {field: replacement}}

    report = verify_trace_fidelity(original, output)

    assert report.passed is False
    assert report.has_code(code)


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


def test_segment_tuple_is_one_narrow_path_not_multiple_broad_allow_rules() -> None:
    original = {"payload": {"prompt": "SYNTHETIC_INPUT", "role": "user"}}
    output = {"payload": {"prompt": "SYNTHETIC_OUTPUT", "role": "assistant"}}

    report = verify_trace_fidelity(
        original,
        output,
        content_paths=("payload", "prompt"),
    )

    assert report.passed is False
    assert report.failing_paths == ("$.payload.role",)
    assert report.allowed_content_paths == ("$.payload.prompt",)


def test_content_path_strings_are_not_whitespace_coerced() -> None:
    original = {"prompt": "SYNTHETIC_INPUT"}
    output = {"prompt": "SYNTHETIC_OUTPUT"}

    report = verify_trace_fidelity(
        original,
        output,
        content_paths=" prompt ",
    )

    assert report.passed is False
    assert report.failing_paths == ("$.prompt",)


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


def test_caller_controlled_keys_are_hashed_in_reports() -> None:
    sensitive = "PatientJaneDoe"
    identifier_key = f"{sensitive}_id"
    original = {identifier_key: "synthetic-id-1"}
    output = {identifier_key: "synthetic-id-2"}

    report = verify_trace_fidelity(original, output, content_paths=[])

    assert report.has_code("identifier")
    assert sensitive not in report.to_json()
    assert sensitive not in report.summary()
    assert "key_sha256_" in report.failing_paths[0]


def test_caller_controlled_allowed_paths_are_hashed() -> None:
    sensitive = "PatientJaneDoe"
    original = {sensitive: "before"}
    output = {sensitive: "after"}

    report = verify_trace_fidelity(original, output, content_paths=[sensitive])

    assert report.passed
    assert sensitive not in report.to_json()
    assert report.allowed_content_paths[0].startswith("$.key_sha256_")


def test_content_path_iteration_errors_are_value_free() -> None:
    sensitive = "PatientJaneDoe"

    def failing_paths():
        raise RuntimeError(sensitive)
        yield "content"

    with pytest.raises(ValueError) as caught:
        verify_trace_fidelity({}, {}, content_paths=failing_paths())

    assert sensitive not in str(caught.value)


def test_content_path_string_subclass_hooks_are_not_used() -> None:
    sensitive = "PatientJaneDoe"

    class FailingPath(str):
        def strip(self, *args: object) -> str:
            raise RuntimeError(sensitive)

    report = verify_trace_fidelity(
        {"content": "SYNTHETIC_INPUT"},
        {"content": "SYNTHETIC_OUTPUT"},
        content_paths=FailingPath("content"),
    )

    assert report.passed
    assert sensitive not in report.to_json()


def test_direct_issue_path_errors_fail_closed() -> None:
    sensitive = "PatientJaneDoe"

    class FailingPath:
        def __eq__(self, other: object) -> bool:
            raise RuntimeError(sensitive)

    issue = TraceFidelityIssue(
        path=FailingPath(),  # type: ignore[arg-type]
        code="structure",
    )

    assert issue.path == "$"


def test_direct_issue_string_subclass_hooks_are_not_used() -> None:
    sensitive = "PatientJaneDoe"

    class HostileString(str):
        def __hash__(self) -> int:
            raise RuntimeError(sensitive)

        def strip(self, *args: object) -> str:
            del args
            raise RuntimeError(sensitive)

    issue = TraceFidelityIssue(
        path=HostileString("$.messages"),
        code=HostileString("identifier"),
        expected_type=HostileString("string"),
        actual_type=HostileString("string"),
    )

    assert issue.path == "$.messages"
    assert issue.code == "identifier"
    assert issue.expected_type == "string"


def test_cyclic_inputs_fail_closed_without_recursion_errors() -> None:
    original: dict[str, object] = {"messages": []}
    output: dict[str, object] = {"messages": []}
    original["messages"].append(original)
    output["messages"].append(output)

    report = verify_trace_fidelity(original, output)

    assert report.passed is False
    assert report.has_code("structure")


def test_direct_report_construction_sanitizes_all_display_metadata() -> None:
    sensitive = "PatientJaneDoe"
    issue = TraceFidelityIssue(
        path=f"$.{sensitive}",
        code=sensitive,
        expected_type=sensitive,
        actual_type=sensitive,
    )
    report = TraceFidelityReport(
        passed=True,
        issues=(issue,),
        allowed_content_paths=(f"$.{sensitive}",),
    )

    serialized = report.to_json()
    assert sensitive not in serialized
    assert report.passed is False
    assert report.issues[0].code == "structure"
    assert report.issues[0].expected_type == "other"
    assert report.allowed_content_paths[0].startswith("$.key_sha256_")
