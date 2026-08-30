"""Focused tests for role-message training redaction."""

from __future__ import annotations

import copy
import json

import pytest

from openmed.traces.schemas.chat import (
    ChatMessageRedactionError,
    ChatRedactionReport,
    ChatRedactionResult,
    ChatSchemaError,
    RoleMessageSchemaAdapter,
    redact_chat_messages,
    redact_chat_record,
    redact_chat_record_with_report,
)
from openmed.traces.schemas.registry import (
    AmbiguousSchemaError,
    TrainingSchemaRegistry,
)

REPLACEMENTS = {
    "synthetic person": "[PERSON]",
    "synthetic phone": "[PHONE]",
    "synthetic structured text": "[TEXT]",
    "synthetic tool result": "[TOOL_RESULT]",
    "synthetic value": "[VALUE]",
}


def _redact(text: str) -> str:
    return REPLACEMENTS.get(text, text)


def test_record_redaction_preserves_role_order_and_metadata_bytes() -> None:
    record = {
        "row_id": "synthetic-row-1",
        "messages": [
            {
                "role": "system",
                "content": "synthetic person",
                "metadata": {"split": "train", "source": "synthetic"},
            },
            {
                "role": "user",
                "content": "synthetic phone",
                "tool_calls": [
                    {
                        "id": "tool-synthetic-1",
                        "type": "function",
                        "function": {
                            "name": "synthetic_lookup",
                            "arguments": "synthetic protected argument",
                        },
                    }
                ],
            },
            {"role": "assistant", "content": None},
        ],
        "metadata": {
            "annotation": "synthetic auxiliary metadata",
            "order": ["first", "second"],
        },
    }
    original = copy.deepcopy(record)
    metadata_bytes = json.dumps(
        record["metadata"], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")

    redacted = redact_chat_record(record, text_redactor=_redact)

    assert record == original
    assert [message["role"] for message in redacted["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert [message["content"] for message in redacted["messages"]] == [
        "[PERSON]",
        "[PHONE]",
        None,
    ]
    assert redacted["messages"][1]["tool_calls"][0]["id"] == "tool-synthetic-1"
    assert (
        redacted["messages"][1]["tool_calls"][0]["function"]["arguments"]
        == "synthetic protected argument"
    )
    assert (
        json.dumps(
            redacted["metadata"], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        == metadata_bytes
    )


def test_structured_content_redacts_text_parts_only() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "synthetic structured text",
                    "metadata": {"note": "synthetic untouched note"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "synthetic://asset"},
                },
                {
                    "type": "tool_use",
                    "id": "tool-synthetic-2",
                    "name": "synthetic_lookup",
                    "input": {"query": "synthetic protected argument"},
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-synthetic-2",
                    "content": [{"type": "text", "text": "synthetic tool result"}],
                },
                {"type": "custom", "payload": {"text": "synthetic value"}},
                {"type": "text", "value": "synthetic value"},
            ],
        }
    ]
    original = copy.deepcopy(messages)

    redacted = redact_chat_messages(messages, redactor=_redact)

    assert messages == original
    assert [part["type"] for part in redacted[0]["content"]] == [
        "text",
        "image_url",
        "tool_use",
        "tool_result",
        "custom",
        "text",
    ]
    assert redacted[0]["content"][0]["text"] == "[TEXT]"
    assert redacted[0]["content"][0]["metadata"] == {"note": "synthetic untouched note"}
    assert redacted[0]["content"][1]["image_url"]["url"] == "synthetic://asset"
    assert redacted[0]["content"][2]["id"] == "tool-synthetic-2"
    assert redacted[0]["content"][2]["input"] == {
        "query": "synthetic protected argument"
    }
    assert redacted[0]["content"][3]["content"][0]["text"] == "[TOOL_RESULT]"
    assert redacted[0]["content"][4]["payload"]["text"] == "synthetic value"
    assert redacted[0]["content"][5]["value"] == "[VALUE]"


def test_unknown_part_containers_remain_untouched() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "custom",
                    "text": "synthetic value",
                    "parts": [{"text": "synthetic value"}],
                }
            ],
        }
    ]

    redacted = redact_chat_messages(messages, text_redactor=_redact)

    assert redacted == messages


def test_adapter_walk_and_reconstruct_use_stable_content_paths() -> None:
    record = {
        "messages": [
            {"role": "user", "content": "synthetic person"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "synthetic value"}],
            },
        ],
        "metadata": {"split": "synthetic"},
    }
    adapter = RoleMessageSchemaAdapter()

    assert adapter.name == "role_messages"
    assert adapter.detect(record) is True
    assert adapter.walk(record) == (
        (("messages", 0, "content"), "synthetic person"),
        (("messages", 1, "content", 0, "text"), "synthetic value"),
    )

    rebuilt = adapter.reconstruct(
        record,
        {
            path: f"[REDACTED-{index}]"
            for index, (path, _text) in enumerate(adapter.walk(record))
        },
    )

    assert rebuilt["messages"][0]["content"] == "[REDACTED-0]"
    assert rebuilt["messages"][1]["content"][0]["text"] == "[REDACTED-1]"
    assert rebuilt["metadata"] == {"split": "synthetic"}
    assert record["messages"][0]["content"] == "synthetic person"


def test_reconstruct_rejects_paths_that_collide_after_normalization() -> None:
    record = {"messages": [{"role": "user", "content": "synthetic person"}]}
    adapter = RoleMessageSchemaAdapter()

    with pytest.raises(ChatSchemaError, match="must be unique"):
        adapter.reconstruct(
            record,
            {
                ("messages", 0, "content"): "[REDACTED-A]",
                "messages.0.content": "[REDACTED-B]",
            },
        )

    assert record["messages"][0]["content"] == "synthetic person"


def test_report_and_result_are_value_free_and_deterministic() -> None:
    record = {
        "messages": [
            {"role": "user", "content": "synthetic person"},
            {"role": "assistant", "content": "synthetic value"},
        ],
        "metadata": {"source": "synthetic"},
    }

    first = redact_chat_record_with_report(record, text_redactor=_redact)
    second = redact_chat_record_with_report(record, text_redactor=_redact)

    assert isinstance(first, ChatRedactionResult)
    assert first == second
    assert first.report.message_count == 2
    assert first.report.text_value_count == 2
    assert first.report.redacted_text_count == 2
    assert first.report.structured_part_count == 0
    serialized_report = json.dumps(first.report.to_dict(), sort_keys=True)
    assert "synthetic" not in serialized_report
    assert "[PERSON]" not in serialized_report


def test_custom_schema_keys_are_hashed_in_report_paths() -> None:
    sensitive_key = "PatientJaneDoe"
    result = redact_chat_record_with_report(
        {sensitive_key: [{"content": "synthetic value"}]},
        text_redactor=_redact,
        messages_key=sensitive_key,
    )

    serialized_report = json.dumps(result.report.to_dict(), sort_keys=True)
    assert sensitive_key not in serialized_report
    assert "key_sha256_" in serialized_report


def test_direct_report_construction_sanitizes_content_paths() -> None:
    sensitive_key = "PatientJaneDoe"
    report = ChatRedactionReport(content_paths=(f"$.{sensitive_key}",))

    serialized_report = json.dumps(report.to_dict(), sort_keys=True)
    assert sensitive_key not in serialized_report
    assert report.content_paths[0].startswith("$.key_sha256_")


def test_cyclic_structured_content_fails_closed() -> None:
    content: list[object] = []
    content.append({"type": "text", "items": content})

    with pytest.raises(ChatSchemaError, match="cyclic"):
        redact_chat_messages(
            [{"role": "user", "content": content}],
            text_redactor=_redact,
        )


def test_redactor_failures_do_not_expose_source_text() -> None:
    source_text = "synthetic private phrase"

    def failing_redactor(_text: str) -> str:
        raise RuntimeError(source_text)

    with pytest.raises(ChatMessageRedactionError) as error:
        redact_chat_messages(
            [{"role": "user", "content": source_text}],
            text_redactor=failing_redactor,
        )

    assert source_text not in str(error.value)
    assert "text redactor failed" in str(error.value)


def test_text_subclass_hooks_are_not_used_at_adapter_boundaries() -> None:
    sensitive = "synthetic private string hook"

    class HostileText(str):
        def __hash__(self) -> int:
            raise RuntimeError(sensitive)

        def __eq__(self, other: object) -> bool:
            del other
            raise RuntimeError(sensitive)

        def strip(self, chars: str | None = None) -> str:
            del chars
            raise RuntimeError(sensitive)

    seen_types: list[type[object]] = []

    def redact(value: str) -> str:
        seen_types.append(type(value))
        return HostileText("[REDACTED]")

    adapter = RoleMessageSchemaAdapter(
        messages_key=HostileText("messages"),
        content_key=HostileText("content"),
    )
    rebuilt = adapter.transform(
        {"messages": [{"role": "user", "content": HostileText("private")}]},
        redact,
    )

    replacement = rebuilt["messages"][0]["content"]
    assert seen_types == [str]
    assert type(replacement) is str
    assert str.encode(replacement, "utf-8") == b"[REDACTED]"


def test_message_iterator_failures_are_value_free() -> None:
    sensitive = "synthetic private message iterator"

    class FailingMessages(list[object]):
        def __iter__(self):
            yield {"role": "user", "content": "synthetic value"}
            raise RuntimeError(sensitive)

    with pytest.raises(ChatSchemaError) as error:
        redact_chat_messages(FailingMessages(), text_redactor=_redact)

    assert sensitive not in str(error.value)


def test_replacement_iterator_failures_are_value_free() -> None:
    sensitive = "synthetic private replacement iterator"

    class FailingReplacements(dict[object, object]):
        def items(self):
            yield (("messages", 0, "content"), "[REDACTED]")
            raise RuntimeError(sensitive)

    adapter = RoleMessageSchemaAdapter()
    record = {"messages": [{"role": "user", "content": "synthetic value"}]}
    with pytest.raises(ChatSchemaError) as error:
        adapter.reconstruct(record, FailingReplacements())

    assert sensitive not in str(error.value)


def test_role_message_adapter_registers_without_alias_collision() -> None:
    record = {"messages": [{"role": "user", "content": "synthetic value"}]}
    adapter = RoleMessageSchemaAdapter()
    registry = TrainingSchemaRegistry()

    registry.register(adapter)

    assert registry.get("role_messages") is adapter
    assert registry.walk(record, schema="role_messages") == (
        (("messages", 0, "content"), "synthetic value"),
    )
    with pytest.raises(AmbiguousSchemaError):
        registry.resolve(record)
