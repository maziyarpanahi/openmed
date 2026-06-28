"""Tests for JSONL chat-log PHI redaction."""

from __future__ import annotations

import io
import json
from pathlib import Path

from openmed.multimodal import ExtractedDocument, redact_document
from openmed.multimodal.chatlog_jsonl import (
    redact_chatlog_jsonl,
    write_redacted_chatlog_jsonl,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fake_redactor(text: str) -> str:
    replacements = {
        "John Doe": "[PERSON]",
        "555-0101": "[PHONE]",
        "john@example.com": "[EMAIL]",
    }
    redacted = text
    for source, replacement in replacements.items():
        redacted = redacted.replace(source, replacement)
    return redacted


def _load_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_one_turn_schema_redacts_content_and_configured_metadata_only():
    result = redact_chatlog_jsonl(
        FIXTURES / "synthetic_phi_chat.jsonl",
        nested_fields=("metadata.patient",),
        text_redactor=_fake_redactor,
    )

    rows = _load_jsonl(result.text)
    first = rows[0]

    assert first["role"] == "user"
    assert first["turn"] == 0
    assert first["timestamp"] == "2026-04-01T09:00:00Z"
    assert first["content"] == "I am [PERSON] and my phone is [PHONE]."
    assert first["metadata"]["patient"] == "[PERSON]"
    assert first["metadata"]["clinic"] == "North Wing"
    assert first["metadata"]["unconfigured_note"] == "John Doe should remain here"


def test_messages_list_schema_redacts_message_content_and_nested_string_leaves():
    result = redact_chatlog_jsonl(
        FIXTURES / "synthetic_phi_chat.jsonl",
        nested_fields=(
            "metadata.patient",
            "tool_call.arguments",
        ),
        text_redactor=_fake_redactor,
    )

    rows = _load_jsonl(result.text)
    conversation = rows[2]
    user_message = conversation["messages"][1]
    assistant_message = conversation["messages"][2]

    assert [message["role"] for message in conversation["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert conversation["timestamp"] == "2026-04-01T09:01:00Z"
    assert user_message["content"] == "My email is [EMAIL]."
    assert user_message["metadata"]["patient"] == "[PERSON]"
    assert assistant_message["content"] == "I removed [EMAIL] from the note."
    assert assistant_message["tool_call"]["arguments"] == {
        "patient": "[PERSON]",
        "note": "Call [PHONE]",
    }
    assert conversation["metadata"]["source"] == "synthetic"


def test_speaker_pseudonymization_is_consistent_within_file():
    result = redact_chatlog_jsonl(
        FIXTURES / "synthetic_phi_chat.jsonl",
        pseudonymize_speakers=True,
        text_redactor=_fake_redactor,
    )

    rows = _load_jsonl(result.text)
    patient_turn_speaker = rows[0]["speaker"]
    clinician_turn_speaker = rows[1]["speaker"]
    patient_message_speaker = rows[2]["messages"][1]["name"]
    clinician_message_speaker = rows[2]["messages"][2]["name"]

    assert patient_turn_speaker == patient_message_speaker
    assert clinician_turn_speaker == clinician_message_speaker
    assert patient_turn_speaker.startswith("USERNAME_")
    assert clinician_turn_speaker.startswith("USERNAME_")
    assert patient_turn_speaker != clinician_turn_speaker
    assert rows[0]["role"] == "user"
    assert rows[2]["messages"][1]["role"] == "user"


def test_write_redacted_chatlog_jsonl_streams_line_iterables_without_reading():
    class LineOnlySource:
        def __iter__(self):
            yield (
                '{"turn":0,"role":"user","content":"John Doe",'
                '"timestamp":"2026-04-01T09:00:00Z"}\n'
            )
            yield (
                '{"turn":1,"role":"assistant","content":"Call 555-0101",'
                '"timestamp":"2026-04-01T09:00:01Z"}\n'
            )

        def read(self):  # pragma: no cover - this must never be called
            raise AssertionError("streaming path should not call read()")

    output = io.StringIO()
    summary = write_redacted_chatlog_jsonl(
        LineOnlySource(),
        output,
        text_redactor=_fake_redactor,
    )

    rows = _load_jsonl(output.getvalue())
    assert summary.line_count == 2
    assert summary.message_count == 2
    assert rows[0]["content"] == "[PERSON]"
    assert rows[1]["content"] == "Call [PHONE]"


def test_redact_document_dispatches_jsonl_handler_with_summary_metadata():
    doc = redact_document(
        str(FIXTURES / "synthetic_phi_chat.jsonl"),
        policy={"nested_fields": ("metadata.patient",)},
        models={"text_redactor": _fake_redactor},
    )

    assert isinstance(doc, ExtractedDocument)
    assert doc.metadata["format"] == "jsonl_chatlog"
    assert doc.metadata["redaction_summary"]["line_count"] == 3
    assert "[PERSON]" in doc.text
