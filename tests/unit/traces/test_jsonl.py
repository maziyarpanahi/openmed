"""Tests for the schema-preserving JSONL trace content walker."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from openmed.traces.jsonl import (
    TraceContentWalker,
    TraceJSONLIOError,
    TraceJSONLLineError,
    TraceJSONLTransformError,
    rewrite_trace_jsonl,
    walk_trace_content,
    write_trace_jsonl,
)


def _line(record: object, ending: str = "\n") -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + ending


def test_walks_only_configured_nested_string_locations() -> None:
    source = _line(
        {
            "trace_id": "trace-001",
            "timestamp": "2026-08-08T10:00:00Z",
            "events": [
                {
                    "role": "user",
                    "content": "synthetic note for the first event",
                    "event_id": "event-001",
                },
                {
                    "role": "assistant",
                    "content": "synthetic response",
                    "metadata": {"content": "synthetic metadata content"},
                },
            ],
            "labels": ["user", "replay"],
        }
    )

    locations = list(
        walk_trace_content(
            source,
            content_paths=("events.*.content", "events.*.metadata.content"),
        )
    )

    assert [(item.line_number, item.path, item.value) for item in locations] == [
        (1, ("events", 0, "content"), "synthetic note for the first event"),
        (1, ("events", 1, "content"), "synthetic response"),
        (1, ("events", 1, "metadata", "content"), "synthetic metadata content"),
    ]
    assert locations[0].json_path == "$.events[0].content"
    assert locations[0].field_path == locations[0].path
    assert locations[0].as_tuple() == (
        1,
        ("events", 0, "content"),
        "synthetic note for the first event",
    )


def test_default_path_selects_content_at_any_depth_but_not_identifiers() -> None:
    source = _line(
        {
            "trace_id": "trace-002",
            "event": {
                "role": "user",
                "content": "synthetic nested content",
                "tool": {"name": "lookup", "content": "synthetic tool content"},
            },
            "identifier": "synthetic identifier",
        }
    )

    locations = list(TraceContentWalker().walk(source))

    assert [location.path for location in locations] == [
        ("event", "content"),
        ("event", "tool", "content"),
    ]
    assert all(location.value != "synthetic identifier" for location in locations)


def test_rewrite_preserves_order_scalar_types_and_untouched_fields() -> None:
    record = {
        "trace_id": "trace-003",
        "sequence": 7,
        "timestamp": "2026-08-08T10:02:00Z",
        "event": {
            "role": "user",
            "content": "synthetic patient note",
            "count": 2,
            "enabled": True,
            "optional": None,
        },
        "events": [
            {"event_id": "event-002", "content": "synthetic tool result"},
            {"event_id": "event-003", "content": 42},
        ],
        "untouched": {"message": "synthetic unconfigured text"},
    }
    source = _line(record, ending="\r\n")

    output = "".join(
        rewrite_trace_jsonl(
            source,
            lambda value: f"[REDACTED:{len(value)}]",
            content_paths=("event.content", "events.*.content"),
        )
    )
    rewritten = json.loads(output)

    assert output.endswith("\r\n")
    assert list(rewritten) == list(record)
    assert list(rewritten["event"]) == list(record["event"])
    assert [event["event_id"] for event in rewritten["events"]] == [
        "event-002",
        "event-003",
    ]
    assert rewritten["trace_id"] == record["trace_id"]
    assert rewritten["sequence"] == 7
    assert isinstance(rewritten["sequence"], int)
    assert rewritten["timestamp"] == record["timestamp"]
    assert rewritten["event"]["role"] == "user"
    assert rewritten["event"]["count"] == 2
    assert isinstance(rewritten["event"]["count"], int)
    assert rewritten["event"]["enabled"] is True
    assert rewritten["event"]["optional"] is None
    assert rewritten["event"]["content"] == "[REDACTED:22]"
    assert rewritten["events"][0]["content"] == "[REDACTED:21]"
    assert rewritten["events"][1]["content"] == 42
    assert rewritten["untouched"] == record["untouched"]
    assert "synthetic patient note" not in output
    assert "synthetic tool result" not in output
    assert "synthetic unconfigured text" in output


def test_rewrite_streams_blank_lines_and_writes_to_text_output() -> None:
    source = io.StringIO(
        _line({"content": "synthetic content"}) + "\n" + _line({"id": "trace-004"})
    )
    destination = io.StringIO()

    count = write_trace_jsonl(
        source,
        destination,
        lambda value: "[REDACTED]",
    )

    assert count == 3
    assert destination.getvalue().splitlines() == [
        '{"content":"[REDACTED]"}',
        "",
        '{"id":"trace-004"}',
    ]


def test_write_rejects_same_input_output_without_truncating(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    original = _line({"content": "synthetic content"})
    trace_path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="must be different"):
        write_trace_jsonl(trace_path, trace_path, lambda _value: "[REDACTED]")

    assert trace_path.read_text(encoding="utf-8") == original


def test_tuple_path_with_array_index_is_treated_as_one_path() -> None:
    source = _line({"events": [{"content": "first"}, {"content": "second"}]})

    locations = list(walk_trace_content(source, ("events", 1, "content")))

    assert [(item.path, item.value) for item in locations] == [
        (("events", 1, "content"), "second")
    ]


def test_malformed_json_reports_line_without_raw_values() -> None:
    source = (
        _line({"content": "synthetic valid content"})
        + '{"content":"synthetic malformed content"\n'
    )
    locations = walk_trace_content(source)

    first = next(locations)
    assert first.value == "synthetic valid content"

    with pytest.raises(TraceJSONLLineError) as exc_info:
        next(locations)

    assert exc_info.value.line_number == 2
    assert "malformed JSON" in exc_info.value.message
    assert "synthetic malformed content" not in str(exc_info.value)


def test_non_object_and_duplicate_key_records_are_rejected() -> None:
    for source in ('["synthetic list record"]\n', '{"content":1,"content":2}\n'):
        with pytest.raises(TraceJSONLLineError) as exc_info:
            next(walk_trace_content(source))

        assert exc_info.value.line_number == 1
        assert "synthetic list record" not in str(exc_info.value)


def test_transform_failures_are_value_free_and_include_line_number() -> None:
    def fail(value: str) -> str:
        raise RuntimeError(value)

    with pytest.raises(TraceJSONLTransformError) as exc_info:
        next(rewrite_trace_jsonl(_line({"content": "synthetic secret"}), fail))

    assert exc_info.value.line_number == 1
    assert exc_info.value.path == ("content",)
    assert exc_info.value.json_path == "$.content"
    assert "synthetic secret" not in str(exc_info.value)


def test_transform_error_hashes_caller_controlled_path_keys() -> None:
    sensitive_key = "PatientJaneDoe"

    def fail(value: str) -> str:
        raise RuntimeError(value)

    with pytest.raises(TraceJSONLTransformError) as exc_info:
        next(
            rewrite_trace_jsonl(
                _line({sensitive_key: "synthetic secret"}),
                fail,
                content_paths=[sensitive_key],
            )
        )

    assert sensitive_key not in str(exc_info.value)
    assert sensitive_key not in exc_info.value.path
    assert exc_info.value.json_path.startswith("$.key_sha256_")


def test_walker_does_not_read_until_consumed() -> None:
    consumed = False

    def source():
        nonlocal consumed
        consumed = True
        yield _line({"content": "synthetic lazy content"})

    locations = walk_trace_content(source())
    assert consumed is False
    assert next(locations).value == "synthetic lazy content"
    assert consumed is True


def test_source_and_destination_failures_are_value_free() -> None:
    sensitive = "PatientJaneDoe"

    class FailingPath(os.PathLike[str]):
        def __fspath__(self) -> str:
            raise RuntimeError(sensitive)

    with pytest.raises(TraceJSONLIOError) as source_error:
        next(walk_trace_content(FailingPath()))
    assert sensitive not in str(source_error.value)

    class FailingOutput:
        def write(self, _line: str) -> None:
            raise RuntimeError(sensitive)

    with pytest.raises(TraceJSONLIOError) as destination_error:
        write_trace_jsonl(
            _line({"content": "synthetic content"}),
            FailingOutput(),
            lambda _value: "[REDACTED]",
        )
    assert sensitive not in str(destination_error.value)
