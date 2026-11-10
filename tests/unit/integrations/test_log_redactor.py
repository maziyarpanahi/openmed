from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any

from openmed.integrations import log_redactor


def _fake_batch_result(texts: list[str]) -> SimpleNamespace:
    items = [
        SimpleNamespace(
            success=True,
            result=SimpleNamespace(deidentified_text=_fake_redact_text(text)),
        )
        for text in texts
    ]
    return SimpleNamespace(items=items)


def _fake_redact_text(text: str) -> str:
    return (
        text.replace("Jane Roe", "[NAME]")
        .replace("John Doe", "[NAME]")
        .replace("555-0100", "[PHONE]")
        .replace("jane.roe@example.org", "[EMAIL]")
    )


def _jsonl(events: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(event) + "\n" for event in events)


def test_ndjson_stream_redacts_configured_message_fields_and_preserves_metadata(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_process_batch(texts: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append({"texts": list(texts), "kwargs": dict(kwargs)})
        return _fake_batch_result(list(texts))

    monkeypatch.setattr(log_redactor, "process_batch", fake_process_batch)
    events = [
        {
            "sequence": 1,
            "service": "api",
            "message": "Patient Jane Roe called 555-0100",
            "metadata": {"trace_id": "abc", "severity": "warning"},
        },
        {
            "sequence": 2,
            "service": "worker",
            "error": {"message": "Escalated John Doe to triage"},
            "metadata": {"trace_id": "def", "severity": "error"},
        },
        {
            "sequence": 3,
            "service": "api",
            "message": "Heartbeat from scheduler",
            "metadata": {"trace_id": "ghi", "severity": "info"},
        },
    ]
    input_stream = io.StringIO(_jsonl(events))
    output_stream = io.StringIO()

    emitted = log_redactor.redact_ndjson_stream(
        input_stream,
        output_stream,
        message_fields=("message", "error.message"),
        batch_size=2,
        model_name="pii-model",
        method="mask",
        use_safety_sweep=False,
    )

    rows = [json.loads(line) for line in output_stream.getvalue().splitlines()]
    assert emitted == 3
    assert [row["sequence"] for row in rows] == [1, 2, 3]
    assert [row["metadata"] for row in rows] == [event["metadata"] for event in events]
    assert rows[0]["message"] == "Patient [NAME] called [PHONE]"
    assert rows[1]["error"]["message"] == "Escalated [NAME] to triage"
    assert rows[2]["message"] == "Heartbeat from scheduler"
    assert "Jane Roe" not in output_stream.getvalue()
    assert "John Doe" not in output_stream.getvalue()
    assert "555-0100" not in output_stream.getvalue()
    assert [call["texts"] for call in calls] == [
        ["Patient Jane Roe called 555-0100", "Escalated John Doe to triage"],
        ["Heartbeat from scheduler"],
    ]
    assert calls[0]["kwargs"]["operation"] == "deidentify"
    assert calls[0]["kwargs"]["batch_size"] == 2
    assert calls[0]["kwargs"]["continue_on_error"] is False
    assert calls[0]["kwargs"]["use_safety_sweep"] is False


def test_embeddable_callable_batches_events_and_preserves_order(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_process_batch(texts: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append(list(texts))
        return _fake_batch_result(list(texts))

    monkeypatch.setattr(log_redactor, "process_batch", fake_process_batch)
    events = [
        {"event_id": f"evt-{index}", "message": f"Patient Jane Roe #{index}"}
        for index in range(5)
    ]

    rows = list(
        log_redactor.redact_log_events(
            events,
            message_fields=("message",),
            batch_size=2,
            use_safety_sweep=False,
        )
    )

    assert [row["event_id"] for row in rows] == [
        "evt-0",
        "evt-1",
        "evt-2",
        "evt-3",
        "evt-4",
    ]
    assert all("Jane Roe" not in row["message"] for row in rows)
    assert calls == [
        ["Patient Jane Roe #0", "Patient Jane Roe #1"],
        ["Patient Jane Roe #2", "Patient Jane Roe #3"],
        ["Patient Jane Roe #4"],
    ]


def test_cli_diagnostics_do_not_echo_malformed_raw_phi() -> None:
    stdin = io.StringIO('{"message":"Patient Jane Roe called"\n')
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = log_redactor.main(
        ["--field", "message"],
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "input line 1" in stderr.getvalue()
    assert "Jane Roe" not in stderr.getvalue()


def test_cli_diagnostics_do_not_echo_batch_error_phi(monkeypatch) -> None:
    def fake_process_batch(texts: list[str], **kwargs: Any) -> SimpleNamespace:
        raise RuntimeError(f"model failed on {texts[0]}")

    monkeypatch.setattr(log_redactor, "process_batch", fake_process_batch)
    stdin = io.StringIO(json.dumps({"message": "Patient Jane Roe called"}) + "\n")
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = log_redactor.main(
        ["--field", "message"],
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert "failed to redact a log event batch" in stderr.getvalue()
    assert "Jane Roe" not in stderr.getvalue()
