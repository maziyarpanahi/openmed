"""Focused tests for the offline session-end trace hook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.guard.session_hook import (
    SessionTraceError,
    main,
    scrub_trace,
)


def test_scrub_trace_is_deterministic_structural_and_value_free(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "completed-trace.json"
    source = {
        "trace_id": "trace-0001",
        "spans": [
            {
                "name": "synthetic-operation",
                "attributes": [
                    {
                        "key": "patient.email",
                        "value": {"stringValue": "synthetic.person@example.test"},
                    },
                    {
                        "key": "patient.phone",
                        "value": {"stringValue": "+1 555 010 2020"},
                    },
                ],
                "events": [
                    {
                        "name": "exception",
                        "attributes": [
                            {
                                "key": "exception.message",
                                "value": {"stringValue": "Patient Synthetic Person"},
                            }
                        ],
                    }
                ],
            }
        ],
    }
    original = json.dumps(source, indent=2).encode("utf-8")
    trace_path.write_bytes(original)

    result = scrub_trace(trace_path)
    scrubbed = json.loads(trace_path.read_text(encoding="utf-8"))

    assert result.changed is True
    assert result.format == "json"
    assert result.redaction_count >= 3
    assert scrubbed.keys() == source.keys()
    assert scrubbed["spans"][0]["attributes"][0]["value"] == {
        "stringValue": "[REDACTED:EMAIL]"
    }
    assert scrubbed["spans"][0]["attributes"][1]["value"] == {
        "stringValue": "[REDACTED:PHONE]"
    }
    output = trace_path.read_text(encoding="utf-8")
    assert "synthetic.person@example.test" not in output
    assert "+1 555 010 2020" not in output
    assert "Synthetic Person" not in output

    second = scrub_trace(trace_path)
    assert second.changed is False
    assert second.redaction_count == 0
    assert second.output_sha256 == result.output_sha256


def test_cli_success_is_quiet_and_json_summary_is_value_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps({"message": "Synthetic Person called +1 555 010 2020"}),
        encoding="utf-8",
    )

    assert main(["--trace", str(trace_path)]) == 0
    quiet = capsys.readouterr()
    assert quiet.out == ""
    assert quiet.err == ""

    assert main([str(trace_path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert "Synthetic Person" not in json.dumps(report)
    assert str(trace_path) not in json.dumps(report)


def test_malformed_trace_fails_without_replacing_input_or_echoing_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "trace.json"
    original = '{"message":"Synthetic Person",'
    trace_path.write_text(original, encoding="utf-8")

    assert main([str(trace_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Synthetic Person" not in captured.err
    assert trace_path.read_text(encoding="utf-8") == original


def test_replace_failure_is_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = tmp_path / "trace.json"
    original = json.dumps({"email": "synthetic.person@example.test"})
    trace_path.write_text(original, encoding="utf-8")

    def fail_replace(_source: str, _destination: Path) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("openmed.guard.session_hook.os.replace", fail_replace)

    with pytest.raises(SessionTraceError) as exc_info:
        scrub_trace(trace_path)

    assert exc_info.value.code == "write_failed"
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    assert str(exc_info.value) == "the scrubbed trace could not be committed"
    assert trace_path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".openmed-session-scrub-*.tmp"))


def test_jsonl_path_preserves_record_and_value_shapes(tmp_path: Path) -> None:
    trace_path = tmp_path / "completed.ndjson"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps({"event": "start", "sequence": 1}),
                json.dumps({"email": "synthetic.person@example.test"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = scrub_trace(trace_path)
    records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.format == "jsonl"
    assert len(records) == 2
    assert records[0] == {"event": "start", "sequence": 1}
    assert records[1] == {"email": "[REDACTED:EMAIL]"}
