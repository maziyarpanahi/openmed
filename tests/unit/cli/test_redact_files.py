"""Focused tests for the offline ``openmed redact-files`` command."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.core.pii import DeidentificationResult, PIIEntity

SYNTHETIC_NAME = "Synthetic Rowan"
SYNTHETIC_PHONE = "555-0107"


@pytest.mark.parametrize("method", ["aadhaar_mask", "format_preserve"])
def test_current_deidentification_methods_are_accepted(method: str) -> None:
    args = main_module.build_parser().parse_args(
        ["redact-files", "input.txt", "output.txt", "--method", method]
    )

    assert args.method == method


def test_text_file_redaction_reports_phi_free_offsets_and_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    input_path = tmp_path / "support-export.txt"
    output_path = tmp_path / "support-export.redacted.txt"
    report_path = tmp_path / "support-export.report.json"
    input_text = f"Caller {SYNTHETIC_NAME} used {SYNTHETIC_PHONE}.\n"
    input_path.write_text(input_text, encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_deidentify(text: str, **kwargs: object) -> DeidentificationResult:
        calls.append(dict(kwargs))
        return _fake_deidentify(text, **kwargs)

    monkeypatch.setattr("openmed.core.pii.deidentify", fake_deidentify)

    exit_code = main_module.main(
        [
            "redact-files",
            str(input_path),
            str(output_path),
            "--policy",
            "strict_no_leak",
            "--lang",
            "en",
            "--method",
            "replace",
            "--seed",
            "42",
            "--no-safety-sweep",
            "--report",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert output_path.read_text(encoding="utf-8") == ("Caller [NAME] used [PHONE].\n")
    assert payload["format"] == "text"
    assert payload["documents"] == 1
    assert payload["redacted_documents"] == 1
    assert payload["total_spans"] == 2
    assert payload["per_label_counts"] == {"NAME": 1, "PHONE": 1}
    assert [item["start"] for item in payload["offsets"]] == [7, 28]
    assert all(SYNTHETIC_NAME not in json.dumps(item) for item in payload["offsets"])
    assert all(SYNTHETIC_PHONE not in json.dumps(item) for item in payload["offsets"])
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload
    assert SYNTHETIC_NAME not in report_path.read_text(encoding="utf-8")
    assert SYNTHETIC_PHONE not in report_path.read_text(encoding="utf-8")
    assert calls[0]["policy"] == "strict_no_leak"
    assert calls[0]["lang"] == "en"
    assert calls[0]["method"] == "replace"
    assert calls[0]["seed"] == 42
    assert calls[0]["consistent"] is True
    assert calls[0]["use_safety_sweep"] is False


def test_jsonl_file_redaction_preserves_line_endings_and_line_offsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    input_path = tmp_path / "events.ndjson"
    output_path = tmp_path / "events.redacted.ndjson"
    input_path.write_bytes(
        (f"event one {SYNTHETIC_NAME}\r\nevent two {SYNTHETIC_PHONE}\n\n").encode(
            "utf-8"
        )
    )
    monkeypatch.setattr("openmed.core.pii.deidentify", _fake_deidentify)

    exit_code = main_module.main(
        [
            "redact-file",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--format",
            "jsonl",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    data = envelope["data"]
    assert exit_code == 0
    assert envelope["ok"] is True
    assert envelope["command"] == "redact-files"
    assert output_path.read_bytes() == (b"event one [NAME]\r\nevent two [PHONE]\n\n")
    assert data["format"] == "jsonl"
    assert data["documents"] == 2
    assert data["redacted_documents"] == 2
    assert data["total_spans"] == 2
    assert [item["line"] for item in data["offsets"]] == [1, 2]
    assert SYNTHETIC_NAME not in captured.out
    assert SYNTHETIC_PHONE not in captured.out


def test_file_redaction_errors_do_not_echo_raw_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    input_path.write_text(SYNTHETIC_NAME, encoding="utf-8")

    def failing_deidentify(text: str, **_kwargs: object) -> None:
        raise RuntimeError(f"model failed on {text}")

    monkeypatch.setattr("openmed.core.pii.deidentify", failing_deidentify)

    exit_code = main_module.main(
        [
            "redact-files",
            str(input_path),
            str(output_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"]["code"] == "redaction_failed"
    assert SYNTHETIC_NAME not in captured.out
    assert captured.err == ""


def _fake_deidentify(text: str, **kwargs: object) -> DeidentificationResult:
    replacements = {
        SYNTHETIC_NAME: ("[NAME]", "NAME"),
        SYNTHETIC_PHONE: ("[PHONE]", "PHONE"),
    }
    redacted = text
    entities: list[PIIEntity] = []
    for source, (replacement, label) in replacements.items():
        start = text.find(source)
        if start == -1:
            continue
        entities.append(
            PIIEntity(
                text=source,
                label=label,
                confidence=0.99,
                start=start,
                end=start + len(source),
                entity_type=label,
                original_text=source,
                redacted_text=replacement,
            )
        )
        redacted = redacted.replace(source, replacement)

    return DeidentificationResult(
        original_text=text,
        deidentified_text=redacted,
        pii_entities=entities,
        method=str(kwargs.get("method", "mask")),
        timestamp=datetime(2026, 1, 1),
    )
