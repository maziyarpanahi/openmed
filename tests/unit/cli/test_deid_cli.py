"""Tests for the top-level ``openmed deid`` command."""

from __future__ import annotations

import io
import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.cli import main_module
from openmed.core import pii as pii_module


def test_deid_stdin_prints_deidentified_text_and_forwards_policy_method(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {}

    def fake_deidentify(text: str, **kwargs: Any) -> SimpleNamespace:
        calls["text"] = text
        calls.update(kwargs)
        return SimpleNamespace(deidentified_text="Patient [NAME]", pii_entities=[])

    monkeypatch.setattr(pii_module, "deidentify", fake_deidentify)
    monkeypatch.setattr(
        main_module.sys,
        "stdin",
        io.StringIO("Patient John Doe\n"),
    )

    result = main_module.main(
        ["deid", "--policy", "hipaa_safe_harbor", "--method", "replace"]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert captured.out == "Patient [NAME]\n"
    assert captured.err == ""
    assert calls["text"] == "Patient John Doe\n"
    assert calls["policy"] == "hipaa_safe_harbor"
    assert calls["method"] == "replace"
    assert calls["keep_mapping"] is False
    assert calls["audit"] is False


def test_deid_audit_writes_report_and_prints_path(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {}
    audit_path = tmp_path / "audit.json"

    def fake_deidentify(text: str, **kwargs: Any) -> SimpleNamespace:
        calls["text"] = text
        calls.update(kwargs)
        return SimpleNamespace(
            to_json=lambda: json.dumps(
                {"policy": kwargs["policy"], "input_hash": "safe-hash"},
                sort_keys=True,
            )
        )

    monkeypatch.setattr(pii_module, "deidentify", fake_deidentify)
    monkeypatch.setattr(main_module.sys, "stdin", io.StringIO("Patient John Doe"))

    result = main_module.main(
        ["deid", "--audit", "--output", str(audit_path), "--keep-mapping"]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert captured.out == f"{audit_path}\n"
    assert captured.err == ""
    assert json.loads(audit_path.read_text(encoding="utf-8")) == {
        "input_hash": "safe-hash",
        "policy": "hipaa_safe_harbor",
    }
    assert calls["audit"] is True
    assert calls["keep_mapping"] is True


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["deid", "--policy", "not_a_policy"], "unknown policy"),
        (["deid", "--method", "not_a_method"], "invalid choice"),
    ],
)
def test_deid_rejects_unknown_policy_or_method(
    argv: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main_module.main(argv)

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert message in captured.err


def test_deid_run_does_not_emit_raw_phi_to_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_phi = "Patient John Doe called 555-1234"

    def fake_deidentify(text: str, **_kwargs: Any) -> SimpleNamespace:
        assert text == raw_phi
        return SimpleNamespace(deidentified_text="Patient [NAME] called [PHONE]")

    monkeypatch.setattr(pii_module, "deidentify", fake_deidentify)
    monkeypatch.setattr(main_module.sys, "stdin", io.StringIO(raw_phi))

    with caplog.at_level(logging.INFO):
        result = main_module.main(["deid", "--method", "mask"])
    captured = capsys.readouterr()

    assert result == 0
    assert raw_phi not in caplog.text
    assert "John Doe" not in captured.out
    assert "555-1234" not in captured.out
    assert raw_phi not in captured.err
