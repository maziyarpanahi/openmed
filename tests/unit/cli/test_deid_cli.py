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


def test_deid_forwards_explicit_date_shift_days(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {}

    def fake_deidentify(text: str, **kwargs: Any) -> SimpleNamespace:
        calls["text"] = text
        calls.update(kwargs)
        return SimpleNamespace(deidentified_text="Visit on 2024-06-30", pii_entities=[])

    monkeypatch.setattr(pii_module, "deidentify", fake_deidentify)
    monkeypatch.setattr(main_module.sys, "stdin", io.StringIO("Visit on 2024-01-02"))

    result = main_module.main(
        ["deid", "--method", "shift_dates", "--date-shift-days", "180"]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ""
    assert calls["method"] == "shift_dates"
    assert calls["date_shift_days"] == 180


@pytest.mark.parametrize(
    ("argv", "expected_method", "expected_shift_dates", "expected_date_shift_days"),
    [
        (["pii", "deidentify", "--method", "shift_dates"], "shift_dates", None, None),
        (["pii", "deidentify", "--shift-dates"], "mask", True, None),
        (
            [
                "pii",
                "deidentify",
                "--method",
                "shift_dates",
                "--date-shift-days",
                "180",
            ],
            "shift_dates",
            None,
            180,
        ),
    ],
)
def test_pii_deidentify_forwards_date_shift_options(
    argv: list[str],
    expected_method: str,
    expected_shift_dates: bool | None,
    expected_date_shift_days: int | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {}

    def fake_deidentify(text: str, **kwargs: Any) -> SimpleNamespace:
        pii_module._resolve_deidentification_method(
            kwargs["method"],
            kwargs["shift_dates"],
            kwargs["date_shift_days"],
        )
        calls["text"] = text
        calls.update(kwargs)
        return SimpleNamespace(deidentified_text="Visit on 2024-06-30", pii_entities=[])

    monkeypatch.setattr(pii_module, "deidentify", fake_deidentify)

    result = main_module.main([*argv, "--text", "Visit on 2024-01-02"])
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == "\n[Redacted 0 entities]\n"
    assert calls["method"] == expected_method
    assert calls["shift_dates"] is expected_shift_dates
    assert calls["date_shift_days"] == expected_date_shift_days


def test_pii_deidentify_prints_date_shift_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_deidentify(text: str, **kwargs: Any) -> SimpleNamespace:
        raise ValueError("date_shift_days requires method='shift_dates'")

    monkeypatch.setattr(pii_module, "deidentify", fake_deidentify)

    result = main_module.main(
        [
            "pii",
            "deidentify",
            "--text",
            "Visit on 2024-01-02",
            "--date-shift-days",
            "180",
        ]
    )
    captured = capsys.readouterr()

    assert result == 2
    assert captured.out == ""
    assert captured.err == "date_shift_days requires method='shift_dates'\n"


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
