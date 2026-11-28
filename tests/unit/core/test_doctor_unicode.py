"""Unicode database reporting in normal diagnostics and the public CLI."""

from __future__ import annotations

import json
import unicodedata

import pytest

from openmed.cli import main_module
from openmed.core import doctor

UNICODE_VERSION = "99.42.7"


@pytest.fixture(autouse=True)
def controlled_runtime(monkeypatch):
    monkeypatch.setattr(unicodedata, "unidata_version", UNICODE_VERSION)
    monkeypatch.setattr(doctor.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(doctor, "_effective_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(doctor, "OPTIONAL_EXTRAS", {})


def test_diagnostics_report_runtime_unicode_version():
    checks = [
        check
        for check in doctor.run_diagnostics()
        if check["name"] == "unicode_data_version"
    ]

    assert checks == [
        {"name": "unicode_data_version", "status": "PASS", "details": UNICODE_VERSION}
    ]


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("has_failure", [False, True])
def test_doctor_unicode_output_preserves_exit_behavior(
    monkeypatch, capsys, json_output, has_failure
):
    if has_failure:
        monkeypatch.setattr(doctor, "MIN_PYTHON_VERSION", (999, 0))

    arguments = ["doctor", "--json"] if json_output else ["doctor"]
    exit_code = main_module.main(arguments)
    captured = capsys.readouterr()

    assert exit_code == (1 if has_failure else 0)
    assert captured.err == ""
    if json_output:
        payload = json.loads(captured.out)
        assert set(payload) == {"ok", "command", "data", "outcome"}
        assert payload["ok"] is True
        assert payload["command"] == "doctor"
        assert payload["outcome"] == ("check_failed" if has_failure else "completed")
        assert set(payload["data"]) == {"checks", "has_failure"}
        assert payload["data"]["has_failure"] is has_failure
        checks = [
            check
            for check in payload["data"]["checks"]
            if check["name"] == "unicode_data_version"
        ]
        assert checks == [
            {
                "name": "unicode_data_version",
                "status": "PASS",
                "details": UNICODE_VERSION,
            }
        ]
    else:
        assert (
            f"PASS unicode_data_version: {UNICODE_VERSION}" in captured.out.splitlines()
        )


def test_value_free_health_report_excludes_unicode_version(monkeypatch):
    before = doctor.value_free_health_checks(environ={})
    monkeypatch.setattr(unicodedata, "unidata_version", "88.21.6")

    after = doctor.value_free_health_checks(environ={})

    assert after == before
    assert all(set(check) == {"name", "status", "code"} for check in after)
    assert "unicode_data_version" not in {check["name"] for check in after}
    assert UNICODE_VERSION not in json.dumps(after)
    assert "88.21.6" not in json.dumps(after)
