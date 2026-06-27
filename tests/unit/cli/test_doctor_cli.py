from __future__ import annotations

import argparse

from openmed.cli.main import _handle_doctor
from openmed.core.doctor import run_diagnostics


def test_run_diagnostics_returns_list():
    results = run_diagnostics()

    assert isinstance(results, list)
    assert len(results) > 0


def test_required_checks_present():
    results = run_diagnostics()

    names = {item["name"] for item in results}

    assert "python_version" in names
    assert "python_arch" in names
    assert "openmed_version" in names
    assert "hf_token" in names
    assert "openmed_offline" in names
    assert "manifest_exists" in names


def test_hf_token_not_exposed(monkeypatch):
    monkeypatch.setenv(
        "HF_TOKEN",
        "SUPER_SECRET_TOKEN_VALUE",
    )

    results = run_diagnostics()

    serialized = str(results)

    assert "SUPER_SECRET_TOKEN_VALUE" not in serialized

    token_check = next(item for item in results if item["name"] == "hf_token")

    assert token_check["status"] == "PASS"


def test_optional_dependencies_are_warn_or_pass():
    results = run_diagnostics()

    optional_checks = [
        item
        for item in results
        if item["name"]
        in {
            "mlx",
            "coreml",
            "onnx",
            "hf",
            "multimodal",
        }
    ]

    for check in optional_checks:
        assert check["status"] in {
            "PASS",
            "WARN",
        }


def test_doctor_returns_zero_when_no_fail():
    args = argparse.Namespace(
        json=False,
    )

    exit_code = _handle_doctor(args)

    assert exit_code == 0


def test_fail_returns_nonzero(monkeypatch):
    def fake_run_diagnostics():
        return [
            {
                "name": "fake_fail",
                "status": "FAIL",
                "details": "testing",
            }
        ]

    monkeypatch.setattr(
        "openmed.core.doctor.run_diagnostics",
        fake_run_diagnostics,
    )

    args = argparse.Namespace(
        json=False,
    )

    exit_code = _handle_doctor(args)

    assert exit_code == 1


def test_json_mode_returns_zero():
    args = argparse.Namespace(
        json=True,
    )

    exit_code = _handle_doctor(args)

    assert exit_code == 0
