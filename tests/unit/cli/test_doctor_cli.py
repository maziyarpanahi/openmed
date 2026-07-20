from __future__ import annotations

import argparse
import json

from openmed.cli import main_module
from openmed.cli.main import _handle_doctor
from openmed.core import doctor as doctor_module
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
    assert "manifest_rows" in names


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
    assert token_check["present"] is True


def test_hf_token_absence_reports_boolean(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)

    results = run_diagnostics()

    token_check = next(item for item in results if item["name"] == "hf_token")

    assert token_check["status"] == "WARN"
    assert token_check["present"] is False


def test_unsupported_python_version_fails(monkeypatch):
    monkeypatch.setattr(doctor_module.sys, "version_info", (3, 9, 18))
    monkeypatch.setattr(doctor_module.platform, "python_version", lambda: "3.9.18")

    results = run_diagnostics()

    python_check = next(item for item in results if item["name"] == "python_version")

    assert python_check["status"] == "FAIL"


def test_unsupported_python_architecture_fails(monkeypatch):
    monkeypatch.setattr(doctor_module.platform, "machine", lambda: "i386")

    results = run_diagnostics()

    arch_check = next(item for item in results if item["name"] == "python_arch")

    assert arch_check["status"] == "FAIL"


def test_manifest_check_uses_canonical_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    results = run_diagnostics()

    manifest_check = next(item for item in results if item["name"] == "manifest_exists")

    assert manifest_check["status"] == "PASS"
    assert manifest_check["details"].endswith("models.jsonl")


def test_missing_manifest_reports_row_check(monkeypatch, tmp_path):
    monkeypatch.setattr(
        doctor_module,
        "MANIFEST_PATH",
        tmp_path / "missing-models.jsonl",
    )

    results = run_diagnostics()

    manifest_exists = next(
        item for item in results if item["name"] == "manifest_exists"
    )
    manifest_rows = next(item for item in results if item["name"] == "manifest_rows")

    assert manifest_exists["status"] == "WARN"
    assert manifest_rows["status"] == "WARN"
    assert "missing" in manifest_rows["details"]


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


def test_doctor_warns_when_optional_dependencies_are_missing(monkeypatch, capsys):
    def missing_dependency(module_name):
        raise ImportError(f"{module_name} missing")

    monkeypatch.setattr(
        doctor_module.importlib,
        "import_module",
        missing_dependency,
    )

    exit_code = _handle_doctor(argparse.Namespace(json=False))

    captured = capsys.readouterr()
    assert exit_code == 0

    for name in doctor_module.OPTIONAL_EXTRAS:
        assert f"WARN {name}:" in captured.out


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


def test_doctor_json_command_is_registered(monkeypatch, capsys):
    diagnostics = [
        {
            "name": "python_version",
            "status": "PASS",
            "details": "3.11.10",
        }
    ]

    monkeypatch.setattr(
        "openmed.core.doctor.run_diagnostics",
        lambda: diagnostics,
    )

    exit_code = main_module.main(["doctor", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["command"] == "doctor"
    assert payload["data"]["checks"] == diagnostics
