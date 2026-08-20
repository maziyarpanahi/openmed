"""Focused tests for the offline install smoke check."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from scripts.install import smoke_check


def _fake_install_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_name = "python.exe" if os.name == "nt" else "python"
    entry_point_name = "openmed.exe" if os.name == "nt" else "openmed"
    python_executable = bin_dir / python_name
    python_executable.touch(mode=0o755)
    (bin_dir / entry_point_name).touch(mode=0o755)
    return python_executable


def _completed(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, "")


def test_report_serialization_is_compact_and_stable() -> None:
    report = smoke_check.SmokeReport(
        status="passed",
        checks=(
            smoke_check.CheckResult(
                name="synthetic_offline_command",
                status="passed",
                details={
                    "change_count": 1,
                    "document_hash": "sha256:" + "a" * 64,
                },
            ),
        ),
    )

    rendered = report.to_json()
    assert "\n" not in rendered
    assert json.loads(rendered) == {
        "checks": [
            {
                "change_count": 1,
                "document_hash": "sha256:" + "a" * 64,
                "name": "synthetic_offline_command",
                "status": "passed",
            }
        ],
        "offline": True,
        "schema_version": 1,
        "status": "passed",
    }


def test_run_smoke_check_uses_offline_clean_environment(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    python_executable = _fake_install_bin(tmp_path)

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, **kwargs})
        if command[-1] == "--version":
            return _completed(command, stdout="openmed 2.0.0\n")
        if command[-3:] == ["models", "validate", "--json"]:
            return _completed(
                command,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "data": {
                            "ok": True,
                            "violation_count": 0,
                            "messages": ["manifest: OK (3 rows checked)"],
                        },
                    }
                ),
            )
        return _completed(
            command,
            stdout=json.dumps(
                {
                    "change_count": 1,
                    "document_hash": "sha256:" + "b" * 64,
                    "entry_point_declared": True,
                    "package_version": "2.0.0",
                    "surface_hash": "sha256:" + "c" * 64,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

    report = smoke_check.run_smoke_check(
        python_executable=str(python_executable),
        runner=runner,
    )

    assert report.status == "passed"
    assert [check.name for check in report.checks] == [
        "entry_point",
        "bundled_manifest",
        "synthetic_offline_command",
    ]
    assert len(calls) == 4
    for call in calls:
        assert call["env"]["OPENMED_OFFLINE"] == "1"
        assert call["env"]["HF_HUB_OFFLINE"] == "1"
        assert call["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert call["env"]["HF_DATASETS_OFFLINE"] == "1"
        assert call["env"]["PYTHONNOUSERSITE"] == "1"
        assert call["env"]["PATH"] == str(python_executable.parent)
        assert call["cwd"] != Path.cwd()


def test_manifest_check_rejects_failure_without_echoing_child_output() -> None:
    raw_sensitive_value = "synthetic-only-value-that-must-not-be-reported"
    result = _completed(
        ["openmed", "models", "validate", "--json"],
        returncode=1,
        stdout=json.dumps(
            {
                "ok": False,
                "error": {"message": raw_sensitive_value},
            }
        ),
    )

    check = smoke_check._manifest_check(result)

    assert check.status == "failed"
    assert raw_sensitive_value not in json.dumps(check.to_dict())


def test_manifest_check_requires_positive_validated_row_count() -> None:
    result = _completed(
        ["openmed", "models", "validate", "--json"],
        stdout=json.dumps(
            {
                "ok": True,
                "data": {
                    "ok": True,
                    "violation_count": 0,
                    "messages": ["manifest: OK"],
                },
            }
        ),
    )

    assert smoke_check._manifest_check(result).to_dict() == {
        "name": "bundled_manifest",
        "reason": "invalid_result",
        "status": "failed",
    }


def test_synthetic_check_rejects_nondeterministic_output(tmp_path: Path) -> None:
    outputs = iter(
        [
            json.dumps(
                {
                    "change_count": 1,
                    "document_hash": "sha256:" + "a" * 64,
                    "entry_point_declared": True,
                    "package_version": "2.0.0",
                    "surface_hash": "sha256:" + "b" * 64,
                }
            ),
            json.dumps(
                {
                    "change_count": 1,
                    "document_hash": "sha256:" + "c" * 64,
                    "entry_point_declared": True,
                    "package_version": "2.0.0",
                    "surface_hash": "sha256:" + "d" * 64,
                }
            ),
        ]
    )

    def runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return _completed(command, stdout=next(outputs))

    check = smoke_check._synthetic_check(
        "/tmp/fake-env/bin/python",
        expected_version="2.0.0",
        cwd=tmp_path,
        environment={"OPENMED_OFFLINE": "1"},
        runner=runner,
    )

    assert check.to_dict() == {
        "name": "synthetic_offline_command",
        "reason": "non_deterministic_output",
        "status": "failed",
    }


def test_main_redacts_unexpected_exception(monkeypatch: Any, capsys: Any) -> None:
    raw_sensitive_value = "synthetic-only-value-that-must-not-be-reported"

    def fail(**_: Any) -> smoke_check.SmokeReport:
        raise RuntimeError(raw_sensitive_value)

    monkeypatch.setattr(smoke_check, "run_smoke_check", fail)

    assert smoke_check.main([]) == 1
    output = capsys.readouterr().out
    assert raw_sensitive_value not in output
    assert json.loads(output)["checks"] == [
        {
            "name": "smoke_check",
            "reason": "internal_error",
            "status": "failed",
        }
    ]


def test_run_smoke_check_rejects_console_package_version_mismatch(
    tmp_path: Path,
) -> None:
    python_executable = _fake_install_bin(tmp_path)

    def runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if command[-1] == "--version":
            return _completed(command, stdout="openmed 2.0.0\n")
        if command[-3:] == ["models", "validate", "--json"]:
            return _completed(
                command,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "data": {
                            "ok": True,
                            "violation_count": 0,
                            "messages": ["manifest: OK (3 rows checked)"],
                        },
                    }
                ),
            )
        return _completed(
            command,
            stdout=json.dumps(
                {
                    "change_count": 1,
                    "document_hash": "sha256:" + "b" * 64,
                    "entry_point_declared": True,
                    "package_version": "2.0.1",
                    "surface_hash": "sha256:" + "c" * 64,
                }
            ),
        )

    report = smoke_check.run_smoke_check(
        python_executable=str(python_executable),
        runner=runner,
    )

    assert report.status == "failed"
    assert report.checks[-1].to_dict() == {
        "name": "synthetic_offline_command",
        "reason": "version_mismatch",
        "status": "failed",
    }
    assert "2.0.0" not in report.to_json()
    assert "2.0.1" not in report.to_json()


def test_run_smoke_check_does_not_fall_back_to_unrelated_path_entry_point(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    python_dir = tmp_path / "python-bin"
    python_dir.mkdir()
    python_name = "python.exe" if os.name == "nt" else "python"
    entry_point_name = "openmed.exe" if os.name == "nt" else "openmed"
    python_executable = python_dir / python_name
    python_executable.touch(mode=0o755)
    unrelated_dir = tmp_path / "unrelated-bin"
    unrelated_dir.mkdir()
    (unrelated_dir / entry_point_name).touch(mode=0o755)
    monkeypatch.setenv("PATH", f"{unrelated_dir}{os.pathsep}{python_dir}")

    report = smoke_check.run_smoke_check(
        python_executable=str(python_executable),
        runner=lambda *args, **kwargs: _completed(list(args[0])),
    )

    assert report.to_dict()["checks"] == [
        {"name": "entry_point", "reason": "not_installed", "status": "failed"}
    ]


def test_unsafe_version_output_is_not_relayed(tmp_path: Path) -> None:
    raw_sensitive_value = "synthetic-only-value-that-must-not-be-reported"
    python_executable = _fake_install_bin(tmp_path)

    def runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return _completed(command, stdout=f"openmed {raw_sensitive_value}\n")

    report = smoke_check.run_smoke_check(
        python_executable=str(python_executable),
        runner=runner,
    )

    rendered = report.to_json()
    assert raw_sensitive_value not in rendered
    assert report.checks[0].to_dict() == {
        "name": "entry_point",
        "reason": "invalid_version_output",
        "status": "failed",
    }
