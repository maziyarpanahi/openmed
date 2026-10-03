"""Focused tests for the offline install smoke check."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

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


@pytest.mark.parametrize("profile", smoke_check.BRIEF_PROFILES)
def test_brief_probe_report_requires_the_complete_ordered_contract(profile):
    names = list(smoke_check.BRIEF_CHECKS)
    if profile == "core":
        names.remove("brief_rest_client")
        names.remove("brief_mcp")
    rows = [{"name": name, "status": "passed"} for name in names]
    result = _completed([], stdout=json.dumps({"status": "passed", "checks": rows}))
    assert [c.name for c in smoke_check._brief_probe_results(result, profile)] == names
    rows.pop()
    result = _completed([], stdout=json.dumps({"status": "passed", "checks": rows}))
    assert smoke_check._brief_probe_results(result, profile)[-1].status == "failed"


@pytest.mark.parametrize("mutation", ["extra", "name", "reason", "status", "exit"])
def test_brief_probe_rejects_unsafe_child_output(mutation):
    secret = "synthetic-install-sensitive-value"
    rows = [dict(name="installed_origin", status="failed", reason="contract_failed")]
    payload = dict(status="failed", checks=rows)
    code = 1
    if mutation == "extra":
        rows[0]["text"] = secret
    elif mutation == "name":
        rows[0]["name"] = secret
    elif mutation == "reason":
        rows[0]["reason"] = secret
    elif mutation == "status":
        payload["status"] = secret
    else:
        code = 0
    checks = smoke_check._brief_probe_results(
        _completed([], returncode=code, stdout=json.dumps(payload)), "core"
    )
    assert checks == (smoke_check._failed("brief_probe", "invalid_result"),)
    assert secret not in json.dumps([check.to_dict() for check in checks])


def test_brief_probe_preserves_a_specific_controlled_failure():
    rows = [
        smoke_check._passed("installed_origin"),
        smoke_check._failed("packaged_resources", "contract_failed"),
    ]
    report = smoke_check.SmokeReport("failed", tuple(rows))
    result = _completed([], returncode=1, stdout=report.to_json())
    assert smoke_check._brief_probe_results(result, "core") == tuple(rows)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_missing_resource_control_changes_only_disposable_artifact(tmp_path, kind):
    import hashlib
    import io
    import tarfile
    import zipfile

    name = "openmed/" + smoke_check.BRIEF_RESOURCE
    data = b'{"synthetic": true}'
    output = tmp_path / "control"
    output.mkdir()
    if kind == "wheel":
        artifact = tmp_path / "openmed-0.0.0-py3-none-any.whl"
        with zipfile.ZipFile(artifact, "w") as archive:
            archive.writestr(name, data)
            archive.writestr("openmed/__init__.py", b"")
    else:
        artifact = tmp_path / "openmed-0.0.0.tar.gz"
        with tarfile.open(artifact, "w:gz") as archive:
            for path, content in [(name, data), ("openmed/__init__.py", b"")]:
                member = tarfile.TarInfo("openmed-0.0.0/" + path)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
    before = hashlib.sha256(artifact.read_bytes()).hexdigest()
    damaged = smoke_check._omit_brief_resource(artifact, output)
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == before
    assert damaged.parent == output
    if kind == "wheel":
        with zipfile.ZipFile(damaged) as archive:
            assert archive.namelist() == ["openmed/__init__.py"]
    else:
        with tarfile.open(damaged, "r:gz") as archive:
            assert archive.getnames() == ["openmed-0.0.0/openmed/__init__.py"]


def test_resource_control_cannot_succeed_when_resource_was_already_missing(tmp_path):
    import zipfile

    artifact = tmp_path / "openmed-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("openmed/__init__.py", b"")
    output = tmp_path / "control"
    output.mkdir()
    with pytest.raises(ValueError, match="invalid resource inventory"):
        smoke_check._omit_brief_resource(artifact, output)


def test_brief_fixture_is_synthetic_reviewed_and_not_diagnostic():
    from openmed.clinical.brief import build_clinical_brief

    value, context = smoke_check._brief_fixture()
    result = build_clinical_brief(value, model="extractive", context=context)
    assert result.refusal_reason is None
    assert result.summary == value.deidentified_text
    assert len(result.citations) == 3
    assert result.envelope["requires_human_review"]
    assert not result.envelope["is_diagnostic"]
    assert "dehydration" not in json.dumps(result.to_dict())


def test_artifact_checks_reject_missing_or_ambiguous_inputs(tmp_path):
    with pytest.raises(ValueError, match="expected one wheel and one sdist"):
        smoke_check.run_artifact_checks(tmp_path, python_executable="python")


def test_artifact_install_failures_are_redacted_and_all_six_cases_run(
    tmp_path, monkeypatch, capsys
):
    for name in ("openmed-0.0.0-py3-none-any.whl", "openmed-0.0.0.tar.gz"):
        (tmp_path / name).touch()
    monkeypatch.setattr(smoke_check.shutil, "which", lambda _: "/synthetic/uv")
    monkeypatch.setattr(
        smoke_check, "_omit_brief_resource", lambda artifact, destination: artifact
    )
    secret = "synthetic-sensitive-installer-error"
    monkeypatch.setattr(
        smoke_check.subprocess,
        "run",
        lambda *args, **kwargs: _completed([], returncode=1, stdout=secret),
    )
    payload = smoke_check.run_artifact_checks(tmp_path, python_executable="python")
    assert payload["status"] == "failed" and len(payload["checks"]) == 6
    assert all(row["reason"] == "installation_failed" for row in payload["checks"])
    assert secret not in json.dumps(payload) + capsys.readouterr().out


def test_resource_control_must_fail_at_the_resource_check(tmp_path, monkeypatch):
    for name in ("openmed-0.0.0-py3-none-any.whl", "openmed-0.0.0.tar.gz"):
        (tmp_path / name).touch()
    monkeypatch.setattr(smoke_check.shutil, "which", lambda _: "/synthetic/uv")
    monkeypatch.setattr(
        smoke_check, "_omit_brief_resource", lambda artifact, destination: artifact
    )
    monkeypatch.setattr(
        smoke_check.subprocess, "run", lambda *args, **kwargs: _completed([])
    )
    monkeypatch.setattr(
        smoke_check,
        "run_smoke_check",
        lambda **kwargs: smoke_check.SmokeReport(
            "failed", (smoke_check._failed("entry_point", "command_failed"),)
        ),
    )
    report = smoke_check.run_artifact_checks(tmp_path, python_executable="python")
    controls = [row for row in report["checks"] if row["profile"] == "missing_resource"]
    assert len(controls) == 2
    assert all(row["reason"] == "negative_control_failed" for row in controls)
