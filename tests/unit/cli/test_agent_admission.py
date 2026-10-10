"""Offline operator CLI tests with synthetic keys and controlled metadata."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openmed.cli import typer_app

KEY = b"synthetic-admission-cli-key-32-bytes"
WORKFLOW = "workflow:org.example/synthetic-fhir"


def invoke(command: str, tmp_path: Path, *options: str):
    key = tmp_path / "key.bin"
    key.write_bytes(KEY)
    return CliRunner().invoke(
        typer_app.build_app(),
        [
            "agents",
            command,
            "--state",
            str(tmp_path / "ledger.db"),
            "--anchor",
            str(tmp_path / "anchor.db"),
            "--key-file",
            str(key),
            *options,
        ],
    )


def test_no_configuration_is_disabled() -> None:
    result = CliRunner().invoke(typer_app.build_app(), ["agents", "status"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["reason_code"] == "admission_disabled"


def test_stop_resume_status_are_durable_and_content_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_network(*args, **kwargs):
        raise AssertionError("unexpected network")

    monkeypatch.setattr(socket, "create_connection", no_network)
    enabled = invoke("resume", tmp_path, "--initialize", "--scope", WORKFLOW)
    assert enabled.exit_code == 0, enabled.output
    assert json.loads(enabled.stdout)["generation"] == 2
    global_status = invoke("status", tmp_path)
    assert json.loads(global_status.stdout)["reason_code"] == "admission_disabled"
    stopped = invoke("stop", tmp_path)
    assert stopped.exit_code == 0, stopped.output
    assert json.loads(stopped.stdout)["reason_code"] == "admission_stopped"
    status = invoke("status", tmp_path, "--scope", WORKFLOW)
    assert json.loads(status.stdout)["reason_code"] == "admission_stopped"
    resumed = invoke("resume", tmp_path)
    assert resumed.exit_code == 0, resumed.output
    assert json.loads(resumed.stdout)["generation"] == 4
    for result in (enabled, stopped, status, resumed):
        assert str(tmp_path) not in result.output
        assert KEY.decode() not in result.output
        assert set(json.loads(result.stdout)) == {
            "scope",
            "reason_code",
            "generation",
            "receipt_digest",
        }


def test_missing_state_requires_explicit_provisioning(tmp_path: Path) -> None:
    for command in ("resume", "stop", "status"):
        result = invoke(command, tmp_path)
        assert result.exit_code == 1
        assert "untrusted_state" in result.output
        assert not (tmp_path / "ledger.db").exists()
        assert str(tmp_path) not in result.output


@pytest.mark.parametrize("option", ["--scope", "--role"])
def test_invalid_metadata_never_echoes_rejected_content(
    tmp_path: Path, option: str
) -> None:
    sentinel = "Synthetic Patient; bearer secret; /private/source"
    result = invoke("resume", tmp_path, "--initialize", option, sentinel)
    assert result.exit_code == 1
    assert sentinel not in result.output
    assert "invalid_control_metadata" in result.output
    assert not (tmp_path / "ledger.db").exists()


def test_reinitialize_cannot_bypass_stop(tmp_path: Path) -> None:
    assert invoke("resume", tmp_path, "--initialize").exit_code == 0
    assert invoke("stop", tmp_path).exit_code == 0
    result = invoke("resume", tmp_path, "--initialize")
    assert result.exit_code == 1
    assert "already_initialized" in result.output
    assert "admission_stopped" in invoke("status", tmp_path).output


def test_unreadable_key_never_discloses_private_path(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        typer_app.build_app(),
        [
            "agents",
            "resume",
            "--state",
            str(tmp_path / "ledger.db"),
            "--anchor",
            str(tmp_path / "anchor.db"),
            "--key-file",
            str(tmp_path / "private-secret"),
        ],
    )
    assert result.exit_code == 1
    assert "unreadable_key" in result.output
    assert str(tmp_path) not in result.output


def test_production_entry_point_has_default_off_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from openmed.cli import main

    assert main(["agents", "status"]) == 0
    assert json.loads(capsys.readouterr().out)["reason_code"] == "admission_disabled"
    assert main(["agents", "status", "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["ok"] is True
    assert envelope["command"] == "agents status"
    assert envelope["data"]["reason_code"] == "admission_disabled"


def test_production_entry_point_stop_resume_and_invalid_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from openmed.cli import main

    key = tmp_path / "key.bin"
    key.write_bytes(KEY)
    common = [
        "--state",
        str(tmp_path / "ledger.db"),
        "--anchor",
        str(tmp_path / "anchor.db"),
        "--key-file",
        str(key),
    ]
    assert main(["agents", "resume", *common, "--initialize", "--scope", WORKFLOW]) == 0
    assert json.loads(capsys.readouterr().out)["generation"] == 2
    assert main(["agents", "stop", *common]) == 0
    assert json.loads(capsys.readouterr().out)["reason_code"] == "admission_stopped"
    assert main(["agents", "status", *common, "--scope", WORKFLOW]) == 0
    assert json.loads(capsys.readouterr().out)["reason_code"] == "admission_stopped"
    sentinel = "Synthetic Patient; bearer secret"
    assert main(["agents", "resume", *common, "--role", sentinel, "--json"]) == 1
    result = capsys.readouterr()
    assert sentinel not in result.out + result.err
    assert json.loads(result.out)["error"]["code"] == "invalid_control_metadata"
    assert main(["agents", "resume", *common]) == 0
    assert json.loads(capsys.readouterr().out)["generation"] == 4


def test_actual_console_script_supports_agents_status() -> None:
    import subprocess
    import sys

    console = Path(sys.executable).parent / "openmed"
    result = subprocess.run(
        [str(console), "agents", "status"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["reason_code"] == "admission_disabled"
