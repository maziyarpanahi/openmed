"""Smoke tests for the tracked CLI and MCP packages."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import openmed
from openmed.cli import main as cli_entry
from openmed.cli import main_module

ROOT = Path(__file__).resolve().parents[3]


def _run_module(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_argparse_cli_imports_and_prints_help() -> None:
    assert callable(cli_entry)
    assert callable(main_module.main)

    result = _run_module("openmed.cli.main", "--help")

    assert result.returncode == 0
    assert "Command-line utilities for OpenMed" in result.stdout


def test_argparse_cli_prints_version() -> None:
    result = _run_module("openmed.cli.main", "--version")

    assert result.returncode == 0
    assert openmed.__version__ in result.stdout


def test_argparse_cli_parses_benchmark_pii_modes() -> None:
    parser = main_module.build_parser()

    suite_args = parser.parse_args(
        ["benchmark", "pii", "--suite", "shield", "--models", "fixture-model"]
    )
    assert suite_args.command == "benchmark"
    assert suite_args.benchmark_command == "pii"
    assert suite_args.attack is None
    assert suite_args.models == ["fixture-model"]

    attack_args = parser.parse_args(
        [
            "benchmark",
            "pii",
            "--attack",
            "reid",
            "--suite",
            "golden",
            "--model",
            "unit-model",
        ]
    )
    assert attack_args.command == "benchmark"
    assert attack_args.benchmark_command == "pii"
    assert attack_args.attack == "reid"
    assert attack_args.model == "unit-model"


def test_argparse_cli_without_command_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main_module.main([])

    assert result == 0
    assert "Command-line utilities for OpenMed" in capsys.readouterr().out


def test_profile_command_returns_gate_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jsonl = tmp_path / "extracted.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "document_id": "secret-doc",
                "person_id": "secret-patient",
                "note_text": "Diabetes noted. Mystery term noted.",
                "entities": [
                    {
                        "text": "Diabetes",
                        "domain_id": "Condition",
                        "start": 0,
                        "end": 8,
                        "concept_id": 201826,
                    },
                    {
                        "text": "Mystery term",
                        "domain_id": "Condition",
                        "start": 16,
                        "end": 28,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    passed = main_module.main(
        ["profile", "--input", str(jsonl), "--completeness-floor", "0.7"]
    )
    pass_payload = json.loads(capsys.readouterr().out)
    failed = main_module.main(
        ["profile", "--input", str(jsonl), "--completeness-floor", "0.9"]
    )
    fail_payload = json.loads(capsys.readouterr().out)

    assert passed == 0
    assert pass_payload["pipeline_gate"]["passed"] is True
    assert failed == 2
    assert fail_payload["pipeline_gate"]["passed"] is False


def test_tui_command_is_not_registered() -> None:
    result = _run_module("openmed.cli.main", "tui")

    assert result.returncode == 2
    assert "invalid choice: 'tui'" in result.stderr


def test_typer_surface_is_importable() -> None:
    from openmed.cli import typer_app

    assert callable(typer_app.main)


def test_mcp_package_imports_and_prints_help() -> None:
    from openmed.mcp import server

    assert callable(server.create_mcp_server)

    result = _run_module("openmed.mcp.server", "--help")

    assert result.returncode == 0
    assert "Run the OpenMed MCP server" in result.stdout


def test_console_script_is_declared() -> None:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["openmed"] == "openmed.cli:main"
