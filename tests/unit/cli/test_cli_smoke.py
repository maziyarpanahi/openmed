"""Smoke tests for the tracked CLI and MCP packages."""

from __future__ import annotations

import io
import subprocess
import sys
import types
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


def test_tui_entry_invokes_openmed_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: dict[str, object] = {}

    class FakeTUI:
        def __init__(self, **kwargs: object) -> None:
            launched["kwargs"] = kwargs

        def run(self) -> None:
            launched["ran"] = True

    fake_tui = types.ModuleType("openmed.tui")
    fake_tui.OpenMedTUI = FakeTUI
    monkeypatch.setitem(sys.modules, "openmed.tui", fake_tui)

    result = main_module.main(
        [
            "tui",
            "--model",
            "disease_detection_superclinical",
            "--confidence-threshold",
            "0.6",
        ]
    )

    assert result == 0
    assert launched == {
        "kwargs": {
            "model_name": "disease_detection_superclinical",
            "confidence_threshold": 0.6,
        },
        "ran": True,
    }


def test_tui_entry_has_base_install_fallback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delitem(sys.modules, "openmed.tui", raising=False)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    result = main_module.main(["tui"])

    assert result == 0
    assert "OpenMed TUI entry is installed" in capsys.readouterr().out


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
