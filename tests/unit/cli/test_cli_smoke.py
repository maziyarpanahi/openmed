"""Smoke tests for the tracked CLI and MCP packages."""

from __future__ import annotations

import subprocess
import sys
from importlib import metadata
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


def test_tui_entry_is_a_clear_not_implemented_stub() -> None:
    args = main_module.build_parser().parse_args(
        [
            "tui",
            "--model",
            "synthetic-model",
            "--confidence-threshold",
            "0.6",
        ]
    )

    with pytest.raises(NotImplementedError, match="backend was removed"):
        args.handler(args)


def test_tui_entry_reports_the_stub_at_the_console_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main_module.main(["tui"])

    assert result == 1
    assert "TUI is not implemented" in capsys.readouterr().err


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


def test_console_script_resolves_to_tracked_cli_package() -> None:
    entry_point = next(
        item
        for item in metadata.entry_points(group="console_scripts")
        if item.name == "openmed"
    )

    assert entry_point.value == "openmed.cli:main"
    assert entry_point.load() is cli_entry
