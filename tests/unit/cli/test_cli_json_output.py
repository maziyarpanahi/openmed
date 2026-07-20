"""Uniform CLI JSON output, exit-code, and scriptability guards (#1742).

Covers three contracts:

* every scriptable subcommand exposes ``--json`` and, when set, emits the
  ``{"ok", "command", "data"}`` success envelope or the
  ``{"ok", "command", "error"}`` failure envelope with a documented exit code;
* the exit-code table in ``openmed/cli/_output.py`` is respected; and
* no CLI code path blocks on interactive input (a static scriptability guard).
"""

from __future__ import annotations

import argparse
import ast
import importlib
import io
import json
from pathlib import Path

import pytest

from openmed.cli._output import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_USAGE,
    CliError,
    emit,
    emit_error,
)
from openmed.cli.main import build_parser, main

CLI_DIR = Path(__file__).resolve().parents[3] / "openmed" / "cli"


# ---------------------------------------------------------------------------
# Every leaf subcommand exposes --json + a command_path
# ---------------------------------------------------------------------------


def _leaf_parsers(parser, path=""):
    sub = next(
        (a for a in parser._actions if isinstance(a, argparse._SubParsersAction)),
        None,
    )
    if sub is not None:
        for name, child in sub.choices.items():
            yield from _leaf_parsers(child, f"{path} {name}".strip())
        if parser.get_default("handler") is None:
            return
    if parser.get_default("handler") is not None:
        yield path, parser


def test_every_subcommand_exposes_json_flag_and_command_path():
    parser = build_parser()
    seen = set()
    missing_json = []
    missing_path = []
    for path, leaf in _leaf_parsers(parser):
        if id(leaf) in seen:
            continue
        seen.add(id(leaf))
        if not any("--json" in a.option_strings for a in leaf._actions):
            missing_json.append(path)
        if leaf.get_default("command_path") is None:
            missing_path.append(path)

    assert missing_json == [], f"subcommands without --json: {missing_json}"
    assert missing_path == [], f"subcommands without command_path: {missing_path}"


# ---------------------------------------------------------------------------
# Envelope helpers behave per contract
# ---------------------------------------------------------------------------


def _ns(**kwargs):
    return argparse.Namespace(**kwargs)


def test_emit_wraps_success_in_stable_envelope():
    stream = io.StringIO()
    rc = emit(
        _ns(json_output=True, command_path="models list"),
        {"count": 1},
        human="ignored",
        stream=stream,
    )
    payload = json.loads(stream.getvalue())
    assert rc == EXIT_OK
    assert payload == {"ok": True, "command": "models list", "data": {"count": 1}}


def test_emit_writes_human_when_not_json():
    stream = io.StringIO()
    emit(
        _ns(json_output=False, command_path="x"),
        {"count": 1},
        human="hello",
        stream=stream,
    )
    assert stream.getvalue() == "hello\n"


def test_emit_error_envelope_and_exit_code():
    stream = io.StringIO()
    rc = emit_error(
        _ns(json_output=True, command_path="audit verify"),
        CliError("nope", code="load_failed", exit_code=EXIT_ERROR),
        json_stream=stream,
    )
    payload = json.loads(stream.getvalue())
    assert rc == EXIT_ERROR
    assert payload == {
        "ok": False,
        "command": "audit verify",
        "error": {"code": "load_failed", "message": "nope"},
    }


def test_emit_error_writes_message_to_text_stream_without_json():
    stream = io.StringIO()
    rc = emit_error(
        _ns(json_output=False, command_path="x"),
        CliError("bad usage", code="invalid_argument", exit_code=EXIT_USAGE),
        text_stream=stream,
    )
    assert rc == EXIT_USAGE
    assert stream.getvalue() == "bad usage\n"


def test_main_renders_cli_error_from_a_real_failing_command(capsys):
    # audit verify on a missing report file must render the error envelope
    # (JSON to stdout) and return a non-zero exit code, not crash.
    rc = main(["audit", "verify", "/no/such/report.json", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc != EXIT_OK
    assert payload["ok"] is False
    assert payload["command"] == "audit verify"
    assert set(payload["error"]) == {"code", "message"}


def test_analyze_missing_input_file_uses_error_envelope(tmp_path, capsys):
    missing = tmp_path / "missing.txt"
    rc = main(["analyze", "--input-file", str(missing), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == EXIT_ERROR
    assert payload["ok"] is False
    assert payload["command"] == "analyze"
    assert payload["error"]["code"] == "input_not_found"


def test_main_sanitizes_unexpected_runtime_errors(monkeypatch, capsys):
    cli_main = importlib.import_module("openmed.cli.main")
    secret = "MRN 123456789"

    def fail(_args):
        raise RuntimeError(secret)

    monkeypatch.setattr(cli_main, "_handle_analyze", fail)
    rc = cli_main.main(["analyze", "--text", "synthetic", "--json"])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert rc == EXIT_ERROR
    assert payload["ok"] is False
    assert payload["command"] == "analyze"
    assert payload["error"] == {
        "code": "runtime_error",
        "message": "Command failed with RuntimeError.",
    }
    assert secret not in output


# ---------------------------------------------------------------------------
# Static scriptability guard: no interactive input anywhere in openmed/cli
# ---------------------------------------------------------------------------

_INTERACTIVE_CALLS = {"input", "getpass"}
_INTERACTIVE_ATTRS = {"prompt", "confirm", "getpass"}


def _interactive_input_sites(source: str) -> list[str]:
    tree = ast.parse(source)
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in _INTERACTIVE_CALLS:
            hits.append(func.id)
        elif isinstance(func, ast.Attribute) and func.attr in _INTERACTIVE_ATTRS:
            hits.append(func.attr)
    return hits


def test_guard_detects_planted_interactive_input():
    assert _interactive_input_sites("x = input('name? ')") == ["input"]
    assert _interactive_input_sites("import typer\ntyper.confirm('ok?')") == ["confirm"]


def test_no_cli_path_blocks_on_interactive_input():
    offenders = {}
    for path in sorted(CLI_DIR.rglob("*.py")):
        hits = _interactive_input_sites(path.read_text(encoding="utf-8"))
        if hits:
            offenders[path.name] = hits
    assert offenders == {}, f"interactive input in scriptable CLI paths: {offenders}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
