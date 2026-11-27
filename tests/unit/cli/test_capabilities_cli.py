"""``openmed capabilities`` prints the canonical capability descriptor."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from openmed.cli import main_module
from openmed.core import capabilities
from openmed.core.capabilities import (
    CAPABILITY_SURFACES,
    capability_descriptor,
    load_capability_descriptor_schema,
    registered_backends,
    validate_capability_descriptor,
)

ROOT = Path(__file__).resolve().parents[3]
TOKEN_SENTINEL = "synthetic-capabilities-cli-token-sentinel"


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    exit_code = main_module.main(["capabilities", *argv])
    return exit_code, capsys.readouterr().out


def test_json_output_validates_and_matches_the_python_digest(capsys) -> None:
    exit_code, out = _run(capsys, "--json")

    assert exit_code == 0
    envelope = json.loads(out)
    assert set(envelope) == {"ok", "command", "data"}
    assert envelope["ok"] is True
    assert envelope["command"] == "capabilities"
    data = envelope["data"]
    Draft202012Validator(load_capability_descriptor_schema()).validate(data)
    validate_capability_descriptor(data)
    expected = capability_descriptor()
    assert data == expected
    assert data["digest"] == expected["digest"]
    assert data["served_models"] == []


def test_json_output_matches_a_separate_python_process() -> None:
    probe = (
        "import sys\n"
        "from openmed.cli import main_module\n"
        "from openmed.core.capabilities import capability_descriptor\n"
        "print(capability_descriptor()['digest'])\n"
        "sys.stdout.flush()\n"
        "sys.exit(main_module.main(['capabilities', '--json']))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    python_digest, cli_json = result.stdout.split("\n", 1)
    assert json.loads(cli_json)["data"]["digest"] == python_digest


def test_planted_token_and_paths_never_appear_in_output(
    capsys, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_TOKEN", TOKEN_SENTINEL)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", TOKEN_SENTINEL)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "synthetic-hf-home"))
    monkeypatch.setenv("OPENMED_CHINESE_USER_DICT", str(tmp_path / "synthetic.txt"))

    json_exit, json_out = _run(capsys, "--json")
    text_exit, text_out = _run(capsys)

    assert json_exit == text_exit == 0
    for out in (json_out, text_out):
        assert TOKEN_SENTINEL not in out
        assert str(tmp_path) not in out
        assert str(Path.home()) not in out


def test_missing_extras_are_reported_without_failing(
    capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capabilities, "_unimportable", lambda modules: tuple(modules))

    json_exit, json_out = _run(capsys, "--json")
    text_exit, text_out = _run(capsys)

    assert json_exit == 0
    assert text_exit == 0
    data = json.loads(json_out)["data"]
    assert not any(item["available"] for item in data["backends"])
    surfaces = {item["name"]: item["available"] for item in data["surfaces"]}
    assert surfaces == {
        "cli": True,
        "graphql": False,
        "grpc": False,
        "mcp": False,
        "python": True,
        "rest": False,
    }
    validate_capability_descriptor(data)
    assert "missing    hf" in text_out


def test_human_output_lists_sections_in_a_stable_order(capsys) -> None:
    exit_code, out = _run(capsys)
    descriptor = capability_descriptor()

    assert exit_code == 0
    lines = out.splitlines()
    assert lines[0] == "OpenMed capabilities (openmed.capabilities v1)"
    assert lines[1] == f"Digest: {descriptor['digest']}"
    headings = [line for line in lines if line and not line.startswith(" ")][2:]
    assert headings == [
        "Surfaces:",
        "Backends:",
        "Schema versions:",
        "Policy profiles:",
        "Served models: none",
    ]

    def section(title: str) -> list[str]:
        start = lines.index(title) + 1
        end = lines.index("", start)
        return [line.split()[-1] for line in lines[start:end]]

    assert section("Surfaces:") == list(CAPABILITY_SURFACES)
    backend_lines = lines[lines.index("Backends:") + 1 :]
    backend_names = [
        line.split()[1] for line in backend_lines[: len(registered_backends())]
    ]
    assert backend_names == list(registered_backends())
    schema_lines = lines[lines.index("Schema versions:") + 1 :]
    schema_names = [
        line.split()[0] for line in schema_lines[: len(descriptor["schema_versions"])]
    ]
    assert schema_names == [item["name"] for item in descriptor["schema_versions"]]
    assert section("Policy profiles:") == descriptor["policy_profiles"]
    second_exit, second = _run(capsys)
    assert second_exit == 0
    assert second == out
