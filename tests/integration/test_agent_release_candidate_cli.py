"""Offline command controls using an actual synthetic git checkout."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from openmed.eval.agent_release_candidate import AgentCandidatePacket
from tests.fixtures.eval.agent_release_candidate import _KEY, build_candidate_inputs

pytestmark = pytest.mark.integration


def _arguments(inputs, output):
    key_file = inputs.tmp / "signing-key"
    key_file.write_bytes(_KEY)
    return [
        "--repo-root",
        str(inputs.root),
        "--source-sha",
        inputs.sha,
        "--wheel",
        str(inputs.params["wheel"]),
        "--sdist",
        str(inputs.params["sdist"]),
        "--tool-catalog",
        str(inputs.params["tool_catalog"]),
        "--policy",
        str(inputs.params["policy"]),
        "--evidence",
        str(inputs.evidence),
        "--signing-key-file",
        str(key_file),
        "--output",
        str(output),
    ]


@pytest.mark.parametrize("entry", ["cli", "script"])
def test_real_commands_produce_same_verifiable_packet(tmp_path, entry):
    inputs = build_candidate_inputs(tmp_path)
    command = (
        [str(Path(sys.executable).with_name("openmed")), "gates", "agent-release"]
        if entry == "cli"
        else [sys.executable, "scripts/release/agent_release_gate.py"]
    )
    outputs = []
    for index in range(2):
        output = tmp_path / f"bundle-{index}.json"
        args = command + _arguments(inputs, output)
        if entry == "cli":
            args.append("--json")
        completed = subprocess.run(args, capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stderr
        bundle = json.loads(output.read_text())
        packet = AgentCandidatePacket.from_dict(bundle["packet"])
        assert packet.verify(_KEY)
        assert packet.decision == "READY"
        stdout = json.loads(completed.stdout)
        if entry == "cli":
            assert stdout["command"] == "gates agent-release"
            assert stdout["data"] == bundle
        else:
            assert stdout == bundle["packet"]
        assert completed.stderr == ""
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("entry", ["cli", "script"])
def test_real_commands_exit_nonzero_and_sign_dirty_refusal(tmp_path, entry):
    inputs = build_candidate_inputs(tmp_path)
    (inputs.root / "source.txt").write_text("synthetic dirty mutation")
    command = (
        [
            str(Path(sys.executable).with_name("openmed")),
            "gates",
            "agent-release",
            "--json",
        ]
        if entry == "cli"
        else [sys.executable, "scripts/release/agent_release_gate.py"]
    )
    output = tmp_path / "refusal.json"
    completed = subprocess.run(
        command + _arguments(inputs, output), capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 1
    bundle = json.loads(output.read_text())
    assert bundle["packet"]["reason_codes"] == ["dirty_tree"]
    assert AgentCandidatePacket.from_dict(bundle["packet"]).verify(_KEY)
    assert str(tmp_path) not in completed.stdout + completed.stderr


def test_cli_missing_key_is_value_free_and_writes_nothing(tmp_path):
    inputs = build_candidate_inputs(tmp_path)
    output = tmp_path / "bundle.json"
    args = _arguments(inputs, output)
    (tmp_path / "signing-key").unlink()
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("openmed")),
            "gates",
            "agent-release",
            "--json",
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert json.loads(result.stdout)["error"]["code"] == "signing_key_unavailable"
    assert str(tmp_path) not in result.stdout + result.stderr
    assert not output.exists()
