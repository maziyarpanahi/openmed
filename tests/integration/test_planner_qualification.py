"""Offline command and unchanged governance primitives without dispatch."""

from __future__ import annotations

import json
import socket
import sys
from types import ModuleType

import pytest

from openmed.agent.approvals import tokens
from openmed.agent.permissions import grants
from openmed.eval import planner_qualification as harness
from openmed.eval.planner_qualification_cli import main
from tests.unit.eval.test_planner_qualification import (
    compliant_planner,
    injected_planner,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "planner,exit_code,hostile_rate",
    [(compliant_planner, 0, 0.0), (injected_planner, 1, 1.0)],
)
def test_command_is_offline_and_never_dispatches(
    monkeypatch, tmp_path, planner, exit_code, hostile_rate, capsys
):
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append("side_effect")
        raise AssertionError("network_or_tool_execution")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(grants, "dispatch_with_capability_grant", forbidden)
    monkeypatch.setattr(tokens, "dispatch_with_approval_token", forbidden)
    module = ModuleType("synthetic_planner")
    module.propose = planner
    monkeypatch.setitem(sys.modules, module.__name__, module)
    output = tmp_path / "qualification.json"
    assert (
        main(["--planner", "synthetic_planner:propose", "--output", str(output)])
        == exit_code
    )
    report = json.loads(output.read_text())
    assert report["rates"]["hostile_instruction_compliance"] == hostile_rate
    assert attempts == []
    assert capsys.readouterr().out == ""


def test_real_governance_verifiers_are_used_without_callbacks(monkeypatch):
    counts = {"grant": 0, "projection": 0, "approval": 0}
    verify = grants.CapabilityGrantVerifier.verify
    project = harness.plan_data_projection
    consume = tokens.ApprovalTokenVerifier.consume

    def grant(self, *args, **kwargs):
        counts["grant"] += 1
        return verify(self, *args, **kwargs)

    def projection(*args, **kwargs):
        counts["projection"] += 1
        return project(*args, **kwargs)

    def approval(self, *args, **kwargs):
        counts["approval"] += 1
        return consume(self, *args, **kwargs)

    monkeypatch.setattr(grants.CapabilityGrantVerifier, "verify", grant)
    monkeypatch.setattr(harness, "plan_data_projection", projection)
    monkeypatch.setattr(tokens.ApprovalTokenVerifier, "consume", approval)
    assert harness.qualify_planner(compliant_planner).qualified
    assert counts == {"grant": 8, "projection": 8, "approval": 1}


@pytest.mark.parametrize(
    "reference", ["private/path:secret", "missing_private_module:secret"]
)
def test_command_errors_do_not_echo_private_references(
    monkeypatch, tmp_path, capsys, reference
):
    assert (
        main(["--planner", reference, "--output", str(tmp_path / "report.json")]) == 2
    )
    captured = capsys.readouterr()
    assert reference not in captured.err
    assert not (tmp_path / "report.json").exists()


def test_import_and_planner_output_are_discarded(monkeypatch, tmp_path, capsys):
    def importer(name):
        print("synthetic-private-import-canary")
        raise RuntimeError("synthetic-private-import-canary")

    monkeypatch.setattr(
        "openmed.eval.planner_qualification_cli.importlib.import_module", importer
    )
    assert (
        main(
            [
                "--planner",
                "private_module:propose",
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "qualification_unavailable\n"
