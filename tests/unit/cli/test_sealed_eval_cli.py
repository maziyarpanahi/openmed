"""Offline tests for ``openmed eval sealed`` using a fake service."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

cli_main = importlib.import_module("openmed.cli.main")
from openmed.cli import sealed_eval

SECRET = "CASE-TEXT-SECRET-PATIENT"


class FakeService:
    def __init__(self, **check):
        self.check = {"signed": True, "digest_match": True, "requires_network": False}
        self.check.update(check)
        self.evaluated = False

    def preflight(self, model_ref, holdout_manifest):
        return self.check

    def evaluate(self, model_ref, holdout_manifest):
        self.evaluated = True
        return {"aggregate": {"f1": 0.9, "n": 10}, "signature": "sig-abc"}


def _run(monkeypatch, tmp_path, service, *extra, manifest=True):
    holdout = tmp_path / "holdout.json"
    if manifest:
        holdout.write_text(SECRET)
    monkeypatch.setattr(sealed_eval, "default_service", lambda: service)
    out = tmp_path / "out" / "result.json"
    argv = [
        "eval",
        "sealed",
        "--model",
        "cand-1",
        "--holdout-manifest",
        str(holdout),
        "--output",
        str(out),
        *extra,
    ]
    return cli_main.main(argv), out


def test_success_writes_only_aggregate_envelope(monkeypatch, tmp_path, capsys):
    service = FakeService()
    code, out = _run(monkeypatch, tmp_path, service, "--json")
    assert code == 0
    assert json.loads(out.read_text())["signature"] == "sig-abc"
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert str(tmp_path) not in captured.out + captured.err
    assert json.loads(captured.out)["data"]["status"] == "completed"


@pytest.mark.parametrize(
    "check,error_code",
    [
        ({"signed": False}, "candidate_unsigned"),
        ({"digest_match": False}, "candidate_digest_mismatch"),
        ({"requires_network": True}, "network_required"),
    ],
)
def test_refusals_fail_before_evaluation(
    monkeypatch, tmp_path, capsys, check, error_code
):
    service = FakeService(**check)
    code, out = _run(monkeypatch, tmp_path, service, "--json")
    assert code == 1
    assert not service.evaluated
    assert not out.exists()
    assert json.loads(capsys.readouterr().out)["error"]["code"] == error_code


def test_missing_manifest_is_usage_error(monkeypatch, tmp_path, capsys):
    service = FakeService()
    code, out = _run(monkeypatch, tmp_path, service, "--json", manifest=False)
    assert code == 2
    assert not service.evaluated
    assert (
        json.loads(capsys.readouterr().out)["error"]["code"]
        == "holdout_manifest_missing"
    )


def test_no_service_configured_fails_cleanly(monkeypatch, tmp_path, capsys):
    holdout = tmp_path / "h.json"
    holdout.write_text("{}")
    code = cli_main.main(
        [
            "eval",
            "sealed",
            "--model",
            "m",
            "--holdout-manifest",
            str(holdout),
            "--output",
            str(tmp_path / "o.json"),
            "--json",
        ]
    )
    assert code == 1
    assert (
        json.loads(capsys.readouterr().out)["error"]["code"]
        == "sealed_eval_service_unavailable"
    )


@pytest.mark.parametrize("stage", ["preflight", "evaluate"])
def test_service_network_access_is_blocked(monkeypatch, tmp_path, capsys, stage):
    import socket

    service = FakeService()

    def connect(*args):
        socket.create_connection(("example.invalid", 443))

    monkeypatch.setattr(service, stage, connect)
    code, out = _run(monkeypatch, tmp_path, service, "--json")
    assert code == 1
    assert not out.exists()
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == (
        "sealed_eval_preflight_failed" if stage == "preflight" else "sealed_eval_failed"
    )
    assert "example.invalid" not in error["message"]


@pytest.mark.parametrize("stage", ["preflight", "evaluate"])
def test_service_diagnostics_are_private(monkeypatch, tmp_path, capsys, stage):
    import sys

    from openmed.cli._output import CliError

    service = FakeService()

    def fail(*args):
        print(SECRET)
        print(SECRET, file=sys.stderr)
        raise CliError(SECRET, code=SECRET)

    monkeypatch.setattr(service, stage, fail)
    code, out = _run(monkeypatch, tmp_path, service, "--json")
    captured = capsys.readouterr()
    assert code == 1
    assert not out.exists()
    assert SECRET not in captured.out + captured.err
