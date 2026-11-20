"""The example validates offline and emits no payload values."""

import runpy
import socket
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry
from referencing.exceptions import Unresolvable

from examples.private_training_schema_validation import reject_remote_resolution


def test_example_runs_with_network_blocked(monkeypatch, capsys):
    def blocked(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    script = (
        Path(__file__).resolve().parents[3]
        / "examples/private_training_schema_validation.py"
    )
    runpy.run_path(str(script), run_name="__main__")
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "federated_round_lifecycle: valid",
        "federated_round_schedule: valid",
        "federated_round_status: valid",
        "federated_aggregate_metric: valid",
        "federated_aggregate_metric: rejected",
    ]
    assert captured.err == ""


def test_registry_refuses_external_references():
    validator = Draft202012Validator(
        {"$ref": "https://example.invalid/schema"},
        registry=Registry(retrieve=reject_remote_resolution),
    )
    with pytest.raises(Unresolvable):
        validator.validate({})
