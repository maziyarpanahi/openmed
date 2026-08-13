"""Focused tests for the optional ARX subprocess bridge."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from openmed.interop.bridges import arx_bridge


def test_stubbed_runner_round_trips_policy_and_anonymized_rows(monkeypatch):
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        arx_bridge.shutil,
        "which",
        lambda candidate: f"/opt/{candidate}",
    )

    def fake_run(command, **kwargs):
        paths = {
            command[index]: Path(command[index + 1])
            for index in range(len(command) - 1)
            if command[index] in {"--input", "--config", "--output"}
        }
        observed["command"] = command
        observed["kwargs"] = kwargs
        observed["workspace"] = paths["--input"].parent
        observed["input_mode"] = paths["--input"].stat().st_mode & 0o777
        observed["config_mode"] = paths["--config"].stat().st_mode & 0o777
        observed["output_mode"] = paths["--output"].stat().st_mode & 0o777
        observed["records"] = json.loads(paths["--input"].read_text(encoding="utf-8"))
        observed["config"] = json.loads(paths["--config"].read_text(encoding="utf-8"))
        paths["--output"].write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "age": "40-49",
                            "region": "north",
                            "diagnosis": "A",
                        },
                        {
                            "age": "40-49",
                            "region": "north",
                            "diagnosis": "B",
                        },
                    ],
                    "metadata": {
                        "achieved_k": 2,
                        "target_l": 1,
                    },
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(arx_bridge.subprocess, "run", fake_run)
    records = [
        {"age": 41, "region": "north", "diagnosis": "A"},
        {"age": 42, "region": "north", "diagnosis": "B"},
    ]
    hierarchies = {
        "age": [
            {"name": "exact", "values": [41, 42]},
            {"name": "decade", "values": ["40-49"]},
        ]
    }

    result = arx_bridge.run_arx(
        records,
        quasi_identifiers=("age", "region"),
        sensitive_attributes=("diagnosis",),
        hierarchies=hierarchies,
        target_k=2,
        target_l=1,
        arx_command="approved-arx-runner",
    )

    assert observed["command"][0] == "/opt/approved-arx-runner"
    assert observed["command"][-6::2] == ["--input", "--config", "--output"]
    assert observed["kwargs"] == {
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": 60.0,
    }
    assert observed["input_mode"] == 0o600
    assert observed["config_mode"] == 0o600
    assert observed["output_mode"] == 0o600
    assert observed["records"] == records
    assert observed["config"] == {
        "hierarchies": hierarchies,
        "quasi_identifiers": ["age", "region"],
        "schema_version": 1,
        "sensitive_attributes": ["diagnosis"],
        "target_k": 2,
        "target_l": 1,
    }
    assert result.records == (
        {"age": "40-49", "region": "north", "diagnosis": "A"},
        {"age": "40-49", "region": "north", "diagnosis": "B"},
    )
    assert result.metadata == {"achieved_k": 2, "target_l": 1}
    assert result["records"] == list(result.records)
    assert not Path(observed["workspace"]).exists()


def test_missing_arx_runner_raises_actionable_not_available_error(monkeypatch):
    monkeypatch.delenv(arx_bridge.ARX_COMMAND_ENV, raising=False)

    with pytest.raises(
        arx_bridge.ARXNotAvailableError,
        match=arx_bridge.ARX_COMMAND_ENV,
    ):
        arx_bridge.run_arx(
            [{"age": 41, "diagnosis": "A"}],
            quasi_identifiers=("age",),
            sensitive_attributes=("diagnosis",),
        )


def test_runner_unavailable_sentinel_is_not_returned_as_anonymized_data(monkeypatch):
    monkeypatch.setattr(arx_bridge.shutil, "which", lambda _candidate: "/opt/arx")
    monkeypatch.setattr(
        arx_bridge.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            69,
            stdout="",
            stderr=f"{arx_bridge.ARX_UNAVAILABLE_SENTINEL}\n",
        ),
    )

    with pytest.raises(arx_bridge.ARXUnavailableError, match="ARX installation"):
        arx_bridge.run_arx(
            [{"age": 41, "diagnosis": "A"}],
            quasi_identifiers=("age",),
            sensitive_attributes=("diagnosis",),
            arx_command="arx-runner",
        )
