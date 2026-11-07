"""Focused tests for the optional JSON-over-stdin ARX bridge."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from openmed.interop.bridges.arx import (
    ArxBridge,
    ArxBridgeError,
    ArxNotAvailableError,
)


def test_absent_arx_adapter_fails_with_actionable_fallback() -> None:
    bridge = ArxBridge()

    assert bridge.available is False
    with pytest.raises(ArxNotAvailableError, match="pure-Python"):
        bridge.anonymize(
            [{"age": 31}],
            quasi_identifiers={"age": "age"},
        )


def test_stubbed_arx_adapter_round_trips_config_over_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = [
        {"age": 31, "condition": "A"},
        {"age": 32, "condition": "B"},
    ]

    def fake_run(command, **kwargs):
        assert command == (sys.executable, "synthetic-arx-adapter")
        assert kwargs["stderr"] is subprocess.DEVNULL
        assert "shell" not in kwargs
        assert set(kwargs["env"]) <= {
            "JAVA_HOME",
            "LANG",
            "LC_ALL",
            "PATH",
            "SYSTEMROOT",
            "TMPDIR",
        }
        request = json.loads(kwargs["input"])
        assert request["privacy"] == {"k": 2, "l": 2, "t": 0.3}
        assert request["quasi_identifiers"] == {"age": "age"}
        assert request["sensitive_attributes"] == ["condition"]
        response = {
            "records": [
                {"age": "30-34", "condition": "A"},
                {"age": "30-34", "condition": "B"},
            ],
            "generalization_levels": {"age": 1},
            "suppressed_count": 0,
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(response).encode(),
        )

    monkeypatch.setattr("openmed.interop.bridges.arx.subprocess.run", fake_run)
    bridge = ArxBridge(command=(sys.executable, "synthetic-arx-adapter"))

    result = bridge.anonymize(
        source,
        quasi_identifiers={"age": "age"},
        sensitive_attributes=["condition"],
        k=2,
        l=2,
        t=0.3,
    )

    assert result.records[0]["age"] == "30-34"
    assert result.manifest["engine"] == "arx"
    assert result.manifest["generalization_levels"] == {"age": 1}
    assert "condition" not in result.manifest


def test_adapter_failure_does_not_echo_source_or_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_canary = "synthetic-private-canary"

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            7,
            stdout=f"failed: {source_canary}".encode(),
        )

    monkeypatch.setattr("openmed.interop.bridges.arx.subprocess.run", fake_run)
    bridge = ArxBridge(command=(sys.executable, "synthetic-arx-adapter"))

    with pytest.raises(ArxBridgeError) as raised:
        bridge.anonymize(
            [{"age": 31, "note": source_canary}],
            quasi_identifiers={"age": "age"},
        )

    assert source_canary not in str(raised.value)
