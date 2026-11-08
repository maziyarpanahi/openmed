"""Tests for the OpenMed MCP console entry point."""

from __future__ import annotations

import json
import sys
from importlib import metadata
from typing import Any

import pytest

from openmed.mcp import server


def test_console_entry_point_resolves() -> None:
    entry_point = next(
        item
        for item in metadata.entry_points(group="console_scripts")
        if item.name == "openmed-mcp"
    )

    assert entry_point.value == "openmed.mcp.server:main"
    assert entry_point.load() is server.main


@pytest.mark.parametrize(
    ("requested_transport", "expected_transport"),
    [("stdio", "stdio"), ("streamable-http", "streamable-http")],
)
def test_console_entry_point_runs_transport(
    monkeypatch: pytest.MonkeyPatch,
    requested_transport: str,
    expected_transport: str,
) -> None:
    calls: dict[str, Any] = {}

    class FakeServer:
        def run(self, *, transport: str) -> None:
            calls["transport"] = transport

    def fake_create_mcp_server(**kwargs: Any) -> FakeServer:
        calls["server_kwargs"] = kwargs
        return FakeServer()

    monkeypatch.setattr(server, "create_mcp_server", fake_create_mcp_server)

    exit_code = server.main(
        [
            "--transport",
            requested_transport,
            "--host",
            "127.0.0.1",
            "--port",
            "9081",
            "--streamable-http-path",
            "/openmed-mcp",
        ]
    )

    assert exit_code == 0
    assert calls == {
        "server_kwargs": {
            "host": "127.0.0.1",
            "port": 9081,
            "streamable_http_path": "/openmed-mcp",
        },
        "transport": expected_transport,
    }


def test_list_tools_prints_every_registered_tool(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_create_mcp_server(**kwargs: Any) -> None:
        raise AssertionError("--list-tools must not create the MCP server")

    monkeypatch.setattr(server, "create_mcp_server", fail_create_mcp_server)

    exit_code = server.main(["--list-tools"])

    output = capsys.readouterr().out
    assert exit_code == 0
    for spec in server.TOOL_REGISTRY.latest_specs():
        assert spec.name in output
        assert spec.title in output
        assert f"v{spec.version}" in output
    assert "readOnlyHint=" in output


def test_list_tools_json_has_the_four_hints_per_tool(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = server.main(["--list-tools", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert [entry["name"] for entry in payload] == [
        spec.name for spec in server.TOOL_REGISTRY.latest_specs()
    ]
    for entry in payload:
        assert set(entry) == {
            "name",
            "title",
            "version",
            "readOnlyHint",
            "destructiveHint",
            "idempotentHint",
            "openWorldHint",
        }
        for hint in (
            "readOnlyHint",
            "destructiveHint",
            "idempotentHint",
            "openWorldHint",
        ):
            assert isinstance(entry[hint], bool)


def test_list_tools_works_without_the_mcp_package(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in [key for key in sys.modules if key == "mcp" or key.startswith("mcp.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "mcp", None)

    exit_code = server.main(["--list-tools", "--json"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)


def test_json_requires_list_tools() -> None:
    with pytest.raises(SystemExit) as excinfo:
        server.main(["--json"])

    assert excinfo.value.code == 2
