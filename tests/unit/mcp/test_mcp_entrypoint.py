"""Tests for the OpenMed MCP console entry point."""

from __future__ import annotations

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
