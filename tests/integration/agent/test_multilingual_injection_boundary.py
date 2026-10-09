"""Multilingual lexical controls through the actual MCP dispatch wrapper."""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.agent.security.injection_guard import InjectionGuard
from openmed.mcp import server as mcp_server

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[3]
CORPUS = json.loads(
    (ROOT / "tests/fixtures/agent/multilingual-injection-cues.json").read_text()
)
LANGUAGES = tuple(CORPUS["linguistic_reviews"])


class _FakeMCP:
    def __init__(self):
        self._tool_manager = SimpleNamespace(
            get_tool=lambda name: SimpleNamespace(parameters={"type": "object"})
        )
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return arguments


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Synthetic boundary attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


def _payload(language, *, benign=False):
    rows = [
        row
        for row in CORPUS["cases"]
        if row["language"] == language and bool(row["pattern_ids"]) != benign
    ]
    return "\n".join(row["text"] for row in (rows[:1] if benign else rows))


@pytest.mark.parametrize("language", LANGUAGES)
def test_strict_mcp_dispatch_refuses_hostile_language_before_handler(
    language, monkeypatch
):
    monkeypatch.setattr(
        mcp_server,
        "_call_tool_result",
        lambda payload, *, is_error: {**payload, "is_error": is_error},
    )
    server = mcp_server._structured_fastmcp(_FakeMCP, InjectionGuard(mode="strict"))()
    text = _payload(language)
    result = asyncio.run(server.call_tool("synthetic_tool", {"text": text}))
    assert server.calls == []
    assert result["is_error"] is True
    serialized = json.dumps(result, ensure_ascii=False)
    assert text not in serialized
    assert all(
        row["text"] not in serialized
        for row in CORPUS["cases"]
        if row["language"] == language
    )
    assert {
        f"{category}.{language}"
        for category in (
            "instruction_override",
            "tool_name_spoofing",
            "data_exfiltration",
        )
    } <= {row["pattern_id"] for row in result["error"]["findings"]}
    assert all(
        set(row) == {"pattern_id", "start", "end", "severity"}
        for row in result["error"]["findings"]
    )


@pytest.mark.parametrize("language", LANGUAGES)
def test_allow_mcp_dispatch_quarantines_every_localized_cue(language):
    server = mcp_server._structured_fastmcp(_FakeMCP, InjectionGuard(mode="allow"))()
    text = _payload(language)
    result = asyncio.run(server.call_tool("synthetic_tool", {"text": text}))
    assert len(server.calls) == 1
    assert result["text"] != text
    assert "OPENMED_QUARANTINED_PROMPT_INJECTION" in result["text"]
    assert not InjectionGuard().scan(result["text"]).flagged


@pytest.mark.parametrize("language", LANGUAGES)
def test_benign_clinical_mcp_control_reaches_same_handler_unchanged(language):
    server = mcp_server._structured_fastmcp(_FakeMCP, InjectionGuard(mode="strict"))()
    text = _payload(language, benign=True)
    result = asyncio.run(server.call_tool("synthetic_tool", {"text": text}))
    assert result == {"text": text}
    assert len(server.calls) == 1
