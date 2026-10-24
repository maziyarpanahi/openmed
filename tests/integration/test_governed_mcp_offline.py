"""Actual in-memory MCP sessions and default hostile boundary attempts."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from mcp.shared.memory import create_connected_server_and_client_session

from openmed.agent.security.adversarial import DEFAULT_ADVERSARIAL_FIXTURES
from openmed.mcp.server import create_mcp_server

pytestmark = pytest.mark.integration
_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "governed_mcp_synthetic_helpers",
    _ROOT / "tests/unit/mcp/test_governed_workflows.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)


@pytest.fixture
def offline_loop(monkeypatch):
    # ProactorEventLoop creates its internal socketpair via loopback on Windows.
    # All application/session work still runs with every connection denied.
    loop = asyncio.new_event_loop()

    def denied(*args, **kwargs):
        raise AssertionError("Offline MCP session attempted network access")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    try:
        yield loop
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()


def test_actual_sdk_session_preflights_previews_requests_review_and_polls(offline_loop):
    service = _HELPERS._Service()
    policy, provider = _HELPERS._consent()
    server = create_mcp_server(
        governance_service=service,
        governance_consent_policy=policy,
        governance_consent_receipt_provider=provider,
        governance_clock=lambda: _HELPERS.NOW,
    )

    async def session():
        async with create_connected_server_and_client_session(server) as client:
            tools = await client.list_tools()
            assert "openmed_workflow_request_review" in {t.name for t in tools.tools}
            for operation in ("preflight", "preview"):
                result = await client.call_tool(
                    "openmed_workflow_" + operation,
                    {"request": _HELPERS._request().to_dict()},
                )
                assert (
                    not result.isError and result.structuredContent["status"] == "ready"
                )
            result = await client.call_tool(
                "openmed_workflow_request_review",
                {"request": _HELPERS._request().to_dict()},
            )
            assert (
                not result.isError
                and result.structuredContent["status"] == "review_required"
            )
            result = await client.call_tool(
                "openmed_workflow_status", {"request": _HELPERS._request().to_dict()}
            )
            assert (
                not result.isError
                and result.structuredContent["receipt_verification"] == "absent"
            )
            assert "signature" not in json.dumps(result.model_dump(mode="json"))

    offline_loop.run_until_complete(session())
    assert service.calls == [
        "preflight",
        "preview",
        "status",
        "request_review",
        "status",
    ]
    assert service.handoffs == 1 and service.clinical_effects == 0


def _plain(value):
    if hasattr(value, "items"):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


@pytest.mark.parametrize(
    "fixture", DEFAULT_ADVERSARIAL_FIXTURES, ids=lambda f: f.case_id
)
@pytest.mark.parametrize("guard_mode", ["strict", "allow"])
def test_default_boundary_attempts_use_actual_mcp_dispatch(
    fixture, guard_mode, caplog, offline_loop
):
    service = _HELPERS._Service()
    request_data = _HELPERS._request().to_dict()
    payload = _plain(fixture.payload)
    # The benign proposal is pre-registered in service custody. Only its digest
    # crosses MCP. Hostile attempts try to supplement that reference with the
    # original fixture fields, which must not become caller-supplied authority.
    benign = (
        set(payload) == {"operation", "requested_fields", "content"}
        and payload["operation"] == "summarize"
        and payload["requested_fields"] == ["summary"]
    )
    if benign:
        digest = (
            "sha256:"
            + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        )
        request_data["action_digest"] = digest
        service.view = replace(service.view, action_digest=digest)
    else:
        request_data.update(payload)
    server = create_mcp_server(
        governance_service=service, injection_guard_mode=guard_mode
    )

    async def session():
        async with create_connected_server_and_client_session(server) as client:
            return await client.call_tool(
                "openmed_workflow_preflight", {"request": request_data}
            )

    result = offline_loop.run_until_complete(session())
    assert result.isError is (not benign)
    assert service.calls == (["preflight"] if benign else [])
    assert service.handoffs == service.clinical_effects == 0
    assert "synthetic-secret-canary" not in json.dumps(result.model_dump(mode="json"))
    assert "synthetic-secret-canary" not in caplog.text
    if not benign:
        assert result.structuredContent["error"]["code"] in {
            "invalid_arguments",
            "prompt_injection_detected",
        }


def test_offline_session_loop_still_denies_connections(offline_loop):
    async def attempt():
        with pytest.raises(
            AssertionError, match="Offline MCP session attempted network access"
        ):
            socket.create_connection(("127.0.0.1", 9))
        with socket.socket() as client:
            for operation in (client.connect, client.connect_ex):
                with pytest.raises(
                    AssertionError, match="Offline MCP session attempted network access"
                ):
                    operation(("127.0.0.1", 9))

    offline_loop.run_until_complete(attempt())
