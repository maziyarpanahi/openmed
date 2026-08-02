"""Protocol-level tests for registry-rendered MCP tools."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

pytest.importorskip("mcp")

from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from openmed.mcp.server import create_mcp_server
from openmed.mcp.tool_registry import TOOL_REGISTRY
from openmed.service.security import MCPAuthorizationConfig


class _Runtime:
    def __init__(self, models: dict[str, Any] | None = None) -> None:
        self._models = models or {}

    def loaded_models(self) -> dict[str, Any]:
        return {"models": self._models}


def _server(runtime: _Runtime | None = None):
    selected = runtime or _Runtime()
    return create_mcp_server(runtime_provider=lambda: selected)


def test_every_tool_advertises_registry_schemas_and_annotations() -> None:
    server = _server()
    advertised = {tool.name: tool for tool in asyncio.run(server.list_tools())}

    assert set(advertised) == {spec.name for spec in TOOL_REGISTRY.latest_specs()}
    for spec in TOOL_REGISTRY.latest_specs():
        tool = advertised[spec.name]
        assert tool.title == spec.title
        assert tool.inputSchema == spec.input_schema
        assert tool.outputSchema == spec.mcp_output_schema()
        assert tool.annotations is not None
        assert tool.annotations.model_dump(exclude_none=True) == spec.annotations()


def test_structured_result_includes_machine_data_and_text_fallback() -> None:
    result = asyncio.run(
        _server(_Runtime(models={"model-a": {"active_requests": 0}})).call_tool(
            "openmed_loaded_models", {}
        )
    )

    assert result.isError is False
    assert result.structuredContent == {"models": {"model-a": {"active_requests": 0}}}
    assert json.loads(result.content[0].text) == result.structuredContent


def test_malformed_call_returns_phi_safe_structured_error_without_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    request_secret = "Patient Jane Secret has MRN-394-PRIVATE"
    response_secret = "loaded-model-PHI-response-secret"
    server = _server(_Runtime(models={response_secret: {}}))

    with caplog.at_level(logging.DEBUG):
        malformed = asyncio.run(
            server.call_tool(
                "openmed_deidentify",
                {"text": request_secret, "method": "unsupported-method"},
            )
        )
        successful = asyncio.run(server.call_tool("openmed_loaded_models", {}))

    assert malformed.isError is True
    assert malformed.structuredContent == {
        "error": {
            "code": "invalid_arguments",
            "message": "The tool arguments are invalid.",
        },
        "is_error": True,
    }
    assert json.loads(malformed.content[0].text) == malformed.structuredContent
    assert successful.isError is False
    assert response_secret in json.dumps(successful.structuredContent)
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert request_secret not in logged
    assert response_secret not in logged


def test_health_resource_reports_only_version_and_loaded_model_count() -> None:
    model_names = {
        "model-one": {"active_requests": 0},
        "model-two": {"active_requests": 1},
    }
    resources = asyncio.run(
        _server(_Runtime(models=model_names)).read_resource("openmed://health")
    )
    payload = json.loads(resources[0].content)

    assert set(payload) == {"loaded_model_count", "version"}
    assert payload["loaded_model_count"] == 2
    assert payload["version"]
    assert not set(model_names) & set(payload)


def test_remote_server_wires_protected_resource_auth_and_per_tool_scope() -> None:
    resource_url = "https://mcp.synthetic.test/mcp"
    config = MCPAuthorizationConfig(
        enabled=True,
        resource_url=resource_url,
        authorization_server_url="https://issuer.synthetic.test",
    )
    server = create_mcp_server(
        runtime_provider=lambda: _Runtime(models={}),
        authorization_config=config,
    )

    unauthorized = asyncio.run(server.call_tool("openmed_loaded_models", {}))
    assert unauthorized.isError is True
    assert unauthorized.structuredContent["error"]["code"] == "invalid_token"

    access_token = AccessToken(
        token="synthetic-access-token",
        client_id="synthetic-client",
        scopes=["mcp:tool:openmed_loaded_models"],
        resource=resource_url,
        claims={"aud": resource_url},
    )
    context_token = auth_context_var.set(AuthenticatedUser(access_token))
    try:
        authorized = asyncio.run(server.call_tool("openmed_loaded_models", {}))
    finally:
        auth_context_var.reset(context_token)
    assert authorized.isError is False

    routes = {
        getattr(route, "path", "") for route in server.streamable_http_app().routes
    }
    assert "/mcp" in routes
    assert "/.well-known/oauth-protected-resource/mcp" in routes
