"""Protocol-level tests for registry-rendered MCP tools."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

pytest.importorskip("mcp")

from openmed.mcp.server import create_mcp_server
from openmed.mcp.tool_registry import CLINICAL_WORKFLOW_SPEC, TOOL_REGISTRY


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


def test_clinical_workflow_prompt_and_resources_are_exposed_over_mcp() -> None:
    server = _server()
    prompts = asyncio.run(server.list_prompts())
    resources = asyncio.run(server.list_resources())

    assert CLINICAL_WORKFLOW_SPEC.prompt_name in {prompt.name for prompt in prompts}
    resource_uris = {str(resource.uri) for resource in resources}
    assert CLINICAL_WORKFLOW_SPEC.resource_uri in resource_uris
    assert CLINICAL_WORKFLOW_SPEC.fixture_uri in resource_uris

    workflow_contents = asyncio.run(
        server.read_resource(CLINICAL_WORKFLOW_SPEC.resource_uri)
    )
    fixture_contents = asyncio.run(
        server.read_resource(CLINICAL_WORKFLOW_SPEC.fixture_uri)
    )
    workflow = json.loads(workflow_contents[0].content)
    fixture = json.loads(fixture_contents[0].content)

    assert workflow["execution"]["default"] == "local"
    assert workflow["execution"]["network_egress"] is False
    assert fixture["synthetic"] is True
    assert fixture["artifacts"]["export"]["bundle"]["resourceType"] == "Bundle"
