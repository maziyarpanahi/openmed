from __future__ import annotations

import inspect
import json
from copy import deepcopy
from typing import Any

import pytest

from openmed.interop import adapter_tool_definitions, langchain, presidio
from openmed.mcp import server as mcp_server
from openmed.mcp.tool_registry import (
    TOOL_REGISTRY,
    ToolCompatibilityError,
    ToolRegistry,
    ToolSchemaValidationError,
    ToolSpec,
    check_tool_registry_compatibility,
    invoke_tool,
)


class FakeFastMCP:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}

    def tool(self, *, name: str):
        def _decorator(func):
            self.tools[name] = func
            return func

        return _decorator

    def resource(self, uri: str, **kwargs):
        del kwargs

        def _decorator(func):
            self.resources[uri] = func
            return func

        return _decorator


def test_mcp_registers_all_tools_from_registry() -> None:
    fake = FakeFastMCP()

    mcp_server._register_tools(fake, runtime_provider=None)

    expected = {spec.name for spec in TOOL_REGISTRY.latest_specs()}
    assert set(fake.tools) == expected
    assert len(fake.tools) == len(expected)

    deidentify_signature = inspect.signature(fake.tools["openmed_deidentify"])
    assert list(deidentify_signature.parameters) == [
        parameter.name
        for parameter in TOOL_REGISTRY.get("openmed_deidentify").parameters
    ]
    assert deidentify_signature.parameters["method"].default == "mask"


def test_registered_tool_invocation_validates_structured_output() -> None:
    spec = TOOL_REGISTRY.get("openmed_list_models")

    def bad_handler(**kwargs):
        del kwargs
        return {"count": "not-an-integer", "returned": 0, "models": []}

    with pytest.raises(ToolSchemaValidationError, match="openmed_list_models"):
        invoke_tool(spec, bad_handler, category=None, pii_language=None, limit=50)


def test_tool_registry_resource_is_generated_from_specs() -> None:
    fake = FakeFastMCP()

    mcp_server._register_resources(fake)
    payload = json.loads(fake.resources["openmed://tool-registry"]())

    assert payload["schema_version"] == "1.0.0"
    assert [tool["name"] for tool in payload["tools"]] == [
        spec.name for spec in TOOL_REGISTRY.all_specs()
    ]
    assert all(tool["version"] for tool in payload["tools"])
    assert all(tool["stability"] for tool in payload["tools"])


def test_clinical_tool_contracts_are_registered_and_versioned() -> None:
    expected = {
        "openmed_ground",
        "openmed_export_fhir",
        "openmed_risk_score",
        "openmed_clinical_pipeline",
    }

    for name in expected:
        spec = TOOL_REGISTRY.get(name)
        assert spec.version == "1.0.0"
        assert spec.input_schema["type"] == "object"
        assert spec.output_schema["type"] == "object"
        assert "schema_version" in spec.output_schema["required"]

    pipeline = TOOL_REGISTRY.get("openmed_clinical_pipeline")
    assert pipeline.input_schema["required"] == ["stages"]
    stages = pipeline.input_schema["properties"]["stages"]["items"]
    assert list(stages["enum"]) == [
        "detect",
        "context",
        "sections",
        "relations",
        "ground",
        "export",
        "risk",
    ]


def test_clinical_contract_handler_outputs_validate() -> None:
    span = _sample_openmed_span()

    assert mcp_server.openmed_ground([span])["status"] == "unimplemented"
    assert mcp_server.openmed_export_fhir([span])["status"] == "unimplemented"
    assert mcp_server.openmed_risk_score([span])["status"] == "unimplemented"
    assert mcp_server.openmed_clinical_pipeline(["detect", "ground"]) == {
        "schema_version": "openmed.clinical_pipeline.v1",
        "status": "planned",
        "stages": ["detect", "ground"],
        "artifacts": {},
        "final_output": None,
        "error": None,
        "trace": [],
    }


def test_adapter_tool_definitions_match_registry_schemas() -> None:
    langchain_defs = langchain.create_tool_definitions()
    presidio_defs = presidio.create_tool_definitions()
    registry_defs = adapter_tool_definitions("presidio")

    assert len(langchain_defs) == len(TOOL_REGISTRY.latest_specs())
    assert _schema_projection(langchain_defs) == _schema_projection(presidio_defs)
    assert _schema_projection(presidio_defs) == _schema_projection(registry_defs)
    assert _schema_projection(langchain_defs) == [
        {
            "name": spec.name,
            "input_schema": spec.input_schema,
            "output_schema": spec.output_schema,
        }
        for spec in TOOL_REGISTRY.latest_specs()
    ]


def test_breaking_schema_change_without_major_bump_fails() -> None:
    previous = TOOL_REGISTRY.latest_specs()
    target = TOOL_REGISTRY.get("openmed_analyze_text")
    input_schema = deepcopy(dict(target.input_schema))
    input_schema["properties"]["text"]["type"] = "integer"
    broken = _replace_spec(target, input_schema=input_schema)

    current = [broken if spec.name == broken.name else spec for spec in previous]

    with pytest.raises(ToolCompatibilityError, match="without version bump"):
        check_tool_registry_compatibility(previous, current)


def test_breaking_schema_change_with_major_bump_passes() -> None:
    previous = TOOL_REGISTRY.latest_specs()
    target = TOOL_REGISTRY.get("openmed_analyze_text")
    input_schema = deepcopy(dict(target.input_schema))
    input_schema["properties"]["text"]["type"] = "integer"
    bumped = _replace_spec(target, input_schema=input_schema, version="2.0.0")

    current = [bumped if spec.name == bumped.name else spec for spec in previous]

    check_tool_registry_compatibility(previous, current)


def test_breaking_new_clinical_tool_schema_change_is_caught() -> None:
    previous = TOOL_REGISTRY.latest_specs()
    target = TOOL_REGISTRY.get("openmed_clinical_pipeline")
    output_schema = deepcopy(dict(target.output_schema))
    output_schema["properties"] = deepcopy(dict(output_schema["properties"]))
    output_schema["properties"]["status"] = {
        "type": "string",
        "enum": ["completed"],
    }
    broken = _replace_spec(target, output_schema=output_schema)

    current = [broken if spec.name == broken.name else spec for spec in previous]

    with pytest.raises(ToolCompatibilityError, match="openmed_clinical_pipeline"):
        check_tool_registry_compatibility(previous, current)


def test_registry_supports_multiple_versions_side_by_side() -> None:
    original = TOOL_REGISTRY.get("openmed_analyze_text")
    bumped = _replace_spec(original, version="2.0.0")
    registry = ToolRegistry([original, bumped])

    assert registry.get("openmed_analyze_text", "1.0.0") == original
    assert registry.get("openmed_analyze_text", "2.0.0") == bumped
    assert registry.get("openmed_analyze_text") == bumped
    assert [tool["version"] for tool in registry.document()["tools"]] == [
        "1.0.0",
        "2.0.0",
    ]


def _replace_spec(
    spec: ToolSpec,
    *,
    input_schema: dict[str, Any] | None = None,
    output_schema: dict[str, Any] | None = None,
    version: str | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=spec.name,
        description=spec.description,
        input_schema=input_schema or spec.input_schema,
        output_schema=output_schema or spec.output_schema,
        version=version or spec.version,
        stability=spec.stability,
        parameters=spec.parameters,
    )


def _schema_projection(definitions: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": definition["name"],
            "input_schema": definition["input_schema"],
            "output_schema": definition["output_schema"],
        }
        for definition in definitions
    ]


def _sample_openmed_span() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "doc_id": "fixture-note",
        "start": 0,
        "end": 8,
        "text_hash": (
            "hmac-sha256:"
            "0123456789abcdef0123456789abcdef"
            "0123456789abcdef0123456789abcdef"
        ),
        "entity_type": "clinical_problem",
        "canonical_label": "OTHER",
        "policy_label": "CLINICAL_CONCEPT",
        "regulatory_tags": [],
        "score": 0.99,
        "detector": "fixture",
        "evidence": {},
        "action": "keep",
        "replacement": None,
        "reversible_id": None,
        "section": None,
        "metadata": {},
    }
