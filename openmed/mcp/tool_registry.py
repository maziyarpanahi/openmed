"""Canonical MCP tool metadata for OpenMed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class StructuredOutputValidationError(ValueError):
    """Raised when a tool response does not match its declared output schema."""


@dataclass(frozen=True)
class MCPToolDefinition:
    """Metadata used to register and validate an OpenMed MCP tool."""

    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)


WORKFLOW_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "session_id",
        "workflow_id",
        "status",
        "handles",
        "final_handle",
        "final_output",
        "outputs",
        "trace",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "session_id": {"type": "string"},
        "workflow_id": {"type": "string"},
        "status": {"type": "string", "enum": ["completed", "failed"]},
        "handles": {"type": "object"},
        "final_handle": {"type": ["string", "null"]},
        "final_output": {},
        "outputs": {"type": "object"},
        "trace": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "step_id",
                    "tool",
                    "status",
                    "duration_ms",
                    "retry_count",
                ],
                "properties": {
                    "step_id": {"type": "string"},
                    "tool": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["completed", "failed", "resumed", "skipped"],
                    },
                    "duration_ms": {"type": "number"},
                    "retry_count": {"type": "integer"},
                    "attempt_count": {"type": "integer"},
                    "input_handles": {"type": "array"},
                    "output_handle": {"type": ["string", "null"]},
                    "error_type": {"type": "string"},
                    "resumed": {"type": "boolean"},
                },
            },
        },
    },
}


MCP_TOOL_REGISTRY: dict[str, MCPToolDefinition] = {
    definition.name: definition
    for definition in (
        MCPToolDefinition(
            name="openmed_analyze_text",
            description="Run OpenMed named-entity recognition on clinical text.",
        ),
        MCPToolDefinition(
            name="openmed_extract_pii",
            description="Extract PII/PHI entities from clinical text.",
        ),
        MCPToolDefinition(
            name="openmed_deidentify",
            description="De-identify text by masking, removing, replacing, hashing, or shifting.",
        ),
        MCPToolDefinition(
            name="openmed_list_models",
            description="List OpenMed model registry entries.",
        ),
        MCPToolDefinition(
            name="openmed_list_pii_languages",
            description="List supported PII languages and default models.",
        ),
        MCPToolDefinition(
            name="openmed_loaded_models",
            description="Return currently loaded model resources.",
        ),
        MCPToolDefinition(
            name="openmed_unload_model",
            description="Unload one inactive model, or all inactive models.",
        ),
        MCPToolDefinition(
            name="openmed_run_workflow",
            description=(
                "Run a stateful multi-step OpenMed workflow with server-side "
                "intermediate handles and PHI-safe egress."
            ),
            input_schema={
                "type": "object",
                "required": ["pipeline"],
                "properties": {
                    "pipeline": {"type": "object"},
                    "session_id": {"type": ["string", "null"]},
                    "workflow_id": {"type": ["string", "null"]},
                },
            },
            output_schema=WORKFLOW_RESULT_SCHEMA,
        ),
    )
}


def get_mcp_tool_definition(name: str) -> MCPToolDefinition:
    """Return the registered MCP tool definition for *name*."""
    try:
        return MCP_TOOL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown MCP tool: {name}") from exc


def validate_structured_output(tool_name: str, payload: Any) -> bool:
    """Validate *payload* against the registered structured-output schema."""
    schema = get_mcp_tool_definition(tool_name).output_schema
    if not schema:
        return True

    _validate_schema(schema, payload, path="$")
    return True


def _validate_schema(schema: Mapping[str, Any], value: Any, *, path: str) -> None:
    expected_type = schema.get("type")
    if expected_type is not None and not _matches_type(value, expected_type):
        raise StructuredOutputValidationError(
            f"{path} must be {_format_expected_type(expected_type)}"
        )

    if "enum" in schema and value not in schema["enum"]:
        raise StructuredOutputValidationError(f"{path} has an unsupported value")

    if isinstance(value, dict):
        for key in schema.get("required", ()):
            if key not in value:
                raise StructuredOutputValidationError(f"{path}.{key} is required")

        properties = schema.get("properties", {})
        if isinstance(properties, Mapping):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, Mapping):
                    _validate_schema(child_schema, value[key], path=f"{path}.{key}")

    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        item_schema = schema["items"]
        for index, item in enumerate(value):
            _validate_schema(item_schema, item, path=f"{path}[{index}]")


def _matches_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_type(value, item) for item in expected_type)

    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None

    return True


def _format_expected_type(expected_type: Any) -> str:
    if isinstance(expected_type, list):
        return " or ".join(str(item) for item in expected_type)
    return str(expected_type)
