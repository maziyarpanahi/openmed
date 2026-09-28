"""Draft 2020-12 schema export for PHI-safe tool inventories.

The exported schema is self-contained and uses only fragment-local references.
It describes the public ``ToolInventory`` and ``ToolInventoryRecord`` JSON
projections without importing a validator or contacting a schema registry.
"""

from __future__ import annotations

import json
from typing import Any, Final

from openmed.agent.tool_inventory import (
    MAX_TOOL_INVENTORY_RECORDS,
    MAX_TOOL_VERSION_LENGTH,
    TOOL_INVENTORY_SCHEMA_VERSION,
    SideEffectClass,
)

TOOL_INVENTORY_SCHEMA_DIALECT: Final = "https://json-schema.org/draft/2020-12/schema"

_MAX_GOVERNANCE_IDENTIFIER_LENGTH: Final = 512
_LABEL_PATTERN = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_NAMESPACE_PATTERN = rf"{_LABEL_PATTERN}(?:\.{_LABEL_PATTERN})+"
_LOCAL_NAME_PATTERN = r"[a-z][a-z0-9-]{0,63}"
_NUMBER_PATTERN = r"(?:0|[1-9][0-9]*)"
_VERSION_PATTERN = rf"{_NUMBER_PATTERN}\.{_NUMBER_PATTERN}\.{_NUMBER_PATTERN}"
_TOOL_ID_PATTERN = rf"^tool:{_NAMESPACE_PATTERN}/{_LOCAL_NAME_PATTERN}$"
_CAPABILITY_ID_PATTERN = (
    rf"^capability:{_NAMESPACE_PATTERN}/{_LOCAL_NAME_PATTERN}"
    rf"(?:@{_VERSION_PATTERN})?$"
)
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


def build_tool_inventory_schema() -> dict[str, Any]:
    """Build the self-contained JSON Schema for tool inventory payloads.

    Returns:
        A new JSON-compatible Draft 2020-12 schema mapping. Mutating the
        returned mapping cannot affect a later export.
    """

    return {
        "$schema": TOOL_INVENTORY_SCHEMA_DIALECT,
        "$defs": {
            "tool_inventory_record": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "tool_id",
                    "version",
                    "capability_class",
                    "side_effect_class",
                    "schema_digest",
                ],
                "properties": {
                    "tool_id": {
                        "type": "string",
                        "maxLength": _MAX_GOVERNANCE_IDENTIFIER_LENGTH,
                        "pattern": _TOOL_ID_PATTERN,
                    },
                    "version": {
                        "type": "string",
                        "maxLength": MAX_TOOL_VERSION_LENGTH,
                        "pattern": rf"^{_VERSION_PATTERN}$",
                    },
                    "capability_class": {
                        "type": "string",
                        "maxLength": _MAX_GOVERNANCE_IDENTIFIER_LENGTH,
                        "pattern": _CAPABILITY_ID_PATTERN,
                    },
                    "side_effect_class": {
                        "type": "string",
                        "enum": [item.value for item in SideEffectClass],
                    },
                    "schema_digest": {
                        "type": "string",
                        "minLength": 71,
                        "maxLength": 71,
                        "pattern": _DIGEST_PATTERN,
                    },
                },
            }
        },
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "tools"],
        "properties": {
            "schema_version": {
                "type": "string",
                "const": TOOL_INVENTORY_SCHEMA_VERSION,
            },
            "tools": {
                "type": "array",
                "maxItems": MAX_TOOL_INVENTORY_RECORDS,
                "uniqueItems": True,
                "items": {"$ref": "#/$defs/tool_inventory_record"},
            },
        },
    }


def render_tool_inventory_schema() -> str:
    """Render the tool inventory schema as byte-stable compact JSON."""

    return json.dumps(
        build_tool_inventory_schema(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "TOOL_INVENTORY_SCHEMA_DIALECT",
    "build_tool_inventory_schema",
    "render_tool_inventory_schema",
]
