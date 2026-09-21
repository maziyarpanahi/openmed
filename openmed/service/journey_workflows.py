"""Canonical Journey workflow contracts shared by MCP and typed clients."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from importlib import resources
from typing import Any, Final

from .journey_resources import (
    JOURNEY_RESOURCE_COMPATIBILITY,
    JOURNEY_RESOURCE_SCHEMA_VERSION,
    MAX_PAGE_SIZE,
    MAX_SELECTED_FIELDS,
    RESOURCE_FIELDS,
    JourneyAccessPolicy,
    JourneyResourceCatalog,
    JourneyResourceKind,
    JourneyResourceQuery,
    JourneyResourceState,
    parse_resource_fields,
)

JOURNEY_WORKFLOW_SCHEMA_VERSION: Final = "1.0.0"
JOURNEY_WORKFLOW_COMPATIBILITY: Final = JOURNEY_RESOURCE_COMPATIBILITY
_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"
_PAGE_SCHEMA_NAME: Final = "journey_resource_page.schema.json"
_CONTROLLED_PATTERN: Final = r"^[a-z][a-z0-9_.:/-]{0,127}$"
_OPAQUE_ID_PATTERN: Final = r"^[a-z][a-z0-9_]{0,31}_[A-Za-z0-9_-]{8,128}$"


@dataclass(frozen=True, slots=True)
class JourneyWorkflowDefinition:
    """One workflow name mapped to its tool and typed-client methods."""

    name: str
    resource_type: JourneyResourceKind
    tool_name: str
    python_method: str
    typescript_method: str
    title: str
    description: str


JOURNEY_WORKFLOW_DEFINITIONS: Final = (
    JourneyWorkflowDefinition(
        name="journey",
        resource_type=JourneyResourceKind.JOURNEY,
        tool_name="openmed_read_journey",
        python_method="journey",
        typescript_method="journey",
        title="Read Patient Journey",
        description=(
            "Read a bounded, evidence-linked patient journey projection without "
            "returning raw source text."
        ),
    ),
    JourneyWorkflowDefinition(
        name="cohort",
        resource_type=JourneyResourceKind.COHORT,
        tool_name="openmed_read_cohort",
        python_method="cohort",
        typescript_method="cohort",
        title="Read Cohort",
        description=(
            "Read bounded cohort definitions and membership summaries under "
            "minimum-necessary field policy."
        ),
    ),
    JourneyWorkflowDefinition(
        name="dataset",
        resource_type=JourneyResourceKind.DATASET,
        tool_name="openmed_read_dataset",
        python_method="dataset",
        typescript_method="dataset",
        title="Read Dataset Snapshot",
        description=(
            "Read bounded governed-dataset snapshot metadata without exporting "
            "underlying clinical source text."
        ),
    ),
    JourneyWorkflowDefinition(
        name="registry",
        resource_type=JourneyResourceKind.REGISTRY,
        tool_name="openmed_read_registry",
        python_method="registry",
        typescript_method="registry",
        title="Read Clinical Registry",
        description=(
            "Read bounded registry definitions and case-count metadata with an "
            "inspectable access-policy result."
        ),
    ),
    JourneyWorkflowDefinition(
        name="measure",
        resource_type=JourneyResourceKind.MEASURE,
        tool_name="openmed_read_measure",
        python_method="measure",
        typescript_method="measure",
        title="Read Measure Results",
        description=(
            "Read bounded deterministic measure results with snapshot and review "
            "metadata."
        ),
    ),
    JourneyWorkflowDefinition(
        name="trial_review",
        resource_type=JourneyResourceKind.TRIAL_REVIEW,
        tool_name="openmed_read_trial_review",
        python_method="trial_review",
        typescript_method="trialReview",
        title="Read Trial Review",
        description=(
            "Read bounded clinical-trial review results while preserving unknown, "
            "conflict, and review-required states."
        ),
    ),
)

JOURNEY_WORKFLOW_BY_NAME: Final = {
    item.name: item for item in JOURNEY_WORKFLOW_DEFINITIONS
}
JOURNEY_WORKFLOW_BY_TOOL: Final = {
    item.tool_name: item for item in JOURNEY_WORKFLOW_DEFINITIONS
}


def load_journey_resource_page_schema() -> dict[str, Any]:
    """Load the bundled canonical Journey page schema."""

    resource = resources.files(_SCHEMA_PACKAGE).joinpath(_PAGE_SCHEMA_NAME)
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def journey_workflow_query_properties(
    definition: JourneyWorkflowDefinition,
) -> dict[str, dict[str, Any]]:
    """Return query-property schemas for one fixed-resource workflow."""

    return {
        "namespace": {
            "type": "string",
            "pattern": _CONTROLLED_PATTERN,
            "default": "default",
            "description": "Controlled namespace to read.",
        },
        "purpose": {
            "type": "string",
            "pattern": _CONTROLLED_PATTERN,
            "default": "care_review",
            "description": "Controlled purpose used by the access policy.",
        },
        "first": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_PAGE_SIZE,
            "default": 20,
            "description": "Maximum resources to return in this bounded page.",
        },
        "after": {
            "type": ["string", "null"],
            "maxLength": 2048,
            "default": None,
            "description": ("Opaque cursor bound to the query and immutable snapshot."),
        },
        "fields": {
            "type": ["array", "null"],
            "maxItems": MAX_SELECTED_FIELDS,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "enum": sorted(RESOURCE_FIELDS[definition.resource_type]),
            },
            "default": None,
            "description": "Optional minimum-necessary field projection.",
        },
    }


def journey_workflow_input_schema(
    definition: JourneyWorkflowDefinition,
) -> dict[str, Any]:
    """Build the canonical input schema for a fixed-resource workflow."""

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": journey_workflow_query_properties(definition),
        "required": [],
    }


def journey_workflow_output_schema(
    definition: JourneyWorkflowDefinition,
) -> dict[str, Any]:
    """Extend the canonical page schema with agent-facing safety metadata."""

    schema = load_journey_resource_page_schema()
    schema["$id"] = (
        "https://openmed.dev/schemas/journey-workflow-result/"
        f"{JOURNEY_WORKFLOW_SCHEMA_VERSION}/{definition.name}"
    )
    schema["title"] = f"OpenMed {definition.title} Result"
    properties = schema["properties"]
    properties.update(
        {
            "workflow": {"const": definition.name},
            "resource_type": {"const": definition.resource_type.value},
            "evidence": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "resource_ids",
                    "resource_count",
                    "snapshot_digest",
                ],
                "properties": {
                    "resource_ids": {
                        "type": "array",
                        "maxItems": MAX_PAGE_SIZE,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "pattern": _OPAQUE_ID_PATTERN,
                        },
                    },
                    "resource_count": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": MAX_PAGE_SIZE,
                    },
                    "snapshot_digest": {
                        "type": "string",
                        "pattern": r"^sha256:[0-9a-f]{64}$",
                    },
                },
            },
            "snapshot": {
                "type": "object",
                "additionalProperties": False,
                "required": ["digest", "immutable", "schema_version"],
                "properties": {
                    "digest": {
                        "type": "string",
                        "pattern": r"^sha256:[0-9a-f]{64}$",
                    },
                    "immutable": {"type": "boolean", "const": True},
                    "schema_version": {
                        "type": "string",
                        "const": JOURNEY_RESOURCE_SCHEMA_VERSION,
                    },
                },
            },
            "access": {"$ref": "#/$defs/policy"},
            "warnings": {
                "type": "array",
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["code", "severity"],
                    "properties": {
                        "code": {
                            "type": "string",
                            "pattern": _CONTROLLED_PATTERN,
                        },
                        "severity": {"const": "warning"},
                    },
                },
            },
            "review": {
                "type": "object",
                "additionalProperties": False,
                "required": ["required", "reasons"],
                "properties": {
                    "required": {"type": "boolean"},
                    "reasons": {
                        "type": "array",
                        "maxItems": 16,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "pattern": _CONTROLLED_PATTERN,
                        },
                    },
                },
            },
        }
    )
    schema["required"].extend(
        (
            "workflow",
            "resource_type",
            "evidence",
            "snapshot",
            "access",
            "warnings",
            "review",
        )
    )
    definitions = schema.pop("$defs")
    return _inline_local_schema_refs(schema, definitions)


def _inline_local_schema_refs(value: Any, definitions: Mapping[str, Any]) -> Any:
    """Inline bundled local definitions for strict MCP schema generators."""

    if isinstance(value, Mapping):
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            name = reference.removeprefix("#/$defs/")
            if name not in definitions:
                raise ValueError(
                    "Journey workflow schema contains an unknown reference"
                )
            resolved = deepcopy(definitions[name])
            resolved.update({key: item for key, item in value.items() if key != "$ref"})
            return _inline_local_schema_refs(resolved, definitions)
        return {
            str(key): _inline_local_schema_refs(item, definitions)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_inline_local_schema_refs(item, definitions) for item in value]
    return deepcopy(value)


def execute_journey_workflow(
    definition: JourneyWorkflowDefinition,
    *,
    catalog: JourneyResourceCatalog,
    policy: JourneyAccessPolicy | None = None,
    namespace: str = "default",
    purpose: str = "care_review",
    first: int = 20,
    after: str | None = None,
    fields: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run one fixed-resource workflow and attach safe agent metadata."""

    page = catalog.list_resources(
        JourneyResourceQuery(
            resource_type=definition.resource_type,
            namespace=namespace,
            purpose=purpose,
            first=first,
            after=after,
            fields=parse_resource_fields(fields),
        ),
        policy=policy,
    )
    payload = page.to_dict()
    resource_ids = list(
        dict.fromkeys(str(item["resource_id"]) for item in payload["resources"])
    )
    snapshot_digest = str(payload["page_info"]["snapshot_digest"])
    warning_codes = [] if payload["code"] is None else [str(payload["code"])]
    review_states = {
        JourneyResourceState.PARTIAL.value,
        JourneyResourceState.UNKNOWN.value,
        JourneyResourceState.CONFLICT.value,
        JourneyResourceState.FAILURE.value,
    }
    resource_review_required = any(
        isinstance(item.get("data"), Mapping)
        and item["data"].get("status")
        in {"conflict", "pending_review", "review_required", "unknown"}
        for item in payload["resources"]
    )
    review_required = payload["state"] in review_states or resource_review_required
    if review_required and not warning_codes:
        warning_codes = ["resource_review_required"]
    review_reasons = warning_codes if review_required else []
    payload.update(
        {
            "workflow": definition.name,
            "resource_type": definition.resource_type.value,
            "evidence": {
                "resource_ids": resource_ids,
                "resource_count": len(resource_ids),
                "snapshot_digest": snapshot_digest,
            },
            "snapshot": {
                "digest": snapshot_digest,
                "immutable": True,
                "schema_version": JOURNEY_RESOURCE_SCHEMA_VERSION,
            },
            "access": deepcopy(payload["policy"]),
            "warnings": [
                {"code": code, "severity": "warning"} for code in warning_codes
            ],
            "review": {
                "required": review_required,
                "reasons": review_reasons,
            },
        }
    )
    return payload


def journey_workflow_client_contract() -> dict[str, Any]:
    """Return the client-generation document derived from workflow schemas."""

    page_schema = load_journey_resource_page_schema()
    return {
        "schema_version": JOURNEY_WORKFLOW_SCHEMA_VERSION,
        "compatibility_policy": JOURNEY_WORKFLOW_COMPATIBILITY,
        "endpoint": "/v1/journey/resources",
        "page_schema": page_schema,
        "workflows": [
            {
                "name": definition.name,
                "resource_type": definition.resource_type.value,
                "tool_name": definition.tool_name,
                "python_method": definition.python_method,
                "typescript_method": definition.typescript_method,
                "input_schema": journey_workflow_input_schema(definition),
                "output_schema": journey_workflow_output_schema(definition),
            }
            for definition in JOURNEY_WORKFLOW_DEFINITIONS
        ],
    }


def render_python_journey_client() -> str:
    """Render the generated Python typed-client mixin."""

    resource_type_lines = [
        "JourneyResourceType = Literal[",
        *(f'    "{item.value}",' for item in JourneyResourceKind),
        "]",
    ]
    workflow_names = ", ".join(
        f'"{item.name}"' for item in JOURNEY_WORKFLOW_DEFINITIONS
    )
    lines = [
        '"""Generated Journey workflow client methods. Do not edit manually."""',
        "",
        "from __future__ import annotations",
        "",
        "from collections.abc import Sequence",
        "from typing import Any, Literal, Optional",
        "",
        *resource_type_lines,
        "JourneyWorkflowName = Literal[",
        f"    {workflow_names}",
        "]",
        "",
        "",
        "class JourneyWorkflowClientMixin:",
        '    """Generated convenience methods for fixed-resource workflows."""',
        "",
        "    def journey_resources(",
        "        self,",
        "        resource_type: JourneyResourceType,",
        "        *,",
        '        namespace: str = "default",',
        '        purpose: str = "care_review",',
        "        first: int = 20,",
        "        after: Optional[str] = None,",
        "        fields: Sequence[str] = (),",
        "        request_id: Optional[str] = None,",
        "    ) -> dict[str, Any]:",
        '        """Implemented by the concrete REST client."""',
        "",
        "        raise NotImplementedError",
    ]
    for definition in JOURNEY_WORKFLOW_DEFINITIONS:
        lines.extend(
            [
                "",
                f"    def {definition.python_method}(",
                "        self,",
                "        *,",
                '        namespace: str = "default",',
                '        purpose: str = "care_review",',
                "        first: int = 20,",
                "        after: Optional[str] = None,",
                "        fields: Sequence[str] = (),",
                "        request_id: Optional[str] = None,",
                "    ) -> dict[str, Any]:",
                f'        """Read the {definition.name} workflow page."""',
                "",
                "        return self.journey_resources(",
                f'            "{definition.resource_type.value}",',
                "            namespace=namespace,",
                "            purpose=purpose,",
                "            first=first,",
                "            after=after,",
                "            fields=fields,",
                "            request_id=request_id,",
                "        )",
            ]
        )
    lines.extend(
        [
            "",
            "",
            '__all__ = ["JourneyResourceType", "JourneyWorkflowClientMixin", '
            '"JourneyWorkflowName"]',
            "",
        ]
    )
    return "\n".join(lines)


def render_typescript_journey_client() -> str:
    """Render TypeScript types and fixed workflow metadata from the schema."""

    page_schema = load_journey_resource_page_schema()
    resource_types = page_schema["$defs"]["resource"]["properties"]["resource_type"][
        "enum"
    ]
    states = page_schema["properties"]["state"]["enum"]
    resource_union = "\n".join(f'  | "{item}"' for item in resource_types)
    state_union = "\n".join(f'  | "{item}"' for item in states)
    workflow_union = "\n".join(
        f'  | "{item.name}"' for item in JOURNEY_WORKFLOW_DEFINITIONS
    )
    mapping = "\n".join(
        f'  {item.name}: "{item.resource_type.value}",'
        for item in JOURNEY_WORKFLOW_DEFINITIONS
    )
    return f"""// Generated from the canonical Journey workflow registry. Do not edit.

export type JourneyResourceType =
{resource_union};

export type JourneyResourceState =
{state_union};

export type JourneyWorkflowName =
{workflow_union};

export interface JourneyResourceQuery {{
  resource_type: JourneyResourceType;
  namespace?: string;
  purpose?: string;
  first?: number;
  after?: string | null;
  fields?: string[];
}}

export type JourneyWorkflowQuery = Omit<JourneyResourceQuery, "resource_type">;

export interface JourneyResourcePage {{
  state: JourneyResourceState;
  code: string | null;
  resources: Array<{{
    resource_type: JourneyResourceType;
    resource_id: string;
    namespace: string;
    data: Record<string, unknown>;
    state: JourneyResourceState;
    version: number;
    revision: number;
    schema_version: string;
    compatibility_policy: "same_major";
    extensions: Record<string, unknown>;
  }}>;
  page_info: {{
    has_next_page: boolean;
    end_cursor: string | null;
    page_size: number;
    snapshot_digest: string;
  }};
  policy: {{
    state: "success" | "denied";
    namespace: string;
    purpose: string;
    allowed_fields: string[];
    code: string | null;
    policy_version: string;
  }};
  schema_version: string;
  compatibility_policy: "same_major";
}}

export const JOURNEY_WORKFLOW_RESOURCE_TYPES = {{
{mapping}
}} as const satisfies Record<JourneyWorkflowName, JourneyResourceType>;
"""


def render_journey_workflow_contract_json() -> str:
    """Render the stable machine-readable client-generation contract."""

    return (
        json.dumps(
            journey_workflow_client_contract(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def assert_no_raw_source_fields(payload: Mapping[str, Any]) -> None:
    """Reject accidental raw-source keys in a workflow result tree."""

    forbidden = {"raw", "raw_text", "source_text", "text", "phi", "payload"}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            overlap = forbidden.intersection(str(key).casefold() for key in value)
            if overlap:
                raise ValueError("Journey workflow result contains raw source fields")
            for item in value.values():
                visit(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                visit(item)

    visit(payload)


__all__ = [
    "JOURNEY_WORKFLOW_BY_NAME",
    "JOURNEY_WORKFLOW_BY_TOOL",
    "JOURNEY_WORKFLOW_COMPATIBILITY",
    "JOURNEY_WORKFLOW_DEFINITIONS",
    "JOURNEY_WORKFLOW_SCHEMA_VERSION",
    "JourneyWorkflowDefinition",
    "assert_no_raw_source_fields",
    "execute_journey_workflow",
    "journey_workflow_client_contract",
    "journey_workflow_input_schema",
    "journey_workflow_output_schema",
    "journey_workflow_query_properties",
    "load_journey_resource_page_schema",
    "render_journey_workflow_contract_json",
    "render_python_journey_client",
    "render_typescript_journey_client",
]
