"""Discoverable resources for the canonical local clinical MCP workflow."""

from __future__ import annotations

import json
from copy import deepcopy
from importlib import resources
from typing import Any, Mapping

from openmed.mcp.tool_registry import (
    CLINICAL_STAGE_ORDER,
    CLINICAL_WORKFLOW_NAME,
    TOOL_REGISTRY,
    ToolSchemaValidationError,
    validate_registered_workflow_artifact,
)

GOLDEN_AGENT_RUN_RESOURCE = "fixtures/clinical_workflow_golden.json"


def render_clinical_workflow_prompt(
    text: str = (
        "Synthetic subject Cedar Example, record SYN-1303-ALPHA, reports "
        "aster syndrome."
    ),
) -> str:
    """Return the canonical privacy-first workflow prompt for an MCP client.

    Args:
        text: Synthetic or operator-approved clinical text. Real PHI should only
            be supplied to an OpenMed runtime that the operator controls.

    Returns:
        A prompt that names the discoverable resources and safe stage order.
    """

    return (
        "Run the canonical OpenMed clinical workflow using default local "
        "execution and zero network egress. First call openmed_deidentify with "
        "method='mask'. Treat that call as the privacy boundary: keep its "
        "original_text and entity surfaces process-local, never log them, and "
        "pass only deidentified_text beyond the boundary. Then call "
        "openmed_clinical_pipeline with allow_external_llm=false and stages="
        f"{list(CLINICAL_STAGE_ORDER)!r}. Preserve the declared order; do not "
        "silently reorder or skip validation. Discover tool and artifact "
        "contracts from openmed://tool-registry and workflow guidance from "
        "openmed://clinical-workflow instead of hardcoding schemas. Validate "
        "every intermediate artifact. Return only the grounded FHIR Bundle and "
        "aggregate residual-risk report. Do not expose the source note, direct "
        "identifiers, entity surfaces, reversible mappings, or raw PHI in "
        "outputs or logs. Grounded codes require human review and must not "
        "auto-trigger clinical, treatment, billing, or medical-device decisions. "
        f"Approved input: {text!r}"
    )


def clinical_workflow_resource_document() -> dict[str, Any]:
    """Return stage, privacy, artifact, and discovery guidance for clients."""

    spec = TOOL_REGISTRY.get_workflow(CLINICAL_WORKFLOW_NAME)
    document = spec.document(include_schemas=True)
    document.update(
        {
            "execution": {
                "default": "local",
                "network_egress": False,
                "external_llm": (
                    "Disabled by default. If explicitly enabled by an operator, "
                    "the stage must use the OpenMed privacy gateway."
                ),
            },
            "privacy_boundaries": [
                {
                    "boundary": "input_to_deidentify",
                    "rule": (
                        "Source text may exist only inside the trusted local "
                        "runtime and the de-identification call."
                    ),
                },
                {
                    "boundary": "post_deidentify",
                    "rule": (
                        "Only de-identified text, offsets, hashes, canonical "
                        "spans, terminology provenance, aggregate risk, and "
                        "identifier-free FHIR resources may cross this boundary."
                    ),
                },
                {
                    "boundary": "external_stage",
                    "rule": (
                        "External-LLM-capable stages are opt-in and reachable "
                        "only through the privacy gateway; raw text is excluded."
                    ),
                },
            ],
            "output": {
                "fhir_artifact": "export.bundle",
                "risk_artifact": "risk.risk_report",
                "clinical_use": (
                    "Human review required; outputs are not autonomous clinical "
                    "or medical-device decisions."
                ),
            },
        }
    )
    return document


def load_golden_agent_run() -> dict[str, Any]:
    """Load and validate the bundled synthetic offline agent-run fixture."""

    resource = resources.files("openmed.mcp").joinpath(GOLDEN_AGENT_RUN_RESOURCE)
    with resource.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_golden_agent_run(payload)


def validate_golden_agent_run(payload: Any) -> dict[str, Any]:
    """Validate fixture structure, every artifact schema, and PHI boundaries."""

    if not isinstance(payload, Mapping):
        raise ToolSchemaValidationError("golden agent run must be an object")
    fixture = deepcopy(dict(payload))
    workflow = fixture.get("workflow")
    if not isinstance(workflow, Mapping):
        raise ToolSchemaValidationError("golden agent run workflow must be an object")
    spec = TOOL_REGISTRY.get_workflow(CLINICAL_WORKFLOW_NAME)
    if workflow.get("name") != spec.name or workflow.get("version") != spec.version:
        raise ToolSchemaValidationError(
            "golden agent run does not match the registered workflow"
        )
    if workflow.get("stage_order") != list(spec.stage_order):
        raise ToolSchemaValidationError(
            "golden agent run does not use the registered stage order"
        )
    if fixture.get("synthetic") is not True:
        raise ToolSchemaValidationError("golden agent run must be explicitly synthetic")

    input_payload = fixture.get("input")
    if not isinstance(input_payload, Mapping):
        raise ToolSchemaValidationError("golden agent run input must be an object")
    note = input_payload.get("note")
    identifiers = input_payload.get("synthetic_direct_identifiers")
    if not isinstance(note, str) or not isinstance(identifiers, list):
        raise ToolSchemaValidationError("golden agent run input is invalid")
    if not identifiers or not all(
        isinstance(identifier, str) and identifier and identifier in note
        for identifier in identifiers
    ):
        raise ToolSchemaValidationError(
            "golden agent run identifiers must be synthetic input markers"
        )

    artifacts = fixture.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ToolSchemaValidationError("golden agent run artifacts must be an object")
    if set(artifacts) != set(spec.stage_order):
        raise ToolSchemaValidationError(
            "golden agent run artifacts must match registered stages"
        )
    for artifact_name in spec.stage_order:
        validate_registered_workflow_artifact(
            spec.name,
            artifact_name,
            artifacts[artifact_name],
        )

    post_boundary = json.dumps(artifacts, sort_keys=True)
    leaked = [identifier for identifier in identifiers if identifier in post_boundary]
    if leaked or note in post_boundary or "original_text" in post_boundary:
        raise ToolSchemaValidationError(
            "golden agent run exposes source text after de-identification"
        )

    export = artifacts["export"]
    risk = artifacts["risk"]
    if export.get("bundle", {}).get("resourceType") != "Bundle":
        raise ToolSchemaValidationError("golden agent run has no FHIR Bundle output")
    if not isinstance(risk.get("risk_report"), Mapping):
        raise ToolSchemaValidationError("golden agent run has no risk output")
    return fixture


__all__ = [
    "GOLDEN_AGENT_RUN_RESOURCE",
    "clinical_workflow_resource_document",
    "load_golden_agent_run",
    "render_clinical_workflow_prompt",
    "validate_golden_agent_run",
]
