"""Journey MCP and generated-client contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from openmed.mcp import server as mcp_server
from openmed.mcp.tool_registry import (
    TOOL_REGISTRY,
    validate_registered_tool_output,
)
from openmed.service.client import OpenMedClient
from openmed.service.journey_resources import (
    JourneyAccessPolicy,
    JourneyResourceCatalog,
    JourneyResourceKind,
    JourneyResourceRecord,
)
from openmed.service.journey_workflows import (
    JOURNEY_WORKFLOW_DEFINITIONS,
    assert_no_raw_source_fields,
    journey_workflow_input_schema,
    journey_workflow_output_schema,
    render_python_journey_client,
    render_typescript_journey_client,
)

ROOT = Path(__file__).resolve().parents[3]
PYTHON_CLIENT = ROOT / "openmed" / "service" / "journey_client_generated.py"
TYPESCRIPT_CLIENT = (
    ROOT / "clients" / "typescript" / "src" / "journey-workflows.generated.ts"
)
TYPESCRIPT_INDEX = ROOT / "clients" / "typescript" / "src" / "index.ts"


def _catalog() -> JourneyResourceCatalog:
    records = {
        JourneyResourceKind.JOURNEY: {
            "subject_id": "subj_synthetic01",
            "snapshot_id": "snap_synthetic01",
            "event_count": 3,
        },
        JourneyResourceKind.COHORT: {
            "cohort_id": "cohort_synthetic01",
            "status": "ready",
            "member_count": 1,
        },
        JourneyResourceKind.DATASET: {
            "dataset_id": "dataset_synthetic01",
            "snapshot_hash": f"sha256:{'1' * 64}",
            "row_count": 1,
        },
        JourneyResourceKind.REGISTRY: {
            "registry_id": "registry_synthetic01",
            "status": "active",
            "record_count": 1,
        },
        JourneyResourceKind.MEASURE: {
            "measure_id": "measure_synthetic01",
            "value": 1,
            "unit": "count",
            "computed_at": "2026-01-01T00:00:00Z",
        },
        JourneyResourceKind.TRIAL_REVIEW: {
            "trial_id": "trial_synthetic01",
            "status": "review_required",
            "criterion_count": 2,
        },
    }
    return JourneyResourceCatalog(
        JourneyResourceRecord(
            resource_type=kind,
            resource_id=f"{kind.value}_synthetic01",
            namespace="default",
            data=data,
        )
        for kind, data in records.items()
    )


def test_six_workflow_tools_derive_their_schemas_from_one_contract() -> None:
    assert [item.name for item in JOURNEY_WORKFLOW_DEFINITIONS] == [
        "journey",
        "cohort",
        "dataset",
        "registry",
        "measure",
        "trial_review",
    ]

    for definition in JOURNEY_WORKFLOW_DEFINITIONS:
        spec = TOOL_REGISTRY.get(definition.tool_name)
        assert spec.input_schema == journey_workflow_input_schema(definition)
        assert spec.output_schema == journey_workflow_output_schema(definition)
        assert spec.annotations() == {
            "title": definition.title,
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
        assert spec.document()["authorization"] == {
            "state_changing": False,
            "consent_receipt_required": False,
        }


def test_workflow_handlers_return_evidence_policy_snapshot_and_review_metadata() -> (
    None
):
    handlers = mcp_server.build_mcp_tool_handlers(
        None,
        journey_catalog_provider=_catalog,
    )

    for definition in JOURNEY_WORKFLOW_DEFINITIONS:
        payload = handlers[definition.tool_name]()
        assert validate_registered_tool_output(definition.tool_name, payload) == payload
        assert payload["workflow"] == definition.name
        assert payload["resource_type"] == definition.resource_type.value
        assert payload["schema_version"] == "1.0.0"
        assert payload["snapshot"]["digest"] == payload["page_info"]["snapshot_digest"]
        assert payload["evidence"]["snapshot_digest"] == payload["snapshot"]["digest"]
        assert payload["access"] == payload["policy"]
        if definition.name == "trial_review":
            assert payload["warnings"] == [
                {"code": "resource_review_required", "severity": "warning"}
            ]
            assert payload["review"] == {
                "required": True,
                "reasons": ["resource_review_required"],
            }
        else:
            assert payload["warnings"] == []
            assert payload["review"] == {"required": False, "reasons": []}
        assert_no_raw_source_fields(payload)


def test_denied_workflow_preserves_access_state_without_resource_values() -> None:
    handlers = mcp_server.build_mcp_tool_handlers(
        None,
        journey_catalog_provider=_catalog,
        journey_access_policy=JourneyAccessPolicy(
            allowed_namespaces=frozenset({"approved"})
        ),
    )

    payload = handlers["openmed_read_journey"]()

    assert payload["state"] == "denied"
    assert payload["code"] == "namespace_denied"
    assert payload["resources"] == []
    assert payload["evidence"]["resource_ids"] == []
    assert payload["access"]["state"] == "denied"
    assert payload["warnings"] == [{"code": "namespace_denied", "severity": "warning"}]
    assert validate_registered_tool_output("openmed_read_journey", payload) == payload


def test_state_changing_tools_declare_and_receive_consent_contract() -> None:
    for spec in TOOL_REGISTRY.latest_specs():
        authorization = spec.document()["authorization"]
        assert authorization["state_changing"] is (not spec.read_only_hint)
        assert authorization["consent_receipt_required"] is (not spec.read_only_hint)
        if spec.read_only_hint:
            continue
        consented = mcp_server._consented_tool_spec(spec)
        assert "consent_receipt" in consented.input_schema["properties"]
        assert "consent_receipt" in consented.signature.parameters


def test_generated_clients_are_exactly_current() -> None:
    assert PYTHON_CLIENT.read_text(encoding="utf-8") == render_python_journey_client()
    assert TYPESCRIPT_CLIENT.read_text(encoding="utf-8") == (
        render_typescript_journey_client()
    )
    typescript_index = TYPESCRIPT_INDEX.read_text(encoding="utf-8")
    for definition in JOURNEY_WORKFLOW_DEFINITIONS:
        assert f"  async {definition.typescript_method}(" in typescript_index


@pytest.mark.parametrize(
    ("method_name", "resource_type"),
    [
        (definition.python_method, definition.resource_type.value)
        for definition in JOURNEY_WORKFLOW_DEFINITIONS
    ],
)
def test_python_generated_workflow_methods_select_fixed_resource_type(
    method_name: str,
    resource_type: str,
) -> None:
    observed: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(dict(request.url.params))
        return httpx.Response(
            200,
            json={
                "state": "empty",
                "code": "no_resources",
                "resources": [],
                "page_info": {},
                "policy": {},
                "schema_version": "1.0.0",
                "compatibility_policy": "same_major",
            },
        )

    with OpenMedClient(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    ) as client:
        payload = getattr(client, method_name)(first=7)

    assert payload["state"] == "empty"
    assert observed["resource_type"] == resource_type
    assert observed["first"] == "7"


def test_existing_clinical_and_grounding_tools_remain_registered() -> None:
    handlers = mcp_server.build_mcp_tool_handlers(None)
    assert {
        "openmed_clinical_pipeline",
        "openmed_ground",
        "openmed_export_fhir",
        "openmed_risk_score",
    }.issubset(handlers)
