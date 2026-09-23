"""Cross-surface tests for the fixed-option decision API."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from fastapi.testclient import TestClient

from openmed.mcp.server import build_mcp_tool_handlers
from openmed.mcp.tool_registry import TOOL_REGISTRY
from openmed.service.app import create_app
from openmed.service.client import OpenMedClient
from openmed.structured.decision import (
    DECISION_ADVISORY,
    DecisionRequest,
    decide,
    decision_request_schema,
    decision_result_schema,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"


def _payload() -> dict[str, Any]:
    return {
        "mode": "fixed_choice",
        "input_text": "Synthetic review priority is urgent.",
        "options": ["urgent", "routine"],
        "namespace": "default",
        "purpose": "care_review",
        "calibration_id": "openmed.synthetic.fixed_option.v1",
        "timeout_ms": 5000,
        "schema_version": "1.0.0",
        "compatibility_policy": "same_major",
    }


def test_python_rest_client_and_mcp_return_identical_decisions() -> None:
    payload = _payload()
    expected = decide(DecisionRequest.from_dict(payload)).to_dict()
    mcp_result = build_mcp_tool_handlers(None)["openmed_decide"](**payload)

    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as rest_client:
        response = rest_client.post("/v1/decisions", json=payload)
        typed_result = OpenMedClient(client=rest_client).decision(payload)

    assert response.status_code == 200
    assert response.json() == typed_result == mcp_result == expected
    assert expected["advisory"] == DECISION_ADVISORY
    assert expected["autonomous_action"] is False
    assert expected["review"]["required"] is True


def test_rest_rejects_invalid_options_without_echoing_values() -> None:
    payload = _payload()
    payload["options"] = ["private-marker", " private-marker "]

    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        response = client.post("/v1/decisions", json=payload)

    assert response.status_code == 400
    rendered = response.text
    assert "private-marker" not in rendered
    assert response.json()["error"]["code"] == "input_error"


def test_openapi_and_mcp_share_the_canonical_decision_contract() -> None:
    request_schema = decision_request_schema()
    result_schema = decision_result_schema()
    openapi = create_app().openapi()
    operation = openapi["paths"]["/v1/decisions"]["post"]
    request_ref = operation["requestBody"]["content"]["application/json"]["schema"][
        "$ref"
    ]
    component = openapi["components"]["schemas"][request_ref.rsplit("/", 1)[-1]]
    rest_result = operation["responses"]["200"]["content"]["application/json"]["schema"]
    mcp_spec = TOOL_REGISTRY.get("openmed_decide")
    mcp_request = deepcopy(request_schema)
    for metadata_key in ("$id", "$schema", "title"):
        mcp_request.pop(metadata_key)

    assert component["properties"] == request_schema["properties"]
    assert component["required"] == request_schema["required"]
    assert component["additionalProperties"] is False
    assert rest_result == result_schema
    assert mcp_spec.input_schema == mcp_request
    assert mcp_spec.output_schema == result_schema
    assert mcp_spec.read_only_hint is True
    assert mcp_spec.destructive_hint is False
    assert mcp_spec.open_world_hint is False
