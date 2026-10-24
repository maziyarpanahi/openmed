"""Cross-surface tests for the fixed-option decision API."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from openmed.mcp.server import build_mcp_tool_handlers
from openmed.mcp.tool_registry import TOOL_REGISTRY, check_tool_registry_compatibility
from openmed.service.app import create_app
from openmed.service.client import OpenMedClient
from openmed.structured.decision import (
    DECISION_ADVISORY,
    DECISION_SCHEMA_VERSION,
    MAX_INPUT_CHARS,
    DecisionRequest,
    decide,
    decision_request_schema,
    decision_result_schema,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"
COMPAT_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "structured"
    / "decision_compat_v1_0_0.json"
)


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


def _question_set_payload() -> dict[str, Any]:
    return {
        "mode": "question_set",
        "input_text": '{"note": "Synthetic priority is urgent; severity is high."}',
        "input_format": "json",
        "questions": [
            {
                "question_id": "priority",
                "mode": "fixed_choice",
                "options": ["urgent", "routine"],
                "option_descriptions": ["Synthetic same-day review", None],
                "instructions": "Which synthetic priority applies?",
            },
            {
                "question_id": "flag",
                "mode": "boolean_choice",
                "options": ["true", "false"],
            },
            {
                "question_id": "severity",
                "mode": "scalar_score",
                "levels": ["Low", "High"],
            },
        ],
    }


def test_question_sets_are_identical_across_python_rest_client_and_mcp() -> None:
    payload = _question_set_payload()
    expected = decide(DecisionRequest.from_dict(payload)).to_dict()
    mcp_result = build_mcp_tool_handlers(None)["openmed_decide"](**payload)

    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as rest_client:
        response = rest_client.post("/v1/decisions", json=payload)
        typed_result = OpenMedClient(client=rest_client).decision(
            DecisionRequest.from_dict(payload)
        )

    assert response.status_code == 200
    assert response.json() == typed_result == mcp_result == expected
    assert expected["schema_version"] == DECISION_SCHEMA_VERSION
    assert [item["question_id"] for item in expected["questions"]] == [
        "priority",
        "flag",
        "severity",
    ]
    assert expected["questions"][0]["choice"] == "urgent"
    assert expected["questions"][1]["state"] == "abstained"
    assert expected["state"] == "partial"
    Draft202012Validator(decision_result_schema()).validate(expected)
    rendered = response.text
    assert "Synthetic same-day review" not in rendered
    assert "Which synthetic priority applies?" not in rendered
    assert "Synthetic priority is urgent" not in rendered


def test_mcp_transport_validates_and_decides_question_sets() -> None:
    pytest.importorskip("mcp")
    from openmed.mcp.server import create_mcp_server

    server = create_mcp_server()
    payload = _question_set_payload()
    accepted = asyncio.run(server.call_tool("openmed_decide", payload))
    legacy = asyncio.run(server.call_tool("openmed_decide", _payload()))
    invalid = deepcopy(payload)
    invalid["schema_version"] = "1.0.0"
    rejected = asyncio.run(server.call_tool("openmed_decide", invalid))

    assert accepted.isError is False
    assert (
        accepted.structuredContent
        == decide(DecisionRequest.from_dict(payload)).to_dict()
    )
    assert legacy.isError is False
    assert (
        legacy.structuredContent
        == decide(DecisionRequest.from_dict(_payload())).to_dict()
    )
    assert rejected.isError is True
    assert "Which synthetic priority applies?" not in json.dumps(
        rejected.structuredContent
    )


def test_requests_without_1_1_fields_keep_the_1_0_0_result_on_every_surface() -> None:
    payload = _payload()
    payload.pop("schema_version")
    expected = decide(DecisionRequest.from_dict(payload)).to_dict()
    mcp_result = build_mcp_tool_handlers(None)["openmed_decide"](
        mode=payload["mode"],
        input_text=payload["input_text"],
        options=payload["options"],
    )

    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        response = client.post("/v1/decisions", json=payload)

    assert response.json() == mcp_result == expected
    assert expected["schema_version"] == "1.0.0"
    assert "questions" not in expected


def test_rest_keeps_the_1_0_0_input_bound_unless_raised() -> None:
    payload = _payload()
    payload["input_text"] = "Synthetic urgent " + "x" * MAX_INPUT_CHARS

    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        too_long = client.post("/v1/decisions", json=payload)
        payload.pop("schema_version")
        payload["max_input_chars"] = len(payload["input_text"])
        raised = client.post("/v1/decisions", json=payload)

    assert too_long.status_code == 422
    assert too_long.json()["error"]["code"] == "validation_error"
    assert "xxxx" not in too_long.text
    assert raised.status_code == 200
    assert raised.json()["choice"] == "urgent"


def test_rest_rejects_1_0_0_requests_with_1_1_fields_without_echo() -> None:
    payload = _payload()
    payload["instructions"] = "Synthetic private wording."

    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        response = client.post("/v1/decisions", json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "input_error"
    assert "Synthetic private wording." not in response.text


def test_mcp_decision_tool_is_an_additive_minor_version_of_1_0_0() -> None:
    frozen = json.loads(COMPAT_FIXTURE.read_text(encoding="utf-8"))
    current = TOOL_REGISTRY.get("openmed_decide")
    previous_input = deepcopy(frozen["request_schema"])
    for metadata_key in ("$id", "$schema", "title"):
        previous_input.pop(metadata_key)
    previous = replace(
        current,
        version="1.0.0",
        input_schema=previous_input,
        output_schema=frozen["result_schema"],
    )

    assert current.version == DECISION_SCHEMA_VERSION
    check_tool_registry_compatibility([previous], [current])
    assert set(previous_input["properties"]) < set(current.input_schema["properties"])
    assert previous_input["required"] == current.input_schema["required"]
