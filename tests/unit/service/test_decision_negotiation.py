"""Content negotiation of the two decision request shapes on ``/v1/decisions``.

OpenMed requests and OpenAI-compatible Decisions bodies share one route. The
shared fixture pins every exchange; the typed clients replay the same file.
Every request here is synthetic and stays in process or on loopback.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, get_args, get_origin

import httpx
import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from openmed.service.app import create_app
from openmed.service.auth import hash_api_key
from openmed.service.client import OpenMedAPIError, OpenMedClient
from openmed.service.schemas import FixedOptionDecisionRequest, OpenAIDecisionsRequest
from openmed.structured.decision import (
    DECISION_SCHEMA_VERSION,
    SUPPORTED_DECISION_SCHEMA_VERSIONS,
    DecisionAccessPolicy,
    DecisionState,
    decide,
    decision_request_schema,
    decision_result_schema,
)
from openmed.structured.decision_interop import (
    DecisionRequestShape,
    DecisionRequestShapeError,
    decision_request_shape,
    from_openai_decisions_request,
    openai_decisions_request_schema,
    openai_decisions_response_schema,
    to_openai_decisions_response,
)
from openmed.structured.decision_server import DecisionServerBackend

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "service" / "decision-negotiation-v1.json"
LOOPBACK_BASE_URL = "http://127.0.0.1"
SERVED_MODEL = "openmed/deterministic-fixed-option"
PRIVATE_INPUT = "Synthetic marker QX-7731 needs urgent review."
PRIVATE_INSTRUCTIONS = "Is synthetic marker QX-7731 urgent?"
API_KEY = "synthetic-decision-key"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _cases(*, typed: bool | None = None) -> list[dict[str, Any]]:
    return [
        case for case in _fixture()["cases"] if typed is None or case["typed"] is typed
    ]


def _openai_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "model": SERVED_MODEL,
        "input": PRIVATE_INPUT,
        "questions": [
            {
                "type": "predicate",
                "name": "urgent",
                "instructions": PRIVATE_INSTRUCTIONS,
            },
            {
                "type": "choice",
                "name": "priority",
                "choices": [{"value": "urgent"}, {"value": "routine"}],
            },
        ],
    }
    body.update(overrides)
    return body


def _openmed_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "mode": "fixed_choice",
        "input_text": PRIVATE_INPUT,
        "options": ["urgent", "routine"],
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Shape classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "shape"),
    [
        ({"mode": "fixed_choice", "input_text": "x"}, DecisionRequestShape.OPENMED),
        ({"mode": "question_set", "questions": []}, DecisionRequestShape.OPENMED),
        ({"input_text": "x"}, DecisionRequestShape.OPENMED),
        (
            {"model": "m", "input": "x", "questions": []},
            DecisionRequestShape.OPENAI_DECISIONS,
        ),
        ({"model": "m", "questions": []}, DecisionRequestShape.OPENAI_DECISIONS),
        ({"input": "x"}, DecisionRequestShape.OPENAI_DECISIONS),
    ],
)
def test_disjoint_required_keys_select_one_shape(
    body: dict[str, Any], shape: DecisionRequestShape
) -> None:
    assert decision_request_shape(body) is shape


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({"mode": "fixed_choice", "input_text": "x", "model": "m"}, "both_shapes"),
        ({"input_text": "x", "input": "x"}, "both_shapes"),
        ({"questions": [{"type": "predicate", "name": "q"}]}, "no_shape"),
        ({}, "no_shape"),
        (
            {"model": "m", "input": "x", "questions": [], "media": []},
            "media_unsupported",
        ),
    ],
)
def test_mixed_shapeless_and_media_bodies_are_refused_without_echo(
    body: dict[str, Any], reason: str
) -> None:
    with pytest.raises(DecisionRequestShapeError) as raised:
        decision_request_shape({**body, "secret": PRIVATE_INPUT})

    assert raised.value.reason_code == reason
    assert PRIVATE_INPUT not in str(raised.value)


def test_shape_detection_requires_a_mapping() -> None:
    with pytest.raises(TypeError):
        decision_request_shape([PRIVATE_INPUT])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Shared fixture: live service and typed Python client
# ---------------------------------------------------------------------------


def test_shared_fixture_covers_every_negotiation_outcome() -> None:
    fixture = _fixture()
    outcomes = {
        (case["shape"], case["status"], (case.get("error") or {}).get("code"))
        for case in fixture["cases"]
    }

    assert fixture["decision_schema_version"] == DECISION_SCHEMA_VERSION
    assert ("openmed", 200, None) in outcomes
    assert ("openai_decisions", 200, None) in outcomes
    assert ("openai_decisions", 422, "model_not_served") in outcomes
    assert ("openai_decisions", 422, "decision_media_unsupported") in outcomes
    assert {
        (case.get("error") or {}).get("details", {}).get("reason")
        for case in fixture["cases"]
        if (case.get("error") or {}).get("code") == "decision_request_ambiguous"
    } == {"both_shapes", "no_shape"}


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["name"])
def test_live_service_reproduces_the_shared_fixture(case: dict[str, Any]) -> None:
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        response = client.post("/v1/decisions", json=case["request"])

    assert response.status_code == case["status"]
    payload = response.json()
    if case["status"] == 200:
        assert payload == case["response"]
    else:
        error = payload["error"]
        assert {"code": error["code"], "details": error["details"]} == case["error"]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["name"])
def test_fixture_bodies_validate_against_the_canonical_documents(
    case: dict[str, Any],
) -> None:
    openmed_request = Draft202012Validator(decision_request_schema())
    openai_request = Draft202012Validator(openai_decisions_request_schema())
    request = case["request"]

    if case["status"] != 200 and case["shape"] is None:
        assert not openmed_request.is_valid(request)
        assert not openai_request.is_valid(request)
        return
    if case["shape"] == "openmed":
        openmed_request.validate(request)
        Draft202012Validator(decision_result_schema()).validate(case["response"])
        return
    if (case.get("error") or {}).get("code") == "decision_media_unsupported":
        assert not openai_request.is_valid(request)
        return
    openai_request.validate(request)
    if case["status"] == 200:
        Draft202012Validator(openai_decisions_response_schema()).validate(
            case["response"]
        )


@pytest.mark.parametrize("case", _cases(typed=True), ids=lambda case: case["name"])
def test_python_client_round_trips_the_shared_fixture(case: dict[str, Any]) -> None:
    sent: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        if case["status"] == 200:
            return httpx.Response(200, json=case["response"])
        return httpx.Response(
            case["status"],
            json={"error": {**case["error"], "message": "Synthetic refusal"}},
        )

    with OpenMedClient(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    ) as client:
        call = (
            client.openai_decisions
            if case["shape"] == "openai_decisions"
            else client.decision
        )
        if case["status"] == 200:
            assert call(case["request"]) == case["response"]
        else:
            with pytest.raises(OpenMedAPIError) as raised:
                call(case["request"])
            assert raised.value.code == case["error"]["code"]

    assert sent == [case["request"]]


# ---------------------------------------------------------------------------
# One route, two shapes: same results, policy, auth and limits
# ---------------------------------------------------------------------------


def test_openai_branch_matches_the_core_converters_and_policy() -> None:
    body = _openai_body()
    request = from_openai_decisions_request(body)
    expected = to_openai_decisions_response(
        request, decide(request), include_openmed=True
    )

    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        response = client.post("/v1/decisions", json=body)
        typed = OpenMedClient(client=client).openai_decisions(body)

    assert response.status_code == 200
    assert response.json() == typed == expected
    review = expected["openmed"]
    assert review["schema_version"] == DECISION_SCHEMA_VERSION
    assert review["review"]["required"] is True
    assert review["autonomous_action"] is False
    assert review["backend"]["model_id"] == SERVED_MODEL
    assert PRIVATE_INPUT not in response.text
    assert PRIVATE_INSTRUCTIONS not in response.text


def test_both_shapes_apply_the_same_access_policy() -> None:
    app = create_app()
    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        app.state.decision_access_policy = DecisionAccessPolicy(
            allowed_purposes=frozenset({"quality"})
        )
        openmed = client.post("/v1/decisions", json=_openmed_body())
        openai = client.post("/v1/decisions", json=_openai_body())

    assert openmed.json()["state"] == "denied"
    assert openmed.json()["code"] == "purpose_denied"
    assert openai.json()["openmed"]["state"] == "denied"
    assert {answer["state"] for answer in openai.json()["answers"]} == {"denied"}


def test_both_shapes_share_the_decisions_write_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def app_with_scopes(scopes: list[str]) -> Any:
        monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "1")
        monkeypatch.setenv(
            "OPENMED_SERVICE_AUTH_API_KEYS",
            json.dumps(
                [
                    {
                        "id": "synthetic-key",
                        "key_hash": hash_api_key(API_KEY),
                        "principal": "synthetic",
                        "scopes": scopes,
                    }
                ]
            ),
        )
        return create_app()

    headers = {"X-API-Key": API_KEY}
    for scopes, status in ((["pii:read"], 403), (["decisions:write"], 200)):
        with TestClient(app_with_scopes(scopes), base_url=LOOPBACK_BASE_URL) as client:
            for body in (_openmed_body(), _openai_body()):
                response = client.post("/v1/decisions", json=body, headers=headers)
                assert response.status_code == status, (scopes, sorted(body))
            anonymous = client.post("/v1/decisions", json=_openai_body())
            assert anonymous.status_code == 401


def test_both_shapes_share_the_request_body_limit() -> None:
    app = create_app(max_request_body_bytes=256)
    padded = PRIVATE_INPUT + " padding" * 64
    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        for body in (_openmed_body(input_text=padded), _openai_body(input=padded)):
            response = client.post("/v1/decisions", json=body)
            assert response.status_code == 413
            assert response.json()["error"]["code"] == "payload_too_large"
            assert PRIVATE_INPUT not in response.text


def test_both_shapes_keep_the_default_input_bound() -> None:
    too_long = "Synthetic urgent " + "x" * 8192
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        for body in (_openmed_body(input_text=too_long), _openai_body(input=too_long)):
            response = client.post("/v1/decisions", json=body)
            assert response.status_code == 422
            assert response.json()["error"]["code"] == "validation_error"
            assert "xxxx" not in response.text


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (_openmed_body(model=SERVED_MODEL), "decision_request_ambiguous"),
        (_openai_body(mode="question_set"), "decision_request_ambiguous"),
        ({"questions": _openai_body()["questions"]}, "decision_request_ambiguous"),
        (_openai_body(media=[{"data": PRIVATE_INPUT}]), "decision_media_unsupported"),
        (_openai_body(model="synthetic-org/other-decider"), "model_not_served"),
    ],
)
def test_refusals_are_controlled_and_never_echo_the_body(
    body: dict[str, Any], code: str
) -> None:
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        response = client.post("/v1/decisions", json=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == code
    assert PRIVATE_INPUT not in response.text
    assert PRIVATE_INSTRUCTIONS not in response.text
    assert "synthetic-org/other-decider" not in response.text


def test_invalid_openai_questions_are_input_errors_without_echo() -> None:
    body = _openai_body(
        questions=[{"type": "multiple", "name": "q", "instructions": PRIVATE_INPUT}]
    )
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        response = client.post("/v1/decisions", json=body)
        not_object = client.post("/v1/decisions", json=[PRIVATE_INPUT])

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "input_error"
    assert PRIVATE_INPUT not in response.text
    assert not_object.status_code == 422
    assert not_object.json()["error"]["code"] == "validation_error"
    assert PRIVATE_INPUT not in not_object.text


# ---------------------------------------------------------------------------
# OpenMed -> OpenMed bridge over loopback
# ---------------------------------------------------------------------------


class _ForwardingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        response = self.server.client.post(  # type: ignore[attr-defined]
            self.path,
            content=body,
            headers={"Content-Type": "application/json"},
        )
        payload = response.content
        self.send_response(response.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return None


@pytest.fixture
def openmed_server() -> Iterator[str]:
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ForwardingHandler)
        server.daemon_threads = True
        server.client = client  # type: ignore[attr-defined]
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            server.server_close()


def test_openmed_decision_server_backend_reaches_another_openmed_service(
    openmed_server: str,
) -> None:
    backend = DecisionServerBackend(
        endpoint=openmed_server,
        protocol="openai_decisions",
        model_id=SERVED_MODEL,
        revision="1.0.0",
        license_id="Apache-2.0",
    )
    request = from_openai_decisions_request(
        {
            "model": SERVED_MODEL,
            "input": "Synthetic review priority is urgent.",
            "questions": [
                {
                    "type": "choice",
                    "name": "priority",
                    "choices": [{"value": "urgent"}, {"value": "routine"}],
                }
            ],
        }
    )

    result = decide(request, backend=backend)

    (question,) = result.questions
    assert question.state is DecisionState.SUCCESS
    assert question.choice == "urgent"
    assert result.backend["runtime"] == "openai_decisions_server:loopback"


def test_bridge_to_a_service_with_another_model_fails_closed(
    openmed_server: str,
) -> None:
    backend = DecisionServerBackend(
        endpoint=openmed_server,
        protocol="openai_decisions",
        model_id="synthetic-org/other-decider",
        revision="1.0.0",
        license_id="Apache-2.0",
    )
    request = from_openai_decisions_request(
        {
            "model": "synthetic-org/other-decider",
            "input": "Synthetic review priority is urgent.",
            "questions": [{"type": "predicate", "name": "urgent"}],
        }
    )

    result = decide(request, backend=backend)

    assert result.state is not DecisionState.SUCCESS
    assert all(item.choice is None for item in result.questions)


# ---------------------------------------------------------------------------
# OpenAPI and pydantic parity with the canonical documents
# ---------------------------------------------------------------------------


def test_openapi_documents_both_request_and_response_shapes() -> None:
    openapi = create_app().openapi()
    operation = openapi["paths"]["/v1/decisions"]["post"]
    variants = operation["requestBody"]["content"]["application/json"]["schema"][
        "oneOf"
    ]
    components = openapi["components"]["schemas"]
    responses = operation["responses"]["200"]["content"]["application/json"]["schema"]

    assert [item["$ref"].rsplit("/", 1)[-1] for item in variants] == [
        "FixedOptionDecisionRequest",
        "OpenAIDecisionsRequest",
    ]
    for name, canonical in (
        ("FixedOptionDecisionRequest", decision_request_schema()),
        ("OpenAIDecisionsRequest", openai_decisions_request_schema()),
    ):
        assert components[name]["properties"] == canonical["properties"]
        assert components[name]["required"] == canonical["required"]
        assert components[name]["$id"] == canonical["$id"]
        assert components[name]["additionalProperties"] is False
    assert responses["oneOf"] == [
        decision_result_schema(),
        openai_decisions_response_schema(),
    ]


def _literal_values(annotation: Any) -> set[Any]:
    if get_origin(annotation) is not None and str(get_origin(annotation)).endswith(
        "Literal"
    ):
        return set(get_args(annotation))
    values: set[Any] = set()
    for argument in get_args(annotation):
        values |= _literal_values(argument)
    return values


def _constraints(field: Any) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for item in field.metadata:
        for name in ("min_length", "max_length", "ge", "le"):
            value = getattr(item, name, None)
            if value is not None:
                found[name] = value
    return found


def _schema_bounds(schema: dict[str, Any]) -> dict[str, Any]:
    variants = [
        item for item in schema.get("anyOf", [schema]) if item.get("type") != "null"
    ]
    (active,) = variants
    mapping = {
        "minLength": "min_length",
        "maxLength": "max_length",
        "minItems": "min_length",
        "maxItems": "max_length",
        "minimum": "ge",
        "maximum": "le",
    }
    return {mapping[key]: value for key, value in active.items() if key in mapping}


@pytest.mark.parametrize(
    ("model", "schema"),
    [
        (FixedOptionDecisionRequest, decision_request_schema()),
        (OpenAIDecisionsRequest, openai_decisions_request_schema()),
    ],
    ids=["openmed", "openai_decisions"],
)
def test_rest_request_models_match_the_canonical_schema(
    model: Any, schema: dict[str, Any]
) -> None:
    fields = model.model_fields
    properties = schema["properties"]

    assert set(fields) == set(properties)
    assert {name for name, field in fields.items() if field.is_required()} == set(
        schema["required"]
    )
    for name, field in fields.items():
        prop = properties[name]
        literal = _literal_values(field.annotation)
        if literal:
            expected = set(prop.get("enum", ())) or {prop["const"]}
            assert literal == expected, name
        bounds = _schema_bounds(prop)
        for key, value in _constraints(field).items():
            assert bounds.get(key) == value, (name, key)
        if prop.get("type") == "array" and "maxItems" in prop:
            assert _constraints(field).get("max_length") == prop["maxItems"], name


def test_rest_model_accepts_exactly_the_supported_schema_versions() -> None:
    annotation = FixedOptionDecisionRequest.model_fields["schema_version"].annotation

    assert _literal_values(annotation) == set(SUPPORTED_DECISION_SCHEMA_VERSIONS)
