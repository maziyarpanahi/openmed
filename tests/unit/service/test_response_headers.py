"""Tests for the REST no-store, nosniff, referrer and framing response headers."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Iterable

import pytest
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import openmed
from openmed.core import pii as openmed_pii
from openmed.processing.outputs import PredictionResult
from openmed.service import auth as service_auth
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.auth import _policy_path, _policy_routes, hash_api_key

LOOPBACK_BASE_URL = "http://127.0.0.1"
SYNTHETIC_TEXT = "Synthetic note: patient Jane Example, MRN 000-SYNTH-0001."
SYNTHETIC_PATH_ID = "synthetic-id-0001"

PHI_RESPONSE_HEADERS = {
    "cache-control": "no-store",
    "pragma": "no-cache",
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
    "content-security-policy": "frame-ancestors 'none'",
}
STATIC_DOCUMENT_CACHE_CONTROL = "no-cache"

# Explicit exemptions. Probe routes keep their responses byte-for-byte; static,
# generated API documents declare their own cache policy; WebSocket routes have
# no HTTP response headers to carry.
PROBE_PATHS = frozenset({"/health", "/livez", "/readyz"})
STATIC_DOCUMENT_PATHS = frozenset(
    {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
)
NON_HTTP_ROUTES = frozenset({("WEBSOCKET", "/graphql")})

# Routes that carry or can return PHI; the enumeration must reach every one.
REQUIRED_ENUMERATED_ROUTES = frozenset(
    {
        ("POST", "/pii/extract"),
        ("POST", "/pii/deidentify"),
        ("POST", "/pii/deidentify/stream"),
        ("POST", "/brief"),
        ("GET", "/jobs/{job_id}"),
        ("GET", "/v1/journey/resources"),
        ("POST", "/graphql"),
        ("GET", "/metrics"),
    }
)

_PATH_PARAMETER = re.compile(r"\{[^}]+\}")


class _FakeLoader:
    """Model loader double so no test downloads or loads a model."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        return object()

    def loaded_models(self) -> dict[str, Any]:
        return {}

    def unload_model(self, model_name: str) -> dict[str, Any]:
        return {"model_name": model_name, "models": 0, "tokenizers": 0, "pipelines": 0}

    def unload_all_models(self) -> dict[str, int]:
        return {"models": 0, "tokenizers": 0, "pipelines": 0}


def _empty_prediction(text: str, *args: Any, **kwargs: Any) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name=str(kwargs.get("model_name") or "stub"),
        timestamp=datetime.now().isoformat(),
    )


@pytest.fixture(autouse=True)
def _service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for name in list(os.environ):
        if name.startswith("OPENMED_SERVICE_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", _FakeLoader)


@pytest.fixture
def fake_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace PII extraction on both public and core entry points."""

    monkeypatch.setattr(openmed, "extract_pii", _empty_prediction)
    monkeypatch.setattr(openmed_pii, "extract_pii", _empty_prediction)


@pytest.fixture
def client() -> Iterable[TestClient]:
    app = create_app()
    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client


def _declare_synthetic_routes(monkeypatch: pytest.MonkeyPatch, *paths: str) -> None:
    """Declare test-only routes so the app's fail-closed policy check passes."""

    for path in paths:
        monkeypatch.setitem(
            service_auth.ROUTE_POLICIES,
            ("GET", path),
            service_auth.RoutePolicy((), "control"),
        )


def _header_problems(response: Any, expected: dict[str, str]) -> list[str]:
    problems = []
    for name, value in expected.items():
        values = response.headers.get_list(name)
        if values != [value]:
            problems.append(f"{name}={values!r}")
    return problems


def _assert_phi_headers(response: Any, label: str = "") -> None:
    problems = _header_problems(response, PHI_RESPONSE_HEADERS)
    assert not problems, f"{label} {response.status_code}: {problems}"


def _assert_no_policy_headers(response: Any, label: str = "") -> None:
    present = [name for name in PHI_RESPONSE_HEADERS if name in response.headers]
    assert not present, f"{label}: {present}"


def _concrete_path(template: str) -> str:
    return _PATH_PARAMETER.sub(SYNTHETIC_PATH_ID, template)


def _route_header_failures(
    client: TestClient,
    app: FastAPI,
    *,
    exempt_paths: frozenset[str] = PROBE_PATHS,
    static_paths: frozenset[str] = STATIC_DOCUMENT_PATHS,
    non_http_routes: frozenset[tuple[str, str]] = NON_HTTP_ROUTES,
) -> tuple[list[str], set[tuple[str, str]]]:
    """Call every registered route once and report missing response headers."""

    failures: list[str] = []
    checked: set[tuple[str, str]] = set()
    for route in _policy_routes(app):
        template = _policy_path(route)
        methods = getattr(route, "methods", None) or ("WEBSOCKET",)
        for method in sorted(methods):
            key = (method, template)
            if template in exempt_paths or key in non_http_routes:
                continue
            assert method != "WEBSOCKET", f"undeclared non-HTTP route: {key}"
            response = client.request(method, _concrete_path(template))
            checked.add(key)
            expected = dict(PHI_RESPONSE_HEADERS)
            if template in static_paths:
                del expected["pragma"]
                expected["cache-control"] = STATIC_DOCUMENT_CACHE_CONTROL
            problems = _header_problems(response, expected)
            if problems:
                failures.append(
                    f"{method} {template} {response.status_code} {problems}"
                )
    return failures, checked


@pytest.mark.usefixtures("fake_extraction")
def test_phi_routes_carry_headers_on_success(client: TestClient) -> None:
    responses = {
        "extract": client.post("/pii/extract", json={"text": SYNTHETIC_TEXT}),
        "deidentify": client.post(
            "/pii/deidentify",
            json={"text": SYNTHETIC_TEXT, "method": "mask", "keep_mapping": True},
        ),
        "journey": client.get(
            "/v1/journey/resources",
            params={"resource_type": "cohort_run", "fields": "member_count"},
        ),
        "models": client.get("/models/loaded"),
    }

    for label, response in responses.items():
        assert response.status_code == 200, label
        _assert_phi_headers(response, label)


def test_error_envelopes_carry_headers(client: TestClient) -> None:
    responses = {
        "unknown_route_404": client.get("/no-such-route"),
        "missing_job_404": client.get(f"/jobs/{SYNTHETIC_PATH_ID}"),
        "wrong_method_405": client.get("/pii/extract"),
        "validation_422": client.post("/pii/extract", json={}),
        "malformed_json_422": client.post(
            "/pii/deidentify",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        ),
        "untrusted_host_400": client.get(
            "/models/loaded", headers={"Host": "attacker.invalid"}
        ),
    }

    expected_status = {
        "unknown_route_404": 404,
        "missing_job_404": 404,
        "wrong_method_405": 405,
        "validation_422": 422,
        "malformed_json_422": 422,
        "untrusted_host_400": 400,
    }
    for label, response in responses.items():
        assert response.status_code == expected_status[label], label
        assert "error" in response.json(), label
        _assert_phi_headers(response, label)


def test_authentication_failure_envelope_carries_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [
                {
                    "id": "test-key",
                    "key_hash": hash_api_key("synthetic-secret"),
                    "principal": "test-service",
                    "scopes": ["pii:write"],
                }
            ]
        ),
    )
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        missing = test_client.post("/pii/extract", json={"text": SYNTHETIC_TEXT})
        invalid = test_client.post(
            "/pii/extract",
            headers={"X-API-Key": "wrong-secret"},
            json={"text": SYNTHETIC_TEXT},
        )
        probe = test_client.get("/livez")

    for label, response in {"missing": missing, "invalid": invalid}.items():
        assert response.status_code == 401, label
        assert response.json()["error"]["code"] in {
            "authentication_required",
            "invalid_credentials",
        }
        _assert_phi_headers(response, label)
    assert probe.status_code == 200
    _assert_no_policy_headers(probe, "livez")


def test_unhandled_exception_envelope_carries_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Responses rendered by Starlette's ServerErrorMiddleware are covered too."""

    _declare_synthetic_routes(monkeypatch, "/synthetic-unhandled-failure")
    app = create_app()

    @app.get("/synthetic-unhandled-failure", include_in_schema=False)
    async def _fail() -> None:
        raise RuntimeError("synthetic failure")

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        response = test_client.get("/synthetic-unhandled-failure")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "synthetic failure" not in response.text
    _assert_phi_headers(response, "unhandled_500")


@pytest.mark.usefixtures("fake_extraction")
def test_ndjson_stream_carries_headers(client: TestClient) -> None:
    payload = {"text": SYNTHETIC_TEXT * 4, "method": "mask", "chunk_size": 32}
    with client.stream("POST", "/pii/deidentify/stream", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        _assert_phi_headers(response, "deidentify_stream")
        events = [json.loads(line) for line in response.iter_lines() if line]

    assert events[-1]["type"] == "final"


def test_probe_responses_are_unchanged(client: TestClient) -> None:
    for path in sorted(PROBE_PATHS):
        response = client.get(path)
        assert response.status_code == 200, path
        _assert_no_policy_headers(response, path)


def test_static_documents_use_the_declared_cache_policy(client: TestClient) -> None:
    for path in ("/openapi.json", "/docs", "/redoc"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers.get_list("cache-control") == [
            STATIC_DOCUMENT_CACHE_CONTROL
        ], path
        assert "pragma" not in response.headers, path
        for name in (
            "x-content-type-options",
            "referrer-policy",
            "x-frame-options",
            "content-security-policy",
        ):
            assert response.headers.get_list(name) == [PHI_RESPONSE_HEADERS[name]]


def test_static_cache_policy_does_not_leak_to_other_routes(
    client: TestClient,
) -> None:
    response = client.get("/docs/extra-segment")

    assert response.status_code == 404
    _assert_phi_headers(response, "docs_prefix")


def test_graphql_responses_do_not_duplicate_headers(client: TestClient) -> None:
    responses = {
        "post": client.post("/graphql", json={"query": "{ entityTypes { label } }"}),
        "get": client.get("/graphql", headers={"Accept": "text/html"}),
        "put": client.request("PUT", "/graphql", json={"query": "{ __typename }"}),
    }

    for label, response in responses.items():
        _assert_phi_headers(response, f"graphql_{label}")


class _GraphQLTransportHeaders:
    """Stand-in for the GraphQL transport header middleware proposed in #3614."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") != "/graphql":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["cache-control"] = "no-store"
                headers["pragma"] = "no-cache"
                headers["referrer-policy"] = "no-referrer"
            await send(message)

        await self.app(scope, receive, send_with_headers)


def test_graphql_headers_stay_single_with_transport_middleware() -> None:
    app = create_app()
    app.add_middleware(_GraphQLTransportHeaders)

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        post = test_client.post("/graphql", json={"query": "{ entityTypes { label } }"})
        rest = test_client.get("/models/loaded")

    assert post.status_code == 200
    _assert_phi_headers(post, "graphql_with_transport_middleware")
    _assert_phi_headers(rest, "rest_with_transport_middleware")


def test_handler_headers_keep_stricter_cache_but_enforce_privacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _declare_synthetic_routes(
        monkeypatch, "/synthetic-weak-cache", "/synthetic-strict-cache"
    )
    app = create_app()

    @app.get("/synthetic-weak-cache", include_in_schema=False)
    async def _weak_cache() -> Response:
        return JSONResponse(
            {"ok": True},
            headers={
                "Cache-Control": "public, max-age=600",
                "Pragma": "public",
                "X-Content-Type-Options": "sniff",
            },
        )

    @app.get("/synthetic-strict-cache", include_in_schema=False)
    async def _strict_cache() -> Response:
        return JSONResponse(
            {"ok": True},
            headers={
                "Cache-Control": "no-store, max-age=0",
                "Referrer-Policy": "same-origin",
                "X-Frame-Options": "SAMEORIGIN",
            },
        )

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as test_client:
        weak = test_client.get("/synthetic-weak-cache")
        strict = test_client.get("/synthetic-strict-cache")

    _assert_phi_headers(weak, "weak_cache")
    assert strict.headers.get_list("cache-control") == ["no-store, max-age=0"]
    assert strict.headers.get_list("pragma") == ["no-cache"]
    assert strict.headers.get_list("referrer-policy") == ["no-referrer"]
    assert strict.headers.get_list("x-frame-options") == ["DENY"]
    assert strict.headers.get_list("x-content-type-options") == ["nosniff"]


def test_install_is_idempotent() -> None:
    from openmed.service.response_headers import install_response_headers

    app = create_app()
    install_response_headers(app)

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as test_client:
        response = test_client.get("/models/loaded")

    _assert_phi_headers(response, "double_install")


@pytest.mark.parametrize("metrics_enabled", [False, True])
def test_every_registered_route_carries_headers(
    monkeypatch: pytest.MonkeyPatch,
    metrics_enabled: bool,
) -> None:
    """Fail when a newly registered route is served without the headers."""

    if metrics_enabled:
        monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as test_client:
        failures, checked = _route_header_failures(test_client, app)

    assert not failures, "\n".join(failures)
    required = REQUIRED_ENUMERATED_ROUTES
    if not metrics_enabled:
        required = required - {("GET", "/metrics")}
    assert required <= checked, sorted(required - checked)


def test_route_enumeration_flags_a_route_without_headers() -> None:
    """Negative control: the enumeration fails on an app without the policy."""

    bare = FastAPI()

    @bare.get("/synthetic-unprotected")
    async def _unprotected() -> dict[str, bool]:
        return {"ok": True}

    with TestClient(bare, base_url=LOOPBACK_BASE_URL) as test_client:
        failures, checked = _route_header_failures(test_client, bare)

    assert ("GET", "/synthetic-unprotected") in checked
    assert any("/synthetic-unprotected" in failure for failure in failures)


def test_existing_headers_cannot_relax_referrer_or_framing_policy():
    from openmed.service.response_headers import apply_response_headers

    headers = MutableHeaders(
        raw=[
            (b"referrer-policy", b"unsafe-url"),
            (b"x-frame-options", b"ALLOWALL"),
            (b"content-security-policy", b"script-src 'self'; frame-ancestors *"),
            (b"content-security-policy", b"object-src 'none'"),
        ]
    )
    apply_response_headers(headers)
    assert headers.getlist("referrer-policy") == ["no-referrer"]
    assert headers.getlist("x-frame-options") == ["DENY"]
    assert headers.getlist("content-security-policy") == [
        "script-src 'self'; frame-ancestors *, object-src 'none', frame-ancestors 'none'"
    ]


def test_malformed_no_store_directive_does_not_bypass_policy():
    from openmed.service.response_headers import apply_response_headers

    headers = MutableHeaders({"cache-control": "public, no-store=false"})
    apply_response_headers(headers)
    assert headers["cache-control"] == "no-store"
