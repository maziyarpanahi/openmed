"""Tests for REST CORS and trusted-host middleware."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.types import Message

from openmed.service import runtime as service_runtime
from openmed.service import security_headers
from openmed.service.app import create_app
from openmed.service.auth import hash_api_key
from openmed.service.security_headers import (
    DEFAULT_TRUSTED_HOSTS,
    parse_cors_origins,
    parse_service_security_config,
    parse_trusted_hosts,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"
ALLOWED_ORIGIN = "https://frontend.example.com"
DISALLOWED_ORIGIN = "https://attacker.example.com"
UNTRUSTED_HOST = {"Host": "attacker.example.com"}
SYNTHETIC_API_KEY = "synthetic-cors-key"


@pytest.fixture(autouse=True)
def service_env(monkeypatch):
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_DEFAULT_MODEL_FOOTPRINT_BYTES", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MODEL_ADMISSION_WAIT_SECONDS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_TEXT_LENGTH", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCHING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_SIZE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_WAIT_MS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_BATCH_MAX_QUEUE_SIZE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_TRUSTED_HOSTS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_COALESCING_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_RPS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_BURST", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_THROTTLE_KEY", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_METRICS_ENABLED", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_TRACING_ENABLED", raising=False)
    for env_var in (
        "OPENMED_SERVICE_AUTH_ENABLED",
        "OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT",
        "OPENMED_SERVICE_AUTH_API_KEYS",
        "OPENMED_SERVICE_AUTH_JWKS",
        "OPENMED_SERVICE_AUTH_JWKS_FILE",
        "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
        "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS",
        "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST",
        "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_KEY",
        "OPENMED_SERVICE_MTLS_ENABLED",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", _FakeLoader)


class _FakeLoader:
    """Model loader double so throttled routes never load real models."""

    def __init__(self, config: Any):
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        return object()

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


def _enable_api_key_auth(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scopes: list[str],
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [
                {
                    "id": "synthetic-key",
                    "key_hash": hash_api_key(SYNTHETIC_API_KEY),
                    "principal": "synthetic-browser-client",
                    "scopes": scopes,
                }
            ]
        ),
    )


def _middleware_names(app) -> list[str]:
    names = []
    for middleware in app.user_middleware:
        dispatch = middleware.kwargs.get("dispatch")
        names.append(dispatch.__name__ if dispatch else middleware.cls.__name__)
    return names


def _assert_allowed_origin_cors(response) -> None:
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    exposed = response.headers["access-control-expose-headers"].lower()
    assert "x-request-id" in exposed
    assert "retry-after" in exposed


def test_security_config_defaults_to_no_cors_and_loopback_hosts() -> None:
    config = parse_service_security_config()

    assert config.cors_origins == ()
    assert config.trusted_hosts == DEFAULT_TRUSTED_HOSTS


def test_security_config_parses_deduped_allowlists(monkeypatch) -> None:
    monkeypatch.setenv(
        "OPENMED_SERVICE_CORS_ORIGINS",
        " https://app.example.com,https://app.example.com, http://localhost:3000 ",
    )
    monkeypatch.setenv(
        "OPENMED_SERVICE_TRUSTED_HOSTS",
        " api.example.com,localhost,api.example.com ",
    )

    config = parse_service_security_config()

    assert config.cors_origins == (
        "https://app.example.com",
        "http://localhost:3000",
    )
    assert config.trusted_hosts == ("api.example.com", "localhost")


def test_security_config_rejects_wildcard_cors_origin() -> None:
    with pytest.raises(ValueError, match="exact origins"):
        parse_cors_origins("https://app.example.com,*")


def test_security_config_rejects_global_trusted_host() -> None:
    with pytest.raises(ValueError, match="global"):
        parse_trusted_hosts("localhost,*")


def test_no_cors_env_does_not_grant_cross_origin_preflight() -> None:
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        response = client.options(
            "/health",
            headers={
                "Origin": "https://frontend.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.headers.get("access-control-allow-origin") is None
    assert response.headers.get("access-control-allow-origin") != "*"


def test_configured_origin_receives_matching_cors_header(monkeypatch) -> None:
    monkeypatch.setenv(
        "OPENMED_SERVICE_CORS_ORIGINS",
        "https://frontend.example.com,http://localhost:5173",
    )
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        response = client.options(
            "/pii/extract",
            headers={
                "Origin": "https://frontend.example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type, X-Request-ID",
            },
        )

    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "https://frontend.example.com"
    )
    assert "*" not in response.headers["access-control-allow-origin"]
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "x-request-id" in response.headers["access-control-allow-headers"].lower()


def test_non_allowlisted_origin_is_not_reflected(monkeypatch) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_CORS_ORIGINS", "https://allowed.example.com")
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        response = client.options(
            "/pii/extract",
            headers={
                "Origin": "https://attacker.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.headers.get("access-control-allow-origin") is None
    assert response.headers.get("access-control-allow-origin") != "*"


def test_disallowed_host_returns_standard_error_envelope() -> None:
    app = create_app()

    with TestClient(
        app,
        base_url="http://attacker.example.com",
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/health")

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "bad_request",
            "message": "Invalid host header",
            "details": {"reason": "Invalid host header"},
        }
    }


def test_loopback_host_passes_by_default() -> None:
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_configured_trusted_host_passes(monkeypatch) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_TRUSTED_HOSTS", "api.example.com")
    app = create_app()

    with TestClient(app, base_url="http://api.example.com") as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_cors_allow_headers_add_credentials_only_when_auth_is_enabled() -> None:
    base_headers = security_headers.SERVICE_CORS_HEADERS
    auth_headers = security_headers.SERVICE_CORS_AUTH_HEADERS
    expose_headers = security_headers.SERVICE_CORS_EXPOSE_HEADERS
    allow_headers = security_headers.service_cors_allow_headers

    assert "X-OpenMed-Priority" in base_headers
    assert allow_headers(auth_enabled=False) == base_headers
    with_auth = allow_headers(auth_enabled=True)
    assert set(auth_headers) == {"Authorization", "X-API-Key"}
    assert set(with_auth) == set(base_headers) | set(auth_headers)
    assert expose_headers == ("X-Request-ID", "Retry-After")
    for header_list in (with_auth, expose_headers):
        assert "*" not in header_list


def test_create_app_rejects_wildcard_cors_origin(monkeypatch) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_CORS_ORIGINS", "*")

    with pytest.raises(ValueError, match="exact origins"):
        create_app()


def test_middleware_stack_validates_host_cors_and_auth_before_bodies(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_CORS_ORIGINS", ALLOWED_ORIGIN)
    app = create_app()

    # Starlette runs user_middleware[0] outermost.
    assert _middleware_names(app) == [
        "ErrorEnvelopeTrustedHostMiddleware",
        "CORSMiddleware",
        "_auth_middleware",
        "WorkflowBoundaryMiddleware",
        "BoundedRequestBodyMiddleware",
        "CorrelationIdMiddleware",
        "_BoundedRequestBodyMiddleware",
        "_throttle_middleware",
        "_readiness_middleware",
    ]


def test_middleware_stack_keeps_trusted_host_outermost_without_cors() -> None:
    names = _middleware_names(create_app())

    assert names[:2] == ["ErrorEnvelopeTrustedHostMiddleware", "_auth_middleware"]
    assert "CORSMiddleware" not in names


def test_allowed_origin_preflight_passes_without_credentials(monkeypatch) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_CORS_ORIGINS", ALLOWED_ORIGIN)
    _enable_api_key_auth(monkeypatch, scopes=["pii:read"])
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        response = client.options(
            "/pii/extract",
            headers={
                "Origin": ALLOWED_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": (
                    "Authorization, X-API-Key, X-OpenMed-Priority, "
                    "Content-Type, X-Request-ID"
                ),
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    allowed = response.headers["access-control-allow-headers"].lower()
    for header in ("authorization", "x-api-key", "x-openmed-priority"):
        assert header in allowed
    assert "*" not in response.headers["access-control-allow-origin"]


def test_disallowed_origin_receives_no_cors_grant_with_auth(monkeypatch) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_CORS_ORIGINS", ALLOWED_ORIGIN)
    _enable_api_key_auth(monkeypatch, scopes=["pii:read"])
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        preflight = client.options(
            "/pii/extract",
            headers={
                "Origin": DISALLOWED_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization",
            },
        )
        rejected = client.post(
            "/pii/extract",
            headers={"Origin": DISALLOWED_ORIGIN},
            json={"text": "synthetic"},
        )

    assert preflight.status_code != 200
    assert rejected.status_code == 401
    for response in (preflight, rejected):
        assert response.headers.get("access-control-allow-origin") is None
        assert response.headers.get("access-control-allow-credentials") is None


def test_untrusted_host_is_rejected_before_auth_failure_limiter(monkeypatch) -> None:
    _enable_api_key_auth(monkeypatch, scopes=["models:read"])
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS", "0.01")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST", "1")
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        host_rejected = client.get("/models/loaded", headers=UNTRUSTED_HOST)
        unauthenticated = client.get("/models/loaded")

    assert host_rejected.status_code == 400
    assert host_rejected.json()["error"]["code"] == "bad_request"
    assert "www-authenticate" not in host_rejected.headers
    # The host rejection must not have spent the single auth-failure token.
    assert unauthenticated.status_code == 401
    assert unauthenticated.json()["error"]["code"] == "authentication_required"


def test_untrusted_host_is_rejected_before_throttle_consumes_token(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_RPS", "0.01")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_BURST", "1")
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        host_rejected = client.post(
            "/analyze", headers=UNTRUSTED_HOST, json={"text": " "}
        )
        first_trusted = client.post("/analyze", json={"text": " "})
        second_trusted = client.post("/analyze", json={"text": " "})

    assert host_rejected.status_code == 400
    assert host_rejected.json()["error"]["code"] == "bad_request"
    assert first_trusted.status_code == 422
    assert second_trusted.status_code == 429


def test_unauthenticated_request_is_rejected_before_body_is_read(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_CORS_ORIGINS", ALLOWED_ORIGIN)
    _enable_api_key_auth(monkeypatch, scopes=["pii:read"])
    app = create_app()
    receive_calls = 0
    sent: list[Message] = []

    async def stalled_receive() -> Message:
        nonlocal receive_calls
        receive_calls += 1
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(message: Message) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/pii/extract",
        "raw_path": b"/pii/extract",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"127.0.0.1"),
            (b"origin", ALLOWED_ORIGIN.encode("ascii")),
            (b"content-type", b"application/json"),
            (b"content-length", b"1024"),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 80),
    }

    async def run() -> None:
        await asyncio.wait_for(app(scope, stalled_receive, send), timeout=2.0)

    asyncio.run(run())

    assert receive_calls == 0
    start = next(m for m in sent if m["type"] == "http.response.start")
    headers = {key.decode().lower(): value.decode() for key, value in start["headers"]}
    body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    assert start["status"] == 401
    assert json.loads(body)["error"]["code"] == "authentication_required"
    assert headers["access-control-allow-origin"] == ALLOWED_ORIGIN


def test_auth_errors_to_allowed_origin_carry_cors_headers(monkeypatch) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_CORS_ORIGINS", ALLOWED_ORIGIN)
    _enable_api_key_auth(monkeypatch, scopes=["models:read"])
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS", "0.01")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST", "1")
    app = create_app()
    origin = {"Origin": ALLOWED_ORIGIN}

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        forbidden = client.post(
            "/pii/extract",
            headers={**origin, "X-API-Key": SYNTHETIC_API_KEY},
            json={"text": "synthetic"},
        )
        unauthenticated = client.post(
            "/pii/extract", headers=origin, json={"text": "synthetic"}
        )
        limited = client.post(
            "/pii/extract", headers=origin, json={"text": "synthetic"}
        )

    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "forbidden"
    assert unauthenticated.status_code == 401
    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "auth_rate_limited"
    assert int(limited.headers["Retry-After"]) >= 1
    for response in (forbidden, unauthenticated, limited):
        _assert_allowed_origin_cors(response)


def test_body_limit_error_to_allowed_origin_carries_cors_headers(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_CORS_ORIGINS", ALLOWED_ORIGIN)
    _enable_api_key_auth(monkeypatch, scopes=["pii:read"])
    app = create_app(max_request_body_bytes=64)

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/pii/extract",
            headers={"Origin": ALLOWED_ORIGIN, "X-API-Key": SYNTHETIC_API_KEY},
            json={"text": "synthetic " * 32},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    _assert_allowed_origin_cors(response)


def test_throttle_error_to_allowed_origin_carries_cors_headers(monkeypatch) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_CORS_ORIGINS", ALLOWED_ORIGIN)
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_RPS", "0.01")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_BURST", "1")
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        first = client.post(
            "/analyze", headers={"Origin": ALLOWED_ORIGIN}, json={"text": " "}
        )
        limited = client.post(
            "/analyze", headers={"Origin": ALLOWED_ORIGIN}, json={"text": " "}
        )

    assert first.status_code == 422
    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "rate_limited"
    _assert_allowed_origin_cors(limited)
