"""Tests for the fail-closed service security-profile self-check."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from openmed.cli.main import main as cli_main
from openmed.service import runtime as service_runtime
from openmed.service import security_profile as sp
from openmed.service.app import create_app
from openmed.service.auth import ServiceAuth, hash_api_key, parse_service_auth_config
from openmed.service.security_headers import ServiceSecurityConfig

ROOT = Path(__file__).resolve().parents[3]
LOOPBACK_BASE_URL = "http://127.0.0.1"
SYNTHETIC_API_KEY = "synthetic-preflight-key-0001"
_HS_SECRET = b"synthetic-hs256-secret-for-security-profile-tests"

_ENV_VARS = (
    "OPENMED_PROFILE",
    "OPENMED_SERVICE_SECURITY_PROFILE",
    "OPENMED_SERVICE_LISTEN_HOST",
    "OPENMED_SERVICE_LISTEN_TLS",
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT",
    "OPENMED_SERVICE_AUTH_API_KEYS",
    "OPENMED_SERVICE_AUTH_JWKS",
    "OPENMED_SERVICE_AUTH_JWKS_FILE",
    "OPENMED_SERVICE_AUTH_JWT_ISSUER",
    "OPENMED_SERVICE_AUTH_JWT_AUDIENCE",
    "OPENMED_SERVICE_AUTH_JWT_MAX_LIFETIME_SECONDS",
    "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
    "OPENMED_SERVICE_MTLS_ENABLED",
    "OPENMED_SERVICE_CORS_ORIGINS",
    "OPENMED_SERVICE_TRUSTED_HOSTS",
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_SERVED_MODELS",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
    "OPENMED_SERVICE_PRIVACY_GATEWAY_ENDPOINT",
    "OPENMED_SERVICE_TRACING_ENABLED",
    "OPENMED_SERVICE_OTLP_ENDPOINT",
    "OPENMED_OPENHIM_MEDIATOR_ENABLED",
    "OPENMED_JOURNEY_POSTGRES_DSN",
)


class FakeLoader:
    """Model loader double that records construction."""

    instances: list["FakeLoader"] = []

    def __init__(self, config: Any) -> None:
        self.config = config
        FakeLoader.instances.append(self)

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        return object()

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    FakeLoader.instances = []
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)


@pytest.fixture
def restore_registry():
    before = sp.registered_security_checks()
    yield
    for check in sp.registered_security_checks():
        sp.unregister_security_check(check.check_id)
    for check in before:
        sp.register_security_check(check)


def _api_keys(principal: str = "synthetic-service", key_id: str = "key-1") -> str:
    return json.dumps(
        [
            {
                "id": key_id,
                "key_hash": hash_api_key(SYNTHETIC_API_KEY),
                "principal": principal,
                "scopes": ["models:read", "pii:read"],
            }
        ]
    )


def _jwks(kid: str = "hs-synthetic", secret: bytes = _HS_SECRET) -> str:
    encoded = base64.urlsafe_b64encode(secret).rstrip(b"=").decode("ascii")
    return json.dumps({"keys": [{"kty": "oct", "kid": kid, "k": encoded}]})


def _configure_production(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply a fully configured synthetic production profile."""
    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_API_KEYS", _api_keys())
    monkeypatch.setenv("OPENMED_SERVICE_SERVED_MODELS", "pii_detection_superclinical")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("OPENMED_SERVICE_LISTEN_HOST", "127.0.0.1")
    monkeypatch.setenv("OPENMED_SERVICE_LISTEN_TLS", "false")


def _result(report: sp.SecurityProfileReport, check_id: str) -> sp.CheckResult:
    matches = [item for item in report.results if item.check_id == check_id]
    assert len(matches) == 1
    return matches[0]


def _assert_invalid(response, *, details: bool) -> dict[str, Any]:
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "security_profile_invalid"
    assert (error["details"] is not None) is details
    return error


# --- resolver ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("environ", "name", "source"),
    [
        ({}, "prod", "default"),
        ({"OPENMED_PROFILE": "prod"}, "prod", "config_profile_env"),
        ({"OPENMED_PROFILE": "dev"}, "dev", "config_profile_env"),
        ({"OPENMED_PROFILE": "test"}, "dev", "config_profile_env"),
        ({"OPENMED_PROFILE": "fast"}, "dev", "config_profile_env"),
        ({"OPENMED_PROFILE": "low_resource"}, "prod", "config_profile_env"),
        ({"OPENMED_PROFILE": "custom-site"}, "prod", "config_profile_env"),
        (
            {"OPENMED_PROFILE": "prod", "OPENMED_SERVICE_SECURITY_PROFILE": "dev"},
            "dev",
            "security_profile_env",
        ),
        (
            {"OPENMED_PROFILE": "dev", "OPENMED_SERVICE_SECURITY_PROFILE": " PROD "},
            "prod",
            "security_profile_env",
        ),
    ],
)
def test_single_resolver_maps_environment_to_security_profile(
    environ: dict[str, str], name: str, source: str
) -> None:
    resolved = sp.resolve_security_profile(environ)

    assert resolved.name.value == name
    assert resolved.source.value == source
    assert resolved.is_production is (name == "prod")


def test_explicit_override_wins_and_profiles_are_versioned() -> None:
    resolved = sp.resolve_security_profile({"OPENMED_PROFILE": "prod"}, override="dev")

    assert resolved.profile is sp.DEV_PROFILE
    assert resolved.source is sp.ProfileSource.EXPLICIT
    assert sp.PROD_PROFILE.identifier == "openmed.service.security_profile.prod.v1"
    assert sp.PROD_PROFILE.enforcing is True
    assert sp.DEV_PROFILE.enforcing is False


def test_unknown_explicit_profile_is_rejected_without_echo() -> None:
    sentinel = "sentinel-profile-value-7731"

    with pytest.raises(ValueError) as caught:
        sp.resolve_security_profile({"OPENMED_SERVICE_SECURITY_PROFILE": sentinel})

    assert sentinel not in str(caught.value)
    assert "OPENMED_SERVICE_SECURITY_PROFILE" in str(caught.value)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"OPENMED_SERVICE_SECURITY_PROFILE": "dev"}, None),
        ({"OPENMED_PROFILE": "test"}, None),
        ({"OPENMED_PROFILE": "low_resource"}, 3600.0),
        ({}, 3600.0),
    ],
)
def test_jwt_lifetime_default_follows_the_resolver(
    monkeypatch: pytest.MonkeyPatch, environ: dict[str, str], expected: float | None
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_JWKS", _jwks())
    for name, value in environ.items():
        monkeypatch.setenv(name, value)

    assert parse_service_auth_config().jwt_max_lifetime_seconds == expected


# --- startup, readiness and refusal -----------------------------------------


def test_fully_configured_synthetic_production_profile_starts_and_becomes_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_production(monkeypatch)
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        ready = client.get("/readyz")
        unauthenticated = client.get("/models/loaded")
        authenticated = client.get(
            "/models/loaded", headers={"x-api-key": SYNTHETIC_API_KEY}
        )

    report = app.state.security_profile_report
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "service": "openmed-rest"}
    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200
    assert report.resolved.is_production
    assert report.ready and not report.blocking
    assert report.exit_code == sp.EXIT_PROFILE_PASSED
    # Controls that cannot be configured on this build stay visible.
    assert report.status is sp.ReportStatus.WARN
    assert _result(report, "graphql.transport").status is sp.CheckStatus.NOT_IMPLEMENTED
    assert not _result(report, "graphql.transport").blocking


def test_invalid_production_profile_refuses_work_and_keeps_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_production(monkeypatch)
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "false")
    monkeypatch.setenv("OPENMED_SERVICE_PRELOAD_MODELS", "pii_detection_superclinical")
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        livez = client.get("/livez")
        health = client.get("/health")
        readyz = client.get("/readyz")
        extract = client.post("/pii/extract", json={"text": "synthetic note"})
        control = client.get("/models/loaded")
        graphql = client.post("/graphql", json={"query": "{ __typename }"})
        unknown = client.get("/not-a-route")

    assert livez.status_code == 200
    assert health.status_code == 200
    error = _assert_invalid(readyz, details=True)
    assert error["details"]["profile"] == "prod"
    assert error["details"]["profile_version"] == 1
    assert error["details"]["blocking_checks"] == ["auth.enabled"]
    assert error["details"]["report_digest"].startswith("sha256:")
    _assert_invalid(extract, details=False)
    _assert_invalid(control, details=False)
    _assert_invalid(graphql, details=False)
    assert unknown.status_code == 404
    assert app.state.ready is False
    assert FakeLoader.instances == []  # No preload under an invalid profile.


def test_gate_applies_when_the_lifespan_never_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app()
    client = TestClient(app, base_url=LOOPBACK_BASE_URL)
    try:
        readyz = client.get("/readyz")
        extract = client.post("/pii/extract", json={"text": "synthetic note"})
    finally:
        client.close()

    _assert_invalid(readyz, details=True)
    _assert_invalid(extract, details=False)


def test_development_profile_keeps_insecure_configuration_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "dev")
    monkeypatch.setenv("OPENMED_SERVICE_LISTEN_HOST", "0.0.0.0")
    monkeypatch.setenv(
        "OPENMED_JOURNEY_POSTGRES_DSN", "postgresql://user@db.example.test/openmed"
    )
    monkeypatch.setenv("OPENMED_SERVICE_CORS_ORIGINS", "http://ui.example.test")
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        readyz = client.get("/readyz")
        loaded = client.get("/models/loaded")

    report = app.state.security_profile_report
    assert readyz.status_code == 200
    assert loaded.status_code == 200
    assert not report.blocking
    assert report.status is sp.ReportStatus.WARN
    assert {item.level for item in report.results} == {sp.CheckLevel.ADVISORY}
    assert _result(report, "auth.enabled").status is sp.CheckStatus.FAIL
    assert _result(report, "journey.store_tls").status is sp.CheckStatus.FAIL


# --- one focused failing test per unsafe production configuration -----------

_JOURNEY_STUBS = {"journey.principal_binding", "journey.encryption_at_rest"}


@pytest.mark.parametrize(
    ("env", "check_id", "reason", "blocking"),
    [
        (
            {"OPENMED_SERVICE_AUTH_ENABLED": "false"},
            "auth.enabled",
            "auth_disabled",
            {"auth.enabled"},
        ),
        (
            {"OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT": "false"},
            "auth.deny_by_default",
            "deny_by_default_disabled",
            {"auth.deny_by_default"},
        ),
        (
            {"OPENMED_SERVICE_AUTH_API_KEYS": ""},
            "auth.credentials",
            "credentials_missing",
            {"auth.credentials"},
        ),
        (
            {
                "OPENMED_SERVICE_AUTH_JWKS": _jwks(),
                "OPENMED_SERVICE_AUTH_JWT_AUDIENCE": "openmed-api",
            },
            "auth.jwt_claims",
            "jwt_issuer_missing",
            {"auth.jwt_claims"},
        ),
        (
            {
                "OPENMED_SERVICE_AUTH_JWKS": _jwks(),
                "OPENMED_SERVICE_AUTH_JWT_ISSUER": "https://issuer.example.test",
            },
            "auth.jwt_claims",
            "jwt_audience_missing",
            {"auth.jwt_claims"},
        ),
        (
            {
                "OPENMED_SERVICE_AUTH_JWKS": _jwks(),
                "OPENMED_SERVICE_AUTH_JWT_ISSUER": "https://issuer.example.test",
                "OPENMED_SERVICE_AUTH_JWT_AUDIENCE": "openmed-api",
                "OPENMED_SERVICE_AUTH_JWT_MAX_LIFETIME_SECONDS": "172800",
            },
            "auth.jwt_claims",
            "jwt_lifetime_excessive",
            {"auth.jwt_claims"},
        ),
        (
            {"OPENMED_SERVICE_CORS_ORIGINS": "http://ui.example.test"},
            "http.cors",
            "cors_origin_not_https",
            {"http.cors"},
        ),
        (
            {"OPENMED_SERVICE_CORS_ORIGINS": "https://ui.example.test/app"},
            "http.cors",
            "cors_origin_not_https",
            {"http.cors"},
        ),
        (
            {"OPENMED_JOURNEY_POSTGRES_DSN": "postgresql://u@db.example.test/openmed"},
            "journey.store_tls",
            "postgres_tls_not_configured",
            {"journey.store_tls", *_JOURNEY_STUBS},
        ),
        (
            {
                "OPENMED_JOURNEY_POSTGRES_DSN": (
                    "postgresql://u@db.example.test/openmed?sslmode=require"
                )
            },
            "journey.store_tls",
            "postgres_tls_not_verified",
            {"journey.store_tls", *_JOURNEY_STUBS},
        ),
        (
            {"OPENMED_JOURNEY_POSTGRES_DSN": "mysql://u@db.example.test/openmed"},
            "journey.store_tls",
            "postgres_dsn_invalid",
            {"journey.store_tls", *_JOURNEY_STUBS},
        ),
        (
            {
                "OPENMED_JOURNEY_POSTGRES_DSN": (
                    "postgresql://u@db.example.test/openmed?sslmode=verify-full"
                )
            },
            "journey.encryption_at_rest",
            "control_not_implemented",
            _JOURNEY_STUBS,
        ),
        (
            {"OPENMED_SERVICE_PRIVACY_GATEWAY_ENDPOINT": "https://llm.example.test"},
            "egress.configured_adapters",
            "control_not_implemented",
            {"egress.configured_adapters"},
        ),
    ],
    ids=[
        "auth-disabled",
        "deny-by-default-off",
        "no-credentials",
        "jwt-no-issuer",
        "jwt-no-audience",
        "jwt-lifetime-excessive",
        "cors-plain-http",
        "cors-with-path",
        "postgres-no-tls",
        "postgres-tls-unverified",
        "postgres-dsn-invalid",
        "journey-encryption-missing",
        "egress-adapter-without-policy",
    ],
)
def test_unsafe_production_configuration_blocks_readiness(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    check_id: str,
    reason: str,
    blocking: set[str],
) -> None:
    _configure_production(monkeypatch)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        readyz = client.get("/readyz")

    report = app.state.security_profile_report
    result = _result(report, check_id)
    assert result.reason == reason
    assert result.blocking
    assert set(report.blocking_checks) == blocking
    assert report.exit_code == sp.EXIT_PROFILE_INVALID
    error = _assert_invalid(readyz, details=True)
    assert set(error["details"]["blocking_checks"]) == blocking


def _evaluate(app: Any, **kwargs: Any) -> sp.SecurityProfileReport:
    return sp.evaluate_service_security_profile(app, **kwargs)


def test_unbounded_jwt_lifetime_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_production(monkeypatch)
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_JWKS", _jwks())
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_JWT_ISSUER", "https://issuer.example.test")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_JWT_AUDIENCE", "openmed-api")
    app = create_app()
    assert _result(_evaluate(app), "auth.jwt_claims").status is sp.CheckStatus.PASS

    config = replace(app.state.auth.config, jwt_max_lifetime_seconds=None)
    app.state.auth = ServiceAuth(config, error_response=app.state.auth._error_response)

    result = _result(_evaluate(app), "auth.jwt_claims")
    assert (result.reason, result.blocking) == ("jwt_lifetime_unbounded", True)


def test_unrestricted_trusted_hosts_block(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_production(monkeypatch)
    app = create_app()
    app.state.security = ServiceSecurityConfig(trusted_hosts=("*",))

    result = _result(_evaluate(app), "http.trusted_hosts")
    assert (result.reason, result.blocking) == ("trusted_hosts_unrestricted", True)


def test_missing_request_body_limit_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_production(monkeypatch)
    app = create_app()
    app.state.model_body_limit_bytes = None

    result = _result(_evaluate(app), "limits.request_body")
    assert (result.reason, result.blocking) == ("request_body_limit_missing", True)


def test_undeclared_route_policy_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_production(monkeypatch)
    app = create_app()

    @app.get("/undeclared-route")
    async def undeclared() -> dict[str, str]:
        return {}

    result = _result(_evaluate(app), "auth.route_policy")
    assert (result.reason, result.blocking) == ("route_policy_undeclared", True)


@pytest.mark.parametrize(
    "state",
    [
        {"journey_resources": SimpleNamespace(record_count=1)},
        {"openhim_settings": SimpleNamespace(enabled=True)},
        {
            "tracing": SimpleNamespace(
                config=SimpleNamespace(
                    enabled=True, otlp_endpoint="https://otel.example.test"
                )
            )
        },
    ],
    ids=["journey-catalog-records", "openhim-mediator", "otlp-exporter"],
)
def test_enabled_feature_without_its_control_blocks(
    monkeypatch: pytest.MonkeyPatch, state: dict[str, Any]
) -> None:
    _configure_production(monkeypatch)
    app = create_app()
    for name, value in state.items():
        setattr(app.state, name, value)

    report = _evaluate(app)

    assert len(report.blocking_checks) == 1
    result = _result(report, report.blocking_checks[0])
    assert result.status is sp.CheckStatus.NOT_IMPLEMENTED
    assert result.tracking_issue in {3549, 3581}


def test_not_implemented_controls_block_only_when_their_feature_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_production(monkeypatch)
    app = create_app()

    disabled = _evaluate(app)
    for check_id in (*_JOURNEY_STUBS, "egress.configured_adapters", "grpc.auth"):
        result = _result(disabled, check_id)
        assert result.status is sp.CheckStatus.NOT_APPLICABLE
        assert not result.blocking
    assert not disabled.blocking

    dsn = "postgresql://u@db.example.test/openmed?sslmode=verify-full"
    enabled = _evaluate(
        app, environ={"OPENMED_PROFILE": "prod", sp.JOURNEY_POSTGRES_DSN_ENV_VAR: dsn}
    )
    assert set(enabled.blocking_checks) == _JOURNEY_STUBS
    assert _result(enabled, "journey.store_tls").status is sp.CheckStatus.PASS

    development = _evaluate(
        app, environ={sp.JOURNEY_POSTGRES_DSN_ENV_VAR: dsn}, profile="dev"
    )
    assert not development.blocking
    assert _result(development, "journey.principal_binding").status is (
        sp.CheckStatus.NOT_IMPLEMENTED
    )


@pytest.mark.parametrize(
    ("address", "transport", "blocking"),
    [
        ("127.0.0.1:50051", "pass", {"grpc.auth"}),
        ("localhost:50051", "pass", {"grpc.auth"}),
        ("[::1]:50051", "pass", {"grpc.auth"}),
        ("[::]:50051", "not_implemented", {"grpc.transport", "grpc.auth"}),
        ("0.0.0.0:50051", "not_implemented", {"grpc.transport", "grpc.auth"}),
    ],
)
def test_grpc_listener_controls(
    monkeypatch: pytest.MonkeyPatch, address: str, transport: str, blocking: set[str]
) -> None:
    _configure_production(monkeypatch)
    report = _evaluate(create_app(), grpc_address=address)

    assert _result(report, "grpc.transport").status.value == transport
    assert set(report.blocking_checks) == blocking


@pytest.mark.parametrize(
    ("listener", "status", "reason"),
    [
        (None, "fail", "listener_unknown"),
        (sp.ListenerFacts("127.0.0.1"), "pass", "loopback_listener"),
        (sp.ListenerFacts("localhost"), "pass", "loopback_listener"),
        (sp.ListenerFacts("::1"), "pass", "loopback_listener"),
        (sp.ListenerFacts("0.0.0.0", tls=True), "pass", "tls_listener"),
        (sp.ListenerFacts("0.0.0.0"), "fail", "plaintext_remote_listener"),
        (sp.ListenerFacts("api.example.test"), "fail", "plaintext_remote_listener"),
    ],
)
def test_listener_transport_is_reported_as_advisory(
    monkeypatch: pytest.MonkeyPatch,
    listener: sp.ListenerFacts | None,
    status: str,
    reason: str,
) -> None:
    _configure_production(monkeypatch)
    monkeypatch.delenv("OPENMED_SERVICE_LISTEN_HOST")
    report = _evaluate(create_app(), listener=listener)
    result = _result(report, "listener.transport")

    assert (result.status.value, result.reason) == (status, reason)
    assert result.level is sp.CheckLevel.ADVISORY
    assert not report.blocking


# --- report contract ---------------------------------------------------------


def test_report_schema_digest_and_text_are_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_production(monkeypatch)
    app = create_app()
    first = _evaluate(app)
    second = _evaluate(app)
    payload = first.to_dict()

    assert set(payload) == {
        "schema_version",
        "profile",
        "status",
        "ready",
        "summary",
        "blocking_checks",
        "checks",
        "digest",
    }
    assert payload["schema_version"] == "openmed.service.security_profile_report.v1"
    assert payload["profile"]["id"] == "openmed.service.security_profile.prod.v1"
    assert [item["id"] for item in payload["checks"]] == [
        check.check_id for check in sp.registered_security_checks()
    ]
    assert {key for item in payload["checks"] for key in item} <= {
        "id",
        "control",
        "level",
        "status",
        "reason",
        "blocking",
        "tracking_issue",
    }
    assert payload["summary"]["checks"] == len(payload["checks"])
    assert first.digest == second.digest == payload["digest"]
    assert json.loads(first.to_json()) == payload
    text = first.to_text()
    assert all(item["id"] in text for item in payload["checks"])
    assert f"Report digest: {first.digest}" in text

    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "false")
    assert _evaluate(create_app()).digest != first.digest


def test_report_readyz_and_logs_pass_secret_and_phi_sentinel_scan(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sentinels = {
        "principal": "principal-sentinel-a91",
        "key_id": "keyid-sentinel-b72",
        "kid": "kid-sentinel-c63",
        "secret": "jwk-secret-sentinel-d54",
        "issuer": "issuer-sentinel-e45",
        "audience": "audience-sentinel-f36",
        "origin": "origin-sentinel-g27",
        "host": "trustedhost-sentinel-h18",
        "db_user": "dbuser-sentinel-i09",
        "db_password": "dbpass-sentinel-j90",
        "db_host": "dbhost-sentinel-k81",
        "gateway": "gateway-sentinel-l72",
        "listener": "listener-sentinel-m63",
        "phi": "Jane Sentinel-Doe MRN 4417-0042",
    }
    _configure_production(monkeypatch)
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        _api_keys(principal=sentinels["principal"], key_id=sentinels["key_id"]),
    )
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_JWKS",
        _jwks(kid=sentinels["kid"], secret=sentinels["secret"].encode()),
    )
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_JWT_ISSUER", sentinels["issuer"])
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_JWT_AUDIENCE", sentinels["audience"])
    monkeypatch.setenv(
        "OPENMED_SERVICE_CORS_ORIGINS", f"http://{sentinels['origin']}.example.test"
    )
    monkeypatch.setenv(
        "OPENMED_SERVICE_TRUSTED_HOSTS", f"127.0.0.1,{sentinels['host']}.example.test"
    )
    monkeypatch.setenv(
        "OPENMED_JOURNEY_POSTGRES_DSN",
        f"postgresql://{sentinels['db_user']}:{sentinels['db_password']}"
        f"@{sentinels['db_host']}.example.test/openmed",
    )
    monkeypatch.setenv(
        "OPENMED_SERVICE_PRIVACY_GATEWAY_ENDPOINT",
        f"https://{sentinels['gateway']}.example.test/v1",
    )
    monkeypatch.setenv("OPENMED_SERVICE_LISTEN_HOST", f"{sentinels['listener']}.test")
    caplog.set_level(logging.DEBUG)
    app = create_app()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        readyz = client.get("/readyz")
        refused = client.post(
            "/pii/extract",
            json={"text": sentinels["phi"]},
            headers={"x-api-key": SYNTHETIC_API_KEY},
        )

    report = app.state.security_profile_report
    assert report.blocking
    assert "security_profile_invalid" in caplog.text
    _assert_invalid(readyz, details=True)
    _assert_invalid(refused, details=False)
    surfaces = (
        report.to_json(),
        report.to_text(),
        json.dumps(report.readiness_details()),
        readyz.text,
        refused.text,
        caplog.text,
    )
    secret_encoded = base64.urlsafe_b64encode(sentinels["secret"].encode())
    for surface in surfaces:
        for value in (*sentinels.values(), secret_encoded.decode().rstrip("=")):
            assert value not in surface
        assert "sentinel" not in surface.lower()


# --- registry ----------------------------------------------------------------


@pytest.mark.usefixtures("restore_registry")
def test_registry_accepts_plugins_and_fails_closed_on_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_production(monkeypatch)
    app = create_app()

    def passing(_: sp.SecurityCheckContext) -> sp.CheckOutcome:
        return sp.CheckOutcome(sp.CheckStatus.PASS, "plugin_ok")

    def raising(_: sp.SecurityCheckContext) -> sp.CheckOutcome:
        raise RuntimeError("configured value sentinel-raise-value")

    def free_text(_: sp.SecurityCheckContext) -> sp.CheckOutcome:
        return sp.CheckOutcome(sp.CheckStatus.FAIL, "Contains sentinel-free-text")

    plugin = sp.SecurityCheck(
        "plugin.example", sp.SecurityControl.LIMITS, sp.CheckLevel.REQUIRED, passing
    )
    sp.register_security_check(plugin)
    with pytest.raises(ValueError):
        sp.register_security_check(plugin)
    with pytest.raises(ValueError):
        sp.SecurityCheck(
            "Not A Dotted Id",
            sp.SecurityControl.LIMITS,
            sp.CheckLevel.REQUIRED,
            passing,
        )
    assert plugin in sp.registered_security_checks()
    assert _result(_evaluate(app), "plugin.example").status is sp.CheckStatus.PASS

    sp.register_security_check(replace(plugin, evaluate=raising), replace=True)
    sp.register_security_check(
        sp.SecurityCheck(
            "plugin.free_text",
            sp.SecurityControl.LIMITS,
            sp.CheckLevel.ADVISORY,
            free_text,
        )
    )
    report = _evaluate(app)

    raised = _result(report, "plugin.example")
    assert (raised.status, raised.reason, raised.blocking) == (
        sp.CheckStatus.FAIL,
        "check_error",
        True,
    )
    assert _result(report, "plugin.free_text").reason == "check_error"
    assert "sentinel" not in report.to_json()


# --- CLI preflight ------------------------------------------------------------


def _preflight(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, Any]:
    code = cli_main(["service", "preflight", "--json", *args])
    return code, json.loads(capsys.readouterr().out)


def test_cli_preflight_exit_codes_match_the_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, envelope = _preflight(capsys)
    assert code == 1
    assert envelope["ok"] is True
    assert envelope["command"] == "service preflight"
    assert envelope["data"]["status"] == "fail"
    assert envelope["data"]["profile"]["source"] == "default"
    assert "auth.enabled" in envelope["data"]["blocking_checks"]

    code, envelope = _preflight(capsys, "--profile", "dev")
    assert code == 0
    assert envelope["data"]["profile"]["name"] == "dev"
    assert envelope["data"]["profile"]["source"] == "explicit"
    assert "OPENMED_SERVICE_SECURITY_PROFILE" not in os.environ

    _configure_production(monkeypatch)
    code, envelope = _preflight(capsys, "--host", "0.0.0.0", "--tls")
    assert code == 0
    assert envelope["data"]["ready"] is True
    listener = next(
        item
        for item in envelope["data"]["checks"]
        if item["id"] == "listener.transport"
    )
    assert listener["reason"] == "tls_listener"

    code, envelope = _preflight(capsys, "--grpc-address", "[::]:50051")
    assert code == 1
    assert set(envelope["data"]["blocking_checks"]) == {"grpc.transport", "grpc.auth"}


def test_cli_preflight_matches_startup_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _configure_production(monkeypatch)
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "false")
    app = create_app()
    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        details = client.get("/readyz").json()["error"]["details"]

    code, envelope = _preflight(capsys)

    assert code == 1
    assert envelope["data"]["digest"] == details["report_digest"]
    assert envelope["data"]["blocking_checks"] == details["blocking_checks"]


def test_cli_preflight_human_output_and_unreadable_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _configure_production(monkeypatch)
    assert cli_main(["service", "preflight"]) == 0
    human = capsys.readouterr().out
    assert human.startswith("OpenMed service security profile: prod v1")
    assert "Report digest: sha256:" in human

    monkeypatch.setenv("OPENMED_SERVICE_AUTH_API_KEYS", "{not-json sentinel-cfg-55")
    code = cli_main(["service", "preflight", "--json"])
    output = capsys.readouterr()
    envelope = json.loads(output.out)

    assert code == 2
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "security_profile_unreadable"
    assert "sentinel" not in output.out + output.err


# --- launcher and deployment manifests ---------------------------------------


@pytest.mark.parametrize(
    ("argv", "host", "tls"),
    [
        (["--host", "0.0.0.0"], "0.0.0.0", "false"),
        (
            ["--ssl-certfile", "cert.pem", "--ssl-keyfile", "key.pem"],
            "127.0.0.1",
            "true",
        ),
    ],
)
def test_reference_launcher_exports_listener_facts(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], host: str, tls: str
) -> None:
    import uvicorn

    from openmed.service import logging as service_logging

    captured: dict[str, Any] = {}
    monkeypatch.setenv(sp.LISTEN_HOST_ENV_VAR, "unset")
    monkeypatch.setenv(sp.LISTEN_TLS_ENV_VAR, "unset")
    monkeypatch.setattr(uvicorn, "run", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["openmed.service.logging", *argv])

    service_logging.main()

    assert captured["host"] == host
    facts = sp.listener_facts_from_env()
    assert facts is not None
    assert (facts.host, str(facts.tls).lower()) == (host, tls)


def test_deployment_images_start_through_the_checked_launcher() -> None:
    for name in ("Dockerfile", "Dockerfile.distroless"):
        dockerfile = (ROOT / "deploy" / "docker" / name).read_text(encoding="utf-8")
        assert "OPENMED_PROFILE=prod" in dockerfile
        assert '"-m", "openmed.service.logging"' in dockerfile
    compose = (ROOT / "deploy" / "openmed-compose.yaml").read_text(encoding="utf-8")
    assert "http://127.0.0.1:8080/readyz" in compose
    values = (ROOT / "deploy" / "helm" / "openmed-service" / "values.yaml").read_text(
        encoding="utf-8"
    )
    assert "path: /readyz" in values
