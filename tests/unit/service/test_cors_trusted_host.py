"""Tests for REST CORS and trusted-host middleware."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from openmed.service.app import create_app
from openmed.service.security_headers import (
    DEFAULT_TRUSTED_HOSTS,
    parse_cors_origins,
    parse_service_security_config,
    parse_trusted_hosts,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"


@pytest.fixture(autouse=True)
def service_env(monkeypatch):
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_KEEP_ALIVE", raising=False)
    monkeypatch.delenv("OPENMED_SERVICE_MAX_RESIDENT_MODELS", raising=False)
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
