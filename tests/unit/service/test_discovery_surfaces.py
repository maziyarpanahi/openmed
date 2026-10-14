"""Unauthenticated discovery surfaces of the REST service by profile."""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.auth import hash_api_key

LOOPBACK_BASE_URL = "http://127.0.0.1"
DOCS_PATHS = ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect")
METADATA_PATH = "/service/metadata"
API_KEY = "synthetic-discovery-key"
CDN_MARKERS = ("cdn.", "jsdelivr", "unpkg", "https://")

_SERVICE_ENV_VARS = (
    "OPENMED_PROFILE",
    "OPENMED_SERVICE_OPENAPI_ENABLED",
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_CORS_ORIGINS",
    "OPENMED_SERVICE_TRUSTED_HOSTS",
    "OPENMED_SERVICE_METRICS_ENABLED",
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT",
    "OPENMED_SERVICE_AUTH_API_KEYS",
    "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
)


class FakeLoader:
    """Minimal model loader double; discovery routes never load models."""

    def __init__(self, config: Any):
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        return object()


@pytest.fixture(autouse=True)
def clean_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in _SERVICE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: Optional[str],
    auth: bool = False,
    openapi: Optional[str] = None,
) -> None:
    if profile is not None:
        monkeypatch.setenv("OPENMED_PROFILE", profile)
    if openapi is not None:
        monkeypatch.setenv("OPENMED_SERVICE_OPENAPI_ENABLED", openapi)
    if auth:
        monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
        monkeypatch.setenv(
            "OPENMED_SERVICE_AUTH_API_KEYS",
            json.dumps(
                [
                    {
                        "id": "synthetic-discovery",
                        "key_hash": hash_api_key(API_KEY),
                        "principal": "synthetic-operator",
                        "scopes": ["models:read"],
                    }
                ]
            ),
        )


def _get_all(paths: tuple[str, ...], **headers: str) -> dict[str, Any]:
    with TestClient(
        create_app(),
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        return {path: client.get(path, headers=headers) for path in paths}


@pytest.mark.parametrize("profile", [None, "prod", "test"])
@pytest.mark.parametrize("auth", [False, True])
def test_non_development_profiles_hide_docs_and_openapi(
    monkeypatch: pytest.MonkeyPatch,
    profile: Optional[str],
    auth: bool,
) -> None:
    _configure(monkeypatch, profile=profile, auth=auth)
    unknown = "/synthetic-unregistered-path"

    anonymous = _get_all((*DOCS_PATHS, unknown))
    authorized = _get_all(DOCS_PATHS, **{"X-API-Key": API_KEY})

    for path in DOCS_PATHS:
        # Deny-by-default answers anonymous callers with the same 401 it uses
        # for any unregistered path, so the docs routes are indistinguishable.
        assert anonymous[path].status_code == anonymous[unknown].status_code, path
        assert anonymous[path].status_code == (401 if auth else 404), path
        assert authorized[path].status_code == 404, path
        for response in (anonymous[path], authorized[path]):
            assert all(marker not in response.text for marker in CDN_MARKERS), path


def test_production_probe_and_health_bodies_reference_no_cdn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, profile="prod", auth=True)

    responses = _get_all(("/health", "/livez", "/readyz", METADATA_PATH))

    for path, response in responses.items():
        assert all(marker not in response.text for marker in CDN_MARKERS), path


@pytest.mark.parametrize("profile", [None, "prod", "test"])
@pytest.mark.parametrize("auth", [False, True])
def test_unauthenticated_health_omits_version_and_profile(
    monkeypatch: pytest.MonkeyPatch,
    profile: Optional[str],
    auth: bool,
) -> None:
    _configure(monkeypatch, profile=profile, auth=auth)

    response = _get_all(("/health",))["/health"]

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "openmed-rest"}
    assert openmed.__version__ not in response.text
    assert (profile or "prod") not in response.text


@pytest.mark.parametrize("path", ["/livez", "/readyz", "/health"])
def test_probes_stay_unauthenticated_under_production_auth(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    _configure(monkeypatch, profile="prod", auth=True)

    response = _get_all((path,))[path]

    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "ready"}


def test_metadata_route_requires_auth_and_reports_version_and_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, profile="prod", auth=True)

    anonymous = _get_all((METADATA_PATH,))[METADATA_PATH]
    wrong = _get_all((METADATA_PATH,), **{"X-API-Key": "wrong-key"})[METADATA_PATH]
    authorized = _get_all((METADATA_PATH,), **{"X-API-Key": API_KEY})[METADATA_PATH]

    assert anonymous.status_code == 401
    assert anonymous.json()["error"]["code"] == "authentication_required"
    assert "WWW-Authenticate" in anonymous.headers
    assert openmed.__version__ not in anonymous.text
    assert wrong.status_code == 401
    assert openmed.__version__ not in wrong.text
    assert authorized.status_code == 200
    assert authorized.json() == {
        "service": "openmed-rest",
        "version": openmed.__version__,
        "profile": "prod",
    }


def test_metadata_route_requires_auth_without_deny_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, profile="prod", auth=True)
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT", "false")

    response = _get_all((METADATA_PATH,))[METADATA_PATH]

    assert response.status_code == 401


def test_metadata_route_is_an_operator_route_outside_the_contract() -> None:
    from openmed.service.auth import DEFAULT_AUTH_EXEMPT_PATHS, ROUTE_POLICIES

    assert METADATA_PATH not in create_app().openapi()["paths"]
    assert METADATA_PATH not in DEFAULT_AUTH_EXEMPT_PATHS
    assert ROUTE_POLICIES[("GET", METADATA_PATH)].scopes == ()
    assert ROUTE_POLICIES[("GET", METADATA_PATH)].admission == "control"


def test_opted_in_production_openapi_requires_auth_and_serves_no_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, profile="prod", auth=True, openapi="true")

    anonymous = _get_all(DOCS_PATHS)
    authorized = _get_all(DOCS_PATHS, **{"X-API-Key": API_KEY})

    assert anonymous["/openapi.json"].status_code == 401
    assert anonymous["/openapi.json"].json()["error"]["code"] == (
        "authentication_required"
    )
    assert "paths" not in anonymous["/openapi.json"].json()
    assert authorized["/openapi.json"].status_code == 200
    assert "/health" in authorized["/openapi.json"].json()["paths"]
    for path in ("/docs", "/redoc", "/docs/oauth2-redirect"):
        assert anonymous[path].status_code in {401, 404}, path
        assert authorized[path].status_code == 404, path
    for response in (*anonymous.values(), *authorized.values()):
        assert "jsdelivr" not in response.text
        assert "swagger-ui" not in response.text


def test_explicit_opt_out_disables_docs_in_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, profile="dev", openapi="false")

    responses = _get_all(DOCS_PATHS)

    assert {path: r.status_code for path, r in responses.items()} == dict.fromkeys(
        DOCS_PATHS, 404
    )


def test_invalid_openapi_setting_fails_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, profile="dev", openapi="sometimes")

    with pytest.raises(ValueError, match="OPENMED_SERVICE_OPENAPI_ENABLED"):
        create_app()


@pytest.mark.parametrize("auth", [False, True])
def test_development_profile_keeps_current_docs_behavior(
    monkeypatch: pytest.MonkeyPatch,
    auth: bool,
) -> None:
    _configure(monkeypatch, profile="dev", auth=auth)

    responses = _get_all(("/docs", "/redoc", "/openapi.json", "/health"))

    assert responses["/docs"].status_code == 200
    assert "swagger-ui" in responses["/docs"].text
    assert responses["/redoc"].status_code == 200
    assert "redoc" in responses["/redoc"].text
    assert responses["/openapi.json"].status_code == 200
    assert responses["/openapi.json"].json()["info"]["title"] == "OpenMed REST API"
    assert responses["/health"].json() == {
        "status": "ok",
        "service": "openmed-rest",
        "version": openmed.__version__,
        "profile": "dev",
    }


def test_development_oauth2_redirect_keeps_current_auth_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, profile="dev")
    open_response = _get_all(("/docs/oauth2-redirect",))["/docs/oauth2-redirect"]
    _configure(monkeypatch, profile="dev", auth=True)
    auth_response = _get_all(("/docs/oauth2-redirect",))["/docs/oauth2-redirect"]

    assert open_response.status_code == 200
    assert auth_response.status_code == 401


def test_reference_launcher_suppresses_server_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    from openmed.service import logging as service_logging

    captured: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: captured.update(kw))
    monkeypatch.setattr(sys, "argv", ["openmed.service.logging", "--host", "127.0.0.1"])

    service_logging.main()

    assert captured["server_header"] is False
    assert captured["app"] == "openmed.service.app:app"
    assert captured["access_log"] is False


def test_reference_launch_serves_no_server_header_or_docs(tmp_path) -> None:
    import os
    import socket
    import subprocess
    import time
    import urllib.error
    import urllib.request

    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
    environment = os.environ.copy()
    for env_var in _SERVICE_ENV_VARS:
        environment.pop(env_var, None)
    environment.update({"OPENMED_OFFLINE": "1", "OPENMED_SERVICE_LOG_LEVEL": "off"})
    output_file = (tmp_path / "synthetic-launch-output.log").open("w+")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "openmed.service.logging",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=environment,
        stdout=output_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=1
                ) as response:
                    health_headers = dict(response.headers.items())
                    health_body = response.read().decode("utf-8")
                    break
            except OSError:
                if time.monotonic() >= deadline or process.poll() is not None:
                    raise AssertionError(
                        "Reference service did not become ready"
                    ) from None
                time.sleep(0.05)
        with pytest.raises(urllib.error.HTTPError) as docs_error:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=2)
    finally:
        process.terminate()
        process.communicate(timeout=15)
        output_file.close()

    assert "server" not in {name.lower() for name in health_headers}
    assert json.loads(health_body) == {"status": "ok", "service": "openmed-rest"}
    assert docs_error.value.code == 404
    assert "server" not in {name.lower() for name in docs_error.value.headers}
