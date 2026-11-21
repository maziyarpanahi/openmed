"""REST ``GET /v1/capabilities`` serves the canonical capability descriptor."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from openmed.core.capabilities import (
    capability_descriptor,
    capability_descriptor_digest,
    load_capability_descriptor_schema,
    validate_capability_descriptor,
)
from openmed.service import runtime as service_runtime
from openmed.service.app import CAPABILITIES_PATH, create_app
from openmed.service.auth import ROUTE_POLICIES, hash_api_key
from scripts.export_openapi import build_openapi_spec

LOOPBACK_BASE_URL = "http://127.0.0.1"
MODELS_PATH = "/models/loaded"
READ_KEY = "synthetic-capabilities-read-key"
WILDCARD_KEY = "synthetic-capabilities-wildcard-key"
OTHER_KEY = "synthetic-capabilities-other-key"
JWT_SECRET = b"synthetic-capabilities-jwt-secret"
JWT_ISSUER = "https://issuer.synthetic.invalid/"
JWT_AUDIENCE = "openmed-synthetic"
TOKEN_SENTINEL = "synthetic-capability-token-sentinel-value"
DICT_SENTINEL = "/synthetic-capability-sentinel/dictionary.txt"
TERMS_SENTINEL = "synthetic-capability-protect-term-sentinel"

_SERVICE_ENV_VARS = (
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_SERVED_MODELS",
    "OPENMED_SERVICE_KEEP_ALIVE",
    "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
    "OPENMED_SERVICE_CORS_ORIGINS",
    "OPENMED_SERVICE_TRUSTED_HOSTS",
    "OPENMED_SERVICE_BATCHING_ENABLED",
    "OPENMED_SERVICE_COALESCING_ENABLED",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT",
    "OPENMED_SERVICE_AUTH_API_KEYS",
    "OPENMED_SERVICE_AUTH_JWKS",
    "OPENMED_SERVICE_AUTH_JWKS_FILE",
    "OPENMED_SERVICE_AUTH_JWT_ISSUER",
    "OPENMED_SERVICE_AUTH_JWT_AUDIENCE",
    "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST",
)


class FakeLoader:
    """Model loader double; the capability route never loads a model."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        return object()

    def loaded_models(self) -> dict[str, Any]:
        return {}


@pytest.fixture(autouse=True)
def clean_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for env_var in _SERVICE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)


def _b64url(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _jwt(scopes: list[str]) -> str:
    header = {"alg": "HS256", "kid": "synthetic-hs", "typ": "JWT"}
    now = int(time.time())
    claims = {
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "sub": "synthetic-subject",
        "iat": now,
        "exp": now + 300,
        "scope": " ".join(scopes),
    }
    encoded = [
        _b64url(json.dumps(part, separators=(",", ":"), sort_keys=True).encode())
        for part in (header, claims)
    ]
    signing_input = ".".join(encoded)
    signature = hmac.new(JWT_SECRET, signing_input.encode(), hashlib.sha256)
    return f"{signing_input}.{_b64url(signature.digest())}"


def _configure_auth(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> dict[str, Optional[dict[str, str]]]:
    """Configure one auth mode and return its synthetic callers."""
    anonymous: dict[str, Optional[dict[str, str]]] = {"anonymous": None}
    if mode == "disabled":
        return anonymous
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    if mode == "api_key_allow_undeclared":
        monkeypatch.setenv("OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT", "false")
    if mode.startswith("api_key"):
        keys = [
            {"id": "read", "key": READ_KEY, "scopes": ["models:read"]},
            {"id": "wildcard", "key": WILDCARD_KEY, "scopes": ["models:*"]},
            {"id": "other", "key": OTHER_KEY, "scopes": ["pii:read"]},
        ]
        monkeypatch.setenv(
            "OPENMED_SERVICE_AUTH_API_KEYS",
            json.dumps(
                [
                    {
                        "id": item["id"],
                        "key_hash": hash_api_key(item["key"]),
                        "principal": f"synthetic-{item['id']}",
                        "scopes": item["scopes"],
                    }
                    for item in keys
                ]
            ),
        )
        return {
            **anonymous,
            "invalid": {"X-API-Key": "synthetic-unknown-key"},
            **{item["id"]: {"X-API-Key": item["key"]} for item in keys},
        }
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_JWKS",
        json.dumps(
            {
                "keys": [
                    {
                        "kty": "oct",
                        "kid": "synthetic-hs",
                        "alg": "HS256",
                        "k": _b64url(JWT_SECRET),
                    }
                ]
            }
        ),
    )
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_JWT_ISSUER", JWT_ISSUER)
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_JWT_AUDIENCE", JWT_AUDIENCE)
    return {
        **anonymous,
        "read": {"Authorization": f"Bearer {_jwt(['models:read'])}"},
        "other": {"Authorization": f"Bearer {_jwt(['pii:read'])}"},
    }


def test_route_policy_matches_loaded_models() -> None:
    assert (
        ROUTE_POLICIES[("GET", CAPABILITIES_PATH)]
        == ROUTE_POLICIES[("GET", MODELS_PATH)]
    )


def test_response_validates_and_matches_the_python_digest() -> None:
    app = create_app()
    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        response = client.get(CAPABILITIES_PATH)
        served = app.state.runtime.served_model_ids()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    validate_capability_descriptor(body)
    Draft202012Validator(load_capability_descriptor_schema()).validate(body)
    expected = capability_descriptor(
        served_models=served,
        enabled_capabilities=[row["code"] for row in body["capabilities"]],
    )
    assert body == expected
    assert body["digest"] == expected["digest"]
    assert body["digest"] == capability_descriptor_digest(body)
    assert body["served_models"] == sorted(served)
    assert {"name": "rest", "available": True} in body["surfaces"]


def test_operator_selected_served_models_reach_the_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OPENMED_SERVICE_SERVED_MODELS",
        "synthetic-org/synthetic-model-b,synthetic-alias-a",
    )
    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        body = client.get(CAPABILITIES_PATH).json()

    assert body["served_models"] == [
        "synthetic-alias-a",
        "synthetic-org/synthetic-model-b",
    ]
    assert body["digest"] == capability_descriptor_digest(body)
    assert (
        body["digest"]
        == capability_descriptor(
            served_models=reversed(body["served_models"]),
            enabled_capabilities=[row["code"] for row in body["capabilities"]],
        )["digest"]
    )


@pytest.mark.parametrize(
    "mode", ["disabled", "api_key", "api_key_allow_undeclared", "jwt"]
)
def test_capabilities_answers_the_same_callers_as_loaded_models(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    callers = _configure_auth(monkeypatch, mode)

    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        outcomes = {
            name: (
                client.get(MODELS_PATH, headers=headers).status_code,
                client.get(CAPABILITIES_PATH, headers=headers).status_code,
            )
            for name, headers in callers.items()
        }

    for name, (models_status, capabilities_status) in outcomes.items():
        assert capabilities_status == models_status, name
    if mode == "disabled":
        assert outcomes["anonymous"] == (200, 200)
    else:
        assert outcomes["anonymous"][1] == 401
        assert outcomes["read"] == (200, 200)
        assert outcomes["other"] == (403, 403)


def test_planted_path_and_environment_sentinels_never_appear(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "synthetic-capability-sentinel" / "private-model"
    model_dir.mkdir(parents=True)
    path_sentinel = str(model_dir)
    monkeypatch.setenv("HF_TOKEN", TOKEN_SENTINEL)
    monkeypatch.setenv("OPENMED_SERVICE_PRELOAD_MODELS", path_sentinel)
    monkeypatch.setenv("OPENMED_CHINESE_USER_DICT", DICT_SENTINEL)
    monkeypatch.setenv("OPENMED_CLINICAL_PROTECT_TERMS", TERMS_SENTINEL)
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [
                {
                    "id": "read",
                    "key_hash": hash_api_key(READ_KEY),
                    "principal": "synthetic-read",
                    "scopes": ["models:read"],
                }
            ]
        ),
    )

    app = create_app()
    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        loaded = client.get(MODELS_PATH, headers={"X-API-Key": READ_KEY})
        response = client.get(CAPABILITIES_PATH, headers={"X-API-Key": READ_KEY})

    assert response.status_code == 200
    # Negative control: the planted path is a real preload entry, so it does
    # reach the operator-facing /models/loaded view.
    assert path_sentinel in loaded.text
    text = response.text
    for sentinel in (
        TOKEN_SENTINEL,
        path_sentinel,
        str(tmp_path),
        DICT_SENTINEL,
        TERMS_SENTINEL,
        READ_KEY,
        hash_api_key(READ_KEY),
        "synthetic-capability-sentinel",
        "127.0.0.1",
        "synthetic-read",
    ):
        assert sentinel not in text
    validate_capability_descriptor(response.json())


def test_openapi_types_the_response_with_the_descriptor_schema() -> None:
    operation = build_openapi_spec()["paths"][CAPABILITIES_PATH]["get"]

    assert "requestBody" not in operation
    schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema == load_capability_descriptor_schema()


def test_committed_openapi_lists_the_capabilities_route() -> None:
    with open("docs/api/openapi.json", encoding="utf-8") as handle:
        spec = json.load(handle)

    schema = spec["paths"][CAPABILITIES_PATH]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert schema["$id"] == load_capability_descriptor_schema()["$id"]
