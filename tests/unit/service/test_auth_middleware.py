"""Unit tests for REST authentication middleware."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

import openmed
from openmed.processing.outputs import PredictionResult
from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.auth import hash_api_key

LOOPBACK_BASE_URL = "http://127.0.0.1"
_HS_SECRET = b"synthetic-test-secret"
_RSA_N = (
    "3O8uls5vuPQurwdtlL7CclqrswG1af7xVrfnn-kWe5PVdlRG3BQ7vczvWb-wWbF53Pkd"
    "NYCwEXWx9Ym4qSTsJbWmx1NH-b1dEQOYoIVqYhYbMNUZ3R4v9IN2yp6c1ocZ8jTbU_8f"
    "ZSPjFeyaZO5vgvo0-fMhDWuN6YThrZGXAP0"
)
_RSA_E = "AQAB"
_RSA_D = (
    "beOVo7LYRQFHOw2Rxps_MgvBPQ8LgcYpmf1s-s-_vAWi9fEjMZHqyRPmtRgwCdzJhUx"
    "u586zRGvq8ProW1EfFxHKxec5L4iFIDucxSUvhPvydL7PNQuiCIN72mD-VEIhLsft3d"
    "Q-ivi3JyuV-evKRVx0wKalUL4_liDEM6tcTkE"
)
_SHA256_DIGEST_INFO_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")
_SERVICE_ENV_VARS = (
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_KEEP_ALIVE",
    "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
    "OPENMED_SERVICE_MAX_TEXT_LENGTH",
    "OPENMED_SERVICE_CORS_ORIGINS",
    "OPENMED_SERVICE_TRUSTED_HOSTS",
    "OPENMED_SERVICE_BATCHING_ENABLED",
    "OPENMED_SERVICE_BATCH_MAX_SIZE",
    "OPENMED_SERVICE_BATCH_MAX_WAIT_MS",
    "OPENMED_SERVICE_COALESCING_ENABLED",
    "OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
    "OPENMED_SERVICE_THROTTLE_KEY",
    "OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS",
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT",
    "OPENMED_SERVICE_AUTH_API_KEYS",
    "OPENMED_SERVICE_AUTH_JWKS",
    "OPENMED_SERVICE_AUTH_JWKS_FILE",
    "OPENMED_SERVICE_AUTH_JWT_ISSUER",
    "OPENMED_SERVICE_AUTH_JWT_AUDIENCE",
    "OPENMED_SERVICE_AUTH_JWT_LEEWAY_SECONDS",
    "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_KEY",
)


class FakeLoader:
    """Minimal model loader double for authenticated service tests."""

    def __init__(self, config: Any):
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        return object()

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {}


@pytest.fixture(autouse=True)
def clean_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for env_var in _SERVICE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


@pytest.fixture(autouse=True)
def fake_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)


def _prediction_result(text: str) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name="disease_detection_superclinical",
        timestamp=datetime.now().isoformat(),
        processing_time=0.01,
    )


def _assert_error_payload(response, status_code: int, code: str) -> dict[str, Any]:
    assert response.status_code == status_code
    payload = response.json()
    assert payload["error"]["code"] == code
    assert "message" in payload["error"]
    assert "details" in payload["error"]
    return payload


def _api_key_config(
    raw_key: str,
    *,
    scopes: list[str],
    principal: str = "test-service",
) -> str:
    return json.dumps(
        [
            {
                "id": "test-key",
                "key_hash": hash_api_key(raw_key),
                "principal": principal,
                "scopes": scopes,
            }
        ]
    )


def _jwks() -> str:
    return json.dumps(
        {
            "keys": [
                {
                    "kty": "oct",
                    "kid": "hs-test",
                    "alg": "HS256",
                    "k": _b64url(_HS_SECRET),
                },
                {
                    "kty": "RSA",
                    "kid": "rs-test",
                    "alg": "RS256",
                    "n": _RSA_N,
                    "e": _RSA_E,
                },
            ]
        }
    )


def _b64url(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _b64url_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return _b64url(encoded)


def _b64url_int(value: int) -> str:
    return _b64url(value.to_bytes((value.bit_length() + 7) // 8, "big"))


def _decode_int(value: str) -> int:
    padding = "=" * (-len(value) % 4)
    return int.from_bytes(base64.urlsafe_b64decode(value + padding), "big")


def _jwt_hs256(payload: dict[str, Any]) -> str:
    header = {"alg": "HS256", "kid": "hs-test", "typ": "JWT"}
    signing_input = f"{_b64url_json(header)}.{_b64url_json(payload)}"
    signature = hmac.new(
        _HS_SECRET,
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{signing_input}.{_b64url(signature)}"


def _jwt_rs256(payload: dict[str, Any]) -> str:
    header = {"alg": "RS256", "kid": "rs-test", "typ": "JWT"}
    signing_input = f"{_b64url_json(header)}.{_b64url_json(payload)}"
    digest = hashlib.sha256(signing_input.encode("ascii")).digest()
    modulus = _decode_int(_RSA_N)
    private_exponent = _decode_int(_RSA_D)
    key_size = (modulus.bit_length() + 7) // 8
    encoded = (
        b"\x00\x01"
        + (b"\xff" * (key_size - len(_SHA256_DIGEST_INFO_PREFIX) - len(digest) - 3))
        + b"\x00"
        + _SHA256_DIGEST_INFO_PREFIX
        + digest
    )
    signature = pow(
        int.from_bytes(encoded, "big"),
        private_exponent,
        modulus,
    )
    return f"{signing_input}.{_b64url(signature.to_bytes(key_size, 'big'))}"


def _tamper_jwt_signature(token: str) -> str:
    header, payload, signature_segment = token.split(".")
    padding = "=" * (-len(signature_segment) % 4)
    signature = bytearray(base64.urlsafe_b64decode(signature_segment + padding))
    signature[0] ^= 0x01
    return f"{header}.{payload}.{_b64url(bytes(signature))}"


def test_auth_defaults_to_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_analyze(text: str, **_: Any) -> PredictionResult:
        return _prediction_result(text)

    monkeypatch.setattr(openmed, "analyze_text", fake_analyze)
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        response = client.post("/analyze", json={"text": "Patient is stable."})

    assert response.status_code == 200
    assert client.app.state.auth.enabled is False


def test_valid_api_key_succeeds_and_principal_is_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        _api_key_config("test-secret", scopes=["custom:read"]),
    )
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
        json.dumps({"GET /whoami": ["custom:read"]}),
    )
    app = create_app()

    @app.get("/whoami")
    async def whoami(request: Request) -> dict[str, Any]:
        principal = request.state.auth_principal
        return {"subject": principal.subject, "scopes": list(principal.scopes)}

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/whoami", headers={"X-API-Key": "test-secret"})

    assert response.status_code == 200
    assert response.json() == {
        "subject": "test-service",
        "scopes": ["custom:read"],
    }


def test_missing_or_invalid_credentials_return_401_without_phi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        _api_key_config("test-secret", scopes=["analyze:write"]),
    )
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        missing = client.post("/analyze", json={"text": "Patient Maria Garcia"})
        invalid = client.post(
            "/analyze",
            headers={"X-API-Key": "wrong-secret"},
            json={"text": "Patient Maria Garcia"},
        )

    _assert_error_payload(missing, 401, "authentication_required")
    _assert_error_payload(invalid, 401, "invalid_credentials")
    assert "WWW-Authenticate" in missing.headers
    assert "Maria Garcia" not in missing.text
    assert "Maria Garcia" not in invalid.text


def test_scope_enforcement_blocks_underprivileged_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        _api_key_config("test-secret", scopes=["pii:read"]),
    )
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/pii/deidentify",
            headers={"X-API-Key": "test-secret"},
            json={"text": "Patient Maria Garcia", "method": "mask"},
        )

    payload = _assert_error_payload(response, 403, "forbidden")
    assert payload["error"]["details"]["required_scopes"] == ["pii:write"]
    assert "Maria Garcia" not in response.text


def test_jwt_hs_and_rs_signatures_and_expiry_are_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = int(time.time())
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_JWKS", _jwks())
    app = create_app()
    valid_claims = {"sub": "jwt-client", "scope": "models:read", "exp": now + 60}

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        hs_response = client.get(
            "/models/loaded",
            headers={"Authorization": f"Bearer {_jwt_hs256(valid_claims)}"},
        )
        rs_token = _jwt_rs256(valid_claims)
        rs_response = client.get(
            "/models/loaded",
            headers={"Authorization": f"Bearer {rs_token}"},
        )
        tampered = _tamper_jwt_signature(rs_token)
        tampered_response = client.get(
            "/models/loaded",
            headers={"Authorization": f"Bearer {tampered}"},
        )
        expired_response = client.get(
            "/models/loaded",
            headers={
                "Authorization": f"Bearer {_jwt_hs256({**valid_claims, 'exp': now - 3600})}"
            },
        )

    assert hs_response.status_code == 200
    assert rs_response.status_code == 200
    _assert_error_payload(tampered_response, 401, "invalid_credentials")
    _assert_error_payload(expired_response, 401, "invalid_credentials")


def test_failed_auth_attempts_are_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        _api_key_config("test-secret", scopes=["models:read"]),
    )
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS", "0.01")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST", "1")
    app = create_app()

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        first = client.get("/models/loaded")
        second = client.get("/models/loaded")

    _assert_error_payload(first, 401, "authentication_required")
    _assert_error_payload(second, 429, "auth_rate_limited")
    assert int(second.headers["Retry-After"]) >= 1
