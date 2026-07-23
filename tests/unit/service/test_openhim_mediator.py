"""Offline conformance coverage for the OpenHIM mediator integration."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import openmed
from openmed.interop.gateway import assert_redacted
from openmed.service.openhim_mediator import (
    DEFAULT_MEDIATOR_URN,
    OPENHIM_MEDIA_TYPE,
    MediatorTransformResult,
    OpenHIMMediatorClient,
    OpenHIMMediatorConfigurationError,
    OpenHIMMediatorSettings,
    build_openhim_envelope,
    default_mediator_registration,
    failed_openhim_result,
    load_mediator_registration,
    mediator_response_headers,
    transform_mediator_payload,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "openhim"
EXAMPLE_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "openhim-mediator"
    / "mediator-config.json"
)
SYNTHETIC_MAPPING = {
    "[NAME]": "Amina Example",
    "[FIRST]": "Amina",
    "[LAST]": "Example",
    "[MRN]": "MRN-871-1448",
    "[PHONE]": "+256-700-871-1448",
}
REPLACEMENTS = sorted(
    ((original, replacement) for replacement, original in SYNTHETIC_MAPPING.items()),
    key=lambda item: len(item[0]),
    reverse=True,
)
service_app = importlib.import_module("openmed.service.app")


@dataclass
class _FakeResult:
    deidentified_text: str


def _fake_deidentify(text: str, **_: Any) -> _FakeResult:
    redacted = text
    for original, replacement in REPLACEMENTS:
        redacted = redacted.replace(original, replacement)
    return _FakeResult(redacted)


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _enabled_settings(
    registration: dict[str, Any],
    *,
    auth_mode: str = "basic",
) -> OpenHIMMediatorSettings:
    return OpenHIMMediatorSettings(
        enabled=True,
        api_url="https://openhim-core.example:8080",
        username="openhim@example.org",
        password="synthetic-password",
        auth_mode=auth_mode,
        verify_tls=True,
        heartbeat_interval_seconds=10,
        request_timeout_seconds=2,
        registration=registration,
    )


def _set_enabled_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_OPENHIM_MEDIATOR_ENABLED", "true")
    monkeypatch.setenv("OPENMED_OPENHIM_CORE_URL", "https://openhim-core.example:8080")
    monkeypatch.setenv("OPENMED_OPENHIM_USERNAME", "openhim@example.org")
    monkeypatch.setenv("OPENMED_OPENHIM_PASSWORD", "synthetic-password")
    monkeypatch.setenv("OPENMED_OPENHIM_CONFIG_PATH", str(EXAMPLE_CONFIG))
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.delenv("OPENMED_SERVICE_PRELOAD_MODELS", raising=False)


def test_feature_flag_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENMED_OPENHIM_MEDIATOR_ENABLED", raising=False)

    settings = OpenHIMMediatorSettings.from_env()
    app = service_app.create_app()

    assert settings.enabled is False
    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/health").status_code == 200
        assert client.get("/openhim/heartbeat").status_code == 404
        assert (
            client.post("/openhim/deidentify", content=b"unchanged").status_code == 404
        )


def test_registration_payload_has_openhim_required_fields() -> None:
    generated = default_mediator_registration()
    mounted = load_mediator_registration(str(EXAMPLE_CONFIG))

    for payload in (generated, mounted):
        assert payload["urn"] == DEFAULT_MEDIATOR_URN
        assert payload["version"] == openmed.__version__
        assert payload["name"] == "OpenMed De-identification Mediator"
        assert payload["defaultChannelConfig"][0]["routes"][0]["path"] == (
            "/openhim/deidentify"
        )
        assert payload["endpoints"][0]["path"] == "/openhim/deidentify"


def test_enabled_settings_require_core_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_OPENHIM_MEDIATOR_ENABLED", "true")
    for name in (
        "OPENMED_OPENHIM_CORE_URL",
        "OPENMED_OPENHIM_USERNAME",
        "OPENMED_OPENHIM_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(OpenHIMMediatorConfigurationError, match="required settings"):
        OpenHIMMediatorSettings.from_env()


def test_heartbeat_interval_is_bounded_by_openhim_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)
    monkeypatch.setenv("OPENMED_OPENHIM_HEARTBEAT_INTERVAL_SECONDS", "31")

    with pytest.raises(OpenHIMMediatorConfigurationError, match="must not exceed 30"):
        OpenHIMMediatorSettings.from_env()


def test_management_api_requires_tls_unless_fixture_override_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)
    monkeypatch.setenv("OPENMED_OPENHIM_CORE_URL", "http://openhim-fixture:8081")

    with pytest.raises(OpenHIMMediatorConfigurationError, match="must use https"):
        OpenHIMMediatorSettings.from_env()

    monkeypatch.setenv("OPENMED_OPENHIM_ALLOW_INSECURE_HTTP", "true")
    settings = OpenHIMMediatorSettings.from_env()

    assert settings.allow_insecure_http is True


def test_management_api_url_rejects_embedded_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)
    monkeypatch.setenv(
        "OPENMED_OPENHIM_CORE_URL",
        "https://user:password@openhim-core.example:8080",
    )

    with pytest.raises(
        OpenHIMMediatorConfigurationError,
        match="must not contain credentials",
    ):
        OpenHIMMediatorSettings.from_env()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("OPENMED_OPENHIM_AUTH_MODE", "session", "must be one of"),
        ("OPENMED_OPENHIM_METHOD", "keep", "must be one of"),
        ("OPENMED_OPENHIM_POLICY", "not_a_policy", "bundled policy"),
    ],
)
def test_processing_and_authentication_settings_are_validated_at_startup(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    _set_enabled_env(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(OpenHIMMediatorConfigurationError, match=message):
        OpenHIMMediatorSettings.from_env()


def test_recorded_token_registration_and_heartbeat_handshake_is_idempotent() -> None:
    fixture = _fixture("handshake.json")
    registration = default_mediator_registration()
    observed: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raw_path = request.url.raw_path.decode("ascii")
        observed.append(
            {
                "method": request.method,
                "path": raw_path,
                "authorization": request.headers.get("authorization"),
                "auth_username": request.headers.get("auth-username"),
                "auth_timestamp": request.headers.get("auth-ts"),
                "auth_salt": request.headers.get("auth-salt"),
                "auth_token": request.headers.get("auth-token"),
                "body": json.loads(request.content) if request.content else None,
            }
        )
        for step_name in ("authenticate", "registration", "heartbeat"):
            step = fixture[step_name]
            if request.method == step["method"] and raw_path == step["path"]:
                return httpx.Response(
                    step["response_status"], json=step["response_body"]
                )
        return httpx.Response(404)

    async def scenario() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            ticks = iter((100.0, 105.0))
            client = OpenHIMMediatorClient(
                _enabled_settings(registration, auth_mode="token"),
                client=http_client,
                monotonic=lambda: next(ticks),
            )
            await client.start()
            await client.start()
            assert client.status()["registered"] is True
            assert client.status()["last_heartbeat_at"] is not None
            await client.stop()

    asyncio.run(scenario())

    assert [item["path"] for item in observed] == [
        fixture["authenticate"]["path"],
        fixture["registration"]["path"],
        fixture["authenticate"]["path"],
        fixture["heartbeat"]["path"],
    ]
    salt = fixture["authenticate"]["response_body"]["salt"]
    timestamp = fixture["authenticate"]["response_body"]["ts"]
    password_hash = hashlib.sha512(
        f"{salt}synthetic-password".encode("utf-8")
    ).hexdigest()
    expected_token = hashlib.sha512(
        f"{password_hash}{salt}{timestamp}".encode("utf-8")
    ).hexdigest()
    for request in (observed[1], observed[3]):
        assert request["authorization"] is None
        assert request["auth_username"] == "openhim@example.org"
        assert request["auth_timestamp"] == timestamp
        assert request["auth_salt"] == salt
        assert request["auth_token"] == expected_token
    assert observed[1]["body"] == registration
    assert observed[3]["body"]["config"] is True
    assert observed[3]["body"]["uptime"] == 5.0


def test_basic_authentication_does_not_call_token_preflight() -> None:
    registration = default_mediator_registration()
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path == "/mediators":
            return httpx.Response(201)
        if request.url.path.endswith("/heartbeat"):
            return httpx.Response(200)
        return httpx.Response(404)

    async def scenario() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = OpenHIMMediatorClient(
                _enabled_settings(registration),
                client=http_client,
            )
            await client.heartbeat(force_config=True)
            await client.stop()

    asyncio.run(scenario())

    assert [request.url.path for request in observed] == [
        "/mediators",
        f"/mediators/{DEFAULT_MEDIATOR_URN}/heartbeat",
    ]
    assert all(
        request.headers["authorization"].startswith("Basic ") for request in observed
    )


def test_fhir_bundle_envelope_redacts_phi_and_preserves_coded_bytes() -> None:
    source = _fixture("fhir_bundle.json")
    result = transform_mediator_payload(
        json.dumps(source).encode("utf-8"),
        {"content-type": "application/fhir+json", "x-request-id": "synthetic-871"},
        202,
        deidentifier=_fake_deidentify,
    )
    envelope = build_openhim_envelope(result, urn=DEFAULT_MEDIATOR_URN)
    output = json.loads(envelope["response"]["body"])

    assert result.transformed is True
    assert envelope["x-mediator-urn"] == DEFAULT_MEDIATOR_URN
    assert envelope["status"] == "Successful"
    assert envelope["response"]["status"] == 202
    assert envelope["response"]["headers"]["x-request-id"] == "synthetic-871"
    assert envelope["orchestrations"][0]["name"] == ("OpenMed local de-identification")
    assert envelope["orchestrations"][0]["request"].get("body") is None
    assert_redacted(json.dumps(output), SYNTHETIC_MAPPING)

    for index, entry in enumerate(source["entry"]):
        output_entry = output["entry"][index]
        assert _canonical_bytes(output_entry["fullUrl"]) == _canonical_bytes(
            entry["fullUrl"]
        )
        assert _canonical_bytes(output_entry["request"]) == _canonical_bytes(
            entry["request"]
        )
    source_observation = source["entry"][1]["resource"]
    output_observation = output["entry"][1]["resource"]
    assert _canonical_bytes(output_observation["code"]) == _canonical_bytes(
        source_observation["code"]
    )
    assert _canonical_bytes(output_observation["subject"]["reference"]) == (
        _canonical_bytes(source_observation["subject"]["reference"])
    )


def test_fhir_parameters_cannot_override_deployment_policy_or_method() -> None:
    observed: list[dict[str, Any]] = []

    def deidentifier(text: str, **kwargs: Any) -> _FakeResult:
        observed.append(dict(kwargs))
        return _FakeResult(text.replace("Amina Example", "[NAME]"))

    source = {
        "resourceType": "Parameters",
        "parameter": [
            {
                "name": "resource",
                "resource": {
                    "resourceType": "Patient",
                    "name": [{"text": "Amina Example"}],
                },
            },
            {"name": "policy", "valueString": "clinical_minimal_redaction"},
            {"name": "method", "valueCode": "remove"},
        ],
    }

    result = transform_mediator_payload(
        json.dumps(source).encode(),
        {"content-type": "application/fhir+json"},
        200,
        deidentifier=deidentifier,
        policy="hipaa_safe_harbor",
        method="replace",
    )
    output = json.loads(result.body)

    assert observed == [
        {
            "policy": "hipaa_safe_harbor",
            "method": "replace",
            "consistent": True,
        }
    ]
    assert "Amina Example" not in result.body.decode()
    assert {item["name"]: item for item in output["parameter"]}["policy"] == {
        "name": "policy",
        "valueString": "hipaa_safe_harbor",
    }
    assert {item["name"]: item for item in output["parameter"]}["method"] == {
        "name": "method",
        "valueCode": "replace",
    }


def test_non_text_payload_preserves_bytes_headers_and_status() -> None:
    body = b"\x00\xff\x10synthetic-binary\x00"
    headers = {
        "content-type": "application/octet-stream",
        "content-disposition": 'attachment; filename="scan.bin"',
        "x-request-id": "opaque-871",
    }

    result = transform_mediator_payload(body, headers, 206)

    assert result == MediatorTransformResult(
        body=body,
        headers=headers,
        status_code=206,
        transformed=False,
        operation="pass-through",
    )


def test_response_headers_drop_credentials_and_unbounded_clinical_metadata() -> None:
    headers = mediator_response_headers(
        {
            "content-type": "application/octet-stream",
            "content-encoding": "gzip",
            "x-request-id": "synthetic-871",
            "content-disposition": 'attachment; filename="Amina Example.pdf"',
            "authorization": "Basic secret",
            "cookie": "session=secret",
            "traceparent": "Amina Example",
            "x-patient-name": "Amina Example",
        }
    )

    assert headers == {
        "content-type": "application/octet-stream",
        "content-encoding": "gzip",
        "x-request-id": "synthetic-871",
    }


def test_openhim_envelope_decodes_declared_text_charset() -> None:
    result = MediatorTransformResult(
        body="dé-identifié".encode("iso-8859-1"),
        headers={"content-type": "text/plain; charset=iso-8859-1"},
        status_code=200,
        transformed=True,
        operation="text-de-identify",
    )

    envelope = build_openhim_envelope(result, urn=DEFAULT_MEDIATOR_URN)

    assert envelope["response"]["body"] == "dé-identifié"


def test_failed_envelope_requests_retry_without_exposing_exception_details() -> None:
    envelope = build_openhim_envelope(
        failed_openhim_result(500, "processing_failed"),
        urn=DEFAULT_MEDIATOR_URN,
    )

    assert envelope["status"] == "Failed"
    assert envelope["error"] == {"message": "Mediator request failed"}
    assert "stack" not in envelope["error"]


class _FakeLifecycleClient:
    def __init__(self, urn: str) -> None:
        self.urn = urn
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "registered": self.started,
            "urn": self.urn,
            "last_heartbeat_at": "2026-01-01T00:00:00Z",
            "last_error": None,
        }


def test_enabled_app_returns_well_formed_openhim_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)
    lifecycle = _FakeLifecycleClient(DEFAULT_MEDIATOR_URN)
    monkeypatch.setattr(service_app, "create_openhim_mediator", lambda _: lifecycle)
    app = service_app.create_app()
    app.state.openhim_deidentifier = _fake_deidentify
    source = _fixture("fhir_bundle.json")

    with TestClient(app, base_url="http://localhost") as client:
        heartbeat = client.get("/openhim/heartbeat")
        response = client.post(
            "/openhim/deidentify",
            content=json.dumps(source).encode("utf-8"),
            headers={
                "content-type": "application/fhir+json",
                "x-request-id": "app-fixture-871",
            },
        )

    assert lifecycle.started is True
    assert lifecycle.stopped is True
    assert heartbeat.json()["registered"] is True
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(OPENHIM_MEDIA_TYPE)
    envelope = response.json()
    assert envelope["response"]["headers"]["x-request-id"] == "app-fixture-871"
    assert envelope["properties"]["openmed.operation"] == ("fhir-bundle-de-identify")
    assert_redacted(envelope["response"]["body"], SYNTHETIC_MAPPING)


def test_enabled_app_keeps_opaque_body_outside_json_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)
    lifecycle = _FakeLifecycleClient(DEFAULT_MEDIATOR_URN)
    monkeypatch.setattr(service_app, "create_openhim_mediator", lambda _: lifecycle)
    app = service_app.create_app()
    body = b"\x00\xff\x10opaque"

    with TestClient(app, base_url="http://localhost") as client:
        response = client.post(
            "/openhim/deidentify",
            content=body,
            headers={"content-type": "application/octet-stream"},
        )

    assert response.status_code == 200
    assert response.content == body
    assert response.headers["content-type"] == "application/octet-stream"


def test_request_query_cannot_weaken_deidentification_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_enabled_env(monkeypatch)
    lifecycle = _FakeLifecycleClient(DEFAULT_MEDIATOR_URN)
    monkeypatch.setattr(service_app, "create_openhim_mediator", lambda _: lifecycle)
    observed: list[dict[str, Any]] = []

    def deidentifier(text: str, **kwargs: Any) -> _FakeResult:
        observed.append(dict(kwargs))
        return _FakeResult(text.replace("Amina Example", "[NAME]"))

    app = service_app.create_app()
    app.state.openhim_deidentifier = deidentifier

    with TestClient(app, base_url="http://localhost") as client:
        response = client.post(
            "/openhim/deidentify?policy=clinical_minimal_redaction&method=remove",
            content="Patient Amina Example".encode(),
            headers={"content-type": "text/plain"},
        )

    assert response.status_code == 200
    assert observed == [{"policy": "hipaa_safe_harbor", "method": "replace"}]
    assert "Amina Example" not in response.json()["response"]["body"]
