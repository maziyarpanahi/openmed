"""Offline REST preflight parity and content-free failure tests."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile

import pytest
from fastapi.testclient import TestClient

from openmed.multimodal.preflight import preflight_asset
from openmed.service.app import create_app

PATH = "/experimental/multimodal/preflight"
FLAG = "OPENMED_SERVICE_EXPERIMENTAL_MULTIMODAL_PREFLIGHT"
CANARY = "SyntheticPatient_PrivateFilename.png"
PNG = b"\x89PNG\r\n\x1a\n" + CANARY.encode()
CASES = [
    (PNG, "image/png", {"width": 8, "height": 8}),
    (b"%PDF-1.7\nsynthetic", "application/pdf", {"pages": 1}),
    (
        b"\0" * 128 + b"DICMsynthetic",
        "application/dicom",
        {"width": 4, "height": 4, "frames": 1},
    ),
    (b"RIFF\x10\0\0\0WAVEsynthetic", "audio/wav", {"duration_seconds": 1.5}),
]


def manifest(source=PNG, media_type="image/png", **fields):
    return {
        "asset_id": "synthetic-asset-1",
        "media_type": media_type,
        "sha256": hashlib.sha256(source).hexdigest(),
        "byte_size": len(source),
        **fields,
    }


def headers(data):
    return {
        "content-type": "application/octet-stream",
        "X-OpenMed-Manifest": json.dumps(data),
    }


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    for name in (
        FLAG,
        "OPENMED_SERVICE_AUTH_ENABLED",
        "OPENMED_SERVICE_CORS_ORIGINS",
        "OPENMED_SERVICE_TRUSTED_HOSTS",
        "OPENMED_SERVICE_PRELOAD_MODELS",
        "OPENMED_SERVICE_MAX_REQUEST_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)


def client(**kwargs):
    return TestClient(
        create_app(experimental_multimodal_preflight=True, **kwargs),
        base_url="http://127.0.0.1",
    )


def test_default_app_does_not_register_route_or_openapi():
    app = create_app()
    assert PATH not in {route.path for route in app.routes}
    assert PATH not in app.openapi()["paths"]
    with TestClient(app, base_url="http://127.0.0.1") as transport:
        assert transport.post(PATH, content=PNG, headers=headers({})).status_code == 404


def test_explicit_opt_in_and_environment(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    operation = create_app().openapi()["paths"][PATH]["post"]
    assert operation["x-experimental"] is True
    assert operation["x-enabled-by-default"] is False
    assert "application/octet-stream" in operation["requestBody"]["content"]
    assert (
        PATH
        not in create_app(experimental_multimodal_preflight=False).openapi()["paths"]
    )
    monkeypatch.setenv(FLAG, CANARY)
    with pytest.raises(ValueError, match="flag is invalid") as exc:
        create_app()
    assert CANARY not in str(exc.value)


def test_catalog_keeps_opt_in_route_separate_from_default_client_contract():
    from scripts.export_openapi import build_openapi_spec

    catalog = build_openapi_spec()
    assert PATH not in catalog["paths"]
    assert (
        catalog["x-openmed-experimental-paths"][PATH]
        == create_app(experimental_multimodal_preflight=True).openapi()["paths"][PATH]
    )


@pytest.mark.parametrize(("source", "media_type", "fields"), CASES)
def test_report_bytes_match_direct_call_without_disk_or_inference(
    source, media_type, fields, monkeypatch
):
    data = manifest(source, media_type, **fields)
    expected = preflight_asset(data, source).to_json().encode()

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not spool files or load models")

    with client() as transport:
        monkeypatch.setattr(tempfile, "TemporaryFile", forbidden)
        monkeypatch.setattr(tempfile, "NamedTemporaryFile", forbidden)
        monkeypatch.setattr(tempfile, "SpooledTemporaryFile", forbidden)
        monkeypatch.setattr(
            "openmed.service.runtime.ServiceRuntime.begin_model_request", forbidden
        )
        response = transport.post(PATH, headers=headers(data), content=source)
    assert response.status_code == 200
    assert response.content == expected
    assert response.headers["content-type"] == "application/json"


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "{",
        "[]",
        "null",
        json.dumps({"filename": CANARY}),
        '{"asset_id":"a","asset_id":"b"}',
        '"' + CANARY + '"',
        "[" * 9000,
    ],
)
def test_malformed_manifest_is_never_echoed(raw, caplog):
    request_headers = {
        "content-type": "application/octet-stream",
        "X-Request-ID": CANARY,
    }
    if raw is not None:
        request_headers["X-OpenMed-Manifest"] = raw
    with (
        caplog.at_level(logging.INFO, logger="openmed.service.access"),
        client() as transport,
    ):
        response = transport.post(PATH, headers=request_headers, content=PNG)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "malformed_manifest"
    assert CANARY not in response.text
    assert CANARY not in response.headers["X-Request-ID"]
    assert CANARY not in caplog.text


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("sha256", "0" * 64, "sha256_mismatch"),
        ("byte_size", len(PNG) + 1, "byte_count_mismatch"),
        ("byte_size", len(PNG) - 1, "byte_count_mismatch"),
    ],
)
def test_integrity_errors_use_stable_content_free_envelope(field, value, code):
    data = manifest(width=8, height=8)
    data.update({"asset_id": CANARY, field: value})
    with client() as transport:
        response = transport.post(PATH, headers=headers(data), content=PNG)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["details"] is None
    assert CANARY not in response.text
    assert "manifest" not in response.json()["error"]


def test_duplicate_manifest_headers_are_rejected():
    raw = json.dumps(manifest(width=8, height=8))
    with client() as transport:
        response = transport.post(
            PATH,
            headers=[
                ("content-type", "application/octet-stream"),
                ("X-OpenMed-Manifest", raw),
                ("X-OpenMed-Manifest", raw),
            ],
            content=PNG,
        )
    assert response.json()["error"]["code"] == "malformed_manifest"


@pytest.mark.parametrize("log_format", ["json", "plain"])
def test_logs_have_only_route_outcome_and_controlled_reasons(
    caplog, monkeypatch, log_format
):
    monkeypatch.setenv("OPENMED_SERVICE_LOG_FORMAT", log_format)
    monkeypatch.setenv("OPENMED_SERVICE_MAX_REQUEST_BYTES", str(len(PNG)))
    data = manifest(width=8, height=8)
    data["asset_id"] = CANARY
    request_headers = {
        **headers(data),
        "X-Request-ID": CANARY,
        "Content-Disposition": 'attachment; filename="' + CANARY + '"',
    }
    with (
        caplog.at_level(logging.INFO, logger="openmed.service.access"),
        client() as transport,
    ):
        assert (
            transport.post(PATH, headers=request_headers, content=PNG).status_code
            == 200
        )
        invalid = {**data, "sha256": "0" * 64}
        assert (
            transport.post(PATH, headers=headers(invalid), content=PNG).status_code
            == 422
        )
        assert (
            transport.post(
                PATH, headers=request_headers, content=PNG + b"x"
            ).status_code
            == 413
        )
    records = [
        record.openmed_access_log
        for record in caplog.records
        if record.name == "openmed.service.access"
    ]
    assert records == [
        {"route": PATH, "outcome": "accept", "reason_codes": ()},
        {"route": PATH, "outcome": "rejected", "reason_codes": ("sha256_mismatch",)},
        {
            "route": PATH,
            "outcome": "rejected",
            "reason_codes": ("request_size_exceeded",),
        },
    ]
    assert CANARY not in caplog.text
    assert hashlib.sha256(PNG).hexdigest() not in caplog.text


def test_wrong_transport_and_configured_body_ceiling():
    with client(max_request_body_bytes=len(PNG) - 1) as transport:
        response = transport.post(PATH, headers=headers(manifest()), content=PNG)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    with client() as transport:
        response = transport.post(
            PATH, content=CANARY, headers={"content-type": "multipart/form-data"}
        )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"
    assert CANARY not in response.text


@pytest.mark.parametrize(
    ("scopes", "status"), [([], 403), (["models:read"], 403), (["pii:read"], 200)]
)
def test_existing_authentication_policy_protects_opt_in_route(
    monkeypatch, caplog, scopes, status
):
    from openmed.service.auth import hash_api_key

    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [{"id": "synthetic", "key_hash": hash_api_key(CANARY), "scopes": scopes}]
        ),
    )
    data = manifest(width=8, height=8)
    with (
        caplog.at_level(logging.INFO, logger="openmed.service.access"),
        client() as transport,
    ):
        missing = transport.post(PATH, headers=headers(data), content=PNG)
        response = transport.post(
            PATH, headers={**headers(data), "X-API-Key": CANARY}, content=PNG
        )
    assert missing.status_code == 401
    assert response.status_code == status
    if status == 200:
        assert response.content == preflight_asset(data, PNG).to_json().encode()
    else:
        assert response.json()["error"]["code"] == "forbidden"
    assert CANARY not in caplog.text
    assert CANARY not in missing.text
    assert CANARY not in response.text
