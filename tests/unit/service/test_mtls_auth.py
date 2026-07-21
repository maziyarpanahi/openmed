"""Unit tests for mutual-TLS client certificate authentication."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

from openmed.service.app import create_app
from openmed.service.auth import hash_api_key
from openmed.service.logging import ACCESS_LOGGER_NAME
from openmed.service.mtls import (
    SERVICE_MTLS_CA_BUNDLE_ENV_VAR,
    SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR,
    SERVICE_MTLS_ENABLED_ENV_VAR,
    SERVICE_MTLS_PRINCIPALS_ENV_VAR,
    SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR,
    parse_service_mtls_config,
    verify_client_certificate_chain,
)

LOOPBACK_BASE_URL = "https://127.0.0.1"
SYNTHETIC_URI_SAN = "uri:spiffe://openmed.test/clinic-api"
SYNTHETIC_DNS_SAN = "dns:clinic-api.test"
_SERVICE_ENV_VARS = (
    SERVICE_MTLS_ENABLED_ENV_VAR,
    SERVICE_MTLS_CA_BUNDLE_ENV_VAR,
    SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR,
    SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR,
    SERVICE_MTLS_PRINCIPALS_ENV_VAR,
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT",
    "OPENMED_SERVICE_AUTH_API_KEYS",
    "OPENMED_SERVICE_AUTH_JWKS",
    "OPENMED_SERVICE_AUTH_JWKS_FILE",
    "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_LOG_FORMAT",
    "OPENMED_SERVICE_LOG_LEVEL",
)


@dataclass(frozen=True)
class SyntheticMTLSMaterial:
    """Short-lived synthetic certificates used by the mTLS tests."""

    ca_bundle: Path
    client_pem: str
    untrusted_client_pem: str
    subject: str


class TLSClientScopeMiddleware:
    """Add synthetic ASGI TLS extension data to HTTP request scopes."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        client_cert_chain: Sequence[str] = (),
        client_cert_error: Optional[str] = None,
        client: Optional[tuple[str, int]] = None,
    ) -> None:
        self.app = app
        self.client_cert_chain = tuple(client_cert_chain)
        self.client_cert_error = client_cert_error
        self.client = client

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_scope = dict(scope)
        extensions = dict(request_scope.get("extensions", {}))
        extensions["tls"] = {
            "server_cert": None,
            "client_cert_chain": self.client_cert_chain,
            "client_cert_name": None,
            "client_cert_error": self.client_cert_error,
        }
        request_scope["extensions"] = extensions
        if self.client is not None:
            request_scope["client"] = self.client
        await self.app(request_scope, receive, send)


class PeerScopeMiddleware:
    """Override the HTTP peer address for trusted-proxy tests."""

    def __init__(self, app: ASGIApp, client: tuple[str, int]) -> None:
        self.app = app
        self.client = client

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope = dict(scope)
            scope["client"] = self.client
        await self.app(scope, receive, send)


@pytest.fixture(autouse=True)
def clean_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for env_var in _SERVICE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


@pytest.fixture(scope="module")
def synthetic_mtls_material(
    tmp_path_factory: pytest.TempPathFactory,
) -> SyntheticMTLSMaterial:
    fixture_dir = tmp_path_factory.mktemp("openmed-mtls")
    now = datetime.now(timezone.utc)

    ca_key, ca_certificate = _build_ca("OpenMed Synthetic Test CA", now=now)
    client_certificate = _build_client_certificate(
        ca_key,
        ca_certificate,
        common_name="synthetic-clinic-client",
        now=now,
    )
    untrusted_ca_key, untrusted_ca = _build_ca("Untrusted Synthetic CA", now=now)
    untrusted_client = _build_client_certificate(
        untrusted_ca_key,
        untrusted_ca,
        common_name="untrusted-synthetic-client",
        now=now,
    )

    ca_bundle = fixture_dir / "ca.pem"
    ca_bundle.write_bytes(ca_certificate.public_bytes(serialization.Encoding.PEM))
    return SyntheticMTLSMaterial(
        ca_bundle=ca_bundle,
        client_pem=client_certificate.public_bytes(serialization.Encoding.PEM).decode(
            "ascii"
        ),
        untrusted_client_pem=untrusted_client.public_bytes(
            serialization.Encoding.PEM
        ).decode("ascii"),
        subject=client_certificate.subject.rfc4514_string(),
    )


def test_mtls_defaults_to_disabled() -> None:
    config = parse_service_mtls_config()

    assert config.enabled is False
    assert config.ca_bundle is None


def test_enabling_mtls_requires_a_ca_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SERVICE_MTLS_ENABLED_ENV_VAR, "true")

    with pytest.raises(ValueError, match=SERVICE_MTLS_CA_BUNDLE_ENV_VAR):
        parse_service_mtls_config()


def test_valid_synthetic_client_certificate_verifies_and_exposes_identity(
    synthetic_mtls_material: SyntheticMTLSMaterial,
) -> None:
    identity = verify_client_certificate_chain(
        [synthetic_mtls_material.client_pem],
        synthetic_mtls_material.ca_bundle,
    )

    assert identity.subject == synthetic_mtls_material.subject
    assert identity.sans == (SYNTHETIC_URI_SAN, SYNTHETIC_DNS_SAN)
    assert len(identity.fingerprint_sha256) == 64


def test_untrusted_synthetic_client_certificate_is_rejected(
    synthetic_mtls_material: SyntheticMTLSMaterial,
) -> None:
    with pytest.raises(ValueError, match="untrusted"):
        verify_client_certificate_chain(
            [synthetic_mtls_material.untrusted_client_pem],
            synthetic_mtls_material.ca_bundle,
        )


def test_mtls_toggle_requires_certificate_without_code_changes(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_mtls_material: SyntheticMTLSMaterial,
) -> None:
    disabled_app = create_app()
    with TestClient(
        disabled_app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        disabled_response = client.get("/health")

    _enable_mtls(monkeypatch, synthetic_mtls_material)
    enabled_app = create_app()
    with TestClient(
        enabled_app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        enabled_response = client.get("/health")

    assert disabled_response.status_code == 200
    _assert_error(enabled_response, 401, "mtls_certificate_required")


def test_valid_request_maps_san_to_route_authorization_principal(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_mtls_material: SyntheticMTLSMaterial,
) -> None:
    _enable_mtls(
        monkeypatch,
        synthetic_mtls_material,
        scopes=["identity:read"],
    )
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
        json.dumps({"GET /whoami": ["identity:read"]}),
    )
    app = create_app()

    @app.get("/whoami")
    async def whoami(request: Request) -> dict[str, Any]:
        principal = request.state.auth_principal
        identity = request.state.mtls_identity
        return {
            "principal": principal.subject,
            "credential_type": principal.credential_type,
            "scopes": list(principal.scopes),
            "subject": identity.subject,
            "sans": list(identity.sans),
        }

    asgi_app = TLSClientScopeMiddleware(
        app,
        client_cert_chain=[synthetic_mtls_material.client_pem],
    )
    with TestClient(
        asgi_app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/whoami")

    assert response.status_code == 200
    assert response.json() == {
        "principal": "synthetic-clinic-api",
        "credential_type": "mtls",
        "scopes": ["identity:read"],
        "subject": synthetic_mtls_material.subject,
        "sans": [SYNTHETIC_URI_SAN, SYNTHETIC_DNS_SAN],
    }


def test_untrusted_request_returns_generic_authentication_error(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_mtls_material: SyntheticMTLSMaterial,
) -> None:
    _enable_mtls(monkeypatch, synthetic_mtls_material)
    app = TLSClientScopeMiddleware(
        create_app(),
        client_cert_chain=[synthetic_mtls_material.untrusted_client_pem],
    )

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/health")

    payload = _assert_error(response, 401, "mtls_certificate_invalid")
    rendered = json.dumps(payload)
    assert "Untrusted Synthetic CA" not in rendered
    assert synthetic_mtls_material.untrusted_client_pem not in rendered


def test_forwarded_certificate_requires_a_trusted_proxy(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_mtls_material: SyntheticMTLSMaterial,
) -> None:
    _enable_mtls(monkeypatch, synthetic_mtls_material)
    monkeypatch.setenv(
        SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR,
        "X-OpenMed-Client-Cert",
    )
    monkeypatch.setenv(SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR, "127.0.0.1/32")
    header_value = quote(synthetic_mtls_material.client_pem, safe="")

    trusted_app = PeerScopeMiddleware(create_app(), ("127.0.0.1", 54321))
    with TestClient(
        trusted_app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        trusted = client.get(
            "/health",
            headers={"X-OpenMed-Client-Cert": header_value},
        )

    untrusted_app = PeerScopeMiddleware(create_app(), ("203.0.113.9", 54321))
    with TestClient(
        untrusted_app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        untrusted = client.get(
            "/health",
            headers={"X-OpenMed-Client-Cert": header_value},
        )

    assert trusted.status_code == 200
    _assert_error(untrusted, 401, "mtls_certificate_invalid")


def test_explicit_api_key_can_authorize_inside_mtls_connection(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_mtls_material: SyntheticMTLSMaterial,
) -> None:
    _enable_mtls(monkeypatch, synthetic_mtls_material)
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [
                {
                    "id": "synthetic-key",
                    "key_hash": hash_api_key("synthetic-secret"),
                    "principal": "jwt-layer-principal",
                    "scopes": ["identity:read"],
                }
            ]
        ),
    )
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
        json.dumps({"GET /whoami": ["identity:read"]}),
    )
    app = create_app()

    @app.get("/whoami")
    async def whoami(request: Request) -> dict[str, Any]:
        return {
            "principal": request.state.auth_principal.subject,
            "credential_type": request.state.auth_principal.credential_type,
            "mtls_subject": request.state.mtls_identity.subject,
        }

    asgi_app = TLSClientScopeMiddleware(
        app,
        client_cert_chain=[synthetic_mtls_material.client_pem],
    )
    with TestClient(
        asgi_app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        response = client.get(
            "/whoami",
            headers={"X-API-Key": "synthetic-secret"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "principal": "jwt-layer-principal",
        "credential_type": "api_key",
        "mtls_subject": synthetic_mtls_material.subject,
    }


def test_access_log_contains_only_stable_mtls_identity(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_mtls_material: SyntheticMTLSMaterial,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _enable_mtls(monkeypatch, synthetic_mtls_material)
    caplog.set_level(logging.INFO, logger=ACCESS_LOGGER_NAME)
    app = TLSClientScopeMiddleware(
        create_app(),
        client_cert_chain=[synthetic_mtls_material.client_pem],
    )

    with TestClient(
        app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/health")

    assert response.status_code == 200
    records = [record for record in caplog.records if record.name == ACCESS_LOGGER_NAME]
    assert len(records) == 1
    rendered = records[0].getMessage()
    payload = json.loads(rendered)
    assert payload["identity"] == "synthetic-clinic-api"
    assert payload["credential_type"] == "mtls"
    assert synthetic_mtls_material.subject not in rendered
    assert SYNTHETIC_URI_SAN not in rendered
    assert SYNTHETIC_DNS_SAN not in rendered
    assert synthetic_mtls_material.client_pem not in rendered


def _enable_mtls(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticMTLSMaterial,
    *,
    scopes: Optional[list[str]] = None,
) -> None:
    monkeypatch.setenv(SERVICE_MTLS_ENABLED_ENV_VAR, "true")
    monkeypatch.setenv(SERVICE_MTLS_CA_BUNDLE_ENV_VAR, str(material.ca_bundle))
    monkeypatch.setenv(
        SERVICE_MTLS_PRINCIPALS_ENV_VAR,
        json.dumps(
            [
                {
                    "identities": [SYNTHETIC_URI_SAN],
                    "principal": "synthetic-clinic-api",
                    "scopes": scopes or [],
                }
            ]
        ),
    )


def _assert_error(response: Any, status_code: int, code: str) -> dict[str, Any]:
    assert response.status_code == status_code
    payload = response.json()
    assert payload["error"]["code"] == code
    assert "message" in payload["error"]
    assert "details" in payload["error"]
    return payload


def _build_ca(
    common_name: str,
    *,
    now: datetime,
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OpenMed Test"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    return key, certificate


def _build_client_certificate(
    ca_key: rsa.RSAPrivateKey,
    ca_certificate: x509.Certificate,
    *,
    common_name: str,
    now: datetime,
) -> x509.Certificate:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OpenMed Test"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_certificate.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(hours=1))
        .not_valid_after(now + timedelta(days=7))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.UniformResourceIdentifier("spiffe://openmed.test/clinic-api"),
                    x509.DNSName("clinic-api.test"),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(
                ca_certificate.public_key()
            ),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
