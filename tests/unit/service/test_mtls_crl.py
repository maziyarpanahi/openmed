"""Unit tests for offline mTLS certificate revocation list (CRL) checks."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

from openmed.service.app import create_app
from openmed.service.mtls import (
    SERVICE_MTLS_CA_BUNDLE_ENV_VAR,
    SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR,
    SERVICE_MTLS_CRL_FAIL_CLOSED_ENV_VAR,
    SERVICE_MTLS_CRL_FILES_ENV_VAR,
    SERVICE_MTLS_CRL_GRACE_SECONDS_ENV_VAR,
    SERVICE_MTLS_ENABLED_ENV_VAR,
    SERVICE_MTLS_PRINCIPALS_ENV_VAR,
    SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR,
    LocalCRLChecker,
    MTLSAuthenticationError,
    ServiceMTLS,
    parse_service_mtls_config,
    verify_client_certificate_chain,
)

LOOPBACK_BASE_URL = "https://127.0.0.1"
MTLS_LOGGER_NAME = "openmed.service.mtls"
PRINCIPAL_SAN = "uri:spiffe://openmed.test/crl-clinic-api"

# Sentinel values that must never appear in responses, exceptions or logs.
REVOKED_CN_SENTINEL = "sentinel-revoked-client-cn"
REVOKED_SAN_SENTINEL = "sentinel-revoked-san.test"
REVOKED_SERIAL_SENTINEL = 0x5E171E1ABEEFC0DE7A11
CRL_DIR_SENTINEL = "sentinel-crl-directory"

_SERVICE_ENV_VARS = (
    SERVICE_MTLS_ENABLED_ENV_VAR,
    SERVICE_MTLS_CA_BUNDLE_ENV_VAR,
    SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR,
    SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR,
    SERVICE_MTLS_PRINCIPALS_ENV_VAR,
    SERVICE_MTLS_CRL_FILES_ENV_VAR,
    SERVICE_MTLS_CRL_GRACE_SECONDS_ENV_VAR,
    SERVICE_MTLS_CRL_FAIL_CLOSED_ENV_VAR,
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
class SyntheticCRLMaterial:
    """Synthetic CA, client certificates and CRL files for revocation tests."""

    now: datetime
    ca_bundle: Path
    ca_certificates: tuple[x509.Certificate, ...]
    good_client_pem: str
    revoked_client_pem: str
    revoked_client: x509.Certificate
    crl_dir: Path
    crl_valid: Path
    crl_valid_der: Path
    crl_empty: Path
    crl_wrong_key: Path
    crl_foreign_ca: Path
    crl_corrupt_signature: Path
    crl_malformed: Path
    crl_stale: Path
    crl_stale_within_grace: Path
    crl_delta: Path
    crl_next_update: datetime


class TLSClientScopeMiddleware:
    """Add synthetic ASGI TLS extension data to HTTP request scopes."""

    def __init__(self, app: ASGIApp, *, client_cert_chain: Sequence[str]) -> None:
        self.app = app
        self.client_cert_chain = tuple(client_cert_chain)

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
            "client_cert_error": None,
        }
        request_scope["extensions"] = extensions
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
def material(tmp_path_factory: pytest.TempPathFactory) -> SyntheticCRLMaterial:
    fixture_dir = tmp_path_factory.mktemp("openmed-mtls-crl")
    crl_dir = fixture_dir / CRL_DIR_SENTINEL
    crl_dir.mkdir()
    now = datetime.now(timezone.utc)

    ca_key, ca_certificate = _build_ca("OpenMed Synthetic CRL CA", now=now)
    good_client = _build_client_certificate(
        ca_key,
        ca_certificate,
        common_name="synthetic-good-client",
        san_dns="synthetic-good-client.test",
        serial_number=x509.random_serial_number(),
        now=now,
    )
    revoked_client = _build_client_certificate(
        ca_key,
        ca_certificate,
        common_name=REVOKED_CN_SENTINEL,
        san_dns=REVOKED_SAN_SENTINEL,
        serial_number=REVOKED_SERIAL_SENTINEL,
        now=now,
    )
    foreign_ca_key, foreign_ca = _build_ca("Foreign Synthetic CA", now=now)
    impostor_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    next_update = now + timedelta(days=1)
    revoked = [REVOKED_SERIAL_SENTINEL]
    valid_crl = _build_crl(
        ca_key, ca_certificate.subject, revoked, now=now, next_update=next_update
    )
    ca_bundle = fixture_dir / "ca.pem"
    ca_bundle.write_bytes(ca_certificate.public_bytes(serialization.Encoding.PEM))

    def write(name: str, data: bytes) -> Path:
        path = crl_dir / name
        path.write_bytes(data)
        return path

    corrupt = bytearray(valid_crl.public_bytes(serialization.Encoding.DER))
    corrupt[-8] ^= 0xFF

    return SyntheticCRLMaterial(
        now=now,
        ca_bundle=ca_bundle,
        ca_certificates=(ca_certificate,),
        good_client_pem=_pem(good_client),
        revoked_client_pem=_pem(revoked_client),
        revoked_client=revoked_client,
        crl_dir=crl_dir,
        crl_valid=write(
            "valid.crl.pem", valid_crl.public_bytes(serialization.Encoding.PEM)
        ),
        crl_valid_der=write(
            "valid.crl.der", valid_crl.public_bytes(serialization.Encoding.DER)
        ),
        crl_empty=write(
            "empty.crl.pem",
            _build_crl(
                ca_key, ca_certificate.subject, [], now=now, next_update=next_update
            ).public_bytes(serialization.Encoding.PEM),
        ),
        crl_wrong_key=write(
            "wrong-key.crl.pem",
            _build_crl(
                impostor_key,
                ca_certificate.subject,
                revoked,
                now=now,
                next_update=next_update,
            ).public_bytes(serialization.Encoding.PEM),
        ),
        crl_foreign_ca=write(
            "foreign.crl.pem",
            _build_crl(
                foreign_ca_key,
                foreign_ca.subject,
                revoked,
                now=now,
                next_update=next_update,
            ).public_bytes(serialization.Encoding.PEM),
        ),
        crl_corrupt_signature=write("corrupt.crl.der", bytes(corrupt)),
        crl_malformed=write(
            "malformed.crl.pem",
            b"-----BEGIN X509 CRL-----\nbm90IGEgY3Js\n-----END X509 CRL-----\n",
        ),
        crl_stale=write(
            "stale.crl.pem",
            _build_crl(
                ca_key,
                ca_certificate.subject,
                [],
                now=now - timedelta(days=3),
                next_update=now - timedelta(days=2),
            ).public_bytes(serialization.Encoding.PEM),
        ),
        crl_stale_within_grace=write(
            "stale-within-grace.crl.pem",
            _build_crl(
                ca_key,
                ca_certificate.subject,
                [],
                now=now - timedelta(days=2),
                next_update=now - timedelta(minutes=5),
            ).public_bytes(serialization.Encoding.PEM),
        ),
        crl_delta=write(
            "delta.crl.pem",
            _build_crl(
                ca_key,
                ca_certificate.subject,
                [],
                now=now,
                next_update=next_update,
                delta=True,
            ).public_bytes(serialization.Encoding.PEM),
        ),
        crl_next_update=next_update.replace(microsecond=0),
    )


# --- configuration -----------------------------------------------------------


def test_crl_settings_default_to_empty_and_production_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
) -> None:
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)
    _enable_mtls(monkeypatch, material, crl_files=())

    config = parse_service_mtls_config()

    assert config.crl_files == ()
    assert config.crl_grace_seconds == 0.0
    assert config.crl_fail_closed is True


def test_crl_settings_parse_comma_separated_files_and_grace(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
) -> None:
    _enable_mtls(monkeypatch, material, crl_files=(material.crl_valid,))
    monkeypatch.setenv(
        SERVICE_MTLS_CRL_FILES_ENV_VAR,
        f" {material.crl_valid} , ,{material.crl_empty}",
    )
    monkeypatch.setenv(SERVICE_MTLS_CRL_GRACE_SECONDS_ENV_VAR, "90")

    config = parse_service_mtls_config()

    assert config.crl_files == (material.crl_valid, material.crl_empty)
    assert config.crl_grace_seconds == 90.0
    assert config.crl_fail_closed is False


@pytest.mark.parametrize("profile", ["dev", "test"])
def test_development_profiles_default_to_report_only(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
    profile: str,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", profile)
    _enable_mtls(monkeypatch, material, crl_files=(material.crl_valid,))

    assert parse_service_mtls_config().crl_fail_closed is False


@pytest.mark.parametrize("profile", ["prod", "fast", "low_resource", "custom"])
def test_non_development_profiles_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
    profile: str,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", profile)
    _enable_mtls(monkeypatch, material, crl_files=(material.crl_valid,))

    assert parse_service_mtls_config().crl_fail_closed is True


def test_production_rejects_fail_open_override(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    _enable_mtls(monkeypatch, material, crl_files=(material.crl_valid,))
    monkeypatch.setenv(SERVICE_MTLS_CRL_FAIL_CLOSED_ENV_VAR, "false")

    with pytest.raises(ValueError, match=SERVICE_MTLS_CRL_FAIL_CLOSED_ENV_VAR):
        parse_service_mtls_config()


def test_development_can_opt_in_to_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "dev")
    _enable_mtls(monkeypatch, material, crl_files=(material.crl_valid,))
    monkeypatch.setenv(SERVICE_MTLS_CRL_FAIL_CLOSED_ENV_VAR, "true")

    assert parse_service_mtls_config().crl_fail_closed is True


@pytest.mark.parametrize("raw_value", ["-1", "nan", "soon"])
def test_invalid_grace_period_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
    raw_value: str,
) -> None:
    _enable_mtls(monkeypatch, material, crl_files=(material.crl_valid,))
    monkeypatch.setenv(SERVICE_MTLS_CRL_GRACE_SECONDS_ENV_VAR, raw_value)

    with pytest.raises(ValueError, match=SERVICE_MTLS_CRL_GRACE_SECONDS_ENV_VAR):
        parse_service_mtls_config()


# --- checker -----------------------------------------------------------------


def test_revoked_synthetic_certificate_is_rejected_by_checker(
    material: SyntheticCRLMaterial,
) -> None:
    checker = _checker(material, (material.crl_valid,), fail_closed=True)

    with pytest.raises(MTLSAuthenticationError) as excinfo:
        verify_client_certificate_chain(
            [material.revoked_client_pem],
            material.ca_bundle,
            crl_checker=checker,
        )

    assert excinfo.value.code == "client_certificate_revoked"
    assert excinfo.value.status_code == 401
    _assert_no_sentinels(str(excinfo.value), material)
    _assert_no_sentinels(repr(excinfo.value), material)


def test_unrevoked_certificate_authenticates_with_checker(
    material: SyntheticCRLMaterial,
) -> None:
    checker = _checker(material, (material.crl_valid,), fail_closed=True)

    identity = verify_client_certificate_chain(
        [material.good_client_pem],
        material.ca_bundle,
        crl_checker=checker,
    )

    assert "synthetic-good-client" in identity.subject


def test_der_encoded_crl_is_accepted(material: SyntheticCRLMaterial) -> None:
    checker = _checker(material, (material.crl_valid_der,), fail_closed=True)

    with pytest.raises(MTLSAuthenticationError) as excinfo:
        verify_client_certificate_chain(
            [material.revoked_client_pem],
            material.ca_bundle,
            crl_checker=checker,
        )

    assert excinfo.value.code == "client_certificate_revoked"


@pytest.mark.parametrize(
    "attribute",
    [
        "crl_wrong_key",
        "crl_foreign_ca",
        "crl_corrupt_signature",
        "crl_malformed",
        "crl_delta",
    ],
)
def test_invalid_crl_fails_closed_at_load(
    material: SyntheticCRLMaterial,
    attribute: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    crl_path = getattr(material, attribute)

    with pytest.raises(MTLSAuthenticationError) as excinfo:
        _checker(material, (crl_path,), fail_closed=True)

    assert excinfo.value.code == "crl_invalid"
    rendered = f"{excinfo.value!s} {excinfo.value!r}"
    _assert_no_sentinels(rendered, material)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None or excinfo.value.__suppress_context__
    _assert_no_sentinels(caplog.text, material)


def test_missing_crl_file_fails_closed_without_path(
    material: SyntheticCRLMaterial,
) -> None:
    missing = material.crl_dir / "missing.crl.pem"

    with pytest.raises(MTLSAuthenticationError) as excinfo:
        _checker(material, (missing,), fail_closed=True)

    assert excinfo.value.code == "crl_invalid"
    assert "missing.crl.pem" not in str(excinfo.value)
    _assert_no_sentinels(str(excinfo.value), material)


def test_invalid_crl_is_reported_and_skipped_in_report_only_mode(
    material: SyntheticCRLMaterial,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=MTLS_LOGGER_NAME)

    checker = _checker(
        material,
        (material.crl_wrong_key, material.crl_valid),
        fail_closed=False,
    )

    codes = _logged_codes(caplog)
    assert codes == ["crl_invalid"]
    _assert_no_sentinels(caplog.text, material)
    with pytest.raises(MTLSAuthenticationError) as excinfo:
        verify_client_certificate_chain(
            [material.revoked_client_pem],
            material.ca_bundle,
            crl_checker=checker,
        )
    assert excinfo.value.code == "client_certificate_revoked"


def test_multiple_crl_files_are_all_consulted(
    material: SyntheticCRLMaterial,
) -> None:
    checker = _checker(
        material,
        (material.crl_empty, material.crl_valid),
        fail_closed=True,
    )

    with pytest.raises(MTLSAuthenticationError) as excinfo:
        verify_client_certificate_chain(
            [material.revoked_client_pem],
            material.ca_bundle,
            crl_checker=checker,
        )
    assert excinfo.value.code == "client_certificate_revoked"
    verify_client_certificate_chain(
        [material.good_client_pem],
        material.ca_bundle,
        crl_checker=checker,
    )


def test_freshness_uses_injected_clock_and_grace_period(
    material: SyntheticCRLMaterial,
) -> None:
    next_update = material.crl_next_update
    within = _checker(
        material,
        (material.crl_valid,),
        fail_closed=True,
        grace_seconds=60,
        clock=lambda: next_update + timedelta(seconds=30),
    )
    beyond = _checker(
        material,
        (material.crl_valid,),
        fail_closed=True,
        grace_seconds=60,
        clock=lambda: next_update + timedelta(seconds=61),
    )

    verify_client_certificate_chain(
        [material.good_client_pem], material.ca_bundle, crl_checker=within
    )
    with pytest.raises(MTLSAuthenticationError) as excinfo:
        verify_client_certificate_chain(
            [material.good_client_pem], material.ca_bundle, crl_checker=beyond
        )
    assert excinfo.value.code == "crl_stale"
    assert excinfo.value.status_code == 503


def test_crl_that_goes_stale_while_running_is_caught(
    material: SyntheticCRLMaterial,
) -> None:
    current = [material.now]
    checker = _checker(
        material,
        (material.crl_valid,),
        fail_closed=True,
        clock=lambda: current[0],
    )
    verify_client_certificate_chain(
        [material.good_client_pem], material.ca_bundle, crl_checker=checker
    )

    current[0] = material.crl_next_update + timedelta(seconds=1)

    with pytest.raises(MTLSAuthenticationError) as excinfo:
        verify_client_certificate_chain(
            [material.good_client_pem], material.ca_bundle, crl_checker=checker
        )
    assert excinfo.value.code == "crl_stale"


def test_stale_crl_is_reported_once_and_allowed_in_report_only_mode(
    material: SyntheticCRLMaterial,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=MTLS_LOGGER_NAME)
    checker = _checker(
        material,
        (material.crl_valid,),
        fail_closed=False,
        clock=lambda: material.crl_next_update + timedelta(days=1),
    )

    for _ in range(3):
        verify_client_certificate_chain(
            [material.good_client_pem], material.ca_bundle, crl_checker=checker
        )

    assert _logged_codes(caplog) == ["crl_stale"]
    _assert_no_sentinels(caplog.text, material)


def test_revoked_certificate_is_rejected_even_when_crl_is_stale_in_report_mode(
    material: SyntheticCRLMaterial,
) -> None:
    checker = _checker(
        material,
        (material.crl_valid,),
        fail_closed=False,
        clock=lambda: material.crl_next_update + timedelta(days=1),
    )

    with pytest.raises(MTLSAuthenticationError) as excinfo:
        verify_client_certificate_chain(
            [material.revoked_client_pem], material.ca_bundle, crl_checker=checker
        )
    assert excinfo.value.code == "client_certificate_revoked"


def test_revoked_intermediate_ca_rejects_its_clients(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    root = _build_ca("Synthetic Root CA", now=now, path_length=1)
    intermediate = _build_ca("Synthetic Issuing CA", now=now, parent=root)
    client = _build_client_certificate(
        intermediate[0],
        intermediate[1],
        common_name="synthetic-intermediate-client",
        san_dns="synthetic-intermediate-client.test",
        serial_number=x509.random_serial_number(),
        now=now,
    )
    ca_bundle = tmp_path / "root.pem"
    ca_bundle.write_bytes(root[1].public_bytes(serialization.Encoding.PEM))
    revoking_crl = tmp_path / "root-revokes-intermediate.crl.pem"
    revoking_crl.write_bytes(
        _build_crl(
            root[0],
            root[1].subject,
            [intermediate[1].serial_number],
            now=now,
            next_update=now + timedelta(days=1),
        ).public_bytes(serialization.Encoding.PEM)
    )
    empty_crl = tmp_path / "root-empty.crl.pem"
    empty_crl.write_bytes(
        _build_crl(
            root[0], root[1].subject, [], now=now, next_update=now + timedelta(days=1)
        ).public_bytes(serialization.Encoding.PEM)
    )
    chain = [_pem(client), _pem(intermediate[1])]

    verify_client_certificate_chain(
        chain,
        ca_bundle,
        crl_checker=LocalCRLChecker((empty_crl,), (root[1],), fail_closed=True),
    )
    with pytest.raises(MTLSAuthenticationError) as excinfo:
        verify_client_certificate_chain(
            chain,
            ca_bundle,
            crl_checker=LocalCRLChecker((revoking_crl,), (root[1],), fail_closed=True),
        )
    assert excinfo.value.code == "client_certificate_revoked"


# --- REST service ------------------------------------------------------------


def test_service_rejects_revoked_certificate_and_accepts_unrevoked(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    _enable_mtls(monkeypatch, material, crl_files=(material.crl_valid,))

    good = _get(create_app(), material.good_client_pem)
    revoked = _get(create_app(), material.revoked_client_pem)

    assert good.status_code == 200
    payload = _assert_error(revoked, 401, "client_certificate_revoked")
    _assert_no_sentinels(json.dumps(payload), material)
    _assert_no_sentinels(json.dumps(dict(revoked.headers)), material)
    _assert_no_sentinels(caplog.text, material)


def test_revocation_check_runs_before_principal_mapping(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    _enable_mtls(monkeypatch, material, crl_files=(material.crl_valid,))
    calls: list[Any] = []
    original = ServiceMTLS._principal_for

    def recording_principal_for(self: ServiceMTLS, identity: Any) -> Any:
        calls.append(identity)
        return original(self, identity)

    monkeypatch.setattr(ServiceMTLS, "_principal_for", recording_principal_for)

    revoked = _get(create_app(), material.revoked_client_pem)

    _assert_error(revoked, 401, "client_certificate_revoked")
    assert calls == []

    good = _get(create_app(), material.good_client_pem)
    assert good.status_code == 200
    assert len(calls) == 1


@pytest.mark.parametrize(
    "attribute",
    ["crl_wrong_key", "crl_foreign_ca", "crl_corrupt_signature", "crl_malformed"],
)
def test_production_refuses_to_start_with_invalid_crl(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
    caplog: pytest.LogCaptureFixture,
    attribute: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    _enable_mtls(monkeypatch, material, crl_files=(getattr(material, attribute),))

    with pytest.raises(MTLSAuthenticationError) as excinfo:
        create_app()

    assert excinfo.value.code == "crl_invalid"
    _assert_no_sentinels(str(excinfo.value), material)
    _assert_no_sentinels(caplog.text, material)


def test_development_reports_invalid_crl_and_proceeds(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("OPENMED_PROFILE", "dev")
    _enable_mtls(monkeypatch, material, crl_files=(material.crl_wrong_key,))

    response = _get(create_app(), material.good_client_pem)

    assert response.status_code == 200
    assert "crl_invalid" in _logged_codes(caplog)
    _assert_no_sentinels(caplog.text, material)


def test_production_fails_closed_on_stale_crl(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    _enable_mtls(monkeypatch, material, crl_files=(material.crl_stale,))

    response = _get(create_app(), material.good_client_pem)

    payload = _assert_error(response, 503, "crl_stale")
    _assert_no_sentinels(json.dumps(payload), material)
    _assert_no_sentinels(caplog.text, material)


def test_production_accepts_stale_crl_within_grace_period(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    _enable_mtls(monkeypatch, material, crl_files=(material.crl_stale_within_grace,))
    monkeypatch.setenv(SERVICE_MTLS_CRL_GRACE_SECONDS_ENV_VAR, "3600")

    response = _get(create_app(), material.good_client_pem)

    assert response.status_code == 200


def test_development_reports_stale_crl_and_proceeds(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("OPENMED_PROFILE", "dev")
    _enable_mtls(monkeypatch, material, crl_files=(material.crl_stale,))

    response = _get(create_app(), material.good_client_pem)

    assert response.status_code == 200
    assert "crl_stale" in _logged_codes(caplog)
    _assert_no_sentinels(caplog.text, material)


def test_forwarded_revoked_certificate_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    _enable_mtls(monkeypatch, material, crl_files=(material.crl_valid,))
    monkeypatch.setenv(SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR, "X-Client-Cert")
    monkeypatch.setenv(SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR, "127.0.0.1/32")
    app = PeerScopeMiddleware(create_app(), ("127.0.0.1", 54321))

    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, raise_server_exceptions=False
    ) as client:
        revoked = client.get(
            "/health",
            headers={"X-Client-Cert": quote(material.revoked_client_pem, safe="")},
        )
        good = client.get(
            "/health",
            headers={"X-Client-Cert": quote(material.good_client_pem, safe="")},
        )

    payload = _assert_error(revoked, 401, "client_certificate_revoked")
    _assert_no_sentinels(json.dumps(payload), material)
    assert good.status_code == 200


# --- helpers -----------------------------------------------------------------


def _checker(
    material: SyntheticCRLMaterial,
    crl_files: Sequence[Path],
    *,
    fail_closed: bool,
    grace_seconds: float = 0.0,
    clock: Optional[Any] = None,
) -> LocalCRLChecker:
    return LocalCRLChecker(
        crl_files,
        material.ca_certificates,
        fail_closed=fail_closed,
        grace_seconds=grace_seconds,
        clock=clock,
    )


def _enable_mtls(
    monkeypatch: pytest.MonkeyPatch,
    material: SyntheticCRLMaterial,
    *,
    crl_files: Sequence[Path],
) -> None:
    monkeypatch.setenv(SERVICE_MTLS_ENABLED_ENV_VAR, "true")
    monkeypatch.setenv(SERVICE_MTLS_CA_BUNDLE_ENV_VAR, str(material.ca_bundle))
    monkeypatch.setenv(
        SERVICE_MTLS_PRINCIPALS_ENV_VAR,
        json.dumps(
            [
                {
                    "identities": [PRINCIPAL_SAN],
                    "principal": "synthetic-crl-clinic-api",
                    "scopes": [],
                }
            ]
        ),
    )
    if crl_files:
        monkeypatch.setenv(
            SERVICE_MTLS_CRL_FILES_ENV_VAR,
            ",".join(str(path) for path in crl_files),
        )


def _get(app: ASGIApp, client_pem: str) -> Any:
    asgi_app = TLSClientScopeMiddleware(app, client_cert_chain=[client_pem])
    with TestClient(
        asgi_app,
        base_url=LOOPBACK_BASE_URL,
        raise_server_exceptions=False,
    ) as client:
        return client.get("/health")


def _assert_error(response: Any, status_code: int, code: str) -> dict[str, Any]:
    assert response.status_code == status_code
    payload = response.json()
    assert payload["error"]["code"] == code
    return payload


def _logged_codes(caplog: pytest.LogCaptureFixture) -> list[str]:
    codes: list[str] = []
    for record in caplog.records:
        if record.name != MTLS_LOGGER_NAME:
            continue
        code = getattr(record, "openmed_code", None)
        if code is not None:
            codes.append(code)
    return codes


def _sentinels(material: SyntheticCRLMaterial) -> Iterator[str]:
    serial = REVOKED_SERIAL_SENTINEL
    hex_lower = f"{serial:x}"
    yield REVOKED_CN_SENTINEL
    yield REVOKED_SAN_SENTINEL
    yield str(serial)
    yield hex_lower
    yield hex_lower.upper()
    padded = hex_lower.zfill(len(hex_lower) + len(hex_lower) % 2)
    yield ":".join(padded[index : index + 2] for index in range(0, len(padded), 2))
    yield material.revoked_client.subject.rfc4514_string()
    yield CRL_DIR_SENTINEL
    yield str(material.crl_dir)


def _assert_no_sentinels(rendered: str, material: SyntheticCRLMaterial) -> None:
    lowered = rendered.lower()
    for sentinel in _sentinels(material):
        assert sentinel.lower() not in lowered


def _pem(certificate: x509.Certificate) -> str:
    return certificate.public_bytes(serialization.Encoding.PEM).decode("ascii")


def _build_crl(
    signing_key: Any,
    issuer: x509.Name,
    revoked_serials: Sequence[int],
    *,
    now: datetime,
    next_update: datetime,
    delta: bool = False,
) -> x509.CertificateRevocationList:
    builder = (
        x509.CertificateRevocationListBuilder()
        .issuer_name(issuer)
        .last_update(now - timedelta(minutes=1))
        .next_update(next_update)
    )
    for serial_number in revoked_serials:
        builder = builder.add_revoked_certificate(
            x509.RevokedCertificateBuilder()
            .serial_number(serial_number)
            .revocation_date(now - timedelta(minutes=1))
            .build()
        )
    if delta:
        builder = builder.add_extension(x509.DeltaCRLIndicator(1), critical=True)
    return builder.sign(signing_key, hashes.SHA256())


def _build_ca(
    common_name: str,
    *,
    now: datetime,
    path_length: Optional[int] = 0,
    parent: Optional[tuple[ec.EllipticCurvePrivateKey, x509.Certificate]] = None,
) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "OpenMed Test"),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )
    signing_key, issuer_name = (
        (key, subject) if parent is None else (parent[0], parent[1].subject)
    )
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=path_length), critical=True
        )
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
    )
    if parent is not None:
        builder = builder.add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(parent[1].public_key()),
            critical=False,
        )
    return key, builder.sign(signing_key, hashes.SHA256())


def _build_client_certificate(
    ca_key: ec.EllipticCurvePrivateKey,
    ca_certificate: x509.Certificate,
    *,
    common_name: str,
    san_dns: str,
    serial_number: int,
    now: datetime,
) -> x509.Certificate:
    key = ec.generate_private_key(ec.SECP256R1())
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
        .serial_number(serial_number)
        .not_valid_before(now - timedelta(hours=1))
        .not_valid_after(now + timedelta(days=7))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
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
                    x509.UniformResourceIdentifier(
                        "spiffe://openmed.test/crl-clinic-api"
                    ),
                    x509.DNSName(san_dns),
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
