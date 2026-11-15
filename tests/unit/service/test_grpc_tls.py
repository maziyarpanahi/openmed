"""Listener policy and TLS/mTLS credential tests for the OpenMed gRPC service.

All certificates are ephemeral and synthetic; assertions record only reason
codes, controlled categories and certificate SHA-256 fingerprints.
"""

from __future__ import annotations

import json
import logging
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import grpc
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from openmed.service import grpc_server
from openmed.service import runtime as service_runtime
from openmed.service.grpc_security import (
    CATEGORY_CERTIFICATE_CHAIN,
    CATEGORY_CLIENT_AUTH,
    CATEGORY_CLIENT_CA,
    CATEGORY_LISTENER,
    CATEGORY_OVERRIDE,
    CATEGORY_PRIVATE_KEY,
    GRPC_REASON_ADDRESS_INVALID,
    GRPC_REASON_BIND_FAILED,
    GRPC_REASON_INSECURE_DEV_OVERRIDE_REJECTED,
    GRPC_REASON_INSECURE_REMOTE_BIND,
    GRPC_REASON_TLS_CERTIFICATE_EXPIRED,
    GRPC_REASON_TLS_CERTIFICATE_NOT_YET_VALID,
    GRPC_REASON_TLS_CLIENT_AUTH_AMBIGUOUS,
    GRPC_REASON_TLS_INCOMPLETE,
    GRPC_REASON_TLS_KEY_MISMATCH,
    GRPC_REASON_TLS_MALFORMED,
    GRPC_REASON_TLS_UNREADABLE,
    MAX_CREDENTIAL_FILE_BYTES,
    GrpcListener,
    GrpcStartupError,
    GrpcTLSConfig,
    bind_grpc_listener,
    certificate_sha256,
    is_loopback_grpc_address,
    load_grpc_server_credentials,
    resolve_grpc_listener,
)
from openmed.service.proto.generated import openmed_pb2, openmed_pb2_grpc

SERVER_NAME = "openmed-grpc.test"
_CHANNEL_OPTIONS = (
    ("grpc.ssl_target_name_override", SERVER_NAME),
    ("grpc.enable_http_proxy", 0),
)
_RPC_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class SyntheticTLSMaterial:
    """File paths and PEM bytes for ephemeral synthetic TLS identities."""

    directory: Path
    ca_file: Path
    ca_pem: bytes
    server_cert_file: Path
    server_key_file: Path
    server_sha256: str
    server_key_pem: bytes
    other_key_file: Path
    encrypted_key_file: Path
    expired_cert_file: Path
    expired_sha256: str
    future_cert_file: Path
    future_sha256: str
    client_cert_pem: bytes
    client_key_pem: bytes
    untrusted_ca_file: Path
    untrusted_ca_pem: bytes
    untrusted_client_cert_pem: bytes
    untrusted_client_key_pem: bytes


class _UnusedLoader:
    """Loader double that proves policy tests never load a model."""

    def __init__(self, config: Any) -> None:
        raise AssertionError("model loading is not expected in transport tests")


class _FakeServer:
    """Records listener calls without opening sockets."""

    def __init__(self, *, port: Any = 50051, error: Optional[Exception] = None):
        self.port = port
        self.error = error
        self.bound: list[tuple[str, str]] = []
        self.started = False
        self.stopped = False

    def add_insecure_port(self, address: str) -> Any:
        self.bound.append(("insecure", address))
        if self.error is not None:
            raise self.error
        return self.port

    def add_secure_port(self, address: str, credentials: Any) -> Any:
        self.bound.append(("secure", address))
        if self.error is not None:
            raise self.error
        return self.port

    def start(self) -> None:
        self.started = True

    def stop(self, grace: Any) -> None:
        self.stopped = True


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> service_runtime.ServiceRuntime:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for env_var in (
        "OPENMED_SERVICE_PRELOAD_MODELS",
        "OPENMED_SERVICE_KEEP_ALIVE",
        "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", _UnusedLoader)
    return service_runtime.ServiceRuntime.from_env()


@pytest.fixture(scope="module")
def tls_material(tmp_path_factory: pytest.TempPathFactory) -> SyntheticTLSMaterial:
    directory = tmp_path_factory.mktemp("openmed-grpc-tls")
    now = datetime.now(timezone.utc)

    ca_key, ca_cert = _build_ca("OpenMed Synthetic gRPC CA", now=now)
    server_key = ec.generate_private_key(ec.SECP256R1())
    server_cert = _build_leaf(
        ca_key,
        ca_cert,
        server_key,
        common_name=SERVER_NAME,
        usage=ExtendedKeyUsageOID.SERVER_AUTH,
        not_before=now - timedelta(hours=1),
        not_after=now + timedelta(days=7),
    )
    expired_cert = _build_leaf(
        ca_key,
        ca_cert,
        server_key,
        common_name=SERVER_NAME,
        usage=ExtendedKeyUsageOID.SERVER_AUTH,
        not_before=now - timedelta(days=10),
        not_after=now - timedelta(days=1),
    )
    future_cert = _build_leaf(
        ca_key,
        ca_cert,
        server_key,
        common_name=SERVER_NAME,
        usage=ExtendedKeyUsageOID.SERVER_AUTH,
        not_before=now + timedelta(days=1),
        not_after=now + timedelta(days=10),
    )
    client_key = ec.generate_private_key(ec.SECP256R1())
    client_cert = _build_leaf(
        ca_key,
        ca_cert,
        client_key,
        common_name="synthetic-grpc-client",
        usage=ExtendedKeyUsageOID.CLIENT_AUTH,
        not_before=now - timedelta(hours=1),
        not_after=now + timedelta(days=7),
    )
    untrusted_ca_key, untrusted_ca = _build_ca("Untrusted Synthetic CA", now=now)
    untrusted_client_key = ec.generate_private_key(ec.SECP256R1())
    untrusted_client = _build_leaf(
        untrusted_ca_key,
        untrusted_ca,
        untrusted_client_key,
        common_name="untrusted-synthetic-client",
        usage=ExtendedKeyUsageOID.CLIENT_AUTH,
        not_before=now - timedelta(hours=1),
        not_after=now + timedelta(days=7),
    )
    other_key = ec.generate_private_key(ec.SECP256R1())

    def write(name: str, data: bytes) -> Path:
        path = directory / name
        path.write_bytes(data)
        return path

    ca_pem = _cert_pem(ca_cert)
    server_key_pem = _key_pem(server_key)
    untrusted_ca_pem = _cert_pem(untrusted_ca)
    return SyntheticTLSMaterial(
        directory=directory,
        ca_file=write("ca.pem", ca_pem),
        ca_pem=ca_pem,
        server_cert_file=write("server.pem", _cert_pem(server_cert)),
        server_key_file=write("server-key.pem", server_key_pem),
        server_sha256=certificate_sha256(server_cert),
        server_key_pem=server_key_pem,
        other_key_file=write("other-key.pem", _key_pem(other_key)),
        encrypted_key_file=write(
            "encrypted-key.pem",
            server_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.BestAvailableEncryption(b"synthetic-passphrase"),
            ),
        ),
        expired_cert_file=write("expired.pem", _cert_pem(expired_cert)),
        expired_sha256=certificate_sha256(expired_cert),
        future_cert_file=write("future.pem", _cert_pem(future_cert)),
        future_sha256=certificate_sha256(future_cert),
        client_cert_pem=_cert_pem(client_cert),
        client_key_pem=_key_pem(client_key),
        untrusted_ca_file=write("untrusted-ca.pem", untrusted_ca_pem),
        untrusted_ca_pem=untrusted_ca_pem,
        untrusted_client_cert_pem=_cert_pem(untrusted_client),
        untrusted_client_key_pem=_key_pem(untrusted_client_key),
    )


def test_default_address_is_loopback() -> None:
    assert grpc_server.DEFAULT_GRPC_ADDRESS == "127.0.0.1:50051"
    assert is_loopback_grpc_address(grpc_server.DEFAULT_GRPC_ADDRESS)
    listener = resolve_grpc_listener(grpc_server.DEFAULT_GRPC_ADDRESS, profile="prod")
    assert listener.loopback is True
    assert listener.transport == "plaintext"
    assert listener.insecure_dev_override is False


def test_serve_default_binds_only_loopback(
    monkeypatch: pytest.MonkeyPatch, runtime: service_runtime.ServiceRuntime
) -> None:
    fake = _FakeServer()
    monkeypatch.setattr(grpc_server, "create_grpc_server", lambda **_: fake)

    assert grpc_server.serve(runtime=runtime) is fake
    assert fake.bound == [("insecure", "127.0.0.1:50051")]
    assert fake.started is True


def test_serve_default_answers_on_loopback(
    monkeypatch: pytest.MonkeyPatch, runtime: service_runtime.ServiceRuntime
) -> None:
    ports = _capture_bound_ports(monkeypatch)
    server = grpc_server.serve("127.0.0.1:0", runtime=runtime, max_workers=2)
    try:
        channel = grpc.insecure_channel(
            f"127.0.0.1:{ports[0]}", options=_CHANNEL_OPTIONS
        )
        with channel:
            assert _rpc_reaches_service(channel)
    finally:
        server.stop(0).wait()


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1:0",
        "127.8.9.10:50051",
        "[::1]:50051",
        "localhost:50051",
        "LOCALHOST:1",
        "[::ffff:127.0.0.1]:5",
        "unix:/var/run/openmed/grpc.sock",
        "unix-abstract:openmed-grpc",
    ],
)
def test_loopback_and_local_socket_addresses(address: str) -> None:
    assert is_loopback_grpc_address(address) is True


@pytest.mark.parametrize(
    "address",
    [
        "0.0.0.0:50051",
        "[::]:50051",
        ":50051",
        "10.0.0.5:50051",
        "[fe80::1]:50051",
        "[::ffff:10.0.0.5]:50051",
        "grpc.openmed.test:50051",
    ],
)
def test_wildcard_routable_and_named_addresses_are_not_loopback(address: str) -> None:
    assert is_loopback_grpc_address(address) is False


@pytest.mark.parametrize(
    "address",
    [
        "",
        " 127.0.0.1:50051",
        "127.0.0.1",
        "127.0.0.1:",
        "127.0.0.1:port",
        "127.0.0.1:70000",
        "127.0.0.1:-1",
        "::1:50051",
        "[::1]50051",
        "[::1]:",
        "[not-an-ip]:50051",
        "dns:///localhost:50051",
        "ipv4:127.0.0.1:50051",
        "unix:",
        50051,
    ],
)
def test_invalid_addresses_are_refused(address: Any) -> None:
    with pytest.raises(GrpcStartupError) as excinfo:
        resolve_grpc_listener(address, profile="test")
    assert excinfo.value.reason == GRPC_REASON_ADDRESS_INVALID
    assert excinfo.value.category == CATEGORY_LISTENER


@pytest.mark.parametrize("address", ["0.0.0.0:50051", "[::]:50051", "10.0.0.5:1"])
def test_external_plaintext_fails_before_server_exists(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    address: str,
) -> None:
    def unexpected(**_: Any) -> None:
        raise AssertionError("server must not be created")

    monkeypatch.setattr(grpc_server, "create_grpc_server", unexpected)

    with pytest.raises(GrpcStartupError) as excinfo:
        grpc_server.serve(address, runtime=runtime)

    assert excinfo.value.reason == GRPC_REASON_INSECURE_REMOTE_BIND
    assert excinfo.value.category == CATEGORY_LISTENER
    assert address not in str(excinfo.value)


@pytest.mark.parametrize("profile", ["dev", "test", " Dev "])
def test_dev_override_allows_external_plaintext_in_dev_profiles(
    caplog: pytest.LogCaptureFixture, profile: str
) -> None:
    caplog.set_level(logging.WARNING, logger="openmed.service.grpc_security")

    listener = resolve_grpc_listener(
        "0.0.0.0:50051", allow_insecure_remote_dev=True, profile=profile
    )

    assert listener.loopback is False
    assert listener.insecure_dev_override is True
    assert listener.transport == "plaintext"
    assert [record.getMessage() for record in caplog.records] == [
        "grpc_insecure_dev_override_active"
    ]
    assert "0.0.0.0" not in caplog.text


def test_dev_override_is_not_recorded_for_loopback() -> None:
    listener = resolve_grpc_listener(
        "127.0.0.1:50051", allow_insecure_remote_dev=True, profile="dev"
    )
    assert listener.insecure_dev_override is False


@pytest.mark.parametrize("profile", ["prod", "PROD", "", "low_resource", "staging"])
@pytest.mark.parametrize("address", ["0.0.0.0:50051", "127.0.0.1:50051"])
def test_dev_override_is_rejected_outside_dev_profiles(
    profile: str, address: str
) -> None:
    with pytest.raises(GrpcStartupError) as excinfo:
        resolve_grpc_listener(address, allow_insecure_remote_dev=True, profile=profile)

    assert excinfo.value.reason == GRPC_REASON_INSECURE_DEV_OVERRIDE_REJECTED
    assert excinfo.value.category == CATEGORY_OVERRIDE


def test_dev_override_defaults_to_prod_when_profile_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)

    def unexpected(**_: Any) -> None:
        raise AssertionError("server must not be created")

    monkeypatch.setattr(grpc_server, "create_grpc_server", unexpected)
    with pytest.raises(GrpcStartupError) as excinfo:
        grpc_server.serve("0.0.0.0:50051", allow_insecure_remote_dev=True)

    assert excinfo.value.reason == GRPC_REASON_INSECURE_DEV_OVERRIDE_REJECTED


def test_dev_override_reads_profile_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    with pytest.raises(GrpcStartupError) as excinfo:
        resolve_grpc_listener("0.0.0.0:50051", allow_insecure_remote_dev=True)
    assert excinfo.value.reason == GRPC_REASON_INSECURE_DEV_OVERRIDE_REJECTED

    monkeypatch.setenv("OPENMED_PROFILE", "dev")
    listener = resolve_grpc_listener("0.0.0.0:50051", allow_insecure_remote_dev=True)
    assert listener.insecure_dev_override is True


@pytest.mark.parametrize("value", ["true", 1, None])
def test_dev_override_must_be_a_real_boolean(value: Any) -> None:
    with pytest.raises(GrpcStartupError) as excinfo:
        resolve_grpc_listener(
            "0.0.0.0:50051", allow_insecure_remote_dev=value, profile="dev"
        )
    assert excinfo.value.reason == GRPC_REASON_INSECURE_DEV_OVERRIDE_REJECTED


def test_external_tls_bind_is_allowed_in_prod(
    tls_material: SyntheticTLSMaterial,
) -> None:
    listener = resolve_grpc_listener(
        "0.0.0.0:50051", tls=_tls_config(tls_material), profile="prod"
    )

    assert listener.loopback is False
    assert listener.transport == "tls"
    assert listener.insecure_dev_override is False


def test_tls_client_succeeds_and_untrusted_clients_fail(
    tls_material: SyntheticTLSMaterial,
    runtime: service_runtime.ServiceRuntime,
) -> None:
    listener = resolve_grpc_listener(
        "127.0.0.1:0", tls=_tls_config(tls_material), profile="prod"
    )
    assert listener.report()["transport"] == "tls"
    assert listener.report()["certificate_sha256"] == tls_material.server_sha256

    with _running(listener, runtime) as port:
        target = f"127.0.0.1:{port}"
        trusted = grpc.ssl_channel_credentials(root_certificates=tls_material.ca_pem)
        with grpc.secure_channel(target, trusted, options=_CHANNEL_OPTIONS) as channel:
            assert _rpc_reaches_service(channel)

        wrong_root = grpc.ssl_channel_credentials(
            root_certificates=tls_material.untrusted_ca_pem
        )
        with grpc.secure_channel(
            target, wrong_root, options=_CHANNEL_OPTIONS
        ) as channel:
            assert not _rpc_reaches_service(channel)

        with grpc.insecure_channel(target, options=_CHANNEL_OPTIONS) as channel:
            assert not _rpc_reaches_service(channel)


def test_mtls_trusted_client_succeeds_and_untrusted_clients_fail(
    tls_material: SyntheticTLSMaterial,
    runtime: service_runtime.ServiceRuntime,
) -> None:
    listener = resolve_grpc_listener(
        "127.0.0.1:0",
        tls=_tls_config(tls_material, client_ca_file=tls_material.ca_file),
        profile="prod",
    )
    report = listener.report()
    assert report["transport"] == "mtls"
    assert report["client_ca_sha256"] == [
        certificate_sha256(x509.load_pem_x509_certificate(tls_material.ca_pem))
    ]

    with _running(listener, runtime) as port:
        target = f"127.0.0.1:{port}"
        trusted_client = grpc.ssl_channel_credentials(
            root_certificates=tls_material.ca_pem,
            private_key=tls_material.client_key_pem,
            certificate_chain=tls_material.client_cert_pem,
        )
        with grpc.secure_channel(
            target, trusted_client, options=_CHANNEL_OPTIONS
        ) as channel:
            assert _rpc_reaches_service(channel)

        no_client_cert = grpc.ssl_channel_credentials(
            root_certificates=tls_material.ca_pem
        )
        with grpc.secure_channel(
            target, no_client_cert, options=_CHANNEL_OPTIONS
        ) as channel:
            assert not _rpc_reaches_service(channel)

        untrusted_client = grpc.ssl_channel_credentials(
            root_certificates=tls_material.ca_pem,
            private_key=tls_material.untrusted_client_key_pem,
            certificate_chain=tls_material.untrusted_client_cert_pem,
        )
        with grpc.secure_channel(
            target, untrusted_client, options=_CHANNEL_OPTIONS
        ) as channel:
            assert not _rpc_reaches_service(channel)


def test_serve_with_tls_answers_trusted_clients(
    monkeypatch: pytest.MonkeyPatch,
    tls_material: SyntheticTLSMaterial,
    runtime: service_runtime.ServiceRuntime,
) -> None:
    ports = _capture_bound_ports(monkeypatch)
    server = grpc_server.serve(
        "127.0.0.1:0", runtime=runtime, max_workers=2, tls=_tls_config(tls_material)
    )
    try:
        trusted = grpc.ssl_channel_credentials(root_certificates=tls_material.ca_pem)
        with grpc.secure_channel(
            f"127.0.0.1:{ports[0]}", trusted, options=_CHANNEL_OPTIONS
        ) as channel:
            assert _rpc_reaches_service(channel)
    finally:
        server.stop(0).wait()


def test_prevalidated_credentials_are_accepted(
    tls_material: SyntheticTLSMaterial,
) -> None:
    credentials = load_grpc_server_credentials(_tls_config(tls_material))
    listener = resolve_grpc_listener("[::]:50051", tls=credentials, profile="prod")

    assert listener.credentials is credentials
    assert listener.transport == "tls"


def _credential_cases() -> list[tuple[str, Any, str, str, Optional[str]]]:
    return [
        (
            "missing_certificate_file",
            lambda m: _tls_config(m, cert=m.directory / "missing.pem"),
            GRPC_REASON_TLS_UNREADABLE,
            CATEGORY_CERTIFICATE_CHAIN,
            None,
        ),
        (
            "key_path_is_directory",
            lambda m: _tls_config(m, key=m.directory),
            GRPC_REASON_TLS_UNREADABLE,
            CATEGORY_PRIVATE_KEY,
            None,
        ),
        (
            "missing_client_ca_file",
            lambda m: _tls_config(m, client_ca_file=m.directory / "missing-ca.pem"),
            GRPC_REASON_TLS_UNREADABLE,
            CATEGORY_CLIENT_CA,
            None,
        ),
        (
            "malformed_certificate",
            lambda m: _tls_config(m, cert=_write(m, "bad.pem", b"not a certificate")),
            GRPC_REASON_TLS_MALFORMED,
            CATEGORY_CERTIFICATE_CHAIN,
            None,
        ),
        (
            "empty_certificate",
            lambda m: _tls_config(m, cert=_write(m, "empty.pem", b"")),
            GRPC_REASON_TLS_MALFORMED,
            CATEGORY_CERTIFICATE_CHAIN,
            None,
        ),
        (
            "oversized_certificate",
            lambda m: _tls_config(
                m,
                cert=_write(m, "huge.pem", b"0" * (MAX_CREDENTIAL_FILE_BYTES + 1)),
            ),
            GRPC_REASON_TLS_MALFORMED,
            CATEGORY_CERTIFICATE_CHAIN,
            None,
        ),
        (
            "certificate_in_key_file",
            lambda m: _tls_config(m, key=m.server_cert_file),
            GRPC_REASON_TLS_MALFORMED,
            CATEGORY_PRIVATE_KEY,
            None,
        ),
        (
            "encrypted_private_key",
            lambda m: _tls_config(m, key=m.encrypted_key_file),
            GRPC_REASON_TLS_MALFORMED,
            CATEGORY_PRIVATE_KEY,
            None,
        ),
        (
            "malformed_client_ca",
            lambda m: _tls_config(
                m, client_ca_file=_write(m, "bad-ca.pem", b"-----BEGIN junk")
            ),
            GRPC_REASON_TLS_MALFORMED,
            CATEGORY_CLIENT_CA,
            None,
        ),
        (
            "mismatched_private_key",
            lambda m: _tls_config(m, key=m.other_key_file),
            GRPC_REASON_TLS_KEY_MISMATCH,
            CATEGORY_PRIVATE_KEY,
            "server",
        ),
        (
            "expired_certificate",
            lambda m: _tls_config(m, cert=m.expired_cert_file),
            GRPC_REASON_TLS_CERTIFICATE_EXPIRED,
            CATEGORY_CERTIFICATE_CHAIN,
            "expired",
        ),
        (
            "not_yet_valid_certificate",
            lambda m: _tls_config(m, cert=m.future_cert_file),
            GRPC_REASON_TLS_CERTIFICATE_NOT_YET_VALID,
            CATEGORY_CERTIFICATE_CHAIN,
            "future",
        ),
        (
            "certificate_not_configured",
            lambda m: GrpcTLSConfig(
                certificate_chain_file=None, private_key_file=m.server_key_file
            ),
            GRPC_REASON_TLS_INCOMPLETE,
            CATEGORY_CERTIFICATE_CHAIN,
            None,
        ),
        (
            "key_not_configured",
            lambda m: GrpcTLSConfig(
                certificate_chain_file=m.server_cert_file, private_key_file=" "
            ),
            GRPC_REASON_TLS_INCOMPLETE,
            CATEGORY_PRIVATE_KEY,
            None,
        ),
        (
            "client_cert_required_without_ca",
            lambda m: _tls_config(m, require_client_cert=True),
            GRPC_REASON_TLS_INCOMPLETE,
            CATEGORY_CLIENT_CA,
            None,
        ),
        (
            "client_ca_without_requirement",
            lambda m: _tls_config(
                m, client_ca_file=m.ca_file, require_client_cert=False
            ),
            GRPC_REASON_TLS_CLIENT_AUTH_AMBIGUOUS,
            CATEGORY_CLIENT_AUTH,
            None,
        ),
    ]


@pytest.mark.parametrize(
    ("build", "reason", "category", "fingerprint"),
    [case[1:] for case in _credential_cases()],
    ids=[case[0] for case in _credential_cases()],
)
def test_invalid_credential_material_fails_value_free(
    tls_material: SyntheticTLSMaterial,
    build: Any,
    reason: str,
    category: str,
    fingerprint: Optional[str],
) -> None:
    config = build(tls_material)

    with pytest.raises(GrpcStartupError) as excinfo:
        resolve_grpc_listener("0.0.0.0:50051", tls=config, profile="prod")

    error = excinfo.value
    assert error.reason == reason
    assert error.category == category
    expected_fingerprint = (
        getattr(tls_material, f"{fingerprint}_sha256") if fingerprint else None
    )
    assert error.fingerprint_sha256 == expected_fingerprint
    _assert_value_free(str(error), tls_material)
    assert error.__cause__ is None
    assert error.__suppress_context__ or error.__context__ is None


def test_bind_failure_is_a_startup_error_and_never_serves(
    monkeypatch: pytest.MonkeyPatch, runtime: service_runtime.ServiceRuntime
) -> None:
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        with pytest.raises(GrpcStartupError) as excinfo:
            grpc_server.serve(f"127.0.0.1:{port}", runtime=runtime, max_workers=1)
    finally:
        holder.close()

    assert excinfo.value.reason == GRPC_REASON_BIND_FAILED
    assert excinfo.value.category == CATEGORY_LISTENER
    assert str(port) not in str(excinfo.value)
    assert excinfo.value.__suppress_context__ is True


@pytest.mark.parametrize(
    "fake",
    [
        _FakeServer(port=0),
        _FakeServer(port=None),
        _FakeServer(error=RuntimeError("Failed to bind to address 10.0.0.1:1")),
    ],
    ids=["port_zero", "port_none", "grpc_raises"],
)
def test_serve_stops_and_never_starts_on_bind_failure(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    fake: _FakeServer,
) -> None:
    monkeypatch.setattr(grpc_server, "create_grpc_server", lambda **_: fake)

    with pytest.raises(GrpcStartupError) as excinfo:
        grpc_server.serve("127.0.0.1:50051", runtime=runtime)

    assert excinfo.value.reason == GRPC_REASON_BIND_FAILED
    assert "10.0.0.1" not in str(excinfo.value)
    assert fake.started is False
    assert fake.stopped is True


def test_bind_uses_secure_port_for_tls(tls_material: SyntheticTLSMaterial) -> None:
    fake = _FakeServer(port=4433)
    listener = resolve_grpc_listener(
        "127.0.0.1:4433", tls=_tls_config(tls_material), profile="prod"
    )

    assert bind_grpc_listener(fake, listener) == 4433  # type: ignore[arg-type]
    assert fake.bound == [("secure", "127.0.0.1:4433")]


def test_create_grpc_server_applies_interceptors(
    runtime: service_runtime.ServiceRuntime,
) -> None:
    seen: list[str] = []

    class RecordingInterceptor(grpc.ServerInterceptor):
        def intercept_service(self, continuation, handler_call_details):
            seen.append(handler_call_details.method)
            return continuation(handler_call_details)

    server = grpc_server.create_grpc_server(
        runtime=runtime, max_workers=2, interceptors=[RecordingInterceptor()]
    )
    port = bind_grpc_listener(server, resolve_grpc_listener("127.0.0.1:0"))
    server.start()
    try:
        with grpc.insecure_channel(
            f"127.0.0.1:{port}", options=_CHANNEL_OPTIONS
        ) as channel:
            assert _rpc_reaches_service(channel)
    finally:
        server.stop(0).wait()

    assert seen == ["/openmed.service.v1.OpenMedService/Analyze"]


def test_listener_report_is_value_free(tls_material: SyntheticTLSMaterial) -> None:
    listener = resolve_grpc_listener(
        "10.0.0.5:50051",
        tls=_tls_config(tls_material, client_ca_file=tls_material.ca_file),
        profile="prod",
    )
    report = listener.report()

    assert set(report) == {
        "loopback",
        "transport",
        "insecure_dev_override",
        "certificate_sha256",
        "certificate_not_valid_after",
        "client_ca_sha256",
    }
    serialized = json.dumps(report)
    assert "10.0.0.5" not in serialized
    _assert_value_free(serialized, tls_material)
    assert "10.0.0.5" not in repr(listener)
    assert "server_credentials" not in repr(listener.credentials)


def test_startup_error_drops_uncontrolled_values() -> None:
    error = GrpcStartupError(
        "/etc/openmed/key.pem", category="secret", fingerprint_sha256="not-a-digest"
    )

    assert error.reason == "grpc_failed"
    assert error.category is None
    assert error.fingerprint_sha256 is None
    assert str(error) == "gRPC startup refused: grpc_failed"


def test_listener_is_frozen() -> None:
    listener = GrpcListener(address="127.0.0.1:1", loopback=True)
    with pytest.raises(AttributeError):
        listener.loopback = False  # type: ignore[misc]


def _tls_config(
    material: SyntheticTLSMaterial,
    *,
    cert: Optional[Path] = None,
    key: Optional[Path] = None,
    client_ca_file: Optional[Path] = None,
    require_client_cert: Optional[bool] = None,
) -> GrpcTLSConfig:
    return GrpcTLSConfig(
        certificate_chain_file=cert or material.server_cert_file,
        private_key_file=key or material.server_key_file,
        client_ca_file=client_ca_file,
        require_client_cert=require_client_cert,
    )


def _write(material: SyntheticTLSMaterial, name: str, data: bytes) -> Path:
    path = material.directory / name
    path.write_bytes(data)
    return path


def _assert_value_free(text: str, material: SyntheticTLSMaterial) -> None:
    assert str(material.directory) not in text
    assert ".pem" not in text
    assert "BEGIN" not in text
    assert material.server_key_pem.decode("ascii") not in text


def _capture_bound_ports(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    ports: list[int] = []
    real_bind = grpc_server.bind_grpc_listener

    def capture(server: grpc.Server, listener: GrpcListener) -> int:
        port = real_bind(server, listener)
        ports.append(port)
        return port

    monkeypatch.setattr(grpc_server, "bind_grpc_listener", capture)
    return ports


class _running:
    """Context manager that binds a listener and serves the OpenMed service."""

    def __init__(
        self, listener: GrpcListener, runtime: service_runtime.ServiceRuntime
    ) -> None:
        self.listener = listener
        self.runtime = runtime
        self.server: Optional[grpc.Server] = None

    def __enter__(self) -> int:
        self.server = grpc_server.create_grpc_server(
            runtime=self.runtime, max_workers=2
        )
        port = bind_grpc_listener(self.server, self.listener)
        self.server.start()
        return port

    def __exit__(self, *exc_info: Any) -> None:
        assert self.server is not None
        self.server.stop(0).wait()


def _rpc_reaches_service(channel: grpc.Channel) -> bool:
    """Return True when the RPC reached the servicer, False on transport failure.

    The request names an undeclared model, so a reached servicer answers with
    ``INVALID_ARGUMENT`` before any model is loaded.
    """
    stub = openmed_pb2_grpc.OpenMedServiceStub(channel)
    try:
        stub.Analyze(
            openmed_pb2.AnalyzeRequest(text="synthetic", model_name="unknown/model"),
            timeout=_RPC_TIMEOUT_SECONDS,
        )
    except grpc.RpcError as exc:
        if exc.code() == grpc.StatusCode.INVALID_ARGUMENT:
            assert exc.details() == "model_not_served"
            return True
        assert exc.code() == grpc.StatusCode.UNAVAILABLE
        return False
    raise AssertionError("undeclared model must not be served")


def _build_ca(
    common_name: str, *, now: datetime
) -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    key = ec.generate_private_key(ec.SECP256R1())
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


def _build_leaf(
    ca_key: ec.EllipticCurvePrivateKey,
    ca_certificate: x509.Certificate,
    key: ec.EllipticCurvePrivateKey,
    *,
    common_name: str,
    usage: x509.ObjectIdentifier,
    not_before: datetime,
    not_after: datetime,
) -> x509.Certificate:
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
        .not_valid_before(not_before)
        .not_valid_after(not_after)
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
        .add_extension(x509.ExtendedKeyUsage([usage]), critical=False)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(common_name)]),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )


def _cert_pem(certificate: x509.Certificate) -> bytes:
    return certificate.public_bytes(serialization.Encoding.PEM)


def _key_pem(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
