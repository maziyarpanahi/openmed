"""Authentication and per-RPC authorization tests for the OpenMed gRPC service.

Every credential, certificate and text is synthetic. Servers run in-process on
loopback, model work is replaced by spies, and assertions record only status
codes and value-free ``openmed-reason`` trailers.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import grpc
import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID

from openmed.core.streaming import StreamingDeidentificationEvent
from openmed.service import grpc_policy, grpc_server
from openmed.service import runtime as service_runtime
from openmed.service.auth import (
    ROUTE_POLICIES,
    APIKeyCredential,
    AuthPrincipal,
    ServiceAuthConfig,
    authenticate_api_key,
    hash_api_key,
)
from openmed.service.grpc_policy import (
    GRPC_FAILURES,
    GRPC_RPC_ROUTES,
    OPENMED_REASON_METADATA_KEY,
    REASON_AUTH_RATE_LIMITED,
    REASON_AUTHENTICATION_REQUIRED,
    REASON_CREDENTIALS_MALFORMED,
    REASON_INSUFFICIENT_SCOPE,
    REASON_INVALID_CREDENTIALS,
    GrpcLimits,
    GrpcPolicyError,
    GrpcSecurityInterceptor,
    GrpcServicePolicy,
    grpc_peer_host,
    grpc_rpc_policies,
    require_authenticated_listener,
    validate_grpc_rpc_policies,
)
from openmed.service.grpc_security import (
    CATEGORY_LISTENER,
    GRPC_REASON_AUTH_REQUIRED,
    GrpcListener,
    GrpcStartupError,
    GrpcTLSConfig,
    bind_grpc_listener,
    resolve_grpc_listener,
)
from openmed.service.mtls import (
    MTLSPrincipalMapping,
    principal_for_mtls_identity,
    verify_client_certificate_chain,
)
from openmed.service.proto.generated import openmed_pb2, openmed_pb2_grpc
from openmed.service.runtime import ServiceThrottleConfig
from tests.unit.service.test_grpc_tls import (
    _CHANNEL_OPTIONS,
    SERVER_NAME,
    _build_ca,
    _build_leaf,
    _cert_pem,
    _key_pem,
)

RPCS = ("Analyze", "Extract", "Deidentify", "StreamDeidentify")
RPC_SCOPES = {
    "Analyze": ("analyze:write",),
    "Extract": ("pii:read",),
    "Deidentify": ("pii:write",),
    "StreamDeidentify": ("pii:write",),
}
SYNTHETIC_TEXT = "Synthetic visit note for record SYNTH-MRN-000417."
FULL_KEY = "synthetic-grpc-full-key-7f3a9c"
NARROW_KEY = "synthetic-grpc-narrow-key-1b2c3d"
UNKNOWN_KEY = "synthetic-grpc-unknown-key-000000"
JWT_SECRET = b"synthetic-grpc-jwt-secret-material-0001"
JWT_KID = "synthetic-grpc-kid"
MAPPED_CLIENT = "synthetic-grpc-client"
UNMAPPED_CLIENT = "unmapped-grpc-client"
RPC_TIMEOUT = 10


class FakeLoader:
    """Loader double; the spies below replace every model entry point."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **kwargs: Any) -> Any:
        raise AssertionError("no model may be loaded in gRPC policy tests")

    def loaded_models(self) -> dict[str, Any]:
        return {}


class ModelSpy:
    """Record model-work entry points instead of running inference."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.stream_kwargs: list[dict[str, Any]] = []
        self.stream_chunks: list[tuple[str, ...]] = []
        self._lock = threading.Lock()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "ModelSpy":
        monkeypatch.setattr(grpc_server, "_analyze_payload", self.analyze)
        monkeypatch.setattr(grpc_server, "_pii_extract_payload", self.extract)
        monkeypatch.setattr(grpc_server, "_pii_deidentify_payload", self.deidentify)
        monkeypatch.setattr(grpc_server, "deidentify_stream", self.stream)
        return self

    def _record(self, name: str) -> None:
        with self._lock:
            self.calls.append(name)

    def analyze(self, payload: Any, runtime: Any) -> dict[str, Any]:
        self._record("Analyze")
        return {"text": "", "model_name": "synthetic", "timestamp": "t"}

    def extract(self, payload: Any, runtime: Any) -> dict[str, Any]:
        self._record("Extract")
        return {"text": "", "model_name": "synthetic", "timestamp": "t"}

    def deidentify(self, payload: Any, runtime: Any) -> dict[str, Any]:
        self._record("Deidentify")
        return {
            "original_text": "",
            "deidentified_text": "[REDACTED]",
            "method": "mask",
            "timestamp": "t",
        }

    def stream(self, chunks: Any, **kwargs: Any) -> Iterator[Any]:
        self._record("StreamDeidentify")
        with self._lock:
            self.stream_chunks.append(tuple(chunks))
            self.stream_kwargs.append(kwargs)
        yield StreamingDeidentificationEvent(redacted_text="[REDACTED] ")
        yield StreamingDeidentificationEvent(redacted_text="", final=True)


@dataclass
class Served:
    """A running in-process server and its security interceptor."""

    port: int
    interceptor: GrpcSecurityInterceptor
    channel_credentials: Optional[grpc.ChannelCredentials] = None
    _channels: list[grpc.Channel] = field(default_factory=list)

    def stub(
        self, credentials: Optional[grpc.ChannelCredentials] = None
    ) -> openmed_pb2_grpc.OpenMedServiceStub:
        target = f"127.0.0.1:{self.port}"
        credentials = credentials or self.channel_credentials
        if credentials is None:
            channel = grpc.insecure_channel(target, options=_CHANNEL_OPTIONS)
        else:
            channel = grpc.secure_channel(target, credentials, options=_CHANNEL_OPTIONS)
        self._channels.append(channel)
        return openmed_pb2_grpc.OpenMedServiceStub(channel)

    def close(self) -> None:
        for channel in self._channels:
            channel.close()


class FakeClock:
    """Manually advanced monotonic clock for deterministic budgets."""

    def __init__(self, start: float = 1_000.0) -> None:
        self._now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


@contextmanager
def serving(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    policy: GrpcServicePolicy,
    *,
    listener: Optional[GrpcListener] = None,
    channel_credentials: Optional[grpc.ChannelCredentials] = None,
    clock: Optional[Callable[[], float]] = None,
    max_workers: int = 4,
) -> Iterator[Served]:
    """Serve the OpenMed service on loopback and expose its interceptor."""
    created: list[GrpcSecurityInterceptor] = []
    base = grpc_policy.GrpcSecurityInterceptor

    class Recording(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if clock is not None:
                kwargs["clock"] = clock
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(grpc_server, "GrpcSecurityInterceptor", Recording)
    listener = listener or resolve_grpc_listener("127.0.0.1:0", profile="test")
    server = grpc_server.create_grpc_server(
        runtime=runtime, max_workers=max_workers, policy=policy, listener=listener
    )
    port = bind_grpc_listener(server, listener)
    server.start()
    served = Served(
        port=port, interceptor=created[0], channel_credentials=channel_credentials
    )
    try:
        yield served
    finally:
        served.close()
        server.stop(0).wait()
        monkeypatch.setattr(grpc_server, "GrpcSecurityInterceptor", base)


def make_runtime(monkeypatch: pytest.MonkeyPatch) -> service_runtime.ServiceRuntime:
    """Return a test-profile runtime that can never load a model."""
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for env_var in (
        "OPENMED_SERVICE_PRELOAD_MODELS",
        "OPENMED_SERVICE_KEEP_ALIVE",
        "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
        "OPENMED_SERVICE_AUTH_ENABLED",
        "OPENMED_SERVICE_AUTH_API_KEYS",
        "OPENMED_SERVICE_AUTH_JWKS",
        "OPENMED_SERVICE_AUTH_JWKS_FILE",
        "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
        "OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT",
        "OPENMED_SERVICE_MTLS_PRINCIPALS",
        "OPENMED_SERVICE_RATE_LIMIT_RPS",
        "OPENMED_SERVICE_RATE_LIMIT_BURST",
        "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
        "OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS",
        "OPENMED_SERVICE_THROTTLE_KEY",
        "OPENMED_SERVICE_MAX_REQUEST_BYTES",
        "OPENMED_SERVICE_MAX_TEXT_LENGTH",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    return service_runtime.ServiceRuntime.from_env()


def call_rpc(
    stub: openmed_pb2_grpc.OpenMedServiceStub,
    rpc: str,
    *,
    text: str = SYNTHETIC_TEXT,
    metadata: Optional[tuple[tuple[str, Any], ...]] = None,
    timeout: float = RPC_TIMEOUT,
    **stream_fields: Any,
) -> Any:
    """Invoke one RPC with a synthetic request, draining streams."""
    if rpc == "Analyze":
        return stub.Analyze(
            openmed_pb2.AnalyzeRequest(text=text), metadata=metadata, timeout=timeout
        )
    if rpc == "Extract":
        return stub.Extract(
            openmed_pb2.ExtractRequest(text=text), metadata=metadata, timeout=timeout
        )
    if rpc == "Deidentify":
        return stub.Deidentify(
            openmed_pb2.DeidentifyRequest(text=text, method="mask"),
            metadata=metadata,
            timeout=timeout,
        )
    request = openmed_pb2.DeidentifyStreamRequest(
        request=openmed_pb2.DeidentifyRequest(
            text="" if "chunks" in stream_fields else text, method="mask"
        ),
        **stream_fields,
    )
    return list(stub.StreamDeidentify(request, metadata=metadata, timeout=timeout))


def outcome(
    stub: openmed_pb2_grpc.OpenMedServiceStub, rpc: str, **kwargs: Any
) -> tuple[grpc.StatusCode, Optional[str]]:
    """Return the status code and ``openmed-reason`` trailer of one call."""
    try:
        call_rpc(stub, rpc, **kwargs)
    except grpc.RpcError as exc:
        return exc.code(), reason_of(exc)
    return grpc.StatusCode.OK, None


def reason_of(error: grpc.RpcError) -> Optional[str]:
    """Return the ``openmed-reason`` trailer of a failed call, if any."""
    for key, value in error.trailing_metadata() or ():
        if key == OPENMED_REASON_METADATA_KEY:
            return value
    return None


def failure_of(reason: str) -> tuple[grpc.StatusCode, str]:
    """Return the expected status code and reason for a reason code."""
    return GRPC_FAILURES[reason].status, reason


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def make_jwt(
    claims: dict[str, Any],
    *,
    secret: bytes = JWT_SECRET,
    kid: str = JWT_KID,
) -> str:
    """Return an HS256 JWT signed with a synthetic secret."""
    header = {"alg": "HS256", "typ": "JWT", "kid": kid}
    signing_input = ".".join(
        b64url(json.dumps(part, separators=(",", ":")).encode("utf-8"))
        for part in (header, claims)
    )
    signature = hmac.new(secret, signing_input.encode("ascii"), hashlib.sha256)
    return f"{signing_input}.{b64url(signature.digest())}"


def jwt_claims(scope: str, *, lifetime: float = 6 * 3600.0) -> dict[str, Any]:
    now = int(time.time())
    return {
        "sub": "synthetic-jwt-client",
        "scope": scope,
        "iat": now,
        "exp": now + lifetime,
    }


def auth_config(**overrides: Any) -> ServiceAuthConfig:
    """Return enabled service auth with synthetic keys and an HS256 JWKS."""
    values: dict[str, Any] = {
        "enabled": True,
        "api_keys": (
            APIKeyCredential(
                key_id="full",
                key_hash=hash_api_key(FULL_KEY),
                principal="synthetic-full",
                scopes=("analyze:write", "pii:*"),
            ),
            APIKeyCredential(
                key_id="narrow",
                key_hash=hash_api_key(NARROW_KEY),
                principal="synthetic-narrow",
                scopes=("models:read",),
            ),
        ),
        "jwks": (
            {"kty": "oct", "kid": JWT_KID, "alg": "HS256", "k": b64url(JWT_SECRET)},
        ),
        "failure_rate_limit_rps": 0.0,
        "failure_rate_limit_burst": 0,
    }
    values.update(overrides)
    return ServiceAuthConfig(**values)


def auth_policy(**auth_overrides: Any) -> GrpcServicePolicy:
    return GrpcServicePolicy(auth=auth_config(**auth_overrides))


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> service_runtime.ServiceRuntime:
    return make_runtime(monkeypatch)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> ModelSpy:
    return ModelSpy().install(monkeypatch)


@dataclass(frozen=True)
class MTLSMaterial:
    """Synthetic files and PEM bytes for one mutual-TLS deployment."""

    ca_file: Path
    ca_pem: bytes
    server_cert_file: Path
    server_key_file: Path
    mapped_cert_pem: bytes
    mapped_key_pem: bytes
    unmapped_cert_pem: bytes
    unmapped_key_pem: bytes


@pytest.fixture(scope="module")
def mtls_material(tmp_path_factory: pytest.TempPathFactory) -> MTLSMaterial:
    directory = tmp_path_factory.mktemp("openmed-grpc-auth")
    now = datetime.now(timezone.utc)
    ca_key, ca_cert = _build_ca("OpenMed Synthetic gRPC Auth CA", now=now)

    def leaf(name: str, usage: x509.ObjectIdentifier) -> tuple[bytes, bytes]:
        key = ec.generate_private_key(ec.SECP256R1())
        certificate = _build_leaf(
            ca_key,
            ca_cert,
            key,
            common_name=name,
            usage=usage,
            not_before=now - timedelta(hours=1),
            not_after=now + timedelta(days=7),
        )
        return _cert_pem(certificate), _key_pem(key)

    server_cert, server_key = leaf(SERVER_NAME, ExtendedKeyUsageOID.SERVER_AUTH)
    mapped_cert, mapped_key = leaf(MAPPED_CLIENT, ExtendedKeyUsageOID.CLIENT_AUTH)
    unmapped_cert, unmapped_key = leaf(UNMAPPED_CLIENT, ExtendedKeyUsageOID.CLIENT_AUTH)

    def write(name: str, data: bytes) -> Path:
        path = directory / name
        path.write_bytes(data)
        return path

    ca_pem = _cert_pem(ca_cert)
    return MTLSMaterial(
        ca_file=write("ca.pem", ca_pem),
        ca_pem=ca_pem,
        server_cert_file=write("server.pem", server_cert),
        server_key_file=write("server-key.pem", server_key),
        mapped_cert_pem=mapped_cert,
        mapped_key_pem=mapped_key,
        unmapped_cert_pem=unmapped_cert,
        unmapped_key_pem=unmapped_key,
    )


def mtls_listener(material: MTLSMaterial, address: str = "127.0.0.1:0") -> GrpcListener:
    return resolve_grpc_listener(
        address,
        tls=GrpcTLSConfig(
            certificate_chain_file=material.server_cert_file,
            private_key_file=material.server_key_file,
            client_ca_file=material.ca_file,
        ),
        profile="test",
    )


def client_credentials(
    material: MTLSMaterial, *, mapped: bool
) -> grpc.ChannelCredentials:
    return grpc.ssl_channel_credentials(
        root_certificates=material.ca_pem,
        private_key=material.mapped_key_pem if mapped else material.unmapped_key_pem,
        certificate_chain=(
            material.mapped_cert_pem if mapped else material.unmapped_cert_pem
        ),
    )


MTLS_PRINCIPALS = (
    MTLSPrincipalMapping(
        identities=frozenset({f"dns:{MAPPED_CLIENT}"}),
        principal="synthetic-clinic-api",
        scopes=("analyze:write", "pii:*"),
    ),
)


class FakeContext:
    """Minimal servicer context for direct interceptor checks."""

    def __init__(
        self,
        metadata: tuple[tuple[str, Any], ...] = (),
        *,
        auth_context: Optional[dict[str, list[bytes]]] = None,
        peer: str = "ipv4:127.0.0.1:50000",
        time_remaining: float = 1e18,
    ) -> None:
        self._metadata = metadata
        self._auth_context = auth_context or {}
        self._peer = peer
        self._time_remaining = time_remaining

    def invocation_metadata(self) -> tuple[tuple[str, Any], ...]:
        return self._metadata

    def auth_context(self) -> dict[str, list[bytes]]:
        return self._auth_context

    def peer(self) -> str:
        return self._peer

    def time_remaining(self) -> float:
        return self._time_remaining

    def is_active(self) -> bool:
        return True


# --- Policy table -----------------------------------------------------------


def test_rpc_policy_table_mirrors_rest_routes() -> None:
    policies = grpc_rpc_policies(ServiceAuthConfig())

    assert set(policies) == {
        f"/openmed.service.v1.OpenMedService/{rpc}" for rpc in RPCS
    }
    for rpc in RPCS:
        method = f"/openmed.service.v1.OpenMedService/{rpc}"
        route = GRPC_RPC_ROUTES[method]
        assert policies[method].scopes == RPC_SCOPES[rpc]
        assert policies[method].scopes == ROUTE_POLICIES[route].scopes
        assert policies[method].requires_authentication is True
        assert policies[method].admission is True


def test_every_proto_rpc_has_a_declared_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    validate_grpc_rpc_policies()

    partial = {
        method: route
        for method, route in GRPC_RPC_ROUTES.items()
        if not method.endswith("/StreamDeidentify")
    }
    monkeypatch.setattr(grpc_policy, "GRPC_RPC_ROUTES", partial)
    with pytest.raises(ValueError, match="no declared security policy"):
        validate_grpc_rpc_policies()


def test_rest_route_scope_overrides_apply_to_rpcs() -> None:
    config = ServiceAuthConfig(
        route_scopes={
            **ServiceAuthConfig().route_scopes,
            ("POST", "/analyze"): ("synthetic:override",),
        }
    )
    policies = grpc_rpc_policies(config)

    assert policies["/openmed.service.v1.OpenMedService/Analyze"].scopes == (
        "synthetic:override",
    )
    assert policies["/openmed.service.v1.OpenMedService/Deidentify"].scopes == (
        "pii:write",
    )


def test_unknown_methods_follow_deny_by_default() -> None:
    method = "/synthetic.Probe/Ping"
    strict = GrpcSecurityInterceptor(auth_policy()).rpc_policy(method)
    relaxed = GrpcSecurityInterceptor(auth_policy(deny_by_default=False)).rpc_policy(
        method
    )

    assert strict.requires_authentication is True
    assert strict.scopes == ()
    assert relaxed.requires_authentication is False


# --- Authentication over the wire -------------------------------------------


@pytest.mark.parametrize("rpc", RPCS)
def test_missing_credentials_fail_before_model_work(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    rpc: str,
) -> None:
    with serving(monkeypatch, runtime, auth_policy()) as served:
        assert outcome(served.stub(), rpc) == failure_of(REASON_AUTHENTICATION_REQUIRED)

    assert spy.calls == []
    assert runtime._loader is None


INVALID_CREDENTIALS = {
    "unknown_api_key": (("x-api-key", UNKNOWN_KEY),),
    "unknown_apikey_scheme": (("authorization", f"ApiKey {UNKNOWN_KEY}"),),
    "unsigned_jwt": (
        ("authorization", f"Bearer {make_jwt(jwt_claims('pii:*'), secret=b'x' * 32)}"),
    ),
    "garbage_bearer": (("authorization", "Bearer not-a-jwt"),),
    "expired_jwt": (
        ("authorization", f"Bearer {make_jwt(jwt_claims('pii:*', lifetime=-60))}"),
    ),
    "unsupported_scheme": (("authorization", "Basic c3ludGhldGljOnVzZXI="),),
}


@pytest.mark.parametrize("rpc", RPCS)
@pytest.mark.parametrize("case", sorted(INVALID_CREDENTIALS))
def test_invalid_credentials_fail_before_model_work(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    rpc: str,
    case: str,
) -> None:
    with serving(monkeypatch, runtime, auth_policy()) as served:
        result = outcome(served.stub(), rpc, metadata=INVALID_CREDENTIALS[case])

    assert result == failure_of(REASON_INVALID_CREDENTIALS)
    assert spy.calls == []


MALFORMED_METADATA = {
    "duplicate_api_key": (("x-api-key", FULL_KEY), ("x-api-key", FULL_KEY)),
    "duplicate_authorization": (
        ("authorization", f"ApiKey {FULL_KEY}"),
        ("authorization", f"ApiKey {FULL_KEY}"),
    ),
    "api_key_and_authorization": (
        ("x-api-key", FULL_KEY),
        ("authorization", f"ApiKey {FULL_KEY}"),
    ),
    "scheme_without_value": (("authorization", "Bearer"),),
    "binary_api_key": (("x-api-key-bin", FULL_KEY.encode("ascii")),),
    "binary_authorization": (("authorization-bin", b"Bearer synthetic"),),
}


@pytest.mark.parametrize("rpc", RPCS)
@pytest.mark.parametrize("case", sorted(MALFORMED_METADATA))
def test_malformed_metadata_fails_before_model_work(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    rpc: str,
    case: str,
) -> None:
    with serving(monkeypatch, runtime, auth_policy()) as served:
        result = outcome(served.stub(), rpc, metadata=MALFORMED_METADATA[case])

    assert result == failure_of(REASON_CREDENTIALS_MALFORMED)
    assert spy.calls == []


@pytest.mark.parametrize(
    "value",
    [
        "x" * (grpc_policy.MAX_CREDENTIAL_METADATA_CHARS + 1),
        "synthetic\x00key",
        "synthetic\tkey",
        "synthétique",
        "   ",
        b"synthetic-bytes",
    ],
    ids=["oversized", "nul", "control", "non_ascii", "blank", "bytes"],
)
@pytest.mark.parametrize("name", ["x-api-key", "authorization"])
def test_malformed_credential_values_are_refused(name: str, value: Any) -> None:
    interceptor = GrpcSecurityInterceptor(auth_policy())

    with pytest.raises(GrpcPolicyError) as excinfo:
        interceptor.authenticate(FakeContext(((name, value),)))

    assert excinfo.value.reason == REASON_CREDENTIALS_MALFORMED
    assert str(excinfo.value) == REASON_CREDENTIALS_MALFORMED


@pytest.mark.parametrize("rpc", RPCS)
def test_under_scoped_credentials_are_denied_before_model_work(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    rpc: str,
) -> None:
    other_scope = "pii:read" if rpc != "Extract" else "pii:write"
    with serving(monkeypatch, runtime, auth_policy()) as served:
        stub = served.stub()
        narrow = outcome(stub, rpc, metadata=(("x-api-key", NARROW_KEY),))
        jwt = outcome(
            stub,
            rpc,
            metadata=(
                ("authorization", f"Bearer {make_jwt(jwt_claims(other_scope))}"),
            ),
        )

    assert narrow == failure_of(REASON_INSUFFICIENT_SCOPE)
    assert jwt == failure_of(REASON_INSUFFICIENT_SCOPE)
    assert spy.calls == []


VALID_CREDENTIALS = {
    "x_api_key": (("x-api-key", FULL_KEY),),
    "apikey_scheme": (("authorization", f"ApiKey {FULL_KEY}"),),
    "bearer_jwt": (
        (
            "authorization",
            f"Bearer {make_jwt(jwt_claims('analyze:write pii:read pii:write'))}",
        ),
    ),
    "bearer_jwt_namespace_wildcard": (
        ("authorization", f"Bearer {make_jwt(jwt_claims('analyze:* pii:*'))}"),
    ),
}


@pytest.mark.parametrize("rpc", RPCS)
@pytest.mark.parametrize("case", sorted(VALID_CREDENTIALS))
def test_valid_credentials_reach_every_rpc(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    rpc: str,
    case: str,
) -> None:
    with serving(monkeypatch, runtime, auth_policy()) as served:
        result = outcome(served.stub(), rpc, metadata=VALID_CREDENTIALS[case])

    assert result == (grpc.StatusCode.OK, None)
    assert spy.calls == [rpc]


def test_unary_and_stream_rpcs_make_equivalent_decisions(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
) -> None:
    cases = {
        "missing": None,
        **INVALID_CREDENTIALS,
        **MALFORMED_METADATA,
        **VALID_CREDENTIALS,
        "narrow": (("x-api-key", NARROW_KEY),),
        "read_only_jwt": (
            ("authorization", f"Bearer {make_jwt(jwt_claims('pii:read'))}"),
        ),
    }
    with serving(monkeypatch, runtime, auth_policy()) as served:
        stub = served.stub()
        for name, metadata in cases.items():
            unary = outcome(stub, "Deidentify", metadata=metadata)
            stream = outcome(stub, "StreamDeidentify", metadata=metadata)
            assert unary == stream, name


def test_principal_is_the_rest_principal_type() -> None:
    interceptor = GrpcSecurityInterceptor(auth_policy())

    principal = interceptor.authenticate(FakeContext((("x-api-key", FULL_KEY),)))

    assert isinstance(principal, AuthPrincipal)
    assert principal == authenticate_api_key(FULL_KEY, auth_config().api_keys)
    assert principal.credential_type == "api_key"


def test_failed_authentication_is_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
) -> None:
    clock = FakeClock()
    policy = auth_policy(failure_rate_limit_rps=1.0, failure_rate_limit_burst=2)
    with serving(monkeypatch, runtime, policy, clock=clock) as served:
        stub = served.stub()
        bad = (("x-api-key", UNKNOWN_KEY),)
        results = [outcome(stub, "Deidentify", metadata=bad) for _ in range(3)]
        valid = outcome(stub, "Deidentify", metadata=(("x-api-key", FULL_KEY),))
        clock.advance(1.0)
        recovered = outcome(stub, "Deidentify", metadata=bad)

    assert results == [
        failure_of(REASON_INVALID_CREDENTIALS),
        failure_of(REASON_INVALID_CREDENTIALS),
        failure_of(REASON_AUTH_RATE_LIMITED),
    ]
    assert valid == (grpc.StatusCode.OK, None)
    assert recovered == failure_of(REASON_INVALID_CREDENTIALS)
    assert spy.calls == ["Deidentify"]


# --- Loopback development and listener policy -------------------------------


@pytest.mark.parametrize("rpc", RPCS)
def test_loopback_development_mode_stays_anonymous(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    rpc: str,
) -> None:
    with serving(monkeypatch, runtime, GrpcServicePolicy()) as served:
        assert served.interceptor.authentication_enabled is False
        assert outcome(served.stub(), rpc) == (grpc.StatusCode.OK, None)

    assert spy.calls == [rpc]


def test_default_serve_is_anonymous_on_loopback(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
) -> None:
    ports: list[int] = []
    real_bind = grpc_server.bind_grpc_listener

    def capture(server: grpc.Server, listener: GrpcListener) -> int:
        ports.append(real_bind(server, listener))
        return ports[-1]

    monkeypatch.setattr(grpc_server, "bind_grpc_listener", capture)
    server = grpc_server.serve("127.0.0.1:0", runtime=runtime, max_workers=2)
    try:
        with grpc.insecure_channel(
            f"127.0.0.1:{ports[0]}", options=_CHANNEL_OPTIONS
        ) as channel:
            stub = openmed_pb2_grpc.OpenMedServiceStub(channel)
            assert outcome(stub, "Deidentify") == (grpc.StatusCode.OK, None)
    finally:
        server.stop(0).wait()


def test_env_enabled_auth_applies_to_default_server(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [{"id": "env", "key_hash": hash_api_key(FULL_KEY), "scopes": "pii:*"}]
        ),
    )
    server = grpc_server.create_grpc_server(runtime=runtime, max_workers=2)
    port = bind_grpc_listener(server, resolve_grpc_listener("127.0.0.1:0"))
    server.start()
    try:
        with grpc.insecure_channel(
            f"127.0.0.1:{port}", options=_CHANNEL_OPTIONS
        ) as channel:
            stub = openmed_pb2_grpc.OpenMedServiceStub(channel)
            missing = outcome(stub, "Deidentify")
            allowed = outcome(stub, "Deidentify", metadata=(("x-api-key", FULL_KEY),))
            denied = outcome(stub, "Analyze", metadata=(("x-api-key", FULL_KEY),))
    finally:
        server.stop(0).wait()

    assert missing == failure_of(REASON_AUTHENTICATION_REQUIRED)
    assert allowed == (grpc.StatusCode.OK, None)
    assert denied == failure_of(REASON_INSUFFICIENT_SCOPE)


def test_reachable_tls_listener_without_auth_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    mtls_material: MTLSMaterial,
) -> None:
    def unexpected(*_: Any, **__: Any) -> None:
        raise AssertionError("nothing may be bound")

    monkeypatch.setattr(grpc_server, "bind_grpc_listener", unexpected)
    tls = GrpcTLSConfig(
        certificate_chain_file=mtls_material.server_cert_file,
        private_key_file=mtls_material.server_key_file,
    )
    with pytest.raises(GrpcStartupError) as excinfo:
        grpc_server.serve(
            "0.0.0.0:50051", runtime=runtime, tls=tls, policy=GrpcServicePolicy()
        )

    assert excinfo.value.reason == GRPC_REASON_AUTH_REQUIRED
    assert excinfo.value.category == CATEGORY_LISTENER
    assert "0.0.0.0" not in str(excinfo.value)


def test_reachable_listeners_accept_authenticating_policies(
    mtls_material: MTLSMaterial,
) -> None:
    tls_only = resolve_grpc_listener(
        "0.0.0.0:50051",
        tls=GrpcTLSConfig(
            certificate_chain_file=mtls_material.server_cert_file,
            private_key_file=mtls_material.server_key_file,
        ),
        profile="prod",
    )
    mutual = mtls_listener(mtls_material, "0.0.0.0:50051")
    dev_plaintext = resolve_grpc_listener(
        "0.0.0.0:50051", allow_insecure_remote_dev=True, profile="dev"
    )

    require_authenticated_listener(tls_only, auth_policy())
    require_authenticated_listener(mutual, GrpcServicePolicy())
    require_authenticated_listener(dev_plaintext, GrpcServicePolicy())
    require_authenticated_listener(
        resolve_grpc_listener("127.0.0.1:50051"), GrpcServicePolicy()
    )
    with pytest.raises(GrpcStartupError):
        require_authenticated_listener(tls_only, GrpcServicePolicy())


# --- Mutual TLS principals --------------------------------------------------


@pytest.mark.parametrize("rpc", RPCS)
def test_mapped_client_certificate_authorizes_every_rpc(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    mtls_material: MTLSMaterial,
    rpc: str,
) -> None:
    policy = GrpcServicePolicy(mtls_principals=MTLS_PRINCIPALS)
    with serving(
        monkeypatch,
        runtime,
        policy,
        listener=mtls_listener(mtls_material),
        channel_credentials=client_credentials(mtls_material, mapped=True),
    ) as served:
        assert served.interceptor.authentication_enabled is True
        mapped = outcome(served.stub(), rpc)
        unmapped = outcome(
            served.stub(client_credentials(mtls_material, mapped=False)), rpc
        )

    assert mapped == (grpc.StatusCode.OK, None)
    assert unmapped == failure_of(REASON_INSUFFICIENT_SCOPE)
    assert spy.calls == [rpc]


def test_metadata_credentials_take_precedence_over_client_certificates(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    mtls_material: MTLSMaterial,
) -> None:
    policy = GrpcServicePolicy(auth=auth_config(), mtls_principals=MTLS_PRINCIPALS)
    with serving(
        monkeypatch,
        runtime,
        policy,
        listener=mtls_listener(mtls_material),
        channel_credentials=client_credentials(mtls_material, mapped=True),
    ) as served:
        stub = served.stub()
        certificate_only = outcome(stub, "Deidentify")
        narrow_key = outcome(stub, "Deidentify", metadata=(("x-api-key", NARROW_KEY),))
        unknown_key = outcome(
            stub, "Deidentify", metadata=(("x-api-key", UNKNOWN_KEY),)
        )

    assert certificate_only == (grpc.StatusCode.OK, None)
    assert narrow_key == failure_of(REASON_INSUFFICIENT_SCOPE)
    assert unknown_key == failure_of(REASON_INVALID_CREDENTIALS)


def test_certificate_principal_matches_rest_mapping(
    mtls_material: MTLSMaterial,
) -> None:
    ca = x509.load_pem_x509_certificate(mtls_material.ca_pem)
    interceptor = GrpcSecurityInterceptor(
        GrpcServicePolicy(mtls_principals=MTLS_PRINCIPALS),
        client_ca_certificates=(ca,),
    )
    context = FakeContext(
        auth_context={"x509_pem_cert": [mtls_material.mapped_cert_pem]}
    )

    principal = interceptor.authenticate(context)
    identity = verify_client_certificate_chain(
        (mtls_material.mapped_cert_pem.decode("ascii"),), mtls_material.ca_file
    )

    assert principal == principal_for_mtls_identity(identity, MTLS_PRINCIPALS)
    assert principal is not None
    assert principal.subject == "synthetic-clinic-api"
    assert principal.credential_type == "mtls"


@pytest.mark.parametrize(
    "auth_context",
    [
        {"x509_pem_cert": [b"-----BEGIN CERTIFICATE-----\nnot-a-cert\n"]},
        {"x509_pem_cert": [b"\xff\xfe"]},
        "two_certificates",
        "foreign_certificate",
    ],
    ids=["garbage", "non_ascii", "two_certificates", "foreign_certificate"],
)
def test_untrusted_peer_certificates_are_invalid(
    mtls_material: MTLSMaterial, auth_context: Any
) -> None:
    ca = x509.load_pem_x509_certificate(mtls_material.ca_pem)
    if auth_context == "two_certificates":
        auth_context = {
            "x509_pem_cert": [
                mtls_material.mapped_cert_pem,
                mtls_material.unmapped_cert_pem,
            ]
        }
    elif auth_context == "foreign_certificate":
        now = datetime.now(timezone.utc)
        other_key, other_ca = _build_ca("Foreign Synthetic CA", now=now)
        key = ec.generate_private_key(ec.SECP256R1())
        foreign = _build_leaf(
            other_key,
            other_ca,
            key,
            common_name=MAPPED_CLIENT,
            usage=ExtendedKeyUsageOID.CLIENT_AUTH,
            not_before=now - timedelta(hours=1),
            not_after=now + timedelta(days=1),
        )
        auth_context = {"x509_pem_cert": [_cert_pem(foreign)]}
    interceptor = GrpcSecurityInterceptor(
        GrpcServicePolicy(mtls_principals=MTLS_PRINCIPALS),
        client_ca_certificates=(ca,),
    )

    with pytest.raises(GrpcPolicyError) as excinfo:
        interceptor.authenticate(FakeContext(auth_context=auth_context))

    assert excinfo.value.reason == REASON_INVALID_CREDENTIALS


def test_missing_peer_certificate_requires_authentication(
    mtls_material: MTLSMaterial,
) -> None:
    ca = x509.load_pem_x509_certificate(mtls_material.ca_pem)
    interceptor = GrpcSecurityInterceptor(
        GrpcServicePolicy(), client_ca_certificates=(ca,)
    )

    with pytest.raises(GrpcPolicyError) as excinfo:
        interceptor.authenticate(FakeContext())

    assert excinfo.value.reason == REASON_AUTHENTICATION_REQUIRED


# --- Generic handlers and value-free output ---------------------------------


def test_interceptor_guards_methods_outside_the_service() -> None:
    def ping(request: bytes, context: grpc.ServicerContext) -> bytes:
        return b"pong"

    handler = grpc.method_handlers_generic_handler(
        "synthetic.Probe", {"Ping": grpc.unary_unary_rpc_method_handler(ping)}
    )
    results: dict[str, tuple[grpc.StatusCode, Optional[str]]] = {}
    for name, policy in (
        ("strict", auth_policy()),
        ("relaxed", auth_policy(deny_by_default=False)),
    ):
        server = grpc.server(
            grpc_server.futures.ThreadPoolExecutor(max_workers=2),
            handlers=[handler],
            interceptors=[GrpcSecurityInterceptor(policy)],
        )
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            with grpc.insecure_channel(
                f"127.0.0.1:{port}", options=_CHANNEL_OPTIONS
            ) as channel:
                ping_stub = channel.unary_unary("/synthetic.Probe/Ping")
                for label, metadata in (
                    ("anonymous", None),
                    ("narrow", (("x-api-key", NARROW_KEY),)),
                ):
                    try:
                        ping_stub(b"ping", metadata=metadata, timeout=RPC_TIMEOUT)
                        results[f"{name}-{label}"] = (grpc.StatusCode.OK, None)
                    except grpc.RpcError as exc:
                        results[f"{name}-{label}"] = (exc.code(), reason_of(exc))
        finally:
            server.stop(0).wait()

    assert results == {
        "strict-anonymous": failure_of(REASON_AUTHENTICATION_REQUIRED),
        "strict-narrow": (grpc.StatusCode.OK, None),
        "relaxed-anonymous": (grpc.StatusCode.OK, None),
        "relaxed-narrow": (grpc.StatusCode.OK, None),
    }


def test_refusals_never_reflect_credentials_text_or_certificates(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    spy: ModelSpy,
    mtls_material: MTLSMaterial,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    token = make_jwt(jwt_claims("pii:*"), secret=b"y" * 32)
    observed: list[str] = []
    policy = GrpcServicePolicy(auth=auth_config(), mtls_principals=MTLS_PRINCIPALS)
    with serving(
        monkeypatch,
        runtime,
        policy,
        listener=mtls_listener(mtls_material),
        channel_credentials=client_credentials(mtls_material, mapped=False),
    ) as served:
        stub = served.stub()
        for metadata in (
            None,
            (("x-api-key", UNKNOWN_KEY),),
            (("x-api-key", NARROW_KEY),),
            (("authorization", f"Bearer {token}"),),
            (("x-api-key", FULL_KEY), ("x-api-key", UNKNOWN_KEY)),
        ):
            for rpc in RPCS:
                try:
                    call_rpc(stub, rpc, metadata=metadata)
                except grpc.RpcError as exc:
                    observed.append(exc.details() or "")
                    observed.append(repr(exc.trailing_metadata()))
                    observed.append(str(exc.code()))

    assert observed
    assert spy.calls == []
    corpus = "\n".join([*observed, caplog.text])
    for secret in (
        FULL_KEY,
        NARROW_KEY,
        UNKNOWN_KEY,
        token,
        SYNTHETIC_TEXT,
        "SYNTH-MRN-000417",
        "BEGIN CERTIFICATE",
        UNMAPPED_CLIENT,
        str(mtls_material.ca_file.parent),
    ):
        assert secret not in corpus
    for details in observed[::3]:
        assert details in {failure.message for failure in GRPC_FAILURES.values()}


def test_failure_table_is_canonical_and_value_free() -> None:
    canonical = {
        grpc.StatusCode.UNAUTHENTICATED,
        grpc.StatusCode.PERMISSION_DENIED,
        grpc.StatusCode.INVALID_ARGUMENT,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.INTERNAL,
    }
    for reason, failure in GRPC_FAILURES.items():
        assert failure.reason == reason
        assert failure.status in canonical
        assert failure.category == failure.status.name.lower()
        assert reason.isascii() and reason.replace("_", "").isalpha()

    assert GrpcPolicyError("not a reason: secret").reason == "internal_error"


@pytest.mark.parametrize(
    ("peer", "host"),
    [
        ("ipv4:127.0.0.1:50000", "127.0.0.1"),
        ("ipv6:[::1]:50000", "::1"),
        ("ipv6:%5B::1%5D:50000", "::1"),
        ("unix:/tmp/synthetic.sock", "local"),
        ("unix-abstract:synthetic", "local"),
        ("unknown", None),
        ("dns:synthetic:1", None),
        (None, None),
    ],
)
def test_peer_host_parsing(peer: Optional[str], host: Optional[str]) -> None:
    assert grpc_peer_host(peer) == host


def test_policy_from_env_reads_rest_settings(
    monkeypatch: pytest.MonkeyPatch, runtime: service_runtime.ServiceRuntime
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps([{"id": "env", "key_hash": hash_api_key(FULL_KEY)}]),
    )
    monkeypatch.setenv(
        "OPENMED_SERVICE_MTLS_PRINCIPALS",
        json.dumps([{"identity": f"dns:{MAPPED_CLIENT}", "principal": "env-client"}]),
    )
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_RPS", "3")
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_BURST", "4")

    policy = GrpcServicePolicy.from_env()
    with_runtime = GrpcServicePolicy.from_env(
        runtime=service_runtime.ServiceRuntime.from_env()
    )

    assert policy.auth.enabled is True
    assert [key.key_id for key in policy.auth.api_keys] == ["env"]
    assert policy.mtls_principals[0].principal == "env-client"
    assert policy.throttle == ServiceThrottleConfig(
        rate_limit_rps=3.0, rate_limit_burst=4, key_by=policy.throttle.key_by
    )
    assert with_runtime.throttle.rate_limit_burst == 4
    assert isinstance(policy.limits, GrpcLimits)
