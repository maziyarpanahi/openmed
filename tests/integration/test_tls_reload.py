"""Live TLS/mTLS credential rotation over loopback gRPC and in-process REST.

Every certificate and key is ephemeral and synthetic. Servers bind only to
127.0.0.1 with an OS-assigned port, and assertions record only generation IDs,
states and reason codes.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import grpc
import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

from openmed.service import grpc_server
from openmed.service import runtime as service_runtime
from openmed.service.app import _error_response, create_app
from openmed.service.grpc_security import (
    GrpcCredentialReloader,
    bind_grpc_listener,
    resolve_grpc_listener,
)
from openmed.service.mtls import (
    SERVICE_MTLS_CA_BUNDLE_ENV_VAR,
    SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR,
    SERVICE_MTLS_ENABLED_ENV_VAR,
    SERVICE_MTLS_PRINCIPALS_ENV_VAR,
    SERVICE_MTLS_RELOAD_INTERVAL_ENV_VAR,
    SERVICE_MTLS_TRUST_OVERLAP_ENV_VAR,
    SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR,
    ServiceMTLS,
    ServiceMTLSConfig,
)
from openmed.service.proto.generated import openmed_pb2, openmed_pb2_grpc
from openmed.service.tls_reload import (
    RELOAD_OUTCOME_ACTIVATED,
    RELOAD_OUTCOME_REJECTED,
    TLS_REASON_CLIENT_TRUST_UNENFORCEABLE,
    TLS_REASON_KEY_MISMATCH,
    TLS_REASON_MALFORMED,
    TLSCredentialProvider,
    TLSCredentialSource,
)
from tests.fixtures.tls_reload import (
    SERVER_NAME,
    CredentialFiles,
    FakeClock,
    SyntheticGeneration,
    assert_value_free,
    build_generation,
    credential_files,
    install,
    write_atomic,
)

pytestmark = pytest.mark.integration

_CHANNEL_OPTIONS = (
    ("grpc.ssl_target_name_override", SERVER_NAME),
    ("grpc.enable_http_proxy", 0),
)
_RPC_TIMEOUT_SECONDS = 10
_STREAM_METHOD = "/openmed.test.TLSReload/Hold"
_UNARY_METHOD = "/openmed.test.TLSReload/Ping"
_REST_ENV_VARS = (
    SERVICE_MTLS_ENABLED_ENV_VAR,
    SERVICE_MTLS_CA_BUNDLE_ENV_VAR,
    SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR,
    SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR,
    SERVICE_MTLS_PRINCIPALS_ENV_VAR,
    SERVICE_MTLS_RELOAD_INTERVAL_ENV_VAR,
    SERVICE_MTLS_TRUST_OVERLAP_ENV_VAR,
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT",
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_KEEP_ALIVE",
    "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
)


class _UnusedLoader:
    """Loader double that proves transport tests never load a model."""

    def __init__(self, config: Any) -> None:
        raise AssertionError("model loading is not expected in transport tests")


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> service_runtime.ServiceRuntime:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for env_var in _REST_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", _UnusedLoader)
    return service_runtime.ServiceRuntime.from_env()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def generations(clock: FakeClock) -> dict[str, SyntheticGeneration]:
    return {label: build_generation(label, now=clock.now) for label in ("a", "b")}


@pytest.fixture
def files(
    tmp_path: Path, generations: dict[str, SyntheticGeneration]
) -> CredentialFiles:
    files = credential_files(tmp_path / "credentials")
    install(files, generations["a"])
    return files


class _HoldService:
    """Synthetic streaming and unary handlers registered beside OpenMed."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()

    def handler(self) -> grpc.GenericRpcHandler:
        def hold(request: bytes, context: grpc.ServicerContext) -> Iterator[bytes]:
            yield b"first"
            self.started.set()
            if not self.release.wait(_RPC_TIMEOUT_SECONDS):
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "not released")
            yield b"second"
            yield b"done"

        def ping(request: bytes, context: grpc.ServicerContext) -> bytes:
            return b"pong"

        return grpc.method_handlers_generic_handler(
            "openmed.test.TLSReload",
            {
                "Hold": grpc.unary_stream_rpc_method_handler(hold),
                "Ping": grpc.unary_unary_rpc_method_handler(ping),
            },
        )


@contextmanager
def _serving(
    reloader: GrpcCredentialReloader,
    runtime: service_runtime.ServiceRuntime,
    hold: Optional[_HoldService] = None,
) -> Iterator[int]:
    server = grpc_server.create_grpc_server(
        runtime=runtime, max_workers=8, options=reloader.server_options()
    )
    if hold is not None:
        server.add_generic_rpc_handlers((hold.handler(),))
    listener = resolve_grpc_listener("127.0.0.1:0", tls=reloader, profile="prod")
    port = bind_grpc_listener(server, listener)
    server.start()
    try:
        yield port
    finally:
        if hold is not None:
            hold.release.set()
        server.stop(0).wait()


def _reloader(
    files: CredentialFiles, clock: FakeClock, **kwargs: Any
) -> GrpcCredentialReloader:
    kwargs.setdefault("max_connection_age_seconds", None)
    return GrpcCredentialReloader(
        TLSCredentialSource(
            certificate_chain_file=files.certificate_chain_file,
            private_key_file=files.private_key_file,
            client_ca_file=files.client_ca_file,
        ),
        clock=clock,
        **kwargs,
    )


def _channel(
    port: int,
    *,
    trust: SyntheticGeneration,
    client: Optional[SyntheticGeneration],
) -> grpc.Channel:
    credentials = grpc.ssl_channel_credentials(
        root_certificates=trust.root.cert_pem,
        private_key=client.client.key_pem if client is not None else None,
        certificate_chain=client.client.cert_pem if client is not None else None,
    )
    return grpc.secure_channel(
        f"127.0.0.1:{port}", credentials, options=_CHANNEL_OPTIONS
    )


def _connects(channel: grpc.Channel) -> bool:
    try:
        reply = channel.unary_unary(_UNARY_METHOD)(b"", timeout=_RPC_TIMEOUT_SECONDS)
    except grpc.RpcError as exc:
        assert exc.code() == grpc.StatusCode.UNAVAILABLE
        return False
    assert reply == b"pong"
    return True


def _fresh_connects(
    port: int,
    *,
    trust: SyntheticGeneration,
    client: Optional[SyntheticGeneration],
) -> bool:
    with _channel(port, trust=trust, client=client) as channel:
        return _connects(channel)


def test_grpc_rotation_keeps_inflight_stream_and_switches_new_connections(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
    runtime: service_runtime.ServiceRuntime,
) -> None:
    a, b = generations["a"], generations["b"]
    reloader = _reloader(files, clock, trust_overlap_seconds=120)
    hold = _HoldService()
    with _serving(reloader, runtime, hold) as port:
        old_channel = _channel(port, trust=a, client=a)
        stream = old_channel.unary_stream(_STREAM_METHOD)(
            b"", timeout=_RPC_TIMEOUT_SECONDS * 3
        )
        assert next(stream) == b"first"
        assert hold.started.wait(_RPC_TIMEOUT_SECONDS)

        install(files, b)
        result = reloader.reload()
        assert result.outcome == RELOAD_OUTCOME_ACTIVATED

        # New connections are served the new certificate chain only.
        assert _fresh_connects(port, trust=b, client=b)
        assert not _fresh_connects(port, trust=a, client=a)
        # During the bounded overlap, clients issued by the old CA still connect.
        assert _fresh_connects(port, trust=b, client=a)
        clock.advance(120)
        assert not _fresh_connects(port, trust=b, client=a)
        assert _fresh_connects(port, trust=b, client=b)

        # The established connection keeps its accepted generation.
        assert _connects(old_channel)
        hold.release.set()
        assert list(stream) == [b"second", b"done"]
        old_channel.close()

    report = reloader.report()
    assert report["installed_generation_id"] == result.generation_id
    assert_value_free(json.dumps(report), files, a, b)


def test_grpc_invalid_candidate_keeps_previous_generation_without_plaintext(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
    runtime: service_runtime.ServiceRuntime,
) -> None:
    a, b = generations["a"], generations["b"]
    reloader = _reloader(files, clock)
    with _serving(reloader, runtime, _HoldService()) as port:
        write_atomic(files.certificate_chain_file, b.server.cert_pem)
        result = reloader.reload()

        assert result.outcome == RELOAD_OUTCOME_REJECTED
        assert result.reason == TLS_REASON_KEY_MISMATCH
        assert _fresh_connects(port, trust=a, client=a)
        assert not _fresh_connects(port, trust=b, client=b)
        assert not _fresh_connects(port, trust=a, client=None)
        with grpc.insecure_channel(
            f"127.0.0.1:{port}", options=_CHANNEL_OPTIONS
        ) as plaintext:
            assert not _connects(plaintext)
        assert reloader.readiness().serving is True


def test_grpc_connection_age_retires_old_generation_connections(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
    runtime: service_runtime.ServiceRuntime,
) -> None:
    a, b = generations["a"], generations["b"]
    reloader = _reloader(
        files,
        clock,
        trust_overlap_seconds=0,
        max_connection_age_seconds=0.5,
        connection_grace_seconds=0.5,
    )
    with _serving(reloader, runtime, _HoldService()) as port:
        old_channel = _channel(port, trust=a, client=a)
        assert _connects(old_channel)
        install(files, b)
        assert reloader.reload().outcome == RELOAD_OUTCOME_ACTIVATED

        # The old connection is closed after its maximum age; reconnecting
        # needs the new generation, which this client does not trust.
        waiter = threading.Event()
        retired = False
        for _ in range(100):
            if not _connects(old_channel):
                retired = True
                break
            waiter.wait(0.1)
        old_channel.close()

        assert retired
        assert _fresh_connects(port, trust=b, client=b)


def test_grpc_serve_accepts_a_reloader_for_openmed_rpcs(
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
    runtime: service_runtime.ServiceRuntime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a, b = generations["a"], generations["b"]
    ports: list[int] = []
    real_bind = grpc_server.bind_grpc_listener

    def capture(server: grpc.Server, listener: Any) -> int:
        port = real_bind(server, listener)
        ports.append(port)
        return port

    monkeypatch.setattr(grpc_server, "bind_grpc_listener", capture)
    reloader = _reloader(files, clock, trust_overlap_seconds=0)
    server = grpc_server.serve(
        "127.0.0.1:0", runtime=runtime, max_workers=2, tls=reloader
    )
    try:
        with _channel(ports[0], trust=a, client=a) as channel:
            assert _reaches_openmed(channel)
        install(files, b)
        assert reloader.reload().outcome == RELOAD_OUTCOME_ACTIVATED
        with _channel(ports[0], trust=b, client=b) as channel:
            assert _reaches_openmed(channel)
        with _channel(ports[0], trust=b, client=a) as channel:
            assert not _reaches_openmed(channel)
    finally:
        server.stop(0).wait()


def _reaches_openmed(channel: grpc.Channel) -> bool:
    stub = openmed_pb2_grpc.OpenMedServiceStub(channel)
    try:
        stub.Analyze(
            openmed_pb2.AnalyzeRequest(text="synthetic", model_name="unknown/model"),
            timeout=_RPC_TIMEOUT_SECONDS,
        )
    except grpc.RpcError as exc:
        if exc.code() == grpc.StatusCode.INVALID_ARGUMENT:
            return True
        assert exc.code() == grpc.StatusCode.UNAVAILABLE
        return False
    raise AssertionError("undeclared model must not be served")


class _ClientCertificateScope:
    """Attach the synthetic client chain named by a test-only header."""

    def __init__(self, app: ASGIApp, chains: dict[str, str]) -> None:
        self.app = app
        self.chains = chains

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or ())
            chain = self.chains.get(headers.get(b"x-test-client", b"").decode())
            scope = dict(scope)
            extensions = dict(scope.get("extensions") or {})
            extensions["tls"] = {
                "client_cert_chain": (chain,) if chain else (),
                "client_cert_error": None,
            }
            scope["extensions"] = extensions
        await self.app(scope, receive, send)


def _rest_app(
    monkeypatch: pytest.MonkeyPatch,
    files: CredentialFiles,
    generations: dict[str, SyntheticGeneration],
    *,
    clock: Optional[FakeClock] = None,
    overlap_seconds: float = 0,
) -> tuple[Any, ASGIApp, threading.Event]:
    from openmed.service.auth import ROUTE_POLICIES, RoutePolicy

    monkeypatch.setitem(
        ROUTE_POLICIES,
        ("GET", "/v1/test/hold"),
        RoutePolicy(("identity:read",), "control"),
    )
    monkeypatch.setenv(SERVICE_MTLS_ENABLED_ENV_VAR, "true")
    monkeypatch.setenv(SERVICE_MTLS_CA_BUNDLE_ENV_VAR, str(files.client_ca_file))
    monkeypatch.setenv(SERVICE_MTLS_TRUST_OVERLAP_ENV_VAR, str(overlap_seconds))
    app = create_app()
    if clock is not None:
        app.state.mtls = ServiceMTLS(
            app.state.mtls.config,
            error_response=_error_response,
            credentials=TLSCredentialProvider(
                TLSCredentialSource(client_ca_file=files.client_ca_file),
                trust_overlap_seconds=overlap_seconds,
                clock=clock,
            ),
        )
    release = threading.Event()
    app.state.test_hold_entered = threading.Event()

    @app.get("/v1/test/hold")
    async def hold(request: Request) -> dict[str, Any]:
        if request.query_params.get("wait") == "1":
            request.app.state.test_hold_entered.set()
            await asyncio.to_thread(release.wait, _RPC_TIMEOUT_SECONDS)
        return {"generation_id": request.state.mtls_generation_id}

    chains = {
        label: generation.client.cert_pem.decode("ascii")
        for label, generation in generations.items()
    }
    return app, _ClientCertificateScope(app, chains), release


def test_rest_rotation_lets_inflight_request_finish_on_its_generation(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    files: CredentialFiles,
    generations: dict[str, SyntheticGeneration],
) -> None:
    app, asgi, release = _rest_app(monkeypatch, files, generations)
    provider = app.state.mtls.credentials
    old_id = provider.active.generation_id

    async def scenario() -> tuple[httpx.Response, httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=asgi)
        async with httpx.AsyncClient(
            transport=transport, base_url="https://127.0.0.1"
        ) as client:
            inflight = asyncio.create_task(
                client.get("/v1/test/hold?wait=1", headers={"x-test-client": "a"})
            )
            for _ in range(500):
                if app.state.test_hold_entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert app.state.test_hold_entered.is_set()
            install(files, generations["b"])
            assert provider.reload().outcome == RELOAD_OUTCOME_ACTIVATED
            old_client = await client.get(
                "/v1/test/hold", headers={"x-test-client": "a"}
            )
            new_client = await client.get(
                "/v1/test/hold", headers={"x-test-client": "b"}
            )
            release.set()
            return await inflight, old_client, new_client

    inflight, old_client, new_client = asyncio.run(scenario())

    assert inflight.status_code == 200
    assert inflight.json() == {"generation_id": old_id}
    assert old_client.status_code == 401
    assert old_client.json()["error"]["code"] == "mtls_certificate_invalid"
    assert new_client.status_code == 200
    assert new_client.json() == {"generation_id": provider.active.generation_id}
    assert_value_free(old_client.text, files, *generations.values())


def test_rest_trust_overlap_accepts_old_and_new_clients_until_deadline(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    app, asgi, _ = _rest_app(
        monkeypatch, files, generations, clock=clock, overlap_seconds=60
    )
    provider = app.state.mtls.credentials
    old_id = provider.active.generation_id
    with TestClient(asgi, base_url="https://127.0.0.1") as client:
        install(files, generations["b"])
        assert provider.reload().outcome == RELOAD_OUTCOME_ACTIVATED
        new_id = provider.active.generation_id

        old_during = client.get("/v1/test/hold", headers={"x-test-client": "a"})
        new_during = client.get("/v1/test/hold", headers={"x-test-client": "b"})
        clock.advance(60)
        old_after = client.get("/v1/test/hold", headers={"x-test-client": "a"})
        new_after = client.get("/v1/test/hold", headers={"x-test-client": "b"})

    assert old_during.json() == {"generation_id": old_id}
    assert new_during.json() == {"generation_id": new_id}
    assert old_after.status_code == 401
    assert new_after.json() == {"generation_id": new_id}


def test_rest_invalid_bundle_keeps_previous_trust_and_never_disables_mtls(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    files: CredentialFiles,
    generations: dict[str, SyntheticGeneration],
) -> None:
    app, asgi, _ = _rest_app(monkeypatch, files, generations)
    provider = app.state.mtls.credentials
    with TestClient(asgi, base_url="https://127.0.0.1") as client:
        write_atomic(files.client_ca_file, b"-----BEGIN CERTIFICATE-----\ntrunc")
        result = provider.reload()
        trusted = client.get("/v1/test/hold", headers={"x-test-client": "a"})
        untrusted = client.get("/v1/test/hold", headers={"x-test-client": "b"})
        anonymous = client.get("/v1/test/hold")

    assert result.reason == TLS_REASON_MALFORMED
    assert trusted.status_code == 200
    assert untrusted.status_code == 401
    assert anonymous.status_code == 401
    assert anonymous.json()["error"]["code"] == "mtls_certificate_required"


def test_rest_expired_trust_reports_not_ready_instead_of_serving(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    files: CredentialFiles,
    clock: FakeClock,
    generations: dict[str, SyntheticGeneration],
) -> None:
    app, asgi, _ = _rest_app(monkeypatch, files, generations, clock=clock)
    with TestClient(asgi, base_url="https://127.0.0.1") as client:
        ready = client.get("/readyz", headers={"x-test-client": "a"})
        clock.advance(31 * 86_400)
        expired = client.get("/readyz", headers={"x-test-client": "a"})
        work = client.get("/v1/test/hold", headers={"x-test-client": "a"})

    assert ready.status_code == 200
    assert expired.status_code == 503
    assert expired.json()["error"]["code"] == "not_ready"
    assert expired.json()["error"]["details"] == {
        "reason": TLS_REASON_CLIENT_TRUST_UNENFORCEABLE
    }
    assert work.status_code == 503


def test_rest_lifespan_runs_the_reload_watcher(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    files: CredentialFiles,
    generations: dict[str, SyntheticGeneration],
) -> None:
    monkeypatch.setenv(SERVICE_MTLS_RELOAD_INTERVAL_ENV_VAR, "1")
    app, asgi, _ = _rest_app(monkeypatch, files, generations)
    mtls = app.state.mtls

    with TestClient(asgi, base_url="https://127.0.0.1") as client:
        watcher = mtls._watcher
        assert watcher is not None and watcher.running
        install(files, generations["b"])
        waiter = threading.Event()
        for _ in range(500):
            if mtls.credentials.active.sequence == 2:
                break
            waiter.wait(0.01)
        rotated = client.get("/v1/test/hold", headers={"x-test-client": "b"})

    assert rotated.status_code == 200
    assert not watcher.running


def test_rest_forwarded_chain_uses_the_active_generation(
    monkeypatch: pytest.MonkeyPatch,
    runtime: service_runtime.ServiceRuntime,
    files: CredentialFiles,
    generations: dict[str, SyntheticGeneration],
) -> None:
    monkeypatch.setenv(SERVICE_MTLS_CLIENT_CERT_HEADER_ENV_VAR, "X-OpenMed-Client-Cert")
    monkeypatch.setenv(SERVICE_MTLS_TRUSTED_PROXIES_ENV_VAR, "127.0.0.1/32")
    app, _, _ = _rest_app(monkeypatch, files, generations)
    assert isinstance(app.state.mtls.config, ServiceMTLSConfig)
    provider = app.state.mtls.credentials

    async def from_trusted_proxy(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            scope = dict(scope)
            scope["client"] = ("127.0.0.1", 50000)
        await app(scope, receive, send)

    def forwarded(label: str) -> dict[str, str]:
        pem = generations[label].client.cert_pem.decode("ascii")
        return {"X-OpenMed-Client-Cert": quote(pem)}

    with TestClient(from_trusted_proxy, base_url="https://127.0.0.1") as client:
        before = client.get("/v1/test/hold", headers=forwarded("a"))
        install(files, generations["b"])
        provider.reload()
        rotated = client.get("/v1/test/hold", headers=forwarded("b"))
        retired = client.get("/v1/test/hold", headers=forwarded("a"))

    assert before.status_code == 200
    assert rotated.json() == {"generation_id": provider.active.generation_id}
    assert retired.status_code == 401
