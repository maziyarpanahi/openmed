"""Adapter conformance tests for the service egress policy (#3581).

Each outbound adapter is exercised twice: denied operations must fail before
any transport call or body serialization, and allowed operations must reach a
synthetic loopback service within their declared purpose. Every value is
synthetic and every server binds to the loopback interface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.request import Request as URLRequest

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from openmed.core.offline import OFFLINE_ENV_VAR
from openmed.service.backends.remote_inference import (
    KServeV2HttpTransport,
    RemoteInferenceSettings,
    TritonGrpcTransport,
)
from openmed.service.egress import (
    EGRESS_MODE_ENV_VAR,
    EGRESS_RULES_ENV_VAR,
    EgressAdapter,
    EgressDenied,
    EgressOperation,
    EgressPolicy,
    EgressPurpose,
    egress_urlopen,
)
from openmed.service.openhim_mediator import (
    OpenHIMMediatorClient,
    OpenHIMMediatorSettings,
    default_mediator_registration,
)
from openmed.service.privacy_gateway import (
    HttpExternalLLMTransport,
    PrivacyTransportError,
)
from openmed.service.security import MetadataDiscoveryError, _fetch_metadata
from openmed.service.smart_backend import (
    SMARTBackendBulkIngestor,
    SMARTBackendConfig,
    SMARTBackendJobManager,
    authorize_smart_backend_config,
)
from openmed.service.tracing import ServiceTraceConfig, guarded_otlp_exporter
from openmed.service.webhooks import (
    WEBHOOK_ERROR_EGRESS_DENIED,
    authorize_webhook_destination,
    deliver_webhook,
)

SENTINEL_SECRET = "synthetic-webhook-secret-7f3a"
SENTINEL_PROMPT = "Patient [NAME_1] reports synthetic symptoms"


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: list[tuple[int, dict[str, str], bytes]] = []
        self.default = (200, {"Content-Type": "application/json"}, b"{}")
        self.lock = threading.Lock()

    def next_response(self) -> tuple[int, dict[str, str], bytes]:
        with self.lock:
            return self.responses.pop(0) if self.responses else self.default


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _handle(self) -> None:
        recorder: _Recorder = self.server.recorder  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        recorder.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "host": self.headers.get("Host"),
                "body": body,
            }
        )
        status, headers, payload = recorder.next_response()
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = _handle
    do_POST = _handle

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return None


class _Server(ThreadingHTTPServer):
    daemon_threads = True


@pytest.fixture
def loopback() -> Iterator[tuple[str, _Recorder]]:
    recorder = _Recorder()
    server = _Server(("127.0.0.1", 0), _Handler)
    server.recorder = recorder  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", recorder
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _clean_egress_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (EGRESS_MODE_ENV_VAR, EGRESS_RULES_ENV_VAR, OFFLINE_ENV_VAR):
        monkeypatch.delenv(name, raising=False)


def _allow(purpose: str, destination: str, **extra: Any) -> EgressPolicy:
    return EgressPolicy.from_rules(
        [{"purposes": [purpose], "destination": destination, **extra}]
    )


def _deny_all() -> EgressPolicy:
    return EgressPolicy(mode="deny")


def _counting_transport() -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    return httpx.MockTransport(handler), seen


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------


def test_webhook_delivery_is_denied_before_serialization_by_default() -> None:
    class Unserializable(dict):
        def __iter__(self):  # pragma: no cover - must never be serialized
            raise AssertionError("payload serialized before the egress decision")

    transport, seen = _counting_transport()
    with httpx.Client(transport=transport) as client:
        result = deliver_webhook(
            "https://callbacks.example.org/openmed",
            Unserializable(event="job.terminal"),
            secret=SENTINEL_SECRET,
            client=client,
        )
    assert result.success is False
    assert result.attempts == 0
    assert result.error == WEBHOOK_ERROR_EGRESS_DENIED
    assert seen == []


def test_allowed_loopback_webhook_is_delivered_through_the_pinned_client(
    loopback: tuple[str, _Recorder],
) -> None:
    origin, recorder = loopback
    recorder.default = (204, {}, b"")
    result = deliver_webhook(
        f"{origin}/hooks/job",
        {"event": "job.terminal", "job_id": "synthetic-job"},
        secret=SENTINEL_SECRET,
        egress_policy=_allow("webhook", origin),
    )
    assert result.success is True
    assert [item["path"] for item in recorder.requests] == ["/hooks/job"]


def test_pinned_webhook_keeps_the_host_header_while_dialing_the_checked_ip(
    loopback: tuple[str, _Recorder],
) -> None:
    origin, recorder = loopback
    port = origin.rsplit(":", 1)[1]
    hostname_origin = f"http://callbacks.synthetic.test:{port}"
    resolved: list[tuple[str, int]] = []

    def resolver(host: str, resolved_port: int) -> list[str]:
        resolved.append((host, resolved_port))
        return ["127.0.0.1"]

    policy = EgressPolicy.from_rules(
        [
            {
                "purposes": ["webhook"],
                "destination": hostname_origin,
                "cidrs": ["127.0.0.0/8"],
            }
        ],
        resolver=resolver,
    )
    recorder.default = (204, {}, b"")
    result = deliver_webhook(
        f"{hostname_origin}/hook",
        {"event": "job.terminal"},
        secret=SENTINEL_SECRET,
        egress_policy=policy,
    )
    assert result.success is True
    assert recorder.requests[0]["host"] == f"callbacks.synthetic.test:{port}"
    assert resolved == [("callbacks.synthetic.test", int(port))]


def test_webhook_retries_stop_when_dns_rebinds_to_a_prohibited_address(
    loopback: tuple[str, _Recorder],
) -> None:
    origin, recorder = loopback
    port = origin.rsplit(":", 1)[1]
    hostname_origin = f"http://callbacks.synthetic.test:{port}"
    answers = iter([["127.0.0.1"], ["169.254.169.254"]])
    policy = EgressPolicy.from_rules(
        [
            {
                "purposes": ["webhook"],
                "destination": hostname_origin,
                "cidrs": ["127.0.0.0/8"],
            }
        ],
        resolver=lambda host, resolved_port: next(answers),
    )
    recorder.default = (
        503,
        {"Connection": "close"},
        b"",
    )
    result = deliver_webhook(
        f"{hostname_origin}/hook",
        {"event": "job.terminal"},
        secret=SENTINEL_SECRET,
        max_attempts=3,
        backoff_seconds=0,
        egress_policy=policy,
    )
    assert result.success is False
    assert result.error == WEBHOOK_ERROR_EGRESS_DENIED
    assert len(recorder.requests) == 1


def test_webhook_retries_honor_offline_mode_switched_on_mid_operation(
    loopback: tuple[str, _Recorder],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin, recorder = loopback
    recorder.default = (503, {}, b"")

    def sleeper(_: float) -> None:
        monkeypatch.setenv(OFFLINE_ENV_VAR, "1")

    result = deliver_webhook(
        f"{origin}/hook",
        {"event": "job.terminal"},
        secret=SENTINEL_SECRET,
        max_attempts=3,
        backoff_seconds=0.01,
        sleeper=sleeper,
        egress_policy=_allow("webhook", origin),
    )
    assert result.error == WEBHOOK_ERROR_EGRESS_DENIED
    assert result.attempts == 1
    assert len(recorder.requests) == 1


def test_webhook_redirects_are_not_followed(loopback: tuple[str, _Recorder]) -> None:
    origin, recorder = loopback
    recorder.default = (302, {"Location": "http://169.254.169.254/latest"}, b"")
    result = deliver_webhook(
        f"{origin}/hook",
        {"event": "job.terminal"},
        secret=SENTINEL_SECRET,
        max_attempts=1,
        egress_policy=_allow("webhook", origin),
    )
    assert result.success is False
    assert result.status_code == 302
    assert len(recorder.requests) == 1


def test_webhook_body_budget_is_enforced_before_sending(
    loopback: tuple[str, _Recorder],
) -> None:
    origin, recorder = loopback
    result = deliver_webhook(
        f"{origin}/hook",
        {"event": "job.terminal", "padding": "x" * 256},
        secret=SENTINEL_SECRET,
        egress_policy=_allow("webhook", origin, max_request_bytes=64),
    )
    assert result.error == WEBHOOK_ERROR_EGRESS_DENIED
    assert recorder.requests == []


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("https://169.254.169.254/latest/meta-data", "cloud_metadata_address"),
        ("https://127.0.0.1:9/hook", "loopback_address"),
        ("https://10.0.0.8/hook", "private_address"),
        ("http://callbacks.example.org/hook", "insecure_transport"),
    ],
)
def test_webhook_destinations_are_checked_at_submission(url: str, reason: str) -> None:
    with pytest.raises(EgressDenied) as enforce_error:
        authorize_webhook_destination(url, egress_policy=EgressPolicy())
    with pytest.raises(EgressDenied) as permissive_error:
        authorize_webhook_destination(
            url, egress_policy=EgressPolicy(mode="permissive")
        )
    assert enforce_error.value.reason_code == "no_matching_rule"
    assert permissive_error.value.reason_code == reason


def test_jobs_api_rejects_unapproved_webhook_urls_with_a_stable_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openmed.service import runtime as service_runtime
    from openmed.service.app import create_app

    class FakeLoader:
        def __init__(self, config: Any) -> None:
            self.config = config

        def resolve_model_name(self, model_name: str) -> str:
            return model_name

        def loaded_models(self) -> dict[str, dict[str, int]]:
            return {}

    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    monkeypatch.delenv("OPENMED_PROFILE", raising=False)
    monkeypatch.setenv(
        EGRESS_RULES_ENV_VAR,
        json.dumps(
            [
                {
                    "purposes": ["webhook"],
                    "destination": "https://callbacks.example.org",
                }
            ]
        ),
    )
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        denied = client.post(
            "/jobs",
            json={
                "documents": [{"text": "Synthetic note."}],
                "webhook": {
                    "url": "http://169.254.169.254/latest/meta-data",
                    "secret": SENTINEL_SECRET,
                },
            },
        )
        malformed = client.post(
            "/jobs",
            json={
                "documents": [{"text": "Synthetic note."}],
                "webhook": {
                    "url": "https://user:pw@callbacks.example.org/hook",
                    "secret": SENTINEL_SECRET,
                },
            },
        )
    assert denied.status_code == 403
    error = denied.json()["error"]
    assert error["code"] == "egress_denied"
    assert error["details"] == {"reason": "no_matching_rule"}
    assert "169.254" not in denied.text and SENTINEL_SECRET not in denied.text
    assert malformed.status_code == 422
    assert "user:pw" not in malformed.text


# ---------------------------------------------------------------------------
# Remote inference
# ---------------------------------------------------------------------------


class _JSONClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def post(self, url: str, *, json: Any) -> Any:
        self.calls.append((url, json))

        class Response:
            status_code = 200

            @staticmethod
            def json() -> Any:
                return {
                    "outputs": [
                        {
                            "name": "logits",
                            "shape": [1, 1, 2],
                            "datatype": "FP32",
                            "data": [0.0, 1.0],
                        }
                    ]
                }

        return Response()


def test_remote_inference_requires_an_explicit_rule() -> None:
    settings = RemoteInferenceSettings(
        endpoint="https://kserve.example.org", model_name="openmed_pii"
    )
    with pytest.raises(EgressDenied) as exc_info:
        KServeV2HttpTransport(settings, client=_JSONClient())
    assert exc_info.value.reason_code == "no_matching_rule"
    with pytest.raises(EgressDenied):
        TritonGrpcTransport(
            RemoteInferenceSettings(
                endpoint="grpcs://triton.example.org:8001",
                model_name="openmed_pii",
                protocol="grpc",
            ),
            rpc=lambda request, *, timeout: None,
        )


def test_remote_inference_tensor_budget_is_checked_before_the_request() -> None:
    client = _JSONClient()
    transport = KServeV2HttpTransport(
        RemoteInferenceSettings(
            endpoint="https://kserve.example.org", model_name="openmed_pii"
        ),
        client=client,
        egress_policy=_allow(
            "remote_inference", "https://kserve.example.org", max_request_bytes=8
        ),
    )
    small = transport.infer(
        {"input_ids": np.array([[101]], dtype=np.int64)}, output_name="logits"
    )
    assert small.shape == (1, 1, 2)
    with pytest.raises(EgressDenied) as exc_info:
        transport.infer(
            {"input_ids": np.array([[101, 102]], dtype=np.int64)},
            output_name="logits",
        )
    assert exc_info.value.reason_code == "budget_exceeded"
    assert len(client.calls) == 1


def test_remote_inference_cannot_bypass_offline_mode_after_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _JSONClient()
    transport = KServeV2HttpTransport(
        RemoteInferenceSettings(
            endpoint="https://kserve.example.org", model_name="openmed_pii"
        ),
        client=client,
        egress_policy=_allow("remote_inference", "https://kserve.example.org"),
    )
    monkeypatch.setenv(OFFLINE_ENV_VAR, "1")
    with pytest.raises(EgressDenied) as exc_info:
        transport.infer(
            {"input_ids": np.array([[101]], dtype=np.int64)}, output_name="logits"
        )
    assert exc_info.value.reason_code == "offline_mode"
    assert client.calls == []


def test_remote_inference_reaches_an_allowed_loopback_kserve_endpoint(
    loopback: tuple[str, _Recorder],
) -> None:
    origin, recorder = loopback
    recorder.default = (
        200,
        {"Content-Type": "application/json"},
        json.dumps(
            {
                "outputs": [
                    {
                        "name": "logits",
                        "shape": [1, 1, 2],
                        "datatype": "FP32",
                        "data": [0.5, 0.5],
                    }
                ]
            }
        ).encode(),
    )
    transport = KServeV2HttpTransport(
        RemoteInferenceSettings(endpoint=origin, model_name="openmed_pii"),
        egress_policy=_allow("remote_inference", origin),
    )
    try:
        result = transport.infer(
            {"input_ids": np.array([[101]], dtype=np.int64)}, output_name="logits"
        )
    finally:
        transport.close()
    assert result.tolist() == [[[0.5, 0.5]]]
    assert recorder.requests[0]["path"] == "/v2/models/openmed_pii/infer"


# ---------------------------------------------------------------------------
# SMART backend services and FHIR Bulk Data
# ---------------------------------------------------------------------------


def _smart_config(tmp_path: Any, **overrides: Any) -> SMARTBackendConfig:
    values = {
        "fhir_base_url": "https://fhir.example.org/r4",
        "token_url": "https://auth.example.org/token",
        "client_id": "synthetic-client",
        "private_key_pem": "-----BEGIN PRIVATE KEY-----\nAA==\n-----END PRIVATE KEY-----",
        "output_dir": tmp_path / "out",
    }
    values.update(overrides)
    return SMARTBackendConfig(**values)


def test_smart_job_submission_requires_rules_for_both_endpoints(
    tmp_path: Any,
) -> None:
    config = _smart_config(tmp_path)
    token_only = _allow(
        "smart_token", "https://auth.example.org", allow_credentials=True
    )
    with pytest.raises(EgressDenied) as exc_info:
        authorize_smart_backend_config(config, egress_policy=token_only)
    assert exc_info.value.reason_code == "no_matching_rule"
    without_credentials = EgressPolicy.from_rules(
        [
            {"purposes": ["smart_token"], "destination": "https://auth.example.org"},
            {"purposes": ["fhir_bulk"], "destination": "https://fhir.example.org"},
        ]
    )
    with pytest.raises(EgressDenied) as credential_error:
        authorize_smart_backend_config(config, egress_policy=without_credentials)
    assert credential_error.value.reason_code == "credentials_not_permitted"


def test_smart_job_manager_refuses_to_start_a_denied_job(tmp_path: Any) -> None:
    manager = SMARTBackendJobManager(egress_policy=EgressPolicy())
    with pytest.raises(EgressDenied):
        manager.start(_smart_config(tmp_path))
    assert manager._jobs == {}  # noqa: SLF001 - nothing was scheduled


def test_smart_ingestion_denial_happens_before_the_client_assertion(
    tmp_path: Any,
) -> None:
    transport, seen = _counting_transport()
    assertions: list[Any] = []

    def build_assertion(config: SMARTBackendConfig) -> str:
        assertions.append(config)
        return "synthetic.assertion.value"

    ingestor = SMARTBackendBulkIngestor(
        _smart_config(tmp_path),
        transport=transport,
        client_assertion_builder=build_assertion,
        egress_policy=EgressPolicy(),
    )
    with pytest.raises(EgressDenied):
        asyncio.run(ingestor.run(job_id="synthetic-job"))
    assert assertions == []
    assert seen == []


# ---------------------------------------------------------------------------
# OpenHIM mediator
# ---------------------------------------------------------------------------


def _openhim_settings(api_url: str) -> OpenHIMMediatorSettings:
    return OpenHIMMediatorSettings(
        enabled=True,
        api_url=api_url,
        username="openhim@example.org",
        password="synthetic-password",
        allow_insecure_http=api_url.startswith("http://"),
        registration=default_mediator_registration(),
    )


def test_openhim_registration_is_denied_without_a_credential_rule() -> None:
    transport, seen = _counting_transport()

    async def scenario(policy: EgressPolicy) -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            mediator = OpenHIMMediatorClient(
                _openhim_settings("https://openhim.example.org"),
                client=client,
                egress_policy=policy,
            )
            await mediator.register()

    with pytest.raises(EgressDenied) as default_error:
        asyncio.run(scenario(EgressPolicy()))
    with pytest.raises(EgressDenied) as credential_error:
        asyncio.run(scenario(_allow("openhim", "https://openhim.example.org")))
    assert default_error.value.reason_code == "no_matching_rule"
    assert credential_error.value.reason_code == "credentials_not_permitted"
    assert seen == []


def test_openhim_registers_against_an_allowed_loopback_core(
    loopback: tuple[str, _Recorder],
) -> None:
    origin, recorder = loopback
    recorder.responses = [(201, {}, b""), (200, {}, b"OK")]

    async def scenario() -> None:
        mediator = OpenHIMMediatorClient(
            _openhim_settings(origin),
            egress_policy=_allow("openhim", origin, allow_credentials=True),
        )
        try:
            await mediator.register()
            await mediator.heartbeat()
        finally:
            await mediator.stop()

    asyncio.run(scenario())
    assert [item["method"] for item in recorder.requests] == ["POST", "POST"]
    assert recorder.requests[0]["path"] == "/mediators"


# ---------------------------------------------------------------------------
# Privacy gateway
# ---------------------------------------------------------------------------


def test_privacy_gateway_transport_is_denied_by_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    transport = HttpExternalLLMTransport(
        endpoint="https://llm.example.org/v1/complete", egress_policy=EgressPolicy()
    )
    with pytest.raises(PrivacyTransportError) as exc_info:
        transport(SENTINEL_PROMPT, request_id="req-1", metadata={})
    assert exc_info.value.reason_code == "egress_denied"
    assert SENTINEL_PROMPT not in caplog.text
    assert "llm.example.org" not in caplog.text


def test_privacy_gateway_forwards_to_an_allowed_loopback_endpoint(
    loopback: tuple[str, _Recorder],
) -> None:
    origin, recorder = loopback
    recorder.default = (
        200,
        {"Content-Type": "application/json"},
        json.dumps({"text": "Synthetic reply"}).encode(),
    )
    transport = HttpExternalLLMTransport(
        endpoint=f"{origin}/v1/complete",
        egress_policy=_allow("privacy_gateway", origin),
    )
    reply = transport(SENTINEL_PROMPT, request_id="req-2", metadata={"policy": {}})
    assert reply == "Synthetic reply"
    body = json.loads(recorder.requests[0]["body"])
    assert body["prompt"] == SENTINEL_PROMPT


def test_privacy_gateway_budget_denial_sends_nothing(
    loopback: tuple[str, _Recorder],
) -> None:
    origin, recorder = loopback
    transport = HttpExternalLLMTransport(
        endpoint=f"{origin}/v1/complete",
        egress_policy=_allow("privacy_gateway", origin, max_request_bytes=16),
    )
    with pytest.raises(PrivacyTransportError) as exc_info:
        transport(SENTINEL_PROMPT, request_id="req-3", metadata={})
    assert exc_info.value.reason_code == "egress_denied"
    assert recorder.requests == []


# ---------------------------------------------------------------------------
# Trace export
# ---------------------------------------------------------------------------


class _FakeExporter:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.exported: list[Any] = []

    def export(self, spans: Any) -> str:
        self.exported.append(spans)
        return "exported"

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def test_trace_export_requires_a_rule_and_rechecks_every_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry.sdk.trace.export import SpanExportResult

    endpoint = "http://127.0.0.1:4318/v1/traces"
    config = ServiceTraceConfig(enabled=True, otlp_endpoint=endpoint)
    created: list[_FakeExporter] = []

    def factory(**kwargs: Any) -> _FakeExporter:
        created.append(_FakeExporter(**kwargs))
        return created[-1]

    with pytest.raises(EgressDenied):
        guarded_otlp_exporter(
            config, egress_policy=EgressPolicy(), exporter_factory=factory
        )
    assert created == []

    exporter = guarded_otlp_exporter(
        config,
        egress_policy=_allow("trace_export", "http://127.0.0.1:4318"),
        exporter_factory=factory,
    )
    assert exporter.export(["span"]) == "exported"
    monkeypatch.setenv(OFFLINE_ENV_VAR, "1")
    assert exporter.export(["span"]) is SpanExportResult.FAILURE
    assert created[0].exported == [["span"]]


def test_trace_export_headers_count_as_credentials() -> None:
    config = ServiceTraceConfig(
        enabled=True,
        otlp_endpoint="https://collector.example.org/v1/traces",
        otlp_headers={"authorization": "synthetic-collector-key"},
    )
    with pytest.raises(EgressDenied) as exc_info:
        guarded_otlp_exporter(
            config,
            egress_policy=_allow("trace_export", "https://collector.example.org"),
            exporter_factory=_FakeExporter,
        )
    assert exc_info.value.reason_code == "credentials_not_permitted"


# ---------------------------------------------------------------------------
# MCP OAuth metadata discovery (urllib connector path)
# ---------------------------------------------------------------------------


def test_metadata_discovery_is_denied_without_a_rule(
    loopback: tuple[str, _Recorder],
) -> None:
    origin, recorder = loopback
    with pytest.raises(MetadataDiscoveryError):
        _fetch_metadata(f"{origin}/.well-known/oauth-protected-resource", None)
    assert recorder.requests == []


def test_metadata_discovery_reaches_an_allowed_loopback_issuer(
    loopback: tuple[str, _Recorder],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin, recorder = loopback
    monkeypatch.setenv(
        EGRESS_RULES_ENV_VAR,
        json.dumps([{"purposes": ["connector_metadata"], "destination": origin}]),
    )
    recorder.default = (
        200,
        {"Content-Type": "application/json"},
        json.dumps({"resource": origin}).encode(),
    )
    payload = _fetch_metadata(f"{origin}/.well-known/oauth-protected-resource", None)
    assert payload == {"resource": origin}


def test_urllib_redirects_are_reauthorized_per_hop(
    loopback: tuple[str, _Recorder],
) -> None:
    origin, recorder = loopback

    def fetch(policy: EgressPolicy) -> bytes:
        operation = EgressOperation(
            EgressAdapter.MCP_METADATA, policy=policy, bound_destinations=(origin,)
        )
        request = URLRequest(f"{origin}/first", method="GET")
        with egress_urlopen(
            operation,
            request,
            purpose=EgressPurpose.CONNECTOR_METADATA,
            timeout=5,
        ) as response:
            return response.read()

    recorder.responses = [(302, {"Location": f"{origin}/second"}, b"")]
    with pytest.raises(EgressDenied) as exc_info:
        fetch(_allow("connector_metadata", origin))
    assert exc_info.value.reason_code == "redirect_not_permitted"

    recorder.requests.clear()
    recorder.responses = [
        (302, {"Location": f"{origin}/second"}, b""),
        (200, {}, b"done"),
    ]
    assert fetch(_allow("connector_metadata", origin, allow_redirects=True)) == b"done"
    assert [item["path"] for item in recorder.requests] == ["/first", "/second"]

    recorder.requests.clear()
    recorder.responses = [(302, {"Location": "http://127.0.0.2:9/elsewhere"}, b"")]
    with pytest.raises(EgressDenied) as cross_origin:
        fetch(_allow("connector_metadata", origin, allow_redirects=True))
    assert cross_origin.value.reason_code == "destination_not_bound"
    assert len(recorder.requests) == 1


# ---------------------------------------------------------------------------
# Negative control: no rule, no egress
# ---------------------------------------------------------------------------


def test_external_destinations_stay_blocked_while_loopback_is_allowed(
    loopback: tuple[str, _Recorder],
) -> None:
    origin, recorder = loopback
    policy = _allow("webhook", origin)
    recorder.default = (204, {}, b"")
    allowed = deliver_webhook(
        f"{origin}/hook",
        {"event": "job.terminal"},
        secret=SENTINEL_SECRET,
        egress_policy=policy,
    )
    blocked = [
        deliver_webhook(
            url,
            {"event": "job.terminal"},
            secret=SENTINEL_SECRET,
            egress_policy=policy,
        )
        for url in (
            "https://callbacks.example.org/hook",
            "http://192.0.2.10/hook",
            "http://[2001:db8::1]/hook",
        )
    ]
    assert allowed.success is True
    assert {item.error for item in blocked} == {WEBHOOK_ERROR_EGRESS_DENIED}
    assert all(item.attempts == 0 for item in blocked)
    assert len(recorder.requests) == 1
