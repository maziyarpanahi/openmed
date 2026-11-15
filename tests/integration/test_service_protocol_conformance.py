"""Semantic and privacy conformance of the REST, GraphQL and gRPC transports.

All three transports run in-process over one shared ``ServiceRuntime`` against
the frozen synthetic fixtures in
``tests/fixtures/service/protocol_conformance_v1.json``. Model entry points are
fakes that return the fixture's model output, so the suite proves what each
transport does with the same core result:

* successful operations normalize to the same spans, offsets, labels and
  actions, and reach the core with the same arguments;
* every failure is the outcome ``docs/api/protocol-map.json`` declares, and a
  category difference is accepted only where the map records a divergence;
* deadlines and cancellation leave no warm-pool lease, permit, task or unsafe
  stream tail behind;
* no synthetic sentinel reaches logs, traces, metrics, error bodies, gRPC
  details or trailers, or the conformance records.

Assertions compare digests, offsets, codes and counts only. Requests and
expected results are built from the protocol map's field mappings, so a renamed
field fails here as well as in the static drift check.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import grpc
import pytest
from fastapi.testclient import TestClient
from google.protobuf.json_format import MessageToDict

import openmed
from openmed.core.errors import BudgetExceededError, MissingExtraError
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.core.schemas.span import OpenMedSpan
from openmed.core.streaming import StreamingDeidentificationEvent
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.service import grpc_server
from openmed.service import runtime as service_runtime
from openmed.service import streaming as service_streaming
from openmed.service.auth import hash_api_key
from openmed.service.graphql_app import GRAPHQL_PATH
from openmed.service.grpc_policy import GrpcServicePolicy
from openmed.service.proto.generated import openmed_pb2
from openmed.service.protocol_map import (
    ObservedOutcome,
    ProtocolOperation,
    check_protocol_map,
    load_protocol_map,
)
from openmed.service.resilience import CircuitBreakerOpenError
from openmed.service.warm_pool import WarmPoolBackpressureError
from tests.unit.service.test_grpc_auth import FakeClock, Served, reason_of, serving

pytestmark = pytest.mark.integration

# ``openmed.service.app`` resolves to the FastAPI app object on the package.
service_app = importlib.import_module("openmed.service.app")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads(
    (
        REPOSITORY_ROOT
        / "tests"
        / "fixtures"
        / "service"
        / "protocol_conformance_v1.json"
    ).read_text(encoding="utf-8")
)
OPERATIONS: Mapping[str, Mapping[str, Any]] = FIXTURE["operations"]
SENTINELS: tuple[str, ...] = tuple(FIXTURE["sentinels"])
UNSERVED_MODEL: str = FIXTURE["unserved_model"]
PROTOCOL_MAP = load_protocol_map()
LOOPBACK_BASE_URL = "http://127.0.0.1"
WAIT_SECONDS = 10.0
RPC_TIMEOUT = 10.0
# The expected-failure REST stream lease check polls this long before giving
# up; once #4137 lands the lease returns as soon as the blocked window ends.
LEASE_SETTLE_SECONDS = 2.0
# Long enough for the first stream window to finish; the second window blocks
# until the deadline has passed, so the outcome never depends on speed.
REST_STREAM_DEADLINE = 1.0
STREAM_OPERATION = "stream_deidentify"
UNARY_OPERATIONS = ("analyze", "extract", "deidentify")
ALL_OPERATIONS = (*UNARY_OPERATIONS, STREAM_OPERATION)

FULL_KEY = "synthetic-conformance-full-key-31d9"
NARROW_KEY = "synthetic-conformance-narrow-key-77a0"
UNKNOWN_KEY = "synthetic-conformance-unknown-key-0000"
AUTH_ENV = {
    "OPENMED_SERVICE_AUTH_ENABLED": "true",
    "OPENMED_SERVICE_AUTH_API_KEYS": json.dumps(
        [
            {
                "key_id": "full",
                "key_hash": hash_api_key(FULL_KEY),
                "principal": "synthetic-conformance-full",
                "scopes": ["analyze:write", "pii:*"],
            },
            {
                "key_id": "narrow",
                "key_hash": hash_api_key(NARROW_KEY),
                "principal": "synthetic-conformance-narrow",
                "scopes": ["models:read"],
            },
        ]
    ),
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS": "0",
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST": "0",
}
_SERVICE_ENV = (
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_KEEP_ALIVE",
    "OPENMED_SERVICE_MAX_RESIDENT_MODELS",
    "OPENMED_SERVICE_MODEL_MEMORY_BUDGET_BYTES",
    "OPENMED_SERVICE_DEFAULT_MODEL_FOOTPRINT_BYTES",
    "OPENMED_SERVICE_MODEL_ADMISSION_WAIT_SECONDS",
    "OPENMED_SERVICE_SERVED_MODELS",
    "OPENMED_SERVICE_MAX_TEXT_LENGTH",
    "OPENMED_SERVICE_MAX_REQUEST_BYTES",
    "OPENMED_SERVICE_BATCHING_ENABLED",
    "OPENMED_SERVICE_COALESCING_ENABLED",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
    "OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS",
    "OPENMED_SERVICE_THROTTLE_KEY",
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_AUTH_API_KEYS",
    "OPENMED_SERVICE_AUTH_JWKS",
    "OPENMED_SERVICE_AUTH_JWKS_FILE",
    "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
    "OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT",
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_KEY",
    "OPENMED_SERVICE_MTLS_ENABLED",
    "OPENMED_SERVICE_MTLS_PRINCIPALS",
    "OPENMED_SERVICE_METRICS_ENABLED",
    "OPENMED_SERVICE_TRACING_ENABLED",
    "OPENMED_SERVICE_CORS_ORIGINS",
    "OPENMED_SERVICE_TRUSTED_HOSTS",
    "OPENMED_SERVICE_RETRY_MAX_ATTEMPTS",
    "OPENMED_SERVICE_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)

GRPC_REQUEST_TYPES: Mapping[str, Any] = {
    "analyze": openmed_pb2.AnalyzeRequest,
    "extract": openmed_pb2.ExtractRequest,
    "deidentify": openmed_pb2.DeidentifyRequest,
    STREAM_OPERATION: openmed_pb2.DeidentifyStreamRequest,
}

#: Outcomes this suite drives on every transport that declares them.
EXERCISED_OUTCOMES = frozenset(
    {
        "authentication_required",
        "invalid_credentials",
        "credentials_malformed",
        "insufficient_scope",
        "auth_rate_limited",
        "rate_limited",
        "text_too_long",
        "request_too_large",
        "invalid_request",
        "model_not_served",
        "deadline_exceeded",
        "cancelled",
        "not_ready",
        "model_capacity_busy",
        "budget_exceeded",
        "circuit_breaker_open",
        "capability_unavailable",
        "internal_error",
    }
)
#: Outcomes left to the transport-specific suites, with the reason.
NOT_EXERCISED_OUTCOMES: Mapping[str, str] = {
    "client_certificate_required": (
        "needs mutual-TLS listeners; tests/unit/service/test_mtls_auth.py and "
        "tests/unit/service/test_grpc_auth.py"
    ),
    "client_certificate_invalid": (
        "needs mutual-TLS listeners; tests/unit/service/test_mtls_auth.py and "
        "tests/unit/service/test_grpc_auth.py"
    ),
    "concurrency_limited": (
        "needs overlapping requests; tests/unit/service/test_throttle.py and "
        "tests/unit/service/test_grpc_limits.py"
    ),
    "batch_backpressure": (
        "REST-only dynamic batching; tests/unit/service/test_backpressure_shedding.py"
    ),
    "stream_too_many_chunks": "gRPC-only; tests/unit/service/test_grpc_limits.py",
    "stream_chunk_too_large": "gRPC limit; tests/unit/service/test_grpc_limits.py",
    "stream_buffer_too_large": "gRPC-only; tests/unit/service/test_grpc_limits.py",
}


# --- Model fakes ------------------------------------------------------------


class FakeLoader:
    """Model loader double: any real model load fails the suite."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **kwargs: Any) -> Any:
        raise AssertionError("the conformance suite must never load a model")

    def loaded_models(self) -> dict[str, Any]:
        return {}


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


class CoreSpy:
    """Fixture-backed core entry points that record value-free call summaries.

    Attributes:
        calls: ``(entry point, input digest, keyword summary)`` per call.
        failure: Factory for an exception raised when the core is entered.
        gate: When set, the core blocks here (after the first stream
            fragment for streams) until released.
        on_enter: Hook run when the core is entered, before any fragment.
        on_fragment: Hook run after the first stream fragment.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.failure: Optional[Callable[[], BaseException]] = None
        self.gate: Optional[threading.Event] = None
        self.on_enter: Optional[Callable[[], None]] = None
        self.on_fragment: Optional[Callable[[], None]] = None
        self.entered = threading.Event()
        self.unblocked = threading.Event()
        self.finished = threading.Event()
        self.stream_closed = threading.Event()
        self._lock = threading.Lock()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "CoreSpy":
        monkeypatch.setattr(openmed, "analyze_text", self.analyze_text)
        monkeypatch.setattr(openmed, "extract_pii", self.extract_pii)
        monkeypatch.setattr(openmed, "deidentify", self.deidentify)
        monkeypatch.setattr(service_streaming, "deidentify_stream", self.stream)
        monkeypatch.setattr(grpc_server, "deidentify_stream", self.stream)
        return self

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()
        self.failure = None
        self.gate = None
        self.on_enter = None
        self.on_fragment = None
        for event in (self.entered, self.unblocked, self.finished, self.stream_closed):
            event.clear()

    def snapshot(self) -> list[tuple[str, str, dict[str, Any]]]:
        with self._lock:
            return list(self.calls)

    def _enter(self, entry: str, digest: str, kwargs: Mapping[str, Any]) -> None:
        summary: dict[str, Any] = {}
        for key, value in sorted(kwargs.items()):
            if key in {"loader", "config"}:
                summary[key] = id(value)
            elif value is None or isinstance(value, (bool, int, float, str)):
                summary[key] = value
            else:
                summary[key] = type(value).__name__
        with self._lock:
            self.calls.append((entry, digest, summary))
        self.entered.set()
        if self.on_enter is not None:
            self.on_enter()
        if self.failure is not None:
            raise self.failure()

    def _block(self) -> None:
        if self.gate is not None:
            assert self.gate.wait(WAIT_SECONDS), "core gate was never released"
        self.unblocked.set()

    def analyze_text(self, text: str, **kwargs: Any) -> PredictionResult:
        self._enter("analyze_text", _digest(text), kwargs)
        try:
            self._block()
            return _prediction("analyze", text, kwargs)
        finally:
            self.finished.set()

    def extract_pii(self, text: str, **kwargs: Any) -> PredictionResult:
        self._enter("extract_pii", _digest(text), kwargs)
        try:
            self._block()
            return _prediction("extract", text, kwargs)
        finally:
            self.finished.set()

    def deidentify(self, text: str, **kwargs: Any) -> DeidentificationResult:
        self._enter("deidentify", _digest(text), kwargs)
        try:
            self._block()
            output = OPERATIONS["deidentify"]["model_output"]
            return DeidentificationResult(
                original_text=text,
                deidentified_text=output["deidentified_text"],
                pii_entities=[
                    PIIEntity(
                        text=entity["text"],
                        label=entity["label"],
                        confidence=entity["confidence"],
                        start=entity["start"],
                        end=entity["end"],
                        entity_type=entity["entity_type"],
                        redacted_text=entity["redacted_text"],
                        canonical_label=entity["canonical_label"],
                        action=entity["action"],
                    )
                    for entity in output["entities"]
                ],
                method=output["method"],
                timestamp=datetime(2026, 1, 1),
                metadata={},
            )
        finally:
            self.finished.set()

    def stream(self, chunks: Any, **kwargs: Any) -> Iterator[Any]:
        try:
            received = [str(chunk) for chunk in chunks]
            digest = _digest(json.dumps([_digest(chunk) for chunk in received]))
            self._enter("deidentify_stream", digest, kwargs)
            output = OPERATIONS[STREAM_OPERATION]["model_output"]
            fragments = list(output["fragments"])
            yield StreamingDeidentificationEvent(redacted_text=fragments[0])
            if self.on_fragment is not None:
                self.on_fragment()
            self._block()
            for fragment in fragments[1:]:
                yield StreamingDeidentificationEvent(redacted_text=fragment)
            yield StreamingDeidentificationEvent(
                redacted_text="",
                final=True,
                spans=tuple(
                    OpenMedSpan(doc_id=output["doc_id"], **span)
                    for span in output["spans"]
                ),
                audit_record=dict(output["audit_record"]),
            )
        finally:
            self.finished.set()
            self.stream_closed.set()


def _prediction(
    operation_id: str, text: str, kwargs: Mapping[str, Any]
) -> PredictionResult:
    output = OPERATIONS[operation_id]["model_output"]
    return PredictionResult(
        text=text,
        entities=[
            EntityPrediction(
                text=entity["text"],
                label=entity["label"],
                confidence=entity["confidence"],
                start=entity["start"],
                end=entity["end"],
            )
            for entity in output["entities"]
        ],
        model_name=str(kwargs.get("model_name") or "synthetic"),
        timestamp="2026-01-01T00:00:00",
        processing_time=0.0,
        metadata={},
    )


# --- Service stack ----------------------------------------------------------


@dataclass
class Stack:
    """The REST app (serving REST and GraphQL) and gRPC server on one runtime."""

    app: Any
    client: TestClient
    served: Served
    runtime: service_runtime.ServiceRuntime
    _stub: Any = field(default=None, repr=False)

    @property
    def stub(self) -> Any:
        if self._stub is None:
            self._stub = self.served.stub()
        return self._stub


@pytest.fixture(autouse=True)
def service_env(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> Iterator[None]:
    """Offline, deterministic service settings and a log sentinel sweep."""
    for name in _SERVICE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    # Retries and the breaker would turn repeated synthetic failures into a
    # different outcome; the circuit-breaker outcome is injected explicitly.
    monkeypatch.setenv("OPENMED_SERVICE_RESILIENCE_ENABLED", "false")
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)
    caplog.set_level(logging.DEBUG)
    yield
    assert_no_sentinel(caplog.text)


@pytest.fixture
def core(monkeypatch: pytest.MonkeyPatch) -> CoreSpy:
    return CoreSpy().install(monkeypatch)


@contextmanager
def service_stack(
    monkeypatch: pytest.MonkeyPatch,
    env: Optional[Mapping[str, str]] = None,
    *,
    clock: Optional[Callable[[], float]] = None,
) -> Iterator[Stack]:
    """Serve REST, GraphQL and gRPC in-process from one environment."""
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    app = service_app.create_app()
    with TestClient(
        app, base_url=LOOPBACK_BASE_URL, raise_server_exceptions=False
    ) as client:
        runtime = app.state.runtime
        policy = GrpcServicePolicy.from_env(runtime=runtime)
        with serving(monkeypatch, runtime, policy, clock=clock) as served:
            yield Stack(app=app, client=client, served=served, runtime=runtime)


def eventually(predicate: Callable[[], bool]) -> bool:
    """Wait for background cleanup; never asserts on elapsed time."""
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def active_leases(runtime: service_runtime.ServiceRuntime) -> int:
    models = runtime.loaded_models().get("models", {})
    return sum(int(entry.get("active_requests", 0)) for entry in models.values())


def grpc_permits_balanced(served: Served) -> bool:
    counters = served.interceptor.concurrency_gate.counters()
    return counters["in_flight"] == 0 and counters["acquired"] == counters["released"]


def resources_released(stack: Stack) -> bool:
    return (
        active_leases(stack.runtime) == 0
        and grpc_permits_balanced(stack.served)
        and not service_app._SERVICE_WORK_TASKS
        and getattr(stack.app.state, "inflight", 0) == 0
    )


# --- Building requests from the protocol map --------------------------------


def transports_of(operation: ProtocolOperation) -> tuple[str, ...]:
    return tuple(
        transport
        for transport in ("rest", "graphql", "grpc")
        if operation.supports(transport)
    )


def request_for(operation_id: str, **overrides: Any) -> dict[str, Any]:
    request = dict(OPERATIONS[operation_id]["request"])
    request.update(overrides)
    return request


def mapped_request(
    operation: ProtocolOperation, transport: str, request: Mapping[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for mapping in operation.request_fields:
        path = mapping.paths.get(transport)
        if path is not None and mapping.field in request:
            payload[path] = request[mapping.field]
    assert set(request) <= {mapping.field for mapping in operation.request_fields}
    return payload


def graphql_document(operation: ProtocolOperation) -> str:
    binding = operation.bindings["graphql"]
    scalars: list[str] = []
    collections: dict[str, list[str]] = {}
    for mapping in operation.response_fields:
        path = mapping.paths["graphql"]
        assert path is not None
        if "[]." in path:
            collection, leaf = path.split("[].", 1)
            collections.setdefault(collection, []).append(leaf)
        else:
            scalars.append(path)
    selection = " ".join(
        [
            *scalars,
            *(
                f"{name} {{ {' '.join(leaves)} }}"
                for name, leaves in collections.items()
            ),
        ]
    )
    return (
        f"query Conformance($input: {binding['input_type']}!) "
        f"{{ {binding['field']}({binding['argument']}: $input) {{ {selection} }} }}"
    )


def grpc_message(operation: ProtocolOperation, request: Mapping[str, Any]) -> Any:
    message = GRPC_REQUEST_TYPES[operation.id]()
    for path, value in mapped_request(operation, "grpc", request).items():
        target = message
        *parents, leaf = path.split(".")
        for parent in parents:
            target = getattr(target, parent)
        if isinstance(value, list):
            getattr(target, leaf).extend(value)
        else:
            setattr(target, leaf, value)
    return message


def message_dict(message: Any) -> dict[str, Any]:
    try:
        return MessageToDict(
            message,
            preserving_proto_field_name=True,
            always_print_fields_with_no_presence=True,
        )
    except TypeError:  # pragma: no cover - protobuf < 5.26
        return MessageToDict(
            message,
            preserving_proto_field_name=True,
            including_default_value_fields=True,
        )


def rest_call(
    stack: Stack,
    operation: ProtocolOperation,
    request: Mapping[str, Any],
    headers: Optional[Mapping[str, str]] = None,
) -> Any:
    return stack.client.post(
        operation.bindings["rest"]["path"],
        json=mapped_request(operation, "rest", request),
        headers=dict(headers or {}),
    )


def graphql_call(
    stack: Stack,
    operation: ProtocolOperation,
    request: Mapping[str, Any],
    headers: Optional[Mapping[str, str]] = None,
) -> Any:
    return stack.client.post(
        GRAPHQL_PATH,
        json={
            "query": graphql_document(operation),
            "variables": {"input": mapped_request(operation, "graphql", request)},
        },
        headers=dict(headers or {}),
    )


def grpc_call(
    stack: Stack,
    operation: ProtocolOperation,
    request: Mapping[str, Any],
    metadata: Optional[Sequence[tuple[str, str]]] = None,
    *,
    timeout: float = RPC_TIMEOUT,
) -> Any:
    binding = operation.bindings["grpc"]
    method = getattr(stack.stub, binding["method"].rsplit("/", 1)[-1])
    message = grpc_message(operation, request)
    if binding["framing"] == "server_stream":
        return [
            message_dict(item)
            for item in method(message, metadata=metadata, timeout=timeout)
        ]
    return message_dict(method(message, metadata=metadata, timeout=timeout))


# --- Normalization ----------------------------------------------------------


def normalize_value(kind: Optional[str], value: Any) -> Any:
    if value is None:
        return None
    if kind == "digest":
        return _digest(str(value))
    if kind in {"offset", "count"}:
        return int(value)
    if kind == "score":
        return round(float(value), 6)
    return str(value)


def normalize(
    operation: ProtocolOperation, transport: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Reduce one transport payload to the canonical, value-free fields."""
    result: dict[str, Any] = {}
    spans: Optional[list[dict[str, Any]]] = None
    for mapping in operation.response_fields:
        path = mapping.paths.get(transport)
        if path is None:
            continue
        canonical = mapping.field.removeprefix("final.")
        if "[]." not in path:
            result[canonical] = normalize_value(mapping.normalize, payload.get(path))
            continue
        collection, leaf = path.split("[].", 1)
        canonical_leaf = canonical.split("[].", 1)[1]
        items = list(payload.get(collection) or ())
        if spans is None:
            spans = [{} for _ in items]
        assert len(spans) == len(items)
        for span, item in zip(spans, items):
            span[canonical_leaf] = normalize_value(mapping.normalize, item.get(leaf))
    result["spans"] = spans or []
    return result


def numbers(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: numbers(item) for key, item in value.items()}
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def ndjson(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def stream_result(
    operation: ProtocolOperation, transport: str, messages: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Normalize a REST NDJSON stream or a gRPC response stream."""
    if transport == "rest":
        fragments = [
            event["redacted_text"] for event in messages if event["type"] == "chunk"
        ]
        finals = [event for event in messages if event["type"] == "final"]
        assert not [event for event in messages if event["type"] == "error"]
        audit_key = "audit"
    else:
        fragments = [
            message["redacted_text"] for message in messages if message["redacted_text"]
        ]
        finals = [message for message in messages if message.get("final")]
        audit_key = "audit_record"
    assert len(finals) == 1
    result = normalize(operation, transport, finals[0])
    result["redacted_text"] = _digest("".join(fragments))
    result["fragment_count"] = len(fragments)
    result["audit_record"] = numbers(dict(finals[0].get(audit_key) or {}))
    assert_no_sentinel(json.dumps(fragments))
    return result


# --- Observing failures -----------------------------------------------------


def assert_no_sentinel(text: str) -> None:
    leaked = [sentinel for sentinel in SENTINELS if sentinel in text]
    assert leaked == [], "a synthetic sentinel reached an observable surface"


def observe_rest(response: Any, *, stream: bool) -> ObservedOutcome:
    """Reduce a REST response to status and code.

    Unary success bodies echo the request text by contract, so only error
    bodies and stream output are swept for sentinels.
    """
    if stream and response.status_code == 200:
        assert_no_sentinel(response.text)
        errors = [event for event in ndjson(response.text) if event["type"] == "error"]
        if errors:
            return ObservedOutcome(
                "rest", 200, errors[-1]["error"]["code"], delivery="stream_event"
            )
        return ObservedOutcome("rest", 200)
    if response.status_code == 200:
        return ObservedOutcome("rest", 200)
    assert_no_sentinel(response.text)
    return ObservedOutcome(
        "rest", response.status_code, response.json()["error"]["code"]
    )


def observe_graphql(response: Any) -> ObservedOutcome:
    """Reduce a GraphQL response to status, code and envelope."""
    body = response.json()
    if "error" in body:
        assert_no_sentinel(response.text)
        return ObservedOutcome(
            "graphql", response.status_code, body["error"]["code"], envelope="rest"
        )
    errors = body.get("errors") or []
    if not errors:
        return ObservedOutcome("graphql", response.status_code, envelope="graphql")
    assert_no_sentinel(response.text)
    codes = {(error.get("extensions") or {}).get("code") for error in errors}
    assert len(codes) == 1
    return ObservedOutcome(
        "graphql", response.status_code, codes.pop(), envelope="graphql"
    )


def observe_grpc(error: grpc.RpcError) -> ObservedOutcome:
    assert_no_sentinel(str(error.details()))
    assert_no_sentinel(repr(error.trailing_metadata()))
    return ObservedOutcome("grpc", error.code().name, reason_of(error))


def observe(
    stack: Stack,
    operation: ProtocolOperation,
    transport: str,
    request: Mapping[str, Any],
    *,
    headers: Optional[Mapping[str, str]] = None,
    metadata: Optional[Sequence[tuple[str, str]]] = None,
) -> ObservedOutcome:
    if transport == "rest":
        response = rest_call(stack, operation, request, headers)
        return observe_rest(
            response, stream=operation.bindings["rest"]["framing"] == "ndjson"
        )
    if transport == "graphql":
        return observe_graphql(graphql_call(stack, operation, request, headers))
    try:
        grpc_call(stack, operation, request, metadata)
    except grpc.RpcError as error:
        return observe_grpc(error)
    return ObservedOutcome("grpc", "OK")


def declared_divergences() -> set[tuple[str, str]]:
    return {
        (divergence.transport, outcome)
        for divergence in PROTOCOL_MAP.divergences
        if divergence.kind == "category"
        for outcome in divergence.outcomes
    }


def assert_declared(
    outcome_id: str, operation: ProtocolOperation, observed: ObservedOutcome
) -> dict[str, Any]:
    """Assert an observation is a declared variant and return its record."""
    assert outcome_id in EXERCISED_OUTCOMES
    outcome = PROTOCOL_MAP.outcome(outcome_id)
    assert outcome.applies_to(operation.id)
    assert outcome.expects(observed), (outcome_id, operation.id, observed.to_dict())
    record = PROTOCOL_MAP.classify(observed)
    assert outcome_id in record["outcomes"]
    if record["category"] != outcome.category:
        assert (observed.transport, outcome_id) in declared_divergences()
    assert_value_free(record)
    return record


def assert_value_free(record: Any) -> None:
    """Conformance records hold codes, statuses, categories and counts only."""
    assert_no_sentinel(json.dumps(record, sort_keys=True))
    if isinstance(record, Mapping):
        for value in record.values():
            assert_value_free(value)
    elif isinstance(record, list):
        for value in record:
            assert_value_free(value)
    else:
        assert record is None or isinstance(record, (bool, int, str))


# --- Contract and fixture ---------------------------------------------------


def test_protocol_map_matches_every_transport() -> None:
    report = check_protocol_map()

    assert report.is_clean, [issue.to_dict() for issue in report.issues]


def test_every_declared_outcome_is_exercised_or_delegated() -> None:
    declared = {outcome.id for outcome in PROTOCOL_MAP.outcomes}

    assert EXERCISED_OUTCOMES | set(NOT_EXERCISED_OUTCOMES) == declared
    assert not EXERCISED_OUTCOMES & set(NOT_EXERCISED_OUTCOMES)


def test_frozen_fixture_is_span_consistent_and_redacted() -> None:
    for operation_id, operation in OPERATIONS.items():
        text = operation["request"]["text"]
        output = operation["model_output"]
        for entity in output.get("entities", ()):
            assert text[entity["start"] : entity["end"]] == entity["text"]
        for span, expected in zip(
            output.get("spans", ()), operation["expected"]["spans"]
        ):
            assert (span["start"], span["end"]) == (expected["start"], expected["end"])
        for entity, expected in zip(
            output.get("entities", ()), operation["expected"]["spans"]
        ):
            assert expected["text"] == _digest(text[entity["start"] : entity["end"]])
        redacted = output.get("deidentified_text") or "".join(
            output.get("fragments", ())
        )
        if operation_id in {"deidentify", STREAM_OPERATION}:
            assert operation["expected"]["redacted_text"] == _digest(redacted)
            assert any(sentinel in text for sentinel in SENTINELS)
            assert_no_sentinel(redacted)
        assert_no_sentinel(json.dumps(operation["expected"]))


# --- Success parity ---------------------------------------------------------


@pytest.mark.parametrize("operation_id", UNARY_OPERATIONS)
def test_successful_operations_normalize_identically(
    monkeypatch: pytest.MonkeyPatch, core: CoreSpy, operation_id: str
) -> None:
    operation = PROTOCOL_MAP.operation(operation_id)
    expected = OPERATIONS[operation_id]["expected"]
    normalized: dict[str, dict[str, Any]] = {}
    invocations: dict[str, list[Any]] = {}
    with service_stack(monkeypatch) as stack:
        for transport in transports_of(operation):
            core.reset()
            request = request_for(operation_id)
            if transport == "rest":
                response = rest_call(stack, operation, request)
                assert response.status_code == 200
                payload = response.json()
            elif transport == "graphql":
                response = graphql_call(stack, operation, request)
                body = response.json()
                assert response.status_code == 200 and "errors" not in body
                payload = body["data"][operation.bindings["graphql"]["field"]]
            else:
                payload = grpc_call(stack, operation, request)
            normalized[transport] = normalize(operation, transport, payload)
            invocations[transport] = core.snapshot()
            for _, _, summary in invocations[transport]:
                assert summary["loader"] == id(stack.runtime.get_loader())
                assert summary["config"] == id(stack.runtime.config)
        assert eventually(lambda: resources_released(stack))

    assert set(normalized) == set(transports_of(operation))
    for transport, value in normalized.items():
        assert value == expected, transport
    reference = json.dumps(next(iter(invocations.values())), sort_keys=True)
    for transport, calls in invocations.items():
        assert len(calls) == 1, transport
        assert json.dumps(calls, sort_keys=True) == reference, transport


def test_stream_deidentify_normalizes_identically(
    monkeypatch: pytest.MonkeyPatch, core: CoreSpy
) -> None:
    operation = PROTOCOL_MAP.operation(STREAM_OPERATION)
    expected = OPERATIONS[STREAM_OPERATION]["expected"]
    results: dict[str, dict[str, Any]] = {}
    invocations: dict[str, list[Any]] = {}
    with service_stack(monkeypatch) as stack:
        for transport in transports_of(operation):
            core.reset()
            request = request_for(STREAM_OPERATION)
            if transport == "rest":
                response = rest_call(stack, operation, request)
                assert response.status_code == 200
                messages = ndjson(response.text)
            else:
                messages = grpc_call(stack, operation, request)
            results[transport] = stream_result(operation, transport, messages)
            invocations[transport] = core.snapshot()
            assert core.stream_closed.is_set()
        assert eventually(lambda: resources_released(stack))

    assert set(results) == {"rest", "grpc"}
    for transport, result in results.items():
        assert result == {
            key: expected[key]
            for key in ("redacted_text", "fragment_count", "audit_record", "spans")
        }, transport
    assert invocations["rest"] == invocations["grpc"]
    chunk_count = expected["chunk_count"]
    text = OPERATIONS[STREAM_OPERATION]["request"]["text"]
    size = OPERATIONS[STREAM_OPERATION]["request"]["chunk_size"]
    chunks = [text[start : start + size] for start in range(0, len(text), size)]
    assert len(chunks) == chunk_count
    assert invocations["rest"][0][1] == _digest(
        json.dumps([_digest(chunk) for chunk in chunks])
    )


# --- Failure parity ---------------------------------------------------------


@dataclass(frozen=True)
class FailureCase:
    """One controlled failure driven through every declaring transport."""

    env: Mapping[str, str] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    metadata: tuple[tuple[str, str], ...] = ()
    overrides: Mapping[str, Any] = field(default_factory=dict)
    core_failure: Optional[Callable[[], BaseException]] = None
    runtime_failure: Optional[Callable[[], BaseException]] = None
    shutting_down: bool = False
    reaches_core: bool = False


FAILURE_CASES: Mapping[str, FailureCase] = {
    "authentication_required": FailureCase(env=AUTH_ENV),
    "invalid_credentials": FailureCase(
        env=AUTH_ENV,
        headers={"X-API-Key": UNKNOWN_KEY},
        metadata=(("x-api-key", UNKNOWN_KEY),),
    ),
    "credentials_malformed": FailureCase(
        env=AUTH_ENV,
        headers={"Authorization": "Bearer"},
        metadata=(("x-api-key", FULL_KEY), ("x-api-key", FULL_KEY)),
    ),
    "insufficient_scope": FailureCase(
        env=AUTH_ENV,
        headers={"X-API-Key": NARROW_KEY},
        metadata=(("x-api-key", NARROW_KEY),),
    ),
    "text_too_long": FailureCase(env={"OPENMED_SERVICE_MAX_TEXT_LENGTH": "16"}),
    "request_too_large": FailureCase(env={"OPENMED_SERVICE_MAX_REQUEST_BYTES": "48"}),
    "invalid_request": FailureCase(overrides={"confidence_threshold": 5.0}),
    "model_not_served": FailureCase(overrides={"model_name": UNSERVED_MODEL}),
    "not_ready": FailureCase(shutting_down=True),
    "internal_error": FailureCase(
        core_failure=lambda: RuntimeError(
            "synthetic core failure near " + " ".join(SENTINELS)
        ),
        reaches_core=True,
    ),
    "capability_unavailable": FailureCase(
        core_failure=lambda: MissingExtraError(
            "A synthetic optional extra is missing.", extra="synthetic"
        ),
        reaches_core=True,
    ),
    "budget_exceeded": FailureCase(
        core_failure=lambda: BudgetExceededError(
            kind="input_chars", limit=8, observed=72, checkpoint="synthetic"
        ),
        reaches_core=True,
    ),
    "model_capacity_busy": FailureCase(
        runtime_failure=lambda: WarmPoolBackpressureError(
            model_name="synthetic",
            required_bytes=2,
            budget_bytes=1,
            wait_seconds=0.0,
        )
    ),
    "circuit_breaker_open": FailureCase(
        runtime_failure=lambda: CircuitBreakerOpenError(5)
    ),
}


@pytest.mark.parametrize("outcome_id", sorted(FAILURE_CASES))
def test_failures_map_to_the_declared_outcome(
    monkeypatch: pytest.MonkeyPatch, core: CoreSpy, outcome_id: str
) -> None:
    case = FAILURE_CASES[outcome_id]
    outcome = PROTOCOL_MAP.outcome(outcome_id)
    records: list[dict[str, Any]] = []
    with service_stack(monkeypatch, case.env) as stack:
        if case.runtime_failure is not None:
            failure = case.runtime_failure

            def refuse(*args: Any, **kwargs: Any) -> Any:
                raise failure()

            monkeypatch.setattr(stack.runtime, "run_model_request", refuse)
            monkeypatch.setattr(stack.runtime, "begin_model_request", refuse)
        stack.app.state.shutting_down = case.shutting_down
        for operation_id in ALL_OPERATIONS:
            operation = PROTOCOL_MAP.operation(operation_id)
            if not outcome.applies_to(operation_id):
                continue
            for transport in transports_of(operation):
                if transport in outcome.not_applicable:
                    continue
                core.reset()
                core.failure = case.core_failure
                observed = observe(
                    stack,
                    operation,
                    transport,
                    request_for(operation_id, **case.overrides),
                    headers=case.headers,
                    metadata=case.metadata or None,
                )
                records.append(assert_declared(outcome_id, operation, observed))
                assert bool(core.snapshot()) == case.reaches_core, (
                    transport,
                    operation_id,
                )
        stack.app.state.shutting_down = False
        assert eventually(lambda: resources_released(stack))

    seen = {record["transport"] for record in records}
    expected = {
        transport
        for transport in ("rest", "graphql", "grpc")
        if transport not in outcome.not_applicable
    }
    assert seen == expected


@pytest.mark.parametrize("transport", ["rest", "graphql", "grpc"])
def test_rate_limits_map_to_the_declared_outcome(
    monkeypatch: pytest.MonkeyPatch, core: CoreSpy, transport: str
) -> None:
    operation = PROTOCOL_MAP.operation("deidentify")
    request = request_for("deidentify")
    full = ({"X-API-Key": FULL_KEY}, (("x-api-key", FULL_KEY),))
    unknown = ({"X-API-Key": UNKNOWN_KEY}, (("x-api-key", UNKNOWN_KEY),))
    env = {
        **AUTH_ENV,
        "OPENMED_SERVICE_RATE_LIMIT_RPS": "0.001",
        "OPENMED_SERVICE_RATE_LIMIT_BURST": "1",
    }
    with service_stack(monkeypatch, env) as stack:
        first = observe(
            stack, operation, transport, request, headers=full[0], metadata=full[1]
        )
        limited = observe(
            stack, operation, transport, request, headers=full[0], metadata=full[1]
        )
        assert eventually(lambda: resources_released(stack))

    env = {
        **AUTH_ENV,
        "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS": "0.001",
        "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST": "1",
    }
    with service_stack(monkeypatch, env) as stack:
        refused = observe(
            stack,
            operation,
            transport,
            request,
            headers=unknown[0],
            metadata=unknown[1],
        )
        throttled = observe(
            stack,
            operation,
            transport,
            request,
            headers=unknown[0],
            metadata=unknown[1],
        )

    success_status = "OK" if transport == "grpc" else 200
    assert first.status == success_status and first.code is None
    assert_declared("rate_limited", operation, limited)
    assert_declared("invalid_credentials", operation, refused)
    assert_declared("auth_rate_limited", operation, throttled)


# --- Deadlines --------------------------------------------------------------


def _blocked(core: CoreSpy) -> threading.Event:
    gate = threading.Event()
    core.gate = gate
    return gate


@pytest.mark.parametrize("transport", ["rest", "graphql"])
def test_http_deadline_maps_and_releases_after_work_settles(
    monkeypatch: pytest.MonkeyPatch, core: CoreSpy, transport: str
) -> None:
    operation = PROTOCOL_MAP.operation("deidentify")
    env = {
        "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY": "1",
        "OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS": "0",
    }
    with service_stack(monkeypatch, env) as stack:
        monkeypatch.setattr(stack.runtime.config, "timeout", 0.25)
        gate = _blocked(core)
        try:
            observed = observe(stack, operation, transport, request_for("deidentify"))
            # The timed-out work keeps running, and holding its lease, until
            # the blocked core returns.
            assert core.entered.wait(WAIT_SECONDS)
            assert active_leases(stack.runtime) == 1
        finally:
            gate.set()
        assert core.finished.wait(WAIT_SECONDS)
        assert eventually(lambda: resources_released(stack))
        core.reset()
        monkeypatch.setattr(stack.runtime.config, "timeout", 60)
        after = observe(stack, operation, transport, request_for("deidentify"))

    assert_declared("deadline_exceeded", operation, observed)
    assert after.code is None and after.status == 200


@pytest.mark.parametrize("operation_id", ["deidentify", STREAM_OPERATION])
def test_grpc_server_budget_maps_and_releases(
    monkeypatch: pytest.MonkeyPatch, core: CoreSpy, operation_id: str
) -> None:
    operation = PROTOCOL_MAP.operation(operation_id)
    clock = FakeClock()
    env = {"OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY": "1"}
    with service_stack(monkeypatch, env, clock=clock) as stack:
        budget = stack.served.interceptor.policy.limits.max_rpc_seconds

        def expire() -> None:
            clock.advance(budget + 1.0)

        if operation_id == STREAM_OPERATION:
            core.on_fragment = expire
        else:
            core.on_enter = expire
        observed = observe(stack, operation, "grpc", request_for(operation_id))
        assert core.finished.wait(WAIT_SECONDS)
        assert eventually(lambda: resources_released(stack))

    assert_declared("deadline_exceeded", operation, observed)


def test_grpc_client_deadline_releases_after_work_settles(
    monkeypatch: pytest.MonkeyPatch, core: CoreSpy
) -> None:
    operation = PROTOCOL_MAP.operation("deidentify")
    env = {"OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY": "1"}
    with service_stack(monkeypatch, env) as stack:
        gate = _blocked(core)
        try:
            future = stack.stub.Deidentify.future(
                grpc_message(operation, request_for("deidentify")), timeout=2.0
            )
            assert core.entered.wait(WAIT_SECONDS)
            with pytest.raises(grpc.RpcError) as excinfo:
                future.result()
            assert stack.served.interceptor.concurrency_gate.in_flight() == 1
        finally:
            gate.set()
        assert core.finished.wait(WAIT_SECONDS)
        assert eventually(lambda: resources_released(stack))
        core.reset()
        after = grpc_call(stack, operation, request_for("deidentify"))

    assert_declared("deadline_exceeded", operation, observe_grpc(excinfo.value))
    assert after["deidentified_text"]


# --- Cancellation -----------------------------------------------------------


def test_grpc_stream_cancellation_emits_no_unsafe_tail(
    monkeypatch: pytest.MonkeyPatch, core: CoreSpy
) -> None:
    operation = PROTOCOL_MAP.operation(STREAM_OPERATION)
    env = {"OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY": "1"}
    fragments = OPERATIONS[STREAM_OPERATION]["model_output"]["fragments"]
    with service_stack(monkeypatch, env) as stack:
        gate = _blocked(core)
        try:
            call = stack.stub.StreamDeidentify(
                grpc_message(operation, request_for(STREAM_OPERATION)),
                timeout=RPC_TIMEOUT,
            )
            received = [next(call).redacted_text]
            call.cancel()
            with pytest.raises(grpc.RpcError) as excinfo:
                received.extend(item.redacted_text for item in call)
        finally:
            gate.set()
        assert core.stream_closed.wait(WAIT_SECONDS)
        assert eventually(lambda: resources_released(stack))
        core.reset()
        after = grpc_call(stack, operation, request_for(STREAM_OPERATION))

    assert_declared("cancelled", operation, observe_grpc(excinfo.value))
    assert received == fragments[:1]
    assert_no_sentinel(json.dumps(received))
    assert stream_result(operation, "grpc", after)["fragment_count"] == len(fragments)


def test_grpc_unary_cancellation_releases_after_work_settles(
    monkeypatch: pytest.MonkeyPatch, core: CoreSpy
) -> None:
    operation = PROTOCOL_MAP.operation("deidentify")
    env = {"OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY": "1"}
    with service_stack(monkeypatch, env) as stack:
        gate = _blocked(core)
        try:
            future = stack.stub.Deidentify.future(
                grpc_message(operation, request_for("deidentify")),
                timeout=RPC_TIMEOUT,
            )
            assert core.entered.wait(WAIT_SECONDS)
            assert future.cancel()
            with pytest.raises(grpc.FutureCancelledError):
                future.result()
        finally:
            gate.set()
        assert core.finished.wait(WAIT_SECONDS)
        assert eventually(lambda: resources_released(stack))
        core.reset()
        after = grpc_call(stack, operation, request_for("deidentify"))

    assert_declared("cancelled", operation, ObservedOutcome("grpc", future.code().name))
    assert after["deidentified_text"]


async def _drive_rest_stream(
    app: Any, body: bytes, *, disconnect_after_first: bool
) -> list[bytes]:
    """Run one REST stream over raw ASGI, optionally disconnecting mid-stream."""
    sent = False
    chunks: list[bytes] = []
    disconnect = asyncio.Event()

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: Mapping[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            chunks.append(bytes(message["body"]))
            if disconnect_after_first:
                disconnect.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/pii/deidentify/stream",
        "raw_path": b"/pii/deidentify/stream",
        "query_string": b"",
        "headers": [
            (b"host", b"127.0.0.1"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 80),
    }
    try:
        await app(scope, receive, send)
    except Exception:  # noqa: BLE001 - the response outcome is asserted instead
        pass
    disconnect.set()
    return chunks


async def _settle(predicate: Callable[[], bool], limit: float = WAIT_SECONDS) -> bool:
    """Poll on the event loop so loop callbacks can run; never times work."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


def _rest_stream_run(
    core: CoreSpy, *, deadline: Optional[float], disconnect: bool
) -> tuple[list[dict[str, Any]], bool]:
    """Drive one blocked REST stream and report whether its lease came back.

    The core blocks after its first fragment, so the deadline or disconnect
    always lands while a window is running in a worker thread.
    """
    operation = PROTOCOL_MAP.operation(STREAM_OPERATION)
    body = json.dumps(
        mapped_request(operation, "rest", request_for(STREAM_OPERATION))
    ).encode("utf-8")
    gate = _blocked(core)

    async def scenario() -> tuple[list[bytes], bool]:
        app = service_app.create_app()
        async with app.router.lifespan_context(app):
            runtime = app.state.runtime
            if deadline is not None:
                runtime.config.timeout = deadline
            try:
                chunks = await _drive_rest_stream(
                    app, body, disconnect_after_first=disconnect
                )
            finally:
                gate.set()
            assert await _settle(core.unblocked.is_set)
            released = await _settle(
                lambda: active_leases(runtime) == 0, limit=LEASE_SETTLE_SECONDS
            )
            return chunks, released

    try:
        chunks, released = asyncio.run(scenario())
    finally:
        gate.set()
    return ndjson(b"".join(chunks).decode("utf-8")), released


def test_rest_stream_deadline_reports_one_in_band_timeout(core: CoreSpy) -> None:
    operation = PROTOCOL_MAP.operation(STREAM_OPERATION)
    fragments = OPERATIONS[STREAM_OPERATION]["model_output"]["fragments"]

    events, _ = _rest_stream_run(core, deadline=REST_STREAM_DEADLINE, disconnect=False)

    assert [event["type"] for event in events] == ["chunk", "error"]
    assert events[0]["redacted_text"] == fragments[0]
    observed = ObservedOutcome(
        "rest", 200, events[-1]["error"]["code"], delivery="stream_event"
    )
    assert_declared("deadline_exceeded", operation, observed)
    assert_no_sentinel(json.dumps(events))


def test_rest_stream_disconnect_emits_no_unsafe_tail(core: CoreSpy) -> None:
    fragments = OPERATIONS[STREAM_OPERATION]["model_output"]["fragments"]

    events, _ = _rest_stream_run(core, deadline=None, disconnect=True)

    assert [event["type"] for event in events] == ["chunk"]
    assert events[0]["redacted_text"] == fragments[0]
    assert_no_sentinel(json.dumps(events))


_REST_STREAM_LEASE_LEAK = (
    "The REST de-identification stream calls close() on a generator still "
    "running in its worker thread, so finish_model_request never runs and the "
    "warm-pool lease leaks after a deadline or disconnect; fixed by #4137."
)


@pytest.mark.xfail(strict=True, reason=_REST_STREAM_LEASE_LEAK)
@pytest.mark.parametrize("trigger", ["deadline", "disconnect"])
def test_rest_stream_releases_its_lease_after_deadline_or_disconnect(
    core: CoreSpy, trigger: str
) -> None:
    _, released = _rest_stream_run(
        core,
        deadline=REST_STREAM_DEADLINE if trigger == "deadline" else None,
        disconnect=trigger == "disconnect",
    )

    assert released


# --- Observability sentinel sweep -------------------------------------------


_REST_SERVER_ERROR_RERAISE = (
    "Starlette's ServerErrorMiddleware re-raises an unhandled exception after "
    "the REST 500 envelope is sent, so the ASGI server logs the exception "
    "message and traceback. gRPC aborts and GraphQL masks the same failure."
)


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason=_REST_SERVER_ERROR_RERAISE)
def test_rest_unhandled_errors_stay_inside_the_application(core: CoreSpy) -> None:
    operation = PROTOCOL_MAP.operation("analyze")
    core.failure = FAILURE_CASES["internal_error"].core_failure
    # ``raise_server_exceptions=True`` surfaces exactly what an ASGI server
    # such as uvicorn receives, and logs, after the response.
    with TestClient(service_app.create_app(), base_url=LOOPBACK_BASE_URL) as client:
        response = client.post(
            operation.bindings["rest"]["path"],
            json=mapped_request(operation, "rest", request_for("analyze")),
        )

    assert response.status_code == 500


def test_logs_traces_metrics_and_errors_never_carry_sentinels(
    monkeypatch: pytest.MonkeyPatch,
    core: CoreSpy,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    from openmed.service.tracing import ServiceTraceConfig, ServiceTracing

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "openmed"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing = ServiceTracing(
        config=ServiceTraceConfig(enabled=True),
        tracer_provider=provider,
        tracer=provider.get_tracer("openmed.service.conformance"),
        propagator=TraceContextTextMapPropagator(),
    )
    monkeypatch.setattr(service_app, "service_tracing_from_env", lambda: tracing)
    sentinel_failure = FAILURE_CASES["internal_error"].core_failure
    records: list[dict[str, Any]] = []
    with service_stack(
        monkeypatch, {"OPENMED_SERVICE_METRICS_ENABLED": "true"}
    ) as stack:
        for operation_id in ALL_OPERATIONS:
            operation = PROTOCOL_MAP.operation(operation_id)
            # A rejected value must never be reflected; pick a validated field.
            invalid = "aggregation_strategy" if operation_id == "analyze" else "lang"
            for transport in transports_of(operation):
                for failure, overrides, outcome_id in (
                    (None, {}, None),
                    (sentinel_failure, {}, "internal_error"),
                    (None, {"model_name": SENTINELS[0]}, "model_not_served"),
                    (None, {invalid: SENTINELS[0]}, "invalid_request"),
                ):
                    core.reset()
                    core.failure = failure
                    observed = observe(
                        stack,
                        operation,
                        transport,
                        request_for(operation_id, **overrides),
                    )
                    if outcome_id is not None:
                        records.append(assert_declared(outcome_id, operation, observed))
        metrics = stack.client.get("/metrics")
        assert metrics.status_code == 200
        assert eventually(lambda: resources_released(stack))
    provider.shutdown()

    spans = list(exporter.get_finished_spans())
    assert spans
    rendered = json.dumps(
        [
            {
                "name": span.name,
                "attributes": {
                    key: str(value) for key, value in span.attributes.items()
                },
                "events": [
                    {
                        "name": event.name,
                        "attributes": {
                            key: str(value)
                            for key, value in (event.attributes or {}).items()
                        },
                    }
                    for event in span.events
                ],
                "status": str(span.status.description),
            }
            for span in spans
        ]
    )
    assert_no_sentinel(rendered)
    assert_no_sentinel(metrics.text)
    assert_no_sentinel(caplog.text)
    assert_value_free(records)
    assert {record["transport"] for record in records} == {"rest", "graphql", "grpc"}
