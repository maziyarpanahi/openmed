"""Each surface reports a failure the way the error-code registry records.

The code sets are read from the real tables and source of each surface (the
REST compatibility inventory, the gRPC failure table, GraphQL source, the MCP
mapper), and each mapping is exercised with synthetic failures through the
real REST exception handlers, gRPC abort helpers, GraphQL formatter and MCP
envelope builder. Nothing here loads a model or opens a socket.
"""

from __future__ import annotations

import ast
import importlib
import json
import pkgutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Optional

import grpc
import pytest
from fastapi.testclient import TestClient
from graphql import GraphQLError
from starlette.exceptions import HTTPException

import openmed
from openmed.core import errors as taxonomy
from openmed.core.error_registry import ErrorCodeRegistry, load_error_registry
from openmed.service import runtime as service_runtime
from openmed.service.api_compatibility import discover_service_error_categories
from openmed.service.protocol_map import discover_transport_surface, load_protocol_map
from openmed.utils.gateway import InputValidationError

ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = ROOT / "openmed" / "service"
GRPC_DOC = ROOT / "docs" / "serving" / "grpc.md"
LOOPBACK = "http://127.0.0.1"
SENTINEL = "SYNTHETIC_RAW_PHI_SENTINEL_R2"
OMITTED_DIVERGENCE = "rest_inventory_omits_emitted_codes"
UNEMITTED_DIVERGENCE = "rest_inventory_lists_unemitted_codes"
GRAPHQL_GAP_DIVERGENCE = "protocol_map_graphql_codes_incomplete"


@pytest.fixture(scope="module")
def registry() -> ErrorCodeRegistry:
    return load_error_registry()


@pytest.fixture(scope="module")
def surface():
    return discover_transport_surface()


def _leaf_codes(registry: ErrorCodeRegistry) -> set[str]:
    """Return the codes carried by shared input-validation errors."""
    leaf_class = "openmed.utils.gateway.InputValidationError"
    return {
        entry.code for entry in registry.codes.values() if leaf_class in entry.python
    }


def _http_variants(entry, *, app: str = "service") -> set[tuple]:
    return {
        (mapping.status, mapping.code, mapping.retry_after)
        for mapping in entry.http
        if mapping.delivery == "response" and mapping.app == app
    }


# ---------------------------------------------------------------------------
# REST inventory and status tables
# ---------------------------------------------------------------------------


def _rest_source_calls() -> Iterator[tuple[str, Optional[int], str]]:
    """Yield ``(app, status, code)`` for every REST error call in source.

    A call counts when its first two positional arguments are a literal 4xx
    or 5xx status and a code: a literal, a module attribute (imported names
    included), or the variable of a ``for`` loop over literal codes.
    ``_ndjson_error`` stream events yield a ``None`` status.
    """
    for path in sorted(SERVICE_ROOT.rglob("*.py")):
        if "proto" in path.parts:
            continue
        relative = path.relative_to(SERVICE_ROOT).as_posix()
        module = importlib.import_module(
            "openmed.service." + relative.removesuffix(".py").replace("/", ".")
        )
        app = (
            "redaction"
            if relative == "redaction_app.py"
            else "privacy_proxy"
            if relative.startswith("privacy_proxy/")
            else "service"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        loop_codes: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and isinstance(node.iter, (ast.Tuple, ast.List))
                and all(
                    isinstance(item, ast.Constant) and isinstance(item.value, str)
                    for item in node.iter.elts
                )
            ):
                loop_codes.setdefault(node.target.id, set()).update(
                    item.value for item in node.iter.elts
                )

        def resolve(node: ast.AST) -> set[str]:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return {node.value}
            if isinstance(node, ast.Name):
                value = getattr(module, node.id, None)
                if isinstance(value, str):
                    return {value}
                return loop_codes.get(node.id, set())
            return set()

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else None
            name = function.id if isinstance(function, ast.Name) else name
            if name == "_ndjson_error":
                for code in resolve(node.args[0]):
                    yield app, None, code
                continue
            if len(node.args) < 2:
                continue
            status = node.args[0]
            if not (
                isinstance(status, ast.Constant)
                and type(status.value) is int
                and 400 <= status.value <= 599
            ):
                continue
            for code in resolve(node.args[1]):
                yield app, status.value, code


def test_rest_inventory_and_registry_agree(registry) -> None:
    inventory = set(discover_service_error_categories())
    source = inventory | {code for _, _, code in _rest_source_calls()}
    emitted = {
        mapping.code for entry in registry.codes.values() for mapping in entry.http
    }
    assert inventory <= set(registry.codes)
    assert source - inventory <= emitted
    for code in sorted(inventory - emitted):
        assert UNEMITTED_DIVERGENCE in registry.entry(code).divergences, code
    for code in sorted(emitted - inventory):
        assert OMITTED_DIVERGENCE in registry.entry(code).divergences, code
        assert code in source or code in _leaf_codes(registry), code
    unemitted = {
        entry.code
        for entry in registry.codes.values()
        if UNEMITTED_DIVERGENCE in entry.divergences
    }
    assert unemitted == inventory - emitted


def test_rest_stream_event_codes_match_the_stream_source(registry, surface) -> None:
    stream_codes = {
        mapping.code
        for entry in registry.codes.values()
        for mapping in entry.http
        if mapping.delivery == "stream_event"
    }
    source = set(surface.rest_stream_event_codes) | {
        code for _, status, code in _rest_source_calls() if status is None
    }
    assert stream_codes == source


def test_literal_rest_error_calls_match_the_registry(registry) -> None:
    calls = sorted(
        {call for call in _rest_source_calls() if call[1] is not None},
        key=str,
    )
    assert len(calls) > 20
    for app, status, code in calls:
        mappings = [
            mapping
            for mapping in registry.entry(code).http
            if mapping.app == app and mapping.code == code
        ]
        assert any(
            mapping.status == status
            or (mapping.note or "").endswith(f"mediated status {status}")
            for mapping in mappings
        ), (app, status, code)


def test_rest_status_tables_match_the_registry(registry) -> None:
    from openmed.service import idempotency, redaction_app, request_limits
    from openmed.service.app import _TAXONOMY_HTTP_STATUS

    for cls, status in _TAXONOMY_HTTP_STATUS.items():
        entry = registry.entry(cls.code)
        assert entry.http_status == status, cls.code
    for code, status in idempotency._ERROR_STATUS.items():
        assert (status, code) in {
            (status_, code_)
            for status_, code_, _ in _http_variants(registry.entry(code))
        }
    for code, status in request_limits._REJECTION_STATUS.items():
        assert registry.entry(code).http_status == status, code
    for status, code, _message in redaction_app._ERROR_DETAILS.values():
        assert status in {
            status_
            for status_, code_, _ in _http_variants(
                registry.entry(code), app="redaction"
            )
            if code_ == code
        }, (status, code)


def _service_exception_classes() -> Iterator[type[BaseException]]:
    import openmed.service as package

    for info in pkgutil.walk_packages(package.__path__, "openmed.service."):
        if ".proto." in info.name:
            continue
        module = importlib.import_module(info.name)
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, BaseException)
                and value.__module__ == module.__name__
            ):
                yield value


def test_exception_status_attributes_match_the_registry(registry) -> None:
    checked = 0
    for cls in _service_exception_classes():
        code = getattr(cls, "error_code", None)
        status = getattr(cls, "status_code", None)
        if not isinstance(code, str) or type(status) is not int:
            continue
        entry = registry.entry(code)
        app = "privacy_proxy" if ".privacy_proxy." in cls.__module__ else "service"
        assert status in {
            status_ for status_, code_, _ in _http_variants(entry, app=app)
        }, (cls.__name__, status, code)
        checked += 1
    assert checked >= 10


# ---------------------------------------------------------------------------
# REST behaviour through the real exception handlers
# ---------------------------------------------------------------------------


def _rest_cases() -> list[tuple[str, Callable[[], BaseException]]]:
    from openmed.service import (
        backpressure,
        bulk_data,
        compatibility,
        drain,
        egress,
        idempotency,
        job_admission,
        job_bounds,
        jobs,
        privacy_gateway,
        replica_state,
        resilience,
        throttle,
        warm_pool,
    )
    from openmed.service.app import ServiceTimeoutError, _DecisionShapeRefused

    message = "Synthetic failure."
    raw = f"Synthetic {SENTINEL} failure."
    cases: list[tuple[str, Callable[[], BaseException]]] = [
        (code, lambda name=name: getattr(taxonomy, name)(message))
        for name, code in taxonomy.ERROR_CODES.items()
    ]
    cases += [
        (
            "bad_request",
            lambda: taxonomy.InputError(message, code="bad_request"),
        ),
        (
            "suspicious_content",
            lambda: InputValidationError(message, code="suspicious_content"),
        ),
        ("max_chars", lambda: InputValidationError(message, code="max_chars")),
        ("model_not_served", service_runtime.ModelNotServedError),
        ("timeout", lambda: ServiceTimeoutError(1.0)),
        (
            "circuit_breaker_open",
            lambda: resilience.CircuitBreakerOpenError(retry_after_seconds=2),
        ),
        ("not_ready", drain.ServiceShuttingDownError),
        ("shutting_down", drain.ServiceShuttingDownError),
        (
            "service_busy",
            lambda: warm_pool.WarmPoolBackpressureError(
                model_name="synthetic",
                required_bytes=1,
                budget_bytes=1,
                wait_seconds=0.0,
            ),
        ),
        (
            "service_busy",
            lambda: throttle.ServiceBusyError(
                SimpleNamespace(max_concurrency=1, wait_seconds=0.0), "client"
            ),
        ),
        (
            "backpressure",
            lambda: backpressure.BackpressureError(
                priority="normal",
                queue_depth=1,
                queue_capacity=1,
                retry_after_seconds=1.0,
            ),
        ),
        (
            "bulk_jobs_saturated",
            lambda: job_bounds.BulkJobsSaturatedError(
                jobs="bulk", max_running_jobs=1, retry_after_seconds=1.0
            ),
        ),
        (
            "job_queue_full",
            lambda: job_admission.JobQueueFullError(
                reason="pending_jobs", limit=1, retry_after_seconds=1
            ),
        ),
        (
            "job_too_large",
            lambda: job_admission.JobTooLargeError(reason="text_bytes", limit=1),
        ),
        (
            "job_too_many_documents",
            lambda: job_admission.JobTooManyDocumentsError(reason="documents", limit=1),
        ),
        ("egress_denied", lambda: egress.EgressDenied("budget_exceeded")),
        ("privacy_gateway_error", lambda: privacy_gateway.PrivacyGatewayError(message)),
        (
            "privacy_gateway_blocked",
            lambda: privacy_gateway.PrivacyPolicyViolation(message),
        ),
        (
            "privacy_gateway_reidentification_error",
            lambda: privacy_gateway.PrivacyReidentificationError(message),
        ),
        (
            "privacy_gateway_transport_error",
            lambda: privacy_gateway.PrivacyTransportError(message),
        ),
        (
            "privacy_gateway_not_configured",
            privacy_gateway.PrivacyGatewayConfigurationError,
        ),
        (
            "component_disabled",
            lambda: replica_state.ReplicaComponentUnavailableError(
                "async_jobs", replica_state.COMPONENT_DISABLED
            ),
        ),
        (
            "replica_unsafe_state",
            lambda: replica_state.ReplicaComponentUnavailableError(
                "async_jobs", "replica_unsafe"
            ),
        ),
        (
            "incompatible_state",
            lambda: compatibility.IncompatibleStateError("job_store", "synthetic"),
        ),
        ("incompatible_state", lambda: jobs.JobStoreSchemaError("synthetic")),
        (
            "decision_media_unsupported",
            lambda: _DecisionShapeRefused("media_unsupported"),
        ),
        ("decision_request_ambiguous", lambda: _DecisionShapeRefused("ambiguous")),
        ("bad_request", lambda: ValueError(raw)),
        ("bad_request", lambda: HTTPException(status_code=404, detail="Not Found")),
        ("bad_request", lambda: HTTPException(status_code=409, detail="Conflict")),
        (
            "bad_request",
            lambda: HTTPException(status_code=405, detail="Method Not Allowed"),
        ),
        ("internal_error", lambda: RuntimeError(raw)),
    ]
    cases += [
        (code, lambda code=code: idempotency.IdempotencyError(code))
        for code in sorted(idempotency._ERROR_STATUS)
    ]
    cases += [
        (code, lambda code=code: bulk_data.BulkPathError(code))
        for code in (
            "bulk_paths_disabled",
            "path_invalid",
            "path_outside_root",
            "path_exists",
        )
    ]
    return cases


@pytest.fixture(scope="module")
def probe_client() -> Iterator[tuple[TestClient, dict[str, Callable[[], Any]]]]:
    from openmed.service import auth
    from openmed.service.app import create_app

    factories: dict[str, Callable[[], BaseException]] = {}
    route = "/__registry_probe/{name}"
    with pytest.MonkeyPatch.context() as patch:
        # The probe is a control route without scopes or model admission, so
        # only the exception handlers shape its refusals.
        patch.setenv("OPENMED_PROFILE", "test")
        patch.setitem(
            auth.ROUTE_POLICIES, ("GET", route), auth.RoutePolicy((), "control")
        )
        app = create_app()

        async def probe(name: str) -> None:
            raise factories[name]()

        app.add_api_route(route, probe, methods=["GET"])
        with TestClient(
            app, base_url=LOOPBACK, raise_server_exceptions=False
        ) as client:
            yield client, factories


@pytest.mark.parametrize(
    "index,code",
    [(index, code) for index, (code, _) in enumerate(_rest_cases())],
)
def test_rest_handlers_answer_as_registered(
    registry, probe_client, index, code
) -> None:
    client, factories = probe_client
    factories[str(index)] = _rest_cases()[index][1]
    response = client.get(f"/__registry_probe/{index}")
    body = response.json()
    observed = (
        response.status_code,
        body["error"]["code"],
        "retry-after" in response.headers,
    )
    assert observed in _http_variants(registry.entry(code)), observed
    assert SENTINEL not in response.text


@pytest.mark.parametrize("class_name", sorted(taxonomy.ERROR_CODES))
def test_rest_analyze_route_reports_a_taxonomy_error_as_registered(
    monkeypatch, registry, class_name
) -> None:
    from openmed.service.app import create_app

    def failing_analyze(*_args: Any, **_kwargs: Any) -> None:
        raise getattr(taxonomy, class_name)("Synthetic failure.")

    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setattr(service_runtime, "ModelLoader", _FakeLoader)
    monkeypatch.setattr(openmed, "analyze_text", failing_analyze)
    with TestClient(create_app(), base_url=LOOPBACK) as client:
        response = client.post("/analyze", json={"text": "synthetic"})
    entry = registry.entry(taxonomy.ERROR_CODES[class_name])
    observed = (response.status_code, response.json()["error"]["code"], False)
    assert observed in _http_variants(entry)
    assert SENTINEL not in response.text


def test_middleware_refusals_answer_as_registered(registry) -> None:
    from openmed.service import admission, request_signing
    from openmed.service.app import _error_response

    signer = SimpleNamespace(_error_response=_error_response)
    for code in request_signing._FAILURE_MESSAGES:
        response = request_signing.RequestSignatureVerifier._reject(signer, code)
        observed = (response.status_code, json.loads(response.body)["error"]["code"])
        assert observed + (False,) in _http_variants(registry.entry(code))

    gate = SimpleNamespace(_error_response=_error_response)
    for error in (
        admission.PrincipalAdmissionLimited("principal_share", retry_after_seconds=1.0),
        admission.AdmissionSaturated("global_limit", retry_after_seconds=1.0),
    ):
        response = admission.HttpAdmission.rejection_response(gate, error)
        code = json.loads(response.body)["error"]["code"]
        observed = (response.status_code, code, "retry-after" in response.headers)
        assert observed in _http_variants(registry.entry(code))


class _FakeLoader:
    """Model loader stand-in; no model is ever loaded."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# gRPC
# ---------------------------------------------------------------------------


class _AbortedRpc(Exception):
    pass


class _FakeContext:
    def __init__(self, metadata: tuple[tuple[str, str], ...] = ()) -> None:
        self.metadata = metadata
        self.trailers: dict[str, str] = {}
        self.status: Optional[grpc.StatusCode] = None
        self.details: Optional[str] = None

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self.metadata

    def set_trailing_metadata(self, items) -> None:
        self.trailers.update(dict(items))

    def abort(self, status: grpc.StatusCode, details: str) -> None:
        self.status = status
        self.details = details
        raise _AbortedRpc


def _grpc_variants(entry) -> set[tuple]:
    return {
        (mapping.status, mapping.reason, mapping.details, mapping.retryable_metadata)
        for mapping in entry.grpc
        if mapping.origin == "server"
    }


def _observe_grpc(context: _FakeContext) -> tuple:
    from openmed.service.grpc_policy import OPENMED_REASON_METADATA_KEY
    from openmed.service.grpc_server import GRPC_RETRYABLE_METADATA_KEY

    reason = context.trailers.get(OPENMED_REASON_METADATA_KEY)
    return (
        context.status.name,
        reason,
        None if reason is not None else context.details,
        context.trailers.get(GRPC_RETRYABLE_METADATA_KEY) == "true",
    )


def test_grpc_failure_table_matches_the_registry(registry, surface) -> None:
    from openmed.service.grpc_policy import GRPC_FAILURES

    reasons = {
        mapping.reason
        for entry in registry.codes.values()
        for mapping in entry.grpc
        if mapping.reason is not None
    }
    assert reasons == set(GRPC_FAILURES)
    for reason, failure in GRPC_FAILURES.items():
        entry = registry.entry(reason)
        assert entry.category == failure.category, reason
        assert (failure.status.name, reason, None, False) in _grpc_variants(entry)
    assert set(surface.grpc_failures) == set(GRPC_FAILURES)


def _grpc_unary_cases() -> list[tuple[str, Callable[[], BaseException]]]:
    from openmed.service import drain, resilience, warm_pool
    from openmed.service.grpc_policy import GRPC_FAILURES, GrpcPolicyError

    message = f"Synthetic {SENTINEL} failure."
    cases: list[tuple[str, Callable[[], BaseException]]] = [
        (code, lambda name=name: getattr(taxonomy, name)(message))
        for name, code in taxonomy.ERROR_CODES.items()
    ]
    cases += [
        (reason, lambda reason=reason: GrpcPolicyError(reason))
        for reason in sorted(GRPC_FAILURES)
    ]
    cases += [
        ("max_chars", lambda: InputValidationError(message, code="max_chars")),
        ("model_not_served", service_runtime.ModelNotServedError),
        ("shutting_down", drain.ServiceShuttingDownError),
        ("not_ready", drain.ServiceShuttingDownError),
        ("circuit_breaker_open", lambda: resilience.CircuitBreakerOpenError(1)),
        (
            "service_busy",
            lambda: warm_pool.WarmPoolBackpressureError(
                model_name="synthetic",
                required_bytes=1,
                budget_bytes=1,
                wait_seconds=0.0,
            ),
        ),
    ]
    return cases


@pytest.mark.parametrize(
    "index,code", [(index, code) for index, (code, _) in enumerate(_grpc_unary_cases())]
)
def test_grpc_unary_failures_abort_as_registered(registry, index, code) -> None:
    from openmed.service.grpc_server import _run_unary

    factory = _grpc_unary_cases()[index][1]
    context = _FakeContext()

    def operation() -> None:
        raise factory()

    with pytest.raises(_AbortedRpc):
        _run_unary(context, operation)
    assert _observe_grpc(context) in _grpc_variants(registry.entry(code))
    assert SENTINEL not in str(context.details)


def test_grpc_peer_contract_refusals_match_the_registry(registry) -> None:
    from openmed.service.compatibility import (
        CURRENT_CONTRACT,
        PEER_CONTRACT_METADATA_KEY,
    )
    from openmed.service.grpc_server import _abort_incompatible_peer

    outside = f"{CURRENT_CONTRACT.contract_generation + 100}/" + str(
        CURRENT_CONTRACT.contract_generation + 100
    )
    for header, code in (
        ("not-a-contract", "peer_contract_malformed"),
        (outside, "peer_incompatible"),
    ):
        context = _FakeContext(((PEER_CONTRACT_METADATA_KEY, header),))
        with pytest.raises(_AbortedRpc):
            _abort_incompatible_peer(context)
        assert _observe_grpc(context) in _grpc_variants(registry.entry(code))


def test_grpc_status_details_codes_are_registered(registry) -> None:
    details = {
        mapping.details
        for entry in registry.codes.values()
        for mapping in entry.grpc
        if mapping.details is not None
    }
    assert details == {
        "shutting_down",
        "peer_contract_malformed",
        "peer_generation_outside_window",
    }
    for entry in registry.codes.values():
        if any(mapping.details is not None for mapping in entry.grpc):
            assert "grpc_status_details_codes" in entry.divergences, entry.code


def test_grpc_doc_reason_table_matches_the_registry(registry) -> None:
    lines = GRPC_DOC.read_text(encoding="utf-8").splitlines()
    start = lines.index("| Reason | Status |")
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        reason, status = (
            cell.strip().strip("`") for cell in line.strip("|").split("|")
        )
        rows.append((reason, status))
    from openmed.service.grpc_policy import GRPC_FAILURES

    assert {reason for reason, _ in rows} == set(GRPC_FAILURES)
    for reason, status in rows:
        assert (status, reason, None, False) in _grpc_variants(registry.entry(reason))


# ---------------------------------------------------------------------------
# GraphQL
# ---------------------------------------------------------------------------


def _graphql_source_codes() -> set[str]:
    """Return every ``extensions.code`` GraphQL source can emit."""
    from openmed.service.graphql_limits import (
        GraphQLLimitCategory,
        GraphQLLimitViolation,
    )

    codes: set[str] = set()
    for path in sorted(SERVICE_ROOT.glob("graphql_*.py")):
        module = importlib.import_module(f"openmed.service.{path.stem}")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not (isinstance(key, ast.Constant) and key.value == "code"):
                    continue
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    codes.add(value.value)
                elif isinstance(value, ast.Name):
                    resolved = getattr(module, value.id)
                    assert isinstance(resolved, str), value.id
                    codes.add(resolved)
    for category in GraphQLLimitCategory:
        codes.add(GraphQLLimitViolation(category=category).code)
    return codes


def test_graphql_codes_match_the_graphql_source(registry) -> None:
    registered = {
        mapping.code
        for entry in registry.codes.values()
        for mapping in entry.graphql
        if mapping.code is not None
    }
    assert registered == _graphql_source_codes()
    for code in registered:
        assert code in registry.codes, code


def test_protocol_map_graphql_gap_is_recorded(registry) -> None:
    listed = set(load_protocol_map().codes["graphql"])
    source = _graphql_source_codes()
    assert listed <= source
    for code in source - listed:
        assert GRAPHQL_GAP_DIVERGENCE in registry.entry(code).divergences, code


def test_graphql_formatter_masks_resolver_failures_as_registered(registry) -> None:
    from openmed.service.drain import SHUTTING_DOWN
    from openmed.service.graphql_app import _safe_formatted_error

    for code, error in (
        (
            "input_error",
            GraphQLError("x", original_error=taxonomy.InputError(SENTINEL)),
        ),
        (
            "model_not_served",
            GraphQLError("x", extensions={"code": "model_not_served"}),
        ),
        ("shutting_down", GraphQLError("x", extensions={"code": SHUTTING_DOWN})),
    ):
        formatted = _safe_formatted_error(error)
        observed = (
            formatted["extensions"]["code"],
            formatted["extensions"].get("retryable") is True,
        )
        assert observed in {
            (mapping.code, mapping.retryable_extension)
            for mapping in registry.entry(code).graphql
        }
        assert SENTINEL not in json.dumps(formatted)


@pytest.mark.parametrize("class_name", sorted(taxonomy.ERROR_CODES))
def test_graphql_resolvers_report_taxonomy_errors_as_registered(
    monkeypatch, registry, class_name
) -> None:
    from openmed.service.app import create_app

    def failing_analyze(*_args: Any, **_kwargs: Any) -> None:
        raise getattr(taxonomy, class_name)(f"Synthetic {SENTINEL} failure.")

    monkeypatch.setenv("OPENMED_PROFILE", "test")
    monkeypatch.setattr(service_runtime, "ModelLoader", _FakeLoader)
    monkeypatch.setattr(openmed, "analyze_text", failing_analyze)
    query = "query($input: AnalyzeInput!) { analyze(input: $input) { text } }"
    with TestClient(create_app(), base_url=LOOPBACK) as client:
        response = client.post(
            "/graphql",
            json={"query": query, "variables": {"input": {"text": "synthetic"}}},
        )
    code = response.json()["errors"][0]["extensions"]["code"]
    entry = registry.entry(taxonomy.ERROR_CODES[class_name])
    assert (response.status_code, code) in {
        (mapping.http_status, mapping.code) for mapping in entry.graphql
    }
    assert SENTINEL not in response.text


def test_graphql_batch_refusal_answers_as_registered(registry) -> None:
    from openmed.service.app import create_app

    with TestClient(create_app(), base_url=LOOPBACK) as client:
        response = client.post(
            "/graphql", json=[{"query": "{ entityTypes { label } }"}]
        )
    code = response.json()["errors"][0]["extensions"]["code"]
    assert (response.status_code, code) in {
        (mapping.http_status, mapping.code)
        for mapping in registry.entry("OPENMED_GRAPHQL_BATCH_UNSUPPORTED").graphql
    }


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------


def _mcp_source_codes() -> set[str]:
    """Return every code the MCP error envelope can carry."""
    from openmed.agent.security.injection_guard import PromptInjectionDetected
    from openmed.mcp import governed_workflows, server
    from openmed.service.security import MCPGatewaySecurityError

    codes = set(server.MCP_ERROR_CODES.values())
    codes.update(
        code
        for class_name, code in _input_validation_codes()
        if class_name == "InputValidationError"
    )
    pending = [MCPGatewaySecurityError]
    while pending:
        cls = pending.pop()
        codes.add(cls.code)
        pending.extend(cls.__subclasses__())
    codes.update(governed_workflows._ERROR_CODES)
    codes.add(PromptInjectionDetected([]).to_dict()["code"])
    tree = ast.parse(Path(server.__file__).read_text(encoding="utf-8"))
    envelope = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_error_envelope"
    )
    for node in ast.walk(envelope):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "code" for t in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            codes.add(node.value.value)
    # Clinical tools report a failed stage inside a successful result.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_clinical_handler_error"
            and len(node.args) > 1
            and isinstance(node.args[1], ast.Constant)
        ):
            codes.add(node.args[1].value)
    return codes


def _input_validation_codes() -> Iterator[tuple[str, str]]:
    for path in sorted((ROOT / "openmed" / "utils").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else None
            name = function.id if isinstance(function, ast.Name) else name
            for keyword in node.keywords:
                if keyword.arg == "code" and isinstance(keyword.value, ast.Constant):
                    yield str(name), str(keyword.value.value)


def test_mcp_codes_match_the_mcp_mapper(registry) -> None:
    registered = {
        mapping.code for entry in registry.codes.values() for mapping in entry.mcp
    }
    assert registered == _mcp_source_codes()


def _mcp_cases() -> list[tuple[str, Callable[[], BaseException]]]:
    from openmed.agent.security.injection_guard import PromptInjectionDetected
    from openmed.mcp import consent_receipts, governed_workflows, tool_registry
    from openmed.service import drain, resilience, security, warm_pool

    message = "Synthetic failure."
    raw = f"Synthetic {SENTINEL} failure."
    cases: list[tuple[str, Callable[[], BaseException]]] = [
        (code, lambda name=name: getattr(taxonomy, name)(message))
        for name, code in taxonomy.ERROR_CODES.items()
    ]
    cases += [
        (code, lambda code=code: InputValidationError(message, code=code))
        for _, code in sorted(set(_input_validation_codes()))
    ]
    pending = [security.MCPGatewaySecurityError]
    while pending:
        cls = pending.pop()
        cases.append((cls.code, lambda cls=cls: cls(message)))
        pending.extend(cls.__subclasses__())
    cases += [
        (code, lambda code=code: governed_workflows.GovernedMCPError(code))
        for code in sorted(governed_workflows._ERROR_CODES)
    ]
    cases += [
        ("prompt_injection_detected", lambda: PromptInjectionDetected([])),
        (
            "consent_required",
            lambda: consent_receipts.ConsentReceiptRequiredError(message),
        ),
        (
            "consent_denied",
            lambda: consent_receipts.ConsentReceiptVerificationError(message),
        ),
        (
            "invalid_consent_receipt",
            lambda: consent_receipts.ConsentReceiptError(message),
        ),
        ("invalid_result", lambda: tool_registry.ToolSchemaValidationError(message)),
        ("unknown_tool", lambda: KeyError(raw)),
        ("invalid_arguments", lambda: TypeError(raw)),
        ("execution_error", lambda: RuntimeError(raw)),
        ("model_not_served", service_runtime.ModelNotServedError),
        ("validation_error", lambda: ValueError(raw)),
        ("circuit_breaker_open", lambda: resilience.CircuitBreakerOpenError(1)),
        ("shutting_down", drain.ServiceShuttingDownError),
        (
            "service_busy",
            lambda: warm_pool.WarmPoolBackpressureError(
                model_name="synthetic",
                required_bytes=1,
                budget_bytes=1,
                wait_seconds=0.0,
            ),
        ),
    ]
    return cases


@pytest.mark.parametrize(
    "index,code", [(index, code) for index, (code, _) in enumerate(_mcp_cases())]
)
def test_mcp_envelopes_carry_the_registered_code(registry, index, code) -> None:
    from openmed.mcp.server import _error_envelope

    envelope = _error_envelope(_mcp_cases()[index][1]())
    assert envelope["is_error"] is True
    assert envelope["error"]["code"] in {
        mapping.code for mapping in registry.entry(code).mcp
    }
    assert SENTINEL not in json.dumps(envelope)


def test_mcp_payload_ignores_an_instance_code_as_recorded(registry) -> None:
    from openmed.mcp.server import mcp_error_payload
    from openmed.service.app import _openmed_error_response

    error = taxonomy.InputError("Synthetic failure.", code="bad_request")
    rest_code = json.loads(_openmed_error_response(error).body)["error"]["code"]
    mcp_code = mcp_error_payload(error)["error"]["code"]
    assert (rest_code, mcp_code) == ("bad_request", "input_error")
    assert "mcp_payload_ignores_instance_code" in registry.entry(rest_code).divergences
    assert not registry.entry("bad_request").mcp


# ---------------------------------------------------------------------------
# Protocol map
# ---------------------------------------------------------------------------


def test_protocol_map_codes_agree_with_the_registry(registry) -> None:
    protocol_map = load_protocol_map()
    for code, item in protocol_map.codes["rest"].items():
        entry = registry.entry(code)
        assert entry.http_status == item.status, code
        assert entry.category == item.category, code
    for code, item in protocol_map.codes["grpc"].items():
        entry = registry.entry(code)
        assert entry.category == item.category, code
        assert item.status in {mapping.status for mapping in entry.grpc}, code
    for code in protocol_map.codes["graphql"]:
        assert any(mapping.code == code for mapping in registry.entry(code).graphql)
    for codes in protocol_map.rest_only_codes.values():
        for code in codes:
            assert registry.entry(code).code == code
