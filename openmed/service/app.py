"""FastAPI application for the OpenMed REST service."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Sequence, Tuple

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware

import openmed
from openmed.core.errors import (
    BudgetExceededError,
    CapabilityError,
    ConfigurationError,
    InferenceError,
    InputError,
    InternalError,
    OpenMedError,
    PolicyError,
)
from openmed.core.key_provider import KeyProvider
from openmed.processing import format_predictions
from openmed.structured.decision import (
    DETERMINISTIC_DECISION_BACKEND,
    DecisionAccessPolicy,
    DecisionError,
    DecisionResult,
    decide,
    decision_result_schema,
)
from openmed.structured.decision import (
    DecisionRequest as CoreDecisionRequest,
)
from openmed.structured.decision_interop import (
    DecisionRequestShapeError,
    decision_request_shape,
    from_openai_decisions_request,
    openai_decisions_response_schema,
    to_openai_decisions_response,
)
from openmed.utils.deprecation import SHIFT_DATES_NOTICE
from openmed.utils.validation import validate_model_name

from .admission import HttpAdmission
from .auth import (
    DEVELOPMENT_DOCS_EXEMPT_PATHS,
    ROUTE_POLICIES,
    SERVICE_METADATA_PATH,
    AuthPrincipal,
    ServiceAuth,
    is_development_profile,
    parse_api_docs_config,
    parse_service_auth_config,
    request_needs_admission,
    route_key_for_request,
    validate_route_policies,
)
from .batcher import (
    BackpressureError,
    BatchPriorityHandle,
    BatchResult,
    DynamicBatcher,
    normalize_priority,
)
from .bulk_data import (
    BulkPathError,
    BulkStoragePolicy,
    FHIRBulkJobConfig,
    FHIRBulkJobManager,
    bulk_job_bounds_from_env,
)
from .coalesce import RequestCoalescer, coalescing_key
from .compatibility import (
    CURRENT_CONTRACT,
    JOB_STORE_COMPONENT,
    PEER_CONTRACT_HEADER,
    CompatibilityGate,
    IncompatibleStateError,
    SchemaObservation,
    job_store_observation,
    local_descriptor,
    peer_contract_decision,
)
from .drain import (
    BackgroundJobsDrainParticipant,
    BatcherDrainParticipant,
    CallbackDrainParticipant,
    DrainCoordinator,
    HTTPRequestDrainParticipant,
    JobQueueDrainParticipant,
    NDJSONStreamDrainParticipant,
    ServiceShuttingDownError,
    shutting_down_details,
)
from .egress import EgressDenied, validate_egress_configuration
from .governed_workflows import WorkflowGovernanceService, WorkflowHTTPPolicy
from .idempotency import (
    IDEMPOTENCY_KEY_HEADER,
    IDEMPOTENCY_KEY_MAX_LENGTH,
    IDEMPOTENCY_REPLAYED_HEADER,
    IdempotencyError,
    IdempotencyReservation,
    IdempotentReplay,
    IdempotentStartRequest,
    StartIdempotencyStore,
    parse_idempotency_key,
)
from .job_admission import JobAdmissionError
from .job_bounds import BulkJobsSaturatedError
from .job_ownership import BULK_ADMIN_SCOPE, JOBS_ADMIN_SCOPE, JobOwnershipPolicy
from .jobs import (
    OWNER_DIGEST_FIELD,
    SERVICE_JOBS_STORE_ENV_VAR,
    DeidentifyJobQueue,
    JobDigestPolicy,
    JobStoreSchemaError,
    job_response_payload,
    parse_jobs_store_path,
)
from .journey_access import (
    JourneyAccessContextProvider,
    JourneyAccessGate,
    JourneyAccessRequest,
)
from .journey_resources import (
    JourneyAccessPolicy,
    JourneyResourceCatalog,
    JourneyResourceKind,
    parse_access_attributes,
    parse_resource_fields,
)
from .limits import get_max_text_length
from .logging import (
    CorrelationIdMiddleware,
    current_request_id,
    request_scope_path,
    service_log_config_from_env,
    set_access_log_brief,
    set_access_log_grounding,
    set_access_log_model_name,
)
from .metrics import (
    PROMETHEUS_CONTENT_TYPE,
    PrometheusMetricsRegistry,
    metrics_enabled_from_env,
)
from .mtls import ServiceMTLS, parse_service_mtls_config
from .openhim_mediator import (
    OPENHIM_HEARTBEAT_PATH,
    OPENHIM_MEDIA_TYPE,
    OPENHIM_MEDIATOR_PATH,
    OpenHIMMediatorSettings,
    build_openhim_envelope,
    create_openhim_mediator,
    failed_openhim_result,
    mediator_response_headers,
    transform_mediator_payload,
)
from .operational import OperationalCategory, OperationalEvent, OperationalState
from .privacy_gateway import (
    HttpExternalLLMTransport,
    InMemoryReidentificationStore,
    PrivacyGateway,
    PrivacyGatewayAuditTrail,
    PrivacyGatewayConfigurationError,
    PrivacyGatewayError,
    PrivacyGatewayPolicy,
    PrivacyTransportError,
)
from .replica_state import (
    ASYNC_JOBS,
    AUTH_FAILURE_LIMITER,
    COMPONENT_DISABLED,
    CONCURRENCY_LIMITER,
    FHIR_BULK_JOBS,
    RATE_LIMITER,
    REQUEST_COALESCER,
    SMART_BACKEND_JOBS,
    ReplicaComponentUnavailableError,
    ReplicaStatePolicy,
    ReplicaStateReport,
)
from .request_limits import (
    REQUEST_BODY_TIMEOUT,
    REQUEST_SIZE_INVALID,
    BoundedRequestBodyMiddleware,
    connection_rejection_count,
    request_body_read_limits_from_env,
    request_limits_from_env,
)
from .request_signing import install_request_signing
from .resilience import CircuitBreakerOpenError, circuit_breaker_details
from .response_headers import install_response_headers
from .runtime import (
    ServiceRuntime,
    parse_service_coalescing_config,
    parse_service_throttle_config,
)
from .schemas import (
    AnalyzeRequest,
    BriefRequest,
    BriefResponse,
    CohortResolveRequest,
    DecisionsRequestBody,
    DeidentifyJobRequest,
    FHIRBulkExportRequest,
    FHIRBulkImportRequest,
    GroundRequest,
    GroundResponse,
    JourneyResourcePageResponse,
    ModelUnloadRequest,
    OmopLoadRequest,
    OpenAIDecisionsRequest,
    PIIDeidentifyRequest,
    PIIExtractRequest,
    PIIExtractStreamRequest,
    PrivacyGatewayRequest,
    ProfileRequest,
    SMARTBackendIngestionRequest,
    SystemOneRequest,
)
from .security_headers import (
    SERVICE_CORS_EXPOSE_HEADERS,
    SERVICE_CORS_METHODS,
    ErrorEnvelopeTrustedHostMiddleware,
    parse_service_security_config,
    service_cors_allow_headers,
)
from .security_profile import (
    SECURITY_PROFILE_INVALID_CODE,
    SecurityProfileReport,
    evaluate_service_security_profile,
    log_security_profile_report,
)
from .smart_backend import (
    SMARTBackendConfig,
    SMARTBackendJobManager,
    authorize_smart_backend_config,
)
from .streaming import (
    STREAM_ROUTE_PATHS,
    NDJSONStreamControls,
    PIIDeidentifyStreamRequest,
    StreamWriteIdleMiddleware,
    deidentify_ndjson_stream,
    pii_extract_ndjson_stream,
)
from .throttle import (
    ServiceBusyError,
    ServiceThrottle,
    current_request_work,
    format_retry_after,
)
from .tracing import (
    OpenTelemetryMiddleware,
    operational_trace_attributes,
    result_summary_attributes,
    service_tracing_from_env,
    set_current_span_attributes,
    trace_service_stage,
)
from .warm_pool import WarmPoolBackpressureError

SERVICE_NAME = "openmed-rest"
REQUEST_PRIORITY_HEADER = "x-openmed-priority"
_PRIVACY_GATEWAY_PATH = "/privacy-gateway/complete"
_SMART_BACKEND_START_PATH = "/fhir/smart-backend/ingestions"
_FHIR_BULK_EXPORT_PATH = "/fhir/bulk/exports"
_FHIR_BULK_IMPORT_PATH = "/fhir/bulk/imports"
# Fixed route labels bound into Idempotency-Key scopes.
_SMART_BACKEND_START_ROUTE = "smart-backend-ingestion"
_FHIR_BULK_EXPORT_ROUTE = "fhir-bulk-export"
_FHIR_BULK_IMPORT_ROUTE = "fhir-bulk-import"
_IDEMPOTENCY_OPENAPI_EXTRA: Dict[str, Any] = {
    "parameters": [
        {
            "name": IDEMPOTENCY_KEY_HEADER,
            "in": "header",
            "required": False,
            "description": (
                "Optional client key that makes retries of this start return the "
                "original job instead of starting another one. Scoped to the "
                "authenticated principal; reusing it with a different request "
                "body returns 409 idempotency_key_conflict."
            ),
            "schema": {
                "type": "string",
                "minLength": 1,
                "maxLength": IDEMPOTENCY_KEY_MAX_LENGTH,
                "pattern": "^[A-Za-z0-9._:~+/=-]+$",
            },
        }
    ]
}
_IDEMPOTENT_START_RESPONSES: Dict[int | str, Dict[str, Any]] = {
    200: {
        "description": "Ingestion job started, or the replayed original start.",
        "headers": {
            IDEMPOTENCY_REPLAYED_HEADER: {
                "description": (
                    "`true` when the response replays the start that first used "
                    "this Idempotency-Key."
                ),
                "schema": {"type": "string", "enum": ["true"]},
            }
        },
    },
    400: {"description": "Malformed Idempotency-Key (`idempotency_key_invalid`)."},
    409: {
        "description": (
            "Idempotency-Key reused for a different request "
            "(`idempotency_key_conflict`) or still being processed "
            "(`idempotency_key_in_progress`)."
        )
    },
}
_IDEMPOTENT_JOB_RESPONSES: Dict[int | str, Dict[str, Any]] = {
    202: {
        "description": "Job accepted, or the replayed original submission.",
        "headers": {
            IDEMPOTENCY_REPLAYED_HEADER: {
                "description": (
                    "`true` when the response returns the job that first used "
                    "this Idempotency-Key."
                ),
                "schema": {"type": "string", "enum": ["true"]},
            }
        },
    },
    400: {"description": "Malformed Idempotency-Key (`idempotency_key_invalid`)."},
    409: {
        "description": (
            "Idempotency-Key reused for a different request "
            "(`idempotency_key_conflict`)."
        )
    },
}
_DECISION_PATH = "/v1/decisions"
_MULTIMODAL_PREFLIGHT_PATH = "/experimental/multimodal/preflight"
_MULTIMODAL_PREFLIGHT_ENV = "OPENMED_SERVICE_EXPERIMENTAL_MULTIMODAL_PREFLIGHT"
_MANIFEST_HEADER = "X-OpenMed-Manifest"
_MAX_MANIFEST_HEADER_BYTES = 8192
_SYSTEM_ONE_PATH = "/v1/systemone"
_COMPATIBILITY_PATH = "/v1/compatibility"
_MODEL_BACKED_PATHS = frozenset(
    path
    for (_, path), policy in ROUTE_POLICIES.items()
    if policy.admission in {"model", "heavy"}
)
_ServicePayload = Dict[str, Any]
_ServiceOperation = Callable[[], Awaitable[_ServicePayload]]
_AnalyzeBatcher = DynamicBatcher["_AnalyzeBatchJob", _ServicePayload]
_PIIExtractBatcher = DynamicBatcher["_PIIExtractBatchJob", _ServicePayload]
_GROUNDING_CACHE_ENV_VAR = "OPENMED_GROUNDING_CACHE_DIR"
_REQUEST_BODY_ENCODING_MULTIPLIER = 12
_REQUEST_BODY_OVERHEAD_BYTES = 65_536


@dataclass(frozen=True)
class _AnalyzeBatchJob:
    payload: AnalyzeRequest


@dataclass(frozen=True)
class _PIIExtractBatchJob:
    payload: PIIExtractRequest


_SERVICE_WORK_TASKS: set[asyncio.Future[Any]] = set()


class ServiceTimeoutError(RuntimeError):
    """Raised when a request exceeds the configured service timeout."""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = float(timeout_seconds)
        super().__init__(
            f"Request exceeded configured timeout of {self.timeout_seconds:g} seconds"
        )


class _DecisionShapeRefused(Exception):
    """A ``POST /v1/decisions`` body without exactly one decision shape."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


async def _negotiate_decision_body(request: Request) -> None:
    """Refuse mixed, shapeless or media-carrying OpenAI-compatible bodies.

    FastAPI parses the JSON body before it resolves dependencies, so this runs
    on a decoded body and before either request model validates it. Malformed
    JSON and non-object bodies are left to the standard validation error.
    """

    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        return
    if not isinstance(body, Mapping):
        return
    try:
        decision_request_shape(body)
    except DecisionRequestShapeError as exc:
        raise _DecisionShapeRefused(exc.reason_code) from None


def _decide_with_service_state(
    state: Any,
    decision_request: CoreDecisionRequest,
) -> DecisionResult:
    """Decide with the service's configured backend, policy and profiles."""

    return decide(
        decision_request,
        backend=getattr(state, "decision_backend", DETERMINISTIC_DECISION_BACKEND),
        policy=getattr(state, "decision_access_policy", DecisionAccessPolicy()),
        calibration_profiles=getattr(state, "decision_calibration_profiles", None),
    )


async def _openai_compatible_decision(
    payload: OpenAIDecisionsRequest,
    request: Request,
) -> Any:
    """Decide an OpenAI-compatible body and answer in the same shape."""

    body = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    try:
        decision_request = from_openai_decisions_request(body)
    except (DecisionError, TypeError, ValueError) as exc:
        # Decision errors carry static, value-free messages; others may not.
        reason = str(exc) if isinstance(exc, DecisionError) else "invalid_arguments"
        raise InputError(
            "Invalid OpenAI-compatible decision request.",
            details={"reason": reason},
        ) from exc
    backend = getattr(
        request.app.state,
        "decision_backend",
        DETERMINISTIC_DECISION_BACKEND,
    )
    identity = getattr(backend, "identity", None)
    if body["model"] != getattr(identity, "model_id", None):
        # Never answer with a different model than the caller asked for.
        return _error_response(
            422,
            "model_not_served",
            "Requested model is not served",
        )
    result = await run_in_threadpool(
        _decide_with_service_state,
        request.app.state,
        decision_request,
    )
    return to_openai_decisions_response(
        decision_request,
        result,
        include_openmed=True,
    )


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert an OpenMed result object to a JSON-serializable mapping."""
    if hasattr(result, "to_dict") and callable(result.to_dict):
        payload = result.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
        raise InternalError(
            "An OpenMed result returned a non-mapping payload. Stop processing "
            "and report this serialization invariant failure."
        )

    if isinstance(result, Mapping):
        return dict(result)

    raise InternalError(
        "An OpenMed operation returned an unsupported result type. Stop "
        "processing and report this serialization invariant failure."
    )


def _parse_grounded_jsonl_text(text: str) -> list[Mapping[str, Any]]:
    """Parse newline-delimited grounded note records without echoing content.

    Errors reference only the offending line number so no note text or other
    caller-supplied content leaks into the PHI-free error envelope.
    """
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise InputError(
                f"records_jsonl line {line_number} is not valid JSON. Correct "
                "that synthetic record before retrying.",
                code="bad_request",
                details={"argument": "records_jsonl", "line": line_number},
            ) from exc
        if not isinstance(payload, Mapping):
            raise InputError(
                f"records_jsonl line {line_number} must be a JSON object. Replace "
                "the record with a key-value mapping before retrying.",
                code="bad_request",
                details={"argument": "records_jsonl", "line": line_number},
            )
        records.append(payload)
    if not records:
        raise InputError(
            "records_jsonl must contain at least one record. Add a JSON object "
            "on its own line before retrying.",
            code="bad_request",
            details={"argument": "records_jsonl"},
        )
    return records


def _omop_load_summary(payload: OmopLoadRequest) -> Dict[str, Any]:
    """Build a PHI-free OMOP load summary from an inline JSONL request body."""
    from ..interop.omop import load_grounded_notes, validate_omop_tables

    records = _parse_grounded_jsonl_text(payload.records_jsonl)
    tables = load_grounded_notes(
        records,
        vocabulary_version=payload.vocabulary_version,
        completeness_floor=payload.completeness_floor,
        required_fields=payload.required_fields,
    )
    summary = tables.summary
    response: Dict[str, Any] = {
        "row_counts": dict(summary.row_counts),
        "rejection_counts": dict(summary.rejection_counts),
        "rejected_spans": [span.to_dict() for span in summary.rejected_spans],
        "source_note_hashes": list(summary.source_note_hashes),
    }
    if payload.validate_constraints:
        violations = validate_omop_tables(tables)
        by_reason: Dict[str, int] = {}
        for violation in violations:
            by_reason[violation.reason] = by_reason.get(violation.reason, 0) + 1
        response["constraint_violations"] = {
            "count": len(violations),
            "by_reason": by_reason,
        }
    return response


def _profile_summary(payload: ProfileRequest) -> Dict[str, Any]:
    """Build the deterministic PHI-free quality profile for inline JSONL."""

    from ..structured.quality import profile_results

    records = _parse_grounded_jsonl_text(payload.records_jsonl)
    return profile_results(
        records,
        completeness_floor=payload.completeness_floor,
        required_fields=payload.required_fields,
        athena_index=payload.athena_index,
    ).to_dict()


def _ground_summary(payload: GroundRequest) -> Dict[str, Any]:
    """Build the shared offline grounding response without logging inputs."""

    from openmed.clinical.grounding import (
        RankingConfig,
        VocabLoader,
        ground_payload,
    )

    loader = VocabLoader(
        cache_dir=os.getenv(_GROUNDING_CACHE_ENV_VAR),
        local_only=payload.offline,
    )
    spans: Any = payload.entities if payload.entities is not None else payload.text
    response = ground_payload(
        spans,
        systems=payload.systems,
        loader=loader,
        config=RankingConfig(k=payload.top_k),
        source_language=payload.lang,
        offline=payload.offline,
    )
    # Calibration remains a library-only extension until the versioned REST
    # response contract explicitly includes these fields.
    for result in response["results"]:
        result.pop("calibrated_confidence", None)
        result.pop("confidence_band", None)
    return response


def _cohort_resolve_summary(payload: CohortResolveRequest) -> Dict[str, Any]:
    """Resolve an inline grounded corpus without persisting request content."""

    from ..interop.omop import load_grounded_notes, write_omop_duckdb
    from ..structured.cohort import (
        ConceptHierarchy,
        PhenotypeDefinition,
        resolve_phenotype,
    )

    definition = PhenotypeDefinition.from_dict(payload.phenotype)
    records = _parse_grounded_jsonl_text(payload.records_jsonl)
    tables = load_grounded_notes(
        records,
        completeness_floor=payload.completeness_floor,
        required_fields=payload.required_fields,
    )
    hierarchy_rows = [
        {
            "ancestor_concept_id": edge.ancestor_concept_id,
            "descendant_concept_id": edge.descendant_concept_id,
        }
        for edge in payload.concept_ancestors
    ]
    hierarchy = None
    if hierarchy_rows:
        canonical_rows = json.dumps(
            hierarchy_rows,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        hierarchy = ConceptHierarchy.from_rows(
            hierarchy_rows,
            source_sha256=hashlib.sha256(canonical_rows).hexdigest(),
        )
    connection = write_omop_duckdb(tables)
    try:
        return resolve_phenotype(
            definition,
            connection,
            hierarchy=hierarchy,
        ).to_dict()
    finally:
        connection.close()


def _error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    details: Optional[Any] = None,
    headers: Optional[Mapping[str, str]] = None,
) -> JSONResponse:
    """Return a standardized API error response."""
    error = {
        "code": code,
        "message": message,
        "details": details,
    }
    request_id = current_request_id()
    if request_id is not None:
        error["request_id"] = request_id
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={"error": error},
    )


def _model_route_body_rejection(code: str) -> JSONResponse:
    """Return the model-route envelope for a request-body bound rejection."""

    if code == REQUEST_SIZE_INVALID:
        return _error_response(400, "bad_request", "Request metadata is invalid")
    if code == REQUEST_BODY_TIMEOUT:
        return _error_response(
            408,
            "request_body_timeout",
            "Request body was not received within the configured time limit",
        )
    return _error_response(
        413,
        "payload_too_large",
        "Request body exceeds the configured limit",
    )


_TAXONOMY_HTTP_STATUS: Dict[type[OpenMedError], int] = {
    InputError: 400,
    ConfigurationError: 400,
    PolicyError: 400,
    CapabilityError: 503,
    BudgetExceededError: 503,
    InternalError: 500,
    OpenMedError: 500,
}


def _taxonomy_http_status(exc: OpenMedError) -> int:
    """Return the documented HTTP status for a taxonomy exception."""

    for error_class in type(exc).__mro__:
        status = _TAXONOMY_HTTP_STATUS.get(error_class)
        if status is not None:
            return status
    return 500


def _openmed_error_response(exc: OpenMedError) -> JSONResponse:
    """Return a stable, PHI-safe service envelope for a public API error."""

    status = _taxonomy_http_status(exc)
    details = dict(exc.details) if status < 500 else None
    return _error_response(status, exc.code, exc.message, details=details)


def _format_error_field(location: Any) -> str:
    if not isinstance(location, (list, tuple)):
        return str(location)

    parts = [str(part) for part in location if part != "__root__"]
    if not parts:
        return "body"
    return ".".join(parts)


def _format_validation_details(exc: RequestValidationError) -> list[Dict[str, str]]:
    """Normalize FastAPI/Pydantic validation errors for the API envelope."""
    details = []
    for error in exc.errors():
        details.append(
            {
                "field": _format_error_field(error.get("loc", ("body",))),
                "message": str(error.get("msg", "Invalid value")),
                "type": str(error.get("type", "value_error")),
            }
        )
    return details


def _attach_runtime(app: FastAPI, runtime: ServiceRuntime) -> None:
    """Persist runtime state on the FastAPI application object."""
    app.state.runtime = runtime
    app.state.profile = runtime.profile
    app.state.config = runtime.config
    app.state.batching = runtime.batching
    app.state.coalescing = runtime.coalescing
    app.state.paged_kv_cache = runtime.paged_kv_cache
    app.state.analyze_batcher = None
    app.state.pii_extract_batcher = None
    app.state.request_coalescer = None
    existing_job_queue = getattr(app.state, "job_queue", None)
    if _replica_state_policy(app).refusal_code(ASYNC_JOBS) is not None:
        # A disabled or replica-unsafe job store never loads local job state.
        if existing_job_queue is not None:
            existing_job_queue.shutdown()
        app.state.job_queue = None
    elif (
        existing_job_queue is None
        or getattr(existing_job_queue, "runtime", None) is not runtime
    ):
        if existing_job_queue is not None:
            existing_job_queue.shutdown()
        try:
            app.state.job_queue = DeidentifyJobQueue.from_env(
                runtime,
                key_provider=getattr(app.state, "job_payload_key_provider", None),
                digests=getattr(app.state, "job_digests", None),
            )
        except JobStoreSchemaError:
            # A store outside this build's window keeps the process up but not
            # ready: readiness reports it and job routes answer 503 until the
            # state is compatible, so a mixed-version rollout never crash-loops.
            app.state.job_queue = None
    if runtime.coalescing.enabled:
        app.state.request_coalescer = RequestCoalescer()
    if runtime.batching.enabled:
        app.state.analyze_batcher = DynamicBatcher(
            lambda jobs: _dispatch_analyze_batch(runtime, jobs),
            max_batch_size=runtime.batching.max_batch_size,
            max_wait_ms=runtime.batching.max_wait_ms,
            max_queue_size_per_priority=runtime.batching.max_queue_size,
            high_watermark=runtime.batching.high_watermark,
            low_watermark=runtime.batching.low_watermark,
            max_queue_wait_ms=runtime.batching.max_queue_wait_ms,
            queue_name="analyze",
            metrics=runtime.metrics,
        )
        app.state.pii_extract_batcher = DynamicBatcher(
            lambda jobs: _dispatch_pii_extract_batch(runtime, jobs),
            max_batch_size=runtime.batching.max_batch_size,
            max_wait_ms=runtime.batching.max_wait_ms,
            max_queue_size_per_priority=runtime.batching.max_queue_size,
            high_watermark=runtime.batching.high_watermark,
            low_watermark=runtime.batching.low_watermark,
            max_queue_wait_ms=runtime.batching.max_queue_wait_ms,
            queue_name="pii_extract",
            metrics=runtime.metrics,
        )
    app.state.throttle = ServiceThrottle(
        runtime.throttle,
        error_response=_error_response,
        backpressure_check=(
            _batch_backpressure_check if runtime.batching.enabled else None
        ),
        limited_paths=_MODEL_BACKED_PATHS,
    )
    app.state.admission = HttpAdmission(
        runtime.admission_controller,
        error_response=_error_response,
    )


def _get_service_runtime(request: Request) -> ServiceRuntime:
    """Return the initialized service runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        runtime = ServiceRuntime.from_env(
            metrics=getattr(request.app.state, "metrics", None)
        )
        _attach_runtime(request.app, runtime)
    return runtime


def _replica_state_policy(app: FastAPI) -> ReplicaStatePolicy:
    """Return the replica policy parsed when the app was created."""
    policy = getattr(app.state, "replica_state", None)
    if policy is None:
        policy = ReplicaStatePolicy.from_env()
        app.state.replica_state = policy
    return policy


def replica_state_report(app: FastAPI) -> ReplicaStateReport:
    """Evaluate the value-free replica-local state report for one app.

    The report lists every replica-local component category with its scope,
    multi-replica safety class, and evaluated state. Readiness uses the same
    report, so a preflight check that calls this function sees the decision
    ``/readyz`` makes. Before the lifespan attaches a runtime, limiter and
    coalescer settings are read from the environment.

    Args:
        app: Application returned by :func:`create_app`.

    Returns:
        Report naming component categories only, never paths, hosts, or ids.
    """
    runtime = getattr(app.state, "runtime", None)
    throttle = (
        runtime.throttle if runtime is not None else parse_service_throttle_config()
    )
    coalescing = (
        runtime.coalescing if runtime is not None else parse_service_coalescing_config()
    )
    auth = getattr(app.state, "auth", None)
    auth_config = getattr(auth, "config", None)
    return _replica_state_policy(app).evaluate(
        {
            ASYNC_JOBS: True,
            FHIR_BULK_JOBS: True,
            SMART_BACKEND_JOBS: True,
            RATE_LIMITER: throttle.rate_limit_enabled,
            CONCURRENCY_LIMITER: throttle.concurrency_enabled,
            AUTH_FAILURE_LIMITER: bool(
                auth_config is not None
                and auth_config.enabled
                and auth_config.failure_rate_limit_enabled
            ),
            REQUEST_COALESCER: coalescing.enabled,
        }
    )


def _replica_component_gate(category: str) -> Callable[[Request], Awaitable[None]]:
    """Return a route dependency that refuses a gated replica-local component."""

    async def _require_replica_component(request: Request) -> None:
        _replica_state_policy(request.app).require_available(category)

    return _require_replica_component


async def _run_with_timeout(
    runtime: ServiceRuntime,
    operation: Any,
) -> Any:
    """Run blocking service work in a threadpool under the profile timeout."""
    work = current_request_work()

    async def model_work() -> Any:
        release_slot = await work.acquire_model_slot() if work is not None else None
        try:
            return await run_in_threadpool(operation)
        finally:
            if release_slot is not None:
                release_slot()

    return await _await_with_timeout(runtime, model_work())


def _track_service_work(awaitable: Any) -> asyncio.Future[Any]:
    """Keep a strong reference and consume late failures without logging input."""
    task = asyncio.ensure_future(awaitable)
    _SERVICE_WORK_TASKS.add(task)

    def completed(future: asyncio.Future[Any]) -> None:
        _SERVICE_WORK_TASKS.discard(future)
        if not future.cancelled():
            future.exception()

    task.add_done_callback(completed)
    work = current_request_work()
    if work is not None:
        work.track(task)
    return task


async def _await_with_timeout(
    runtime: ServiceRuntime,
    awaitable: Any,
    *,
    cancel_on_timeout: bool = False,
) -> Any:
    """Bound the client wait without cancelling work that still owns capacity."""
    task = _track_service_work(awaitable)
    timeout_seconds = float(getattr(runtime.config, "timeout", 0) or 0)
    wait = task if cancel_on_timeout else asyncio.shield(task)
    if timeout_seconds <= 0:
        return await wait

    try:
        return await asyncio.wait_for(wait, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise ServiceTimeoutError(timeout_seconds) from exc


async def _submit_with_timeout(
    runtime: ServiceRuntime,
    batcher: DynamicBatcher,
    item: Any,
    *,
    priority: str,
    priority_handle: BatchPriorityHandle,
) -> Any:
    """Cancel abandoned queued jobs while retaining ownership of dispatches."""
    work_done = asyncio.get_running_loop().create_future()
    _track_service_work(work_done)
    started = False

    async def submit() -> Any:
        nonlocal started
        started = True
        return await batcher.submit(
            item,
            priority=priority,
            priority_handle=priority_handle,
            work_done=work_done,
        )

    task = asyncio.create_task(submit())

    def completed_before_start(future: asyncio.Future[Any]) -> None:
        if not started and not work_done.done():
            work_done.set_result(None)

    task.add_done_callback(completed_before_start)
    return await _await_with_timeout(
        runtime,
        task,
        cancel_on_timeout=True,
    )


def _get_analyze_batcher(request: Request) -> _AnalyzeBatcher:
    batcher = getattr(request.app.state, "analyze_batcher", None)
    if batcher is None:
        raise InternalError(
            "The analyze batcher is not initialized. Restart the service and "
            "report the startup invariant if it recurs."
        )
    return batcher


def _get_pii_extract_batcher(request: Request) -> _PIIExtractBatcher:
    batcher = getattr(request.app.state, "pii_extract_batcher", None)
    if batcher is None:
        raise InternalError(
            "The PII extraction batcher is not initialized. Restart the service "
            "and report the startup invariant if it recurs."
        )
    return batcher


def _get_smart_backend_manager(request: Request) -> SMARTBackendJobManager:
    manager = getattr(request.app.state, "smart_backend_jobs", None)
    if manager is None:
        manager = SMARTBackendJobManager(
            bounds=getattr(request.app.state, "bulk_job_bounds", None),
            idempotency=StartIdempotencyStore.from_env(),
        )
        request.app.state.smart_backend_jobs = manager
    return manager


def _get_fhir_bulk_manager(request: Request) -> FHIRBulkJobManager:
    manager = getattr(request.app.state, "fhir_bulk_jobs", None)
    if manager is None:
        manager = FHIRBulkJobManager(
            bounds=getattr(request.app.state, "bulk_job_bounds", None),
            idempotency=StartIdempotencyStore.from_env(),
        )
        request.app.state.fhir_bulk_jobs = manager
    return manager


def _get_job_queue(request: Request) -> DeidentifyJobQueue:
    queue = getattr(request.app.state, "job_queue", None)
    if queue is None:
        runtime = _get_service_runtime(request)
        queue = DeidentifyJobQueue.from_env(
            runtime,
            key_provider=getattr(request.app.state, "job_payload_key_provider", None),
            digests=getattr(request.app.state, "job_digests", None),
        )
        request.app.state.job_queue = queue
    return queue


def _get_job_ownership(request: Request) -> JobOwnershipPolicy:
    policy = getattr(request.app.state, "job_ownership", None)
    if policy is None:
        raise InternalError(
            "The job ownership policy is not initialized. Restart the service "
            "and report the startup invariant if it recurs."
        )
    return policy


def _submitting_owner_digest(request: Request) -> Optional[dict[str, str]]:
    """Return the keyed digest of the authenticated submitter, if any."""
    return _get_job_ownership(request).owner_digest(request.scope.get("openmed.auth"))


def _keyed_start_request(
    request: Request,
    payload: Any,
    *,
    route: str,
    owner_digest: Optional[dict[str, str]],
) -> Optional[IdempotentStartRequest]:
    """Return the digest-only identity of a keyed start, if a key was sent.

    The scope binds the route label, the submitter's owner digest and the key;
    the request digest covers every validated field except credentials. Both
    are keyed HMAC digests, so neither the key nor any field is retained.
    """
    key = parse_idempotency_key(request.headers.getlist(IDEMPOTENCY_KEY_HEADER))
    if key is None:
        return None
    fields = (
        payload.model_dump(mode="json")
        if hasattr(payload, "model_dump")
        else payload.dict()
    )
    return IdempotentStartRequest.derive(
        _get_job_ownership(request).key,
        route=route,
        owner_digest=owner_digest,
        idempotency_key=key,
        request_fields=fields,
    )


def _job_visible(
    request: Request,
    *,
    admin_scope: str,
    owner_digest: Callable[[], Any],
) -> bool:
    """Return whether the caller may read or cancel a job record.

    The owner lookup is deferred so auth-disabled requests never consult it.
    Callers answer ``False`` exactly like an unknown job id, so other
    principals cannot probe which ids exist.
    """
    policy = _get_job_ownership(request)
    principal = request.scope.get("openmed.auth")
    if not isinstance(principal, AuthPrincipal):
        return policy.permits(None, None, admin_scope=admin_scope)
    return policy.permits(principal, owner_digest(), admin_scope=admin_scope)


def _require_visible_bulk_job(
    request: Request,
    manager: FHIRBulkJobManager | SMARTBackendJobManager,
    job_id: str,
) -> None:
    """Raise ``KeyError`` for an unknown job or one the caller does not own."""
    if not _job_visible(
        request,
        admin_scope=BULK_ADMIN_SCOPE,
        owner_digest=lambda: manager.owner_digest(job_id),
    ):
        raise KeyError(job_id)


def _get_journey_resource_catalog(request: Request) -> JourneyResourceCatalog:
    catalog = getattr(request.app.state, "journey_resources", None)
    if catalog is None:
        catalog = JourneyResourceCatalog()
        request.app.state.journey_resources = catalog
    return catalog


def _get_journey_access_gate(request: Request) -> JourneyAccessGate:
    gate = getattr(request.app.state, "journey_access", None)
    if not isinstance(gate, JourneyAccessGate):
        gate = JourneyAccessGate.from_env()
        request.app.state.journey_access = gate
    return gate


def _create_compatibility_gate(
    app: FastAPI, runtime: ServiceRuntime
) -> CompatibilityGate:
    """Gate readiness on the persisted state this process shares with peers."""

    def job_store_probe() -> SchemaObservation:
        queue = getattr(app.state, "job_queue", None)
        if queue is not None:
            return job_store_observation(queue.store.path)
        if _replica_state_policy(app).refusal_code(ASYNC_JOBS) is not None:
            # Async jobs are off on this replica, so no job store is shared.
            return SchemaObservation.absent()
        # No queue yet, for example because the store was refused on open.
        return job_store_observation(
            parse_jobs_store_path(os.getenv(SERVICE_JOBS_STORE_ENV_VAR))
        )

    return CompatibilityGate(
        local_descriptor(runtime),
        {JOB_STORE_COMPONENT: job_store_probe},
    )


def _get_compatibility_gate(request: Request) -> CompatibilityGate:
    gate = getattr(request.app.state, "compatibility", None)
    if gate is None:
        gate = _create_compatibility_gate(request.app, _get_service_runtime(request))
        request.app.state.compatibility = gate
    return gate


def _incompatible_state_response(blocking: Mapping[str, Any]) -> JSONResponse:
    return _error_response(
        503,
        "incompatible_state",
        "Persisted state is outside this build's compatibility window",
        details={
            "components": {
                name: {"state": item.state.value, "reason": item.code}
                for name, item in sorted(blocking.items())
            }
        },
    )


def _peer_contract_rejection(request: Request) -> Optional[JSONResponse]:
    """Refuse an announced peer outside the window before any processing."""
    values = request.headers.getlist(PEER_CONTRACT_HEADER)
    if not values or request.url.path == _COMPATIBILITY_PATH:
        return None
    policy = ROUTE_POLICIES.get(route_key_for_request(request))
    if policy is None or policy.admission == "probe":
        return None
    decision = peer_contract_decision(CURRENT_CONTRACT, values)
    if decision.compatible:
        return None
    if decision.code == "peer_contract_malformed":
        return _error_response(
            400,
            "peer_contract_malformed",
            "Peer contract header is malformed",
            details=None,
        )
    return _error_response(
        409,
        "peer_incompatible",
        "Peer build is outside this build's compatibility window",
        details={
            "reason": decision.code,
            "peer_contract": CURRENT_CONTRACT.peer_contract,
        },
    )


async def _run_maybe_coalesced(
    request: Request,
    endpoint: str,
    payload: Any,
    operation: _ServiceOperation,
    *,
    priority: str,
    on_priority_upgrade: Optional[Callable[[str], Any]] = None,
) -> _ServicePayload:
    coalescer = getattr(request.app.state, "request_coalescer", None)
    if coalescer is None:
        return await operation()

    with trace_service_stage(
        "request_coalescing",
        {
            "openmed.endpoint": endpoint,
            "openmed.coalescing.enabled": True,
        },
    ):
        return await asyncio.shield(
            _track_service_work(
                coalescer.run(
                    coalescing_key(endpoint, payload),
                    operation,
                    priority=priority,
                    on_priority_upgrade=on_priority_upgrade,
                )
            )
        )


def _request_priority(request: Request) -> str:
    return normalize_priority(request.headers.get(REQUEST_PRIORITY_HEADER))


def _backpressure_response(exc: BackpressureError) -> JSONResponse:
    response = _error_response(
        503,
        "backpressure",
        str(exc),
        details=exc.to_details(),
    )
    response.headers["Retry-After"] = format_retry_after(exc.retry_after_seconds)
    return response


async def _batch_backpressure_check(request: Request) -> Optional[BackpressureError]:
    batcher: Any
    if request.url.path == "/analyze":
        batcher = getattr(request.app.state, "analyze_batcher", None)
    elif request.url.path == "/pii/extract":
        batcher = getattr(request.app.state, "pii_extract_batcher", None)
    else:
        return None

    if batcher is None:
        return None
    return await batcher.admission_error(_request_priority(request))


def _shutting_down_response(retry_after_seconds: float) -> JSONResponse:
    """Return the stable, retryable REST refusal used once admission closes."""
    response = _error_response(
        503,
        "not_ready",
        "Service is shutting down and not accepting new requests",
        details=shutting_down_details(retry_after_seconds),
    )
    response.headers["Retry-After"] = format_retry_after(retry_after_seconds)
    return response


def _drain_retry_after(state: Any) -> float:
    coordinator = getattr(state, "drain", None)
    return float(getattr(coordinator, "retry_after_seconds", 1.0))


def _drain_aware_stream(
    request: Request,
    events: Any,
    *,
    retain: Optional[Callable[[asyncio.Future[Any]], None]] = None,
) -> Any:
    """Give an NDJSON body a bounded tail and a typed drain terminal event.

    ``retain`` keeps a read the client abandoned under the request's admission
    while it settles, as the bounded stream controls do for model windows.
    """
    streams = getattr(request.app.state, "drain_streams", None)
    if not isinstance(streams, NDJSONStreamDrainParticipant):
        return events
    return streams.wrap(events, retain=retain)


async def _stop_openhim_mediator(state: Any) -> None:
    mediator = getattr(state, "openhim_mediator", None)
    if mediator is not None:
        await mediator.stop()


def _shutdown_tracing(state: Any) -> None:
    tracing = getattr(state, "tracing", None)
    if tracing is not None:
        tracing.shutdown()


def _install_drain_coordinator(
    app: FastAPI, runtime: ServiceRuntime
) -> DrainCoordinator:
    """Register REST lifecycle participants on the runtime's coordinator.

    Participants look state up when the drain runs, so batchers, queues, and
    managers swapped in after start-up are still drained. Participants added
    with ``register_drain_participant(app, ...)`` join every lifespan.
    """
    state = app.state
    coordinator = runtime.get_drain_coordinator()
    coordinator.on_readiness_lost(lambda: setattr(state, "ready", False))
    streams = NDJSONStreamDrainParticipant(
        retry_after_seconds=coordinator.retry_after_seconds
    )
    state.drain = coordinator
    state.drain_streams = streams
    state.drain_report = None
    for participant in (
        HTTPRequestDrainParticipant(state),
        CallbackDrainParticipant(
            "openhim",
            priority=15,
            stop_admission=lambda _ctx: _stop_openhim_mediator(state),
        ),
        streams,
        BatcherDrainParticipant(
            lambda: (
                getattr(state, "analyze_batcher", None),
                getattr(state, "pii_extract_batcher", None),
            )
        ),
        JobQueueDrainParticipant(lambda: getattr(state, "job_queue", None)),
        BackgroundJobsDrainParticipant(
            lambda: getattr(state, "smart_backend_jobs", None),
            category="smart_jobs",
        ),
        BackgroundJobsDrainParticipant(
            lambda: getattr(state, "fhir_bulk_jobs", None),
            category="bulk_jobs",
        ),
        CallbackDrainParticipant(
            "tracing", priority=70, flush=lambda _ctx: _shutdown_tracing(state)
        ),
        *tuple(getattr(state, "drain_participants", None) or ()),
    ):
        coordinator.register(participant)
    return coordinator


def _metrics_route_label(request: Request) -> str:
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path
    return "unknown"


def _security_profile_report(fastapi_app: FastAPI) -> SecurityProfileReport:
    """Return the app's security-profile report, evaluating it on first use."""
    report = getattr(fastapi_app.state, "security_profile_report", None)
    if report is None:
        report = evaluate_service_security_profile(fastapi_app)
        fastapi_app.state.security_profile_report = report
        log_security_profile_report(report)
    return report


def _security_profile_blocks(request: Request) -> bool:
    """Refuse every non-probe route while the security profile is invalid."""
    if _security_profile_report(request.app).ready:
        return False
    method, path = route_key_for_request(request)
    if not path:
        return False  # No route matches; the router answers 404 without work.
    policy = ROUTE_POLICIES.get((method, path))
    return policy is None or policy.admission != "probe"


def _security_profile_invalid_response(
    report: SecurityProfileReport,
    *,
    include_details: bool,
) -> JSONResponse:
    return _error_response(
        503,
        SECURITY_PROFILE_INVALID_CODE,
        "Service security profile checks failed; the service is not accepting "
        "requests. Run 'openmed service preflight' for the report.",
        details=report.readiness_details() if include_details else None,
    )


def create_app(
    *,
    max_request_body_bytes: Optional[int] = None,
    experimental_multimodal_preflight: Optional[bool] = None,
    workflow_service: Optional[WorkflowGovernanceService] = None,
    workflow_policy: Optional[WorkflowHTTPPolicy] = None,
    workflow_clock: Optional[Callable[[], int]] = None,
    journey_access_context_provider: Optional[JourneyAccessContextProvider] = None,
    job_payload_key_provider: Optional[KeyProvider] = None,
) -> FastAPI:
    """Create the REST app, optionally enabling single-asset preflight.

    Args:
        max_request_body_bytes: Override the route-specific body ceiling.
        experimental_multimodal_preflight: Explicit experimental opt-in. When
            omitted, read OPENMED_SERVICE_EXPERIMENTAL_MULTIMODAL_PREFLIGHT;
            unset or false keeps the route absent.
        workflow_service: Trusted local custody service for workflow routes.
        workflow_policy: Server-owned workflow opt-ins; disabled by default.
        workflow_clock: Integer Unix-seconds source for receipt validity checks.
        journey_access_context_provider: Operator hook that resolves the
            authoritative Journey access context for authenticated principals.
            Defaults to the grants in ``OPENMED_SERVICE_JOURNEY_GRANTS``.
        job_payload_key_provider: Caller-owned key provider that enables
            encrypted retention of async job payloads. Without it, job payloads
            stay in memory only.

    Returns:
        Configured application, without starting its lifespan or an effect executor.
    """

    if experimental_multimodal_preflight is None:
        flag = os.getenv(_MULTIMODAL_PREFLIGHT_ENV, "0").strip().lower()
        if flag not in {"", "0", "false", "1", "true"}:
            raise ValueError("Experimental multimodal preflight flag is invalid")
        experimental_multimodal_preflight = flag in {"1", "true"}

    openhim_settings = OpenHIMMediatorSettings.from_env()

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        import logging

        from openmed.core.settings_registry import service_setting_findings

        for finding in service_setting_findings(os.environ):
            logging.getLogger(__name__).warning(
                "Service setting %s: %s", finding["name"], finding["code"]
            )
        validate_route_policies(fastapi_app)
        validate_egress_configuration()
        runtime = ServiceRuntime.from_env(
            metrics=getattr(fastapi_app.state, "metrics", None)
        )
        _attach_runtime(fastapi_app, runtime)
        fastapi_app.state.compatibility = _create_compatibility_gate(
            fastapi_app, runtime
        )
        await run_in_threadpool(fastapi_app.state.compatibility.report, force=True)
        bulk_job_bounds = getattr(fastapi_app.state, "bulk_job_bounds", None)
        if getattr(fastapi_app.state, "smart_backend_jobs", None) is None:
            fastapi_app.state.smart_backend_jobs = SMARTBackendJobManager(
                bounds=bulk_job_bounds,
                idempotency=StartIdempotencyStore.from_env(),
            )
        if getattr(fastapi_app.state, "fhir_bulk_jobs", None) is None:
            fastapi_app.state.fhir_bulk_jobs = FHIRBulkJobManager(
                bounds=bulk_job_bounds,
                idempotency=StartIdempotencyStore.from_env(),
            )
        fastapi_app.state.ready = False
        fastapi_app.state.shutting_down = False
        fastapi_app.state.inflight = 0
        fastapi_app.state.openhim_mediator = None
        coordinator = _install_drain_coordinator(fastapi_app, runtime)

        # Fail closed: an invalid profile never preloads, contacts OpenHIM or
        # becomes ready, and every non-probe route answers 503.
        fastapi_app.state.security_profile_report = None
        mtls = getattr(fastapi_app.state, "mtls", None)
        if _security_profile_report(fastapi_app).ready:
            if runtime.preload_models:
                await run_in_threadpool(runtime.preload)

            if openhim_settings.enabled:
                mediator = create_openhim_mediator(openhim_settings)
                fastapi_app.state.openhim_mediator = mediator
                try:
                    await mediator.start()
                except Exception:
                    await mediator.stop()
                    raise

            if mtls is not None:
                mtls.start_reload_watcher()
            fastapi_app.state.ready = True

        try:
            yield
        finally:
            # The credential reload watcher stops before the drain so no reload
            # races shutdown. Readiness then drops; admission closes on every
            # surface, in-flight work finishes or ends with a typed result, and
            # models and devices are released, each phase within its budget.
            if mtls is not None:
                mtls.stop_reload_watcher()
            fastapi_app.state.drain_report = await coordinator.drain()

    api_docs = parse_api_docs_config()
    app = FastAPI(
        title="OpenMed REST API",
        version=openmed.__version__,
        description="Hardened REST API for OpenMed text analysis and PII workflows.",
        lifespan=lifespan,
        openapi_url=api_docs.openapi_url,
        docs_url=api_docs.docs_url,
        redoc_url=api_docs.redoc_url,
    )
    security_config = parse_service_security_config()
    app.state.security = security_config
    app.state.job_payload_key_provider = job_payload_key_provider
    app.state.bulk_storage = BulkStoragePolicy.from_env()
    app.state.bulk_job_bounds = bulk_job_bounds_from_env()
    auth_config = parse_service_auth_config()
    if api_docs.public:
        auth_config = replace(
            auth_config,
            exempt_paths=auth_config.exempt_paths | DEVELOPMENT_DOCS_EXEMPT_PATHS,
        )
    app.state.auth = ServiceAuth(auth_config, error_response=_error_response)
    app.state.mtls = ServiceMTLS(
        parse_service_mtls_config(),
        error_response=_error_response,
    )
    app.state.job_ownership = JobOwnershipPolicy.from_env(
        authentication_enabled=app.state.auth.enabled or app.state.mtls.enabled
    )
    # One deployment key serves owner, span, document and webhook URL digests.
    app.state.job_digests = JobDigestPolicy.from_env(key=app.state.job_ownership.key)
    app.state.metrics = (
        PrometheusMetricsRegistry() if metrics_enabled_from_env() else None
    )
    app.state.tracing = service_tracing_from_env()
    install_response_headers(app)
    app.state.openhim_settings = openhim_settings
    app.state.openhim_deidentifier = None
    app.state.journey_resources = JourneyResourceCatalog()
    app.state.journey_access_policy = JourneyAccessPolicy()
    app.state.journey_access = JourneyAccessGate.from_env(
        provider=journey_access_context_provider
    )
    app.state.operational_limits = request_limits_from_env()
    app.state.request_body_read_limits = request_body_read_limits_from_env()
    app.state.replica_state = ReplicaStatePolicy.from_env()
    app.state.system_one_enabled = os.getenv(
        "OPENMED_SERVICE_SYSTEM_ONE", ""
    ).strip().lower() in {"true", "1", "yes"}
    app.state.system_one_audit_key = os.urandom(32)
    app.state.decision_backend = DETERMINISTIC_DECISION_BACKEND
    app.state.decision_access_policy = DecisionAccessPolicy()
    app.state.decision_calibration_profiles = None

    if experimental_multimodal_preflight:

        @app.post(
            _MULTIMODAL_PREFLIGHT_PATH,
            tags=["Experimental"],
            summary="Experimental single-asset multimodal preflight",
            description=(
                "Disabled by default. Supply a strict AssetManifest JSON object "
                "in X-OpenMed-Manifest (at most 8192 bytes) and raw asset bytes "
                "as application/octet-stream. Returns the unchanged core "
                "PreflightReport JSON using desktop-v1 limits. No decoding, "
                "inference or clinical action is performed."
            ),
            openapi_extra={
                "x-experimental": True,
                "x-enabled-by-default": False,
                "parameters": [
                    {
                        "name": _MANIFEST_HEADER,
                        "in": "header",
                        "required": True,
                        "schema": {"type": "string", "maxLength": 8192},
                        "description": "Strict AssetManifest JSON; opaque references only",
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/octet-stream": {
                            "schema": {"type": "string", "format": "binary"}
                        }
                    },
                },
            },
            responses={
                200: {"description": "Byte-stable core PreflightReport JSON"},
                413: {"description": "Request body exceeds the configured ceiling"},
                415: {"description": "unsupported_media_type error envelope"},
                422: {"description": "malformed_manifest or digest mismatch envelope"},
            },
        )
        async def multimodal_preflight(request: Request) -> Response:
            """Validate one in-memory asset using the existing preflight contract."""
            from openmed.multimodal.asset_manifest import (
                AssetManifest,
                AssetManifestError,
            )
            from openmed.multimodal.preflight import preflight_asset

            def reject(status: int, code: str) -> JSONResponse:
                request.scope["openmed.preflight_outcome"] = "rejected"
                request.scope["openmed.preflight_reason_codes"] = (code,)
                return _error_response(status, code, "Multimodal preflight rejected")

            if (
                request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                != "application/octet-stream"
            ):
                return reject(415, "unsupported_media_type")
            headers = request.headers.getlist(_MANIFEST_HEADER)
            if (
                len(headers) != 1
                or len(headers[0].encode("utf-8")) > _MAX_MANIFEST_HEADER_BYTES
            ):
                return reject(422, "malformed_manifest")
            try:
                manifest = AssetManifest.from_json(headers[0])
            except (AssetManifestError, RecursionError):
                return reject(422, "malformed_manifest")
            report = await run_in_threadpool(
                preflight_asset, manifest, await request.body()
            )
            reasons = tuple(finding.reason_code for finding in report.findings)
            # Do not echo even a syntactically valid caller reference when
            # source integrity fails. Other abstentions preserve core parity.
            for code in ("byte_count_mismatch", "sha256_mismatch"):
                if code in reasons:
                    return reject(422, code)
            request.scope["openmed.preflight_outcome"] = report.status.value
            request.scope["openmed.preflight_reason_codes"] = reasons
            return Response(content=report.to_json(), media_type="application/json")

    @app.middleware("http")
    async def _readiness_middleware(request: Request, call_next):
        if _security_profile_blocks(request):
            return _security_profile_invalid_response(
                _security_profile_report(request.app), include_details=False
            )
        peer_rejection = _peer_contract_rejection(request)
        if peer_rejection is not None:
            return peer_rejection
        if request_needs_admission(request):
            state = request.app.state
            if getattr(state, "shutting_down", False):
                return _shutting_down_response(_drain_retry_after(state))
            state.inflight = getattr(state, "inflight", 0) + 1
            try:
                return await call_next(request)
            finally:

                def release_inflight() -> None:
                    state.inflight = getattr(state, "inflight", 0) - 1

                work = current_request_work()
                if work is None:
                    release_inflight()
                else:
                    work.on_finished(release_inflight)
        return await call_next(request)

    if app.state.metrics is not None:

        @app.middleware("http")
        async def _metrics_middleware(request: Request, call_next):
            if request.url.path == "/metrics":
                return await call_next(request)
            metrics = request.app.state.metrics
            metrics.request_started()
            start_time = time.perf_counter()
            status_code = 500
            try:
                response = await call_next(request)
                status_code = response.status_code
                return response
            finally:
                metrics.request_finished(
                    route=_metrics_route_label(request),
                    status_code=status_code,
                    duration_seconds=time.perf_counter() - start_time,
                )

        @app.get("/metrics", include_in_schema=False)
        async def metrics(request: Request) -> Response:
            registry = request.app.state.metrics
            runtime = _get_service_runtime(request)
            registry.set_circuit_breaker_state_counts(
                runtime.circuit_breaker_state_counts()
            )
            registry.set_connection_rejected_total(connection_rejection_count())
            return Response(
                content=registry.render(),
                media_type=PROMETHEUS_CONTENT_TYPE,
            )

    # Declared before the throttle so it runs inside it: rate limits shed first,
    # and the request's tracked work is visible when the permit is released.
    @app.middleware("http")
    async def _admission_middleware(request: Request, call_next):
        admission = getattr(request.app.state, "admission", None)
        if admission is None:
            return await call_next(request)
        return await admission.dispatch(request, call_next)

    @app.middleware("http")
    async def _throttle_middleware(request: Request, call_next):
        throttle = getattr(request.app.state, "throttle", None)
        if throttle is None:
            _get_service_runtime(request)
            throttle = getattr(request.app.state, "throttle", None)
        if throttle is None:
            return await call_next(request)
        return await throttle.dispatch(request, call_next)

    async def _auth_middleware(request: Request, call_next):
        auth = getattr(request.app.state, "auth", None)
        mtls = getattr(request.app.state, "mtls", None)

        async def dispatch_auth(auth_request: Request) -> Response:
            if auth is None:
                return await call_next(auth_request)
            return await auth.dispatch(auth_request, call_next)

        if mtls is None:
            return await dispatch_auth(request)
        return await mtls.dispatch(request, dispatch_auth)

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        # Question IDs and unknown JSON keys may themselves contain PHI. Use the
        # matched route so mounted services receive the same content-free error.
        protected_route = (
            getattr(request.scope.get("route"), "path", None) == _SYSTEM_ONE_PATH
        )
        return _error_response(
            422,
            "validation_error",
            "Request validation failed",
            details=None if protected_route else _format_validation_details(exc),
        )

    @app.exception_handler(ServiceTimeoutError)
    async def _timeout_handler(_: Request, exc: ServiceTimeoutError) -> JSONResponse:
        return _error_response(
            504,
            "timeout",
            str(exc),
            details={"timeout_seconds": exc.timeout_seconds},
        )

    @app.exception_handler(ReplicaComponentUnavailableError)
    async def _replica_component_handler(
        _: Request,
        exc: ReplicaComponentUnavailableError,
    ) -> JSONResponse:
        if exc.code == COMPONENT_DISABLED:
            return _error_response(
                503,
                "component_disabled",
                "This service component is disabled by the replica policy",
                details={"component": exc.category},
            )
        return _error_response(
            503,
            "replica_unsafe_state",
            "This component keeps replica-local state that multi-replica mode "
            "has not accepted",
            details={"component": exc.category, "replica_mode": "multi"},
        )

    @app.exception_handler(JobStoreSchemaError)
    async def _job_store_schema_handler(
        _: Request,
        exc: JobStoreSchemaError,
    ) -> JSONResponse:
        # A store outside this build's window is refused when it is opened.
        return _error_response(
            503,
            "incompatible_state",
            "Persisted state is outside this build's compatibility window",
            details={
                "components": {
                    JOB_STORE_COMPONENT: {
                        "state": "incompatible",
                        "reason": exc.compatibility_reason or exc.code,
                    }
                }
            },
        )

    @app.exception_handler(IncompatibleStateError)
    async def _incompatible_state_handler(
        _: Request,
        exc: IncompatibleStateError,
    ) -> JSONResponse:
        return _error_response(
            503,
            "incompatible_state",
            "Persisted state is outside this build's compatibility window",
            details=exc.to_details(),
        )

    @app.exception_handler(BackpressureError)
    async def _backpressure_handler(
        _: Request,
        exc: BackpressureError,
    ) -> JSONResponse:
        return _backpressure_response(exc)

    @app.exception_handler(BulkJobsSaturatedError)
    async def _bulk_jobs_saturated_handler(
        _: Request,
        exc: BulkJobsSaturatedError,
    ) -> JSONResponse:
        return _error_response(
            503,
            "bulk_jobs_saturated",
            "Bulk job capacity is saturated; retry later",
            details=exc.to_details(),
            headers={"Retry-After": format_retry_after(exc.retry_after_seconds)},
        )

    @app.exception_handler(ServiceBusyError)
    async def _service_busy_handler(_: Request, exc: ServiceBusyError) -> JSONResponse:
        return _error_response(
            503,
            "service_busy",
            str(exc),
            details={
                "max_concurrency": exc.max_concurrency,
                "wait_seconds": exc.wait_seconds,
                "key": exc.key_by,
            },
        )

    @app.exception_handler(EgressDenied)
    async def _egress_denied_handler(
        _: Request,
        exc: EgressDenied,
    ) -> JSONResponse:
        return _error_response(
            403,
            exc.error_code,
            "Outbound destination is not permitted by the egress policy",
            details={"reason": exc.reason_code},
        )

    @app.exception_handler(PrivacyGatewayConfigurationError)
    async def _privacy_gateway_config_handler(
        _: Request,
        exc: PrivacyGatewayConfigurationError,
    ) -> JSONResponse:
        return _error_response(
            503,
            exc.error_code,
            "Privacy gateway external endpoint is not configured",
            details={"reason": exc.reason_code},
        )

    @app.exception_handler(PrivacyTransportError)
    async def _privacy_gateway_transport_handler(
        _: Request,
        exc: PrivacyTransportError,
    ) -> JSONResponse:
        return _error_response(
            502,
            exc.error_code,
            "External LLM transport failed",
            details={"reason": exc.reason_code},
        )

    @app.exception_handler(PrivacyGatewayError)
    async def _privacy_gateway_error_handler(
        _: Request,
        exc: PrivacyGatewayError,
    ) -> JSONResponse:
        return _error_response(
            400,
            exc.error_code,
            str(exc),
            details={"reason": exc.reason_code},
        )

    @app.exception_handler(JobAdmissionError)
    async def _job_admission_handler(
        _: Request,
        exc: JobAdmissionError,
    ) -> JSONResponse:
        headers = None
        if exc.retry_after_seconds is not None:
            headers = {"Retry-After": format_retry_after(exc.retry_after_seconds)}
        return _error_response(
            exc.status_code,
            exc.error_code,
            exc.message,
            details=exc.to_details(),
            headers=headers,
        )

    @app.exception_handler(CircuitBreakerOpenError)
    async def _circuit_breaker_handler(
        _: Request,
        exc: CircuitBreakerOpenError,
    ) -> JSONResponse:
        retry_after = str(exc.retry_after_seconds)
        return _error_response(
            503,
            "circuit_breaker_open",
            "Model backend is temporarily unavailable",
            details=circuit_breaker_details(exc),
            headers={"Retry-After": retry_after},
        )

    @app.exception_handler(ServiceShuttingDownError)
    async def _service_shutting_down_handler(
        _: Request,
        exc: ServiceShuttingDownError,
    ) -> JSONResponse:
        return _shutting_down_response(exc.retry_after_seconds)

    @app.exception_handler(WarmPoolBackpressureError)
    async def _warm_pool_backpressure_handler(
        _: Request,
        exc: WarmPoolBackpressureError,
    ) -> JSONResponse:
        return _error_response(
            503,
            "service_busy",
            str(exc),
            details={
                "model_name": exc.model_name,
                "required_bytes": exc.required_bytes,
                "budget_bytes": exc.budget_bytes,
                "wait_seconds": exc.wait_seconds,
            },
        )

    @app.exception_handler(_DecisionShapeRefused)
    async def _decision_shape_handler(
        _: Request,
        exc: _DecisionShapeRefused,
    ) -> JSONResponse:
        details = {"reason": exc.reason_code}
        if exc.reason_code == "media_unsupported":
            return _error_response(
                422,
                "decision_media_unsupported",
                "OpenAI-compatible decision requests are text-only.",
                details=details,
            )
        return _error_response(
            422,
            "decision_request_ambiguous",
            "Send exactly one of an OpenMed decision request or an "
            "OpenAI-compatible Decisions body.",
            details=details,
        )

    @app.exception_handler(OpenMedError)
    async def _openmed_error_handler(
        _: Request,
        exc: OpenMedError,
    ) -> JSONResponse:
        return _openmed_error_response(exc)

    @app.exception_handler(IdempotencyError)
    async def _idempotency_error_handler(
        _: Request,
        exc: IdempotencyError,
    ) -> JSONResponse:
        headers = (
            None if exc.retry_after is None else {"Retry-After": str(exc.retry_after)}
        )
        return _error_response(
            exc.status_code, exc.code, str(exc), details=None, headers=headers
        )

    @app.exception_handler(ValueError)
    async def _value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        from .runtime import ModelNotServedError

        if isinstance(exc, ModelNotServedError):
            return _error_response(
                422,
                "model_not_served",
                "Requested model is not served",
                details=None,
            )
        if isinstance(exc, BulkPathError):
            return _error_response(
                400, exc.code, "Bulk storage request is not permitted", details=None
            )
        del exc
        return _error_response(
            400,
            "bad_request",
            "Request arguments are invalid. Correct the request and retry.",
            details={"reason": "invalid_arguments"},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception_handler(
        _: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        message = str(exc.detail)
        code = "internal_error" if exc.status_code >= 500 else "bad_request"
        details = None if exc.status_code >= 500 else {"reason": message}
        if exc.status_code >= 500 and not message:
            message = "Internal server error"
        return _error_response(exc.status_code, code, message, details=details)

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        return _error_response(
            500,
            "internal_error",
            "Internal server error",
            details=None,
        )

    async_jobs_gate = [Depends(_replica_component_gate(ASYNC_JOBS))]
    fhir_bulk_gate = [Depends(_replica_component_gate(FHIR_BULK_JOBS))]
    smart_backend_gate = [Depends(_replica_component_gate(SMART_BACKEND_JOBS))]

    @app.get("/health")
    async def health(request: Request) -> Dict[str, str]:
        runtime = _get_service_runtime(request)
        payload = {"status": "ok", "service": SERVICE_NAME}
        if is_development_profile(runtime.profile):
            # Outside development, version and profile need an authenticated
            # call to the metadata route.
            payload["version"] = openmed.__version__
            payload["profile"] = runtime.profile
        return payload

    @app.get(SERVICE_METADATA_PATH, include_in_schema=False)
    async def service_metadata(request: Request) -> Dict[str, str]:
        runtime = _get_service_runtime(request)
        return {
            "service": SERVICE_NAME,
            "version": openmed.__version__,
            "profile": runtime.profile,
        }

    @app.get("/livez")
    async def livez() -> Dict[str, str]:
        return {"status": "ok", "service": SERVICE_NAME}

    @app.get("/readyz")
    async def readyz(request: Request):
        security_report = _security_profile_report(request.app)
        if not security_report.ready:
            return _security_profile_invalid_response(
                security_report, include_details=True
            )
        if not getattr(request.app.state, "ready", False):
            if getattr(request.app.state, "shutting_down", False):
                return _shutting_down_response(_drain_retry_after(request.app.state))
            return _error_response(
                503,
                "not_ready",
                "Service preload has not completed",
                details=None,
            )
        report = replica_state_report(request.app)
        if not report.ready:
            return _error_response(
                503,
                "replica_unsafe_state",
                "Replica-local service state is not safe behind multiple replicas",
                details=report.readiness_details(),
            )
        compatibility = await run_in_threadpool(_get_compatibility_gate(request).report)
        if not compatibility.ready:
            return _incompatible_state_response(compatibility.blocking())
        return {"status": "ready", "service": SERVICE_NAME}

    @app.get(_COMPATIBILITY_PATH, include_in_schema=False)
    async def compatibility(request: Request) -> Dict[str, Any]:
        """Publish the value-free contract and persisted-state compatibility."""
        snapshot = await run_in_threadpool(_get_compatibility_gate(request).snapshot)
        return {
            **snapshot,
            "service": SERVICE_NAME,
            "admission": (
                "draining"
                if getattr(request.app.state, "shutting_down", False)
                else "open"
            ),
        }

    @app.get(
        "/v1/journey/resources",
        response_model=JourneyResourcePageResponse,
        tags=["journey"],
    )
    async def list_journey_resources(
        request: Request,
        resource_type: JourneyResourceKind,
        namespace: Optional[str] = None,
        purpose: Optional[str] = None,
        role: Optional[str] = None,
        attributes: Optional[str] = Query(default=None, max_length=1024),
        consent_state: Optional[str] = None,
        export_policy: Optional[str] = None,
        first: int = Query(default=20, ge=1, le=100),
        after: Optional[str] = Query(default=None, max_length=2048),
        fields: Optional[str] = Query(default=None, max_length=1024),
    ) -> Dict[str, Any]:
        """List a bounded, principal-bound page of versioned Journey resources.

        Access context comes from the authenticated principal. The namespace,
        purpose, role, attribute, consent, and export parameters may only
        narrow that context; anything wider returns a typed denied page.
        """

        catalog = _get_journey_resource_catalog(request)
        policy = getattr(request.app.state, "journey_access_policy", None)
        access_request = JourneyAccessRequest(
            resource_type=resource_type,
            namespace=namespace,
            purpose=purpose,
            role=role,
            attributes=(
                None if attributes is None else parse_access_attributes(attributes)
            ),
            consent_state=consent_state,
            export_policy=export_policy,
            first=first,
            after=after,
            fields=parse_resource_fields(fields),
        )
        started_at = time.perf_counter()
        page = _get_journey_access_gate(request).list_resources(
            catalog,
            access_request,
            principal=request.scope.get("openmed.auth"),
            policy=policy,
            transport="rest",
        )
        event = OperationalEvent(
            category=OperationalCategory.QUERY,
            operation="list",
            state=OperationalState(page.state.value),
            duration_seconds=time.perf_counter() - started_at,
        )
        metrics = getattr(request.app.state, "metrics", None)
        if metrics is not None:
            metrics.record_operational_event(event)
        set_current_span_attributes(operational_trace_attributes(event))
        return page.to_dict()

    @app.post(
        _DECISION_PATH,
        response_model=None,
        tags=["decision"],
        dependencies=[Depends(_negotiate_decision_body)],
        responses={
            200: {
                "description": (
                    "A calibrated decision in the shape of the request: an "
                    "OpenMed result for an OpenMed request, or an "
                    "OpenAI-compatible Decisions response for that body."
                ),
                "content": {
                    "application/json": {
                        "schema": {
                            "oneOf": [
                                decision_result_schema(),
                                openai_decisions_response_schema(),
                            ]
                        }
                    }
                },
            }
        },
    )
    async def fixed_option_decision(
        payload: DecisionsRequestBody,
        request: Request,
    ) -> Any:
        """Evaluate a bounded local decision and return a typed review result.

        The body is either an OpenMed decision request or an OpenAI-compatible
        Decisions body; the response uses the same shape as the request.
        """

        if isinstance(payload, OpenAIDecisionsRequest):
            return await _openai_compatible_decision(payload, request)
        # Only caller-set fields reach the core contract, so a request without
        # extension fields still resolves to schema 1.0.0.
        body = (
            payload.model_dump(exclude_unset=True)
            if hasattr(payload, "model_dump")
            else payload.dict(exclude_unset=True)
        )
        try:
            # Optional video preflight and metadata scrubbing must not block
            # the event loop serving unrelated requests.
            decision_request = await run_in_threadpool(
                CoreDecisionRequest.from_dict, body
            )
        except (DecisionError, TypeError, ValueError) as exc:
            raise InputError(
                "Invalid fixed-option decision request.",
                details={"reason": str(exc)},
            ) from exc
        result = await run_in_threadpool(
            _decide_with_service_state,
            request.app.state,
            decision_request,
        )
        return result.to_dict()

    @app.post(
        _SYSTEM_ONE_PATH,
        tags=["decision"],
        response_model=None,
        responses={
            200: {
                "description": "System One answers plus an OpenMed decision ID.",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["decision_id", "model", "answers"],
                            "properties": {
                                "decision_id": {
                                    "type": "string",
                                    "pattern": "^[0-9a-f]{32}$",
                                },
                                "model": {"type": "string"},
                                "answers": {"type": "object"},
                            },
                        }
                    }
                },
            },
            404: {"description": "Route disabled (default)."},
            422: {"description": "De-identification refused; nothing forwarded."},
        },
    )
    async def private_system_one(payload: SystemOneRequest, request: Request):
        """De-identify state and every question before scoring with the configured local backend."""
        from openmed.core.offline import is_local_only

        from .system_one import (
            SystemOneInputError,
            SystemOnePrivacyError,
            protect_and_decide,
        )

        app_state = request.app.state
        if not app_state.system_one_enabled:
            raise HTTPException(status_code=404, detail="System One route is disabled.")
        runtime = _get_service_runtime(request)
        if not (
            app_state.auth.config.enabled
            and runtime.throttle.rate_limit_enabled
            and is_local_only(runtime.config)
        ):
            return _error_response(
                503,
                "privacy_gateway_not_configured",
                "Local-only mode, service authentication and rate limits are required.",
            )

        def clean(text: str):
            model = (
                runtime.config.pii_model
                or "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
            )
            return runtime.run_model_request(
                model,
                None,
                lambda: openmed.deidentify(
                    text,
                    method="mask",
                    model_name=model,
                    config=runtime.config,
                    loader=runtime.get_loader(),
                    keep_mapping=False,
                    use_safety_sweep=True,
                ),
            )

        try:
            return await run_in_threadpool(
                protect_and_decide,
                {"state": payload.state, "questions": payload.questions},
                cleaner=clean,
                backend=app_state.decision_backend,
                policy=app_state.decision_access_policy,
                profiles=app_state.decision_calibration_profiles,
                audit_key=app_state.system_one_audit_key,
            )
        except SystemOneInputError:
            return _error_response(400, "bad_request", "Invalid System One request.")
        except SystemOnePrivacyError:
            return _error_response(
                422, "outbound_redaction_failed", "Unable to de-identify the request."
            )
        except Exception:
            return _error_response(
                503,
                "not_ready",
                "The local decision backend is unavailable.",
            )

    if openhim_settings.enabled:

        @app.get(OPENHIM_HEARTBEAT_PATH)
        async def openhim_heartbeat(request: Request) -> Dict[str, Any]:
            mediator = getattr(request.app.state, "openhim_mediator", None)
            if mediator is None:
                return {
                    "enabled": True,
                    "registered": False,
                    "urn": str(openhim_settings.registration["urn"]),
                    "last_heartbeat_at": None,
                    "last_error": "not_started",
                }
            return mediator.status()

        @app.post(OPENHIM_MEDIATOR_PATH)
        async def openhim_deidentify(request: Request) -> Response:
            runtime = _get_service_runtime(request)
            headers = mediator_response_headers(request.headers)
            body = await request.body()
            configured_deidentifier = getattr(
                request.app.state, "openhim_deidentifier", None
            )

            def _runtime_deidentifier(text: str, **kwargs: Any) -> Any:
                if configured_deidentifier is not None:
                    return configured_deidentifier(text, **kwargs)
                return openmed.deidentify(
                    text,
                    config=runtime.config,
                    loader=runtime.get_loader(),
                    **kwargs,
                )

            def _transform() -> Any:
                return transform_mediator_payload(
                    body,
                    headers,
                    200,
                    deidentifier=_runtime_deidentifier,
                    policy=openhim_settings.policy,
                    method=openhim_settings.method,
                )

            try:
                result = await _run_with_timeout(runtime, _transform)
            except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
                result = failed_openhim_result(400, "invalid_payload")
            except Exception:
                result = failed_openhim_result(500, "processing_failed")

            if not result.transformed:
                return Response(
                    content=result.body,
                    status_code=result.status_code,
                    headers=dict(result.headers),
                )

            envelope = build_openhim_envelope(
                result,
                urn=str(openhim_settings.registration["urn"]),
                request_method=request.method,
                request_path=request.url.path,
            )
            return Response(
                content=json.dumps(envelope, ensure_ascii=False, separators=(",", ":")),
                status_code=200,
                media_type=OPENHIM_MEDIA_TYPE,
            )

    @app.get("/models/loaded")
    async def loaded_models(request: Request) -> Dict[str, Any]:
        runtime = _get_service_runtime(request)
        return runtime.loaded_models()

    @app.post("/models/unload")
    async def unload_models(
        payload: ModelUnloadRequest,
        request: Request,
    ) -> Dict[str, Any]:
        runtime = _get_service_runtime(request)
        set_access_log_model_name(request, payload.model_name)
        if payload.all:
            return runtime.unload_all_models()
        if payload.model_name is None:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=400, detail="model_name is required when 'all' is not set"
            )
        return runtime.unload_model(payload.model_name)

    @app.post("/analyze")
    async def analyze(payload: AnalyzeRequest, request: Request) -> Dict[str, Any]:
        set_access_log_model_name(request, payload.model_name)
        runtime = _get_service_runtime(request)
        priority = _request_priority(request)
        priority_handle = BatchPriorityHandle(priority)

        async def _operation() -> Dict[str, Any]:
            if runtime.batching.enabled:
                return await _submit_with_timeout(
                    runtime,
                    _get_analyze_batcher(request),
                    _AnalyzeBatchJob(payload),
                    priority=priority,
                    priority_handle=priority_handle,
                )

            def _model_operation() -> Dict[str, Any]:
                return runtime.run_model_request(
                    payload.model_name,
                    payload.keep_alive,
                    lambda: _analyze_payload(payload, runtime),
                )

            with trace_service_stage(
                "model_request",
                {
                    "openmed.endpoint": "/analyze",
                    "openmed.input.length": len(payload.text),
                    "openmed.model_name": payload.model_name,
                },
            ):
                return await _run_with_timeout(runtime, _model_operation)

        return await _run_maybe_coalesced(
            request,
            "/analyze",
            payload,
            _operation,
            priority=priority,
            on_priority_upgrade=priority_handle.promote,
        )

    @app.post("/pii/extract")
    async def pii_extract(
        payload: PIIExtractRequest, request: Request
    ) -> Dict[str, Any]:
        set_access_log_model_name(request, payload.model_name)
        runtime = _get_service_runtime(request)
        priority = _request_priority(request)
        priority_handle = BatchPriorityHandle(priority)

        async def _operation() -> Dict[str, Any]:
            if runtime.batching.enabled:
                return await _submit_with_timeout(
                    runtime,
                    _get_pii_extract_batcher(request),
                    _PIIExtractBatchJob(payload),
                    priority=priority,
                    priority_handle=priority_handle,
                )

            def _model_operation() -> Dict[str, Any]:
                return runtime.run_model_request(
                    payload.model_name,
                    payload.keep_alive,
                    lambda: _pii_extract_payload(payload, runtime),
                )

            with trace_service_stage(
                "model_request",
                {
                    "openmed.endpoint": "/pii/extract",
                    "openmed.input.length": len(payload.text),
                    "openmed.model_name": payload.model_name,
                },
            ):
                return await _run_with_timeout(runtime, _model_operation)

        return await _run_maybe_coalesced(
            request,
            "/pii/extract",
            payload,
            _operation,
            priority=priority,
            on_priority_upgrade=priority_handle.promote,
        )

    @app.post("/pii/extract/stream")
    async def pii_extract_stream(
        payload: PIIExtractStreamRequest,
        request: Request,
    ) -> StreamingResponse:
        runtime = _get_service_runtime(request)
        runtime.validate_served_model(payload.model_name)
        controls = NDJSONStreamControls.for_request(request.scope, runtime)
        return controls.response(
            _drain_aware_stream(
                request,
                pii_extract_ndjson_stream(
                    payload,
                    runtime,
                    retain_abandoned_work=controls.retain,
                ),
                retain=controls.retain,
            )
        )

    @app.post("/pii/deidentify")
    async def pii_deidentify(
        payload: PIIDeidentifyRequest,
        request: Request,
        response: Response,
    ) -> Dict[str, Any]:
        if payload.shift_dates is not None:
            response.headers["OpenMed-Deprecation"] = SHIFT_DATES_NOTICE
        set_access_log_model_name(request, payload.model_name)
        runtime = _get_service_runtime(request)
        priority = _request_priority(request)

        async def _operation() -> Dict[str, Any]:
            def _model_operation() -> Dict[str, Any]:
                return runtime.run_model_request(
                    payload.model_name,
                    payload.keep_alive,
                    lambda: _pii_deidentify_payload(payload, runtime),
                )

            with trace_service_stage(
                "model_request",
                {
                    "openmed.endpoint": "/pii/deidentify",
                    "openmed.input.length": len(payload.text),
                    "openmed.model_name": payload.model_name,
                },
            ):
                return await _run_with_timeout(runtime, _model_operation)

        return await _run_maybe_coalesced(
            request,
            "/pii/deidentify",
            payload,
            _operation,
            priority=priority,
        )

    @app.post("/pii/deidentify/stream")
    async def pii_deidentify_stream(
        payload: PIIDeidentifyStreamRequest,
        request: Request,
    ) -> StreamingResponse:
        set_access_log_model_name(request, payload.model_name)
        runtime = _get_service_runtime(request)
        events = deidentify_ndjson_stream(payload, runtime)
        controls = NDJSONStreamControls.for_request(request.scope, runtime)
        response = controls.response(
            _drain_aware_stream(request, events, retain=controls.retain)
        )
        if payload.shift_dates is not None:
            response.headers["OpenMed-Deprecation"] = SHIFT_DATES_NOTICE
        return response

    @app.post(_PRIVACY_GATEWAY_PATH)
    async def privacy_gateway_complete(
        payload: PrivacyGatewayRequest,
        request: Request,
    ) -> Dict[str, Any]:
        runtime = _get_service_runtime(request)

        async def _operation() -> Dict[str, Any]:
            def _model_operation() -> Dict[str, Any]:
                return runtime.run_model_request(
                    payload.model_name,
                    payload.keep_alive,
                    lambda: _privacy_gateway_payload(
                        payload,
                        runtime,
                        request.app,
                    ),
                )

            return await _run_with_timeout(runtime, _model_operation)

        return await _operation()

    @app.post("/omop/load")
    async def omop_load(payload: OmopLoadRequest, request: Request) -> Dict[str, Any]:
        from ..structured.quality import QualityGateError

        with trace_service_stage(
            "omop_load",
            {
                "openmed.endpoint": "/omop/load",
                "openmed.input.length": len(payload.records_jsonl),
            },
        ):
            try:
                return await run_in_threadpool(_omop_load_summary, payload)
            except QualityGateError as exc:
                return JSONResponse(
                    status_code=409,
                    content={
                        "status": "rejected",
                        "quality_gate": exc.report.to_dict(),
                    },
                )

    @app.post("/profile")
    async def profile_route(
        payload: ProfileRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Profile extracted output without returning note text or values."""

        with trace_service_stage(
            "profile",
            {
                "openmed.endpoint": "/profile",
                "openmed.input.length": len(payload.records_jsonl),
            },
        ):
            return await run_in_threadpool(_profile_summary, payload)

    @app.post("/brief", response_model=BriefResponse, response_model_exclude_unset=True)
    async def brief_route(payload: BriefRequest, request: Request) -> Dict[str, Any]:
        """Return the shared local brief without recording protected content."""
        from openmed.clinical.brief_cancellation import BriefCancellation

        from .brief import brief_response

        cancellation = BriefCancellation()
        provider = getattr(request.app.state, "brief_context_provider", None)
        try:
            response = await asyncio.to_thread(
                brief_response,
                payload.text,
                model=payload.model,
                profile=payload.profile,
                review_id=payload.review_id,
                context_provider=provider,
                cancellation=cancellation,
            )
            set_access_log_brief(request, response)
            return response
        except asyncio.CancelledError:
            cancellation.cancel()
            response = brief_response("", cancellation=cancellation)
            set_access_log_brief(request, response)
            return response
        except (TypeError, ValueError):
            return _error_response(
                422, "validation_error", "Invalid local brief request."
            )

    @app.post("/ground", response_model=GroundResponse)
    async def ground_route(payload: GroundRequest, request: Request) -> Dict[str, Any]:
        """Ground text or pre-extracted entities against local snapshots."""

        input_count = len(payload.entities) if payload.entities is not None else 1
        set_access_log_grounding(
            request,
            input_count=input_count,
            systems=payload.systems,
            lang=payload.lang,
        )
        try:
            response = await run_in_threadpool(_ground_summary, payload)
            set_access_log_grounding(
                request,
                input_count=input_count,
                result_count=len(response["results"]),
                systems=payload.systems,
                lang=payload.lang,
            )
            return response
        except Exception as exc:
            from openmed.clinical.grounding import (
                RestrictedVocabularyError,
                VocabLoaderError,
            )
            from openmed.core.offline import OfflineModeError

            if isinstance(exc, RestrictedVocabularyError):
                return _error_response(
                    400,
                    "restricted_terminology_unconfigured",
                    "Restricted terminology requires a configured user-supplied "
                    "out-of-process endpoint.",
                )
            if isinstance(exc, OfflineModeError):
                return _error_response(
                    503,
                    "offline_snapshot_unavailable",
                    "The requested vocabulary snapshot is unavailable offline.",
                )
            if isinstance(exc, VocabLoaderError):
                return _error_response(
                    400,
                    "snapshot_invalid",
                    "The requested vocabulary snapshot could not be verified.",
                )
            if isinstance(exc, (TypeError, ValueError)):
                return _error_response(
                    400,
                    "grounding_invalid_request",
                    "The grounding request is invalid.",
                )
            raise

    @app.post(
        _FHIR_BULK_EXPORT_PATH,
        status_code=202,
        include_in_schema=False,
        dependencies=fhir_bulk_gate,
    )
    async def start_fhir_bulk_export(
        payload: FHIRBulkExportRequest,
        request: Request,
        response: Response,
    ) -> Dict[str, Any]:
        return _start_fhir_bulk_route(
            payload, request, response, route=_FHIR_BULK_EXPORT_ROUTE
        )

    @app.post(
        _FHIR_BULK_IMPORT_PATH,
        status_code=202,
        include_in_schema=False,
        dependencies=fhir_bulk_gate,
    )
    async def start_fhir_bulk_import(
        payload: FHIRBulkImportRequest,
        request: Request,
        response: Response,
    ) -> Dict[str, Any]:
        return _start_fhir_bulk_route(
            payload, request, response, route=_FHIR_BULK_IMPORT_ROUTE
        )

    @app.get(
        f"{_FHIR_BULK_EXPORT_PATH}/{{job_id}}",
        include_in_schema=False,
        dependencies=fhir_bulk_gate,
    )
    @app.get(
        f"{_FHIR_BULK_IMPORT_PATH}/{{job_id}}",
        include_in_schema=False,
        dependencies=fhir_bulk_gate,
    )
    async def fhir_bulk_job_status(
        job_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        manager = _get_fhir_bulk_manager(request)
        try:
            _require_visible_bulk_job(request, manager, job_id)
            return manager.get(job_id).to_dict()
        except KeyError as exc:
            raise StarletteHTTPException(
                status_code=404,
                detail="bulk job not found",
            ) from exc

    @app.get(
        f"{_FHIR_BULK_EXPORT_PATH}/{{job_id}}/manifest",
        include_in_schema=False,
        dependencies=fhir_bulk_gate,
    )
    @app.get(
        f"{_FHIR_BULK_IMPORT_PATH}/{{job_id}}/manifest",
        include_in_schema=False,
        dependencies=fhir_bulk_gate,
    )
    async def fhir_bulk_job_manifest(
        job_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        manager = _get_fhir_bulk_manager(request)
        try:
            _require_visible_bulk_job(request, manager, job_id)
            return manager.manifest(job_id)
        except KeyError as exc:
            raise StarletteHTTPException(
                status_code=404,
                detail="bulk job not found",
            ) from exc
        except ValueError as exc:
            raise StarletteHTTPException(status_code=409, detail=str(exc)) from exc

    @app.get(
        f"{_FHIR_BULK_EXPORT_PATH}/{{job_id}}/report",
        include_in_schema=False,
        dependencies=fhir_bulk_gate,
    )
    @app.get(
        f"{_FHIR_BULK_IMPORT_PATH}/{{job_id}}/report",
        include_in_schema=False,
        dependencies=fhir_bulk_gate,
    )
    async def fhir_bulk_job_report(
        job_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        manager = _get_fhir_bulk_manager(request)
        try:
            _require_visible_bulk_job(request, manager, job_id)
            return manager.report(job_id)
        except KeyError as exc:
            raise StarletteHTTPException(
                status_code=404,
                detail="bulk job not found",
            ) from exc
        except ValueError as exc:
            raise StarletteHTTPException(status_code=409, detail=str(exc)) from exc

    @app.delete(
        f"{_FHIR_BULK_EXPORT_PATH}/{{job_id}}",
        status_code=202,
        include_in_schema=False,
        dependencies=fhir_bulk_gate,
    )
    @app.delete(
        f"{_FHIR_BULK_IMPORT_PATH}/{{job_id}}",
        status_code=202,
        include_in_schema=False,
        dependencies=fhir_bulk_gate,
    )
    async def cancel_fhir_bulk_job(
        job_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        manager = _get_fhir_bulk_manager(request)
        try:
            _require_visible_bulk_job(request, manager, job_id)
            status = await manager.cancel(job_id)
            return status.to_dict()
        except KeyError as exc:
            raise StarletteHTTPException(
                status_code=404,
                detail="bulk job not found",
            ) from exc

    @app.post("/cohort/resolve")
    async def cohort_resolve(
        payload: CohortResolveRequest,
        request: Request,
    ) -> Dict[str, Any]:
        from ..structured.quality import QualityGateError

        with trace_service_stage(
            "cohort_resolve",
            {
                "openmed.endpoint": "/cohort/resolve",
                "openmed.input.length": len(payload.records_jsonl),
            },
        ):
            try:
                return await run_in_threadpool(_cohort_resolve_summary, payload)
            except QualityGateError as exc:
                return JSONResponse(
                    status_code=409,
                    content={
                        "status": "rejected",
                        "quality_gate": exc.report.to_dict(),
                    },
                )

    @app.post(
        _SMART_BACKEND_START_PATH,
        dependencies=smart_backend_gate,
        responses=_IDEMPOTENT_START_RESPONSES,
        openapi_extra=_IDEMPOTENCY_OPENAPI_EXTRA,
    )
    async def start_smart_backend_ingestion(
        payload: SMARTBackendIngestionRequest,
        request: Request,
        response: Response,
    ) -> Dict[str, Any]:
        manager = _get_smart_backend_manager(request)
        owner_digest = _submitting_owner_digest(request)
        keyed = _keyed_start_request(
            request,
            payload,
            route=_SMART_BACKEND_START_ROUTE,
            owner_digest=owner_digest,
        )
        reservation: Optional[IdempotencyReservation] = None
        if keyed is not None:
            outcome = manager.reserve_idempotent_start(keyed)
            if isinstance(outcome, IdempotentReplay):
                response.headers[IDEMPOTENCY_REPLAYED_HEADER] = "true"
                return outcome.response
            reservation = outcome
        try:
            runtime = _get_service_runtime(request)
            runtime.validate_served_model(payload.model_name)
            config = _smart_backend_config_from_payload(payload)
            authorize_smart_backend_config(config)
            # Admission precedes path reservation so a rejected start leaves no
            # output directory behind.
            manager.ensure_capacity()
            job_id = uuid.uuid4().hex
            _, output, checkpoint = request.app.state.bulk_storage.prepare(
                input_dir=None,
                output_dir=payload.output_dir,
                checkpoint_path=payload.checkpoint_path,
                job_id=job_id,
            )
            config = replace(config, output_dir=output, checkpoint_path=checkpoint)
            deidentifier = _smart_backend_deidentifier(payload, runtime)
            status = manager.start(
                config,
                deidentifier=deidentifier,
                job_id=job_id,
                owner_digest=owner_digest,
            )
            if reservation is not None:
                manager.complete_idempotent_start(reservation, status)
        finally:
            if reservation is not None:
                manager.release_idempotent_start(reservation)
        return status.to_dict()

    @app.get(
        "/fhir/smart-backend/ingestions/{job_id}",
        dependencies=smart_backend_gate,
    )
    async def smart_backend_ingestion_status(
        job_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        manager = _get_smart_backend_manager(request)
        try:
            _require_visible_bulk_job(request, manager, job_id)
            return manager.get(job_id).to_dict()
        except KeyError as exc:
            raise StarletteHTTPException(
                status_code=404,
                detail="ingestion job not found",
            ) from exc

    @app.get(
        "/fhir/smart-backend/ingestions/{job_id}/summary",
        dependencies=smart_backend_gate,
    )
    async def smart_backend_ingestion_summary(
        job_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        manager = _get_smart_backend_manager(request)
        try:
            _require_visible_bulk_job(request, manager, job_id)
            return manager.summary(job_id).to_dict()
        except KeyError as exc:
            raise StarletteHTTPException(
                status_code=404,
                detail="ingestion job not found",
            ) from exc
        except ValueError as exc:
            raise StarletteHTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc

    @app.delete(
        "/fhir/smart-backend/ingestions/{job_id}",
        status_code=202,
        include_in_schema=False,
        dependencies=smart_backend_gate,
    )
    async def cancel_smart_backend_ingestion(
        job_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        manager = _get_smart_backend_manager(request)
        try:
            _require_visible_bulk_job(request, manager, job_id)
            return (await manager.cancel(job_id)).to_dict()
        except KeyError as exc:
            raise StarletteHTTPException(
                status_code=404,
                detail="ingestion job not found",
            ) from exc

    @app.post(
        "/jobs",
        status_code=202,
        dependencies=async_jobs_gate,
        responses=_IDEMPOTENT_JOB_RESPONSES,
        openapi_extra=_IDEMPOTENCY_OPENAPI_EXTRA,
    )
    async def create_job(
        payload: DeidentifyJobRequest,
        request: Request,
        response: Response,
    ) -> Dict[str, Any]:
        _get_service_runtime(request).validate_served_model(payload.model_name)
        queue = _get_job_queue(request)
        submission = queue.submit_job(
            payload,
            owner_digest=_submitting_owner_digest(request),
            idempotency_key=parse_idempotency_key(
                request.headers.getlist(IDEMPOTENCY_KEY_HEADER)
            ),
        )
        if payload.shift_dates is not None:
            response.headers["OpenMed-Deprecation"] = SHIFT_DATES_NOTICE
        record = submission.record
        if not submission.created:
            response.headers[IDEMPOTENCY_REPLAYED_HEADER] = "true"
        return job_response_payload(record, status_url=f"/jobs/{record['id']}")

    @app.get("/jobs/{job_id}", dependencies=async_jobs_gate)
    async def get_job(job_id: str, request: Request) -> Dict[str, Any]:
        queue = _get_job_queue(request)
        record = queue.get(job_id)
        if record is None or not _job_visible(
            request,
            admin_scope=JOBS_ADMIN_SCOPE,
            owner_digest=lambda: record.get(OWNER_DIGEST_FIELD),
        ):
            raise HTTPException(status_code=404, detail="job not found")
        return job_response_payload(record, status_url=f"/jobs/{job_id}")

    from .graphql_app import GraphQLPrivacyHeadersMiddleware, mount_graphql

    mount_graphql(
        app,
        runtime_getter=_get_service_runtime,
        resource_getter=_get_journey_resource_catalog,
        access_getter=_get_journey_access_gate,
    )

    from .workflow_routes import WorkflowBoundaryMiddleware, mount_workflow_routes

    mount_workflow_routes(
        app, service=workflow_service, policy=workflow_policy, clock=workflow_clock
    )

    if app.state.tracing.enabled:
        app.add_middleware(
            OpenTelemetryMiddleware,
            tracing=app.state.tracing,
        )
    # Signature checks sit inside both body-limit layers and before routing.
    install_request_signing(
        app,
        error_response=_error_response,
        max_body_bytes=app.state.operational_limits.max_request_bytes,
    )
    body_limit = (
        int(max_request_body_bytes)
        if max_request_body_bytes is not None
        else (
            get_max_text_length() * _REQUEST_BODY_ENCODING_MULTIPLIER
            + _REQUEST_BODY_OVERHEAD_BYTES
        )
    )
    app.state.model_body_limit_bytes = body_limit
    app.add_middleware(
        BoundedRequestBodyMiddleware,
        max_bytes=body_limit,
        limited_paths=_MODEL_BACKED_PATHS
        | (
            {_MULTIMODAL_PREFLIGHT_PATH} if experimental_multimodal_preflight else set()
        ),
        read_limits=app.state.request_body_read_limits,
        rejection_response=_model_route_body_rejection,
        metrics=app.state.metrics,
        # The global limiter below already checked the JSON structure.
        scan_json=False,
    )
    app.add_middleware(
        BoundedRequestBodyMiddleware,
        read_limits=app.state.request_body_read_limits,
        metrics=app.state.metrics,
        limits=app.state.operational_limits,
    )
    app.add_middleware(
        CorrelationIdMiddleware,
        log_config=service_log_config_from_env(),
        minimal_log_paths=(_MULTIMODAL_PREFLIGHT_PATH,),
    )
    app.add_middleware(
        WorkflowBoundaryMiddleware,
        read_limits=app.state.request_body_read_limits,
        metrics=app.state.metrics,
    )
    # Starlette runs the last-added middleware first. Host, CORS and
    # header-only credential checks must finish before the body-limit and
    # throttle layers registered earlier read a body or spend a token.
    app.middleware("http")(_auth_middleware)
    if security_config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=security_config.cors_origins,
            allow_methods=SERVICE_CORS_METHODS,
            allow_headers=service_cors_allow_headers(
                auth_enabled=app.state.auth.enabled
            ),
            expose_headers=SERVICE_CORS_EXPOSE_HEADERS,
        )
    app.add_middleware(
        ErrorEnvelopeTrustedHostMiddleware,
        allowed_hosts=security_config.trusted_hosts,
        www_redirect=False,
    )
    # Outside the app's own layers, so a stalled NDJSON consumer is bounded at
    # the server's send.
    app.add_middleware(StreamWriteIdleMiddleware, paths=STREAM_ROUTE_PATHS)

    # Registered last so it is outermost: every /graphql response, including
    # rejections raised by the middlewares above, carries the privacy headers.
    app.add_middleware(GraphQLPrivacyHeadersMiddleware)
    validate_route_policies(app)
    return app


async def _dispatch_analyze_batch(
    runtime: ServiceRuntime,
    jobs: Sequence[_AnalyzeBatchJob],
) -> Sequence[BatchResult[_ServicePayload]]:
    return await run_in_threadpool(_dispatch_analyze_batch_sync, runtime, list(jobs))


def _dispatch_analyze_batch_sync(
    runtime: ServiceRuntime,
    jobs: Sequence[_AnalyzeBatchJob],
) -> Sequence[BatchResult[_ServicePayload]]:
    results: list[Optional[BatchResult[_ServicePayload]]] = [None] * len(jobs)
    groups: dict[Tuple[Any, ...], list[int]] = {}
    for index, job in enumerate(jobs):
        groups.setdefault(_analyze_batch_key(job.payload), []).append(index)

    for indexes in groups.values():
        _dispatch_analyze_group(runtime, jobs, indexes, results)

    return _completed_batch_results(results)


def _dispatch_analyze_group(
    runtime: ServiceRuntime,
    jobs: Sequence[_AnalyzeBatchJob],
    indexes: Sequence[int],
    results: list[Optional[BatchResult[_ServicePayload]]],
) -> None:
    active_jobs: list[Tuple[int, str, Any]] = []

    for index in indexes:
        try:
            model_key = runtime.begin_model_request(jobs[index].payload.model_name)
        except Exception as exc:
            results[index] = exc
        else:
            active_jobs.append((index, model_key, jobs[index].payload.keep_alive))

    try:
        active_indexes = [index for index, _, _ in active_jobs]
        if not active_indexes:
            return

        payloads = [jobs[index].payload for index in active_indexes]
        if _can_backend_batch_analyze(payloads, runtime):
            try:
                with trace_service_stage(
                    "analyze_batch",
                    {
                        "openmed.batch.size": len(payloads),
                        "openmed.input.count": len(payloads),
                        "openmed.input.total_length": sum(
                            len(payload.text) for payload in payloads
                        ),
                        "openmed.model_name": payloads[0].model_name,
                    },
                ):
                    batch_results = _analyze_payload_batch(payloads, runtime)
                if len(batch_results) != len(active_indexes):
                    raise InferenceError(
                        "Analyze batch returned "
                        f"{len(batch_results)} results for {len(active_indexes)} "
                        "jobs. Retry without batching or inspect the model backend.",
                        details={
                            "expected_count": len(active_indexes),
                            "actual_count": len(batch_results),
                        },
                    )
            except Exception:
                for index in active_indexes:
                    try:
                        results[index] = _analyze_payload(jobs[index].payload, runtime)
                    except Exception as exc:
                        results[index] = exc
            else:
                for index, result in zip(active_indexes, batch_results):
                    results[index] = result
        else:
            for index in active_indexes:
                try:
                    results[index] = _analyze_payload(jobs[index].payload, runtime)
                except Exception as exc:
                    results[index] = exc
    finally:
        _finish_active_batch_requests(runtime, active_jobs, results)


async def _dispatch_pii_extract_batch(
    runtime: ServiceRuntime,
    jobs: Sequence[_PIIExtractBatchJob],
) -> Sequence[BatchResult[_ServicePayload]]:
    return await run_in_threadpool(
        _dispatch_pii_extract_batch_sync,
        runtime,
        list(jobs),
    )


def _dispatch_pii_extract_batch_sync(
    runtime: ServiceRuntime,
    jobs: Sequence[_PIIExtractBatchJob],
) -> Sequence[BatchResult[_ServicePayload]]:
    results: list[Optional[BatchResult[_ServicePayload]]] = [None] * len(jobs)
    groups: dict[Tuple[Any, ...], list[int]] = {}
    for index, job in enumerate(jobs):
        groups.setdefault(_pii_extract_batch_key(job.payload), []).append(index)

    for indexes in groups.values():
        _dispatch_pii_extract_group(runtime, jobs, indexes, results)

    return _completed_batch_results(results)


def _dispatch_pii_extract_group(
    runtime: ServiceRuntime,
    jobs: Sequence[_PIIExtractBatchJob],
    indexes: Sequence[int],
    results: list[Optional[BatchResult[_ServicePayload]]],
) -> None:
    active_jobs: list[Tuple[int, str, Any]] = []
    for index in indexes:
        try:
            model_key = runtime.begin_model_request(jobs[index].payload.model_name)
        except Exception as exc:
            results[index] = exc
        else:
            active_jobs.append((index, model_key, jobs[index].payload.keep_alive))

    try:
        active_indexes = [index for index, _, _ in active_jobs]
        if not active_indexes:
            return
        payloads = [jobs[index].payload for index in active_indexes]
        try:
            with trace_service_stage(
                "pii_extract_batch",
                {
                    "openmed.batch.size": len(payloads),
                    "openmed.input.count": len(payloads),
                    "openmed.input.total_length": sum(
                        len(payload.text) for payload in payloads
                    ),
                    "openmed.model_name": payloads[0].model_name,
                },
            ):
                batch_results = _pii_extract_payload_batch(payloads, runtime)
            if len(batch_results) != len(active_indexes):
                raise InferenceError(
                    "PII extract batch returned "
                    f"{len(batch_results)} results for {len(active_indexes)} "
                    "jobs. Retry without batching or inspect the model backend.",
                    details={
                        "expected_count": len(active_indexes),
                        "actual_count": len(batch_results),
                    },
                )
        except Exception:
            for index in active_indexes:
                try:
                    results[index] = _pii_extract_payload(jobs[index].payload, runtime)
                except Exception as exc:
                    results[index] = exc
        else:
            for index, result in zip(active_indexes, batch_results):
                results[index] = result
    except Exception as exc:
        for index, _, _ in active_jobs:
            if results[index] is None:
                results[index] = exc
    finally:
        _finish_active_batch_requests(runtime, active_jobs, results)


def _finish_active_batch_requests(
    runtime: ServiceRuntime,
    active_jobs: Sequence[Tuple[int, str, Any]],
    results: list[Optional[BatchResult[_ServicePayload]]],
) -> None:
    for index, model_key, keep_alive in active_jobs:
        try:
            result = results[index]
            error = result if isinstance(result, BaseException) else None
            runtime.finish_model_request(model_key, keep_alive, error=error)
        except Exception as exc:
            if not isinstance(results[index], BaseException):
                results[index] = exc


def _completed_batch_results(
    results: Sequence[Optional[BatchResult[_ServicePayload]]],
) -> list[BatchResult[_ServicePayload]]:
    completed: list[BatchResult[_ServicePayload]] = []
    for result in results:
        if result is None:
            completed.append(
                InternalError(
                    "A batch job produced no result. Retry the request and report "
                    "this batch invariant if it recurs."
                )
            )
        else:
            completed.append(result)
    return completed


def _analyze_batch_key(payload: AnalyzeRequest) -> Tuple[Any, ...]:
    return (
        payload.model_name,
        payload.confidence_threshold,
        payload.group_entities,
        payload.aggregation_strategy,
        payload.sentence_detection,
        payload.sentence_language,
        payload.sentence_clean,
        payload.use_fast_tokenizer,
    )


def _can_backend_batch_analyze(
    payloads: Sequence[AnalyzeRequest],
    runtime: ServiceRuntime,
) -> bool:
    if not payloads:
        return False
    first = payloads[0]
    return not first.sentence_detection and not bool(
        getattr(runtime.config, "use_medical_tokenizer", False)
    )


def _normalize_batch_predictions(
    raw_predictions: Any, expected_count: int
) -> list[Any]:
    if expected_count == 1:
        if isinstance(raw_predictions, list) and (
            not raw_predictions or isinstance(raw_predictions[0], dict)
        ):
            return [raw_predictions]
        if isinstance(raw_predictions, list) and len(raw_predictions) == 1:
            return [raw_predictions[0]]
        return [raw_predictions]

    if not isinstance(raw_predictions, list):
        raise InferenceError(
            "The analyze backend returned a non-list batch result. Retry without "
            "batching or inspect the model backend."
        )
    if len(raw_predictions) != expected_count:
        raise InferenceError(
            "Analyze backend returned "
            f"{len(raw_predictions)} results for {expected_count} inputs. Retry "
            "without batching or inspect the model backend.",
            details={
                "expected_count": expected_count,
                "actual_count": len(raw_predictions),
            },
        )
    return list(raw_predictions)


def _analyze_payload_batch(
    payloads: Sequence[AnalyzeRequest],
    runtime: ServiceRuntime,
) -> list[_ServicePayload]:
    if not payloads:
        return []

    first = payloads[0]
    model_name = validate_model_name(first.model_name)
    loader = runtime.get_loader()
    pipeline = loader.create_pipeline(
        model_name,
        task="token-classification",
        aggregation_strategy=first.aggregation_strategy,
        use_fast_tokenizer=first.use_fast_tokenizer,
    )

    effective_max_length = loader.get_max_sequence_length(
        model_name,
        tokenizer=getattr(pipeline, "tokenizer", None),
    )
    tokenizer = getattr(pipeline, "tokenizer", None)
    if tokenizer is not None and effective_max_length is not None:
        try:
            tokenizer.model_max_length = int(effective_max_length)
        except (AttributeError, OverflowError, TypeError, ValueError):
            pass

    import time

    texts = [payload.text for payload in payloads]
    start_time = time.time()
    raw_predictions = pipeline(texts, batch_size=len(texts))
    processing_time = time.time() - start_time
    per_item_time = processing_time / len(texts) if texts else 0.0
    normalized = _normalize_batch_predictions(raw_predictions, len(texts))

    responses: list[_ServicePayload] = []
    for payload, predictions in zip(payloads, normalized):
        if predictions is None:
            predictions = []
        elif isinstance(predictions, dict):
            predictions = [predictions]
        else:
            predictions = list(predictions)

        metadata = {"sentence_detection": False}
        if effective_max_length is not None:
            metadata["max_length"] = effective_max_length
        result = format_predictions(
            predictions,
            payload.text,
            model_name=model_name,
            output_format="dict",
            include_confidence=True,
            confidence_threshold=payload.confidence_threshold,
            group_entities=payload.group_entities,
            metadata=metadata,
            processing_time=per_item_time,
        )
        responses.append(_result_to_dict(result))
    return responses


def _pii_extract_batch_key(payload: PIIExtractRequest) -> Tuple[Any, ...]:
    return (
        payload.model_name,
        payload.confidence_threshold,
        payload.use_smart_merging,
        payload.lang,
        payload.normalize_accents,
    )


def _pii_extract_payload_batch(
    payloads: Sequence[PIIExtractRequest],
    runtime: ServiceRuntime,
) -> list[_ServicePayload]:
    from openmed.core.pii import _extract_pii_batch

    if not payloads:
        return []

    first = payloads[0]
    results = _extract_pii_batch(
        [payload.text for payload in payloads],
        model_name=first.model_name,
        confidence_threshold=first.confidence_threshold,
        config=runtime.config,
        use_smart_merging=first.use_smart_merging,
        lang=first.lang,
        normalize_accents=first.normalize_accents,
        loader=runtime.get_loader(),
        batch_size=len(payloads),
    )
    return [_result_to_dict(result) for result in results]


def _analyze_payload(
    payload: AnalyzeRequest, runtime: ServiceRuntime
) -> Dict[str, Any]:
    with trace_service_stage(
        "analyze_pipeline",
        {
            "openmed.endpoint": "/analyze",
            "openmed.input.length": len(payload.text),
            "openmed.model_name": payload.model_name,
        },
    ):
        result = openmed.analyze_text(
            payload.text,
            model_name=payload.model_name,
            config=runtime.config,
            loader=runtime.get_loader(),
            aggregation_strategy=payload.aggregation_strategy,
            output_format="dict",
            confidence_threshold=payload.confidence_threshold,
            group_entities=payload.group_entities,
            sentence_detection=payload.sentence_detection,
            sentence_language=payload.sentence_language,
            sentence_clean=payload.sentence_clean,
            use_fast_tokenizer=payload.use_fast_tokenizer,
        )
        response = _result_to_dict(result)
        set_current_span_attributes(result_summary_attributes(response))
        return response


def _pii_extract_payload(
    payload: PIIExtractRequest,
    runtime: ServiceRuntime,
) -> Dict[str, Any]:
    with trace_service_stage(
        "pii_extract_pipeline",
        {
            "openmed.endpoint": "/pii/extract",
            "openmed.input.length": len(payload.text),
            "openmed.model_name": payload.model_name,
        },
    ):
        result = openmed.extract_pii(
            payload.text,
            model_name=payload.model_name,
            confidence_threshold=payload.confidence_threshold,
            config=runtime.config,
            use_smart_merging=payload.use_smart_merging,
            lang=payload.lang,
            normalize_accents=payload.normalize_accents,
            loader=runtime.get_loader(),
        )
        response = _result_to_dict(result)
        set_current_span_attributes(result_summary_attributes(response))
        return response


def _pii_deidentify_payload(
    payload: PIIDeidentifyRequest,
    runtime: ServiceRuntime,
) -> Dict[str, Any]:
    from openmed.core.policy import canonical_policy_name, load_policy

    with trace_service_stage(
        "pii_deidentify_pipeline",
        {
            "openmed.endpoint": "/pii/deidentify",
            "openmed.input.length": len(payload.text),
            "openmed.model_name": payload.model_name,
        },
    ):
        policy_name = canonical_policy_name(payload.policy) if payload.policy else None
        policy_profile = load_policy(policy_name) if policy_name is not None else None

        result = openmed.deidentify(
            payload.text,
            method=payload.method,
            model_name=payload.model_name,
            confidence_threshold=payload.confidence_threshold,
            keep_year=payload.keep_year,
            **(
                {"shift_dates": payload.shift_dates}
                if payload.shift_dates is not None
                else {}
            ),
            date_shift_days=payload.date_shift_days,
            keep_mapping=payload.keep_mapping,
            config=runtime.config,
            use_smart_merging=payload.use_smart_merging,
            use_safety_sweep=payload.use_safety_sweep,
            lang=payload.lang,
            normalize_accents=payload.normalize_accents,
            loader=runtime.get_loader(),
            policy=policy_name,
        )

        response = _result_to_dict(result)
        set_current_span_attributes(result_summary_attributes(response))
        should_emit_mapping = payload.keep_mapping or bool(
            policy_profile is not None and policy_profile.keep_mapping
        )
        if should_emit_mapping and getattr(result, "mapping", None):
            response["mapping"] = result.mapping
        return response


def _privacy_gateway_payload(
    payload: PrivacyGatewayRequest,
    runtime: ServiceRuntime,
    fastapi_app: FastAPI,
) -> Dict[str, Any]:
    transport = _get_privacy_gateway_transport(fastapi_app)
    audit_trail = _get_privacy_gateway_audit_trail(fastapi_app)
    store = _get_privacy_gateway_store(fastapi_app)

    def _extractor(text: str, **kwargs: Any) -> Any:
        return openmed.extract_pii(
            text,
            config=runtime.config,
            loader=runtime.get_loader(),
            **kwargs,
        )

    gateway = PrivacyGateway(
        transport=transport,
        extractor=_extractor,
        audit_trail=audit_trail,
        store=store,
    )
    policy = PrivacyGatewayPolicy(
        name=payload.policy,
        min_confidence=payload.confidence_threshold,
        detector_confidence_floor=payload.detector_confidence_floor,
        disallowed_entity_categories=frozenset(payload.disallowed_entity_categories),
    )
    result = gateway.complete(
        payload.text,
        policy=policy,
        model_name=payload.model_name,
        use_smart_merging=payload.use_smart_merging,
        lang=payload.lang,
        normalize_accents=payload.normalize_accents,
    )
    return result.to_dict()


def _get_privacy_gateway_transport(fastapi_app: FastAPI) -> Any:
    transport = getattr(fastapi_app.state, "privacy_gateway_transport", None)
    if transport is not None:
        return transport
    transport = HttpExternalLLMTransport.from_env()
    fastapi_app.state.privacy_gateway_transport = transport
    return transport


def _get_privacy_gateway_audit_trail(
    fastapi_app: FastAPI,
) -> PrivacyGatewayAuditTrail:
    audit_trail = getattr(fastapi_app.state, "privacy_gateway_audit_trail", None)
    if audit_trail is None:
        audit_trail = PrivacyGatewayAuditTrail()
        fastapi_app.state.privacy_gateway_audit_trail = audit_trail
    return audit_trail


def _get_privacy_gateway_store(fastapi_app: FastAPI) -> InMemoryReidentificationStore:
    store = getattr(fastapi_app.state, "privacy_gateway_store", None)
    if store is None:
        store = InMemoryReidentificationStore()
        fastapi_app.state.privacy_gateway_store = store
    return store


def _smart_backend_config_from_payload(
    payload: SMARTBackendIngestionRequest,
) -> SMARTBackendConfig:
    return SMARTBackendConfig(
        fhir_base_url=payload.fhir_base_url,
        token_url=payload.token_url,
        client_id=payload.client_id,
        private_key_pem=payload.private_key_pem,
        output_dir=payload.output_dir,
        checkpoint_path=payload.checkpoint_path,
        key_id=payload.key_id,
        scope=payload.scope,
        export_path=payload.export_path,
        max_inflight_downloads=payload.max_inflight_downloads,
        max_buffered_resources=getattr(payload, "max_buffered_resources", 1),
        poll_interval_seconds=payload.poll_interval_seconds,
        request_timeout_seconds=payload.request_timeout_seconds,
        policy=payload.policy or "hipaa_safe_harbor",
        method=payload.method,
    )


def _fhir_bulk_config_from_payload(
    payload: FHIRBulkExportRequest | FHIRBulkImportRequest,
) -> FHIRBulkJobConfig:
    """Convert a validated route payload without retaining request metadata."""

    input_dir = (
        payload.input_dir if payload.input_dir is not None else payload.source_dir
    )
    return FHIRBulkJobConfig(
        input_dir=input_dir,
        output_dir=payload.output_dir,
        checkpoint_path=payload.checkpoint_path,
        policy=payload.policy,
        method=payload.method,
        max_buffered_resources=payload.max_buffered_resources,
        max_inflight_downloads=payload.max_inflight_downloads,
        poll_interval_seconds=payload.poll_interval_seconds,
        request_timeout_seconds=payload.request_timeout_seconds,
        fhir_base_url=payload.fhir_base_url,
        token_url=payload.token_url,
        client_id=payload.client_id,
        private_key_pem=payload.private_key_pem,
        key_id=payload.key_id,
        scope=payload.scope,
        export_path=payload.export_path,
    )


def _start_fhir_bulk_route(
    payload: FHIRBulkExportRequest | FHIRBulkImportRequest,
    request: Request,
    response: Response,
    *,
    route: str,
) -> Dict[str, Any]:
    """Start, or replay the keyed start of, one bulk export/import job.

    A repeated ``Idempotency-Key`` is resolved before any validation, storage
    reservation or task scheduling, so a replay or conflict has no side effect.
    """
    manager = _get_fhir_bulk_manager(request)
    owner_digest = _submitting_owner_digest(request)
    keyed = _keyed_start_request(
        request, payload, route=route, owner_digest=owner_digest
    )
    reservation: Optional[IdempotencyReservation] = None
    if keyed is not None:
        outcome = manager.reserve_idempotent_start(keyed)
        if isinstance(outcome, IdempotentReplay):
            response.headers[IDEMPOTENCY_REPLAYED_HEADER] = "true"
            return _fhir_bulk_start_payload(
                request, response, outcome.job_id, outcome.response
            )
        reservation = outcome
    try:
        status = _start_fhir_bulk_job(
            payload, request, manager=manager, owner_digest=owner_digest
        )
        if reservation is not None:
            manager.complete_idempotent_start(reservation, status)
    finally:
        if reservation is not None:
            manager.release_idempotent_start(reservation)
    return _fhir_bulk_start_payload(request, response, status.job_id, status.to_dict())


def _fhir_bulk_start_payload(
    request: Request,
    response: Response,
    job_id: str,
    status_payload: Dict[str, Any],
) -> Dict[str, Any]:
    status_url = f"{request.url.path}/{job_id}"
    response.headers["Content-Location"] = status_url
    payload = dict(status_payload)
    payload["status_url"] = status_url
    return payload


def _start_fhir_bulk_job(
    payload: FHIRBulkExportRequest | FHIRBulkImportRequest,
    request: Request,
    *,
    manager: FHIRBulkJobManager,
    owner_digest: Optional[dict[str, str]],
):
    runtime = _get_service_runtime(request)
    runtime.validate_served_model(payload.model_name)
    config = _fhir_bulk_config_from_payload(payload)
    if payload.input_dir is not None and payload.source_dir is not None:
        raise BulkPathError("path_invalid")
    if not config.is_local:
        authorize_smart_backend_config(config.to_smart_config())
    # Admission precedes path reservation so a rejected start leaves no output
    # directory behind.
    manager.ensure_capacity()
    job_id = uuid.uuid4().hex
    source, output, checkpoint = request.app.state.bulk_storage.prepare(
        input_dir=payload.input_dir
        if payload.input_dir is not None
        else payload.source_dir,
        output_dir=payload.output_dir,
        checkpoint_path=payload.checkpoint_path,
        job_id=job_id,
    )
    config = replace(
        config, input_dir=source, output_dir=output, checkpoint_path=checkpoint
    )
    configured_deidentifier = getattr(request.app.state, "fhir_bulk_deidentifier", None)
    deidentifier = configured_deidentifier or _fhir_bulk_deidentifier(payload, runtime)
    return manager.start(
        config,
        deidentifier=deidentifier,
        job_id=job_id,
        owner_digest=owner_digest,
    )


def _fhir_bulk_deidentifier(
    payload: FHIRBulkExportRequest | FHIRBulkImportRequest,
    runtime: ServiceRuntime,
):
    def _deidentifier(text: str, **kwargs: Any) -> Any:
        method = str(kwargs.get("method", payload.method))
        policy = kwargs.get("policy") or payload.policy
        consistent = bool(kwargs.get("consistent", False))

        def _operation() -> Any:
            return openmed.deidentify(
                text,
                method=method,
                model_name=payload.model_name,
                confidence_threshold=payload.confidence_threshold,
                config=runtime.config,
                use_smart_merging=payload.use_smart_merging,
                use_safety_sweep=payload.use_safety_sweep,
                lang=payload.lang,
                normalize_accents=payload.normalize_accents,
                loader=runtime.get_loader(),
                policy=policy,
                consistent=consistent,
            )

        return runtime.run_model_request(
            payload.model_name,
            payload.keep_alive,
            _operation,
        )

    return _deidentifier


def _smart_backend_deidentifier(
    payload: SMARTBackendIngestionRequest,
    runtime: ServiceRuntime,
):
    def _deidentifier(text: str, **kwargs: Any) -> Any:
        method = str(kwargs.get("method", payload.method))
        policy = kwargs.get("policy") or payload.policy
        consistent = bool(kwargs.get("consistent", False))

        def _operation() -> Any:
            return openmed.deidentify(
                text,
                method=method,
                model_name=payload.model_name,
                confidence_threshold=payload.confidence_threshold,
                config=runtime.config,
                use_smart_merging=payload.use_smart_merging,
                use_safety_sweep=payload.use_safety_sweep,
                lang=payload.lang,
                normalize_accents=payload.normalize_accents,
                loader=runtime.get_loader(),
                policy=policy,
                consistent=consistent,
            )

        return runtime.run_model_request(
            payload.model_name,
            payload.keep_alive,
            _operation,
        )

    return _deidentifier


app = create_app()
