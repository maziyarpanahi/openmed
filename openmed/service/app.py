"""FastAPI application for the OpenMed REST service."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Sequence, Tuple

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware

import openmed
from openmed.processing import format_predictions
from openmed.utils.validation import validate_model_name

from .batcher import BatchResult, DynamicBatcher
from .coalesce import RequestCoalescer, coalescing_key
from .metrics import (
    PROMETHEUS_CONTENT_TYPE,
    PrometheusMetricsRegistry,
    metrics_enabled_from_env,
)
from .runtime import ServiceRuntime
from .schemas import (
    AnalyzeRequest,
    ModelUnloadRequest,
    PIIDeidentifyRequest,
    PIIExtractRequest,
)
from .security_headers import (
    SERVICE_CORS_HEADERS,
    SERVICE_CORS_METHODS,
    ErrorEnvelopeTrustedHostMiddleware,
    parse_service_security_config,
)
from .throttle import ServiceThrottle
from .warm_pool import WarmPoolBackpressureError

SERVICE_NAME = "openmed-rest"
_MODEL_BACKED_PATHS = frozenset({"/analyze", "/pii/extract", "/pii/deidentify"})
_ServicePayload = Dict[str, Any]
_ServiceOperation = Callable[[], Awaitable[_ServicePayload]]
_AnalyzeBatcher = DynamicBatcher["_AnalyzeBatchJob", _ServicePayload]
_PIIExtractBatcher = DynamicBatcher["_PIIExtractBatchJob", _ServicePayload]


@dataclass(frozen=True)
class _AnalyzeBatchJob:
    payload: AnalyzeRequest


@dataclass(frozen=True)
class _PIIExtractBatchJob:
    payload: PIIExtractRequest


class ServiceTimeoutError(RuntimeError):
    """Raised when a request exceeds the configured service timeout."""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = float(timeout_seconds)
        super().__init__(
            f"Request exceeded configured timeout of {self.timeout_seconds:g} seconds"
        )


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert an OpenMed result object to a JSON-serializable mapping."""
    if hasattr(result, "to_dict") and callable(result.to_dict):
        payload = result.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
        raise TypeError("Result to_dict() must return a mapping.")

    if isinstance(result, Mapping):
        return dict(result)

    raise TypeError("Unsupported result payload type.")


def _error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    details: Optional[Any] = None,
) -> JSONResponse:
    """Return a standardized API error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": details,
            }
        },
    )


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
    app.state.throttle = ServiceThrottle(
        runtime.throttle,
        error_response=_error_response,
        limited_paths=_MODEL_BACKED_PATHS,
    )
    app.state.analyze_batcher = None
    app.state.pii_extract_batcher = None
    app.state.request_coalescer = None
    if runtime.coalescing.enabled:
        app.state.request_coalescer = RequestCoalescer()
    if runtime.batching.enabled:
        app.state.analyze_batcher = DynamicBatcher(
            lambda jobs: _dispatch_analyze_batch(runtime, jobs),
            max_batch_size=runtime.batching.max_batch_size,
            max_wait_ms=runtime.batching.max_wait_ms,
        )
        app.state.pii_extract_batcher = DynamicBatcher(
            lambda jobs: _dispatch_pii_extract_batch(runtime, jobs),
            max_batch_size=runtime.batching.max_batch_size,
            max_wait_ms=runtime.batching.max_wait_ms,
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


async def _run_with_timeout(
    runtime: ServiceRuntime,
    operation: Any,
) -> Any:
    """Run blocking service work in a threadpool under the profile timeout."""
    timeout_seconds = float(getattr(runtime.config, "timeout", 0) or 0)
    if timeout_seconds <= 0:
        return await run_in_threadpool(operation)

    try:
        return await asyncio.wait_for(
            run_in_threadpool(operation),
            timeout=float(timeout_seconds),
        )
    except asyncio.TimeoutError as exc:
        raise ServiceTimeoutError(timeout_seconds) from exc


async def _await_with_timeout(
    runtime: ServiceRuntime,
    awaitable: Any,
) -> Any:
    """Await async service work under the profile timeout."""
    timeout_seconds = float(getattr(runtime.config, "timeout", 0) or 0)
    if timeout_seconds <= 0:
        return await awaitable

    try:
        return await asyncio.wait_for(awaitable, timeout=float(timeout_seconds))
    except asyncio.TimeoutError as exc:
        raise ServiceTimeoutError(timeout_seconds) from exc


def _get_analyze_batcher(request: Request) -> _AnalyzeBatcher:
    batcher = getattr(request.app.state, "analyze_batcher", None)
    if batcher is None:
        raise RuntimeError("Analyze batcher is not initialized")
    return batcher


def _get_pii_extract_batcher(request: Request) -> _PIIExtractBatcher:
    batcher = getattr(request.app.state, "pii_extract_batcher", None)
    if batcher is None:
        raise RuntimeError("PII extract batcher is not initialized")
    return batcher


async def _run_maybe_coalesced(
    request: Request,
    endpoint: str,
    payload: Any,
    operation: _ServiceOperation,
) -> _ServicePayload:
    coalescer = getattr(request.app.state, "request_coalescer", None)
    if coalescer is None:
        return await operation()

    return await coalescer.run(coalescing_key(endpoint, payload), operation)


def _metrics_route_label(request: Request) -> str:
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path
    return "unknown"


def create_app() -> FastAPI:
    """Create and configure the OpenMed REST FastAPI app."""

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        runtime = ServiceRuntime.from_env(
            metrics=getattr(fastapi_app.state, "metrics", None)
        )
        _attach_runtime(fastapi_app, runtime)
        fastapi_app.state.ready = False
        fastapi_app.state.shutting_down = False
        fastapi_app.state.inflight = 0

        if runtime.preload_models:
            await run_in_threadpool(runtime.preload)

        fastapi_app.state.ready = True

        try:
            yield
        finally:
            fastapi_app.state.ready = False
            fastapi_app.state.shutting_down = True
            loop = asyncio.get_event_loop()
            deadline = loop.time() + float(
                getattr(runtime, "shutdown_drain_seconds", 0.0) or 0.0
            )
            while getattr(fastapi_app.state, "inflight", 0) > 0 and (
                loop.time() < deadline
            ):
                await asyncio.sleep(0.05)

    app = FastAPI(
        title="OpenMed REST API",
        version=openmed.__version__,
        description="Hardened REST API for OpenMed text analysis and PII workflows.",
        lifespan=lifespan,
    )
    security_config = parse_service_security_config()
    app.state.security = security_config
    if security_config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=security_config.cors_origins,
            allow_methods=SERVICE_CORS_METHODS,
            allow_headers=SERVICE_CORS_HEADERS,
        )
    app.add_middleware(
        ErrorEnvelopeTrustedHostMiddleware,
        allowed_hosts=security_config.trusted_hosts,
        www_redirect=False,
    )
    app.state.metrics = (
        PrometheusMetricsRegistry() if metrics_enabled_from_env() else None
    )

    @app.middleware("http")
    async def _readiness_middleware(request: Request, call_next):
        path = request.url.path
        if path in _MODEL_BACKED_PATHS:
            state = request.app.state
            if getattr(state, "shutting_down", False):
                return _error_response(
                    503,
                    "not_ready",
                    "Service is shutting down and not accepting new requests",
                    details=None,
                )
            state.inflight = getattr(state, "inflight", 0) + 1
            try:
                return await call_next(request)
            finally:
                state.inflight = getattr(state, "inflight", 0) - 1
        return await call_next(request)

    if app.state.metrics is not None:

        @app.middleware("http")
        async def _metrics_middleware(request: Request, call_next):
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
            return Response(
                content=registry.render(),
                media_type=PROMETHEUS_CONTENT_TYPE,
            )

    @app.middleware("http")
    async def _throttle_middleware(request: Request, call_next):
        throttle = getattr(request.app.state, "throttle", None)
        if throttle is None:
            _get_service_runtime(request)
            throttle = getattr(request.app.state, "throttle", None)
        if throttle is None:
            return await call_next(request)
        return await throttle.dispatch(request, call_next)

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(
        _: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            422,
            "validation_error",
            "Request validation failed",
            details=_format_validation_details(exc),
        )

    @app.exception_handler(ServiceTimeoutError)
    async def _timeout_handler(_: Request, exc: ServiceTimeoutError) -> JSONResponse:
        return _error_response(
            504,
            "timeout",
            str(exc),
            details={"timeout_seconds": exc.timeout_seconds},
        )

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

    @app.exception_handler(ValueError)
    async def _value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        reason = str(exc)
        return _error_response(
            400,
            "bad_request",
            reason,
            details={"reason": reason},
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

    @app.get("/health")
    async def health(request: Request) -> Dict[str, str]:
        runtime = _get_service_runtime(request)
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "version": openmed.__version__,
            "profile": runtime.profile,
        }

    @app.get("/livez")
    async def livez() -> Dict[str, str]:
        return {"status": "ok", "service": SERVICE_NAME}

    @app.get("/readyz")
    async def readyz(request: Request):
        if getattr(request.app.state, "ready", False):
            return {"status": "ready", "service": SERVICE_NAME}
        return _error_response(
            503,
            "not_ready",
            "Service preload has not completed",
            details=None,
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
        if payload.all:
            return runtime.unload_all_models()
        assert payload.model_name is not None
        return runtime.unload_model(payload.model_name)

    @app.post("/analyze")
    async def analyze(payload: AnalyzeRequest, request: Request) -> Dict[str, Any]:
        runtime = _get_service_runtime(request)

        async def _operation() -> Dict[str, Any]:
            if runtime.batching.enabled:
                return await _await_with_timeout(
                    runtime,
                    _get_analyze_batcher(request).submit(_AnalyzeBatchJob(payload)),
                )

            def _model_operation() -> Dict[str, Any]:
                return runtime.run_model_request(
                    payload.model_name,
                    payload.keep_alive,
                    lambda: _analyze_payload(payload, runtime),
                )

            return await _run_with_timeout(runtime, _model_operation)

        return await _run_maybe_coalesced(request, "/analyze", payload, _operation)

    @app.post("/pii/extract")
    async def pii_extract(
        payload: PIIExtractRequest, request: Request
    ) -> Dict[str, Any]:
        runtime = _get_service_runtime(request)

        async def _operation() -> Dict[str, Any]:
            if runtime.batching.enabled:
                return await _await_with_timeout(
                    runtime,
                    _get_pii_extract_batcher(request).submit(
                        _PIIExtractBatchJob(payload)
                    ),
                )

            def _model_operation() -> Dict[str, Any]:
                return runtime.run_model_request(
                    payload.model_name,
                    payload.keep_alive,
                    lambda: _pii_extract_payload(payload, runtime),
                )

            return await _run_with_timeout(runtime, _model_operation)

        return await _run_maybe_coalesced(
            request,
            "/pii/extract",
            payload,
            _operation,
        )

    @app.post("/pii/deidentify")
    async def pii_deidentify(
        payload: PIIDeidentifyRequest,
        request: Request,
    ) -> Dict[str, Any]:
        runtime = _get_service_runtime(request)

        async def _operation() -> Dict[str, Any]:
            def _model_operation() -> Dict[str, Any]:
                return runtime.run_model_request(
                    payload.model_name,
                    payload.keep_alive,
                    lambda: _pii_deidentify_payload(payload, runtime),
                )

            return await _run_with_timeout(runtime, _model_operation)

        return await _run_maybe_coalesced(
            request,
            "/pii/deidentify",
            payload,
            _operation,
        )

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
                batch_results = _analyze_payload_batch(payloads, runtime)
                if len(batch_results) != len(active_indexes):
                    raise ValueError(
                        "Analyze batch returned "
                        f"{len(batch_results)} results for {len(active_indexes)} jobs"
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
            batch_results = _pii_extract_payload_batch(payloads, runtime)
            if len(batch_results) != len(active_indexes):
                raise ValueError(
                    "PII extract batch returned "
                    f"{len(batch_results)} results for {len(active_indexes)} jobs"
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
            runtime.finish_model_request(model_key, keep_alive)
        except Exception as exc:
            if not isinstance(results[index], BaseException):
                results[index] = exc


def _completed_batch_results(
    results: Sequence[Optional[BatchResult[_ServicePayload]]],
) -> list[BatchResult[_ServicePayload]]:
    completed: list[BatchResult[_ServicePayload]] = []
    for result in results:
        if result is None:
            completed.append(RuntimeError("Batch job did not produce a result"))
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
        raise ValueError("Analyze backend returned a non-list batch result")
    if len(raw_predictions) != expected_count:
        raise ValueError(
            "Analyze backend returned "
            f"{len(raw_predictions)} results for {expected_count} inputs"
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
        except Exception:
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
    return _result_to_dict(result)


def _pii_extract_payload(
    payload: PIIExtractRequest,
    runtime: ServiceRuntime,
) -> Dict[str, Any]:
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
    return _result_to_dict(result)


def _pii_deidentify_payload(
    payload: PIIDeidentifyRequest,
    runtime: ServiceRuntime,
) -> Dict[str, Any]:
    from openmed.core.policy import canonical_policy_name, load_policy

    policy_name = canonical_policy_name(payload.policy) if payload.policy else None
    policy_profile = load_policy(policy_name) if policy_name is not None else None

    result = openmed.deidentify(
        payload.text,
        method=payload.method,
        model_name=payload.model_name,
        confidence_threshold=payload.confidence_threshold,
        keep_year=payload.keep_year,
        shift_dates=payload.shift_dates,
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
    should_emit_mapping = payload.keep_mapping or bool(
        policy_profile is not None and policy_profile.keep_mapping
    )
    if should_emit_mapping and getattr(result, "mapping", None):
        response["mapping"] = result.mapping
    return response


app = create_app()
