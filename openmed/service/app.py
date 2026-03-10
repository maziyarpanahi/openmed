"""FastAPI application for the OpenMed REST service."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Mapping, Optional

import openmed
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .runtime import ServiceRuntime
from .schemas import AnalyzeRequest, PIIDeidentifyRequest, PIIExtractRequest

SERVICE_NAME = "openmed-rest"


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


def _get_service_runtime(request: Request) -> ServiceRuntime:
    """Return the initialized service runtime."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        runtime = ServiceRuntime.from_env()
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


def create_app() -> FastAPI:
    """Create and configure the OpenMed REST FastAPI app."""

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        runtime = ServiceRuntime.from_env()
        _attach_runtime(fastapi_app, runtime)

        if runtime.preload_models:
            await run_in_threadpool(runtime.preload)

        yield

    app = FastAPI(
        title="OpenMed REST API",
        version=openmed.__version__,
        description="Hardened REST API for OpenMed text analysis and PII workflows.",
        lifespan=lifespan,
    )

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

    @app.post("/analyze")
    async def analyze(payload: AnalyzeRequest, request: Request) -> Dict[str, Any]:
        runtime = _get_service_runtime(request)

        def _operation() -> Dict[str, Any]:
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

        return await _run_with_timeout(runtime, _operation)

    @app.post("/pii/extract")
    async def pii_extract(payload: PIIExtractRequest, request: Request) -> Dict[str, Any]:
        runtime = _get_service_runtime(request)

        def _operation() -> Dict[str, Any]:
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

        return await _run_with_timeout(runtime, _operation)

    @app.post("/pii/deidentify")
    async def pii_deidentify(
        payload: PIIDeidentifyRequest,
        request: Request,
    ) -> Dict[str, Any]:
        runtime = _get_service_runtime(request)

        def _operation() -> Dict[str, Any]:
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
                lang=payload.lang,
                normalize_accents=payload.normalize_accents,
                loader=runtime.get_loader(),
            )

            response = _result_to_dict(result)
            if payload.keep_mapping and getattr(result, "mapping", None):
                response["mapping"] = result.mapping
            return response

        return await _run_with_timeout(runtime, _operation)

    return app


app = create_app()
