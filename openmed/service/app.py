"""FastAPI application for the OpenMed REST MVP."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Mapping

import openmed
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from openmed.core.config import OpenMedConfig, PROFILE_ENV_VAR

from .schemas import AnalyzeRequest, PIIDeidentifyRequest, PIIExtractRequest

SERVICE_NAME = "openmed-rest"


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


def _resolve_service_config(request: Request) -> OpenMedConfig:
    """Return service config from app state, lazily initializing if missing."""
    config = getattr(request.app.state, "config", None)
    if config is not None:
        return config

    profile = os.getenv(PROFILE_ENV_VAR, "prod")
    config = OpenMedConfig.from_profile(profile)
    request.app.state.profile = profile
    request.app.state.config = config
    return config


def create_app() -> FastAPI:
    """Create and configure the OpenMed REST FastAPI app."""
    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI):
        profile = os.getenv(PROFILE_ENV_VAR, "prod")
        fastapi_app.state.profile = profile
        fastapi_app.state.config = OpenMedConfig.from_profile(profile)
        yield

    app = FastAPI(
        title="OpenMed REST API",
        version=openmed.__version__,
        description="Dockerized REST MVP for OpenMed text analysis and PII workflows.",
        lifespan=lifespan,
    )

    @app.exception_handler(ValueError)
    async def _value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc), "error_type": "ValueError"},
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
            )

        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    @app.get("/health")
    async def health(request: Request) -> Dict[str, str]:
        profile = getattr(request.app.state, "profile", os.getenv(PROFILE_ENV_VAR, "prod"))
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "version": openmed.__version__,
            "profile": profile,
        }

    @app.post("/analyze")
    async def analyze(payload: AnalyzeRequest, request: Request) -> Dict[str, Any]:
        config = _resolve_service_config(request)
        result = openmed.analyze_text(
            payload.text,
            model_name=payload.model_name,
            config=config,
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

    @app.post("/pii/extract")
    async def pii_extract(payload: PIIExtractRequest, request: Request) -> Dict[str, Any]:
        config = _resolve_service_config(request)
        result = openmed.extract_pii(
            payload.text,
            model_name=payload.model_name,
            confidence_threshold=payload.confidence_threshold,
            config=config,
            use_smart_merging=payload.use_smart_merging,
            lang=payload.lang,
            normalize_accents=payload.normalize_accents,
        )
        return _result_to_dict(result)

    @app.post("/pii/deidentify")
    async def pii_deidentify(
        payload: PIIDeidentifyRequest, request: Request
    ) -> Dict[str, Any]:
        config = _resolve_service_config(request)
        result = openmed.deidentify(
            payload.text,
            method=payload.method,
            model_name=payload.model_name,
            confidence_threshold=payload.confidence_threshold,
            keep_year=payload.keep_year,
            shift_dates=payload.shift_dates,
            date_shift_days=payload.date_shift_days,
            keep_mapping=payload.keep_mapping,
            config=config,
            use_smart_merging=payload.use_smart_merging,
            lang=payload.lang,
            normalize_accents=payload.normalize_accents,
        )

        response = _result_to_dict(result)
        if payload.keep_mapping and getattr(result, "mapping", None):
            response["mapping"] = result.mapping
        return response

    return app


app = create_app()
