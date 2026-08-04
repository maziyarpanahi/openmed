"""MCP server for OpenMed agent integrations."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from inspect import Signature
from typing import Annotated, Any, Callable, Dict, Optional

import openmed
from openmed.agent.security.injection_guard import (
    InjectionGuard,
    PromptInjectionDetected,
)
from openmed.clinical.exporters.fhir import to_bundle, to_fhir
from openmed.clinical.grounding import (
    DEFAULT_GROUNDING_SYSTEMS,
    Candidate,
    GroundedSpan,
    VocabLoader,
    ground,
)
from openmed.clinical.grounding.provenance import GroundingProvenance
from openmed.core.model_registry import ModelInfo
from openmed.core.pii_i18n import (
    DEFAULT_PII_MODELS,
    INDIC_NER_LANGUAGES,
    LANGUAGE_NAMES,
    SUPPORTED_LANGUAGES,
)
from openmed.core.schemas import OpenMedSpan
from openmed.mcp.clinical_workflow import (
    clinical_workflow_resource_document,
    load_golden_agent_run,
    render_clinical_workflow_prompt,
)
from openmed.mcp.tool_registry import (
    CLINICAL_WORKFLOW_SPEC,
    TOOL_REGISTRY,
    ToolSchemaValidationError,
    ToolSpec,
    render_mcp_tool,
    render_tool_registry_document,
    validate_registered_tool_output,
)
from openmed.mcp.workflow import (
    ClinicalPipelineArtifact,
    WorkflowRunner,
    builtin_workflow_step_executors,
    execute_clinical_pipeline,
    plan_clinical_pipeline,
)
from openmed.risk import safe_risk_summary
from openmed.risk.reid import risk_report
from openmed.service.runtime import ServiceRuntime
from openmed.utils.gateway import normalize_text, validate_language
from openmed.utils.validation import validate_model_name

RuntimeProvider = Callable[[], ServiceRuntime]
PrivacyGatewayProvider = Callable[[], Any]


def _safe_int_env(name: str, default: int) -> int:
    """Read an environment variable as an int, falling back to *default* on error."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


MCP_INSTRUCTIONS = (
    "OpenMed exposes local clinical NLP, PII extraction, and de-identification "
    "tools. Use synthetic examples for tests and docs. Only send real PHI to "
    "OpenMed instances the user operates and trusts. Prefer local model paths "
    "or approved OpenMed/Hugging Face model identifiers in regulated flows."
)

_DEFAULT_RUNTIME: Optional[ServiceRuntime] = None


def _load_fastmcp() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised by packaging users
        raise RuntimeError(
            "The MCP SDK is not installed. Install OpenMed with the MCP extra: "
            'pip install "openmed[mcp]"'
        ) from exc
    return FastMCP


def _get_default_runtime() -> ServiceRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = ServiceRuntime.from_env()
    return _DEFAULT_RUNTIME


def _runtime(runtime_provider: Optional[RuntimeProvider] = None) -> ServiceRuntime:
    if runtime_provider is not None:
        return runtime_provider()
    return _get_default_runtime()


def _result_to_dict(result: Any) -> Dict[str, Any]:
    if hasattr(result, "to_dict") and callable(result.to_dict):
        payload = result.to_dict()
        if isinstance(payload, dict):
            return dict(payload)
        raise TypeError("Result to_dict() must return a dictionary.")

    if isinstance(result, dict):
        return dict(result)

    raise TypeError("Unsupported OpenMed result type.")


def _run_model_request(
    runtime: ServiceRuntime,
    model_name: str,
    keep_alive: Any,
    operation: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    return runtime.run_model_request(model_name, keep_alive, operation)


def _model_info_to_dict(key: str, model: ModelInfo) -> Dict[str, Any]:
    payload = asdict(model)
    payload["key"] = key
    payload["size_mb"] = model.size_mb
    return payload


def _json_resource(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _error_envelope(error: BaseException) -> Dict[str, Any]:
    """Return a PHI-safe structured tool error without echoing input or output."""

    if isinstance(error, PromptInjectionDetected):
        return {"error": error.to_dict(), "is_error": True}
    error_module = error.__class__.__module__
    if isinstance(error, ToolSchemaValidationError):
        code = "invalid_result"
        message = "The tool returned an invalid structured result."
    elif isinstance(error, KeyError):
        code = "unknown_tool"
        message = "The requested tool is not available."
    elif isinstance(error, (TypeError, ValueError)) or error_module.startswith(
        ("jsonschema", "pydantic")
    ):
        code = "invalid_arguments"
        message = "The tool arguments are invalid."
    else:
        code = "execution_error"
        message = "The tool could not complete the request."
    return {"error": {"code": code, "message": message}, "is_error": True}


def _call_tool_result(payload: Dict[str, Any], *, is_error: bool) -> Any:
    """Return structured content plus a JSON text fallback for older clients."""

    from mcp.types import CallToolResult, TextContent

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, sort_keys=True),
            )
        ],
        structuredContent=payload,
        isError=is_error,
    )


def _mcp_return_annotation(spec: ToolSpec) -> Any:
    """Build a typed result annotation from the registry's output contract."""

    try:
        from mcp.types import CallToolResult
        from pydantic import ConfigDict, RootModel
    except ImportError:
        return Dict[str, Any]

    model_name = "".join(part.title() for part in spec.name.split("_")) + "Result"
    output_model = type(
        model_name,
        (RootModel[Dict[str, Any]],),
        {
            "model_config": ConfigDict(
                json_schema_extra=spec.mcp_output_schema(),
            )
        },
    )
    return Annotated[CallToolResult, output_model]


def _render_structured_mcp_tool(
    spec: ToolSpec,
    handler: Callable[..., Dict[str, Any]],
    injection_guard: Optional[InjectionGuard] = None,
) -> Callable[..., Any]:
    """Render one registry tool with structured success and error results."""

    registry_tool = render_mcp_tool(spec, handler)
    return_annotation = _mcp_return_annotation(spec)

    def _tool(*args: Any, **kwargs: Any) -> Any:
        try:
            if injection_guard is None:
                payload = registry_tool(*args, **kwargs)
            else:
                bound = spec.signature.bind_partial(*args, **kwargs)
                guarded = injection_guard.guard_input(bound.arguments)
                payload = registry_tool(**guarded.value)
        except Exception as error:
            return _call_tool_result(_error_envelope(error), is_error=True)
        return _call_tool_result(payload, is_error=False)

    _tool.__name__ = registry_tool.__name__
    _tool.__doc__ = spec.description
    _tool.__signature__ = Signature(  # type: ignore[attr-defined]
        parameters=tuple(spec.signature.parameters.values()),
        return_annotation=return_annotation,
    )
    _tool.__annotations__ = dict(registry_tool.__annotations__)
    _tool.__annotations__["return"] = return_annotation
    return _tool


def _mcp_annotations(spec: ToolSpec) -> Any:
    """Return SDK annotations when available, or the equivalent plain mapping."""

    payload = spec.annotations()
    try:
        from mcp.types import ToolAnnotations
    except ImportError:
        return payload
    return ToolAnnotations(**payload)


def _synchronize_registered_schemas(server: Any, spec: ToolSpec) -> None:
    """Make FastMCP advertise the registry schemas verbatim."""

    manager = getattr(server, "_tool_manager", None)
    if manager is None:
        return
    registered = manager.get_tool(spec.name)
    if registered is None:
        return
    registered.parameters = deepcopy(dict(spec.input_schema))
    registered.fn_metadata.output_schema = spec.mcp_output_schema()
    registered.__dict__.pop("output_schema", None)


def _structured_fastmcp(
    base_class: Any,
    injection_guard: Optional[InjectionGuard] = None,
) -> Any:
    """Return a FastMCP class that envelopes malformed calls without logging data."""

    from jsonschema import validate

    selected_guard = injection_guard or InjectionGuard.from_env(
        "OPENMED_MCP_INJECTION_GUARD_MODE"
    )

    class _OpenMedFastMCP(base_class):
        async def call_tool(
            self,
            name: str,
            arguments: Dict[str, Any],
        ) -> Any:
            try:
                tool = self._tool_manager.get_tool(name)
                if tool is None:
                    raise KeyError(name)
                guarded = selected_guard.guard_arguments(arguments)
                validate(instance=guarded.value, schema=tool.parameters)
                return await super().call_tool(name, guarded.value)
            except Exception as error:
                return _call_tool_result(_error_envelope(error), is_error=True)

    return _OpenMedFastMCP


def openmed_analyze_text(
    text: str,
    model_name: str = "disease_detection_superclinical",
    confidence_threshold: Optional[float] = 0.0,
    group_entities: bool = False,
    aggregation_strategy: Optional[str] = "simple",
    sentence_detection: bool = True,
    sentence_language: str = "en",
    sentence_clean: bool = False,
    use_fast_tokenizer: bool = True,
    keep_alive: Optional[str] = None,
    *,
    runtime_provider: Optional[RuntimeProvider] = None,
) -> Dict[str, Any]:
    """Run OpenMed named-entity recognition and return a JSON-ready result."""
    from openmed.service.schemas import AnalyzeRequest

    # Validate through the shared gateway so the MCP surface applies the same
    # length/size/encoding guardrails as the library and REST entry points.
    text = normalize_text(text)

    payload = AnalyzeRequest(
        text=text,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        group_entities=group_entities,
        aggregation_strategy=aggregation_strategy,
        sentence_detection=sentence_detection,
        sentence_language=sentence_language,
        sentence_clean=sentence_clean,
        use_fast_tokenizer=use_fast_tokenizer,
        keep_alive=keep_alive,
    )
    runtime = _runtime(runtime_provider)

    def operation() -> Dict[str, Any]:
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

    response = _run_model_request(
        runtime,
        payload.model_name,
        payload.keep_alive,
        operation,
    )
    return validate_registered_tool_output("openmed_analyze_text", response)


def openmed_extract_pii(
    text: str,
    model_name: str = DEFAULT_PII_MODELS["en"],
    confidence_threshold: float = 0.5,
    use_smart_merging: bool = True,
    lang: str = "en",
    normalize_accents: Optional[bool] = None,
    keep_alive: Optional[str] = None,
    *,
    runtime_provider: Optional[RuntimeProvider] = None,
) -> Dict[str, Any]:
    """Extract PII/PHI entities and return a JSON-ready result."""
    from openmed.service.schemas import PIIExtractRequest

    # Shared gateway: normalize text and guard the language before dispatch so
    # the MCP surface rejects the same bad inputs as the REST and library paths.
    text = normalize_text(text)
    lang = validate_language(lang, include_national_id=False)

    payload = PIIExtractRequest(
        text=text,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        use_smart_merging=use_smart_merging,
        lang=lang,
        normalize_accents=normalize_accents,
        keep_alive=keep_alive,
    )
    runtime = _runtime(runtime_provider)

    def operation() -> Dict[str, Any]:
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

    response = _run_model_request(
        runtime,
        payload.model_name,
        payload.keep_alive,
        operation,
    )
    return validate_registered_tool_output("openmed_extract_pii", response)


def openmed_deidentify(
    text: str,
    method: str = "mask",
    model_name: str = DEFAULT_PII_MODELS["en"],
    confidence_threshold: float = 0.7,
    keep_year: bool = False,
    shift_dates: Optional[bool] = None,
    date_shift_days: Optional[int] = None,
    keep_mapping: bool = False,
    use_smart_merging: bool = True,
    lang: str = "en",
    normalize_accents: Optional[bool] = None,
    keep_alive: Optional[str] = None,
    *,
    runtime_provider: Optional[RuntimeProvider] = None,
) -> Dict[str, Any]:
    """De-identify text by masking, removing, replacing, hashing, or shifting PII."""
    from openmed.service.schemas import PIIDeidentifyRequest

    # Shared gateway: normalize text and guard the language before dispatch so
    # the MCP surface rejects the same bad inputs as the REST and library paths.
    text = normalize_text(text)
    lang = validate_language(lang, include_national_id=False)

    payload = PIIDeidentifyRequest(
        text=text,
        method=method,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        keep_year=keep_year,
        shift_dates=shift_dates,
        date_shift_days=date_shift_days,
        keep_mapping=keep_mapping,
        use_smart_merging=use_smart_merging,
        lang=lang,
        normalize_accents=normalize_accents,
        keep_alive=keep_alive,
    )
    runtime = _runtime(runtime_provider)

    def operation() -> Dict[str, Any]:
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

    response = _run_model_request(
        runtime,
        payload.model_name,
        payload.keep_alive,
        operation,
    )
    return validate_registered_tool_output("openmed_deidentify", response)


def openmed_list_models(
    category: Optional[str] = None,
    pii_language: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """List OpenMed registry models with optional category or PII language filters."""
    if pii_language is not None:
        pii_language = validate_language(
            pii_language,
            include_national_id=False,
        )

    models = openmed.get_all_models()

    if category:
        category_lower = category.strip().lower()
        models = {
            key: model
            for key, model in models.items()
            if model.category.lower() == category_lower
        }

    if pii_language is not None:
        allowed = openmed.get_pii_models_by_language(pii_language)
        models = {key: model for key, model in models.items() if key in allowed}

    limited_items = list(sorted(models.items()))[: max(limit, 0)]
    response = {
        "count": len(models),
        "returned": len(limited_items),
        "models": [_model_info_to_dict(key, model) for key, model in limited_items],
    }
    return validate_registered_tool_output("openmed_list_models", response)


def openmed_list_pii_languages() -> Dict[str, Any]:
    """List supported PII languages and their default model IDs."""
    languages = []
    for code in sorted(SUPPORTED_LANGUAGES | INDIC_NER_LANGUAGES):
        languages.append(
            {
                "code": code,
                "name": LANGUAGE_NAMES.get(code, code),
                "default_pii_model": DEFAULT_PII_MODELS[code],
                "model_count": len(openmed.get_pii_models_by_language(code)),
            }
        )
    response = {"count": len(languages), "languages": languages}
    return validate_registered_tool_output("openmed_list_pii_languages", response)


def openmed_loaded_models(
    *,
    runtime_provider: Optional[RuntimeProvider] = None,
) -> Dict[str, Any]:
    """Return currently loaded model resources for the MCP runtime."""
    response = _runtime(runtime_provider).loaded_models()
    return validate_registered_tool_output("openmed_loaded_models", response)


def openmed_fhir_bundle(
    resources: list[Dict[str, Any]],
    doc_id: str = "openmed-document",
    bundle_type: str = "transaction",
) -> Dict[str, Any]:
    """Assemble FHIR resources into a R4 bundle."""
    bundle = to_bundle(resources, doc_id=doc_id, bundle_type=bundle_type)
    return validate_registered_tool_output("openmed_fhir_bundle", bundle)


def openmed_risk_report(
    deidentified: Any,
    original: Optional[Any] = None,
    aux: Optional[Any] = None,
) -> Dict[str, Any]:
    """residual re-identification risk for de-identified records."""
    response = risk_report(deidentified, original, aux)
    return validate_registered_tool_output("openmed_risk_report", response)


def openmed_signed_audit_report(
    text: str,
    method: str = "mask",
    model_name: str = DEFAULT_PII_MODELS["en"],
    confidence_threshold: float = 0.7,
    lang: str = "en",
    signing_key: Optional[str] = None,
    key_id: str = "release",
    keep_alive: Optional[str] = None,
    *,
    runtime_provider: Optional[RuntimeProvider] = None,
) -> Dict[str, Any]:
    """returns a signed PHI-sage audit report"""
    text = normalize_text(text)
    lang = validate_language(lang, include_national_id=False)
    if not signing_key:
        raise ValueError("A signing key is required")
    runtime = _runtime(runtime_provider)

    def operation() -> Dict[str, Any]:
        report = openmed.deidentify(
            text,
            method=method,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            config=runtime.config,
            lang=lang,
            loader=runtime.get_loader(),
            audit=True,
        )
        report.sign(signing_key, key_id=key_id)
        return report.to_dict()

    response = _run_model_request(runtime, model_name, keep_alive, operation)
    return validate_registered_tool_output("openmed_signed_audit_report", response)


def openmed_search_models(
    category: Optional[str] = None,
    language: Optional[str] = None,
    max_size_mb: Optional[float] = None,
    license: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """search openmed model by category, language, size, license"""
    models = openmed.get_all_models()

    def _matches(model: ModelInfo) -> bool:
        if category and category.strip().lower() != model.category.strip().lower():
            return False
        if language:
            languages = {str(code).strip().lower() for code in model.languages}
            if language.strip().lower() not in languages:
                return False
        if max_size_mb is not None:
            size_mb = model.size_mb
            if size_mb is None or size_mb > max_size_mb:
                return False
        if license:
            declared = (model.license or "").strip().lower()
            if license.strip().lower() not in declared:
                return False
        return True

    matched = sorted((key, model) for key, model in models.items() if _matches(model))
    limited = matched[: max(limit, 0)]
    response = {
        "count": len(matched),
        "returned": len(limited),
        "models": [_model_info_to_dict(key, model) for key, model in limited],
    }
    return validate_registered_tool_output("openmed_search_models", response)


def openmed_health(
    *,
    runtime_provider: Optional[RuntimeProvider] = None,
) -> Dict[str, Any]:
    """Return a PHI-free MCP health summary."""

    loaded = _runtime(runtime_provider).loaded_models()
    models = loaded.get("models")
    loaded_model_count = len(models) if isinstance(models, Mapping) else 0
    return {
        "version": openmed.__version__,
        "loaded_model_count": loaded_model_count,
    }


def openmed_unload_model(
    model_name: Optional[str] = None,
    all_models: bool = False,
    *,
    runtime_provider: Optional[RuntimeProvider] = None,
) -> Dict[str, Any]:
    """Unload one inactive model or all inactive models from memory."""
    runtime = _runtime(runtime_provider)
    if all_models:
        response = runtime.unload_all_models()
        return validate_registered_tool_output("openmed_unload_model", response)
    if model_name is None:
        raise ValueError("model_name is required unless all_models=true")
    response = runtime.unload_model(validate_model_name(model_name))
    return validate_registered_tool_output("openmed_unload_model", response)


def openmed_run_workflow(
    pipeline: Dict[str, Any],
    session_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    *,
    runtime_provider: Optional[RuntimeProvider] = None,
) -> Dict[str, Any]:
    """Run a stateful multi-step workflow with PHI-safe result egress."""
    runtime = _runtime(runtime_provider)
    runner = WorkflowRunner(
        store=runtime.get_workflow_store(),
        executors=_workflow_step_executors(runtime_provider),
        deidentifier=_workflow_egress_deidentifier(runtime_provider),
    )
    response = runner.run(pipeline, session_id=session_id, workflow_id=workflow_id)
    return validate_registered_tool_output("openmed_run_workflow", response)


def openmed_ground(
    spans: list[Dict[str, Any]],
    vocabularies: Optional[list[str]] = None,
    max_candidates: int = 5,
    allow_external_llm: bool = False,
) -> Dict[str, Any]:
    """Ground text-free clinical spans with cache-only local vocabularies.

    The handler uses an input-only ``metadata.grounding_surface`` when present,
    falling back to the source ``entity_type`` label. Grounding surfaces and
    matched aliases are never returned. The emitted concept records contain
    only offsets, the existing span HMAC, and vocabulary-derived metadata.
    """

    del allow_external_llm
    validated, safe_spans = _prepare_clinical_spans(spans)
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")

    groundable = [
        (index, span)
        for index, span in enumerate(validated)
        if span.policy_label == "CLINICAL_CONCEPT"
    ]
    try:
        grounded = ground(
            [_grounding_input(span) for _, span in groundable],
            systems=vocabularies or DEFAULT_GROUNDING_SYSTEMS,
            loader=VocabLoader(local_only=True),
        )
        concepts: list[Dict[str, Any]] = []
        for (span_index, source_span), result in zip(groundable, grounded):
            candidates = tuple(result.candidates[:max_candidates])
            provenance = GroundingProvenance.from_candidates(
                start=source_span.start,
                end=source_span.end,
                candidates=candidates,
                method="composite",
                text_hash=source_span.text_hash,
                calibrated_score=result.calibrated_score,
                abstained=result.abstained or not candidates,
            ).to_dict()
            concept = {
                "span_index": span_index,
                "canonical_label": source_span.canonical_label,
                **provenance,
            }
            concepts.append(concept)
            metadata = dict(safe_spans[span_index]["metadata"])
            metadata["grounding"] = deepcopy(concept)
            safe_spans[span_index]["metadata"] = metadata
        response = {
            "schema_version": "openmed.ground.v1",
            "status": "completed",
            "spans": safe_spans,
            "grounded_concepts": concepts,
            "error": None,
        }
    except Exception as error:
        response = {
            "schema_version": "openmed.ground.v1",
            "status": "failed",
            "spans": safe_spans,
            "grounded_concepts": [],
            "error": _clinical_handler_error(
                "ground",
                "grounding_failed",
                "Local clinical grounding could not complete.",
                error,
            ),
        }
    return validate_registered_tool_output("openmed_ground", response)


def openmed_export_fhir(
    spans: list[Dict[str, Any]],
    resources: Optional[list[Dict[str, Any]]] = None,
    doc_id: str = "workflow",
    bundle_type: str = "collection",
) -> Dict[str, Any]:
    """Export prebuilt or locally grounded clinical artifacts to FHIR R4."""

    _, safe_spans = _prepare_clinical_spans(spans)
    try:
        fhir_resources = [
            _sanitize_fhir_resource(resource) for resource in (resources or [])
        ]
        for span in safe_spans:
            grounded_span = _grounded_span_from_artifact(span)
            if grounded_span is None:
                continue
            exported = to_fhir(grounded_span, document_id=doc_id)
            if exported is not None:
                fhir_resources.append(exported)

        bundle = to_bundle(
            fhir_resources,
            doc_id=doc_id,
            bundle_type=bundle_type,
        )
        response = {
            "schema_version": "openmed.export_fhir.v1",
            "status": "completed",
            "spans": safe_spans,
            "bundle": bundle,
            "resource_count": len(bundle.get("entry", [])),
            "error": None,
        }
    except Exception as error:
        response = {
            "schema_version": "openmed.export_fhir.v1",
            "status": "failed",
            "spans": safe_spans,
            "bundle": {},
            "resource_count": 0,
            "error": _clinical_handler_error(
                "export",
                "fhir_export_failed",
                "FHIR export could not complete.",
                error,
            ),
        }
    return validate_registered_tool_output("openmed_export_fhir", response)


def openmed_risk_score(
    spans: list[Dict[str, Any]],
    deidentified_text: Optional[str] = None,
    records: Optional[list[Dict[str, Any]]] = None,
    quasi_identifiers: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Score residual risk and return only the aggregate PHI-safe summary."""

    _, safe_spans = _prepare_clinical_spans(spans)
    try:
        risk_input: Any
        if records is not None:
            risk_input = deepcopy(records)
            record_count = len(records)
        else:
            risk_input = {
                "deidentified_text": deidentified_text or "",
                "spans": safe_spans,
            }
            record_count = 1 if deidentified_text is not None or safe_spans else 0
        detailed = risk_report(
            risk_input,
            quasi_identifier_fields=quasi_identifiers,
        )
        detailed["record_count"] = record_count
        response = {
            "schema_version": "openmed.risk_score.v1",
            "status": "completed",
            "spans": safe_spans,
            "risk_report": safe_risk_summary(detailed),
            "error": None,
        }
    except Exception as error:
        response = {
            "schema_version": "openmed.risk_score.v1",
            "status": "failed",
            "spans": safe_spans,
            "risk_report": {},
            "error": _clinical_handler_error(
                "risk",
                "risk_scoring_failed",
                "Residual-risk scoring could not complete.",
                error,
            ),
        }
    return validate_registered_tool_output("openmed_risk_score", response)


def openmed_clinical_pipeline(
    stages: list[str],
    text: Optional[str] = None,
    spans: Optional[list[Dict[str, Any]]] = None,
    options: Optional[Dict[str, Any]] = None,
    allow_external_llm: bool = False,
    session_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    *,
    runtime_provider: Optional[RuntimeProvider] = None,
    privacy_gateway_provider: Optional[PrivacyGatewayProvider] = None,
) -> Dict[str, Any]:
    """Execute a validated clinical pipeline with text-free artifact egress."""

    if text is None and spans is None:
        response = plan_clinical_pipeline(stages)
    else:
        preflight = plan_clinical_pipeline(stages)
        if preflight["status"] == "rejected":
            response = preflight
        else:
            normalized_text = normalize_text(text) if text is not None else None
            gateway = _clinical_external_stage_gateway(privacy_gateway_provider)
            response = execute_clinical_pipeline(
                stages,
                text=normalized_text,
                spans=spans,
                stage_handlers=_clinical_pipeline_stage_handlers(runtime_provider),
                options=options,
                allow_external_llm=allow_external_llm,
                external_stage_gateway=gateway,
                session_id=session_id,
                workflow_id=workflow_id,
            )
    return validate_registered_tool_output("openmed_clinical_pipeline", response)


def _clinical_pipeline_stage_handlers(
    runtime_provider: Optional[RuntimeProvider],
) -> Dict[str, Callable[..., Mapping[str, Any]]]:
    """Return local adapters for every declarative clinical pipeline stage."""

    return {
        "detect": lambda artifact, stage_options: _clinical_detect_stage(
            artifact,
            stage_options,
            runtime_provider=runtime_provider,
        ),
        "context": _clinical_context_stage,
        "sections": _clinical_sections_stage,
        "relations": _clinical_relations_stage,
        "ground": _clinical_ground_stage,
        "export": _clinical_export_stage,
        "risk": _clinical_risk_stage,
    }


def _clinical_detect_stage(
    artifact: ClinicalPipelineArtifact,
    stage_options: Mapping[str, Any],
    *,
    runtime_provider: Optional[RuntimeProvider],
) -> Mapping[str, Any]:
    """Run local detection and convert results to canonical text-free spans."""

    if artifact.spans:
        spans = artifact.public_spans()
        return {
            "spans": spans,
            "model_name": "provided-openmed-spans",
            "entity_count": len(spans),
        }
    if artifact.text is None:
        raise ValueError("the detect stage requires text or canonical spans")

    from openmed.core.labels import normalize_label, policy_label_for
    from openmed.core.pipeline import DEFAULT_HASH_SECRET
    from openmed.core.schemas import OpenMedSpan, hmac_text_hash

    options = dict(stage_options)
    doc_id = str(options.pop("doc_id", "clinical-pipeline"))
    language = str(options.pop("language", "en"))
    response = openmed_analyze_text(
        text=artifact.text,
        runtime_provider=runtime_provider,
        **options,
    )
    canonical_spans: list[dict[str, Any]] = []
    for entity in response.get("entities", []):
        if not isinstance(entity, Mapping):
            continue
        start = entity.get("start")
        end = entity.get("end")
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end < start
            or end > len(artifact.text)
        ):
            continue
        label = str(entity.get("label") or "OTHER")
        canonical_label = normalize_label(label, language)
        surface = artifact.text[start:end]
        canonical_spans.append(
            OpenMedSpan(
                doc_id=doc_id,
                start=start,
                end=end,
                text_hash=hmac_text_hash(surface, DEFAULT_HASH_SECRET),
                entity_type=label,
                canonical_label=canonical_label,
                policy_label=policy_label_for(canonical_label, language),
                score=float(entity.get("confidence") or 0.0),
                detector=str(response.get("model_name") or "openmed"),
                evidence={"pipeline_stage": "detect"},
                metadata={"pipeline_stage": "detect"},
            ).to_dict()
        )
    return {
        "spans": canonical_spans,
        "model_name": str(response.get("model_name") or "openmed"),
        "entity_count": len(canonical_spans),
    }


def _clinical_context_stage(
    artifact: ClinicalPipelineArtifact,
    stage_options: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Attach deterministic ConText axes without exposing source surfaces."""

    from openmed.clinical.context import resolve_span_context

    options = dict(stage_options)
    language = options.pop("language", None)
    if options:
        raise ValueError("the context stage received unsupported options")

    spans: list[dict[str, Any]] = []
    for span in artifact.public_spans():
        view: dict[str, Any] = {
            "start": span["start"],
            "end": span["end"],
        }
        if artifact.text is not None:
            view["document_text"] = artifact.text
        context = resolve_span_context(view, language=language)
        metadata = dict(span.get("metadata") or {})
        metadata["clinical_context"] = {
            "temporality": context.temporality,
            "certainty": context.certainty,
            "negation": context.negation,
        }
        span["metadata"] = metadata
        spans.append(span)
    return {"spans": spans, "context_count": len(spans)}


def _clinical_sections_stage(
    artifact: ClinicalPipelineArtifact,
    stage_options: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Detect local sections and attach canonical labels to span artifacts."""

    from openmed.clinical.sections import detect_sections

    if artifact.text is None:
        raise ValueError("the sections stage requires source text")
    options = dict(stage_options)
    sections = detect_sections(artifact.text, **options)
    safe_sections = [
        {key: deepcopy(value) for key, value in section.items() if key != "header"}
        for section in sections
    ]
    spans: list[dict[str, Any]] = []
    for span in artifact.public_spans():
        containing = next(
            (
                section
                for section in sections
                if section["start"] <= span["start"] < section["end"]
            ),
            None,
        )
        if containing is not None:
            span["section"] = str(containing["label"])
        spans.append(span)
    return {"spans": spans, "sections": safe_sections}


def _clinical_relations_stage(
    artifact: ClinicalPipelineArtifact,
    stage_options: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Run deterministic local relation extraction with surface-free output."""

    from openmed.clinical.relations import (
        available_multilingual_relation_languages,
        extract_relations,
    )

    if artifact.text is None:
        raise ValueError("the relations stage requires source text")
    options = dict(stage_options)
    language = str(options.pop("language", "en")).lower()
    relations: list[dict[str, Any]] = []
    if language in available_multilingual_relation_languages():
        relation_spans = [
            {
                "text": artifact.text[span["start"] : span["end"]],
                "label": span["canonical_label"],
                "start": span["start"],
                "end": span["end"],
                "score": span["score"] or 0.0,
                "section": span["section"],
            }
            for span in artifact.spans
        ]
        extracted = extract_relations(
            artifact.text,
            relation_spans,
            language=language,
            **options,
        )
        relations = [
            _safe_relation_payload(relation.to_dict()) for relation in extracted
        ]
    elif options:
        raise ValueError("relation options require a supported relation language")
    return {"spans": artifact.public_spans(), "relations": relations}


def _safe_relation_payload(relation: Mapping[str, Any]) -> dict[str, Any]:
    """Remove source surfaces from a relation result."""

    payload = deepcopy(dict(relation))
    for endpoint_name in ("head", "tail"):
        endpoint = payload.get(endpoint_name)
        if isinstance(endpoint, Mapping):
            payload[endpoint_name] = {
                key: deepcopy(value) for key, value in endpoint.items() if key != "text"
            }
    return payload


def _clinical_ground_stage(
    artifact: ClinicalPipelineArtifact,
    stage_options: Mapping[str, Any],
) -> Mapping[str, Any]:
    options = dict(stage_options)
    options.pop("allow_external_llm", None)
    return openmed_ground(
        spans=artifact.public_spans(),
        allow_external_llm=False,
        **options,
    )


def _clinical_export_stage(
    artifact: ClinicalPipelineArtifact,
    stage_options: Mapping[str, Any],
) -> Mapping[str, Any]:
    return openmed_export_fhir(
        spans=artifact.public_spans(),
        **dict(stage_options),
    )


def _clinical_risk_stage(
    artifact: ClinicalPipelineArtifact,
    stage_options: Mapping[str, Any],
) -> Mapping[str, Any]:
    return openmed_risk_score(
        spans=artifact.public_spans(),
        **dict(stage_options),
    )


def _clinical_external_stage_gateway(
    gateway_provider: Optional[PrivacyGatewayProvider],
) -> Callable[[str, Mapping[str, Any]], Mapping[str, Any]]:
    """Build a fail-closed external stage route over the privacy gateway."""

    def route(stage: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        from openmed.service.privacy_gateway import (
            HttpExternalLLMTransport,
            PrivacyGateway,
        )

        gateway = (
            gateway_provider()
            if gateway_provider is not None
            else PrivacyGateway(transport=HttpExternalLLMTransport.from_env())
        )
        result = gateway.complete(json.dumps(dict(request), sort_keys=True))
        response_text = getattr(result, "reidentified_text", None)
        if not isinstance(response_text, str):
            raise TypeError("privacy gateway returned no structured stage response")
        response = json.loads(response_text)
        if not isinstance(response, Mapping):
            raise TypeError("privacy gateway stage response must be an object")
        if stage != "ground":
            raise ValueError("unsupported external clinical stage")
        return dict(response)

    return route


_INPUT_ONLY_TEXT_KEYS = frozenset(
    {
        "deidentified_surface",
        "deidentified_text",
        "entity_text",
        "grounding_surface",
        "note",
        "span_text",
        "surface",
        "text",
        "word",
    }
)
_PATIENT_DIRECT_IDENTIFIER_FIELDS = frozenset(
    {
        "address",
        "birthDate",
        "contact",
        "generalPractitioner",
        "identifier",
        "link",
        "managingOrganization",
        "name",
        "photo",
        "telecom",
    }
)


def _prepare_clinical_spans(
    spans: Sequence[Mapping[str, Any]],
) -> tuple[list[OpenMedSpan], list[Dict[str, Any]]]:
    """Validate canonical spans and remove input-only text from artifacts."""

    if isinstance(spans, (str, bytes, bytearray)) or not isinstance(spans, Sequence):
        raise TypeError("spans must be a sequence of OpenMedSpan mappings")
    validated: list[OpenMedSpan] = []
    safe_spans: list[Dict[str, Any]] = []
    for index, payload in enumerate(spans):
        if not isinstance(payload, Mapping):
            raise TypeError(f"span at index {index} must be a mapping")
        span = OpenMedSpan.from_dict(payload)
        safe = span.to_dict()
        safe["evidence"] = _remove_input_text(safe["evidence"])
        safe["metadata"] = _remove_input_text(safe["metadata"])
        validated.append(span)
        safe_spans.append(safe)
    return validated, safe_spans


def _remove_input_text(value: Any) -> Any:
    """Deep-copy structured metadata while dropping raw-text carrier keys."""

    if isinstance(value, Mapping):
        return {
            str(key): _remove_input_text(item)
            for key, item in value.items()
            if str(key).casefold() not in _INPUT_ONLY_TEXT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_remove_input_text(item) for item in value]
    return deepcopy(value)


def _grounding_input(span: OpenMedSpan) -> Dict[str, Any]:
    """Adapt a text-free artifact to the local grounding facade."""

    metadata = span.metadata
    surface = next(
        (
            value.strip()
            for key in (
                "grounding_surface",
                "deidentified_surface",
                "surface",
                "text",
            )
            if isinstance((value := metadata.get(key)), str) and value.strip()
        ),
        span.entity_type,
    )
    language = metadata.get("source_language")
    return {
        "text": surface,
        "start": span.start,
        "end": span.end,
        "canonical_label": span.canonical_label,
        "source_language": language if isinstance(language, str) else "en",
    }


def _grounded_span_from_artifact(span: Mapping[str, Any]) -> GroundedSpan | None:
    """Rebuild a grounding object from safe MCP artifact metadata."""

    metadata = span.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    grounding = metadata.get("grounding")
    if not isinstance(grounding, Mapping) or grounding.get("abstained") is True:
        return None

    candidates: list[Candidate] = []
    chosen = _candidate_from_grounding_record(grounding)
    if chosen is not None:
        candidates.append(chosen)
    alternatives = grounding.get("alternatives")
    if isinstance(alternatives, list):
        candidates.extend(
            candidate
            for item in alternatives
            if isinstance(item, Mapping)
            if (candidate := _candidate_from_grounding_record(item)) is not None
        )
    if not candidates:
        return None

    export_metadata: Dict[str, Any] = {}
    value = metadata.get("value")
    if isinstance(value, (bool, int, float)):
        export_metadata["value"] = value
    unit = metadata.get("unit")
    if isinstance(unit, str) and unit.strip():
        export_metadata["unit"] = unit.strip()
    return GroundedSpan(
        text=candidates[0].display,
        start=int(span["start"]),
        end=int(span["end"]),
        candidates=tuple(candidates),
        canonical_label=str(span["canonical_label"]),
        metadata=export_metadata,
    )


def _candidate_from_grounding_record(
    record: Mapping[str, Any],
) -> Candidate | None:
    system = record.get("system")
    code = record.get("code")
    display = record.get("display")
    score = record.get("score")
    if not all(isinstance(value, str) and value for value in (system, code, display)):
        return None
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    return Candidate(
        system=system,
        code=code,
        display=display,
        score=float(score),
        source=str(record.get("source") or "mcp"),
        match_kind=str(record.get("match_kind") or ""),
        vocab_version=str(record.get("vocab_version") or "") or None,
    )


def _sanitize_fhir_resource(resource: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop direct Patient identifiers before deterministic Bundle assembly."""

    if not isinstance(resource, Mapping):
        raise TypeError("FHIR resources must be mappings")
    sanitized = deepcopy(dict(resource))
    if sanitized.get("resourceType") == "Patient":
        for field in _PATIENT_DIRECT_IDENTIFIER_FIELDS:
            sanitized.pop(field, None)
    return sanitized


def _clinical_handler_error(
    stage: str,
    code: str,
    message: str,
    error: Exception,
) -> Dict[str, Any]:
    """Return a deterministic error without including exception or input text."""

    return {
        "code": code,
        "message": message,
        "stage": stage,
        "details": {"error_type": error.__class__.__name__},
    }


def _workflow_step_executors(
    runtime_provider: Optional[RuntimeProvider],
) -> Dict[str, Callable[..., Any]]:
    executors = builtin_workflow_step_executors()
    executors.update(
        {
            "openmed_analyze_text": lambda **kwargs: openmed_analyze_text(
                runtime_provider=runtime_provider,
                **kwargs,
            ),
            "openmed_extract_pii": lambda **kwargs: openmed_extract_pii(
                runtime_provider=runtime_provider,
                **kwargs,
            ),
            "openmed_deidentify": lambda **kwargs: _workflow_deidentify_step(
                runtime_provider=runtime_provider,
                **kwargs,
            ),
        }
    )
    return executors


def _workflow_deidentify_step(
    *,
    runtime_provider: Optional[RuntimeProvider],
    text: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    if not isinstance(text, str):
        text = json.dumps(text, sort_keys=True)
    return openmed_deidentify(
        text=text,
        runtime_provider=runtime_provider,
        **kwargs,
    )


def _workflow_egress_deidentifier(
    runtime_provider: Optional[RuntimeProvider],
) -> Callable[[str], str]:
    def deidentify_text(text: str) -> str:
        response = openmed_deidentify(
            text=text,
            runtime_provider=runtime_provider,
        )
        deidentified = response.get("deidentified_text")
        if isinstance(deidentified, str):
            return deidentified
        return "[REDACTED_TEXT]" if text else text

    return deidentify_text


def build_mcp_tool_handlers(
    runtime_provider: Optional[RuntimeProvider],
) -> dict[str, Callable[..., Dict[str, Any]]]:
    """Return the MCP tool-name -> handler mapping bound to a runtime provider.

    Exposed at module level so the tool-schema drift guard can assert this set
    of registered tool names matches the canonical registry specs.
    """

    handlers: dict[str, Callable[..., Dict[str, Any]]] = {
        "openmed_analyze_text": lambda **kwargs: openmed_analyze_text(
            **kwargs,
            runtime_provider=runtime_provider,
        ),
        "openmed_extract_pii": lambda **kwargs: openmed_extract_pii(
            **kwargs,
            runtime_provider=runtime_provider,
        ),
        "openmed_deidentify": lambda **kwargs: openmed_deidentify(
            **kwargs,
            runtime_provider=runtime_provider,
        ),
        "openmed_list_models": lambda **kwargs: openmed_list_models(**kwargs),
        "openmed_list_pii_languages": (
            lambda **kwargs: openmed_list_pii_languages(**kwargs)
        ),
        "openmed_loaded_models": lambda **kwargs: openmed_loaded_models(
            **kwargs,
            runtime_provider=runtime_provider,
        ),
        "openmed_unload_model": lambda **kwargs: openmed_unload_model(
            **kwargs,
            runtime_provider=runtime_provider,
        ),
        "openmed_run_workflow": lambda **kwargs: openmed_run_workflow(
            **kwargs,
            runtime_provider=runtime_provider,
        ),
        "openmed_ground": lambda **kwargs: openmed_ground(**kwargs),
        "openmed_export_fhir": lambda **kwargs: openmed_export_fhir(**kwargs),
        "openmed_risk_score": lambda **kwargs: openmed_risk_score(**kwargs),
        "openmed_clinical_pipeline": (
            lambda **kwargs: openmed_clinical_pipeline(
                **kwargs,
                runtime_provider=runtime_provider,
            )
        ),
        "openmed_fhir_bundle": lambda **kwargs: openmed_fhir_bundle(**kwargs),
        "openmed_risk_report": lambda **kwargs: openmed_risk_report(**kwargs),
        "openmed_signed_audit_report": (
            lambda **kwargs: openmed_signed_audit_report(
                **kwargs,
                runtime_provider=runtime_provider,
            )
        ),
        "openmed_search_models": lambda **kwargs: openmed_search_models(**kwargs),
    }
    handlers.update(TOOL_REGISTRY.registered_handlers())
    return handlers


# Canonical set of MCP-exposed tool names, kept in sync with TOOL_REGISTRY by
# tests/unit/interop/test_tool_schema_sync.py.
MCP_TOOL_NAMES: frozenset[str] = frozenset(build_mcp_tool_handlers(None))


def _register_tools(
    server: Any,
    runtime_provider: Optional[RuntimeProvider],
    injection_guard: Optional[InjectionGuard] = None,
) -> None:
    handlers = build_mcp_tool_handlers(runtime_provider)
    for spec in TOOL_REGISTRY.latest_specs():
        server.tool(
            name=spec.name,
            title=spec.title,
            description=spec.description,
            annotations=_mcp_annotations(spec),
            structured_output=True,
        )(
            _render_structured_mcp_tool(
                spec,
                handlers[spec.name],
                injection_guard,
            )
        )
        _synchronize_registered_schemas(server, spec)


def _register_resources(
    server: Any,
    runtime_provider: Optional[RuntimeProvider] = None,
) -> None:
    @server.resource(
        "openmed://models",
        name="OpenMed model registry",
        mime_type="application/json",
    )
    def _models_resource() -> str:
        return _json_resource(openmed_list_models(limit=1000))

    @server.resource(
        "openmed://pii-languages",
        name="OpenMed PII languages",
        mime_type="application/json",
    )
    def _pii_languages_resource() -> str:
        return _json_resource(openmed_list_pii_languages())

    @server.resource(
        "openmed://examples",
        name="OpenMed safe examples",
        mime_type="application/json",
    )
    def _examples_resource() -> str:
        return _json_resource(
            {
                "analyze": "Patient received 75mg clopidogrel for NSTEMI.",
                "pii_extract": "Paciente: Maria Garcia, DNI: 12345678Z",
                "pii_deidentify": (
                    "Patient John Doe called 555-123-4567 on 01/15/2020."
                ),
            }
        )

    @server.resource(
        "openmed://tool-registry",
        name="OpenMed tool schema registry",
        mime_type="application/json",
    )
    def _tool_registry_resource() -> str:
        return _json_resource(render_tool_registry_document())

    @server.resource(
        CLINICAL_WORKFLOW_SPEC.resource_uri,
        name="OpenMed canonical clinical workflow",
        mime_type="application/json",
    )
    def _clinical_workflow_resource() -> str:
        return _json_resource(clinical_workflow_resource_document())

    @server.resource(
        CLINICAL_WORKFLOW_SPEC.fixture_uri,
        name="OpenMed synthetic clinical workflow golden run",
        mime_type="application/json",
    )
    def _clinical_workflow_fixture_resource() -> str:
        return _json_resource(load_golden_agent_run())

    @server.resource(
        "openmed://health",
        name="OpenMed health",
        mime_type="application/json",
    )
    def _health_resource() -> str:
        return _json_resource(openmed_health(runtime_provider=runtime_provider))


def _register_prompts(server: Any) -> None:
    @server.prompt(name=CLINICAL_WORKFLOW_SPEC.prompt_name)
    def _clinical_workflow_prompt(
        text: str = (
            "Synthetic subject Cedar Example, record SYN-1303-ALPHA, reports "
            "aster syndrome."
        ),
    ) -> str:
        """Prompt an agent to use the canonical local clinical workflow."""

        return render_clinical_workflow_prompt(text)

    @server.prompt(name="openmed-clinical-ner")
    def _clinical_ner_prompt(
        text: str = "Patient received 75mg clopidogrel for NSTEMI.",
        model_name: str = "disease_detection_superclinical",
    ) -> str:
        """Prompt an agent to use OpenMed clinical NER."""
        return (
            "Use the openmed_analyze_text tool on the provided clinical text. "
            f"Use model_name={model_name!r}. Text: {text!r}"
        )

    @server.prompt(name="openmed-pii-deidentify")
    def _pii_deidentify_prompt(
        text: str = "Patient John Doe called 555-123-4567 on 01/15/2020.",
        method: str = "mask",
        lang: str = "en",
    ) -> str:
        """Prompt an agent to de-identify synthetic or approved text."""
        return (
            "Use the openmed_deidentify tool. Confirm the user controls the "
            "OpenMed runtime before processing real PHI. "
            f"Use method={method!r}, lang={lang!r}. Text: {text!r}"
        )


def create_mcp_server(
    *,
    runtime_provider: Optional[RuntimeProvider] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    streamable_http_path: str = "/mcp",
    injection_guard_mode: Optional[str] = None,
) -> Any:
    """Create a FastMCP server exposing OpenMed tools, resources, and prompts."""
    if injection_guard_mode is None:
        injection_guard = InjectionGuard.from_env("OPENMED_MCP_INJECTION_GUARD_MODE")
    else:
        injection_guard = InjectionGuard(mode=injection_guard_mode)
    FastMCP = _structured_fastmcp(
        _load_fastmcp(),
        injection_guard=injection_guard,
    )
    server = FastMCP(
        "OpenMed",
        instructions=MCP_INSTRUCTIONS,
        website_url="https://openmed.life/docs/",
        host=host or os.getenv("OPENMED_MCP_HOST", "127.0.0.1"),
        port=port or _safe_int_env("OPENMED_MCP_PORT", 8081),
        streamable_http_path=streamable_http_path,
        stateless_http=True,
        json_response=True,
    )
    _register_tools(server, runtime_provider, injection_guard)
    _register_resources(server, runtime_provider)
    _register_prompts(server)
    return server


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OpenMed MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http", "http"),
        default=os.getenv("OPENMED_MCP_TRANSPORT", "stdio"),
        help="MCP transport. Defaults to stdio.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("OPENMED_MCP_HOST", "127.0.0.1"),
        help="Host for streamable HTTP transport.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_safe_int_env("OPENMED_MCP_PORT", 8081),
        help="Port for streamable HTTP transport.",
    )
    parser.add_argument(
        "--streamable-http-path",
        default=os.getenv("OPENMED_MCP_PATH", "/mcp"),
        help="Path for streamable HTTP transport.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the OpenMed package version and exit.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(openmed.__version__)
        return 0

    transport = args.transport
    if transport == "http":
        transport = "streamable-http"

    server = create_mcp_server(
        host=args.host,
        port=args.port,
        streamable_http_path=args.streamable_http_path,
    )
    server.run(transport=transport)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
