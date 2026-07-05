"""MCP server for OpenMed agent integrations."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

import openmed
from openmed.core.errors import (
    InputError,
    InternalError,
    MissingExtraError,
    OpenMedError,
)
from openmed.core.model_registry import ModelInfo
from openmed.core.pii_i18n import (
    DEFAULT_PII_MODELS,
    LANGUAGE_NAMES,
    SUPPORTED_LANGUAGES,
)
from openmed.mcp.tool_registry import (
    TOOL_REGISTRY,
    render_mcp_tool,
    render_tool_registry_document,
    validate_registered_tool_output,
)
from openmed.mcp.workflow import WorkflowRunner, builtin_workflow_step_executors
from openmed.service.runtime import ServiceRuntime
from openmed.utils.validation import validate_model_name

RuntimeProvider = Callable[[], ServiceRuntime]

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
        raise MissingExtraError(
            "The MCP SDK is not installed. Install OpenMed with the MCP extra: "
            'pip install "openmed[mcp]".',
            package="mcp",
            extra="mcp",
            feature="MCP server",
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
        raise InternalError(
            "Result to_dict() returned a "
            f"{type(payload).__name__}, expected a dict. This is an internal "
            "invariant violation; report it."
        )

    if isinstance(result, dict):
        return dict(result)

    raise InternalError(
        f"Cannot serialize result of type {type(result).__name__}. Expected an "
        "OpenMed result with a to_dict() method or a dict; this is an internal "
        "invariant violation, report it."
    )


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


#: Stable, documented MCP error codes for each taxonomy class. The service layer
#: (:mod:`openmed.service.app`) maps the same classes to matching HTTP codes, so
#: an ``input_error`` here corresponds to a ``400`` there. These strings are part
#: of the public MCP contract and must stay stable across releases.
MCP_ERROR_CODES: Dict[str, str] = {
    "OpenMedError": "openmed_error",
    "InputError": "input_error",
    "ConfigurationError": "configuration_error",
    "CapabilityError": "capability_error",
    "MissingExtraError": "missing_extra",
    "ModelLoadError": "model_load_error",
    "PolicyError": "policy_error",
    "BudgetExceededError": "budget_exceeded",
    "InternalError": "internal_error",
    "InferenceError": "inference_error",
}


def mcp_error_payload(exc: OpenMedError) -> Dict[str, Any]:
    """Map an :class:`OpenMedError` to a stable, PHI-free MCP error payload.

    Args:
        exc: A taxonomy exception raised by an OpenMed tool.

    Returns:
        A dict with a stable ``code`` (from :data:`MCP_ERROR_CODES`, falling
        back to the exception's own ``code``), the actionable ``message``, and
        the PHI-free ``details`` mapping. Suitable for returning as an MCP tool
        error result.
    """

    code = MCP_ERROR_CODES.get(type(exc).__name__, exc.code)
    return {
        "error": {
            "code": code,
            "message": exc.message,
            "details": dict(exc.details),
        }
    }


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
    models = openmed.get_all_models()

    if category:
        category_lower = category.strip().lower()
        models = {
            key: model
            for key, model in models.items()
            if model.category.lower() == category_lower
        }

    if pii_language:
        if pii_language not in SUPPORTED_LANGUAGES:
            supported = sorted(SUPPORTED_LANGUAGES)
            raise InputError(
                f"Unsupported language '{pii_language}'. Pass one of the "
                f"supported ISO 639-1 codes: {supported}.",
                details={"lang": pii_language, "supported": supported},
            )
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
    for code in sorted(SUPPORTED_LANGUAGES):
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
        raise InputError(
            "model_name is required unless all_models=true. Pass a model_name to "
            "unload one model, or set all_models=true to unload all inactive models."
        )
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


def _register_tools(
    server: Any,
    runtime_provider: Optional[RuntimeProvider],
) -> None:
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
    }

    for spec in TOOL_REGISTRY.latest_specs():
        server.tool(name=spec.name)(render_mcp_tool(spec, handlers[spec.name]))


def _register_resources(server: Any) -> None:
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


def _register_prompts(server: Any) -> None:
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
) -> Any:
    """Create a FastMCP server exposing OpenMed tools, resources, and prompts."""
    FastMCP = _load_fastmcp()
    server = FastMCP(
        "OpenMed",
        instructions=MCP_INSTRUCTIONS,
        website_url="https://openmed.life/docs/",
        host=host or os.getenv("OPENMED_MCP_HOST", "127.0.0.1"),
        port=port or int(os.getenv("OPENMED_MCP_PORT", "8081")),
        streamable_http_path=streamable_http_path,
        stateless_http=True,
        json_response=True,
    )
    _register_tools(server, runtime_provider)
    _register_resources(server)
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
        default=int(os.getenv("OPENMED_MCP_PORT", "8081")),
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
