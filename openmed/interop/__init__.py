"""Interoperability package for section 4.2.

Intended contents include optional lazily-imported adapters that emit canonical
spans, plus bridges/ for subprocess-only integrations. Adapters live behind
explicit imports so importing :mod:`openmed` or ``openmed.interop`` never imports
optional third-party detector dependencies.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from types import ModuleType
from typing import Any, Final

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdapterSpec:
    """Registry metadata for one optional interoperability adapter."""

    name: str
    module: str
    extra: str
    description: str
    kind: str = "builtin"
    plugin_id: str = ""
    component_id: str = ""
    loaded_by_policy_opt_in: bool = False

    @property
    def is_plugin(self) -> bool:
        """Return whether this spec represents a third-party SDK component."""

        return bool(self.plugin_id and self.component_id)

    @property
    def qualified_id(self) -> str:
        """Return stable plugin provenance or the built-in adapter name."""

        if self.is_plugin:
            return f"{self.plugin_id}:{self.component_id}"
        return self.name


_ADAPTERS: Final[dict[str, AdapterSpec]] = {
    "cda": AdapterSpec(
        name="cda",
        module="openmed.interop.cda",
        extra="core",
        description="CDA/C-CDA XML de-identification adapter",
    ),
    "hl7v2": AdapterSpec(
        name="hl7v2",
        module="openmed.interop.hl7v2",
        extra="",
        description="HL7 v2 segment-aware de-identification",
    ),
    "indic": AdapterSpec(
        name="indic",
        module="openmed.interop.indic",
        extra="indic",
        description="Indic segmentation and transliteration helpers",
    ),
    "icd11_api": AdapterSpec(
        name="icd11_api",
        module="openmed.interop.icd11_api",
        extra="",
        description="Offline ICD-11 MMS snapshot grounding and builder",
    ),
    "duckdb": AdapterSpec(
        name="duckdb",
        module="openmed.interop.duckdb_udf",
        extra="duckdb",
        description="DuckDB scalar UDFs for in-query de-identification",
    ),
    "function_tools": AdapterSpec(
        name="function_tools",
        module="openmed.interop.function_tools",
        extra="",
        description="Generic function-calling and tool-use schema adapters",
    ),
    "graph_orchestration": AdapterSpec(
        name="graph_orchestration",
        module="openmed.interop.graph_orchestration",
        extra="langgraph",
        description="State-graph de-identification and re-identification nodes",
    ),
    "langchain": AdapterSpec(
        name="langchain",
        module="openmed.interop.langchain",
        extra="langchain",
        description="LangChain redaction runnable adapter",
    ),
    "llamaindex": AdapterSpec(
        name="llamaindex",
        module="openmed.interop.llamaindex",
        extra="llamaindex",
        description="LlamaIndex node redaction and FunctionTool adapters",
    ),
    "pandas": AdapterSpec(
        name="pandas",
        module="openmed.interop.pandas_accessor",
        extra="pandas",
        description="Pandas DataFrame de-identification accessor",
    ),
    "presidio": AdapterSpec(
        name="presidio",
        module="openmed.interop.presidio",
        extra="presidio",
        description="Presidio RecognizerResult adapter",
    ),
    "quickumls": AdapterSpec(
        name="quickumls",
        module="openmed.interop.quickumls",
        extra="quickumls",
        description="QuickUMLS licensed-resource linker adapter",
    ),
    "scispacy_linker": AdapterSpec(
        name="scispacy_linker",
        module="openmed.interop.scispacy_linker",
        extra="scispacy",
        description="scispaCy UMLS entity-linker adapter",
    ),
    "philter": AdapterSpec(
        name="philter",
        module="openmed.interop.philter",
        extra="philter",
        description="Philter PHI span adapter",
    ),
    "polars": AdapterSpec(
        name="polars",
        module="openmed.interop.polars_accessor",
        extra="polars",
        description="Polars DataFrame de-identification helpers",
    ),
    "prefect": AdapterSpec(
        name="prefect",
        module="openmed.interop.prefect_tasks",
        extra="prefect",
        description="Prefect task and flow for batch de-identification",
    ),
    "pydeid": AdapterSpec(
        name="pydeid",
        module="openmed.interop.pydeid",
        extra="pydeid",
        description="pyDeid PHI span adapter",
    ),
    "ray": AdapterSpec(
        name="ray",
        module="openmed.interop.ray_data",
        extra="ray",
        description="Ray Data actor operator for batch column de-identification",
    ),
    "scrubadub": AdapterSpec(
        name="scrubadub",
        module="openmed.interop.scrubadub",
        extra="scrubadub",
        description="scrubadub Filth span adapter",
    ),
    "search_pipeline": AdapterSpec(
        name="search_pipeline",
        module="openmed.interop.search_pipeline",
        extra="haystack",
        description="Modular search-pipeline redaction components",
    ),
    "spark": AdapterSpec(
        name="spark",
        module="openmed.interop.spark_udf",
        extra="spark",
        description="PySpark pandas_udf for batch column de-identification",
    ),
    "gliner_biomed": AdapterSpec(
        name="gliner_biomed",
        module="openmed.interop.gliner_biomed",
        extra="gliner",
        description="GLiNER-BioMed zero-shot entity adapter",
    ),
    "haystack": AdapterSpec(
        name="haystack",
        module="openmed.interop.haystack",
        extra="haystack",
        description="Haystack document redaction component",
    ),
    "spacy": AdapterSpec(
        name="spacy",
        module="openmed.interop.spacy_component",
        extra="spacy",
        description="spaCy pipeline component for OpenMed PII spans",
    ),
    "zh": AdapterSpec(
        name="zh",
        module="openmed.interop.zh",
        extra="zh",
        description="Chinese segmentation, script conversion, and pinyin helpers",
    ),
    "cdm_etl": AdapterSpec(
        name="cdm_etl",
        module="openmed.interop.cdm_etl",
        extra="",
        description="Deterministic clinical note to CDM-style ETL helpers",
    ),
    "omop": AdapterSpec(
        name="omop",
        module="openmed.interop.omop",
        extra="",
        description="OMOP CDM loader for grounded clinical note spans",
    ),
    "openmrs": AdapterSpec(
        name="openmrs",
        module="openmed.interop.openmrs",
        extra="openmrs",
        description="Local-first OpenMRS REST and FHIR2 de-identification adapter",
    ),
}

_PLUGIN_ADAPTERS: dict[str, tuple[AdapterSpec, Any]] = {}
_PLUGIN_ADAPTER_LOCK = RLock()

_GATEWAY_EXPORTS: Final[dict[str, str]] = {
    "PrivacyGateway": "PrivacyGateway",
    "PrivacyGatewayConfig": "PrivacyGatewayConfig",
    "RedactionMapping": "RedactionMapping",
    "assert_redacted": "assert_redacted",
    "restore_text": "restore_text",
}
_TOOL_EXPORTS: Final[frozenset[str]] = frozenset(
    {"TOOLS", "ToolDefinition", "get_tool", "list_tools"}
)


def available_adapters(
    *,
    include_plugins: bool = False,
    allow_network_egress: bool = False,
    allow_non_permissive_licenses: bool = False,
    opt_in_plugins: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return registered adapter and exporter names.

    Built-in listing remains dependency-light. Set ``include_plugins`` to run
    explicit SDK discovery before returning first- and third-party names.
    """

    if include_plugins:
        discover_plugin_adapters(
            allow_network_egress=allow_network_egress,
            allow_non_permissive_licenses=allow_non_permissive_licenses,
            opt_in_plugins=opt_in_plugins,
        )

    with _PLUGIN_ADAPTER_LOCK:
        plugin_names = tuple(_PLUGIN_ADAPTERS)
    return tuple(sorted((*_ADAPTERS, *plugin_names)))


def adapter_spec(name: str) -> AdapterSpec:
    """Return registry metadata for *name* without importing the adapter."""

    plugin_name = str(name or "").strip()
    with _PLUGIN_ADAPTER_LOCK:
        plugin_entry = _PLUGIN_ADAPTERS.get(plugin_name)
    if plugin_entry is not None:
        return plugin_entry[0]

    key = _normalize_adapter_name(name)
    try:
        return _ADAPTERS[key]
    except KeyError as exc:
        known = ", ".join(available_adapters())
        raise KeyError(f"unknown interop adapter {name!r}; available: {known}") from exc


def get_adapter(name: str) -> ModuleType | Any:
    """Return a built-in adapter module or registered plugin component."""

    plugin_name = str(name or "").strip()
    with _PLUGIN_ADAPTER_LOCK:
        plugin_entry = _PLUGIN_ADAPTERS.get(plugin_name)
    if plugin_entry is not None:
        return plugin_entry[1]

    spec = adapter_spec(name)
    module = import_module(spec.module)
    ensure_registered = getattr(module, "ensure_registered", None)
    if ensure_registered is not None:
        ensure_registered()
    return module


def discover_plugin_adapters(
    *,
    allow_network_egress: bool = False,
    allow_non_permissive_licenses: bool = False,
    opt_in_plugins: Sequence[str] = (),
) -> tuple[AdapterSpec, ...]:
    """Discover accepted SDK exporters and interop adapters.

    Policy-restricted components are returned by the SDK only when the caller
    supplies an explicit component opt-in or broad policy flag.

    Args:
        allow_network_egress: Allow SDK plugins declaring network access.
        allow_non_permissive_licenses: Allow restricted plugin licenses.
        opt_in_plugins: Plugin or qualified component ids explicitly enabled.

    Returns:
        Newly registered plugin adapter metadata.
    """

    try:
        registrations = _iter_sdk_plugins(
            allow_network_egress=allow_network_egress,
            allow_non_permissive_licenses=allow_non_permissive_licenses,
            opt_in_plugins=opt_in_plugins,
        )
    except Exception as exc:  # pragma: no cover - defensive SDK boundary
        logger.warning(
            "Failed to discover OpenMed SDK interop plugins: %s",
            exc.__class__.__name__,
        )
        return ()
    return register_plugin_adapters(registrations)


def register_plugin_adapters(
    registrations: Iterable[Any],
) -> tuple[AdapterSpec, ...]:
    """Register validated SDK exporters and interop adapter components."""

    registered: list[AdapterSpec] = []
    for registration in registrations:
        try:
            metadata = registration.metadata
            if metadata.kind not in {"exporter", "interop_adapter"}:
                continue
            name = metadata.qualified_id
            spec = AdapterSpec(
                name=name,
                module="",
                extra="",
                description=metadata.description or metadata.name or name,
                kind=metadata.kind,
                plugin_id=metadata.plugin_id,
                component_id=metadata.component_id,
                loaded_by_policy_opt_in=registration.loaded_by_policy_opt_in,
            )
            with _PLUGIN_ADAPTER_LOCK:
                existing = _PLUGIN_ADAPTERS.get(name)
                if existing is None:
                    _PLUGIN_ADAPTERS[name] = (spec, registration.component)
                elif existing[0] != spec:
                    raise ValueError(
                        "plugin adapter metadata changed after registration"
                    )
        except Exception as exc:
            logger.warning(
                "Failed to wire OpenMed SDK interop component %s: %s",
                _safe_registration_id(registration),
                exc.__class__.__name__,
            )
            continue
        registered.append(spec)
    return tuple(registered)


def adapter_tool_definitions(name: str) -> tuple[dict[str, Any], ...]:
    """Return registry-rendered tool definitions for an adapter."""

    spec = adapter_spec(name)
    from openmed.mcp.tool_registry import render_adapter_tool_definitions

    return render_adapter_tool_definitions(spec.name)


def to_function_tools() -> tuple[dict[str, Any], ...]:
    """Return generic function-calling tool definitions."""

    from openmed.interop.function_tools import to_function_tools as _render

    return _render()


def to_tool_use_tools() -> tuple[dict[str, Any], ...]:
    """Return generic tool-use input-schema definitions."""

    from openmed.interop.function_tools import to_tool_use_tools as _render

    return _render()


def get_langchain_tools() -> tuple[Any, ...]:
    """Return LangChain tool objects for every OpenMed registry tool."""

    from openmed.interop.langchain import get_langchain_tools as _render

    return _render()


def get_llamaindex_tools() -> tuple[Any, ...]:
    """Return LlamaIndex tool objects for every OpenMed registry tool."""

    from openmed.interop.llamaindex import get_llamaindex_tools as _render

    return _render()


def _normalize_adapter_name(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def _iter_sdk_plugins(**policy: Any) -> tuple[Any, ...]:
    """Return SDK registrations without importing plugins at module import."""

    try:
        registry = import_module("openmed.plugins.registry")
    except ModuleNotFoundError as exc:
        if exc.name in {"openmed.plugins.protocols", "openmed.plugins.registry"}:
            return ()
        raise
    return tuple(registry.iter_plugins(None, **policy))


def _safe_registration_id(registration: Any) -> str:
    try:
        value = registration.metadata.qualified_id
    except Exception:
        return "<unknown>"
    return value if isinstance(value, str) and value else "<unknown>"


def _reset_plugin_adapters_for_tests() -> None:
    with _PLUGIN_ADAPTER_LOCK:
        _PLUGIN_ADAPTERS.clear()


def __getattr__(name: str) -> Any:
    if name in _ADAPTERS:
        return get_adapter(name)
    if name == "gateway":
        return import_module("openmed.interop.gateway")
    if name in _GATEWAY_EXPORTS:
        module = import_module("openmed.interop.gateway")
        return getattr(module, _GATEWAY_EXPORTS[name])
    if name in _TOOL_EXPORTS:
        module = import_module("openmed.interop.tools")
        return getattr(module, name)
    raise AttributeError(name)


__all__ = [
    "AdapterSpec",
    "PrivacyGateway",
    "PrivacyGatewayConfig",
    "RedactionMapping",
    "TOOLS",
    "ToolDefinition",
    "adapter_tool_definitions",
    "adapter_spec",
    "assert_redacted",
    "available_adapters",
    "discover_plugin_adapters",
    "gateway",
    "get_adapter",
    "get_langchain_tools",
    "get_llamaindex_tools",
    "get_tool",
    "list_tools",
    "register_plugin_adapters",
    "restore_text",
    "to_function_tools",
    "to_tool_use_tools",
]
