"""Lazy interoperability adapter registry.

Adapters live behind explicit imports so importing :mod:`openmed` or
``openmed.interop`` never imports optional third-party detector dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any, Final


@dataclass(frozen=True)
class AdapterSpec:
    """Registry metadata for one optional interoperability adapter."""

    name: str
    module: str
    extra: str
    description: str


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
    "langchain": AdapterSpec(
        name="langchain",
        module="openmed.interop.langchain",
        extra="langchain",
        description="LangChain redaction runnable adapter",
    ),
    "presidio": AdapterSpec(
        name="presidio",
        module="openmed.interop.presidio",
        extra="presidio",
        description="Presidio RecognizerResult adapter",
    ),
    "philter": AdapterSpec(
        name="philter",
        module="openmed.interop.philter",
        extra="philter",
        description="Philter PHI span adapter",
    ),
    "pydeid": AdapterSpec(
        name="pydeid",
        module="openmed.interop.pydeid",
        extra="pydeid",
        description="pyDeid PHI span adapter",
    ),
    "gliner_biomed": AdapterSpec(
        name="gliner_biomed",
        module="openmed.interop.gliner_biomed",
        extra="gliner",
        description="GLiNER-BioMed zero-shot entity adapter",
    ),
    "spacy": AdapterSpec(
        name="spacy",
        module="openmed.interop.spacy_component",
        extra="spacy",
        description="spaCy pipeline component for OpenMed PII spans",
    ),
}

_GATEWAY_EXPORTS: Final[dict[str, str]] = {
    "PrivacyGateway": "PrivacyGateway",
    "PrivacyGatewayConfig": "PrivacyGatewayConfig",
    "RedactionMapping": "RedactionMapping",
    "assert_redacted": "assert_redacted",
    "restore_text": "restore_text",
}


def available_adapters() -> tuple[str, ...]:
    """Return registered adapter names without importing adapter modules."""

    return tuple(sorted(_ADAPTERS))


def adapter_spec(name: str) -> AdapterSpec:
    """Return registry metadata for *name* without importing the adapter."""

    key = _normalize_adapter_name(name)
    try:
        return _ADAPTERS[key]
    except KeyError as exc:
        known = ", ".join(available_adapters())
        raise KeyError(f"unknown interop adapter {name!r}; available: {known}") from exc


def get_adapter(name: str) -> ModuleType:
    """Import and return an adapter module by name."""

    spec = adapter_spec(name)
    return import_module(spec.module)


def adapter_tool_definitions(name: str) -> tuple[dict[str, Any], ...]:
    """Return registry-rendered tool definitions for an adapter."""

    spec = adapter_spec(name)
    from openmed.mcp.tool_registry import render_adapter_tool_definitions

    return render_adapter_tool_definitions(spec.name)


def _normalize_adapter_name(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def __getattr__(name: str) -> Any:
    if name in _ADAPTERS:
        return get_adapter(name)
    if name == "gateway":
        return import_module("openmed.interop.gateway")
    if name in _GATEWAY_EXPORTS:
        module = import_module("openmed.interop.gateway")
        return getattr(module, _GATEWAY_EXPORTS[name])
    raise AttributeError(name)


__all__ = [
    "AdapterSpec",
    "PrivacyGateway",
    "PrivacyGatewayConfig",
    "RedactionMapping",
    "adapter_tool_definitions",
    "adapter_spec",
    "assert_redacted",
    "available_adapters",
    "gateway",
    "get_adapter",
    "restore_text",
]
