"""Generic agent tool definitions rendered from the OpenMed tool registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any

from openmed.mcp.tool_registry import (
    TOOL_REGISTRY,
    ToolSpec,
    render_mcp_tool,
)

RuntimeProvider = Callable[[], Any]


def registry_tool_specs(
    specs: Iterable[ToolSpec] | None = None,
) -> tuple[ToolSpec, ...]:
    """Return the registry specs used by framework adapter renderers."""

    return tuple(specs) if specs is not None else TOOL_REGISTRY.latest_specs()


def to_function_tools(
    specs: Iterable[ToolSpec] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Render generic function-calling JSON tool definitions."""

    return tuple(_function_tool_definition(spec) for spec in registry_tool_specs(specs))


def to_tool_use_tools(
    specs: Iterable[ToolSpec] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Render generic tool-use definitions with ``input_schema`` payloads."""

    return tuple(_tool_use_definition(spec) for spec in registry_tool_specs(specs))


def create_tool_callable(
    spec: ToolSpec,
    *,
    runtime_provider: RuntimeProvider | None = None,
) -> Callable[..., dict[str, Any]]:
    """Return a callable wrapper for one registry tool spec."""

    tool = render_mcp_tool(
        spec,
        _tool_handler(spec.name, runtime_provider=runtime_provider),
    )
    tool.__name__ = spec.name
    return tool


def _function_tool_definition(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": deepcopy(dict(spec.input_schema)),
        },
    }


def _tool_use_definition(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": deepcopy(dict(spec.input_schema)),
    }


def _tool_handler(
    name: str,
    *,
    runtime_provider: RuntimeProvider | None,
) -> Callable[..., dict[str, Any]]:
    from openmed.mcp import server as mcp_server

    handlers: dict[str, Callable[..., dict[str, Any]]] = {
        "openmed_analyze_text": lambda **kwargs: mcp_server.openmed_analyze_text(
            **kwargs,
            runtime_provider=runtime_provider,
        ),
        "openmed_extract_pii": lambda **kwargs: mcp_server.openmed_extract_pii(
            **kwargs,
            runtime_provider=runtime_provider,
        ),
        "openmed_deidentify": lambda **kwargs: mcp_server.openmed_deidentify(
            **kwargs,
            runtime_provider=runtime_provider,
        ),
        "openmed_list_models": lambda **kwargs: mcp_server.openmed_list_models(
            **kwargs
        ),
        "openmed_list_pii_languages": (
            lambda **kwargs: mcp_server.openmed_list_pii_languages(**kwargs)
        ),
        "openmed_loaded_models": lambda **kwargs: mcp_server.openmed_loaded_models(
            **kwargs,
            runtime_provider=runtime_provider,
        ),
        "openmed_unload_model": lambda **kwargs: mcp_server.openmed_unload_model(
            **kwargs,
            runtime_provider=runtime_provider,
        ),
    }
    try:
        return handlers[name]
    except KeyError as exc:
        raise KeyError(f"unknown OpenMed tool {name!r}") from exc


__all__ = [
    "create_tool_callable",
    "registry_tool_specs",
    "to_function_tools",
    "to_tool_use_tools",
]
