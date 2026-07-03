"""LlamaIndex tool adapters rendered from the OpenMed tool registry."""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any

from openmed.interop.function_tools import (
    RuntimeProvider,
    create_tool_callable,
    registry_tool_specs,
)
from openmed.mcp.tool_registry import render_adapter_tool_definitions


def create_tool_definitions() -> tuple[dict[str, Any], ...]:
    """Return LlamaIndex-facing OpenMed tool definitions from the registry."""

    return render_adapter_tool_definitions("llamaindex")


def get_llamaindex_tools(
    *,
    runtime_provider: RuntimeProvider | None = None,
) -> tuple[Any, ...]:
    """Return LlamaIndex ``FunctionTool`` objects for every registry tool."""

    function_tool = _load_function_tool()
    return tuple(
        _function_tool_from_spec(function_tool, spec, runtime_provider)
        for spec in registry_tool_specs()
    )


def _function_tool_from_spec(
    function_tool: Any,
    spec: Any,
    runtime_provider: RuntimeProvider | None,
) -> Any:
    func = create_tool_callable(spec, runtime_provider=runtime_provider)
    if hasattr(function_tool, "from_defaults"):
        return function_tool.from_defaults(
            fn=func,
            name=spec.name,
            description=spec.description,
        )
    if hasattr(function_tool, "from_function"):
        return function_tool.from_function(
            func=func,
            name=spec.name,
            description=spec.description,
        )
    raise ImportError("LlamaIndex tools require llama-index-core with FunctionTool.")


def _load_function_tool() -> Any:
    try:
        module = _import_module("llama_index.core.tools")
    except ImportError as exc:
        raise ImportError(
            "LlamaIndex tools require the 'llamaindex' extra. "
            "Install with `pip install openmed[llamaindex]`."
        ) from exc

    try:
        return module.FunctionTool
    except AttributeError as exc:
        raise ImportError(
            "LlamaIndex tools require llama-index-core with FunctionTool."
        ) from exc


__all__ = [
    "create_tool_definitions",
    "get_llamaindex_tools",
]
