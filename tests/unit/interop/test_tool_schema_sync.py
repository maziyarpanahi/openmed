"""Drift guard: registry, MCP, and framework-adapter tool schemas must agree.

The canonical tool contract lives in ``openmed.mcp.tool_registry.TOOL_REGISTRY``.
Every other agent-facing surface -- the MCP server's registered tools, each
framework adapter's rendered definitions, and the committed ``tools.json``
bundle -- is derived from it. This suite fails loudly if any surface drifts
(a tool added to one place but not another, or an input schema that diverges),
which is what makes the CLI/tool surface reliable for programmatic agents.
"""

from __future__ import annotations

import json
from pathlib import Path

from openmed.interop import (
    adapter_tool_definitions,
    available_adapters,
    to_function_tools,
    to_tool_use_tools,
)
from openmed.mcp import server as mcp_server
from openmed.mcp.tool_registry import TOOL_REGISTRY, render_tool_registry_document

TOOLS_JSON = Path(__file__).resolve().parents[3] / "openmed" / "interop" / "tools.json"


def _registry_names() -> set[str]:
    return {spec.name for spec in TOOL_REGISTRY.latest_specs()}


def _registry_schema_projection() -> dict[str, tuple]:
    return {
        spec.name: (
            json.dumps(spec.input_schema, sort_keys=True),
            json.dumps(spec.output_schema, sort_keys=True),
        )
        for spec in TOOL_REGISTRY.latest_specs()
    }


def _definition_schema_projection(definitions) -> dict[str, tuple]:
    return {
        definition["name"]: (
            json.dumps(definition["input_schema"], sort_keys=True),
            json.dumps(definition["output_schema"], sort_keys=True),
        )
        for definition in definitions
    }


# ---------------------------------------------------------------------------
# Registry <-> MCP
# ---------------------------------------------------------------------------


def test_mcp_handler_set_matches_registry():
    # The module-level handler mapping (source of the MCP tool set) must expose
    # exactly the registry's tools -- a spec without a handler, or a handler
    # without a spec, is drift.
    handlers = mcp_server.build_mcp_tool_handlers(None)
    assert set(handlers) == _registry_names()
    assert mcp_server.MCP_TOOL_NAMES == frozenset(_registry_names())


def test_mcp_server_registers_exactly_the_registry_tools():
    class _FakeServer:
        def __init__(self):
            self.tools = {}

        def tool(self, *, name):
            def _decorator(func):
                self.tools[name] = func
                return func

            return _decorator

    fake = _FakeServer()
    mcp_server._register_tools(fake, runtime_provider=None)
    assert set(fake.tools) == _registry_names()


# ---------------------------------------------------------------------------
# Registry <-> framework adapters
# ---------------------------------------------------------------------------


def test_every_adapter_agrees_with_registry_schemas():
    registry = _registry_schema_projection()
    adapters = available_adapters()
    assert adapters, "expected at least one registered framework adapter"

    for adapter in adapters:
        projection = _definition_schema_projection(adapter_tool_definitions(adapter))
        assert projection == registry, f"adapter {adapter!r} drifted from registry"


def test_generic_tool_renderers_cover_the_registry_by_name():
    registry_names = _registry_names()
    # Function-calling shape nests the name under "function"; tool-use is flat.
    assert {tool["function"]["name"] for tool in to_function_tools()} == registry_names
    assert {tool["name"] for tool in to_tool_use_tools()} == registry_names


def test_tool_use_input_schemas_match_registry():
    registry = {
        spec.name: json.dumps(spec.input_schema, sort_keys=True)
        for spec in TOOL_REGISTRY.latest_specs()
    }
    rendered = {
        tool["name"]: json.dumps(tool["input_schema"], sort_keys=True)
        for tool in to_tool_use_tools()
    }
    assert rendered == registry


# ---------------------------------------------------------------------------
# Committed tools.json bundle must not drift from the registry
# ---------------------------------------------------------------------------


def test_tools_json_bundle_matches_registry_document():
    assert TOOLS_JSON.exists(), (
        "openmed/interop/tools.json missing; regenerate it from "
        "render_tool_registry_document()"
    )
    committed = json.loads(TOOLS_JSON.read_text(encoding="utf-8"))
    assert committed == render_tool_registry_document(), (
        "tools.json is stale; regenerate it from the tool registry"
    )
