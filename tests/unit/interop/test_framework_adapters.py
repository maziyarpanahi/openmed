from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace

import pytest

from openmed.interop import (
    adapter_spec,
    adapter_tool_definitions,
    get_adapter,
    to_function_tools,
    to_tool_use_tools,
)
from openmed.interop import langchain as langchain_adapter
from openmed.interop import llamaindex as llamaindex_adapter
from openmed.mcp.tool_registry import TOOL_REGISTRY


class FakeStructuredTool:
    def __init__(self, *, func, name, description):
        self.func = func
        self.name = name
        self.description = description

    @classmethod
    def from_function(cls, func=None, *, name=None, description=None, **kwargs):
        del kwargs
        return cls(func=func, name=name, description=description)


class FakeFunctionTool:
    def __init__(self, *, fn, name, description):
        self.fn = fn
        self.name = name
        self.description = description

    @classmethod
    def from_defaults(cls, fn=None, *, name=None, description=None, **kwargs):
        del kwargs
        return cls(fn=fn, name=name, description=description)


def test_function_tools_cover_every_registry_tool() -> None:
    definitions = to_function_tools()

    assert len(definitions) == len(TOOL_REGISTRY.latest_specs())
    assert [
        {
            "name": definition["function"]["name"],
            "parameters": definition["function"]["parameters"],
        }
        for definition in definitions
    ] == [
        {
            "name": spec.name,
            "parameters": spec.input_schema,
        }
        for spec in TOOL_REGISTRY.latest_specs()
    ]
    assert all(definition["type"] == "function" for definition in definitions)


def test_tool_use_tools_cover_every_registry_tool() -> None:
    definitions = to_tool_use_tools()

    assert len(definitions) == len(TOOL_REGISTRY.latest_specs())
    assert [
        {
            "name": definition["name"],
            "input_schema": definition["input_schema"],
        }
        for definition in definitions
    ] == [
        {
            "name": spec.name,
            "input_schema": spec.input_schema,
        }
        for spec in TOOL_REGISTRY.latest_specs()
    ]


def test_adapter_definition_input_schemas_match_registry() -> None:
    expected = [
        {"name": spec.name, "input_schema": spec.input_schema}
        for spec in TOOL_REGISTRY.latest_specs()
    ]

    assert [
        {
            "name": definition["function"]["name"],
            "input_schema": definition["function"]["parameters"],
        }
        for definition in to_function_tools()
    ] == expected
    assert [
        {"name": definition["name"], "input_schema": definition["input_schema"]}
        for definition in to_tool_use_tools()
    ] == expected
    assert [
        {"name": definition["name"], "input_schema": definition["input_schema"]}
        for definition in adapter_tool_definitions("langchain")
    ] == expected
    assert [
        {"name": definition["name"], "input_schema": definition["input_schema"]}
        for definition in adapter_tool_definitions("llamaindex")
    ] == expected


def test_langchain_tools_are_rendered_from_registry(monkeypatch) -> None:
    monkeypatch.setattr(
        langchain_adapter,
        "_import_module",
        lambda name: SimpleNamespace(StructuredTool=FakeStructuredTool),
    )

    tools = langchain_adapter.get_langchain_tools()

    assert [tool.name for tool in tools] == [
        spec.name for spec in TOOL_REGISTRY.latest_specs()
    ]
    assert [tool.description for tool in tools] == [
        spec.description for spec in TOOL_REGISTRY.latest_specs()
    ]
    assert [list(inspect.signature(tool.func).parameters) for tool in tools] == [
        [parameter.name for parameter in spec.parameters]
        for spec in TOOL_REGISTRY.latest_specs()
    ]


def test_llamaindex_tools_are_rendered_from_registry(monkeypatch) -> None:
    monkeypatch.setattr(
        llamaindex_adapter,
        "_import_module",
        lambda name: SimpleNamespace(FunctionTool=FakeFunctionTool),
    )

    tools = llamaindex_adapter.get_llamaindex_tools()

    assert [tool.name for tool in tools] == [
        spec.name for spec in TOOL_REGISTRY.latest_specs()
    ]
    assert [tool.description for tool in tools] == [
        spec.description for spec in TOOL_REGISTRY.latest_specs()
    ]
    assert [list(inspect.signature(tool.fn).parameters) for tool in tools] == [
        [parameter.name for parameter in spec.parameters]
        for spec in TOOL_REGISTRY.latest_specs()
    ]


def test_framework_object_factories_raise_clear_errors_without_extras(monkeypatch):
    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(langchain_adapter, "_import_module", missing_dependency)
    monkeypatch.setattr(llamaindex_adapter, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[langchain\]"):
        langchain_adapter.get_langchain_tools()
    with pytest.raises(ImportError, match=r"openmed\[llamaindex\]"):
        llamaindex_adapter.get_llamaindex_tools()


def test_llamaindex_adapter_loads_lazily() -> None:
    for name in list(sys.modules):
        if name == "llama_index" or name.startswith("llama_index."):
            sys.modules.pop(name, None)

    adapter = get_adapter("llamaindex")

    assert adapter is llamaindex_adapter
    assert adapter_spec("llamaindex").extra == "llamaindex"
    assert not any(
        name == "llama_index" or name.startswith("llama_index.") for name in sys.modules
    )
