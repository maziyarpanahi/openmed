"""Tests for the canonical public Pydantic-backed tool registry."""

from __future__ import annotations

import inspect
import subprocess
import sys
from typing import Any

from jsonschema.validators import validator_for

import openmed
from openmed.interop import TOOLS, get_tool, list_tools
from openmed.interop import tools as tool_module
from openmed.interop.tools import (
    AnalyzeTextArgs,
    DeidentifyArgs,
    ExtractPIIArgs,
    UnloadModelArgs,
)

EXPECTED_INPUT_FIELDS = {
    "analyze_text": {
        "text",
        "model_name",
        "confidence_threshold",
        "group_entities",
        "aggregation_strategy",
        "sentence_detection",
        "sentence_language",
        "sentence_clean",
        "use_fast_tokenizer",
    },
    "extract_pii": {
        "text",
        "model_name",
        "confidence_threshold",
        "use_smart_merging",
        "lang",
        "normalize_accents",
    },
    "deidentify": {
        "text",
        "method",
        "model_name",
        "confidence_threshold",
        "keep_year",
        "shift_dates",
        "date_shift_days",
        "keep_mapping",
        "policy",
        "use_smart_merging",
        "use_safety_sweep",
        "lang",
        "normalize_accents",
    },
    "list_models": {"include_registry", "include_remote"},
    "list_pii_languages": set(),
    "loaded_models": set(),
    "unload_model": {"model_name", "all"},
}


class _FixtureLoader:
    config = object()

    def loaded_models(self) -> dict[str, dict[str, int]]:
        return {"fixture-model": {"models": 1, "tokenizers": 1, "pipelines": 1}}

    def unload_model(self, model_name: str) -> dict[str, Any]:
        return {
            "model_name": model_name,
            "models": 1,
            "tokenizers": 1,
            "pipelines": 1,
        }

    def unload_all_models(self) -> dict[str, int]:
        return {"models": 1, "tokenizers": 1, "pipelines": 1}


def test_registry_has_the_canonical_tool_contracts() -> None:
    assert set(TOOLS) == set(EXPECTED_INPUT_FIELDS)
    assert [tool.name for tool in list_tools()] == list(EXPECTED_INPUT_FIELDS)

    for tool in list_tools():
        assert tool.name
        assert tool.description
        assert callable(tool.callable)
        assert get_tool(tool.name) is tool

        for schema in (tool.input_json_schema(), tool.output_json_schema()):
            validator = validator_for(schema)
            validator.check_schema(schema)


def test_input_models_match_registered_callable_signatures() -> None:
    for name, expected_fields in EXPECTED_INPUT_FIELDS.items():
        tool = get_tool(name)
        assert set(tool.args_model.model_fields) == expected_fields
        assert set(inspect.signature(tool.callable).parameters) == expected_fields


def test_registry_callables_invoke_public_apis_with_synthetic_data(
    monkeypatch,
) -> None:
    tool_module._get_model_loader.cache_clear()
    monkeypatch.setattr(openmed, "ModelLoader", lambda: _FixtureLoader())
    monkeypatch.setattr(
        openmed,
        "analyze_text",
        lambda text, **kwargs: {
            "text": text,
            "entities": [],
            "model_name": kwargs["model_name"],
        },
    )
    monkeypatch.setattr(
        openmed,
        "extract_pii",
        lambda text, **kwargs: {
            "text": text,
            "entities": [],
            "model_name": kwargs["model_name"],
        },
    )
    monkeypatch.setattr(
        openmed,
        "deidentify",
        lambda text, **kwargs: {
            "deidentified_text": text,
            "pii_entities": [],
            "method": kwargs["method"],
        },
    )
    monkeypatch.setattr(
        openmed,
        "list_models",
        lambda **kwargs: [
            "fixture-model"
            if kwargs
            == {
                "include_registry": True,
                "include_remote": False,
                "config": _FixtureLoader.config,
            }
            else "unexpected"
        ],
    )
    monkeypatch.setattr(openmed, "get_pii_models_by_language", lambda code: [code])

    assert (
        get_tool("analyze_text").invoke(text="Synthetic asthma note.")["model_name"]
        == "disease_detection_superclinical"
    )
    assert (
        get_tool("extract_pii").invoke(text="Synthetic patient note.")["entities"] == []
    )
    assert (
        get_tool("deidentify").invoke(text="Synthetic contact note.")[
            "deidentified_text"
        ]
        == "Synthetic contact note."
    )
    assert get_tool("list_models").invoke(include_remote=False) == ["fixture-model"]
    assert get_tool("list_pii_languages").invoke()["count"] > 0
    assert "fixture-model" in get_tool("loaded_models").invoke()
    assert get_tool("unload_model").invoke(model_name="fixture-model")["pipelines"] == 1

    tool_module._get_model_loader.cache_clear()


def test_service_requests_reuse_registry_argument_models() -> None:
    from openmed.service.schemas import (
        AnalyzeRequest,
        ModelUnloadRequest,
        PIIDeidentifyRequest,
        PIIExtractRequest,
    )

    assert issubclass(AnalyzeRequest, AnalyzeTextArgs)
    assert issubclass(PIIExtractRequest, ExtractPIIArgs)
    assert issubclass(PIIDeidentifyRequest, DeidentifyArgs)
    assert issubclass(ModelUnloadRequest, UnloadModelArgs)


def test_registry_import_does_not_import_service_or_fastapi() -> None:
    code = """
import sys
from openmed.interop import list_tools
assert list_tools()
blocked = [
    name for name in sys.modules
    if name == 'fastapi'
    or name.startswith('fastapi.')
    or name == 'openmed.service'
    or name.startswith('openmed.service.')
]
assert blocked == [], blocked
"""
    subprocess.run([sys.executable, "-c", code], check=True)
