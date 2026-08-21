from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

import openmed.interop as interop

OPTIONAL_ADAPTER_MODULE_PREFIXES = (
    "apache_beam",
    "duckdb",
    "indicnlp",
    "jieba",
    "langchain",
    "langchain_core",
    "langgraph",
    "pandas",
    "presidio",
    "philter_ucsf",
    "polars",
    "prefect",
    "pyDeid",
    "pydeid",
    "pyspark",
    "ray",
    "gliner",
    "haystack",
    "llama_index",
    "opencc",
    "pypinyin",
    "quickumls",
    "scispacy",
    "scrubadub",
    "spacy",
)


@pytest.fixture(autouse=True)
def reset_runtime_plugin_adapters():
    interop._reset_plugin_adapters_for_tests()
    yield
    interop._reset_plugin_adapters_for_tests()


def _clear_optional_adapter_modules() -> None:
    for name in list(sys.modules):
        if _is_optional_adapter_module(name):
            sys.modules.pop(name, None)


def _is_optional_adapter_module(name: str) -> bool:
    return any(
        name == prefix or name.startswith(f"{prefix}.")
        for prefix in OPTIONAL_ADAPTER_MODULE_PREFIXES
    )


def test_import_openmed_does_not_import_optional_adapter_dependencies():
    _clear_optional_adapter_modules()
    for name in list(sys.modules):
        if name == "openmed.plugins" or name.startswith("openmed.plugins."):
            sys.modules.pop(name, None)

    import openmed  # noqa: F401

    assert not any(_is_optional_adapter_module(name) for name in sys.modules)
    assert "openmed.plugins" not in sys.modules


def test_fresh_core_import_does_not_import_graph_or_search_frameworks():
    code = """
import sys
import openmed
import openmed.interop
blocked = [
    name for name in sys.modules
    if name == 'langgraph'
    or name.startswith('langgraph.')
    or name == 'haystack'
    or name.startswith('haystack.')
]
assert blocked == [], blocked
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_interop_registry_does_not_import_optional_adapter_dependencies():
    _clear_optional_adapter_modules()
    sys.modules.pop("openmed.interop.icd11_api", None)

    from openmed.interop import adapter_spec, available_adapters

    assert available_adapters() == (
        "beam",
        "cda",
        "cdm_etl",
        "duckdb",
        "fhir_server",
        "function_tools",
        "gliner_biomed",
        "graph_orchestration",
        "haystack",
        "hl7v2",
        "icd11_api",
        "indic",
        "langchain",
        "llamaindex",
        "omop",
        "openmrs",
        "pandas",
        "philter",
        "polars",
        "prefect",
        "presidio",
        "pydeid",
        "quickumls",
        "ray",
        "scispacy_linker",
        "scrubadub",
        "search_pipeline",
        "spacy",
        "spark",
        "zh",
    )
    assert adapter_spec("beam").extra == "beam"
    assert adapter_spec("cda").extra == "core"
    assert adapter_spec("cdm_etl").extra == ""
    assert adapter_spec("duckdb").extra == "duckdb"
    assert adapter_spec("hl7v2").extra == ""
    assert adapter_spec("icd11_api").extra == ""
    assert adapter_spec("fhir_server").extra == "fhir"
    assert adapter_spec("indic").extra == "indic"
    assert adapter_spec("function_tools").extra == ""
    assert adapter_spec("graph_orchestration").extra == "langgraph"
    assert adapter_spec("haystack").extra == "haystack"
    assert adapter_spec("langchain").extra == "langchain"
    assert adapter_spec("llamaindex").extra == "llamaindex"
    assert adapter_spec("omop").extra == ""
    assert adapter_spec("openmrs").extra == "openmrs"
    assert adapter_spec("pandas").extra == "pandas"
    assert adapter_spec("presidio").extra == "presidio"
    assert adapter_spec("philter").extra == "philter"
    assert adapter_spec("polars").extra == "polars"
    assert adapter_spec("prefect").extra == "prefect"
    assert adapter_spec("pydeid").extra == "pydeid"
    assert adapter_spec("quickumls").extra == "quickumls"
    assert adapter_spec("ray").extra == "ray"
    assert adapter_spec("scispacy_linker").extra == "scispacy"
    assert adapter_spec("scrubadub").extra == "scrubadub"
    assert adapter_spec("search_pipeline").extra == "haystack"
    assert adapter_spec("gliner_biomed").extra == "gliner"
    assert adapter_spec("spacy").extra == "spacy"
    assert adapter_spec("spark").extra == "spark"
    assert adapter_spec("zh").extra == "zh"
    assert "openmed.interop.icd11_api" not in sys.modules
    assert not any(_is_optional_adapter_module(name) for name in sys.modules)


def test_sdk_adapters_and_exporters_share_registry_with_explicit_policy_opt_in(
    monkeypatch,
):
    class SyntheticAdapter:
        def to_openmed_spans(self, payload, **kwargs):
            del payload, kwargs
            return ()

        def from_openmed_spans(self, spans, **kwargs):
            del spans, kwargs
            return {"schema": "synthetic-adapter.v1"}

    class SyntheticExporter:
        def export(self, spans, **kwargs):
            del spans, kwargs
            return {"schema": "synthetic-exporter.v1"}

    def registration(component_id, kind, component, *, opted_in=False):
        metadata = SimpleNamespace(
            plugin_id="synthetic-interop-plugin",
            component_id=component_id,
            qualified_id=f"synthetic-interop-plugin:{component_id}",
            kind=kind,
            name=f"Synthetic {kind}",
            description=f"Offline synthetic {kind}",
        )
        return SimpleNamespace(
            metadata=metadata,
            component=component,
            loaded_by_policy_opt_in=opted_in,
        )

    adapter = registration(
        "record-adapter",
        "interop_adapter",
        SyntheticAdapter(),
    )
    restricted_exporter = registration(
        "restricted-exporter",
        "exporter",
        SyntheticExporter(),
        opted_in=True,
    )

    def fake_iter_sdk_plugins(**policy):
        registrations = [adapter]
        if "synthetic-interop-plugin:restricted-exporter" in policy["opt_in_plugins"]:
            registrations.append(restricted_exporter)
        return tuple(registrations)

    monkeypatch.setattr(interop, "_iter_sdk_plugins", fake_iter_sdk_plugins)

    default_specs = interop.discover_plugin_adapters()
    assert [spec.qualified_id for spec in default_specs] == [
        "synthetic-interop-plugin:record-adapter"
    ]
    assert "synthetic-interop-plugin:restricted-exporter" not in (
        interop.available_adapters()
    )

    interop.discover_plugin_adapters(
        opt_in_plugins=("synthetic-interop-plugin:restricted-exporter",)
    )

    exporter_name = "synthetic-interop-plugin:restricted-exporter"
    assert exporter_name in interop.available_adapters()
    assert interop.adapter_spec(exporter_name).kind == "exporter"
    assert interop.adapter_spec(exporter_name).loaded_by_policy_opt_in is True
    assert interop.get_adapter(exporter_name) is restricted_exporter.component


def test_presidio_adapter_missing_extra_raises_clear_import_error(monkeypatch):
    from openmed.interop import presidio

    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(presidio, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[presidio\]"):
        presidio.from_canonical([])
