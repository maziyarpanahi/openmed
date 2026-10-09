from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import openmed.interop as interop


@pytest.fixture(scope="module")
def effect_guard():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "security"
        / "check_effect_paths.py"
    )
    spec = importlib.util.spec_from_file_location("openmed_effect_guard", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_effect_inventory_matches_all_sources_and_has_no_source_values(effect_guard):
    sources = effect_guard.read_sources()
    paths = effect_guard.scan_effect_paths(sources)
    labels = effect_guard.load_classifications()
    effect_guard.validate_inventory(paths, labels)
    document = effect_guard.inventory_document(paths, labels)
    assert set(document) == {"governed", "legacy_explicit_opt_in", "operational"}
    assert all(
        set(row) == {"module", "function"} for rows in document.values() for row in rows
    )
    assert all(
        row["module"].startswith("openmed.")
        for rows in document.values()
        for row in rows
    )
    assert not any(
        value in json.dumps(document)
        for value in ("http://", "https://", "INSERT ", "PRIVATE_PAYLOAD")
    )
    assert {
        "FHIRServerClient.put_resource",
        "FHIRServerClient.fetch_and_deidentify",
    } <= {row["function"] for row in document["legacy_explicit_opt_in"]}
    assert "OpenMRSAdapter.write_back" in {
        row["function"] for row in document["legacy_explicit_opt_in"]
    }


@pytest.mark.parametrize(
    "code",
    (
        "def mutate(client): client.post('PRIVATE_ENDPOINT', json={'secret': 'PRIVATE_PAYLOAD'})",
        "def mutate(client): client.request('PATCH', 'PRIVATE_ENDPOINT')",
        "def mutate(client, method): client.request(method, 'PRIVATE_ENDPOINT')",
        "def mutate(connection): connection.execute('INSERT INTO private VALUES (?)')",
        "def mutate(connection, sql): connection.execute(sql)",
        "def mutate(connection): connection.executemany('UPDATE private SET x=?', [])",
        "def mutate(connection): connection.executescript('CREATE TABLE private(x)')",
        "def mutate(connection): connection.execute('WITH x AS (SELECT 1) INSERT INTO private SELECT * FROM x')",
        "def mutate(connection): connection.execute('-- PRIVATE_COMMENT\\nDELETE FROM private')",
        "def mutate(client): getattr(client, 'delete')('PRIVATE_ENDPOINT')",
        "def mutate(): Request('PRIVATE_ENDPOINT', b'PRIVATE_PAYLOAD')",
        "def mutate(method): Request('PRIVATE_ENDPOINT', method=method)",
    ),
)
def test_new_unclassified_effect_module_fails_inventory_lint(effect_guard, code):
    sources = {"openmed.synthetic_effect": ast.parse(code)}
    paths = effect_guard.scan_effect_paths(sources)
    assert len(paths) == 1 and paths[0].function == "mutate"
    with pytest.raises(
        effect_guard.EffectPathError, match="effect_path_unclassified"
    ) as caught:
        effect_guard.validate_inventory(paths, {})
    assert "PRIVATE" not in str(caught.value)


def test_http_route_declarations_and_literal_reads_are_not_write_effects(effect_guard):
    sources = {
        "openmed.synthetic": ast.parse("""
@app.post('/PRIVATE_ENDPOINT')
async def handler(client, connection):
    client.request('GET', '/PRIVATE_ENDPOINT')
    connection.execute('SELECT value FROM private')
""")
    }
    assert effect_guard.scan_effect_paths(sources) == ()


@pytest.mark.parametrize(
    "code",
    (
        "def tool():\n from openmed.interop.fhir_server import FHIRServerClient\n return FHIRServerClient.put_resource",
        "def tool():\n from openmed.interop.openmrs import OpenMRSAdapter as Adapter\n return Adapter.write_back",
        "import openmed.interop.fhir_server as driver",
        "from ..interop import openmrs",
        "def tool():\n import importlib\n return importlib.import_module('openmed.interop.fhir_server')",
        "def tool():\n return __import__('openmed.interop.openmrs')",
        "def tool(client):\n return client.write_back([])",
        "def tool(client):\n return getattr(client, 'put_resource')",
        "def tool():\n return get_adapter('fhir_server')",
        "from importlib import import_module as load\ndef tool(): return load('openmed.interop.openmrs')",
        "def tool(): return import_module('.openmrs', 'openmed.interop')",
    ),
)
def test_synthetic_mcp_legacy_writer_reachability_fails(effect_guard, code):
    sources = {"openmed.mcp.synthetic_tool": ast.parse(code)}
    with pytest.raises(effect_guard.EffectPathError, match="legacy_writer_reachable"):
        effect_guard.assert_no_legacy_reachability(
            sources, ("openmed.mcp.synthetic_tool",)
        )


def test_transitive_dependency_and_unknown_handler_fail_closed(effect_guard):
    sources = {
        "openmed.mcp.synthetic_tool": ast.parse(
            "from openmed.synthetic_bridge import run"
        ),
        "openmed.synthetic_bridge": ast.parse(
            "from openmed.interop.openmrs import OpenMRSAdapter"
        ),
    }
    with pytest.raises(effect_guard.EffectPathError, match="legacy_writer_reachable"):
        effect_guard.assert_no_legacy_reachability(
            sources, ("openmed.mcp.synthetic_tool",)
        )
    with pytest.raises(effect_guard.EffectPathError, match="effect_surface_unverified"):
        effect_guard.assert_no_legacy_reachability(
            sources, ("external.synthetic_handler",)
        )


def test_imported_child_checks_its_package_initializer(effect_guard):
    sources = {
        "openmed.mcp.synthetic_tool": ast.parse(
            "import openmed.synthetic_package.child"
        ),
        "openmed.synthetic_package": ast.parse(
            "from openmed.interop.openmrs import OpenMRSAdapter"
        ),
        "openmed.synthetic_package.child": ast.parse("pass"),
    }
    with pytest.raises(effect_guard.EffectPathError, match="legacy_writer_reachable"):
        effect_guard.assert_no_legacy_reachability(
            sources, ("openmed.mcp.synthetic_tool",)
        )


def test_real_builtin_mcp_rest_cli_and_journey_surfaces_have_no_legacy_dependency(
    effect_guard, monkeypatch
):
    from openmed.cli.main import build_parser
    from openmed.core.models import ModelLoader
    from openmed.core.offline import network_blocked_if_offline
    from openmed.mcp import server, tool_registry
    from openmed.service.app import create_app
    from openmed.service.journey_workflows import (
        JOURNEY_WORKFLOW_DEFINITIONS,
        execute_journey_workflow,
    )

    monkeypatch.setattr(
        ModelLoader,
        "load_model",
        lambda *args, **kwargs: pytest.fail("unexpected model loading"),
    )
    registry = tool_registry.ToolRegistry(
        tool_registry.TOOL_SPECS, workflows=(tool_registry.CLINICAL_WORKFLOW_SPEC,)
    )
    monkeypatch.setattr(server, "TOOL_REGISTRY", registry)
    with network_blocked_if_offline(local_only=True):
        handlers = server.build_mcp_tool_handlers(None)
        assert set(handlers) == {spec.name for spec in registry.latest_specs()}
        app = create_app()
        parser = build_parser()
    modules = {handler.__module__ for handler in handlers.values()}
    modules.update(
        route.endpoint.__module__
        for route in app.routes
        if getattr(route, "endpoint", None) is not None
        and route.endpoint.__module__.startswith("openmed.")
    )

    def visit_parser(current):
        handler = current.get_default("handler")
        if handler is not None:
            modules.add(handler.__module__)
        for action in current._actions:
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                for child in choices.values():
                    if hasattr(child, "get_default"):
                        visit_parser(child)

    visit_parser(parser)
    assert JOURNEY_WORKFLOW_DEFINITIONS
    assert all(item.tool_name in handlers for item in JOURNEY_WORKFLOW_DEFINITIONS)
    modules.add(execute_journey_workflow.__module__)
    sources = effect_guard.read_sources()
    effect_guard.assert_no_legacy_reachability(sources, modules)
    effect_guard.assert_no_legacy_reachability(sources)


def test_stale_inventory_and_misclassified_legacy_paths_refuse(effect_guard):
    path = effect_guard.EffectPath(
        "openmed.interop.fhir_server", "FHIRServerClient.put_resource"
    )
    with pytest.raises(
        effect_guard.EffectPathError, match="legacy_classification_invalid"
    ):
        effect_guard.validate_inventory([path], {path: "governed"})
    with pytest.raises(effect_guard.EffectPathError, match="effect_inventory_stale"):
        effect_guard.validate_inventory([], {path: "legacy_explicit_opt_in"})


def test_lint_cli_and_manifest_refusals_do_not_echo_paths_or_values(
    effect_guard, monkeypatch, tmp_path, capsys
):
    path = tmp_path / "PRIVATE_PATH.json"
    path.write_text(
        json.dumps({"schema_version": 1, "paths": [{"module": "PRIVATE_PAYLOAD"}]})
    )
    with pytest.raises(
        effect_guard.EffectPathError, match="effect_manifest_invalid"
    ) as caught:
        effect_guard.load_classifications(path)
    assert caught.value.__context__ is None
    monkeypatch.setattr(
        effect_guard,
        "read_sources",
        lambda: (_ for _ in ()).throw(ValueError("PRIVATE_PAYLOAD")),
    )
    assert effect_guard.main(["--check"]) == 1
    assert capsys.readouterr().out == "effect_inventory_failed\n"


def test_lint_cli_refuses_unexpected_private_arguments_without_echo(
    effect_guard, capsys
):
    assert effect_guard.main(["--PRIVATE_PATH", "PRIVATE_PAYLOAD"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "effect_arguments_invalid\n"


OPTIONAL_ADAPTER_MODULE_PREFIXES = (
    "apache_beam",
    "duckdb",
    "indicnlp",
    "jieba",
    "langchain",
    "langchain_core",
    "langgraph",
    "medspacy",
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
    "snowflake",
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
        "airflow",
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
        "medspacy_context",
        "omop",
        "openmrs",
        "opensearch",
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
        "snowflake",
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
    assert adapter_spec("opensearch").extra == ""
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
    assert adapter_spec("snowflake").extra == "snowflake"
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
