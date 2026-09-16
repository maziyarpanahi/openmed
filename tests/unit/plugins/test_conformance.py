"""Offline tests for the public plugin example and conformance kit."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from openmed.plugins.conformance import (
    REASON_INVALID_METADATA,
    PluginConformanceError,
    assert_plugin_conforms,
    check_plugin_conformance,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "openmed-plugin-example"
EXAMPLE_SRC = EXAMPLE_ROOT / "src"
MALFORMED_FIXTURE = EXAMPLE_ROOT / "tests" / "fixtures" / "malformed_plugin.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example_plugin(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_SRC))
    sys.modules.pop("openmed_example_plugin", None)
    return __import__("openmed_example_plugin")


def test_example_plugin_passes_conformance_without_network(
    example_plugin,
    monkeypatch,
) -> None:
    def deny_network(*args, **kwargs):
        del args, kwargs
        raise AssertionError("conformance attempted network access")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket, "socket", deny_network)

    report = assert_plugin_conforms(example_plugin.plugin_components)

    assert report.passed
    assert report.components_checked == 2
    assert report.failures == ()


def test_example_recognizer_and_exporter_preserve_privacy_safe_spans(
    example_plugin,
) -> None:
    marker = example_plugin.SYNTHETIC_PERSON_MARKER
    text = f"fixture {marker} complete"
    recognizer, exporter = example_plugin.plugin_components()

    spans = recognizer.recognize(text)
    assert len(spans) == 1
    assert (spans[0].start, spans[0].end) == (
        text.index(marker),
        text.index(marker) + len(marker),
    )
    assert spans[0].canonical_label == "PERSON"
    assert spans[0].text_hash.startswith("hmac-sha256:")

    artifact = exporter.export(spans)
    serialized = json.dumps(artifact, sort_keys=True)
    assert marker not in serialized
    assert artifact["spans"][0]["start"] == text.index(marker)


def test_example_distribution_declares_entry_point_and_permissive_license() -> None:
    with (EXAMPLE_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    project = config["project"]
    assert project["license"] == "Apache-2.0"
    assert project["dependencies"] == ["openmed>=2.0.0"]
    assert project["entry-points"]["openmed.plugins"] == {
        "example": "openmed_example_plugin:plugin_components"
    }
    assert (EXAMPLE_ROOT / "LICENSE").is_file()


def test_malformed_fixture_fails_with_specific_reason() -> None:
    malformed = _load_module("openmed_malformed_plugin", MALFORMED_FIXTURE)

    report = check_plugin_conformance(malformed.plugin_components)

    assert not report.passed
    assert report.components_checked == 1
    assert len(report.failures) == 1
    assert report.failures[0].reason == REASON_INVALID_METADATA
    assert report.failures[0].message == "network_egress must be a boolean"
    with pytest.raises(
        PluginConformanceError,
        match="invalid_metadata: network_egress must be a boolean",
    ):
        assert_plugin_conforms(malformed.plugin_components)


def test_conformance_cli_reports_safe_deterministic_results(
    example_plugin,
    capsys,
) -> None:
    assert main(["openmed_example_plugin:plugin_components"]) == 0
    assert capsys.readouterr().out == "PASS: 2 component(s) conform\n"

    malformed = _load_module("openmed_malformed_cli", MALFORMED_FIXTURE)
    sys.modules["openmed_malformed_cli"] = malformed
    try:
        assert main(["openmed_malformed_cli:plugin_components"]) == 1
    finally:
        sys.modules.pop("openmed_malformed_cli", None)
    output = capsys.readouterr().out
    assert output == (
        "FAIL: 1 conformance error(s)\n"
        "- invalid_metadata: network_egress must be a boolean\n"
    )
