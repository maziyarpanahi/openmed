"""Self-certification tests copied with the example plugin package."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from openmed_example_plugin import plugin_components

from openmed.plugins.conformance import (
    REASON_INVALID_METADATA,
    PluginConformanceError,
    assert_plugin_conforms,
    check_plugin_conformance,
)


def _malformed_components():
    fixture = Path(__file__).parent / "fixtures" / "malformed_plugin.py"
    spec = importlib.util.spec_from_file_location("malformed_plugin", fixture)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plugin_components


def test_example_plugin_conforms_offline() -> None:
    report = assert_plugin_conforms(plugin_components)

    assert report.passed
    assert report.components_checked == 2


def test_malformed_fixture_has_specific_failure() -> None:
    report = check_plugin_conformance(_malformed_components())

    assert not report.passed
    assert report.failures[0].reason == REASON_INVALID_METADATA
    assert report.failures[0].message == "network_egress must be a boolean"
    with pytest.raises(
        PluginConformanceError,
        match="invalid_metadata: network_egress must be a boolean",
    ):
        assert_plugin_conforms(_malformed_components())
