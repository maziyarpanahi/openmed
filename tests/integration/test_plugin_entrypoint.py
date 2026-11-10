"""Offline integration coverage for an installed OpenMed plugin distribution."""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import shutil
import sys
from pathlib import Path

import pytest

from openmed.core import detector_plugins
from openmed.core.pipeline import Pipeline
from openmed.plugins import registry as plugin_registry
from openmed.processing.outputs import PredictionResult

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PACKAGE = (
    REPO_ROOT / "examples" / "openmed-plugin-example" / "src" / "openmed_example_plugin"
)


def _install_example_distribution(site_packages: Path) -> None:
    """Create the example package's installed wheel layout without a network."""

    shutil.copytree(EXAMPLE_PACKAGE, site_packages / "openmed_example_plugin")
    dist_info = site_packages / "openmed_example_plugin-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: openmed-example-plugin\nVersion: 0.1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[openmed.plugins]\nexample = openmed_example_plugin:plugin_components\n",
        encoding="utf-8",
    )


def _empty_prediction(text: str, **kwargs: object) -> PredictionResult:
    """Return a deterministic empty model result for pipeline arbitration."""

    return PredictionResult(
        text=text,
        entities=[],
        model_name=str(kwargs.get("model_name") or "synthetic-stub"),
        timestamp="2026-08-04T00:00:00+00:00",
    )


def test_installed_entry_point_recognizer_participates_in_arbitration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discover the installed example and route its span through arbitration."""

    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    _install_example_distribution(site_packages)
    monkeypatch.syspath_prepend(str(site_packages))
    sys.modules.pop("openmed_example_plugin", None)
    importlib.invalidate_caches()

    installed_entry_points = tuple(
        entry_point
        for distribution in importlib_metadata.distributions(path=[str(site_packages)])
        for entry_point in distribution.entry_points
    )

    def isolated_entry_points(*, group: str | None = None):
        return tuple(
            entry_point
            for entry_point in installed_entry_points
            if group is None or entry_point.group == group
        )

    monkeypatch.setattr(importlib_metadata, "entry_points", isolated_entry_points)
    plugin_registry._reset_plugin_registry_for_tests()
    detector_plugins._reset_detector_registry_for_tests()

    try:
        text = "Synthetic fixture OPENMED_SYNTHETIC_PERSON"
        result = Pipeline(
            model_detector=_empty_prediction,
            use_safety_sweep=False,
        ).run(text, method="mask")

        discovery = plugin_registry.discover_plugins()
        assert [
            registration.metadata.component_id
            for registration in discovery.registrations
        ] == ["toy-exporter", "toy-recognizer"]
        assert discovery.quarantined == ()

        arbitration_span = result.stage("span_arbitration").spans[0]
        assert arbitration_span.canonical_label == "PERSON"
        assert (
            arbitration_span.detector == "plugin:openmed-example-plugin:toy-recognizer"
        )
        assert result.redacted_text == "Synthetic fixture [person]"
    finally:
        detector_plugins._reset_detector_registry_for_tests()
        plugin_registry._reset_plugin_registry_for_tests()
        sys.modules.pop("openmed_example_plugin", None)
