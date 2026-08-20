"""Focused tests for the deterministic standalone redactor manifest."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "packaging" / "standalone_manifest.py"
ABOUT_PATH = ROOT / "openmed" / "__about__.py"
PYPROJECT_PATH = ROOT / "pyproject.toml"


def _load_module() -> ModuleType:
    """Load the repository manifest without importing optional dependencies."""

    spec = importlib.util.spec_from_file_location(
        "openmed_standalone_manifest_test_module",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


manifest_module = _load_module()


def test_default_manifest_is_offline_and_platform_neutral() -> None:
    manifest = manifest_module.STANDALONE_MANIFEST

    assert manifest.network_egress is False
    assert manifest.platforms == (manifest_module.PLATFORM_ANY,)
    assert manifest_module.validate_manifest(manifest) == ()
    assert all(
        component.platforms == (manifest_module.PLATFORM_ANY,)
        for component in manifest.components
    )
    assert all(
        dependency.network_egress is False
        for dependency in manifest.default_dependencies()
    )


def test_default_bundle_excludes_optional_and_restricted_entries() -> None:
    manifest = manifest_module.STANDALONE_MANIFEST
    default_names = set(manifest.default_dependency_names())
    optional_names = {item.name for item in manifest.optional_dependencies}
    restricted_names = {item.name for item in manifest.restricted_dependencies}

    assert default_names.isdisjoint(optional_names)
    assert default_names.isdisjoint(restricted_names)
    assert all(item.bundled is False for item in manifest.restricted_assets)
    assert all(item.name not in default_names for item in manifest.restricted_assets)
    assert restricted_names == {"extract-msg", "sdcMicro"}
    assert {
        item.name for item in manifest.optional_dependencies if item.network_egress
    } == {"huggingface-hub", "transformers"}
    assert set(manifest.default_requirements()) == {
        item.requirement for item in manifest.required_dependencies
    }


def test_rendering_is_byte_stable_and_detached() -> None:
    manifest = manifest_module.get_standalone_manifest()

    first = manifest_module.render_manifest(manifest)
    second = manifest_module.render_manifest(manifest)

    assert first == second
    payload = json.loads(first)
    assert payload["format"] == manifest_module.MANIFEST_FORMAT
    assert payload["dependencies"]["required"]
    assert payload["dependencies"]["optional"]
    assert payload["dependencies"]["restricted"]
    assert json.loads(manifest.to_json()) == payload

    payload["dependencies"]["required"].clear()
    assert manifest.default_dependencies()


def test_validation_does_not_echo_sensitive_metadata() -> None:
    synthetic_sensitive_value = "synthetic-sensitive-canary"
    invalid = replace(
        manifest_module.STANDALONE_MANIFEST,
        network_egress=True,
    )
    # The canary stands in for caller metadata that must never appear in a
    # diagnostic.  It is not included in the manifest's static report.
    invalid_component = replace(
        invalid.components[0],
        purpose=synthetic_sensitive_value,
    )
    invalid = replace(invalid, components=(invalid_component,))

    issues = manifest_module.validate_manifest(invalid)
    assert issues
    with pytest.raises(manifest_module.ManifestValidationError) as caught:
        manifest_module.assert_valid_manifest(invalid)

    assert synthetic_sensitive_value not in str(caught.value)
    assert all(synthetic_sensitive_value not in str(issue) for issue in issues)


def test_manifest_module_imports_no_network_or_package_discovery_modules() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and node.module not in {"__future__", "annotations"}
        ):
            imported_roots.add(node.module.split(".")[0])

    assert imported_roots == {"collections", "dataclasses", "json", "typing"}


def test_manifest_tracks_project_version_and_base_dependencies() -> None:
    """The descriptive boundary cannot drift from shipped project metadata."""

    version_match = re.search(
        r'__version__\s*=\s*"([^"]+)"',
        ABOUT_PATH.read_text(encoding="utf-8"),
    )
    assert version_match is not None
    project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))["project"]
    manifest = manifest_module.STANDALONE_MANIFEST

    assert manifest.version == version_match.group(1)
    assert manifest.python_requires == project["requires-python"]
    assert manifest.license == project["license"]["text"]
    assert set(manifest.default_requirements()) == set(project["dependencies"])


def test_requirement_drift_fails_without_echoing_the_requirement() -> None:
    """A changed default requirement is rejected with a value-free issue."""

    marker = "synthetic-private-package-value-882"
    changed = replace(
        manifest_module.STANDALONE_MANIFEST.required_dependencies[0],
        requirement=f"faker @ https://packages.invalid/{marker}",
    )
    invalid = replace(
        manifest_module.STANDALONE_MANIFEST,
        required_dependencies=(
            changed,
            *manifest_module.STANDALONE_MANIFEST.required_dependencies[1:],
        ),
    )

    with pytest.raises(manifest_module.ManifestValidationError) as caught:
        manifest_module.assert_valid_manifest(invalid)

    assert marker not in str(caught.value)
    assert "approved project boundary" in str(caught.value)


def test_hostile_text_and_collection_hooks_cannot_run() -> None:
    """Manifest construction rejects subclasses and mutable collections early."""

    marker = "synthetic-metadata-hook-value-441"

    class HostileText(str):
        def strip(self, *args, **kwargs):
            raise RuntimeError(marker)

    with pytest.raises(TypeError) as text_error:
        manifest_module.ComponentSpec(
            name=HostileText("local-redactor"),
            version="2.2.0",
            license="Apache-2.0",
            purpose="Local redaction.",
        )

    with pytest.raises(TypeError) as collection_error:
        replace(
            manifest_module.STANDALONE_MANIFEST,
            required_dependencies=list(  # type: ignore[arg-type]
                manifest_module.STANDALONE_MANIFEST.required_dependencies
            ),
        )

    assert marker not in str(text_error.value)
    assert marker not in str(collection_error.value)


def test_package_and_component_versions_must_remain_synchronized() -> None:
    """A stale package or component version fails structural validation."""

    invalid = replace(manifest_module.STANDALONE_MANIFEST, version="9.9.9")
    issues = manifest_module.validate_manifest(invalid)

    assert {issue.path for issue in issues} >= {
        "version",
        "components[0].version",
    }
