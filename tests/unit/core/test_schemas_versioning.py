"""Tests for bundled OpenMed JSON Schemas and drift detection."""

from __future__ import annotations

import copy

from jsonschema.validators import validator_for

from openmed.core.schemas import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_NAMES,
    build_schema_snapshot,
    compare_all_schema_drift,
    compare_schema_drift,
    current_schema_version,
    load_all_schemas,
    load_schema,
    load_schema_bundle,
    load_schema_snapshot,
)


def test_each_shipped_json_schema_is_valid_and_versioned() -> None:
    for name in SCHEMA_NAMES:
        schema = load_schema(name)
        validator = validator_for(schema)
        validator.check_schema(schema)

        assert schema["schema_version"] == CURRENT_SCHEMA_VERSION
        assert schema["properties"]["schema_version"]["const"] == CURRENT_SCHEMA_VERSION


def test_schema_loader_returns_bundle_and_current_version() -> None:
    bundle = load_schema_bundle()

    assert current_schema_version() == CURRENT_SCHEMA_VERSION
    assert bundle["schema_version"] == CURRENT_SCHEMA_VERSION
    assert set(bundle["schemas"]) == set(SCHEMA_NAMES)


def test_committed_snapshot_matches_current_schemas() -> None:
    results = compare_all_schema_drift()

    assert set(results) == set(SCHEMA_NAMES)
    assert all(not result.breaking_change for result in results.values())
    assert build_schema_snapshot(load_all_schemas()) == load_schema_snapshot()


def test_drift_helper_flags_removed_field_without_version_bump() -> None:
    schemas = load_all_schemas()
    snapshot = build_schema_snapshot(schemas)
    changed_span = copy.deepcopy(schemas["span"])
    changed_span["required"].remove("detector")
    del changed_span["properties"]["detector"]

    result = compare_schema_drift("span", changed_span, snapshot=snapshot)

    assert result.breaking_change is True
    assert result.version_bumped is False
    assert result.removed_required == ("detector",)
    assert result.removed_properties == ("detector",)


def test_drift_helper_accepts_breaking_change_with_version_bump() -> None:
    schemas = load_all_schemas()
    snapshot = build_schema_snapshot(schemas)
    changed_span = copy.deepcopy(schemas["span"])
    changed_span["schema_version"] = CURRENT_SCHEMA_VERSION + 1
    changed_span["properties"]["schema_version"]["const"] = CURRENT_SCHEMA_VERSION + 1
    changed_span["required"].remove("detector")
    del changed_span["properties"]["detector"]

    result = compare_schema_drift("span", changed_span, snapshot=snapshot)

    assert result.version_bumped is True
    assert result.breaking_change is False
    assert result.removed_required == ("detector",)
