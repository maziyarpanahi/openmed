"""Tests for bundled OpenMed JSON Schemas and drift detection."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema.validators import validator_for

from openmed.clinical.record_schemas import (
    clinical_record_schema_filename,
    load_clinical_record_snapshot,
)
from openmed.core.schemas import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_NAMES,
    build_packaged_schema_snapshot,
    build_schema_snapshot,
    compare_all_packaged_schema_drift,
    compare_all_schema_drift,
    compare_packaged_schema_drift,
    compare_schema_drift,
    current_schema_version,
    load_all_schemas,
    load_packaged_schema,
    load_schema,
    load_schema_bundle,
    load_schema_snapshot,
    packaged_schema_paths,
    schema_version_bumped,
    schema_version_of,
)

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = ROOT / "openmed"
COMMITTED_SNAPSHOT = ROOT / "scripts" / "release" / "schema_fingerprints.json"
DECISION_RESULT = "openmed/core/schemas/json/decision_result.schema.json"
CLINICAL_MEASURE = "openmed/core/schemas/json/clinical_measure.schema.json"


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


def _committed_packaged_snapshot() -> dict[str, Any]:
    return json.loads(COMMITTED_SNAPSHOT.read_text(encoding="utf-8"))


def _schema_files_on_disk() -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*.schema.json")
        if "__pycache__" not in path.parts
    }


def test_packaged_snapshot_has_exactly_one_entry_per_schema_file() -> None:
    paths = packaged_schema_paths()
    snapshot = _committed_packaged_snapshot()

    assert len(paths) == len(set(paths))
    assert set(paths) == _schema_files_on_disk()
    missing = sorted(set(paths) - set(snapshot))
    stale = sorted(set(snapshot) - set(paths))
    assert not missing, f"schemas without a fingerprint entry: {missing}"
    assert not stale, f"fingerprint entries without a schema: {stale}"
    assert "openmed/core/config.schema.json" in snapshot
    assert "openmed/compliance/schemas/control_evidence.schema.json" in snapshot


def test_committed_packaged_snapshot_matches_current_schemas() -> None:
    snapshot = _committed_packaged_snapshot()
    results = compare_all_packaged_schema_drift(snapshot=snapshot)

    assert all(not result.breaking_change for result in results.values())
    assert build_packaged_schema_snapshot() == snapshot


def test_packaged_snapshot_reuses_core_and_clinical_entries() -> None:
    snapshot = build_packaged_schema_snapshot()

    for name, state in load_schema_snapshot().items():
        assert snapshot[f"openmed/core/schemas/json/{name}.schema.json"] == state
    for name, state in load_clinical_record_snapshot().items():
        filename = clinical_record_schema_filename(name)
        assert snapshot[f"openmed/core/schemas/json/{filename}"] == state


def test_load_packaged_schema_rejects_unknown_paths() -> None:
    assert load_packaged_schema(DECISION_RESULT)["$id"].endswith("/1.2.0")
    with pytest.raises(KeyError):
        load_packaged_schema("openmed/core/schemas/json/synthetic.schema.json")
    with pytest.raises(KeyError):
        load_packaged_schema("decision_result")


def _without_first_required(schema: dict[str, Any]) -> tuple[dict[str, Any], str]:
    changed = copy.deepcopy(schema)
    removed = sorted(changed["required"])[0]
    changed["required"].remove(removed)
    del changed["properties"][removed]
    return changed, removed


def test_removed_required_property_without_version_bump_is_breaking() -> None:
    schema = load_packaged_schema(DECISION_RESULT)
    snapshot = build_packaged_schema_snapshot({DECISION_RESULT: schema})
    changed, removed = _without_first_required(schema)

    result = compare_packaged_schema_drift(DECISION_RESULT, changed, snapshot=snapshot)

    assert result.schema_version == result.snapshot_version == "1.2.0"
    assert result.version_bumped is False
    assert result.breaking_change is True
    assert result.removed_required == (removed,)
    assert result.removed_properties == (removed,)


def test_added_optional_property_is_not_breaking() -> None:
    schema = load_packaged_schema(DECISION_RESULT)
    snapshot = build_packaged_schema_snapshot({DECISION_RESULT: schema})
    changed = copy.deepcopy(schema)
    changed["properties"]["synthetic_optional_field"] = {"type": "string"}

    result = compare_packaged_schema_drift(DECISION_RESULT, changed, snapshot=snapshot)

    assert result.breaking_change is False
    assert result.version_bumped is False
    assert result.fingerprint != result.snapshot_fingerprint
    assert result.removed_required == result.removed_properties == ()


def test_removed_property_with_version_bump_is_not_breaking() -> None:
    schema = load_packaged_schema(DECISION_RESULT)
    snapshot = build_packaged_schema_snapshot({DECISION_RESULT: schema})
    changed, removed = _without_first_required(schema)
    changed["properties"]["schema_version"]["enum"].append("1.10.0")

    result = compare_packaged_schema_drift(DECISION_RESULT, changed, snapshot=snapshot)

    assert result.schema_version == "1.10.0"
    assert result.snapshot_version == "1.2.0"
    assert result.version_bumped is True
    assert result.breaking_change is False
    assert result.removed_required == (removed,)


def test_removed_variant_property_is_breaking() -> None:
    schema = load_packaged_schema(CLINICAL_MEASURE)
    snapshot = build_packaged_schema_snapshot({CLINICAL_MEASURE: schema})
    changed = copy.deepcopy(schema)
    definition = changed["$defs"]["definition"]
    removed = sorted(definition["required"])[0]
    definition["required"].remove(removed)
    del definition["properties"][removed]

    result = compare_packaged_schema_drift(CLINICAL_MEASURE, changed, snapshot=snapshot)

    assert result.breaking_change is True
    assert result.removed_required == (f"definition.{removed}",)
    assert result.removed_properties == (f"definition.{removed}",)


def test_schema_without_snapshot_entry_is_not_breaking_but_unrecorded() -> None:
    result = compare_packaged_schema_drift(DECISION_RESULT, snapshot={})

    assert result.breaking_change is False
    assert result.snapshot_fingerprint is None
    assert result.snapshot_version is None


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ({"schema_version": 1}, 1),
        ({"schema_version": "1.0.0"}, "1.0.0"),
        ({"properties": {"schema_version": {"const": "1.1.0"}}}, "1.1.0"),
        (
            {"properties": {"schema_version": {"enum": ["1.9.0", "1.10.0", "1.2.0"]}}},
            "1.10.0",
        ),
        ({"$id": "https://openmed.dev/schemas/decision-request/1.2.0"}, "1.2.0"),
        (
            {"properties": {"schema_version": {"const": "openmed.synthetic.v2"}}},
            "openmed.synthetic.v2",
        ),
        ({"$id": "https://openmed.dev/schemas/synthetic.schema.json"}, None),
        ({"schema_version": True}, None),
        ({}, None),
    ],
)
def test_schema_version_of_reads_every_declaration_style(
    schema: dict[str, Any], expected: Any
) -> None:
    assert schema_version_of(schema) == expected


@pytest.mark.parametrize(
    ("current", "saved", "expected"),
    [
        (2, 1, True),
        (1, 1, False),
        (1, 2, False),
        ("1.10.0", "1.9.0", True),
        ("1.2.0", "1.2.0", False),
        ("1.1.0", "1.2.0", False),
        ("2.0.0", 1, True),
        ("openmed.synthetic.v2", "openmed.synthetic.v1", True),
        ("openmed.synthetic.v1", "openmed.synthetic.v1", False),
        ("openmed.other.v1", "openmed.synthetic.v1", True),
        (None, 1, False),
        ("1.0.0", None, False),
    ],
)
def test_schema_version_bumped_orders_integers_semver_and_labels(
    current: Any, saved: Any, expected: bool
) -> None:
    assert schema_version_bumped(current, saved) is expected


def test_core_drift_accepts_semver_versions_without_type_errors() -> None:
    schemas = load_all_schemas()
    snapshot = build_schema_snapshot(schemas)
    changed_span = copy.deepcopy(schemas["span"])
    changed_span["schema_version"] = "2.0.0"
    changed_span["required"].remove("detector")
    del changed_span["properties"]["detector"]

    result = compare_schema_drift("span", changed_span, snapshot=snapshot)

    assert result.schema_version == "2.0.0"
    assert result.snapshot_version == CURRENT_SCHEMA_VERSION
    assert result.version_bumped is True
    assert result.breaking_change is False


def test_core_drift_results_keep_integer_versions() -> None:
    for result in compare_all_schema_drift().values():
        assert result.schema_version == CURRENT_SCHEMA_VERSION
        assert result.snapshot_version == CURRENT_SCHEMA_VERSION
        assert type(result.schema_version) is int
