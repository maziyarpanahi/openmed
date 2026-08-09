"""Focused offline tests for the privacy-safe FHIR R5 fidelity diff."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from openmed.interop.fhir_r5_fidelity import (
    FHIRR5FidelityError,
    diff_fhir_r5_bundles,
)


def _patient() -> dict:
    return {
        "resourceType": "Patient",
        "id": "synthetic-patient-1",
        "name": [{"text": "Synthetic Person"}],
        "identifier": [{"system": "https://synthetic.example/id", "value": "S-1"}],
    }


def _observation() -> dict:
    return {
        "resourceType": "Observation",
        "id": "synthetic-observation-1",
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "SYN-1234",
                    "display": "Synthetic measurement",
                }
            ]
        },
        "subject": {"reference": "Patient/synthetic-patient-1"},
        "valueQuantity": {"value": 7.2, "unit": "synthetic-unit"},
    }


def _bundle() -> dict:
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "fullUrl": "urn:uuid:synthetic-patient-1",
                "resource": _patient(),
            },
            {
                "fullUrl": "urn:uuid:synthetic-observation-1",
                "resource": _observation(),
            },
        ],
    }


def test_round_trip_ignores_json_member_order_and_bundle_entry_order():
    before = _bundle()
    after = json.loads(json.dumps(before, sort_keys=True))
    after["entry"].reverse()

    result = diff_fhir_r5_bundles(before, after)

    assert result.equivalent
    assert result.changes == ()
    assert result.before_digest.startswith("sha256:")
    assert result.after_digest.startswith("sha256:")
    assert result.to_json() == result.to_json()


def test_diff_reports_paths_types_and_safe_resource_identifiers():
    before = _bundle()
    after = copy.deepcopy(before)
    observation = after["entry"][1]["resource"]
    observation["code"]["coding"][0]["code"] = "SYN-5678"
    observation["valueQuantity"]["value"] = "7.2"
    observation["subject"]["reference"] = "Patient/synthetic-patient-2"

    result = diff_fhir_r5_bundles(before, after)
    report_text = result.to_json() + result.to_markdown()

    assert not result.equivalent
    assert any(
        path.endswith("resource.code.coding[0].code") for path in result.changed_paths
    )
    assert any(
        path.endswith("resource.valueQuantity.value") for path in result.changed_paths
    )
    value_type_change = next(
        change
        for change in result.type_changes
        if change.path.endswith("resource.valueQuantity.value")
    )
    assert value_type_change.change_type == "type_changed"
    assert value_type_change.before_type == "number"
    assert value_type_change.after_type == "string"
    assert value_type_change.resource_type == "Observation"
    assert value_type_change.resource_id_hash.startswith("sha256:")
    assert "synthetic-patient-1" not in report_text
    assert "synthetic-patient-2" not in report_text
    assert "SYN-5678" not in report_text


def test_declared_serialization_differences_are_ignored_by_path():
    before = _bundle()
    after = copy.deepcopy(before)
    before["entry"][0]["resource"]["meta"] = {
        "lastUpdated": "2099-01-01T00:00:00Z",
        "tag": [
            {"system": "https://synthetic.example/tag", "code": "a"},
            {"system": "https://synthetic.example/tag", "code": "b"},
        ],
    }
    after["entry"][0]["resource"]["meta"] = copy.deepcopy(
        before["entry"][0]["resource"]["meta"]
    )
    after["entry"][0]["resource"]["meta"]["lastUpdated"] = "2099-02-01T00:00:00Z"
    after["entry"][0]["resource"]["meta"]["tag"].reverse()

    strict = diff_fhir_r5_bundles(before, after)
    allowed = diff_fhir_r5_bundles(
        before,
        after,
        allowed_paths=["entry[*].resource.meta.lastUpdated"],
        unordered_paths=["entry[*].resource.meta.tag"],
    )

    assert not strict.equivalent
    assert allowed.equivalent
    assert allowed.ignored_paths == ("entry[*].resource.meta.lastUpdated",)
    assert allowed.unordered_paths == ("entry[*].resource.meta.tag",)


def test_resource_type_changes_are_reported_without_values():
    before = _bundle()
    after = copy.deepcopy(before)
    after["entry"][0]["resource"]["resourceType"] = "Observation"

    result = diff_fhir_r5_bundles(before, after)
    type_change = next(
        change
        for change in result.changes
        if change.path.endswith("resource.resourceType")
    )

    assert type_change.change_type == "resource_type_changed"
    assert type_change.before_resource_type == "Patient"
    assert type_change.after_resource_type == "Observation"
    assert type_change.before_type == "string"
    assert type_change.after_type == "string"


def test_json_input_and_invalid_errors_are_local_and_phi_safe(tmp_path: Path):
    bundle_path = tmp_path / "synthetic-bundle.json"
    bundle_path.write_text(json.dumps(_bundle()), encoding="utf-8")

    assert diff_fhir_r5_bundles(bundle_path, json.dumps(_bundle())).equivalent

    secret = "Synthetic private narrative that must not appear"
    with pytest.raises(FHIRR5FidelityError) as exc_info:
        diff_fhir_r5_bundles(
            f'{{"resourceType": "Bundle", "note": "{secret}',
            _bundle(),
        )

    assert secret not in str(exc_info.value)
