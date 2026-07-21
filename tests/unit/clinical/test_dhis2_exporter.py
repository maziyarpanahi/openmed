"""Offline round-trip and privacy tests for the DHIS2 exporter."""

from __future__ import annotations

import copy
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.clinical.exporters import (
    DHIS2ExportConfig,
    DHIS2Exporter,
    DHIS2ExportError,
    export_dhis2,
)
from openmed.interop import assert_redacted
from tests.unit.clinical.fixtures.dhis2.schema_checker import (
    validate_aggregate_payload,
    validate_tracker_payload,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "dhis2"
ORIGINALS = (
    "Amina Example",
    "+254 712 345 678",
    "ID-851-0001",
)
REPLACEMENTS = {
    "Amina Example": "[NAME]",
    "+254 712 345 678": "[PHONE]",
    "ID-851-0001": "[ID]",
}
REDACTION_MAPPING = {
    replacement: source for source, replacement in REPLACEMENTS.items()
}


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _offline_redactor(value: str) -> str:
    for source, replacement in REPLACEMENTS.items():
        value = value.replace(source, replacement)
    return value


def _org_units(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "orgUnit":
                found.append(item)
            else:
                found.extend(_org_units(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_org_units(item))
    return found


def test_synthetic_payloads_round_trip_without_phi_or_fine_geography():
    aggregate = _load_fixture("aggregate_payload.json")
    tracker = _load_fixture("tracker_payload.json")
    original_aggregate = copy.deepcopy(aggregate)
    original_tracker = copy.deepcopy(tracker)
    config = DHIS2ExportConfig(date_shift_days=14)

    result = export_dhis2(
        aggregate,
        tracker,
        FIXTURE_DIR / "organisation_units.json",
        config=config,
        text_redactor=_offline_redactor,
    )
    repeated = export_dhis2(
        aggregate,
        tracker,
        FIXTURE_DIR / "organisation_units.json",
        config=config,
        text_redactor=_offline_redactor,
    )

    validate_aggregate_payload(result.aggregate_payload)
    validate_tracker_payload(result.tracker_payload)
    assert aggregate == original_aggregate
    assert tracker == original_tracker
    assert result.to_json() == repeated.to_json()
    assert result.aggregate_json() == repeated.aggregate_json()
    assert result.tracker_json() == repeated.tracker_json()

    assert set(_org_units(result.combined_payload)) == {"ouDistrict1"}
    aggregate_values = result.aggregate_payload["dataValueSets"][0]["dataValues"]
    assert [item["dataElement"] for item in aggregate_values] == ["deVisits001"]
    assert result.tracker_payload["events"][0]["occurredAt"] == (
        "2026-07-31T10:30:00.000Z"
    )
    event = result.tracker_payload["events"][0]
    assert not {"geometry", "latitude", "longitude"}.intersection(event)
    assert event["notes"][0]["value"] == "[NAME] called from [PHONE]"
    assert event["dataValues"][0]["value"] == "[NAME] confirmed [ID]"

    serialized_output = result.aggregate_json() + result.tracker_json()
    assert_redacted(serialized_output, REDACTION_MAPPING)
    assert_redacted(result.manifest_json(), REDACTION_MAPPING)
    for original in ORIGINALS:
        assert original not in serialized_output
        assert original not in result.manifest_json()

    counts = result.manifest["counts"]
    assert counts["data_value_sets"] == 1
    assert counts["aggregate_values_input"] == 2
    assert counts["aggregate_values_output"] == 1
    assert counts["suppressed_aggregate_values"] == 1
    assert counts["tracked_entities"] == 1
    assert counts["events"] == 1
    assert counts["org_units_examined"] == 3
    assert counts["org_units_generalized"] == 3
    assert counts["precise_locations_removed"] == 3
    assert counts["dates_transformed"] == 1
    assert result.manifest["generalization_level"] == 3
    assert result.manifest["small_cell_threshold"] == 5
    assert result.manifest["transformed_paths"] == sorted(
        result.manifest["transformed_paths"]
    )


def test_configurable_geo_level_replaces_facilities_with_province():
    result = export_dhis2(
        _load_fixture("aggregate_payload.json"),
        _load_fixture("tracker_payload.json"),
        _load_fixture("organisation_units.json"),
        generalization_level=2,
        small_cell_threshold=0,
        date_mode="none",
        text_redactor=_offline_redactor,
    )

    assert set(_org_units(result.combined_payload)) == {"ouProvince1"}
    assert len(result.aggregate_payload["dataValueSets"][0]["dataValues"]) == 2
    assert result.manifest["counts"]["org_units_generalized"] == 3
    assert result.manifest["counts"]["suppressed_aggregate_values"] == 0


def test_none_convenience_threshold_disables_small_cell_suppression():
    result = export_dhis2(
        _load_fixture("aggregate_payload.json"),
        None,
        _load_fixture("organisation_units.json"),
        small_cell_threshold=None,
        date_mode="none",
        text_redactor=_offline_redactor,
    )

    assert len(result.aggregate_payload["dataValueSets"][0]["dataValues"]) == 2
    assert result.manifest["small_cell_threshold"] is None


def test_period_and_event_dates_can_be_coarsened_to_month():
    aggregate = _load_fixture("aggregate_payload.json")
    aggregate["dataValueSets"][0]["period"] = "20260717"

    result = export_dhis2(
        aggregate,
        _load_fixture("tracker_payload.json"),
        _load_fixture("organisation_units.json"),
        config=DHIS2ExportConfig(
            date_mode="coarsen",
            period_granularity="month",
        ),
        text_redactor=_offline_redactor,
    )

    assert result.aggregate_payload["dataValueSets"][0]["period"] == "202607"
    assert result.tracker_payload["events"][0]["occurredAt"] == ("2026-07-01T00:00:00Z")
    assert result.manifest["counts"]["periods_coarsened"] == 1
    assert result.manifest["counts"]["dates_transformed"] == 1


def test_payload_collections_are_sorted_for_canonical_output():
    aggregate = _load_fixture("aggregate_payload.json")
    aggregate["dataValueSets"][0]["dataValues"].reverse()
    tracker = _load_fixture("tracker_payload.json")
    tracker["trackedEntities"][0]["attributes"].reverse()

    result = export_dhis2(
        aggregate,
        tracker,
        _load_fixture("organisation_units.json"),
        config=DHIS2ExportConfig(
            small_cell_threshold=0,
            date_mode="none",
        ),
        text_redactor=_offline_redactor,
    )

    values = result.aggregate_payload["dataValueSets"][0]["dataValues"]
    assert [item["dataElement"] for item in values] == [
        "deReferrals1",
        "deVisits001",
    ]
    attributes = result.tracker_payload["trackedEntities"][0]["attributes"]
    assert [item["attribute"] for item in attributes] == [
        "attrName001",
        "attrNatId01",
        "attrPhone01",
    ]


def test_single_aggregate_and_event_only_payloads_are_normalized():
    aggregate = _load_fixture("aggregate_payload.json")["dataValueSets"][0]
    event = _load_fixture("tracker_payload.json")["events"][0]

    result = export_dhis2(
        aggregate,
        {"events": [event]},
        _load_fixture("organisation_units.json"),
        config=DHIS2ExportConfig(date_mode="none"),
        text_redactor=_offline_redactor,
    )

    validate_aggregate_payload(result.aggregate_payload)
    validate_tracker_payload(result.tracker_payload)
    assert result.tracker_payload["trackedEntities"] == []


def test_export_uses_only_the_local_snapshot(monkeypatch):
    def reject_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network access is forbidden during DHIS2 export")

    monkeypatch.setattr(socket, "create_connection", reject_network)

    result = export_dhis2(
        _load_fixture("aggregate_payload.json"),
        _load_fixture("tracker_payload.json"),
        FIXTURE_DIR / "organisation_units.json",
        config=DHIS2ExportConfig(date_mode="none"),
        text_redactor=_offline_redactor,
    )

    assert result.manifest["counts"]["data_value_sets"] == 1


def test_default_redactor_uses_existing_local_deidentification_pipeline(monkeypatch):
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_deidentify(value: str, **kwargs: Any) -> SimpleNamespace:
        calls.append((value, kwargs))
        return SimpleNamespace(deidentified_text=_offline_redactor(value))

    monkeypatch.setattr("openmed.core.pii.deidentify", fake_deidentify)
    exporter = DHIS2Exporter(
        _load_fixture("organisation_units.json"),
        config=DHIS2ExportConfig(date_mode="none"),
    )
    result = exporter.export(
        _load_fixture("aggregate_payload.json"),
        _load_fixture("tracker_payload.json"),
    )

    assert calls
    assert all(kwargs["method"] == "mask" for _, kwargs in calls)
    assert all(kwargs["policy"] == "hipaa_safe_harbor" for _, kwargs in calls)
    assert_redacted(result.aggregate_json() + result.tracker_json(), REDACTION_MAPPING)


def test_unknown_org_unit_fails_closed_without_echoing_the_uid():
    tracker = _load_fixture("tracker_payload.json")
    tracker["events"][0]["orgUnit"] = "ouSecret999"

    with pytest.raises(DHIS2ExportError) as exc_info:
        export_dhis2(
            None,
            tracker,
            _load_fixture("organisation_units.json"),
            text_redactor=_offline_redactor,
        )

    assert "absent from the local snapshot" in str(exc_info.value)
    assert "ouSecret999" not in str(exc_info.value)


@pytest.mark.parametrize(
    "target", ["storedBy", "comment", "attribute", "data_value", "note"]
)
def test_sensitive_text_values_fail_closed_on_non_strings(target: str):
    tracker = _load_fixture("tracker_payload.json")
    event = tracker["events"][0]
    invalid_value = {"raw": "Amina Example"}
    if target == "storedBy":
        event["storedBy"] = invalid_value
    elif target == "comment":
        event["dataValues"][0]["comment"] = invalid_value
    elif target == "attribute":
        tracker["trackedEntities"][0]["attributes"][0]["value"] = invalid_value
    elif target == "data_value":
        event["dataValues"][0]["value"] = invalid_value
    else:
        event["notes"][0]["value"] = invalid_value

    with pytest.raises(DHIS2ExportError, match="must be a string or null") as exc_info:
        export_dhis2(
            None,
            tracker,
            _load_fixture("organisation_units.json"),
            text_redactor=_offline_redactor,
        )

    assert "Amina Example" not in str(exc_info.value)


def test_malformed_datetime_suffix_fails_closed_without_echoing_raw_value():
    tracker = _load_fixture("tracker_payload.json")
    tracker["events"][0]["occurredAt"] = "2026-07-17 Amina Example"

    with pytest.raises(DHIS2ExportError, match="ISO YYYY-MM-DD") as exc_info:
        export_dhis2(
            None,
            tracker,
            _load_fixture("organisation_units.json"),
            text_redactor=_offline_redactor,
        )

    assert "Amina Example" not in str(exc_info.value)


def test_snapshot_parent_links_and_levels_are_validated():
    missing_parent = {
        "organisationUnits": [
            {"id": "ouFacility1", "level": 4, "parent": {"id": "missing"}}
        ]
    }
    invalid_levels = {
        "organisationUnits": [
            {"id": "ouDistrict1", "level": 3},
            {
                "id": "ouFacility1",
                "level": 3,
                "parent": {"id": "ouDistrict1"},
            },
        ]
    }

    with pytest.raises(DHIS2ExportError, match="parent absent"):
        DHIS2Exporter(missing_parent)
    with pytest.raises(DHIS2ExportError, match="lower than child"):
        DHIS2Exporter(invalid_levels)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"generalization_level": 0}, "generalization_level"),
        ({"small_cell_threshold": -1}, "small_cell_threshold"),
        ({"date_shift_days": 0}, "date_shift_days"),
        ({"date_shift_days": True}, "date_shift_days"),
        ({"date_shift_days": 1.5}, "date_shift_days"),
        (
            {"date_mode": "coarsen", "date_shift_days": 10},
            "requires date_mode",
        ),
    ],
)
def test_invalid_privacy_configuration_is_rejected(
    kwargs: dict[str, Any], message: str
):
    with pytest.raises((TypeError, ValueError), match=message):
        DHIS2ExportConfig(**kwargs)
