"""Small offline checker for the DHIS2 shapes exercised by the fixtures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate_aggregate_payload(payload: Any) -> None:
    """Assert the documented aggregate data-value-set envelope shape."""

    assert isinstance(payload, Mapping)
    assert isinstance(payload.get("dataValueSets"), list)
    for data_value_set in payload["dataValueSets"]:
        _require_strings(data_value_set, "dataSet", "period", "orgUnit")
        assert isinstance(data_value_set.get("dataValues"), list)
        for data_value in data_value_set["dataValues"]:
            _require_strings(data_value, "dataElement", "value")


def validate_tracker_payload(payload: Any) -> None:
    """Assert the documented tracker import envelope shape."""

    assert isinstance(payload, Mapping)
    assert isinstance(payload.get("trackedEntities"), list)
    assert isinstance(payload.get("events"), list)
    for entity in payload["trackedEntities"]:
        _require_strings(entity, "trackedEntity", "trackedEntityType", "orgUnit")
        assert isinstance(entity.get("attributes"), list)
        for attribute in entity["attributes"]:
            _require_strings(attribute, "attribute", "value")
    for event in payload["events"]:
        _require_strings(
            event,
            "event",
            "program",
            "programStage",
            "orgUnit",
            "occurredAt",
        )
        assert isinstance(event.get("dataValues"), list)
        for data_value in event["dataValues"]:
            _require_strings(data_value, "dataElement", "value")
        if "notes" in event:
            assert isinstance(event["notes"], list)
            for note in event["notes"]:
                _require_strings(note, "value")


def _require_strings(value: Any, *keys: str) -> None:
    assert isinstance(value, Mapping)
    for key in keys:
        assert isinstance(value.get(key), str)
        assert value[key]
