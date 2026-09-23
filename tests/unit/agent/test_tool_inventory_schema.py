"""Tests for the PHI-safe tool inventory JSON Schema export."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import fields
from typing import Any, NoReturn, cast

import pytest
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry

from openmed.agent.tool_inventory import (
    MAX_TOOL_INVENTORY_RECORDS,
    MAX_TOOL_VERSION_LENGTH,
    TOOL_INVENTORY_SCHEMA_VERSION,
    SideEffectClass,
    ToolInventory,
    ToolInventoryRecord,
)
from openmed.agent.tool_inventory_schema import (
    TOOL_INVENTORY_SCHEMA_DIALECT,
    build_tool_inventory_schema,
    render_tool_inventory_schema,
)


def _payload() -> dict[str, Any]:
    return ToolInventory.from_records(
        [
            ToolInventoryRecord(
                tool_id="tool:org.example/summarize",
                version="1.2.3",
                capability_class=("capability:org.example/clinical-transform@1.0.0"),
                side_effect_class=SideEffectClass.NONE,
                schema_digest="sha256:" + "a" * 64,
            )
        ]
    ).to_dict()


def _record(payload: dict[str, Any]) -> dict[str, Any]:
    return cast(list[dict[str, Any]], payload["tools"])[0]


def _validator() -> Draft202012Validator:
    def reject_remote_resolution(uri: str) -> NoReturn:
        raise AssertionError(f"unexpected remote schema resolution: {uri}")

    schema = build_tool_inventory_schema()
    return Draft202012Validator(
        schema,
        registry=Registry(retrieve=reject_remote_resolution),
    )


def test_schema_is_valid_draft_2020_12_and_resolves_locally() -> None:
    schema = build_tool_inventory_schema()

    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == TOOL_INVENTORY_SCHEMA_DIALECT
    _validator().validate(
        {"schema_version": TOOL_INVENTORY_SCHEMA_VERSION, "tools": []}
    )
    _validator().validate(_payload())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"description": "synthetic clinical note"}),
        lambda payload: _record(payload).update(
            {"endpoint": "https://private.example.test"}
        ),
        lambda payload: _record(payload).update(
            {"tool_id": "tool:org.example/summarize@1.2.3"}
        ),
        lambda payload: _record(payload).update({"version": "latest"}),
        lambda payload: _record(payload).update(
            {"capability_class": "clinical summary"}
        ),
        lambda payload: _record(payload).update(
            {"side_effect_class": "sometimes-writes"}
        ),
        lambda payload: _record(payload).update(
            {"schema_digest": "sha256:synthetic-patient-content"}
        ),
        lambda payload: _record(payload).pop("schema_digest"),
    ],
)
def test_malformed_or_content_bearing_payloads_fail(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(ValidationError):
        _validator().validate(payload)


def test_inventory_size_and_strings_are_bounded() -> None:
    schema = build_tool_inventory_schema()
    record = schema["$defs"]["tool_inventory_record"]

    assert schema["properties"]["tools"]["maxItems"] == MAX_TOOL_INVENTORY_RECORDS
    assert record["properties"]["version"]["maxLength"] == MAX_TOOL_VERSION_LENGTH
    assert record["properties"]["tool_id"]["maxLength"] == 512
    assert record["properties"]["capability_class"]["maxLength"] == 512
    assert record["properties"]["schema_digest"]["maxLength"] == 71


def test_schema_fields_enums_and_version_track_python_sources() -> None:
    schema = build_tool_inventory_schema()
    record = schema["$defs"]["tool_inventory_record"]

    assert set(schema["properties"]) == {field.name for field in fields(ToolInventory)}
    assert set(schema["required"]) == set(schema["properties"])
    assert set(record["properties"]) == {
        field.name for field in fields(ToolInventoryRecord)
    }
    assert set(record["required"]) == set(record["properties"])
    assert record["properties"]["side_effect_class"]["enum"] == [
        item.value for item in SideEffectClass
    ]
    assert (
        schema["properties"]["schema_version"]["const"] == TOOL_INVENTORY_SCHEMA_VERSION
    )


def test_schema_rendering_is_byte_stable_and_returns_fresh_mappings() -> None:
    first = build_tool_inventory_schema()
    first["properties"].clear()

    encoded = render_tool_inventory_schema()

    assert build_tool_inventory_schema()["properties"]
    assert encoded == render_tool_inventory_schema()
    assert hashlib.sha256(encoded.encode("ascii")).hexdigest() == (
        "830bd449a4a74d47708c2c842ba4ef5899de333ecdcb3096e198869e2cd66463"
    )
