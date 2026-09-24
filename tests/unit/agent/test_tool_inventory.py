"""Golden tests for PHI-safe registered-agent tool inventories."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from itertools import repeat
from typing import Any

import pytest

from openmed.agent.tool_inventory import (
    MAX_TOOL_INVENTORY_RECORDS,
    TOOL_INVENTORY_SCHEMA_VERSION,
    SideEffectClass,
    ToolInventory,
    ToolInventoryError,
    ToolInventoryRecord,
    render_tool_inventory_json,
    render_tool_inventory_markdown,
)

SCHEMA_A = "sha256:" + "a" * 64
SCHEMA_B = "sha256:" + "b" * 64


def _record(
    *,
    tool_id: str = "tool:org.example/summarize",
    version: str = "1.0.0",
    capability_class: str = "capability:org.example/clinical-transform@1.0.0",
    side_effect_class: SideEffectClass = SideEffectClass.NONE,
    schema_digest: str = SCHEMA_A,
) -> ToolInventoryRecord:
    return ToolInventoryRecord(
        tool_id=tool_id,
        version=version,
        capability_class=capability_class,
        side_effect_class=side_effect_class,
        schema_digest=schema_digest,
    )


def test_empty_inventory_has_stable_golden_output() -> None:
    inventory = ToolInventory.from_records([])

    assert inventory.to_json() == (
        '{"schema_version":"openmed.agent.tool_inventory.v1","tools":[]}'
    )
    assert inventory.to_markdown() == (
        "# Agent tool inventory\n"
        "\n"
        "Schema: `openmed.agent.tool_inventory.v1`\n"
        "\n"
        "Registered tools: 0\n"
        "\n"
        "_No registered tools._\n"
    )


def test_single_tool_inventory_has_stable_golden_output() -> None:
    inventory = ToolInventory.from_records([_record()])

    assert inventory.to_json() == (
        '{"schema_version":"openmed.agent.tool_inventory.v1","tools":['
        '{"capability_class":"capability:org.example/clinical-transform@1.0.0",'
        f'"schema_digest":"{SCHEMA_A}","side_effect_class":"none",'
        '"tool_id":"tool:org.example/summarize","version":"1.0.0"}'
        "]}"
    )
    assert inventory.to_markdown() == (
        "# Agent tool inventory\n"
        "\n"
        "Schema: `openmed.agent.tool_inventory.v1`\n"
        "\n"
        "Registered tools: 1\n"
        "\n"
        "| Tool ID | Version | Capability class | Side-effect class | Schema digest |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `tool:org.example/summarize` | `1.0.0` | "
        "`capability:org.example/clinical-transform@1.0.0` | `none` | "
        f"`{SCHEMA_A}` |\n"
    )


def test_multi_tool_inventory_sorts_by_tool_and_version() -> None:
    first = _record(
        tool_id="tool:org.example/export",
        version="2.0.0",
        capability_class="capability:org.example/clinical-write@1.0.0",
        side_effect_class=SideEffectClass.IDEMPOTENT_WRITE,
        schema_digest=SCHEMA_B,
    )
    second = _record(version="1.1.0")
    third = _record(version="1.0.0")
    fourth = _record(version="10.0.0", schema_digest=SCHEMA_B)
    fifth = _record(version="2.0.0", schema_digest=SCHEMA_B)

    inventory = ToolInventory.from_records([fourth, second, first, fifth, third])

    assert [(item.tool_id, item.version) for item in inventory.tools] == [
        ("tool:org.example/export", "2.0.0"),
        ("tool:org.example/summarize", "1.0.0"),
        ("tool:org.example/summarize", "1.1.0"),
        ("tool:org.example/summarize", "2.0.0"),
        ("tool:org.example/summarize", "10.0.0"),
    ]
    assert (
        inventory.to_json()
        == ToolInventory.from_records([third, fifth, second, fourth, first]).to_json()
    )
    assert (
        inventory.to_markdown()
        == ToolInventory.from_records(
            [third, fifth, second, fourth, first]
        ).to_markdown()
    )


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "description",
        "endpoint",
        "path",
        "credential",
        "secret",
        "arguments",
        "example",
        "headers",
    ],
)
def test_content_bearing_record_fields_fail_closed(forbidden_field: str) -> None:
    sentinel = "Synthetic Patient credential at https://private.example.test"
    payload = _record().to_dict()
    payload[forbidden_field] = sentinel

    with pytest.raises(ToolInventoryError, match="tool: invalid_fields") as caught:
        ToolInventoryRecord.from_dict(payload)

    assert sentinel not in str(caught.value)
    assert sentinel not in repr(caught.value)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("tool_id", "https://private.example.test/tool", "invalid_identifier"),
        ("tool_id", "/private/agent/tool", "invalid_identifier"),
        ("tool_id", "tool:org.example/summarize@1.0.0", "version_must_be_separate"),
        ("version", "v1 latest", "invalid_semver"),
        ("capability_class", "patient discharge summary", "invalid_identifier"),
        ("schema_digest", "secret-api-token", "invalid_digest"),
        ("side_effect_class", "sometimes writes records", "invalid_class"),
    ],
)
def test_free_form_and_identifying_values_fail_closed(
    field: str, value: object, reason: str
) -> None:
    sentinel = str(value)
    payload = _record().to_dict()
    payload[field] = value

    with pytest.raises(ToolInventoryError, match=reason) as caught:
        ToolInventoryRecord.from_dict(payload)

    assert sentinel not in str(caught.value)


def test_inventory_mapping_round_trip_is_byte_stable() -> None:
    original = ToolInventory.from_records(
        [
            _record(),
            _record(
                tool_id="tool:org.example/export",
                capability_class="capability:org.example/clinical-write@1.0.0",
                side_effect_class=SideEffectClass.DESTRUCTIVE,
                schema_digest=SCHEMA_B,
            ),
        ]
    )

    restored = ToolInventory.from_dict(json.loads(original.to_json()))

    assert restored == original
    assert restored.to_json() == original.to_json()
    assert render_tool_inventory_json(restored) == original.to_json()
    assert render_tool_inventory_markdown(restored) == original.to_markdown()


def test_duplicate_tool_versions_fail_closed() -> None:
    with pytest.raises(ToolInventoryError, match="tools: duplicate_record"):
        ToolInventory.from_records([_record(), _record()])


def test_inventory_size_is_bounded_before_full_materialization() -> None:
    records = repeat(_record(), MAX_TOOL_INVENTORY_RECORDS + 1)

    with pytest.raises(ToolInventoryError, match="tools: too_many_records"):
        ToolInventory.from_records(records)


def test_inventory_rejects_top_level_content_metadata() -> None:
    sentinel = "Bearer synthetic-secret-token"
    payload = {
        "schema_version": TOOL_INVENTORY_SCHEMA_VERSION,
        "tools": [],
        "credentials": sentinel,
    }

    with pytest.raises(ToolInventoryError, match="inventory: invalid_fields") as caught:
        ToolInventory.from_dict(payload)

    assert sentinel not in str(caught.value)


def test_unreadable_inputs_fail_without_forwarding_private_exceptions() -> None:
    sentinel = "Synthetic Patient secret"

    class UnreadableMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise RuntimeError(sentinel)

        def __iter__(self) -> Iterator[str]:
            raise RuntimeError(sentinel)

        def __len__(self) -> int:
            raise RuntimeError(sentinel)

    with pytest.raises(ToolInventoryError, match="unreadable_mapping") as caught:
        ToolInventoryRecord.from_dict(UnreadableMapping())

    assert sentinel not in str(caught.value)


def test_renderers_require_a_typed_inventory() -> None:
    with pytest.raises(ToolInventoryError, match="inventory: invalid_type"):
        render_tool_inventory_json({})  # type: ignore[arg-type]
    with pytest.raises(ToolInventoryError, match="inventory: invalid_type"):
        render_tool_inventory_markdown({})  # type: ignore[arg-type]
