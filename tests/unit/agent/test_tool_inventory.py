from __future__ import annotations

import json
from typing import Any

import pytest

from openmed.agent.tool_inventory import (
    TOOL_INVENTORY_SCHEMA_VERSION,
    CapabilityClass,
    SideEffectClass,
    ToolEntry,
    ToolInventory,
    ToolInventoryError,
)

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64


def _entry(
    tool_id: str,
    *,
    version: str = "1.0.0",
    capabilities: tuple[CapabilityClass, ...] = (CapabilityClass.READ,),
    side_effect: SideEffectClass = SideEffectClass.NONE,
    digest: str = _DIGEST_A,
) -> ToolEntry:
    return ToolEntry(
        tool_id=tool_id,
        version=version,
        capability_classes=capabilities,
        side_effect_class=side_effect,
        schema_digest=digest,
    )


def test_empty_inventory_has_stable_output() -> None:
    inventory = ToolInventory.from_entries([])

    assert inventory.entries == ()
    assert json.loads(inventory.to_json()) == {
        "schema_version": TOOL_INVENTORY_SCHEMA_VERSION,
        "tool_count": 0,
        "tools": [],
    }


def test_single_tool_inventory_has_stable_golden_output() -> None:
    inventory = ToolInventory.from_entries([_entry("retrieve-record")])

    assert json.loads(inventory.to_json()) == {
        "schema_version": TOOL_INVENTORY_SCHEMA_VERSION,
        "tool_count": 1,
        "tools": [
            {
                "tool_id": "retrieve-record",
                "version": "1.0.0",
                "capability_classes": ["read"],
                "side_effect_class": "none",
                "schema_digest": _DIGEST_A,
            }
        ],
    }


def test_multi_tool_inventory_is_sorted_by_tool_id() -> None:
    inventory = ToolInventory.from_entries(
        [_entry("zebra"), _entry("apple"), _entry("mango")]
    )

    assert [entry.tool_id for entry in inventory.entries] == [
        "apple",
        "mango",
        "zebra",
    ]


def test_duplicate_tool_ids_are_rejected() -> None:
    with pytest.raises(ToolInventoryError, match="entries: duplicate_tool_id"):
        ToolInventory.from_entries([_entry("shared"), _entry("shared")])


def test_markdown_matches_golden_output() -> None:
    inventory = ToolInventory.from_entries(
        [
            _entry(
                "retrieve-record",
                capabilities=(CapabilityClass.READ, CapabilityClass.WRITE),
                side_effect=SideEffectClass.READ_ONLY,
                digest=_DIGEST_A,
            )
        ]
    )

    assert inventory.to_markdown() == (
        "# Agent Tool Inventory\n"
        "\n"
        "| Tool | Version | Capabilities | Side effect | Schema digest |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `retrieve-record` | `1.0.0` | `read`, `write` | `read_only` | "
        f"`{_DIGEST_A}` |\n"
    )


def test_json_is_deterministic() -> None:
    entries = [_entry("apple"), _entry("zebra")]

    assert (
        ToolInventory.from_entries(entries).to_json()
        == ToolInventory.from_entries(list(reversed(entries))).to_json()
    )


@pytest.mark.parametrize(
    "tool_id",
    [
        "https://api.example.com/v1/tools",
        "/home/clinic/tool",
        "user@example.com",
        "tool with spaces",
        "urn:service:endpoint",
    ],
)
def test_endpoint_and_path_identifiers_fail_closed(tool_id: str) -> None:
    with pytest.raises(ToolInventoryError, match=r"^tool_id: invalid_identifier$"):
        _entry(tool_id)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"tool_id": 42}, "tool_id: invalid_identifier"),
        ({"tool_id": None}, "tool_id: invalid_identifier"),
        ({"version": "1.0"}, "version: invalid_version"),
        ({"version": "v1.0.0"}, "version: invalid_version"),
        ({"version": "1.0.0-beta"}, "version: invalid_version"),
        ({"version": "01.0.0"}, "version: invalid_version"),
        (
            {"schema_digest": "sha256:" + "g" * 64},
            "schema_digest: invalid_digest",
        ),
        ({"schema_digest": _DIGEST_A[:-1]}, "schema_digest: invalid_digest"),
        ({"schema_digest": "not-a-digest"}, "schema_digest: invalid_digest"),
        ({"capability_classes": ()}, "capability_classes: empty"),
        (
            {"capability_classes": ("read", "read")},
            "capability_classes: duplicate_item",
        ),
        (
            {"capability_classes": (CapabilityClass.WRITE, CapabilityClass.READ)},
            "capability_classes: not_sorted_unique",
        ),
        (
            {"capability_classes": ("exfiltrate",)},
            "capability_classes: unknown_class",
        ),
        ({"capability_classes": "read"}, "capability_classes: invalid_sequence"),
        (
            {"side_effect_class": "launch_missiles"},
            "side_effect_class: unknown_class",
        ),
        ({"side_effect_class": 123}, "side_effect_class: unknown_class"),
    ],
)
def test_entry_construction_rejects_unsafe_values_without_echo(
    updates: dict[str, Any], message: str
) -> None:
    fields: dict[str, Any] = {
        "tool_id": "retrieve-record",
        "version": "1.0.0",
        "capability_classes": (CapabilityClass.READ,),
        "side_effect_class": SideEffectClass.NONE,
        "schema_digest": _DIGEST_A,
    }
    fields.update(updates)

    with pytest.raises(ToolInventoryError) as exc_info:
        ToolEntry(**fields)

    assert str(exc_info.value) == message


def test_inventory_construction_rejects_unsorted_and_invalid_entries() -> None:
    apple = _entry("apple")
    zebra = _entry("zebra")

    with pytest.raises(ToolInventoryError, match="entries: not_sorted"):
        ToolInventory(entries=(zebra, apple))
    with pytest.raises(ToolInventoryError, match="entries: invalid_item"):
        ToolInventory(entries=(apple, "not-a-tool"))
    with pytest.raises(ToolInventoryError, match="schema_version: unsupported_version"):
        ToolInventory(entries=(apple,), schema_version="other")


def test_free_form_descriptions_never_enter_output() -> None:
    sentinel = "Synthetic_Patient_Secret_987"
    inventory = ToolInventory.from_entries([_entry("retrieve-record")])

    rendered = inventory.to_json() + inventory.to_markdown()
    assert sentinel not in rendered
    assert "endpoint" not in rendered
    assert "Bearer" not in rendered


def test_capability_and_side_effect_classes_are_closed() -> None:
    assert {capability.value for capability in CapabilityClass} == {
        "read",
        "write",
        "query",
        "compute",
        "notify",
    }
    assert {side_effect.value for side_effect in SideEffectClass} == {
        "none",
        "read_only",
        "state_mutation",
        "external",
    }


def test_inventory_is_exported_from_agent_package() -> None:
    import openmed.agent as agent

    assert agent.ToolInventory is ToolInventory
    assert agent.ToolEntry is ToolEntry
    assert agent.ToolInventoryError is ToolInventoryError
    assert agent.CapabilityClass is CapabilityClass
    assert agent.SideEffectClass is SideEffectClass
    assert agent.TOOL_INVENTORY_SCHEMA_VERSION == TOOL_INVENTORY_SCHEMA_VERSION
