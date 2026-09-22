from __future__ import annotations

import json
from typing import Any

import pytest

from openmed.agent.tool_catalog_diff import (
    TOOL_CATALOG_DIFF_SCHEMA_VERSION,
    ToolCatalogDiff,
    ToolCatalogDiffError,
    ToolChange,
    diff_tool_catalogs,
)
from openmed.agent.tool_inventory import (
    CapabilityClass,
    SideEffectClass,
    ToolEntry,
    ToolInventory,
)

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64


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


def _catalog(*entries: ToolEntry) -> ToolInventory:
    return ToolInventory.from_entries(entries)


def test_equal_catalogs_have_no_changes() -> None:
    before = _catalog(_entry("apple"), _entry("mango"))

    diff = diff_tool_catalogs(before, before)

    assert diff.changed is False
    assert diff.tool_ids_added == ()
    assert diff.tool_ids_removed == ()
    assert diff.changes == ()
    assert diff.tool_ids_unchanged == ("apple", "mango")


def test_added_removed_and_unchanged_tools() -> None:
    before = _catalog(_entry("apple"), _entry("mango"))
    after = _catalog(_entry("apple"), _entry("zebra"))

    diff = diff_tool_catalogs(before, after)

    assert diff.changed is True
    assert diff.tool_ids_added == ("zebra",)
    assert diff.tool_ids_removed == ("mango",)
    assert diff.tool_ids_unchanged == ("apple",)
    assert diff.changes == ()


def test_changed_tool_captures_per_field_deltas() -> None:
    before = _catalog(
        _entry(
            "retrieve-record",
            version="1.0.0",
            capabilities=(CapabilityClass.READ,),
            side_effect=SideEffectClass.NONE,
            digest=_DIGEST_A,
        )
    )
    after = _catalog(
        _entry(
            "retrieve-record",
            version="2.0.0",
            capabilities=(CapabilityClass.READ, CapabilityClass.WRITE),
            side_effect=SideEffectClass.STATE_MUTATION,
            digest=_DIGEST_B,
        )
    )

    diff = diff_tool_catalogs(before, after)

    assert diff.changed is True
    (change,) = diff.changes
    assert change.tool_id == "retrieve-record"
    assert change.version_before == "1.0.0"
    assert change.version_after == "2.0.0"
    assert change.schema_digest_before == _DIGEST_A
    assert change.schema_digest_after == _DIGEST_B
    assert change.capability_classes_added == (CapabilityClass.WRITE,)
    assert change.capability_classes_removed == ()
    assert change.side_effect_class_before is SideEffectClass.NONE
    assert change.side_effect_class_after is SideEffectClass.STATE_MUTATION


def test_reordered_catalogs_produce_identical_diff() -> None:
    before = _catalog(_entry("apple"), _entry("mango"))
    after = _catalog(_entry("zebra"), _entry("apple"))

    forward = diff_tool_catalogs(before, after)
    reordered = diff_tool_catalogs(
        _catalog(_entry("mango"), _entry("apple")),
        _catalog(_entry("apple"), _entry("zebra")),
    )

    assert forward.to_json() == reordered.to_json()


def test_json_is_deterministic_and_ordered() -> None:
    before = _catalog(_entry("apple"), _entry("mango"))
    after = _catalog(_entry("apple"), _entry("zebra"))

    diff = diff_tool_catalogs(before, after)

    assert diff.to_json() == diff_tool_catalogs(before, after).to_json()
    assert list(diff.to_dict()) == [
        "schema_version",
        "changed",
        "tool_ids",
        "changes",
    ]
    payload = json.loads(diff.to_json())
    assert payload["schema_version"] == TOOL_CATALOG_DIFF_SCHEMA_VERSION
    assert payload["tool_ids"] == {
        "added": ["zebra"],
        "removed": ["mango"],
        "unchanged": ["apple"],
    }


def test_markdown_matches_golden_output() -> None:
    before = _catalog(_entry("apple"), _entry("mango"))
    after = _catalog(_entry("apple"), _entry("zebra"))

    assert diff_tool_catalogs(before, after).to_markdown() == (
        "# Agent Tool Catalog Diff\n"
        "\n"
        "Changed: yes\n"
        "\n"
        "## Tools\n"
        "\n"
        "| Change | Tool |\n"
        "| --- | --- |\n"
        "| added | `zebra` |\n"
        "| removed | `mango` |\n"
        "| unchanged | `apple` |\n"
    )


def test_unchanged_markdown_reports_no_change() -> None:
    before = _catalog(_entry("apple"))

    markdown = diff_tool_catalogs(before, before).to_markdown()

    assert "Changed: no\n" in markdown
    assert "| added |" not in markdown
    assert "| removed |" not in markdown
    assert "| changed |" not in markdown


@pytest.mark.parametrize(
    "value",
    [
        None,
        "a catalog",
        {"tools": []},
        [_entry("apple")],
    ],
)
@pytest.mark.parametrize("position", ["before", "after"])
def test_non_catalog_inputs_are_rejected(value: Any, position: str) -> None:
    valid = _catalog(_entry("apple"))
    arguments = {"before": valid, "after": valid, position: value}

    with pytest.raises(ToolCatalogDiffError, match=rf"^{position}: invalid_catalog$"):
        diff_tool_catalogs(**arguments)


def _diff_fields(**updates: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tool_ids_added": (),
        "tool_ids_removed": (),
        "tool_ids_unchanged": (),
        "changes": (),
    }
    fields.update(updates)
    return fields


def _change(**updates: Any) -> ToolChange:
    fields: dict[str, Any] = {
        "tool_id": "retrieve-record",
        "version_before": "1.0.0",
        "version_after": "2.0.0",
        "schema_digest_before": _DIGEST_A,
        "schema_digest_after": _DIGEST_B,
        "capability_classes_added": (CapabilityClass.WRITE,),
        "capability_classes_removed": (),
        "side_effect_class_before": SideEffectClass.NONE,
        "side_effect_class_after": SideEffectClass.STATE_MUTATION,
    }
    fields.update(updates)
    return ToolChange(**fields)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"tool_ids_added": ("b", "a")}, "tool_ids_added: not_sorted_unique"),
        ({"tool_ids_added": ("has spaces",)}, "tool_id: invalid_identifier"),
        (
            {"tool_ids_added": ("apple",), "tool_ids_removed": ("apple",)},
            "tool_ids: added_and_removed",
        ),
        (
            {"tool_ids_added": ("apple",), "tool_ids_unchanged": ("apple",)},
            "tool_ids: added_and_unchanged",
        ),
        ({"changes": ("not-a-change",)}, "changes: invalid_item"),
        ({"schema_version": "other"}, "schema_version: unsupported_version"),
    ],
)
def test_diff_construction_rejects_unsafe_values_without_echo(
    updates: dict[str, Any], message: str
) -> None:
    with pytest.raises(ToolCatalogDiffError) as exc_info:
        ToolCatalogDiff(**_diff_fields(**updates))

    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"version_before": "1.0"}, "version_before: invalid_version"),
        ({"version_after": "v2.0.0"}, "version_after: invalid_version"),
        ({"schema_digest_before": "bad"}, "schema_digest_before: invalid_digest"),
        (
            {"schema_digest_after": _DIGEST_B[:-1]},
            "schema_digest_after: invalid_digest",
        ),
        (
            {
                "capability_classes_added": (
                    CapabilityClass.WRITE,
                    CapabilityClass.WRITE,
                )
            },
            "capability_classes_added: duplicate_item",
        ),
        (
            {
                "capability_classes_added": (CapabilityClass.WRITE,),
                "capability_classes_removed": (CapabilityClass.WRITE,),
            },
            "capability_classes: added_and_removed",
        ),
        (
            {"side_effect_class_before": "exfiltrate"},
            "side_effect_class_before: unknown_class",
        ),
        (
            {
                "version_before": "1.0.0",
                "version_after": "1.0.0",
                "schema_digest_before": _DIGEST_A,
                "schema_digest_after": _DIGEST_A,
                "capability_classes_added": (),
                "capability_classes_removed": (),
                "side_effect_class_before": SideEffectClass.NONE,
                "side_effect_class_after": SideEffectClass.NONE,
            },
            "change: no_field_changed",
        ),
    ],
)
def test_change_construction_rejects_unsafe_values_without_echo(
    updates: dict[str, Any], message: str
) -> None:
    with pytest.raises(ToolCatalogDiffError) as exc_info:
        _change(**updates)

    assert str(exc_info.value) == message


def test_endpoints_paths_and_secrets_cannot_enter_diff_output() -> None:
    sentinel = "Synthetic_Patient_Secret_987"
    before = _catalog(_entry("retrieve-record"))
    after = _catalog(_entry("retrieve-record", version="2.0.0", digest=_DIGEST_B))

    rendered = (
        diff_tool_catalogs(before, after).to_json()
        + diff_tool_catalogs(before, after).to_markdown()
    )

    assert sentinel not in rendered
    assert "endpoint" not in rendered
    assert "Bearer" not in rendered


def test_diff_is_exported_from_agent_package() -> None:
    import openmed.agent as agent

    assert agent.diff_tool_catalogs is diff_tool_catalogs
    assert agent.ToolCatalogDiff is ToolCatalogDiff
    assert agent.ToolCatalogDiffError is ToolCatalogDiffError
    assert agent.ToolChange is ToolChange
    assert agent.TOOL_CATALOG_DIFF_SCHEMA_VERSION == TOOL_CATALOG_DIFF_SCHEMA_VERSION
