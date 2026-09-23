"""Golden tests for content-free clinical agent tool catalog diffs."""

from __future__ import annotations

import json
from typing import Any

import pytest

from openmed.agent.tool_catalog_diff import (
    ToolCatalogDiffError,
    ToolCatalogFieldChange,
    diff_tool_catalogs,
    render_tool_catalog_diff_json,
    render_tool_catalog_diff_markdown,
)
from openmed.agent.tool_inventory import (
    TOOL_INVENTORY_SCHEMA_VERSION,
    SideEffectClass,
    ToolInventory,
    ToolInventoryRecord,
)

SCHEMA_A = "sha256:" + "a" * 64
SCHEMA_B = "sha256:" + "b" * 64
SCHEMA_C = "sha256:" + "c" * 64
SCHEMA_D = "sha256:" + "d" * 64


def _record(
    *,
    tool_id: str,
    version: str = "1.0.0",
    capability_class: str = "capability:org.example/clinical-read@1.0.0",
    side_effect_class: SideEffectClass = SideEffectClass.READ_ONLY,
    schema_digest: str = SCHEMA_A,
) -> ToolInventoryRecord:
    return ToolInventoryRecord(
        tool_id=tool_id,
        version=version,
        capability_class=capability_class,
        side_effect_class=side_effect_class,
        schema_digest=schema_digest,
    )


def _mixed_catalogs() -> tuple[ToolInventory, ToolInventory]:
    unchanged = _record(tool_id="tool:org.example/lookup", schema_digest=SCHEMA_B)
    before_changed = _record(
        tool_id="tool:org.example/summarize",
        capability_class="capability:org.example/clinical-transform@1.0.0",
        side_effect_class=SideEffectClass.NONE,
        schema_digest=SCHEMA_A,
    )
    after_changed = _record(
        tool_id="tool:org.example/summarize",
        capability_class="capability:org.example/clinical-write@1.0.0",
        side_effect_class=SideEffectClass.IDEMPOTENT_WRITE,
        schema_digest=SCHEMA_C,
    )
    removed = _record(
        tool_id="tool:org.example/export",
        capability_class="capability:org.example/clinical-write@1.0.0",
        side_effect_class=SideEffectClass.DESTRUCTIVE,
        schema_digest=SCHEMA_D,
    )
    added = _record(
        tool_id="tool:org.example/approve",
        capability_class="capability:org.example/clinical-write@1.0.0",
        side_effect_class=SideEffectClass.IDEMPOTENT_WRITE,
        schema_digest=SCHEMA_D,
    )
    return (
        ToolInventory.from_records([before_changed, removed, unchanged]),
        ToolInventory.from_records([unchanged, added, after_changed]),
    )


def test_added_removed_changed_and_unchanged_have_stable_golden_json() -> None:
    baseline, candidate = _mixed_catalogs()

    rendered = diff_tool_catalogs(baseline, candidate).to_json()

    assert rendered == (
        '{"added":[{"capability_class":"capability:org.example/clinical-write@1.0.0",'
        f'"schema_digest":"{SCHEMA_D}","side_effect_class":"idempotent-write",'
        '"tool_id":"tool:org.example/approve","version":"1.0.0"}],'
        '"changed":[{"fields":['
        '{"after":"capability:org.example/clinical-write@1.0.0",'
        '"before":"capability:org.example/clinical-transform@1.0.0",'
        '"field":"capability_class"},'
        '{"after":"idempotent-write","before":"none",'
        '"field":"side_effect_class"},'
        f'{{"after":"{SCHEMA_C}","before":"{SCHEMA_A}",'
        '"field":"schema_digest"}],'
        '"tool_id":"tool:org.example/summarize","version":"1.0.0"}],'
        '"removed":[{"capability_class":"capability:org.example/clinical-write@1.0.0",'
        f'"schema_digest":"{SCHEMA_D}","side_effect_class":"destructive",'
        '"tool_id":"tool:org.example/export","version":"1.0.0"}],'
        '"schema_version":"openmed.agent.tool_catalog_diff.v1",'
        '"unchanged":[{"capability_class":"capability:org.example/clinical-read@1.0.0",'
        f'"schema_digest":"{SCHEMA_B}","side_effect_class":"read-only",'
        '"tool_id":"tool:org.example/lookup","version":"1.0.0"}]}'
    )


def test_added_removed_changed_and_unchanged_have_stable_golden_markdown() -> None:
    baseline, candidate = _mixed_catalogs()

    rendered = diff_tool_catalogs(baseline, candidate).to_markdown()

    assert rendered == (
        "# Agent tool catalog diff\n"
        "\n"
        "Schema: `openmed.agent.tool_catalog_diff.v1`\n"
        "\n"
        "Added: 1\n"
        "Removed: 1\n"
        "Changed: 1\n"
        "Unchanged: 1\n"
        "\n"
        "## Added\n"
        "\n"
        "| Tool ID | Version | Capability class | Side-effect class | Schema digest |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `tool:org.example/approve` | `1.0.0` | "
        "`capability:org.example/clinical-write@1.0.0` | `idempotent-write` | "
        f"`{SCHEMA_D}` |\n"
        "\n"
        "## Removed\n"
        "\n"
        "| Tool ID | Version | Capability class | Side-effect class | Schema digest |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `tool:org.example/export` | `1.0.0` | "
        "`capability:org.example/clinical-write@1.0.0` | `destructive` | "
        f"`{SCHEMA_D}` |\n"
        "\n"
        "## Changed\n"
        "\n"
        "| Tool ID | Version | Field | Baseline | Candidate |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `tool:org.example/summarize` | `1.0.0` | `capability_class` | "
        "`capability:org.example/clinical-transform@1.0.0` | "
        "`capability:org.example/clinical-write@1.0.0` |\n"
        "| `tool:org.example/summarize` | `1.0.0` | `side_effect_class` | "
        "`none` | `idempotent-write` |\n"
        "| `tool:org.example/summarize` | `1.0.0` | `schema_digest` | "
        f"`{SCHEMA_A}` | `{SCHEMA_C}` |\n"
        "\n"
        "## Unchanged\n"
        "\n"
        "| Tool ID | Version | Capability class | Side-effect class | Schema digest |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| `tool:org.example/lookup` | `1.0.0` | "
        "`capability:org.example/clinical-read@1.0.0` | `read-only` | "
        f"`{SCHEMA_B}` |\n"
        "\n"
    )


def test_reordered_catalogs_are_unchanged_and_byte_stable() -> None:
    first = _record(tool_id="tool:org.example/first", version="10.0.0")
    second = _record(tool_id="tool:org.example/first", version="2.0.0")
    third = _record(tool_id="tool:org.example/second")
    baseline = {
        "schema_version": TOOL_INVENTORY_SCHEMA_VERSION,
        "tools": [third.to_dict(), first.to_dict(), second.to_dict()],
    }
    candidate = {
        "tools": [second.to_dict(), third.to_dict(), first.to_dict()],
        "schema_version": TOOL_INVENTORY_SCHEMA_VERSION,
    }

    diff = diff_tool_catalogs(baseline, candidate)

    assert diff.is_empty
    assert not diff.added
    assert not diff.removed
    assert not diff.changed
    assert [(record.tool_id, record.version) for record in diff.unchanged] == [
        ("tool:org.example/first", "2.0.0"),
        ("tool:org.example/first", "10.0.0"),
        ("tool:org.example/second", "1.0.0"),
    ]
    assert diff.to_json() == diff_tool_catalogs(candidate, baseline).to_json()
    assert diff.to_markdown() == diff_tool_catalogs(candidate, baseline).to_markdown()


def test_version_replacement_is_an_addition_and_removal() -> None:
    old = _record(tool_id="tool:org.example/summarize", version="1.0.0")
    new = _record(tool_id="tool:org.example/summarize", version="2.0.0")

    diff = diff_tool_catalogs(
        ToolInventory.from_records([old]), ToolInventory.from_records([new])
    )

    assert diff.added == (new,)
    assert diff.removed == (old,)
    assert not diff.changed
    assert not diff.unchanged


@pytest.mark.parametrize(
    "forbidden_field",
    ["endpoints", "headers", "arguments", "examples", "secrets"],
)
def test_content_bearing_catalog_fields_fail_closed(forbidden_field: str) -> None:
    sentinel = "Synthetic Patient secret at https://private.example.test"
    unsafe_record: dict[str, Any] = _record(
        tool_id="tool:org.example/summarize"
    ).to_dict()
    unsafe_record[forbidden_field] = sentinel
    candidate = {
        "schema_version": TOOL_INVENTORY_SCHEMA_VERSION,
        "tools": [unsafe_record],
    }

    with pytest.raises(
        ToolCatalogDiffError, match="candidate.tools: unreadable_records"
    ) as caught:
        diff_tool_catalogs(ToolInventory.from_records([]), candidate)

    assert sentinel not in str(caught.value)
    assert sentinel not in repr(caught.value)


def test_top_level_secrets_fail_closed_without_echoing_values() -> None:
    sentinel = "Bearer synthetic-private-token"
    candidate = {
        "schema_version": TOOL_INVENTORY_SCHEMA_VERSION,
        "tools": [],
        "credentials": sentinel,
    }

    with pytest.raises(
        ToolCatalogDiffError, match="candidate.inventory: invalid_fields"
    ) as caught:
        diff_tool_catalogs(ToolInventory.from_records([]), candidate)

    assert sentinel not in str(caught.value)


def test_public_change_records_reject_content_bearing_values() -> None:
    sentinel = "https://private.example.test/tool"

    with pytest.raises(ToolCatalogDiffError, match="change.after: invalid_identifier"):
        ToolCatalogFieldChange(
            field="capability_class",
            before="capability:org.example/clinical-read@1.0.0",
            after=sentinel,
        )


def test_renderers_require_a_typed_diff() -> None:
    with pytest.raises(ToolCatalogDiffError, match="diff: invalid_type"):
        render_tool_catalog_diff_json({})  # type: ignore[arg-type]
    with pytest.raises(ToolCatalogDiffError, match="diff: invalid_type"):
        render_tool_catalog_diff_markdown({})  # type: ignore[arg-type]


def test_json_projection_round_trips_as_plain_data() -> None:
    baseline, candidate = _mixed_catalogs()
    diff = diff_tool_catalogs(baseline, candidate)

    assert json.loads(diff.to_json()) == diff.to_dict()
