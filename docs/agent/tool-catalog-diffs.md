# Agent Tool Catalog Diffs

`openmed.agent.diff_tool_catalogs()` compares two
[PHI-safe tool inventories](tool-inventory.md) and reports what changed using
only content-free fields: tool identifier, version, schema digest, capability
classes, and side-effect class. It never discloses local endpoints, headers,
arguments, examples, or secrets, because both inputs must already be validated
`ToolInventory` values.

## What is compared

| Category | Diff output |
| --- | --- |
| Added tools | Sorted tool identifiers present only in `after` |
| Removed tools | Sorted tool identifiers present only in `before` |
| Changed tools | Per-field `before`/`after` values for the shared tool |
| Unchanged tools | Sorted tool identifiers that are byte-identical |

A tool is "changed" when its version, schema digest, capability classes, or
side-effect class differs; capability classes are split into sorted `added`
and `removed` sets. Equal catalogs produce a diff whose `changed` property is
`False`.

## Example

```python
from openmed.agent import (
    CapabilityClass,
    SideEffectClass,
    ToolEntry,
    ToolInventory,
    diff_tool_catalogs,
)

def entry(tool_id: str, version: str) -> ToolEntry:
    return ToolEntry(
        tool_id=tool_id,
        version=version,
        capability_classes=(CapabilityClass.READ,),
        side_effect_class=SideEffectClass.NONE,
        schema_digest="sha256:" + "a" * 64,
    )

before = ToolInventory.from_entries(
    [entry("retrieve-record", "1.0.0"), entry("legacy-lookup", "1.0.0")]
)
after = ToolInventory.from_entries(
    [entry("retrieve-record", "2.0.0"), entry("new-exporter", "1.0.0")]
)

diff = diff_tool_catalogs(before, after)
assert diff.changed
assert diff.tool_ids_added == ("new-exporter",)
assert diff.tool_ids_removed == ("legacy-lookup",)
assert diff.changes[0].version_before == "1.0.0"
json_payload = diff.to_json()
markdown_report = diff.to_markdown()
```

## Output contract

- `to_json()` returns compact JSON with sorted keys and the schema identifier
  `openmed.agent.tool_catalog_diff.v1`; `to_dict()` keeps a fixed field order.
- `to_markdown()` renders one table listing added, removed, changed, and
  unchanged tools in a stable order.
- Reversing or reordering the input inventories produces identical output for
  the same tool sets.

Direct construction of `ToolCatalogDiff` and `ToolChange` validates the same
closed vocabulary as inventories: bounded identifiers, semantic versions,
lowercase SHA-256 digests, and closed capability and side-effect classes.
Failures raise `ToolCatalogDiffError` with a stable `field: code` message that
never repeats the submitted value.

## Out of scope

The diff makes no compatibility decisions, performs no endpoint discovery, and
never executes tools. Use it to review catalog changes and apply your own
acceptance policy to the result.
