# PHI-safe Agent Tool Inventory

`openmed.agent.ToolInventory` renders the local agent tool surface without
revealing endpoints, credentials, arguments, or clinical data. Each entry
carries only a tool identifier, a semantic version, a closed set of capability
classes, a closed side-effect class, and a schema digest.

## What an entry carries

| Field | Constraint |
| --- | --- |
| `tool_id` | Bounded identifier; no slashes, colons, spaces, or URL forms |
| `version` | Semantic version `major.minor.patch` without a leading zero |
| `capability_classes` | Non-empty, sorted, unique subset of the closed set |
| `side_effect_class` | One of the closed side-effect classes |
| `schema_digest` | Lowercase `sha256:<64 hex>` digest |

The closed vocabulary is:

- Capability classes: `read`, `write`, `query`, `compute`, `notify`.
- Side-effect classes: `none`, `read_only`, `state_mutation`, `external`.

Anything outside these sets fails closed, so endpoints, paths, secrets,
free-form descriptions, and unknown classifications can never reach output.

## Example

```python
from openmed.agent import (
    CapabilityClass,
    SideEffectClass,
    ToolEntry,
    ToolInventory,
)

entries = [
    ToolEntry(
        tool_id="retrieve-record",
        version="1.0.0",
        capability_classes=(CapabilityClass.READ, CapabilityClass.QUERY),
        side_effect_class=SideEffectClass.READ_ONLY,
        schema_digest="sha256:" + "a" * 64,
    )
]

inventory = ToolInventory.from_entries(entries)
json_payload = inventory.to_json()
markdown_report = inventory.to_markdown()
```

## Output contract

- `ToolInventory.from_entries()` sorts entries by `tool_id`, so identical input
  sets always produce identical output regardless of input order.
- `to_json()` returns compact JSON with sorted keys and the schema identifier
  `openmed.agent.tool_inventory.v1`; `to_dict()` keeps a fixed field order.
- `to_markdown()` renders one table row per tool in the same sorted order.

Failures raise `ToolInventoryError` with a stable `field: code` message that
never repeats the submitted value.

## Out of scope

The inventory does not execute tools, probe endpoints, or render request and
response examples. It describes the content-free tool surface only.
