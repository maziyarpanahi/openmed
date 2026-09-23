# Agent tool inventory JSON Schema

`openmed.agent.tool_inventory_schema` exports a strict Draft 2020-12 contract
for the content-free records documented in
[PHI-safe agent tool inventory](tool-inventory.md). The export is deterministic,
self-contained, and safe to generate offline.

## Export the schema

Use `build_tool_inventory_schema()` when an adapter needs a JSON-compatible
mapping, or `render_tool_inventory_schema()` when it needs canonical compact
JSON bytes:

```python
from openmed.agent.tool_inventory_schema import (
    build_tool_inventory_schema,
    render_tool_inventory_schema,
)

schema = build_tool_inventory_schema()
schema_json = render_tool_inventory_schema()
```

Repeated calls return independent mappings and identical JSON text. The
renderer uses sorted keys, ASCII escaping, and fixed separators. It does not
read configuration, import a schema validator, contact a registry, load an
agent provider, or execute a tool.

The schema uses only a fragment-local `$ref` into its own `$defs`. Consumers
can therefore validate it without network resolution or a hosted registry.

## Closed contract

Both the inventory object and each tool record set
`additionalProperties: false`. The schema requires the same fields exposed by
`ToolInventory` and `ToolInventoryRecord`, and it binds the inventory version
and side-effect enum to their Python sources of truth. Tool and capability
identifiers, semantic versions, and SHA-256 digests use bounded strings and
strict patterns. The tools array is bounded by the public inventory record
limit.

Unknown fields are invalid, including descriptions, endpoints, filesystem
paths, credentials, secrets, request arguments, results, examples, and raw
clinical identifiers. The schema describes inventory records only; it does not
define tool argument, tool result, endpoint, or credential schemas.

JSON Schema validation checks the serialized contract. Constructing a
`ToolInventory` remains the source of truth for Python runtime invariants such
as deterministic record ordering and duplicate tool-version rejection.
