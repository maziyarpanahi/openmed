# PHI-safe agent tool inventory

`openmed.agent.tool_inventory` renders a deterministic snapshot of registered
agent tool metadata without executing a tool, loading a provider, or probing an
endpoint. The inventory contains only canonical tool and capability identifiers,
semantic versions, side-effect classes, and SHA-256 schema digests.

## Build an inventory

Create records from trusted, developer-authored registration metadata. Compute
the schema digest where the schema is registered, then pass only the digest to
the inventory boundary.

```python
from openmed.agent.tool_inventory import (
    SideEffectClass,
    ToolInventory,
    ToolInventoryRecord,
)

inventory = ToolInventory.from_records(
    [
        ToolInventoryRecord(
            tool_id="tool:org.example/summarize",
            version="1.0.0",
            capability_class="capability:org.example/clinical-transform@1.0.0",
            side_effect_class=SideEffectClass.NONE,
            schema_digest="sha256:" + "a" * 64,
        )
    ]
)

json_text = inventory.to_json()
markdown_text = inventory.to_markdown()
```

Records are sorted by tool ID and version. Reordering registration input does
not change either rendering. A duplicate tool ID and version fails closed.
Empty inventories are valid and render an explicit zero count.

The side-effect vocabulary is intentionally closed:

- `none`: pure local computation;
- `read-only`: inspection that cannot mutate the inspected resource;
- `idempotent-write`: replay-safe mutation;
- `non-idempotent-write`: mutation that is not replay-safe;
- `destructive`: deletion or another explicitly destructive mutation.

Side-effect classes describe mutation behavior only. They do not authorize a
tool, prove that an implementation follows its declaration, or replace signed
capability grants and human approval.

## Content-free boundary

`ToolInventoryRecord.from_dict()` and `ToolInventory.from_dict()` accept exact,
closed field sets. Unknown or missing fields fail with stable value-free errors.
Do not add descriptions, endpoints, filesystem paths, headers, credentials,
secrets, arguments, examples, clinical values, tenant names, or person and
record identifiers. Those values are neither needed nor safe in an operator
inventory.

Tool and capability values use the canonical governance identifier grammar
described in [Agent governance identifiers](governance-identifiers.md). The
tool identifier is unversioned because `version` is a separate required field.
Schema digests use lowercase `sha256:` plus 64 hexadecimal characters; schemas,
request examples, and response examples are never rendered.

All operations are local and deterministic. Inventory generation performs no
network access, endpoint discovery, credential validation, or tool execution.
