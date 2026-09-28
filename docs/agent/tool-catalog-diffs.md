# Content-free agent tool catalog diffs

`openmed.agent.tool_catalog_diff` compares two
[PHI-safe agent tool inventories](tool-inventory.md) for deployment review. It
reports only canonical tool and capability identifiers, semantic versions,
closed side-effect classes, and SHA-256 schema digests. It does not inspect a
provider, discover an endpoint, validate credentials, or execute a tool.

## Compare two snapshots

Pass typed `ToolInventory` snapshots or their exact JSON-compatible mappings:

```python
from openmed.agent.tool_catalog_diff import diff_tool_catalogs
from openmed.agent.tool_inventory import (
    SideEffectClass,
    ToolInventory,
    ToolInventoryRecord,
)

baseline = ToolInventory.from_records(
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
candidate = ToolInventory.from_records(
    [
        ToolInventoryRecord(
            tool_id="tool:org.example/summarize",
            version="1.0.0",
            capability_class="capability:org.example/clinical-transform@1.0.0",
            side_effect_class=SideEffectClass.NONE,
            schema_digest="sha256:" + "b" * 64,
        )
    ]
)

catalog_diff = diff_tool_catalogs(baseline, candidate)
json_text = catalog_diff.to_json()
markdown_text = catalog_diff.to_markdown()
```

Both renderers are deterministic. The JSON renderer emits canonical compact
JSON. The Markdown renderer emits stable counts and tables for added, removed,
changed, and unchanged entries. Registration order never affects either
output.

## Identity and change rules

Entries are keyed by `(tool_id, version)`. This preserves inventories that
register multiple versions of one tool:

- a new key is **added**;
- a missing key is **removed**;
- a shared key whose capability class, side-effect class, or schema digest
  differs is **changed**;
- a shared key with identical content-free metadata is **unchanged**.

A version replacement is one removal plus one addition, not an inferred
upgrade or compatibility decision. The diff deliberately does not decide
whether a change is backward compatible, safe to deploy, or authorized.

## Content-free boundary

Mapping inputs are parsed through the exact `ToolInventory` contract before
comparison. Unknown or missing fields fail closed with stable, value-free
errors. Endpoints, filesystem paths, headers, credentials, secrets, arguments,
descriptions, examples, clinical values, tenant names, and person or record
identifiers cannot enter the diff or its renderers.

Keep complete schemas, examples, and runtime configuration in their governed
source systems. Catalog diffs need only schema digests and canonical governance
metadata. This boundary keeps review artifacts suitable for offline sharing
without turning them into endpoint-discovery or credential-disclosure surfaces.
