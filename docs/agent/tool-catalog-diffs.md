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

## Pin implementations for a run

`openmed.agent.tool_catalog_binding` binds a reviewed inventory to exact-version
executable registrations in the existing Python MCP `ToolRegistry`. A catalog
diff remains review evidence; use `RunToolCatalog` when dispatch must preserve
the implementation that was reviewed.

```python
from openmed.agent.correlation import RunId
from openmed.agent.tool_catalog_binding import (
    RunToolCatalog,
    RunToolCatalogSnapshot,
    tool_spec_schema_digest,
    tool_spec_side_effect_class,
)
from openmed.agent.tool_inventory import ToolInventory, ToolInventoryRecord
from openmed.mcp.tool_registry import ToolRegistry, ToolSpec

def local_count(value=0):
    return {"count": value + 1}

spec = ToolSpec(
    name="local_count",
    description="Count synthetic local values",
    input_schema={"type": "object", "properties": {"value": {"type": "integer"}}},
    output_schema={"type": "object", "properties": {"count": {"type": "integer"}}},
    read_only_hint=True,
    destructive_hint=False,
)
registry = ToolRegistry()
registry.register(spec, handler=local_count)
tool_id = "tool:org.example/local-count"
inventory = ToolInventory.from_records([
    ToolInventoryRecord(
        tool_id=tool_id,
        version=spec.version,
        capability_class="capability:org.example/local-read@1.0.0",
        side_effect_class=tool_spec_side_effect_class(spec),
        schema_digest=tool_spec_schema_digest(spec),
    )
])
names = {(tool_id, spec.version): spec.name}
catalog = RunToolCatalog.capture(RunId.generate(), inventory, registry, names)
review_reference = catalog.snapshot.digest
eligibility = catalog.check(registry).to_dict()
# After the application's separate review/permission checks:
result = catalog.invoke(registry, tool_id, spec.version, {"value": 7})
# result == {"count": 8}

evidence = catalog.snapshot.to_dict()
snapshot = RunToolCatalogSnapshot.from_dict(evidence)
resumed = RunToolCatalog.restore(snapshot, registry, names)
assert resumed.snapshot.digest == review_reference
```

The snapshot binds its opaque run ID, canonical tool/capability identifiers,
exact version, input/output schema digest, side-effect class and opaque
implementation identity. Canonical JSON and the snapshot digest are stable
when the run and registrations are unchanged. Registration order is irrelevant.
The runtime-name mapping and executable references are retained only in memory.

Each executable registration receives a random registry-owned identity. A new
registry, even with the same names, versions, schemas and callable, requires
re-review. These identities attest registration instances, not source-code
integrity. Process restarts fail closed rather than claiming reproducible code
attestation. Restore succeeds only against the original registrations; there is
no automatic refresh, implicit version upgrade or fallback to another handler.
Schema digests cover canonical input and output schemas, including all schema
annotations. Risk classes follow registry hints conservatively; pure `none`
is not inferred because the registry does not declare pure computation.

Call `check(current_registry)` after a reload and immediately before using a
preview or resuming. Its deterministic report contains the snapshot digest,
`eligible` or `re-review-required`, `preview_valid`, `resume_eligible` and fixed
reason codes. `invoke` checks the entire snapshot again before dispatch and
calls the captured handler even if a reload races with that final check. A
replacement can never become the dispatched callable through a name lookup.
Removal, missing handlers or missing exact versions produce `tool_unavailable`;
changed registrations, schemas and risk classes produce
`implementation_changed`, `schema_changed` and `side_effect_changed`.
Unavailable registries and malformed schemas fail closed. Re-review requires
capturing and explicitly approving a new snapshot; the original is immutable.

Snapshot parsing rejects unknown fields at every level. Endpoints, credentials,
arguments, patient identifiers, private paths, schema payloads and runtime
names never enter snapshot or eligibility artifacts. Handler failures become
the value-free `invocation_failed` code. Governance identifiers must come from
trusted developer-authored registrations, never patient content; opaque run
and implementation identities are generated independently of clinical data.
Tool results remain protected runtime data and must not be treated as audit
evidence. This binding layer adds no network calls, telemetry or dependencies.

This issue's independent boundary is the existing Python registry and dispatch
binding. It adds no workflow scheduler or automatic clinical effects. Application
owners must still enforce capability, input/output, scope and human-approval
contracts and pass the current registry into each check and invocation.
OpenMedKit's review and receipt surface is tracked separately in #3669; this
Python registry slice adds no Apple Foundation Models or cloud fallback path.
