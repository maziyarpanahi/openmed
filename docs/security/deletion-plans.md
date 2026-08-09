# Deletion impact plans

`openmed.risk.deletion_plan` provides a local, non-destructive preview before a
caller removes an OpenMed-managed cache, map, or evidence artifact. It is a
planning aid, not a compliance certification or a guarantee that every copy of
data is discoverable.

## Manifest shape

Pass a mapping with an `artifacts` list, a sequence of entries, or a path to a
local JSON manifest. Each entry contains an artifact hash, a safe kind, a
retention class, optional dependency links, and an ownership flag:

```python
from openmed.risk.deletion_plan import plan_deletion_impact

cache_hash = "sha256:" + "0" * 64
map_hash = "sha256:" + "1" * 64

manifest = {
    "artifacts": [
        {
            "artifact_hash": cache_hash,
            "kind": "cache",
            "retention_class": "short",
            "dependencies": [],
            "owned": True,
        },
        {
            "artifact_hash": map_hash,
            "kind": "map",
            "retention_class": "standard",
            "dependencies": [cache_hash],
            "owned": True,
        },
    ]
}

plan = plan_deletion_impact(manifest, cache_hash)
print(plan.to_json())
```

`dependencies` point from an entry to the artifacts it needs. The planner
follows those links in reverse and transitively, so deleting the cache above
reports both the cache and the dependent map as affected. Opaque local
references are converted to SHA-256 digests at the input boundary. Existing
SHA-256 and HMAC-SHA-256 digests are preserved in canonical lowercase form.

## Safe dry-run output

Planning is always a dry-run. `to_dict()`, `to_json()`, and `to_markdown()`
contain counts grouped by artifact kind and retention class, plus integrity
digests and safety counters. They do not include source paths, raw identifiers,
unknown manifest fields, or individual resource values. The plan object can
provide canonical hash references to an injected local executor, but those
references are not emitted by the report serializers.

Unowned targets and unresolved dependency links are reported as blocked safety
counters. Execution refuses to proceed while either counter is non-zero.

## Explicit execution boundary

The planner never performs deletion. If an application has its own local
deletion implementation, it must inject that callback and provide the exact
confirmation token for the reviewed plan:

```python
from openmed.risk.deletion_plan import execute_deletion_plan

def delete_owned_artifact(artifact):
    # Resolve artifact.artifact_hash through the application's local registry.
    # Do not pass raw paths or data into the plan or its reports.
    ...

result = execute_deletion_plan(
    plan,
    confirmation=plan.confirmation_token,
    executor=delete_owned_artifact,
)
```

The callback receives only a normalized `DeletionArtifact`, and target
callbacks run in stable hash order. Dependents are included in the impact
preview but are not deleted automatically. Callers remain responsible for
retention policy, legal holds, access control, backups, and human review.

The module has no network dependency or mandatory outbound call. Use synthetic
offline fixtures for tests and never place protected health information,
credentials, or raw paths in a manifest report.
