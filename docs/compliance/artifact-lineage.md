# Privacy-artifact lineage manifests

OpenMed can represent the lineage of derived privacy artifacts without
retaining the source payload. A manifest contains only SHA-256 digests, typed
parent references, transformation names, policy fingerprints, and schema
versions. It is a local review artifact; creating or verifying one performs no
network access and does not provide a compliance certification or clinical
decision guarantee.

## Build a manifest

Use `ArtifactLineageNode.create` so each node hash commits to its complete
metadata. Policy metadata can be fingerprinted before it is placed in the
manifest:

```python
from openmed.compliance import (
    ArtifactLineageNode,
    ArtifactLineageParent,
    build_artifact_lineage_manifest,
    compute_policy_fingerprint,
)

policy_fingerprint = compute_policy_fingerprint(
    {"name": "local-policy", "revision": "2026-08"}
)
source = ArtifactLineageNode.create(
    artifact_type="source-input",
    transformation="ingest",
    policy_fingerprint=policy_fingerprint,
    schema_version=1,
)
redacted = ArtifactLineageNode.create(
    artifact_type="redacted-output",
    parents=(ArtifactLineageParent("source-input", source.artifact_hash),),
    transformation="redact",
    policy_fingerprint=policy_fingerprint,
    schema_version=1,
)
manifest = build_artifact_lineage_manifest((source, redacted))
manifest.write_json("artifact-lineage.json")
```

The resulting JSON contains no source text or raw identifiers. Parent
references include both the expected artifact type and its digest, which
prevents a digest from being silently reused for the wrong kind of artifact.
Records are sorted before hashing, so equivalent manifests built in different
input orders have the same serialized bytes and manifest hash.

## Verify safely

```python
diagnostics = manifest.verify()
if not diagnostics.valid:
    raise ValueError(diagnostics.to_dict())
```

Verification returns counts only:

| Field | Meaning |
| --- | --- |
| `cycle_count` | Strongly connected components containing a cycle. |
| `missing_parent_count` | Parent digests absent from the manifest. |
| `hash_mismatch_count` | Stale node hashes, wrong typed references, or a stale manifest hash. |
| `duplicate_hash_count` | Additional nodes sharing an artifact digest. |
| `node_count` / `parent_reference_count` | Aggregate manifest sizes. |
| `valid` | Whether every integrity count is zero. |

Diagnostics never include the offending digest, transformation, policy value,
or source payload. Callers should retain the manifest and counts according to
their local retention policy and should keep any original payload outside the
manifest.
