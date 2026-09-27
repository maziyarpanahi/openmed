# Sealed workflow manifests

Workflow benchmark results are comparable only when the executable submission
surface cannot change between submission and evaluation. OpenMed represents
that surface as a deterministic, digest-only manifest and verifies it again at
evaluation start. The process is local and performs no network requests.

## Governed components

Every manifest contains exactly one lowercase `sha256:` digest for each of
these components:

- `model`
- `tokenizer`
- `tool_inventory`
- `prompt`
- `policy`
- `container`
- `threshold`
- `post_processing`

The tool-inventory digest must come from a canonical, PHI-safe inventory. Do
not place tool arguments, endpoints, credentials, descriptions, clinical
values, or paths in the manifest. The other entries likewise identify bytes or
canonical configuration documents by digest; they do not embed their content.

## Seal a submission

Compute the component digests locally, then seal the complete mapping:

```python
from openmed.eval.workflows import seal_workflow_manifest

component_digests = {
    "container": "sha256:" + "1" * 64,
    "model": "sha256:" + "2" * 64,
    "policy": "sha256:" + "3" * 64,
    "post_processing": "sha256:" + "4" * 64,
    "prompt": "sha256:" + "5" * 64,
    "threshold": "sha256:" + "6" * 64,
    "tokenizer": "sha256:" + "7" * 64,
    "tool_inventory": "sha256:" + "8" * 64,
}
manifest = seal_workflow_manifest(component_digests)
serialized = manifest.to_json()
```

`to_json()` uses sorted, compact canonical JSON. `manifest_digest` seals the
schema version and the complete component mapping. The returned Python object
is frozen, and its component mapping is read-only.

## Verify at evaluation start

Immediately before evaluation, recompute all eight digests from the local
artifacts and configuration, then compare them with the submitted manifest:

```python
from openmed.eval.workflows import verify_at_evaluation_start

verification = verify_at_evaluation_start(manifest, component_digests)
if not verification.eligible_for_sealed_results:
    # Keep the run out of sealed-result publication.
    print(verification.to_dict())
```

Only a complete, canonical, unmodified manifest whose component digests all
match the evaluation-start snapshot is eligible for sealed results. Missing or
malformed data, extra fields, a broken manifest seal, and changed components
all fail closed.

Verification reports contain only closed reason codes and governed component
names. They never echo submitted values. Store or log the report rather than
the supplied manifest when producing evaluation diagnostics.

This mechanism establishes submission immutability for benchmark comparison.
It is not a compliance certification or an autonomous clinical decision
guarantee.
