# FHIR Bulk Data checkpoints

OpenMed provides a small local-only manifest for resuming a FHIR Bulk Data
export. It records the resource type, digests for the opaque page token,
privacy policy, and endpoint scope, plus aggregate page and resource counts.
It never stores a page token, resource payload, or policy configuration in the
manifest.

## Create and validate a checkpoint

```python
from openmed.interop.fhir.bulk_checkpoint import (
    create_checkpoint,
    validate_resume,
)

checkpoint = create_checkpoint(
    "Patient",
    page_token="synthetic-next-page-token",
    policy={"name": "local-safe-policy", "date_shift_days": 7},
    endpoint_scope={
        "base": "https://synthetic.example/fhir",
        "export": "group-synthetic",
    },
    pages_processed=3,
    resources_processed=42,
)
checkpoint.write("bulk-checkpoint.json")

# Raises BulkCheckpointCompatibilityError if any resume identity changed.
validate_resume(
    checkpoint,
    resource_type="Patient",
    page_token="synthetic-next-page-token",
    policy={"name": "local-safe-policy", "date_shift_days": 7},
    endpoint_scope={
        "base": "https://synthetic.example/fhir",
        "export": "group-synthetic",
    },
)
```

Policy and endpoint values may be mappings or other JSON-compatible values;
mapping keys are sorted before hashing, so equivalent configurations produce
the same fingerprint. `None` is supported for the final page marker. Resume
validation compares the manifest version, resource type, page-token digest,
policy fingerprint, and endpoint-scope digest. A mismatch fails closed and
the exception names only the mismatched field, never its value.

The manifest is deterministic JSON and performs no network call. Filesystem
I/O happens only when `write`, `write_checkpoint`, or `load_checkpoint` is
explicitly called. The checkpoint is progress metadata, not a clinical
decision or a substitute for a FHIR server's export guarantees.
