# OMOP rollback manifest JSON Schema

OpenMed exports a strict Draft 2020-12 JSON Schema for the content-free
rollback manifests described in [OMOP rollback manifests](omop-rollback-manifests.md).
The export is deterministic, requires no network access, and uses only local
`#/$defs/...` references.

## Export and validate locally

Use the mapping form with any Draft 2020-12-compatible validator:

```python
from jsonschema import Draft202012Validator

from openmed.interop.omop_rollback_schema import (
    export_omop_rollback_manifest_schema,
)

schema = export_omop_rollback_manifest_schema()
Draft202012Validator.check_schema(schema)
Draft202012Validator(schema).validate(manifest.to_dict())
```

Use `export_omop_rollback_manifest_schema_json()` when a registry, file, or
transport needs canonical JSON. Repeated calls return byte-identical UTF-8
content for the same OpenMed version.

## Strict interoperability boundary

The schema closes every object and binds:

- the manifest schema discriminator;
- standard OpenMed OMOP table names;
- `insert`, `update`, and `tombstone` operations to their required rollback
  strategies;
- positive operation and mutation counts to the 10,000-mutation batch limit;
- mutation and rollback ordinals to the corresponding zero-based range; and
- batch, manifest, mutation, rollback-artifact, and vocabulary-snapshot
  references to lowercase `sha256:` digests.

Unknown properties are rejected. Row data, row keys, SQL, connection details,
and before-images therefore have no representable field in the schema. Custom
mutation tables supported by a local adapter are intentionally outside this
interoperable standard-table contract.

JSON Schema establishes the portable structural boundary. Before approval,
also call `manifest.validate(batch, vocabulary_snapshot)` to verify semantic
coverage, reverse ordering, count totals, digest integrity, and binding to the
exact staged batch and local vocabulary snapshot. Neither validation path
executes a rollback, accesses a database, or proves semantic reversibility.
