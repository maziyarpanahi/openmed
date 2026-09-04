# FHIR R5 round-trip fidelity

`openmed.interop.fhir_r5_fidelity` compares a local FHIR R5 `Bundle` before
and after a parser/exporter round trip. It is a deterministic exchange check,
not a FHIR profile validator, terminology check, compliance certification, or
clinical decision guarantee.

The comparison is offline and dependency-free. JSON object member order and
whitespace are ignored. Bundle entries are matched by `fullUrl`, then resource
`id`, and finally by remaining position, so a serializer that reorders entries
does not create a false difference.

```python
from openmed.interop.fhir_r5_fidelity import diff_fhir_r5_bundles

result = diff_fhir_r5_bundles(exported_bundle, reparsed_bundle)
if not result.is_faithful:
    print(result.to_markdown())
```

Reports contain structural paths, JSON types, resource types, and SHA-256
digests of values and identifiers. They do not include raw IDs, references,
code values, narrative, or other source values, so `to_dict()`, `to_json()`,
and `to_markdown()` are suitable for local review artifacts.

Known serializer-specific differences must be declared explicitly. Paths use
dot notation with array indexes and support `[*]` wildcards:

```python
result = diff_fhir_r5_bundles(
    before,
    after,
    allowed_paths=["entry[*].resource.meta.lastUpdated"],
    unordered_paths=["entry[*].resource.meta.tag"],
)
```

Only the named subtree is ignored; coded fields, resource references, and
other fields remain part of the fidelity check. An empty `result.changes`
means the two bundles are equivalent under the declarations supplied by the
caller.
