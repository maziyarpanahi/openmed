# FHIR R5 round-trip fidelity

`openmed.interop.fhir_r5_fidelity` compares a local FHIR R5 `Bundle` before
and after a parser/exporter round trip. It is a deterministic exchange check,
not a FHIR profile validator, terminology check, compliance certification, or
clinical decision guarantee.

The comparison is offline and dependency-free. JSON object member order and
whitespace are ignored. Bundle entries are matched by `fullUrl`, then the
resource `resourceType` and `id` pair, and finally by remaining position.
Duplicate stable keys are paired deterministically, so a serializer that
reorders entries does not create a false difference. Valid request and response
entries without a `resource` are supported and compared as complete entries.

```python
from openmed.interop.fhir_r5_fidelity import diff_fhir_r5_bundles

result = diff_fhir_r5_bundles(exported_bundle, reparsed_bundle)
if not result.is_faithful:
    print(result.to_markdown())
```

Reports contain structural paths, JSON types, resource types, and SHA-256
digests of values and identifiers. They do not include raw IDs, references,
code values, narrative, or other source values. Digests can still be
correlatable metadata, so protect and retain `to_dict()`, `to_json()`, and
`to_markdown()` outputs according to the source data policy.

Known serializer-specific differences must be declared explicitly. Paths use
dot notation with array indexes and support `[*]` for one segment or `**` for
recursive matching:

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

JSON text and file inputs are limited to 16 MiB. Object keys must use FHIR JSON
element-name syntax, resource metadata is validated before matching, and
excessive nesting, node counts, cycles, and non-finite numbers are rejected.
Failures use value-free errors and do not include rejected input or path text.
