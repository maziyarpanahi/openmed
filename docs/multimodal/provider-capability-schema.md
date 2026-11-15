# Provider capability matrix schema

Export a self-contained Draft 2020-12 schema for the JSON array emitted by
`render_capability_matrix_json`. This describes content-free compatibility
declarations; it does not discover providers, load models, benchmark execution,
describe endpoints, or choose a provider.

```python
import json
from jsonschema import Draft202012Validator
from openmed.multimodal.capability_matrix import (
    CompatibilityEntry, render_capability_matrix_json,
)
from openmed.multimodal.capability_matrix_schema import (
    build_capability_matrix_schema, render_capability_matrix_schema,
)

schema = build_capability_matrix_schema()
Draft202012Validator.check_schema(schema)
rows = [CompatibilityEntry("local_audio", "1.0.0", "mono", "exact", ())]
payload = json.loads(render_capability_matrix_json(rows))
Draft202012Validator(schema).validate(payload)
assert render_capability_matrix_schema() == render_capability_matrix_schema()
```

`build_capability_matrix_schema()` returns a fresh mutable mapping.
`render_capability_matrix_schema()` returns compact, sorted, byte-stable JSON
with one trailing newline. Mutating a returned mapping cannot affect future
exports. `$id` is `urn:openmed:multimodal:capability-matrix:1`; all `$ref` values
resolve to definitions inside the same document. Exporting or validating this
schema needs no network resolution.

The root is an array of up to 10,000 records; an empty matrix is valid. Each
record requires exactly `provider_id`, `capability_version`, `profile_id`,
`outcome` and `reasons`. Additional fields, including endpoints, model paths
or arbitrary metadata, are refused. IDs use the Python contract's bounded
lowercase code grammar, and capability versions use its bounded numeric triples.
The patterns reject trailing newlines as well as paths, URLs and whitespace.

| Outcome | Allowed reasons |
| --- | --- |
| `exact` | Empty array |
| `transform_required` | `declared_transform_required` |
| `unsupported` | Exactly one of `modality_unsupported` or `feature_unsupported` |
| `missing_capability` | `capability_missing` |

Identifier patterns, record bounds and outcome/reason combinations share constants
with the runtime contract. Drift checks also compare required fields with the
Python record and its actual serialized output. A valid enum used with the wrong
outcome still fails. The exported shape covers rendered matrix records, not the
provider/profile input declarations or a wrapper object.

Standard JSON Schema `uniqueItems` rejects identical records, but cannot enforce
uniqueness of just a provider/profile pair when other fields differ. Use the Python
matrix renderer after constructing `CompatibilityEntry` records to enforce that
semantic rule and canonical ordering. Schema validation does not establish that
the matrix is a complete cross product or that a capability declaration is true.

Use controlled opaque IDs; a grammar check cannot detect sensitive information
encoded into an otherwise valid code. Avoid logging validator exceptions or the
rejected payload: third-party validator diagnostics can include submitted values.
Compatibility declarations remain separate from runtime and clinical qualification.

Run the offline contract checks:

```bash
python -m pytest tests/unit/multimodal/test_capability_matrix_schema.py tests/unit/multimodal/test_capability_matrix.py -q
```
