# Federated update schema fingerprints

Coordinators can compare the declared schemas of anonymous dense adapter
updates without opening tensor values. `fingerprint_update_schema()` accepts an
already validated `FederatedUpdateMetadata` instance and returns a lowercase
`sha256:` reference. `same_update_schema()` fingerprints two validated metadata
instances and compares the resulting references with `hmac.compare_digest`.

```python
from openmed.training.federated_schema_fingerprint import (
    fingerprint_update_schema,
    same_update_schema,
)

# first and second were parsed with independently supplied coordinator policies.
reference = fingerprint_update_schema(first)
compatible = same_update_schema(first, second)
```

The fingerprint binds the update metadata schema version, model digest,
adapter format, and each parameter's name, shape, and dtype. Validated
parameter records have canonical name order. The function serializes these
fields as sorted-key compact JSON and hashes them with the domain prefix
`openmed.training.federated_schema_fingerprint.v1` followed by a NUL byte.
This prefix separates the digest from unrelated uses of SHA-256. A golden
vector in the focused tests pins the exact output bytes.

An individual update's content digest, clipping declaration, and derived
element count are excluded from schema identity. The coordinator must still
validate each update against an independently constructed policy; a matching
schema fingerprint says nothing about tensor contents, clipping truthfulness,
privacy, or safe aggregation. The current update contract accepts only the
`dense` adapter format, but the format field is included so a future validated
format cannot collide with it.

No client or site identifier, file path, endpoint, example, gradient, or tensor
value enters the canonical bytes. Invalid or unvalidated mappings are not
accepted by the fingerprint API.

Run the offline checks with:

```sh
uv run --frozen --extra dev python -m pytest tests/unit/training/test_federated_schema_fingerprint.py -q
```
