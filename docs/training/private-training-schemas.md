# Private-training JSON schemas

External coordinators and air-gapped clients validate private-training
metadata against strict, versioned JSON Schemas. OpenMed exports one
self-contained catalog of Draft 2020-12 schemas covering the federated
training metadata surfaces: round lifecycle, round scheduling, round status,
dense update metadata, and aggregate metric envelopes.

The catalog is import-light and deterministic. Each schema:

- uses only fragment-local references (no `$remote` resolution);
- pins `additionalProperties: false` and exact schema-version constants;
- binds enum values, field sets, and bounds to the Python sources of truth;
- renders byte-identically across runs, so it can be embedded in offline
  artifacts or shipped to peers without a hosted registry.

The schemas describe the public JSON projections only. They never read model
contents, tensors, gradients, client identifiers, or local metrics.

## Catalog

`openmed.training.private_training_schemas` exposes:

```python
from openmed.training.private_training_schemas import (
    build_private_training_schemas,
    build_schema,
    render_private_training_schemas,
    render_schema,
)

catalog = build_private_training_schemas()
# keys: federated_round_lifecycle, federated_round_schedule,
#       federated_round_status, federated_update_metadata,
#       federated_aggregate_metric

lifecycle = build_schema("federated_round_lifecycle")
stable = render_private_training_schemas()  # byte-stable compact JSON
```

Each `build_*` function returns a fresh mapping, so mutating one export cannot
affect a later build. `build_schema(name)` and `render_schema(name)` raise
`PrivateTrainingSchemaError` for unknown names.

## Use with a validator

```python
from jsonschema import Draft202012Validator, validate
from referencing import Registry
from openmed.training.private_training_schemas import build_schema

schema = build_schema("federated_update_metadata")
validate(payload, schema, cls=Draft202012Validator, registry=Registry())
```

Because every reference is fragment-local, the validator resolves fully
offline; a `Registry` that rejects remote fetches is sufficient.
