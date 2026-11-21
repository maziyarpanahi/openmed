# Capability descriptor

The capability descriptor is one value-free document that says what an OpenMed
runtime supports. A client reads it before calling an operation instead of
calling the operation and parsing the failure. Every surface builds it with the
same function, `openmed.core.capabilities.capability_descriptor()`, and seals
it with the same digest, so two answers with equal digests describe the same
capabilities.

The bundled JSON Schema is
`openmed/core/schemas/json/capability_descriptor.schema.json`
(`schema: "openmed.capabilities"`, `schema_version: 1`).

## Fields

| Field | Content |
|---|---|
| `schema`, `schema_version` | `"openmed.capabilities"` and `1`. |
| `backends` | Every registered optional backend, sorted by `name`, with its `extra` and whether all of its modules are importable (`available`). |
| `surfaces` | `cli`, `graphql`, `grpc`, `mcp`, `python` and `rest`, sorted, with whether the packages each surface needs are installed. |
| `served_models` | Sorted public model ids the reporting runtime serves. The plain Python API reports none unless the caller passes them. |
| `schema_versions` | The `schema_version` of every canonical record schema (`span`, `audit`, `entity`, ...), sorted by `name`. |
| `policy_profiles` | Sorted canonical names of the bundled de-identification policy profiles. |
| `digest` | `sha256:` over the canonical JSON form of every other field. |

Backend and surface availability is probed with `importlib.util.find_spec`, so
building the descriptor imports no optional package and loads no model.

## Privacy

The descriptor holds only controlled names, booleans, integers, public model
ids and a digest. It never contains hostnames, filesystem paths, cache
locations, environment values, credentials or request content. A served model
entry that is not a public model id, such as a local checkpoint path, is left
out rather than echoed.

## Digest

The digest is `sha256:` followed by the hex SHA-256 of the descriptor without
its `digest` field, serialized with sorted keys, `","` and `":"` separators and
ASCII escapes. Recompute it to check a descriptor received from another
surface:

```python
from openmed.core.capabilities import (
    capability_descriptor,
    capability_descriptor_digest,
    validate_capability_descriptor,
)

descriptor = capability_descriptor()
validate_capability_descriptor(descriptor)  # schema and digest
assert capability_descriptor_digest(descriptor) == descriptor["digest"]
```

Pass the served model ids to describe a serving runtime:

```python
descriptor = capability_descriptor(
    served_models=["OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"],
)
```

## REST

`GET /v1/capabilities` returns the descriptor of the running server, with the
server's served model ids, as `application/json`. The OpenAPI document types
the response with the descriptor schema. When authentication is enabled the
route requires `models:read`, so it answers the same callers as
`GET /models/loaded`.

```bash
curl -sS --max-time 10 \
  -H "X-API-Key: $OPENMED_API_KEY" \
  "$OPENMED_URL/v1/capabilities"
```

The typed clients expose it as `OpenMedClient.capabilities()` in Python and
TypeScript, and `Client.Capabilities` in Go.

The REST descriptor also includes a sorted `capabilities` array for its mounted
clinical operations. Each entry has a capability `code`, the bundled
`intended_use_digest`, the `regulatory_tier`, its required `obligations` and
the declaration's `excluded_uses`. For example, the decisions route advertises
the clinician-mode obligation. The descriptor digest covers these entries.
Authorization and model readiness are evaluated when the operation is called.
