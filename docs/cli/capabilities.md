# Capabilities CLI

`openmed capabilities` prints the value-free
[capability descriptor](../api/capabilities.md) of the local install: which
optional backends and surfaces are installed, the canonical schema versions,
the bundled policy profile names, and the descriptor digest. Scripts and CI jobs
can check whether a backend or schema version is available without parsing
`openmed doctor` output or importing Python.

```bash
openmed capabilities
openmed capabilities --json
```

The command only reports. It always exits `0`, including when an optional
extra is missing; a missing backend is listed as `missing`. It loads no model,
makes no network call and imports no optional package. Its output never
contains hostnames, filesystem paths, cache locations, tokens or other
environment values.

## Human output

Sections appear in a fixed order, and each section is sorted by name, so the
output is the same for the same install:

```text
OpenMed capabilities (openmed.capabilities v1)
Digest: sha256:<64 hex characters>

Surfaces:
  available  cli
  missing    graphql
  ...

Backends:
  missing    coreml      openmed[coreml]
  ...

Schema versions:
  action                   1
  ...

Policy profiles:
  africa_malabo_baseline
  ...

Served models: none
```

## JSON output

`--json` wraps the descriptor in the standard
[result envelope](result-envelope.md):

```json
{
  "ok": true,
  "command": "capabilities",
  "data": {
    "schema": "openmed.capabilities",
    "schema_version": 1,
    "backends": [{"name": "coreml", "extra": "coreml", "available": false}],
    "surfaces": [{"name": "cli", "available": true}],
    "served_models": [],
    "schema_versions": [{"name": "action", "version": 1}],
    "policy_profiles": ["africa_malabo_baseline"],
    "digest": "sha256:..."
  }
}
```

The example is shortened. `data` validates against the bundled descriptor
schema, and `data.digest` equals the digest that
`openmed.core.capabilities.capability_descriptor()` returns in the same
environment. The CLI serves no models, so `served_models` is empty. A service
reports its served model ids from `GET /v1/capabilities`.

```bash
openmed capabilities --json | jq -r '.data.digest'
openmed capabilities --json \
  | jq -e '.data.backends[] | select(.name == "onnx") | .available'
```
