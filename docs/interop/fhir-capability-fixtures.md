# Synthetic FHIR capability fixtures

`tests/fixtures/fhir_capabilities.py` provides small FHIR R4
`CapabilityStatement` builders for offline write-preflight tests. Every call
returns a new dictionary, and the fixed date and resource types are synthetic.
The fixtures omit endpoints, credentials, publisher and organization details,
server software metadata, and clinical resources.

| Builder | Declared capability | Intended preflight outcome |
| --- | --- | --- |
| `build_read_only_capability_statement()` | `read`, `search-type` | A planned write is incompatible |
| `build_create_capability_statement()` | Resource-level `create` | Ordinary create is compatible |
| `build_update_capability_statement()` | Resource-level `update` | Ordinary update is compatible |
| `build_conditional_capability_statement()` | `create` and `update` with both conditional flags | Conditional create and update are compatible |
| `build_transaction_capability_statement()` | System-level `transaction` | Transaction Bundle submission is compatible |
| `build_unsupported_resource_capability_statement()` | Only `SUPPORTED_RESOURCE_TYPE` is declared | A plan for `UNSUPPORTED_RESOURCE_TYPE` is incompatible |

The intended outcomes document fixture semantics; the builders do not perform
preflight checks or execute FHIR operations.

Use `without_capability_field(statement, path)` for missing-field cases and
`with_invalid_capability_field(statement, path, value)` for invalid-field
cases. Paths are sequences of dictionary keys and list indexes:

```python
from tests.fixtures.fhir_capabilities import (
    build_create_capability_statement,
    with_invalid_capability_field,
    without_capability_field,
)

statement = build_create_capability_statement()
missing_version = without_capability_field(statement, ("fhirVersion",))
invalid_interaction = with_invalid_capability_field(
    statement,
    ("rest", 0, "resource", 0, "interaction", 0, "code"),
    "not-an-interaction",
)
```

Both helpers deep-copy the input and preserve it unchanged. They raise instead
of silently creating a field when the requested path does not exist.
