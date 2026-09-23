# FHIR write capability preflight

`openmed.interop.fhir_capability_preflight` checks a content-free write plan
against an already-cached FHIR R4 `CapabilityStatement`. The check is local and
dependency-free: it never discovers a server, reads credentials, accepts a
clinical resource payload, or executes a write.

Run this check before code obtains credentials or materializes patient data:

```python
from openmed.interop.fhir_capability_preflight import (
    FHIRWriteInteraction,
    FHIRWritePlan,
    preflight_write_plan,
)

plan = FHIRWritePlan(
    interaction=FHIRWriteInteraction.CREATE,
    resource_type="Observation",
)
result = preflight_write_plan(cached_capability_statement, plan)

if not result.is_compatible:
    # Stop before credentials or resource payloads are touched.
    send_for_review(result.to_dict())
```

The write plan contains only an interaction, an optional resource type, and a
conditional-write flag. Do not attach resources, patient identifiers,
credentials, or endpoints to it.

## Decisions and reason codes

Only `compatible` confirms that the cached statement declares the requested
capability. `review` and `incompatible` must not automatically proceed to a
write.

| Status | Reason code | Meaning |
| --- | --- | --- |
| `compatible` | `supported` | The resource/system interaction and any required conditional flag are declared. |
| `review` | `capability_statement_malformed` | Required capability metadata is missing, invalid, or above a parser bound. |
| `review` | `conditional_create_undeclared` | Create is declared but `conditionalCreate` is absent. |
| `review` | `conditional_update_undeclared` | Update is declared but `conditionalUpdate` is absent. |
| `incompatible` | `fhir_version_not_supported` | The statement is not for supported FHIR R4 version metadata. |
| `incompatible` | `resource_not_supported` | The planned resource type is not declared. |
| `incompatible` | `interaction_not_supported` | The resource exists but does not declare the planned create or update interaction. |
| `incompatible` | `conditional_create_not_supported` | Conditional create is explicitly false. |
| `incompatible` | `conditional_update_not_supported` | Conditional update is explicitly false. |
| `incompatible` | `transaction_not_supported` | No system-level transaction interaction is declared. |

Transaction plans are system-level and omit `resource_type`. Create and update
plans require a valid FHIR resource type. A conditional plan also requires the
ordinary create or update interaction; a conditional flag alone is not enough.

## Bounded parsing

`parse_capability_statement()` reads only `resourceType`, `fhirVersion`, and
the write-related portions of `rest`. It enforces fixed limits on REST blocks,
resource declarations, and interactions, ignores client-mode capability
blocks, and returns immutable normalized metadata. Unknown top-level content
is neither copied into the result nor reflected in preflight output.

The parser raises `CapabilityStatementError` for callers that need strict
validation. `preflight_write_plan()` converts malformed capability metadata to
the safe `review` result so a malformed cache entry can never grant write
compatibility.

The synthetic builders in
[Synthetic FHIR capability fixtures](fhir-capability-fixtures.md) cover the
supported, unsupported, missing-field, and malformed cases without any live
FHIR service.
