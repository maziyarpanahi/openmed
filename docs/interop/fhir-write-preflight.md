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

## Optional server profile validation

After capability, permission and minimum-data checks, an application may call
`openmed.interop.fhir.preflight_server_validation` with its protected proposed
resource. **The default is `enabled=False`: no input or transport is inspected
and no request is made.** Enabling it sends the resource only through the
injected transport already bound to the intended write server. This is a
separate disclosure requiring the application's verified read/use authority;
human write review follows validation.

The transport implements `FHIRValidationTransport.validate`. It receives FHIR
`Parameters` containing the intended `create` or `update` mode and a fresh JSON
copy of the proposed resource. Update validation requires an explicit, exact
`resource_id` and uses the instance operation; a resource's supplied ID must
match that target. Create validation uses the type operation and omits an
instance ID. Delete and transaction validation are outside this helper.

An optional `profile_uri` nominates an application-approved canonical profile
in the `profile` parameter. It is protected request metadata, never the HTTP
destination. Select profiles from trusted local policy; the application and
target server govern profile resolution. Neither this module nor its cached
capability parser fetches a profile, an OperationDefinition or an endpoint.

Support requires a same-REST-block server declaration for the resource and
`validate` operation, at resource level or as a shared REST operation. The
operation definition must identify the standard R4
`Resource-validate` canonical, optionally qualified by a supported R4 version.
A missing declaration, client-only declaration, unrelated resource or conflicting
operation definition returns `unsupported` before resolving the transport.
An arbitrary operation named `validate` does not establish standard semantics.

This runnable synthetic fixture uses a fake transport and an opaque fake
custody handle. It deliberately returns an HTTP 200 validation error: no
preview or live write is produced, and only controlled outcomes are printed.

```python
# Runnable: synthetic injected transport only; no server contact.
from openmed.interop.fhir import FHIRValidationResponse, preflight_server_validation

statement = {
    "resourceType": "CapabilityStatement",
    "fhirVersion": "4.0.1",
    "rest": [
        {
            "mode": "server",
            "resource": [
                {
                    "type": "Patient",
                    "interaction": [{"code": "create"}],
                    "operation": [
                        {
                            "name": "validate",
                            "definition": "http://hl7.org/fhir/OperationDefinition/Resource-validate",
                        }
                    ],
                }
            ],
        }
    ],
}


class SyntheticCustodyHandle:
    pass


class SyntheticValidationTransport:
    def validate(self, resource_type, **protected_request):
        return FHIRValidationResponse(
            200,
            {
                "resourceType": "OperationOutcome",
                "issue": [
                    {
                        "severity": "error",
                        "code": "required",
                        "diagnostics": "synthetic diagnostic discarded",
                        "expression": ["Patient.name[0].family"],
                    }
                ],
            },
        )


result = preflight_server_validation(
    statement,
    {"resourceType": "Patient", "active": True},
    mode="create",
    enabled=True,
    transport=SyntheticValidationTransport(),
    credential_handle=SyntheticCustodyHandle(),
)
assert not result.is_valid
print({"status": result.status.value, "reason_code": result.reason_code.value})
```

Only `passed` means this enabled validation check passed. Disabled and
unsupported checks supply no validation evidence; the application decides
whether its deployment policy requires this optional check. `blocked` stops
preview/review in the composed flow. Fatal/error issues block even with HTTP
200; warnings are summarized and pass by default, or block when
`block_warnings=True`. A non-200 response means validation was unavailable,
rather than proof that the resource passed or failed its profile rules.

Results contain controlled severity and R4 issue-type codes, counts, a request
digest, and a conservative structural path vocabulary. Indices become `[]`.
Functions, filters, comparisons, quoted literals, unknown/custom names,
identifiers, XPath `location` values and paths for another resource type are
discarded. The vocabulary is a privacy boundary rather than a complete schema;
it can omit legitimate expressions without changing their issue severity.
Diagnostics, details, narrative, resource content, IDs, profile URIs,
credentials, headers, server addresses and raw exceptions are never copied into
results. Keep the protected `FHIRValidationResponse` and request out of logs;
retain only the sanitized result as evidence.

The SHA-256 request digest binds the resource, intended mode, optional profile
and update instance ID. It does **not** bind server identity, permission,
policy or approval. Bind the result to that same target and proposal in trusted
application code; changing either requires fresh validation. Passing is neither
clinical assurance nor permission to write, and concurrent server changes can
still cause the later approved write to fail.

The helper bounds proposed JSON to 1 MiB, 32 levels and 65,536 values; operation
metadata to 256 declarations; and returned outcomes to 128 issues, each with
at most 16 expressions of 256 characters. Malformed or excessive input blocks
rather than truncating late errors. It sends once and never retries. The
application transport must enforce the supplied positive deadline (at most
60 seconds), response-body limits, credential custody and target/TLS policy,
and must disable redirects, secret/payload logging and automatic retries.
The protocol cannot interrupt a transport that ignores those requirements.
No transport implementation or execution adapter is installed automatically.

The protocol follows the [FHIR R4 Resource `$validate` operation](https://hl7.org/fhir/R4/resource-operation-validate.html),
[CapabilityStatement operation declarations](https://hl7.org/fhir/R4/capabilitystatement-definitions.html#CapabilityStatement.rest.resource.operation)
and [OperationOutcome issue semantics](https://hl7.org/fhir/R4/operationoutcome.html).
Fake-transport checks provide engineering evidence; they do not establish
reference-server conformance or clinical validation.
