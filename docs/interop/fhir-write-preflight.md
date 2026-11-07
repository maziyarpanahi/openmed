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

## Executing an approved write

`openmed.interop.fhir.write_client.FHIRWriteClient` owns bounded execution after
planning and protected human review. It uses an injected one-shot HTTP
transport, verified local `ApprovalAuthorization`, caller-owned SMART credential
custody, a required durable attempt ledger, and repeated authorization and
field-lineage verification. It adds no HTTP dependency or default connection.

The declared FHIR R4 subset is:

| Input contract | Exact request | Required boundary |
| --- | --- | --- |
| Existing `ConditionalWritePlan`, create | POST `ResourceType`, exact `If-None-Exist` predicate | Match readiness, verified lineage, consumed outer approval |
| Existing `ConditionalWritePlan`, update, and `UpdatePrecondition` | PUT `ResourceType?predicate`, exact `If-Match` | Fresh version evidence, match readiness, verified lineage, consumed outer approval |
| Existing `AssembledTransaction` | POST the configured FHIR base with the exact serialized Bundle | POST creates and version-guarded PUT updates, final Provenance POST, atomic transaction capability |

The conditional planner's idempotency key is preserved as `Idempotency-Key`.
Transactions require an operator-supplied aggregate identity in the same
`fhir-cw-v1-<64 lowercase hex>` format; the adapter does not mint one or change
entry predicates or preconditions. It never rewrites or splits a transaction.
The final Provenance entry counts toward the entry cap but is excluded from
the proposed clinical resource count.

Conditional planning, concurrency, SMART custody and field-lineage contracts
are supplied by [#3439](https://github.com/maziyarpanahi/openmed/pull/3439),
[#3441](https://github.com/maziyarpanahi/openmed/pull/3441),
[#3443](https://github.com/maziyarpanahi/openmed/pull/3443), and
[#3449](https://github.com/maziyarpanahi/openmed/pull/3449); transaction assembly
is supplied by [#4000](https://github.com/maziyarpanahi/openmed/pull/4000).
The adapter consumes these structural interfaces without bundling a second
implementation.
Synthetic vectors cover development independently; compatibility must also
be checked against the actual predecessor classes at pinned commits.

### Trusted operator configuration

Configure one HTTPS FHIR audience, private commitment key, capability metadata
observed for that audience, scope context, and inclusive limits. HTTP is
accepted only for explicitly enabled loopback labs. Audiences are preserved
exactly for custody binding; trailing slashes are refused instead of silently
normalized. Plans cannot choose the endpoint or the sender.

The `custody_factory(sender)` must construct the existing SMART broker with
that bound sender. The trusted operator can retain the broker and store
credentials through its existing API. Agents receive only opaque handles.
Calling the bound sender outside an active approved attempt fails closed.
The transport receives the bearer header only after custody checks.

Required SMART v2 scopes use `.c` for creates and `.u` for guarded updates.
Transactions combine operations per resource type in canonical order, and
require `.c` for the appended Provenance. No read, search, delete or wildcard
scope is added by the adapter.

The `authorize(prepared, receipt)` callback must verify the consumed receipt
against trusted local issuance state, active grant, patient scope, default-off
effect admission, emergency stop, match readiness, and fresh update evidence.
It must return exact `True`. After review of the exact proposal, call the approval
verifier's `consume_authorization()` once and pass its protected local result to `submit()`.
The client checks its consumed action and exclusive validity window before
reservation, custody, and dispatch, including after fresh policy and lineage reads.
The callback receives that result's codes-and-digests receipt and verifies the
remaining authority repeatedly without consuming the token again. A serialized
receipt or caller-supplied role/expiry metadata cannot authorize execution. Keep
the protected authorization in memory and serialize only its public receipt.
No approval is issued by the client.

The `verify_lineage(prepared)` callback must reconstruct the original write
intent from `prepared.payload`, retain the exact original plans and target
bindings, and re-run `require_write_provenance()` with its existing approval
verifier. The returned manifest must match the prepared snapshot. Returning a
previously copied manifest does not prove field coverage or payload agreement.
The adapter checks bounded manifest structure and resource counts; the
existing gate owns exact changed-field coverage and target proof.

`prepare_conditional()` and `prepare_transaction()` return frozen protected
proposals. Their metadata-only `to_dict()` contains keyed commitments and
counts. The outer action binds payload bytes, method, predicate, version
headers, endpoint, handle, required scopes, limits and lineage. Existing
lineage approval does not bind every wire detail, so the human approval token
must also bind this outer action. `prepared.payload` is a detached sensitive
copy for trusted review and verification, never an audit export.

The transport must enforce `timeout_seconds` and `max_response_bytes` while
reading, disable every automatic retry and redirect, and suppress URL, header,
payload and driver-exception logging. The adapter cannot undo a driver that
allocates an oversized result before returning it. All request/response object
representations hide their contents. Diagnostic and receipt exports contain
only closed codes, counts and keyed commitments.

### Outcomes and recovery

| Outcome | Meaning | Next step |
| --- | --- | --- |
| `committed` | Valid server acknowledgement, including an existing conditional-create match | Persist the receipt; apply the separate clinical review policy |
| `rejected` | Authorization/custody refusal or a known server rejection | Correct the cause and obtain any required new review; never reuse a changed action under the old key |
| `conflict` | HTTP 409, 412, 428, or a key bound to a different completed action | Fresh read and caller-owned conflict policy |
| `unknown_commit` | Timeout after entering transport, redirect, server uncertainty, malformed/oversized acknowledgement, pending claim, or lost durable receipt acknowledgement | Reconcile under a separate authorized read; no blind retry or automatic compensation |

Resource acknowledgements require the expected resource type, a bounded ID
and version, or an exact Location/ETag pair for a minimal response. If the
proposal supplied an ID, the acknowledgement must match it. Transaction
responses must acknowledge every entry in order with the expected resource
types, locations and versions. Mixed success/failure transaction responses
remain unknown; they are not interpreted as partial success.

The required `FHIRWriteLedger` must atomically reserve a key across processes
and durably retain it across restart, even when no outcome was recorded.
`finish()` must acknowledge durable storage with exact `True`. A duplicate
key returns its matching recorded result; a pending key cannot dispatch again.
`reconcile()` reads this ledger only. Missing evidence stays unknown, and
server-side reconciliation remains owned by the existing recovery workflow.
The adapter assumes neither server support for its idempotency header nor
that a conditional predicate makes replay safe.

This is a transport boundary, not clinical validation or permission to make
autonomous clinical changes. Planning, OAuth/refresh, conflict merging,
compensation and server-side recovery policy remain separate boundaries.
