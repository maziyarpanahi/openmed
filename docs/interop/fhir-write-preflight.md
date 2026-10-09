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

## Executing an approved write

`openmed.interop.fhir.write_client.FHIRWriteClient` owns bounded execution after
planning and protected human review. It uses an injected one-shot HTTP
transport, the existing `ApprovalReceipt`, caller-owned SMART credential
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
These predecessor PRs are open at development time. The adapter consumes their
structural interfaces without importing or bundling their implementations.
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
It must return exact `True`. Approval consumption happens once before submit;
the callback verifies that consumed receipt repeatedly without consuming the
token again. No approval is issued by the client.

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
