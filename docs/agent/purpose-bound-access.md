# Purpose-bound data-access tickets

`openmed.agent.permissions.access_tickets` provides an offline, fail-closed
boundary for data access during one local agent run. A ticket binds all of the
following authority together:

- one opaque `RunId`, which makes the ticket non-transferable to another run;
- one developer-authored `PurposeId`;
- permitted data classes for minimum-necessary projection;
- exact keyed record selectors;
- exact tool and action pairs; and
- an exclusive expiry time.

Verification rejects a missing or expired ticket, a different run or purpose,
and any projection, selector, tool, or action outside that complete scope. It
does not use wildcard, prefix, inheritance, or implicit-default rules.

## Create opaque record selectors

Record identifiers can themselves be sensitive. `RecordSelector.from_value()`
uses an application-supplied HMAC-SHA-256 key and retains only the keyed digest.
Use at least 32 bytes of locally managed key material and a stable,
developer-authored selector kind:

```python
from openmed.agent.permissions import RecordSelector

selector = RecordSelector.from_value(
    kind="selector:org.example/record-id@1.0.0",
    value="synthetic-record-001",
    key=b"replace-with-32-or-more-local-key-bytes",
)
```

The example value is synthetic. Do not place raw patient, encounter, account,
or document identifiers in logs, fixtures, governance identifiers, or audit
artifacts. A keyed digest reduces offline guessing risk compared with a plain
hash, but it can still be linkable metadata. Keep tickets and selector digests
under the same access controls as the underlying workflow and rotate keys
according to the application's policy.

## Verify before local dispatch

```python
from openmed.agent import RunId
from openmed.agent.permissions import (
    AccessTicket,
    AccessTicketRequest,
    AccessTicketVerifier,
    ToolAction,
    dispatch_with_access_ticket,
)

run_id = RunId.generate()
purpose = "purpose:org.example/care-summary@1.0.0"
data_class = "data:org.example/medications@1.0.0"
tool_action = ToolAction(
    tool="tool:org.example/summarize@1.0.0",
    action="action:org.example/read@1.0.0",
)

ticket = AccessTicket(
    run_id=run_id,
    purpose=purpose,
    permitted_data_classes=(data_class,),
    record_selectors=(selector,),
    permitted_tool_actions=(tool_action,),
    expires_at=2_000_000_000,
)
request = AccessTicketRequest(
    run_id=run_id,
    purpose=purpose,
    projection=(data_class,),
    record_selectors=(selector,),
    tool_action=tool_action,
)

result = dispatch_with_access_ticket(
    ticket,
    request,
    AccessTicketVerifier(),
    lambda: "local result",
    now=1_999_999_999,
)
```

The callback takes no arguments so record contents and tool arguments remain
outside the authorization layer. It is not invoked unless the full request is
authorized. A requested projection and selector set may be narrower than the
ticket, but every requested item must be present in it. Pass `now` explicitly
for deterministic replay and tests, or inject a local integer clock into
`AccessTicketVerifier` for runtime use.

Tickets and requests are immutable and canonicalize their scope tuples, so
equivalent inputs compare identically. Expiry is exclusive: a ticket is expired
when `now >= expires_at`. Creating and checking a ticket performs no network
request.

## Value-free denials

Every access-ticket failure exposes an `AccessTicketDenialEvidence` object with
only these controlled fields:

- `schema_version`;
- `reason_code`; and
- `field_name`.

The exception message and evidence never copy run identifiers, purposes, data
classes, selector kinds or digests, tool identifiers, actions, record values,
or callback arguments. Applications may record `error.evidence.to_dict()` and
must not log the rejected ticket or request. A denial envelope is evidence that
this local check failed, not a complete audit ledger.

## Relationship to capability grants

A signed capability-grant manifest answers whether the agent may request an
operation class under a policy profile. An access ticket independently answers
why this run may access which records and data classes for that operation. Use
both checks where both boundaries apply; neither one replaces the other.

Access tickets do not establish consent, validate that a stated purpose is
legally sufficient, grant host operating-system permissions, provide replay
protection after expiry, certify compliance, or authorize autonomous clinical
decisions. The trusted local issuer remains responsible for approving the
purpose and constructing the narrow ticket scope.

Run the focused offline tests with:

```text
.venv/bin/python -m pytest tests/unit/agent/permissions/test_access_tickets.py -q
```

## Verify FHIR write references before preview

`FHIRWriteScopeVerifier` uses the same ticket authority to reject wrong-patient
references in a proposed create, update or transaction Bundle before a local
preview callback runs. This Python agent/FHIR boundary does not execute writes
or replace preview, approval, provenance, read-side scope or clinical review.

Configure it with the issuer's patient and encounter selector kinds, the same
local HMAC selector key, an exact FHIR server base and an injected
`ReferenceScopeResolver`. Direct `Patient/id` and `Encounter/id` references,
including same-base absolute URLs and versioned references, map the unversioned
**id** through `RecordSelector.from_value()` and must match the active ticket.
A transaction URN targeting a Patient or Encounter with an id uses that same
mapping. The ticket issuer must establish the patient/encounter relationship;
the verifier does not perform identity discovery.

For other resource types, contained fragments and URN targets without stable
ids, `resolver.resolve(reference, resource=target)` returns a sequence of
candidate scopes. A scope is a tuple of keyed `RecordSelector` objects. The
resolver receives the exact local target for internal references, or `None`
for existing server resources. It must independently establish ownership;
presence in the proposed Bundle, a contained id, or a proposed subject claim is
not permission. One candidate must contain exactly one patient selector and,
when the ticket contains an encounter selector, exactly one encounter selector.
Every returned selector must belong to the active ticket. Zero candidates,
multiple candidates, missing scope evidence and resolver exceptions fail closed.
The injected resolver must remain local-first and never log its transient inputs.

```python
from openmed.agent.permissions import (
    FHIRWriteScopeVerifier,
    preview_with_fhir_write_scope,
)

# resolver, selector_key, ticket, request, and proposed_resource are provided
# by the trusted local application. The payload and resolver are synthetic
# in tests; no model inference or server connection is required.
guard = FHIRWriteScopeVerifier(
    selector_key=selector_key,
    patient_selector_kind="selector:org.example/patient@1.0.0",
    encounter_selector_kind="selector:org.example/encounter@1.0.0",
    server_base="https://synthetic.example/fhir",
    resolver=resolver,
)
preview = preview_with_fhir_write_scope(
    proposed_resource,
    ticket,
    request,
    guard,
    lambda: render_local_preview(proposed_resource),
)
```

The wrapper verifies the full run, purpose, expiry, projection, selector and
write tool/action request before walking every reference, including transaction
entries, contained resources, extension `valueReference` elements and nested
R5 CodeableReferences. In-scope inputs and callback results pass unchanged.
Internal `urn:uuid` targets must be unique transaction entries; fragment targets
must be unique within the referring resource's contained collection. Cross-entry
fragments cannot borrow another entry's contained resource. Only transaction
`POST`/`PUT` entries are supported; other Bundle types and operations fail closed.

`FHIRWriteScopeError.findings` contains only `path` and `code`. Logical
identifier-only references produce `identifier_only_reference`; foreign bases
produce `foreign_base_reference`; unresolved targets produce
`unresolvable_reference`; ambiguous identities/candidates produce
`ambiguous_reference`. Invalid syntax, absent scope evidence and unauthorized
selectors produce `invalid_reference`, `scope_evidence_missing` and
`selector_out_of_scope` respectively. Structural/configuration failures use
`invalid_payload`/`invalid_configuration`. Unknown profile keys become positional
`field[index]` paths so arbitrary input key text cannot enter diagnostics. Do not
log payloads, tickets, resolver inputs, keys, tracebacks with captured locals or
selector values. Resolver exception messages are discarded.

This is a reference-scope guard, not complete FHIR schema validation or
execution authorization. Run schema/type checks separately. The shared
reference-integrity walker visits unknown profile fields; explicit Reference
objects and identifier-only wrappers are checked conservatively, while known
resource-specific reference fields are also checked for malformed values.
Payloads are limited to 64 nesting levels and 100,000 visited values, and cycles
or non-JSON shapes fail closed. The application must preview the verified payload
without mutation and bind that exact payload to its later approval/provenance
checks; verification does not create an approval receipt or authorize execution.

```text
.venv/bin/python -m pytest tests/unit/agent/permissions/test_fhir_write_scope.py tests/unit/interop/test_fhir_reference_integrity.py tests/integration/agent/test_fhir_write_scope_boundary.py -q
```
