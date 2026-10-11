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

## Authorize tool results before downstream use

Request authorization does not authorize whatever a tool returns. The Python
`openmed.agent.permissions.result_scope` boundary uses the existing ticket,
`RecordSelector`, `ArtifactReference` and minimum-data projection contracts.
It owns result authorization; it does not resolve identities, scan arguments,
judge clinical correctness or grant permission to perform clinical actions.
This slice targets the existing Python agent ticket API; it adds no Swift or
Apple Foundation Models execution path.

Construct a trusted expected `ResultScope` with four separate selector kinds:
patient, encounter, namespace and evidence snapshot. Include **exactly those
four selectors** in the read's `AccessTicketRequest`. The issuer may grant a
broader ticket, but results must match this narrower request. Use stable
application-owned selector kinds and the same local keyed mapping for expected
and observed identities. The snapshot selector must bind its exact version or
digest; a namespace selector must distinguish tenants/sources even when their
local patient identifiers coincide. Missing dimensions and repeated kinds fail
closed. A workflow without encounter or snapshot evidence cannot use this
boundary; do not invent placeholder identities.

A trusted local adapter independently derives `ResultScope` from authoritative
resource metadata on **each** `ToolResultPage`, `ToolResultRecord` and nested
record in `children`. Do not trust scope asserted by a model, copy request scope
onto returned records, inherit parent/page scope or hide independent resources
inside ordinary field values. Nested structured values in `fields` are governed
by the reviewed schema; independent resource records belong in `children` and
each requires its own scope and evidence identity. This module compares supplied
evidence, rather than proving that an adapter's metadata is authentic.

The per-record output schema uses the same annotations as
[minimum-data projections](minimum-data-projections.md). The gate derives its
projection using the request's purpose and data classes. Declared containers
do not authorize arbitrary descendants: objects are closed, nested properties
must be declared and arrays follow their `items` schema. Extra fields quarantine
the **whole batch**, rather than silently dropping unauthorized content. Missing
fields, incompatible shapes, opaque object values and non-finite numbers also
fail closed. This is an authorization projection, not a complete JSON Schema or
clinical validator.

```python
from openmed.agent.permissions import dispatch_with_authorized_results

# expected_scope, output_schema and local_read are reviewed local bindings.
# request.record_selectors exactly covers expected_scope.selectors().
result = dispatch_with_authorized_results(
    ticket,
    request,
    AccessTicketVerifier(),
    scope=expected_scope,
    schema=output_schema,
    read=local_read,
    consume=screen_then_run_next_step,
)
```

The local `read` returns a tuple of all collected pages, in contiguous zero-based
order, with `final=True` only on the last page. The adapter must establish source
pagination completeness; the boundary cannot discover omitted pages or fetch
continuation URLs. Lazy batches, partial batches and missing scope on even an
empty page fail closed. No first-page streaming occurs. Limits are 64 pages,
10,000 traversed record/value nodes and depth 32. The reviewed schema is copied
before invoking the adapter; later caller-side schema mutation cannot expand it.
Ticket validity is checked before reading, after reading and immediately before
release using the verifier's clock. Explicit `now` is for deterministic replay.

Accepted results receive fresh nested field dictionaries/lists and retain their
original scope and `ArtifactReference` object, including its opaque identity and
digest. The reference identifies **original evidence**, not the newly projected
field encoding. Every record needs an evidence reference; duplicate artifact IDs
in one batch fail closed rather than guessing record custody. No new evidence
digest or model output is fabricated.

On rejection, `consume` is never invoked and no partial result is returned.
Record only `ResultQuarantinedError.to_dict()` in an action trace. It contains a
fixed schema version and controlled reason code, with no payloads, unexpected
field names, selector digests or provider exception messages. Read failures also
discard the original exception context before raising the controlled quarantine
error. Existing ticket and projection errors retain their own value-free
diagnostics. Quarantine means
withholding output: this library does not log, cache, save or otherwise persist
rejected content. Applications must keep protected inputs outside audit traces
and must not capture exception traceback locals.

Authorization and privacy screening are independent. Correct patient scope can
still contain PHI or hostile instructions; `consume` must apply the application's
separate privacy preflight/injection guards before model context or subsequent
actions. Conversely, a PHI-free result for the wrong patient is still rejected.
No network call, identity resolver, mandatory dependency or cloud fallback is
added.

```text
.venv/bin/python -m pytest tests/unit/agent/permissions/test_result_scope.py tests/integration/agent/test_result_scope_boundary.py -q
```

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

## Record access decisions

Every call to `AccessTicketVerifier.verify()` creates exactly one `AccessEvent`
for the authorization decision, including missing, expired, mismatched and
out-of-scope tickets. The default `MemoryAccessEventSink` retains events on
`verifier.sink.events` for the lifetime of that verifier. Hosts should inject a
sink with appropriate retention for long-running workflows. No network call or
filesystem write is required by the default verifier.

```python
from openmed.agent.permissions import AccessTicketVerifier, LocalAccessEventSink

# The host chooses a protected local directory; this is a synthetic path.
sink = LocalAccessEventSink("synthetic-access-events.jsonl")
verifier = AccessTicketVerifier(sink=sink, clock=lambda: 1_999_999_999)
verifier.verify(ticket, request)
head = sink.verify_chain()
counts = sink.export_counts()
assert counts["outcomes"] == {"allow": 1}
```

An injected `AccessEventSink` implements `emit(event)`. Verification calls it
once, before returning authority or raising the original denial. A sink failure
raises a value-free `AccessEventError("sink_unavailable")`, blocks dispatch and
is never automatically retried. Hosts must avoid retrying a verification to
repair an uncertain sink write: a custom sink may have persisted before raising.
No transaction with the tool's data store is implied.

The closed event schema contains only an opaque requesting run ID, a
**developer-authored** purpose, requested data-class codes, requested selector
count, developer-authored tool/action codes, `allow` or `deny`, the denial reason
code and verification epoch seconds. Keep these governance codes free of patient
identifiers and values. Events contain no ticket, selector kind, selector digest,
record identifier, tool arguments, result, credential, or source payload.
The event reflects the requested narrow scope, including rejected requests,
rather than copying the ticket's wider authority. Malformed request objects
produce `invalid_request` with absent scope metadata. Unavailable or invalid
clock values produce a denial with `timestamp=None`; invalid requests also have
no timestamp. Construction errors before verification and invalid dispatch
callbacks are not verification decisions and do not produce access events.

Events describe authorization, not proof that a tool returned data. A verified
read that subsequently fails still has one `allow` event. Repeated verification
calls each produce a distinct event. This Python verifier extension does not
change the action ledger, Journey audit format, result authorization, or Swift
contracts. It adds no cloud fallback or clinical action.

### Local chain and aggregate export

`LocalAccessEventSink` writes canonical JSON lines with an event, sequence,
previous digest and SHA-256 digest. It verifies the entire existing chain before
append, flushes and fsyncs before returning, and uses a POSIX advisory file lock
for cooperating writers, including separate sink instances. New files use mode
0600; symbolic links and non-regular files are refused. POSIX is required only
for this durable sink. Hosts own directory protection, existing file permissions,
retention and backups, and must prevent file replacement or rotation while
writers are active. Diagnostics omit private log paths and parser payloads.

`verify_chain()` returns the chain head or raises a controlled error for
corruption, modification, interior deletion, reordering, duplicate entries or
partial writes. To detect suffix truncation, compare with an independently saved
head using `verify_chain(expected_head=head)`. Deeply nested malformed JSON also
fails with a controlled code without retaining the parser exception.
A plain digest chain cannot detect
an attacker rewriting the whole log and recomputing all digests without that
independent anchor. No signatures or compliance guarantee are supplied.

`export_counts()` first verifies the chain, then returns event and selector
counts, outcome counts, denial-reason counts, purpose counts, data-class counts
and tool/action counts. It omits run IDs and per-record metadata. Data-class
counts count decisions containing the class; selector counts sum requested
selectors, including denied requests. These are not counts of returned records.

Focused offline validation:

```text
.venv/bin/python -m pytest tests/unit/agent/permissions/test_access_tickets.py tests/unit/agent/permissions/test_access_events.py tests/integration/agent/test_access_event_log.py -q
```
