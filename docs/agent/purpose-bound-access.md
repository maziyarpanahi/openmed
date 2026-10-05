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
field names, selector digests or provider exception messages. Existing ticket
and projection errors retain their own value-free diagnostics. Quarantine means
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
