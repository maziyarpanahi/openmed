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
