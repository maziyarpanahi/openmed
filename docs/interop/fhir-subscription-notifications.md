# Value-free FHIR Subscription intake

`parse_subscription_notification()` consumes an in-memory notification Bundle
and returns numbers, counts, closed classifications and SHA-256 digests. It
performs no network call, checkpoint write, resource export or workflow dispatch.
It does not return resource content, URLs, identifiers, timestamps or server
error text.

The caller supplies the negotiated notification release explicitly:

| `version` | Bundle type | Leading resource |
| --- | --- | --- |
| `R4` or `4.0.1` | `history` | Backport `Parameters` |
| `R4B` or `4.3.0` | `history` | `SubscriptionStatus` |
| `R5` or `5.0.0` | `subscription-notification` | `SubscriptionStatus` |

The supported formats follow the [Subscriptions Backport 1.1 R4 status
profile](https://hl7.org/fhir/uv/subscriptions-backport/STU1.1/StructureDefinition-backport-subscription-status-r4.html),
[FHIR R4B SubscriptionStatus](https://hl7.org/fhir/R4B/subscriptionstatus.html)
and [FHIR R5 SubscriptionStatus](https://hl7.org/fhir/R5/subscriptionstatus.html).
The [R4B notification Bundle profile](https://hl7.org/fhir/uv/subscriptions-backport/STU1.1/StructureDefinition-backport-subscription-notification.html)
requires the `history` type and a leading `SubscriptionStatus`.
R4B support here is limited to notification intake; the general OpenMed
R4/R5 exchange converter does not gain R4B conversion support.

## Offline synthetic example

```python
from openmed.interop.fhir.subscription_notifications import (
    SubscriptionContentMode,
    parse_subscription_notification,
)

bundle = {
    "resourceType": "Bundle",
    "type": "subscription-notification",
    "entry": [{"resource": {
        "resourceType": "SubscriptionStatus",
        "status": "active",
        "type": "event-notification",
        "subscription": {"reference": "Subscription/synthetic-subscription"},
        "eventsSinceSubscriptionStart": "1",
        "notificationEvent": [{
            "eventNumber": "1",
            "focus": {"reference": "Observation/synthetic-observation"},
        }],
    }}],
}
report = parse_subscription_notification(
    bundle, version="R5", previous_event_number=0,
)
assert not report.requires_reconciliation
assert len(report.workflow_events) == 1
event = report.events[0]
assert event.content_mode is SubscriptionContentMode.ID_ONLY
assert event.requires_authorized_read
assert "synthetic-observation" not in report.to_json()
assert "synthetic-subscription" not in report.to_json()
```

This is parser smoke evidence using invented references. It does not establish
server compatibility, clinical validity or authority to read a referenced
resource.

## Content and control boundaries

Each `SubscriptionEvent` records its input position, event number, optional
focus-reference digest, additional-reference digests and observed content mode.
Reference strings stay in the caller's input and are never included in the
returned object or an error.

| Observed event content | Classification | Authorized follow-up read needed |
| --- | --- | --- |
| No focus or context references | `empty` | Yes |
| At least one referenced resource is absent | `id-only` | Yes |
| Focus and every context reference resolve within the Bundle | `full-resource` | No missing content to fetch |

The classifier matches exact `fullUrl` values and bounded `ResourceType/id`
references. It does not equate different server origins, dereference a URL,
choose among ambiguous entries, or claim the sender's declared payload setting
is correct. Partial context is treated conservatively as `id-only`. Full
resource content still is not returned; any use of the original input remains
subject to separate authorization and clinical validation.

`workflow_events` includes metadata only for `event-notification` messages with
no reconciliation findings. Handshakes and heartbeats may contain historical
event metadata, but expose no new workflow events. Neither `query-status` nor
historical `query-event` results are promoted into a new workflow. An event
appearing in this property grants no execution permission.

## Sequence reconciliation

For best-effort global numbering, pass the last processed number from a
separately governed checkpoint as `previous_event_number`. Missing interior or
trailing ranges are represented by their first number, last number and count;
the parser never expands a large missing range into a list.

- Repeated numbers, previously processed events and backward delivery produce
  explicit duplicate or out-of-order findings.
- An event number greater than the server counter, a regressed counter, a
  missing event counter or a server-reported error requires reconciliation.
- Without a prior checkpoint, a first number above one produces
  `history_not_examined`; the parser does not invent missing preceding history.
- Query-event history does not imply that the server's newer events were lost.
- A heartbeat, handshake or query-status counter ahead of a known checkpoint
  also reports the undelivered range. With no checkpoint, preceding history
  stays unknown; control messages still expose no workflow events.

R5 also supports guaranteed-delivery channels with Bundle-relative numbering.
Use `global_event_numbers=False` for that negotiated mode. The parser checks
relative gaps and order without comparing numbers to the global counter, and
refuses a global checkpoint in that mode. This distinction follows the
[R5 event-numbering contract](https://hl7.org/fhir/R5/subscriptionstatus.html).

Checkpoint persistence, delivery deduplication across calls, authorized reads
and workflow dispatch are separate integrations. No notification advances a
checkpoint automatically.

## Bounded refusals and version checks

Limits are one MiB of canonical JSON, 32,768 JSON nodes, depth 32, 1,024 Bundle
entries, 256 events and 64 context references per event. Counter strings use
unsigned decimal notation bounded by the signed 64-bit maximum. Builtin JSON
containers are snapshotted; caller input must remain unchanged during parsing.

Malformed JSON, duplicate keys, unsupported versions, invalid parsed structures,
unhandled modifiers, ambiguous references and exceeded bounds raise
`SubscriptionNotificationError`. Its `code` and optional numeric `position`
contain no rejected value, source path or decoder message. Reports retain only
digests, classifications, sequence ranges and counts.
Unhandled extension keys, primitive extension companions and `implicitRules`
on parsed Bundle, entry, status, event, parameter or reference structures are
refused even when their supplied value is empty or malformed. Resource payloads
remain outside this parser's clinical schema validation boundary.

The parser rejects incompatible leading status formats, additional status
resources and conflicting recognized core or notification profile release
declarations. FHIR JSON does not establish a complete release identity by
itself: undeclared payload resources and custom profiles are not validated
against every FHIR schema. Supply the negotiated release and perform resource
validation separately before using any original clinical content. This intake
parser does not interpret clinical fields or timestamps.
