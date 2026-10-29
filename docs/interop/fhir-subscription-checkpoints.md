# FHIR Subscription intake checkpoints

`SubscriptionCheckpoint` is a local, ordered intake gate for notification
metadata. It stores keyed SHA-256 digests of the subscription, notification,
resource identity, and resource version; sequence numbers; and closed status
codes. It never stores resource bodies, raw identifiers, or the HMAC key.
Supply a private local database path and a stable, randomly generated 32-byte
(or longer) secret from your own key store. Losing or changing that key makes
old deliveries impossible to match safely.

```python
from openmed.interop.fhir.subscription_checkpoint import SubscriptionCheckpoint

# Load a stable secret from a caller-controlled, local secret store.
with SubscriptionCheckpoint("private/subscription.sqlite", secret=secret) as checkpoint:
    delivery = dict(
        subscription_id="opaque-subscription",
        notification_id="opaque-notification",
        resource_id="opaque-resource-reference",
        resource_version="opaque-version",
        sequence=0,
    )
    decision = checkpoint.claim(**delivery)
    if decision.status == "claimed":
        # Durably accept the work under the same idempotency identity first.
        workflow_queue.accept_once(delivery)
        checkpoint.commit(**delivery)
    elif decision.status == "quarantined":
        review_queue.record_code(decision.reason, decision.sequence)
```

Each subscription starts at sequence zero by default. Set
`initial_sequence=1` when the source's first event number is one. `claim`
durably reserves the next
sequence. The caller commits only after durable workflow acceptance. A retry
of a committed notification or resource version is a duplicate; a retry of an
unconfirmed claim is quarantined because its downstream outcome is unknown.
A crash between downstream acceptance and `commit` also needs reconciliation:
use the downstream idempotency identity to prove acceptance before calling
`commit`. This checkpoint alone cannot guarantee exactly-once external side
effects.

A future sequence is quarantined as a gap. After preceding sequences commit,
call `claim(..., replay_gap=True)` with the original delivery metadata to
reserve that gap for replay. Conflicting notification IDs or sequence numbers,
old deliveries beyond the deduplication window, and gaps beyond `max_gap`
require review. The default history window is 128 committed sequences, and
at most 512 unresolved records are retained per subscription. When capacity
is reached, new anomalies return `capacity` without entering the store.
`quarantines(subscription_id)` lists only event digests, sequence numbers, and
reason codes, including unconfirmed claims. After reconciling a quarantined
conflict with the source, an operator can call
`discard_quarantine(subscription_id, event_digest)` to release that record's
capacity. Pending claims cannot be discarded; commit them only with durable
downstream acknowledgment. Re-delivery after discard is evaluated against the
current sequence and deduplication state.
The database uses local SQLite transactions and makes no network calls.
Keep the database and secret protected by local access controls; digests are
metadata, not a substitute for access control.
