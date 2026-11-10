# Consent-aware cached outputs

`openmed.clinical.consent_cache` provides a local, in-memory cache for derived
clinical outputs whose consent context can later be revoked. It is an
infrastructure safeguard, not a compliance certification or a clinical
decision guarantee.

## Privacy and determinism

The cache never makes a network call. Cache keys, consent scopes, and consent
revisions are canonicalized only long enough to calculate deterministic
SHA-256 fingerprints. Entry metadata stores those fingerprints, not the raw
inputs. Cached outputs remain in memory for normal cache use, but are excluded
from entry `to_dict()` output and from revocation audit events.

Revocation events contain only the event type and the number of invalidated
entries:

```python
{"event_type": "consent_cache_invalidation", "invalidated_count": 2}
```

Applications should retain this aggregate event through their own approved
audit channel. The module does not write files, emit logs, contact a consent
registry, or decide whether a new processing request is legally permitted.

## Usage

```python
from openmed.clinical.consent_cache import ConsentCache

cache = ConsentCache()
cache.put(
    "synthetic-result-key",
    {"summary": "synthetic derived output"},
    scope="summary",
    revision="receipt-v1",
)

result = cache.get(
    "synthetic-result-key",
    scope="summary",
    revision="receipt-v1",
)

event = cache.revoke(scope="summary", revision="receipt-v1")
assert event.invalidated_count == 1
assert cache.get(
    "synthetic-result-key",
    scope="summary",
    revision="receipt-v1",
) is None
```

The scope and revision supplied to `get()` must match the context used by
`put()`. If they are omitted, a lookup succeeds only when one consent context
exists for that cache key. This avoids selecting an arbitrary result when a
key has entries for multiple revisions.

An exact revocation removes all entries with the matching scope and revision
and rejects later writes for that pair. Calling `revoke(scope="summary")`
removes all revisions currently held for that scope and blocks those observed
revisions while allowing a previously unseen revision to represent a new
consent receipt.

The cache is bounded by `max_entries` (1024 by default). Eviction due to the
bound is not a consent revocation and is not included in the revocation audit
count.
