# Provenance-aware terminology cache

`openmed.structured.terminology_cache` caches caller-supplied responses locally,
without network requests, environment configuration, or vocabulary downloads.

## Store and retrieve an exact release

```python
from openmed.structured import TerminologyCache

cache = TerminologyCache()
entry = cache.put(
    "synthetic-vocabulary",
    "2026.01",
    {"codes": [{"code": "SYN-001", "display": "Synthetic finding"}]},
    source="local-fixture",
)
cached = cache.get("synthetic-vocabulary", "2026.01")
assert cached is not None
assert cached.response == entry.response
assert cached.provenance.release == "2026.01"
assert cached.provenance.fingerprint.startswith("sha256:")
```

Responses are copied and canonicalized; mapping order does not affect their
SHA-256 fingerprints. `entry.response` returns a detached copy. Inputs must be
JSON-compatible: non-finite numbers, cycles, and nesting beyond 64 containers
raise value-free errors.

A cached vocabulary/release key is immutable. Repeating `put()` with identical
content and source is idempotent. Changing either raises
`TerminologyProvenanceError`; call `invalidate()` before intentional replacement.

## Refuse stale releases

Requesting a different release of a cached vocabulary raises
`StaleTerminologyError`. An exact key with a mismatched source raises
`TerminologyProvenanceError`. Load the requested release explicitly; the cache
never substitutes an older response.

`get_or_compute()` calls the supplied computation only on a cache miss, never
on an exact hit or after detecting a stale release. Any network retrieval is
the caller's responsibility.

## Privacy-safe metadata

`TerminologyCacheEntry.to_dict()` and `TerminologyCache.report()` omit response
contents by default. They expose keys, source identifiers, schema values, and
fingerprints for auditing. Use stable, non-sensitive source labels; never use
patient identifiers or note text. Access `entry.response` only where needed.

No restricted vocabulary, credentials, or clinical decision logic is bundled.
Callers remain responsible for licensing, validation, and qualified review.
