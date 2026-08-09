# Provenance-aware terminology cache

`openmed.structured.terminology_cache` provides a small local cache for
caller-supplied terminology responses. It is deterministic and does not make
network requests, read environment configuration, or download vocabulary
content.

## Store and retrieve a release-pinned response

Use both the vocabulary identifier and the exact release identifier for every
entry. The returned entry carries the source and SHA-256 fingerprints needed to
audit which response was used:

```python
from openmed.structured.terminology_cache import TerminologyCache

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
assert cached.provenance.source == "local-fixture"
assert cached.provenance.fingerprint.startswith("sha256:")
```

Responses are copied and canonicalized before storage. Mapping key order does
not change the response fingerprint, and callers receive a detached copy from
`entry.response`. Responses must be JSON-compatible and must not contain
non-finite numbers.

## Refuse stale releases

The cache never substitutes one release for another. If a vocabulary is cached
under one release and a caller requests a different release, `get()` raises
`StaleTerminologyError` instead of returning the older response. A source
mismatch for an otherwise exact key raises `TerminologyProvenanceError`.

```python
from openmed.structured.terminology_cache import (
    StaleTerminologyError,
    TerminologyCache,
)

try:
    cache.get("synthetic-vocabulary", "2026.02")
except StaleTerminologyError:
    # Load the caller's 2026.02 response explicitly, then call cache.put().
    pass
```

`get_or_compute()` accepts a caller-owned computation for a cache miss. It does
not run that computation for an exact hit and does not run it after a stale
release is detected. Any network access required to obtain a response remains
an explicit caller concern; the cache itself is local-only.

## Privacy-safe metadata

`TerminologyCacheEntry.to_dict()` and `TerminologyCache.report()` contain only
cache keys, source identifiers, schema values, and fingerprints. They omit the
terminology response by default, so they can be used for logs and audit
metadata without copying response contents. Use `entry.response` only at the
explicit terminology operation that needs the caller-supplied data. Source
identifiers should be stable non-sensitive labels, never patient identifiers
or note text.

The cache contains no bundled restricted vocabulary, credentials, or clinical
decision logic. It provides provenance and reuse only; callers remain
responsible for licensing, validation, and qualified review of terminology
content.
