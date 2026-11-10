# Surrogate-key providers

OpenMed's reversible surrogate mappings need one explicit key-custody boundary.
The contract in `openmed.core.surrogate_key_provider` lets an in-memory vault,
an encrypted local store, or a user-managed key service expose the same
lifecycle semantics without making the mapping implementation responsible for
key custody.

This boundary is about key material, not clinical decisions or compliance
certification. Deployments still own authorization, retention, backup, access
review, and the custody of any root secret.

## Contract

Implement `SurrogateKeyProvider` with these members:

- `capabilities`: `SurrogateKeyCapabilities` metadata. Providers must opt in to
  `lookup`, `rotation`, `scope`, and `destruction` independently.
- `lookup(scope, key_id=None)`: return the active key, or a specified historical
  version, without exposing a mapping or source value.
- `rotate(scope)`: create the next key version for a scope.
- `scope(scope)`: return lifecycle metadata such as the active key ID and
  versions. It never returns key bytes.
- `destroy(scope, key_id=None)`: destroy one version, or the complete scope when
  no key ID is supplied.

Call `require_surrogate_key_capabilities(provider, required)` before composing a
workflow that needs more than one operation. The check fails closed: a provider
with missing or invalid capability metadata cannot silently fall back to a
weaker lifecycle path.

```python
from openmed.core.surrogate_key_provider import (
    InMemorySurrogateKeyProvider,
    SurrogateKeyCapability,
    require_surrogate_key_capabilities,
)

provider = InMemorySurrogateKeyProvider(seed="synthetic-demo-seed")
require_surrogate_key_capabilities(
    provider,
    (SurrogateKeyCapability.LOOKUP, SurrogateKeyCapability.ROTATION),
)

key = provider.lookup("synthetic-study")
assert key is not None
assert key.scope == "synthetic-study"
assert key.material  # pass only to a local cryptographic operation

next_key = provider.rotate("synthetic-study")
assert next_key.version == 2
```

The scope identifier is an opaque application label. Do not put names, medical
record numbers, encounter text, or other sensitive values in a scope, key ID,
exception, log, or report.

## Bundled local implementation

`InMemorySurrogateKeyProvider` is an alias for
`LocalSyntheticSurrogateKeyProvider`. It uses only Python's standard library,
derives deterministic 32-byte synthetic material with HMAC-SHA256, and makes no
network call or file write. Two providers with the same seed, scope, and version
produce the same material, which makes the implementation suitable for offline
fixtures and contract tests.

The implementation keeps key versions in process memory. Rotation keeps older
versions available until they are explicitly destroyed; destroying a scope
removes its material and makes the scope terminal for that provider instance.
This is useful local lifecycle behavior, not a guarantee of hardware-backed
erasure. Production deployments should provide a provider whose destruction and
authorization semantics match their key-management requirements.

## Privacy-safe evidence

`SurrogateKeyMaterial.as_metadata()` returns only the key ID, scope, version,
material length, and a SHA-256 fingerprint. `SurrogateKeyScope.as_dict()` and
`capability_metadata` likewise exclude raw key bytes and source mappings. Use
these metadata methods for logs, reports, or audit artifacts; never serialize
`material` or a reversible mapping.

The provider does not log, persist, or phone home. The no-raw-PHI logging policy
still applies to caller-owned scope labels and exception handling:

- log operation names, versions, counts, and capability names;
- keep key bytes, source identifiers, and mapping payloads out of messages;
- use synthetic offline fixtures for tests and examples;
- treat fingerprints as linkage metadata and protect them according to the
  deployment's audit policy.

This contract does not make a seeded synthetic key suitable for production
custody, and it does not make a reversible mapping cryptographically
non-invertible.
