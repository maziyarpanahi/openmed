# API-key and JWT Verifier Generations

`openmed.service.auth_generations` rotates REST API-key hashes and JWT
verification keys without restarting the service. The caller supplies each
complete set of verifiers as one *generation*. OpenMed validates the whole
generation, publishes it as one immutable snapshot, and keeps the previous
generation only for a bounded overlap window. A rejected reload never changes
what is active.

This feature owns the verifier lifecycle only. It does not reload TLS or mTLS
certificates, issue tokens, or discover keys from an identity provider. Key
custody stays with the caller, and nothing here makes network calls.

## Generation contents

A generation is a `VerifierGenerationSpec` or an equivalent JSON document:

```json
{
  "generation_id": "2026-10-rotation",
  "serial": 7,
  "api_keys": [
    {
      "id": "clinic-api",
      "principal": "clinic-api",
      "key_hash": "sha256:<sha256-hex-digest>",
      "scopes": ["pii:read"]
    }
  ],
  "jwks": {"keys": [{"kty": "RSA", "kid": "rs-2026-10", "alg": "RS256",
                     "n": "<base64url>", "e": "AQAB"}]},
  "jwt": {
    "issuer": "https://issuer.example.invalid",
    "audience": ["openmed"],
    "leeway_seconds": 30,
    "max_lifetime_seconds": 900
  }
}
```

`api_keys` and `jwks` use the same shapes as `OPENMED_SERVICE_AUTH_API_KEYS`
and `OPENMED_SERVICE_AUTH_JWKS` (see [REST Authentication](../serving/authentication.md))
and are parsed by the same functions. Token checks reuse
`authenticate_api_key` and `authenticate_jwt` from `openmed.service.auth`; the
signature, `exp`/`nbf`/`iat`, issuer, audience and lifetime rules are unchanged.

A generation is complete only when all of these hold:

- `generation_id` is 1-64 ASCII letters, digits, `.`, `_`, `:` or `-`, and
  `serial` is a positive integer.
- It holds at least one API key or JWK.
- API-key hashes are `sha256:<64 hex>`; key IDs and hashes are unique.
- Every JWK has a unique `kid`, is `oct`/HS256 or `RSA`/RS256 with `use`
  absent or `sig`, and carries no private members (`d`, `p`, `q`, `dp`, `dq`,
  `qi`, `oth`).
- HS256 secrets are at least 32 bytes. RSA moduli are at least 2048 bits by
  default (`min_rsa_modulus_bits`, floor 1024) with an odd exponent of 3 or more.
- When JWKS are present, an issuer and at least one audience are set.
  Leeway is 0-300 seconds and the maximum token lifetime is finite and positive.

## Lifecycle

```python
from openmed.service.auth_generations import VerifierGenerationRegistry

registry = VerifierGenerationRegistry(max_overlap_seconds=3600)

receipt = registry.reload(generation_document)                  # first generation
receipt = registry.reload(next_document, overlap_seconds=600)   # rotate
if not receipt.accepted:
    print(receipt.reason)   # fixed code; the last valid generation stays active
```

| Operation | Effect |
| --- | --- |
| `stage(spec)` | Validate and hold a generation without accepting it. |
| `activate(generation_id, overlap_seconds=...)` | Atomically make a staged generation active. |
| `reload(spec, overlap_seconds=...)` | Validate and activate in one step. |
| `revoke(generation_id)` | Stop accepting a staged, active or overlapping generation now. |
| `status()` / `metrics()` | Labels, serials, epochs, boundaries and counters only. |

Rules that keep rotation bounded:

- `serial` must increase on every activation, so an older generation cannot be
  replayed. A generation ID cannot be reused while it is known to the registry
  or among the last 256 accepted IDs.
- `overlap_seconds` must be between 0 and `max_overlap_seconds`. With 0 the
  previous generation stops immediately. Otherwise it is accepted while
  `now < retire_at` and rejected from `retire_at` on.
- At most one previous generation overlaps. Activating again drops any older
  overlapping generation.
- During an overlap, a credential carried over from the previous generation
  must be identical. The same key hash with a different key ID, principal or
  scopes (`api_key_conflict`), or the same `kid` with different key material
  (`jwk_kid_conflict`), is rejected. Changing a key's scopes needs a zero-overlap
  cutover or a new key.
- Revoking the active generation fails closed: requests are rejected until
  another generation is activated. Revocation works even if the clock fails.

The clock is injected (`clock=`, default `time.time`). JWT time claims and
overlap boundaries use the same reading for each decision. A non-finite or
non-numeric reading rejects lifecycle changes with `clock_invalid` and rejects
authentication.

## Pinned decisions

Lifecycle changes are serialized and publish a new snapshot with one reference
assignment. Each decision reads one snapshot and checks the active generation,
then the overlapping one. Every check runs against a single generation's keys
and JWT bounds. `authenticate_api_key` and `authenticate_bearer` return a
`GenerationDecision` whose `pin` records the generation ID, serial, snapshot
epoch and role (`active` or `overlap`). `registry.view()` returns a view pinned
to one snapshot for callers that need several consistent checks. Inputs are
deep-copied at validation, so changing the caller's objects later has no effect.

To use generations in the REST service, replace the app's auth middleware. The
service reads `app.state.auth` on every request:

```python
from openmed.service.app import create_app
from openmed.service.auth_generations import GenerationalServiceAuth

app = create_app()
app.state.auth = GenerationalServiceAuth.replacing(app.state.auth, registry=registry)
```

Route scopes, exempt paths and failure rate limits still come from the
`OPENMED_SERVICE_AUTH_*` settings. API keys and JWKS come only from the
registry. Authenticated requests carry the `GenerationPin` in
`request.state.auth_generation` and `request.scope["openmed.auth_generation"]`.
Failures return the same `401 invalid_credentials` envelope as static auth.

## Privacy of receipts and diagnostics

Every lifecycle call returns a `GenerationReceipt` with the action, outcome,
fixed reason code, epoch, clock time, generation ID and serial (only for a
validated or known generation), overlap boundary, key counts, and a canonical
`sha256:` digest over those fields. Receipts, `status()`, `metrics()`, object
representations and `GenerationValidationError` never contain raw API keys,
key hashes, JWK material, tokens, token claims, principals or invalid input
values. Errors raised from validation carry no chained exception context.

Reason codes: `document_malformed`, `document_field_unknown`,
`generation_id_invalid`, `generation_serial_invalid`, `generation_serial_stale`,
`generation_id_reused`, `generation_empty`, `generation_unknown`,
`api_keys_malformed`, `api_key_hash_invalid`, `api_key_duplicate`,
`api_key_conflict`, `jwks_malformed`, `jwk_unsupported`, `jwk_kid_missing`,
`jwk_kid_duplicate`, `jwk_kid_conflict`, `jwk_private_material`,
`jwk_key_invalid`, `jwk_key_too_weak`, `jwt_binding_incomplete`,
`jwt_leeway_invalid`, `jwt_lifetime_invalid`, `overlap_out_of_bounds`,
`staged_capacity_exceeded`, `clock_invalid`.

Metrics are counters keyed by `<action>_accepted`,
`<action>_rejected:<reason>`, `decision_active`, `decision_overlap`,
`decision_rejected`, `decision_no_active_generation` and
`decision_clock_invalid`.

## Validation

```bash
PYTHONPATH=$PWD python -m pytest tests/unit/service/test_auth_verifier_generations.py -q
```

The tests use synthetic keys and an injected clock. They cover overlap
admission and rejection at the boundary, malformed and conflicting
generations, revocation, stale serials, caller mutation, a multi-threaded
reload test that fails if a reader ever sees a partial key set, sentinel leak
checks, and REST middleware rotation without a restart.
