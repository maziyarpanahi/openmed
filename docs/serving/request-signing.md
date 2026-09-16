# HMAC Request Signing

OpenMed provides a shared HMAC-SHA256 scheme for callers that need request
integrity and replay protection. It is available for inbound request
verification and is also used by async job webhooks.

This is an application-level integrity check. Keep using HTTPS, service
authentication, and a secret manager; request signing does not replace them.

## Signature Headers

Every signed request carries exactly these headers:

- `X-OpenMed-Timestamp`: Unix time in whole seconds
- `X-OpenMed-Nonce`: a unique, opaque value for this request
- `X-OpenMed-Signature`: `sha256=<64 lowercase hex characters>`

The default freshness window is 300 seconds. Keep clocks synchronized and
generate a new nonce for every new request.

## Canonical String

The version 1 canonical string is UTF-8 encoded with newline separators:

```text
OPENMED-HMAC-SHA256-V1
<UPPERCASE HTTP METHOD>
<EXACT PATH AND QUERY>
<UNIX TIMESTAMP>
<NONCE>
sha256:<LOWERCASE SHA-256 OF THE EXACT BODY BYTES>
```

Use the origin-form request target, such as `/jobs?mode=async`. Do not include
the scheme, host, or URL fragment, and do not decode or reorder query
parameters. The body hash—not the body—is placed in the canonical string, so
raw PHI is never copied into signing or replay-cache state.

The signature is the lowercase hexadecimal HMAC-SHA256 digest of those bytes.
Header names are matched case-insensitively, but their values and the request
target must be preserved exactly.

## Sign a Client Request

Serialize the body once, sign those exact bytes, and send those same bytes:

```python
import json
import os

import httpx

from openmed.service.signing import sign_request

body = json.dumps(
    {"document_hash": "sha256:...", "offsets": [[8, 20]]},
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
headers = sign_request(
    "POST",
    "/jobs",
    body,
    secret=os.environ["OPENMED_HMAC_SECRET"],
)
headers["Content-Type"] = "application/json"

response = httpx.post(
    "https://openmed.example.com/jobs",
    content=body,
    headers=headers,
)
response.raise_for_status()
```

`sign_request_headers` is an equivalent, explicitly named helper for clients.
Both helpers generate a cryptographically random nonce unless one is supplied.

## Verify a Request

Create one cache per shared-secret verification domain and keep it for the life
of the process:

```python
import os

from openmed.service.signing import (
    NonceCache,
    SignatureVerificationError,
    verify_request_signature,
)

nonce_cache = NonceCache(max_entries=10_000, window_seconds=300)


def verify(method, path_and_query, body, headers):
    try:
        verify_request_signature(
            method,
            path_and_query,
            body,
            headers,
            secret=os.environ["OPENMED_HMAC_SECRET"],
            nonce_cache=nonce_cache,
            max_skew_seconds=300,
        )
    except SignatureVerificationError:
        return False
    return True
```

Verification checks freshness and the HMAC before atomically consuming the
nonce. A tampered method, path, timestamp, nonce, or body fails without
occupying cache space. A successful request consumes its nonce, so calling the
verifier again for that request correctly reports a replay.

The cache removes nonces only after their acceptance windows pass. It never
evicts an active nonce just to make room: if `max_entries` is reached, new
requests fail closed with `NonceCacheFullError`. Increase the bound for the
expected peak number of signed requests per window. The cache is local to one
process; deployments that require cross-process replay protection need an
external nonce store, which is outside this feature's scope.

## Signed Webhooks

Async job callbacks use this same canonical scheme with method `POST`, the
callback URL's exact path and query, and the canonical JSON bytes sent on the
wire. Receivers can pass the callback request directly to
`verify_request_signature`. See [Async REST Jobs & Webhooks](async-jobs.md) for
delivery and retry behavior.
