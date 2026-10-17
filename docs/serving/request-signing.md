# HMAC Request Signing

OpenMed provides a shared HMAC-SHA256 scheme for callers that need request
integrity and replay protection. The REST service can require it on selected
routes (see [Require Signatures on Service Routes](#require-signatures-on-service-routes)),
and async job webhooks use the same scheme.

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

## Require Signatures on Service Routes

The REST service can require valid signatures on chosen routes. It is off by
default. When it is off, no middleware is installed and every route behaves as
before.

Write the shared secret to a file readable only by the service account, then
name the file and the routes:

```bash
OPENMED_SERVICE_REQUEST_SIGNING_SECRET_FILE=/run/secrets/openmed-signing-key \
OPENMED_SERVICE_REQUEST_SIGNING_ROUTES="POST /jobs,GET /jobs/{job_id}" \
python -m openmed.service.logging --host 127.0.0.1 --port 8080
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `OPENMED_SERVICE_REQUEST_SIGNING_ROUTES` | unset (off) | Comma-separated `METHOD /route-template` entries that must be signed |
| `OPENMED_SERVICE_REQUEST_SIGNING_SECRET_FILE` | unset | File that holds the shared secret |
| `OPENMED_SERVICE_REQUEST_SIGNING_MAX_SKEW_SECONDS` | `300` | Accepted clock skew; also the nonce replay window |
| `OPENMED_SERVICE_REQUEST_SIGNING_NONCE_CACHE_SIZE` | `10000` | Maximum active nonces held by one process |

Route entries use the route template, not a concrete path. `GET /jobs/{job_id}`
covers `/jobs/abc` and `/jobs/xyz`. The same template also guards `HEAD`
requests to it. Requests to routes that are not listed pass through unchanged
and their bodies are not read.

The secret file is read once at startup. Leading and trailing whitespace,
such as a final newline, is removed. The secret must be 32 to 4096 bytes.
The service refuses to start when:

- routes are set without a secret file, or a secret file without routes;
- the secret file is missing, unreadable, empty, too short, or too long;
- an entry is malformed or names an unknown or unregistered route template;
- a skew or cache-size value is not a positive integer.

Startup errors name only the setting. They never include the file path, the
route entry, or the secret.

### Failure Codes

A rejected request never reaches the route handler. Failures use the standard
service error envelope with `details: null`, the request ID, and a
`WWW-Authenticate: OPENMED-HMAC-SHA256` header:

| Status | Code | Cause |
| --- | --- | --- |
| `401` | `signature_missing` | One or more of the three signature headers is absent |
| `401` | `signature_invalid` | Malformed or duplicate headers, or the HMAC does not match the method, target, timestamp, nonce, or body |
| `401` | `signature_stale` | The timestamp is outside the skew window |
| `401` | `signature_replayed` | The nonce was already accepted inside its window |
| `503` | `service_busy` | The nonce cache is full of active entries, so the request is refused |
| `413` | `payload_too_large` | A signed route body exceeds the service request-byte limit |

Responses, logs, and metrics never contain signatures, nonces, secrets, or
body content. The service does not write a separate log line for signature
failures; the access log records the route template and status, never the
raw target or headers.

### Middleware Order

Signature checks run after the request-body limits and before routing:

1. The request-size limit (`OPENMED_SERVICE_MAX_REQUEST_BYTES`) and the
   model-route body limit reject oversized bodies first.
2. The signing middleware buffers the body, verifies it, and replays the
   identical bytes to the handler.
3. Tracing, readiness, metrics, throttling, authentication, trusted-host
   checks, and routing then run as usual.

Because signatures are checked before authentication, an unsigned request to
a signed route gets `signature_missing` even when it also lacks credentials.
Signature failures are not counted by the authentication failure rate limiter
or by the request metrics.

The signed target is the path and query exactly as the OpenMed process
receives them. If a proxy rewrites the path, sign the rewritten path.

### Nonce Cache Limits

Each service process keeps its own in-memory nonce cache:

- It holds at most `OPENMED_SERVICE_REQUEST_SIGNING_NONCE_CACHE_SIZE` active
  nonces. A nonce stays active until its timestamp plus the skew window has
  passed.
- It never evicts an active nonce to make room. When it is full, new signed
  requests get `503 service_busy` until older entries expire. Size it for the
  peak number of signed requests per window, per process.
- It is lost on restart. A request captured shortly before a restart can be
  replayed once against the new process while its timestamp is still fresh.

Replay protection is per process. With several replicas or worker processes,
each one has its own cache, so the same signed request can be accepted once
by each process. Use sticky routing or a single worker when that matters.
A shared nonce store is out of scope; webhook delivery and key rotation are
covered separately.

## Signed Webhooks

Async job callbacks use this same canonical scheme with method `POST`, the
callback URL's exact path and query, and the canonical JSON bytes sent on the
wire. Receivers can pass the callback request directly to
`verify_request_signature`. See [Async REST Jobs & Webhooks](async-jobs.md) for
delivery and retry behavior.
