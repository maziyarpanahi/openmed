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

Async job callbacks are signed with the version 2 webhook-delivery contract,
which wraps the version 1 canonical string above without changing it. It lets a
receiver tell one logical delivery from its retries, reject a retransmitted
attempt even after a restart, and keep verifying during a key rotation. See
[Async REST Jobs & Webhooks](async-jobs.md) for delivery and retry behavior.

### Delivery Headers

Every attempt carries:

- `X-OpenMed-Event`: event type, such as `job.done`
- `X-OpenMed-Delivery-Id`: opaque id, identical on every retry of one delivery
- `X-OpenMed-Attempt`: attempt number, `1`, `2`, ...
- `X-OpenMed-Timestamp`: Unix time of this attempt
- `X-OpenMed-Nonce`: fresh one-time value for this attempt
- `X-OpenMed-Webhook-Signature`: `v2,kid=<key id>,sha256=<64 lowercase hex>`

The key id names the key generation that signed the attempt; it is never key
material. A job webhook signed with the per-job `secret` uses a key id derived
from that secret (`hk_` followed by 24 hex characters: the first 12 bytes of
HMAC-SHA256 keyed by the secret over `OPENMED-SIGNING-KEY-ID-V1`), so a
receiver that holds the secret derives the same id with
`openmed.service.signing_keys.derive_key_id`.

For receivers that still verify version 1, each attempt also carries the V1
`X-OpenMed-Signature: sha256=<hex>` header, computed with the same key. Senders
that call `deliver_webhook` directly can turn it off with
`include_legacy_signature=False`.

### Version 2 Canonical String

```text
OPENMED-WEBHOOK-HMAC-SHA256-V2
key-id:<KEY ID>
event:<EVENT TYPE>
delivery-id:<DELIVERY ID>
attempt:<ATTEMPT NUMBER>
<THE VERSION 1 CANONICAL STRING>
```

The signature is the lowercase hexadecimal HMAC-SHA256 of those UTF-8 bytes.
Test vector, for receivers in other languages:

| Input | Value |
| --- | --- |
| secret | `synthetic-webhook-secret` |
| method, path | `POST`, `/callbacks/openmed?source=jobs` |
| timestamp, nonce | `1800000000`, `synthetic-attempt-nonce` |
| body | `{"event":"job.done","job_id":"synthetic-job","status":"done"}` |
| key id, event | `webhooks-2026-11`, `job.done` |
| delivery id, attempt | `dlv_00000000000000000000000000000001`, `2` |
| signature | `bf5c4494b17722c287c69ecf8b1c8ffa5425067eeafbadc80ffc57d08bc529a3` |

### Verify and De-duplicate Deliveries

`openmed.service.webhook_replay.WebhookReceiver` checks the headers, freshness,
key state, and HMAC, then claims the attempt in a durable `FileReplayLedger`:

```python
import os

from openmed.service.signing_keys import SigningKeyring
from openmed.service.webhook_replay import FileReplayLedger, WebhookReceiver

receiver = WebhookReceiver(
    keyring=SigningKeyring.single(os.environ["OPENMED_WEBHOOK_SECRET"]),
    ledger=FileReplayLedger(
        "/var/lib/receiver/openmed-webhooks.json",
        integrity_key=bytes.fromhex(os.environ["RECEIVER_LEDGER_KEY_HEX"]),
    ),
)


def handle(method, path_and_query, body, headers):
    verification = receiver.verify(method, path_and_query, body, headers)
    if verification.accepted:
        try:
            apply_effect(body)  # your idempotent side effect
        except Exception:
            receiver.release(verification)  # let a later retry claim it
            raise
        receiver.mark_applied(verification)
    return verification.http_status
```

Generate the ledger integrity key once (for example
`secrets.token_bytes(32).hex()`) and keep it separate from webhook secrets.
`verify` never raises for hostile input and returns one typed outcome:

| Outcome | Meaning | Suggested status |
| --- | --- | --- |
| `accepted` | New delivery; apply it, then call `mark_applied` | 200 |
| `duplicate` | A fresh retry of a delivery already claimed or applied | 200 |
| `replayed` | A retransmitted attempt or reused nonce | 409 |
| `stale` | Timestamp outside the window (default 300 seconds) | 401 |
| `bad_signature` | Missing, malformed, or mismatched signature headers | 401 |
| `unknown_key` | Unknown key id, or a key that is not active yet | 401 |
| `retired_key` | Retired key, or a retiring key after its overlap | 401 |
| `unavailable` | Ledger or keyring integrity or capacity not established | 503 |

`reason` gives a controlled code for each outcome, such as `delivery_applied`
or `ledger_integrity_failed`. Delivery and key fields are filled in only after
the signature verifies. A failed verification never writes to the ledger, so a
forged request cannot use up a genuine delivery id, and a wrong secret produces
the same `bad_signature` result as any other mismatch.

The ledger is one `0600` JSON file replaced atomically on every change, with a
sibling `.lock` file that serializes receivers across threads and processes,
so two workers cannot both claim one delivery. It stores only HMAC digests of
delivery ids, attempts, and nonces, plus key ids, attempt numbers, states, and
timestamps, and an HMAC over the whole file detects tampering. It fails closed
with `unavailable` when the file is corrupt, signed with another key, rolled
back or deleted while the receiver runs, or full of unexpired entries
(`max_entries`, default 10,000). Attempt and nonce records expire with the
freshness window; delivery records expire after `delivery_ttl_seconds`
(default 24 hours), which is the receiver's idempotency horizon.

Requests that carry only the V1 header are rejected with
`signature_version_unsupported` unless the receiver is built with
`accept_legacy_v1=True`. Legacy requests then get nonce replay protection but
no retry de-duplication, because V1 has no delivery id.

### Rotate Signing Keys

A `SigningKeyringFile` holds one complete key generation: exactly one `active`
key, `retiring` keys that verify until `verify_until`, and `retired` key ids
that are refused. Point `OPENMED_WEBHOOK_SIGNING_KEYRING_FILE` at it, or use
`OPENMED_WEBHOOK_SIGNING_SECRET` or `OPENMED_WEBHOOK_SIGNING_SECRET_FILE` (with
an optional `OPENMED_WEBHOOK_SIGNING_KEY_ID`) for a single static key, and load
it with `signing_keyring_from_env()`.

```json
{
  "format": "openmed.signing-keyring",
  "schema_version": 1,
  "generation": 2,
  "rotated_at": 1800000000,
  "keys": [
    {"key_id": "webhooks-2026-11", "state": "active",
     "secret_file": "webhooks-2026-11.key", "not_before": 1800000000},
    {"key_id": "webhooks-2026-10", "state": "retiring",
     "secret_env": "WEBHOOKS_2026_10_SECRET", "verify_until": 1800086400},
    {"key_id": "webhooks-2026-09", "state": "retired"}
  ]
}
```

Each active or retiring key names one secret source: `secret_file` (resolved
next to the keyring file), `secret_env`, or inline `secret_b64`. A generation
that is malformed, or whose referenced secret is missing, is rejected as a
whole. `SigningKeyringFile.rotate(new_secret=..., overlap_seconds=...)`
writes the next generation in one atomic replace: the new key becomes active,
the previous active key becomes retiring for the overlap window (at most seven
days), and the previously retiring key is retired. An interrupted rotation
leaves the previous complete generation in place. A loaded resolver refuses to
move back to an older generation, or to a different file with the same
generation number, and keeps using the generation it has; pass
`min_generation` to carry that floor across restarts.

When the sender and receiver keep separate keyring files, rotate the receiver
first so that it already trusts the new key, then rotate the sender before the
receiver's overlap window ends. In-flight retries signed with the old key keep
verifying until then. Rotation receipts and `SigningKeyring.describe()` hold
only generations, key ids, states, and timestamps.
