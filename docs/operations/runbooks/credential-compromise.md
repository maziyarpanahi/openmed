# Credential compromise runbook

Use this runbook when a REST API key, JWT signing material, or an async job
webhook secret may have been exposed. For example, it may have been pasted in a
ticket or chat, committed to a repository, found in a support archive, or used
by an unknown caller. Read the [shared rules](service-runbooks.md#rules-for-tickets-logs-and-chat)
first. For a client certificate, use the
[client certificate runbook](client-certificate-compromise.md).

Credential changes take effect only after a restart. Verifier hot reload is
pending in #3699.

## Leaked API key

API keys are configured in `OPENMED_SERVICE_AUTH_API_KEYS` as a JSON list of
entries with `id`, `principal`, `key_hash` and `scopes`. The service stores only
the SHA-256 hash. A caller sends the raw key in the `X-API-Key` header or as
`Authorization: ApiKey <key>`.

### Detection signals

- A secret scanner or a person reports the raw key outside your secret store.
  See [secret scanning](../../security/secret-scanning.md).
- `openmed_service_request_total` shows more `401`, `403` or `429` responses on
  a route. This metric exists only when `OPENMED_SERVICE_METRICS_ENABLED` is
  on. The `429` responses use the `auth_rate_limited` code, which the failed
  authentication limiter (`OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS`,
  `OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST`) returns.
- Access logs show unusual volume, routes or times. Each access-log record has
  the method, route template, status code, duration, model name and request ID.
  API-key and JWT requests do not record the key `id` in the access log, so you
  cannot attribute traffic to one key from these logs alone.

### Containment

1. Identify the affected entry by its `id`. Do this by hashing the reported key
   locally (see [Rotation](#rotation)) and comparing the hash with the
   configured `key_hash` values. Do not send the key to anyone.
2. If the exposure is confirmed or likely, remove that entry from
   `OPENMED_SERVICE_AUTH_API_KEYS` and restart the service. Requests with the
   removed key then fail with `invalid_credentials`.
3. If you cannot restart quickly, block the affected routes or callers at your
   reverse proxy until the restart is done. OpenMed cannot revoke a key while
   it is running.
4. Make sure `OPENMED_SERVICE_AUTH_ENABLED` is `true`. Keep
   `OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT` at its default of `true` so that a
   route without a declared scope still requires a credential.

### Rotation

Use an overlap only when the old key was not exposed. Examples are scheduled
rotation and an operator leaving. For a confirmed leak, remove the old entry in
the same restart as you add the new one.

1. Generate the new key in your secret manager. Hash it without echoing it.
   The command below reads the key from a hidden prompt and prints only the
   `sha256:` hash:

    ```bash
    python -c "import getpass; from openmed.service.auth import hash_api_key; print(hash_api_key(getpass.getpass('New API key: ')))"
    ```

2. Add a new entry with a new `id` and the printed `key_hash`. Do not reuse an
   old `id`. Keep the scopes as narrow as the caller needs.
3. Restart the service and give the new key to the caller through the secret
   manager.
4. For an overlap, remove the old entry in a second restart after the caller
   has switched.

### Recovery

- Review what the compromised key could reach: its `scopes` and the routes in
  [REST authentication](../../serving/authentication.md#route-scopes). For
  example, a key with `jobs:read` could read job metadata with
  `GET /jobs/{job_id}` if the caller also knew a job ID.
- If a route that returns PHI may have been used, start the
  [breach notification runbook](../../compliance/breach-notification-runbook.md).

### Verification

1. Confirm that the old key is rejected. Put the header line
   `X-API-Key: <old key>` in a file with mode `0600`, written by your secret
   manager. Then send it to a scoped route. Using a header file keeps the key
   out of the command line and shell history:

    ```bash
    curl -sS -o /dev/null -w '%{http_code}\n' -H @"$OLD_KEY_HEADER_FILE" \
      http://127.0.0.1:8080/models/loaded
    ```

    The expected result is `401`. Repeat with the new key's header file. The
    expected result is `200`, or `403` if that key does not have
    `models:read`. Delete both header files afterwards.

2. Check that the `401` and `429` rates return to their normal level.

### Keep out of tickets and logs

- The raw key, the header file, and any shell history that contains the key.
  Record only the entry `id` and the time it was removed.
- The configured `OPENMED_SERVICE_AUTH_API_KEYS` value. A hash cannot recover
  the key, but the full list shows which principals and scopes exist.

## Leaked JWKS or JWT signing key

`OPENMED_SERVICE_AUTH_JWKS_FILE` and `OPENMED_SERVICE_AUTH_JWKS` hold the keys
that verify bearer tokens. The service supports `HS256` keys (`kty` `oct`),
whose `k` member is a shared secret, and `RS256` public keys (`kty` `RSA`).
Keys from the file and the inline value are combined. The impact depends on
what was exposed:

- An `oct` key, or a JWKS document that contains one, is a shared secret.
  Anyone who has it can mint accepted tokens.
- An `RSA` public key is not secret. The incident is the issuer's private key,
  which does not belong in OpenMed configuration.

### Detection signals

- A JWKS document that contains an `oct` key appears outside your secret store,
  or your identity provider reports a signing-key compromise.
- Tokens arrive with an unexpected `kid`, issuer or audience. The service
  rejects them with `invalid_credentials`. Watch for `401` spikes in
  `openmed_service_request_total`.
- Valid-looking traffic arrives from callers you do not recognize. Token claims
  are not written to access logs, so use your identity provider's issuance
  records.

### Containment

1. Remove the compromised key from both `OPENMED_SERVICE_AUTH_JWKS_FILE` and
   `OPENMED_SERVICE_AUTH_JWKS`, then restart. All tokens signed with that key
   then fail with `invalid_credentials`. They fail even if they have not
   expired yet.
2. Check that `OPENMED_SERVICE_AUTH_JWT_ISSUER` and
   `OPENMED_SERVICE_AUTH_JWT_AUDIENCE` are set. This stops tokens minted for
   other services from being accepted.
3. Keep `OPENMED_SERVICE_AUTH_JWT_MAX_LIFETIME_SECONDS` short. The production
   profile defaults it to one hour. It limits how long any token signed before
   the incident stays usable.

### Rotation

1. Create the new signing key at the issuer with a new, unique `kid`.
2. Add the new public key, or the new `oct` key, to the JWKS with that `kid`.
   When more than one key matches the token algorithm, the service requires a
   `kid` in the token header and accepts only an exact match. Tokens without a
   `kid` are rejected during an overlap. A `kid` used twice rejects every token
   that carries it.
3. For a confirmed compromise, remove the old key in the same restart. For a
   scheduled rotation, keep both keys until tokens signed with the old key have
   expired, then remove the old key in a second restart.

### Recovery

- Ask the issuer to stop issuing tokens with the old key and to revoke any
  client credentials that may have been exposed with it.
- Review which scopes those tokens could carry (`scope`, `scp` or `scopes`
  claims). Start the [breach notification runbook](../../compliance/breach-notification-runbook.md)
  if a PHI-returning route may have been reached.

### Verification

1. Present a token signed with the old key. The expected result is `401` with
   `invalid_credentials`. Use a synthetic test client that the issuer can
   revoke afterwards.
2. Present a token from the new key with the expected issuer and audience. The
   expected result is `200`, or `403` with `forbidden` if a scope is missing.

### Keep out of tickets and logs

- JWKS documents, the `k` member of any key, private keys, and tokens. Record
  only the `kid` values that were removed and added.
- Decoded token claims. They can name people or systems. Record counts of
  rejected requests instead.

## Leaked job webhook secret

A webhook secret is sent with each `POST /jobs` request in `webhook.secret`.
The service signs the terminal webhook for that job with it. There is no
service-wide webhook setting. The secret is kept in memory with the queued job
and is not written to the job store. The store keeps only `url_hash`, the
retry settings, and the `webhook_delivery` result. Your receiver verifies the
signature, for example with `openmed.service.signing.verify_request_signature`.

### Detection signals

- The receiver accepts a webhook with a valid signature for a job that you did
  not submit.
- The receiver's replay protection (`NonceCache`) reports reused nonces. Or it
  sees deliveries whose timestamps fall outside its window.
- The secret appears in a client config file, a log, or a ticket.

### Containment

1. Change the receiver so it accepts only the new secret. The service does not
   need a restart for this.
2. Stop sending the old secret in new `POST /jobs` requests.

### Rotation

Jobs that were queued before the change still sign with the old secret. To keep
them deliverable during a scheduled rotation, the receiver can accept either
secret for a short, fixed window. Try the new secret first:

```python
from openmed.service.signing import (
    InvalidSignatureError,
    verify_request_signature,
)


def verify_with_overlap(method, path, body, headers, *, secrets, nonce_cache):
    """Accept a delivery signed by any secret in ``secrets``.

    Failed attempts do not consume the nonce, so trying a second secret
    keeps replay protection intact. Timestamp and replay errors still raise.
    """
    for secret in secrets:
        try:
            return verify_request_signature(
                method,
                path,
                body,
                headers,
                secret=secret,
                nonce_cache=nonce_cache,
            )
        except InvalidSignatureError:
            continue
    raise InvalidSignatureError("Invalid request signature")
```

For a confirmed leak, do not use an overlap. Pass only the new secret.

### Recovery

- When a delivery fails, the job's `webhook_delivery` records it. Callers can
  still poll `GET /jobs/{job_id}` for the terminal status, so a rejected
  webhook does not lose the result.
- Treat webhooks that were accepted with the old secret during the exposure
  window as untrusted. Re-read their status with `GET /jobs/{job_id}`.

### Verification

1. Submit a synthetic job with the new secret. Confirm the receiver accepts the
   delivery. Confirm `webhook_delivery` shows success in `GET /jobs/{job_id}`.
2. Confirm the receiver rejects a synthetic delivery signed with the old
   secret.

### Keep out of tickets and logs

- Both secrets, the webhook URL, and receiver host names. The store keeps only
  `url_hash` for this reason.
- The text of `webhook_delivery` errors. It comes from the HTTP client and can
  name the receiver.
- Webhook bodies. They include job IDs, document IDs, span offsets and label
  counts.
