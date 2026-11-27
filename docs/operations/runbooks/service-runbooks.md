# Service incident runbooks

These runbooks cover four incidents for a self-hosted OpenMed REST service and
its Journey deployment profile:

| Incident | Runbook |
| --- | --- |
| Leaked API key, JWKS signing material or job webhook secret | [Credential compromise](credential-compromise.md) |
| Compromised mTLS client certificate | [Client certificate compromise](client-certificate-compromise.md) |
| Full disk under the model cache, async job store or Journey artifacts | [Disk exhaustion](disk-exhaustion.md) |
| Damaged async job store or Journey store | [Job and Journey store corruption](store-corruption.md) |

Each runbook gives detection signals, containment, rotation or recovery,
verification, and what must stay out of tickets and logs. They describe the
settings, commands and reason codes that exist in the current release. Steps
that need recovery tooling that has not shipped yet are marked **pending**. Use
the [breach notification runbook](../../compliance/breach-notification-runbook.md)
if you suspect PHI was exposed, and the general
[key management guide](../../security/key-management.md) for audit and
surrogate keys.

All examples use synthetic values. Replace `<placeholders>` with values from
your own secret manager or deployment. Do not paste real values into a ticket.

## How settings take effect

The REST service reads authentication, mTLS, and job-store settings once, when
`openmed.service.app.create_app` builds the application. To change an API key,
JWKS, CA bundle, principal mapping or job-store path:

1. Update the value in your secret store, then in the process environment.
   With Helm, add them as Helm `extraEnv` entries that reference a Kubernetes
   Secret.
2. Restart the service. `OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS` bounds how
   long in-flight requests can finish during shutdown.
3. Run the verification steps in the runbook.

Restarts drop work that exists only in memory. Queued and running async jobs
keep their submitted text in memory only, so a restart loses them. Their stored
metadata stays at its last state. With more than one replica, restart them one
at a time so the service stays available.

## Rules for tickets, logs and chat

These rules apply to every step in every runbook:

- Never paste API keys, bearer tokens, JWKS documents, HMAC or webhook secrets,
  private keys, certificates in PEM form, DSNs or database passwords. Refer to a
  credential by its configured `id`, its JWKS `kid`, or a certificate SHA-256
  fingerprint. Only do this when the label contains no personal name.
- Never attach the job store file, Journey database dumps, artifact blobs, model
  inputs, request or response bodies, or webhook payloads. These can contain
  document IDs, span offsets, label counts and text hashes linked to patients.
- Record reason codes, HTTP status codes, counts, timestamps, sizes, and
  SHA-256 digests of quarantined files. Do not record contents.
- Do not record private paths, host names of webhook receivers, or client IP
  addresses unless your incident process requires it and stores it with
  restricted access.
- Do not turn on debug logging, shell tracing, or environment dumps to
  investigate. These can print secrets. Never print a secret in a shell. Use
  files with mode `0600` or a secret manager to pass values.
- Keep quarantined files on encrypted storage that only the incident owners
  can read. Destroy them according to your retention policy.

## Reason codes

The service and Journey tools return these controlled codes. Their output never
includes the input value. Use the code to choose a runbook.

| Code | Where it appears | Runbook |
| --- | --- | --- |
| `invalid_credentials` | REST `401` error envelope for an unknown API key or a rejected JWT | [Credential compromise](credential-compromise.md) |
| `authentication_required` | REST `401` when no credential was sent | [Credential compromise](credential-compromise.md) |
| `auth_rate_limited` | REST `429` after repeated failed authentication | [Credential compromise](credential-compromise.md) |
| `forbidden` | REST `403` when a valid principal lacks a route scope | [Client certificate compromise](client-certificate-compromise.md) |
| `mtls_certificate_required` | REST `401` when no verified client certificate arrived | [Client certificate compromise](client-certificate-compromise.md) |
| `mtls_certificate_invalid` | REST `401` when the chain does not verify against the CA bundle | [Client certificate compromise](client-certificate-compromise.md) |
| `not_ready` | REST `503` from `GET /readyz` or during shutdown | [Disk exhaustion](disk-exhaustion.md) |
| `internal_error` | REST `500`, for example when the job store cannot be written | [Disk exhaustion](disk-exhaustion.md) |
| `artifact_write_failed` | Journey probe, worker readiness, or artifact store write | [Disk exhaustion](disk-exhaustion.md) |
| `artifact_store_unavailable` | Journey probe when the artifact root cannot be opened | [Disk exhaustion](disk-exhaustion.md) |
| `artifact_read_failed` | Journey artifact read or probe read-back failure | [Job and Journey store corruption](store-corruption.md) |
| `artifact_integrity_failed` | Journey artifact read whose bytes no longer match their digest | [Job and Journey store corruption](store-corruption.md) |
| `artifact_verify_failed` | Journey artifact write that failed its read-back digest check | [Disk exhaustion](disk-exhaustion.md) |
| `transaction_failed` | Journey metadata transaction that rolled back | [Disk exhaustion](disk-exhaustion.md) |
| `commit_status_unknown` | Journey metadata commit whose outcome cannot be proven | [Job and Journey store corruption](store-corruption.md) |
| `postgres_connect_failed` | Journey store connection failure | [Job and Journey store corruption](store-corruption.md) |
| `store_open_failed` | Journey store failed to open or migrate | [Job and Journey store corruption](store-corruption.md) |
| `schema_unsupported` | Journey schema is newer than this release, or a migration checksum drifted | [Job and Journey store corruption](store-corruption.md) |
| `sqlite_integrity_failed` | Local SQLite Journey store failed its structure check | [Job and Journey store corruption](store-corruption.md) |
| `foreign_key_integrity_failed` | Journey store has broken or unvalidated relationships | [Job and Journey store corruption](store-corruption.md) |
| `stored_payload_invalid` | Journey store row payload cannot be parsed | [Job and Journey store corruption](store-corruption.md) |
| `payload_hash_mismatch` | Journey store row no longer matches its stored digest | [Job and Journey store corruption](store-corruption.md) |

The async job store does not return a reason code when it refuses to start. The
service stops at startup with one of the fixed messages listed in
[Job and Journey store corruption](store-corruption.md#async-job-store).

## Pending capabilities

The current release does not include these capabilities. The runbooks give the
manual steps that work today and do not depend on them.

| Pending capability | Tracking |
| --- | --- |
| Reload API-key and JWT verifier sets without a restart, with an overlap window | Pending in #3699 |
| Reload TLS and mTLS credentials without dropping in-flight work | Pending in #3580 |
| Durable async jobs with leases, checkpoints, quarantine and replay | Pending in #3574 |
| Transactional model download, verification, activation and rollback | Pending in #3575 |
| Encrypted, integrity-checked backup and restore of service state | Pending in #3577 |
| Bounded no-PHI diagnostic and recovery bundle | Pending in #3600 |
| Local operator CLI for inspect, plan, verify and apply recovery actions | Pending in #3703 |
| Release recovery drill that exercises these runbooks | Pending in #3603 |

Until these ship, there is no read-only store inspection command, no bulk
artifact sweep, no job replay, and no credential hot reload. Do not describe any
of them as available in an incident record.

## Related documentation

- [REST authentication](../../serving/authentication.md)
- [Mutual TLS client authentication](../../serving/mtls.md)
- [Async REST jobs and webhooks](../../serving/async-jobs.md)
- [HMAC request signing](../../serving/request-signing.md)
- [Self-hosted Compose deployment](../../deploy/self-hosted-compose.md)
- [Helm deployment](../../deploy/helm.md)
- [Durable local Journey storage](../../structured/local-journey-store.md)
- [PostgreSQL and object-store Journey storage](../../structured/postgres-object-journey-store.md)
- [Secret handling](../../security/secret-handling.md)
