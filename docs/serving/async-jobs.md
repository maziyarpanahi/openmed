# Async De-identification Jobs

`POST /jobs` accepts a multi-document de-identification batch and returns a job
id immediately. The service processes jobs on a bounded local worker pool and
persists only job metadata: status, progress, document hashes, label counts,
offsets, span hashes, and webhook delivery status.

Raw submitted text is never written to the local job store. The submission is
snapshotted once into an immutable input payload, and the terminal webhook event
is sealed once into an immutable terminal-event payload. Both live behind opaque
payload references with an explicit, bounded retention deadline; see
[Payload Protection and Retention](#payload-protection-and-retention).

## Submit a Job

```bash
curl -s http://127.0.0.1:8080/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {"id": "note-1", "text": "Patient Maria Garcia arrived at 09:00."},
      {"id": "note-2", "text": "Call Maria Garcia at 555-1212."}
    ],
    "method": "mask",
    "webhook": {
      "url": "https://pipeline.example.com/openmed/jobs",
      "secret": "replace-with-shared-secret",
      "max_attempts": 3,
      "backoff_seconds": 0.5
    }
  }'
```

The response uses HTTP `202` and includes `id`, `status`, `progress_percent`,
`document_count`, and `status_url`.

To make a retried submission safe, send an `Idempotency-Key` header with a
random value (16-128 letters, digits, or `._~:-`; a UUID works). A repeat with
the same key and the same request returns the existing job with HTTP `202` and
`Idempotent-Replayed: true` instead of creating a second one. Reusing the key
for a different request returns HTTP `409` with error code
`idempotency_key_conflict`. The key is bound to the authenticated principal and
stored only as a scoped digest; it is remembered for as long as the job record
is retained.

## Poll Status

```bash
curl -s http://127.0.0.1:8080/jobs/<job-id>
```

Status moves from `queued` to `running` and then to `done` or `failed`. A job
can also end `cancelled`, or wait in `quarantined` until an operator replays it
(see [Durability and Recovery](#durability-and-recovery)). A retried job goes
back to `queued` until its backoff elapses. With memory-only payloads, a job
whose work never started when the service shut down is recorded as `cancelled`
with error code `cancelled_on_shutdown`; running and finished jobs keep their
own states. Progress is updated after each document. Every write increments
`state_version`, and `recovery` carries the value-free recovery metadata.
Terminal records include:

- `processed_count` and `failed_count`
- `label_histogram`
- `spans` with `document_id`, offsets, labels, confidence, and text hashes
- `documents` with ids, lengths, and document hashes
- `error` with a value-free `code`, the exception `type`, and a fixed message
- `payloads` with the opaque payload references and their lifecycle state
- `webhook_delivery` after callback attempts complete

The job API does not return raw redacted text. Use synchronous
`POST /pii/deidentify` when a caller needs the redacted payload in the response.

## Webhook Payloads

Terminal jobs trigger a signed webhook when `webhook` is provided. The payload
contains no raw PHI:

```json
{
  "event": "job.done",
  "job_id": "<job-id>",
  "status": "done",
  "progress_percent": 100.0,
  "document_count": 2,
  "processed_count": 2,
  "failed_count": 0,
  "label_histogram": {"NAME": 2, "PHONE": 1},
  "spans": [
    {
      "document_id": "note-1",
      "start": 8,
      "end": 20,
      "label": "NAME",
      "text_hash": "sha256:...",
      "confidence": 0.99
    }
  ],
  "documents": [
    {"id": "note-1", "length": 42, "text_hash": "sha256:..."}
  ],
  "error": null,
  "completed_at": "2026-07-01T10:00:00Z"
}
```

Webhook requests include:

- `X-OpenMed-Event`: `job.done` or `job.failed`
- `X-OpenMed-Timestamp`: Unix timestamp
- `X-OpenMed-Nonce`: unique opaque request nonce
- `X-OpenMed-Signature`: `sha256=<hmac>`

The signature uses the shared
[HMAC request-signing scheme](request-signing.md), covering `POST`, the exact
callback path and query, timestamp, nonce, and SHA-256 digest of the canonical
JSON body. Each delivery attempt is signed independently with a fresh nonce, so
retries remain compatible with receiver-side replay protection. Non-2xx
responses and transport errors are retried with exponential backoff.

The body is built once, when the job reaches its terminal state, and sealed
into the job's terminal-event payload. Delivery and every retry read that
immutable payload; they never rebuild the body from mutable job state.
`webhook_delivery` records `success`, `attempts`, `status_code`, a value-free
`error` (`HTTP <status>`, `timeout`, `transport_error`,
`webhook_sender_failed`, or a payload failure code), and `payload_ref`, the
opaque reference of the terminal-event payload that was delivered.

## Payload Protection and Retention

Every job record carries a `payloads` object:

```json
{
  "schema_version": 1,
  "mode": "encrypted",
  "retention_seconds": 3600,
  "input": {
    "ref": "jp_<32 hex characters>",
    "schema_version": 1,
    "state": "deleted",
    "created_at": "2026-07-01T10:00:00Z",
    "expires_at": "2026-07-01T11:00:00Z",
    "digest": "sha256:...",
    "reason": "consumed"
  },
  "terminal_event": null
}
```

References are random and opaque. Records never expose storage paths, key
material, or payload content. `digest` is the SHA-256 digest of the stored
ciphertext (it is `null` in memory-only mode), never a digest of the content.
`state` is `retained`, `deleted`, or `expired`, with a controlled `reason`.

Lifecycle:

- The input payload is created at submission and deleted (`consumed`) when the
  job reaches a terminal state, or (`cancelled`) when the job is cancelled. A
  quarantined job keeps its input until a replay or the retention deadline.
- The terminal-event payload holds the webhook body plus its delivery target.
  It is deleted after a successful delivery (`delivered`). In encrypted mode a
  failed delivery keeps it until its deadline so a later retry can reference
  it; memory-only mode drops it (`delivery_failed`).
- Each payload expires `retention_seconds` after it was created. Expired and
  deleted payloads cannot be reopened through a stale job or delivery record,
  and a payload in use by a delivery is not removed until the delivery ends.

Two modes are available:

- **Memory only** (default). Payloads stay in process memory and nothing is
  written to disk. Payloads do not survive a restart.
- **Encrypted.** Payloads are retained in a local directory, sealed with the
  versioned AEAD envelope (`openmed.core.aead_envelope`) under a caller-owned
  key provider. Each envelope is bound to its reference, job, payload kind,
  schema version, and retention deadline, so a payload copied to another job or
  given a longer deadline fails authentication. Temporary files hold the same
  ciphertext and are published with an atomic rename. At startup, crash
  remnants, expired payloads, and payloads no job record still references are
  deleted. Encrypted mode needs the `service` (or `integrity`) extra.

OpenMed never stores or generates the key. Pass the provider when you build the
app, for example with the bundled process-local provider or your own custody
adapter:

```python
from openmed.core.key_provider import InMemoryKeyProvider, KeyState
from openmed.service.app import create_app
from openmed.service.job_payloads import JOB_PAYLOAD_KEY_PURPOSE

provider = InMemoryKeyProvider()
provider.add_key(
    "jobs-key-1",
    key_material_from_your_custody_system,  # exactly 32 bytes
    purpose=JOB_PAYLOAD_KEY_PURPOSE,
    state=KeyState.ACTIVE,
)
app = create_app(job_payload_key_provider=provider)
```

A missing, retired, revoked, or wrong key, a damaged envelope, a full disk, or
an interrupted write fails closed with a controlled code (for example
`envelope_authentication_failed`, `key_unavailable`, `payload_damaged`,
`payload_storage_full`, `payload_write_failed`) and leaves the last verified
payload untouched. A job whose input cannot be read fails with that code in
`error.code`.

Callers may request a shorter retention with `payload_retention_seconds` in the
`POST /jobs` body. A value above the service maximum is rejected with HTTP 400.

Each lifecycle transition (`created`, `read`, `read_refused`, `expired`,
`deleted`, `cleanup_failed`) emits a value-free receipt holding only the opaque
reference, payload kind, ciphertext digest, timestamp, and a controlled reason
code. `DeidentifyJobQueue.payloads.lifecycle_evidence()` returns recent
receipts and per-event counts for audit evidence.

## Durability and Recovery

The queue is durable: the input payload reference, the request digest, and the
idempotency key digest are written to the job store before a job is handed to
a worker. The recovery records reuse the Journey ingestion contracts (lease,
checkpoint, retry, cancellation, quarantine result, and quarantine promotion
from `openmed.interop.ingest.contracts`), adapted in
`openmed.service.job_recovery`.

- **Leases.** A worker claims a job with an expiring lease
  (`OPENMED_SERVICE_JOBS_LEASE_SECONDS`, default 60). Each claim increments the
  lease epoch, and every progress, retry, quarantine, and terminal write checks
  that epoch, so two workers can never hold a valid lease for the same epoch
  and a worker that lost its lease cannot commit. Workers renew the lease at
  every checkpoint, and the periodic reconcile pass renews leases held by live
  workers.
- **Checkpoints.** Each document's result is committed together with a
  checkpoint and a chained output digest. A job that is resumed after a crash
  or a lease reclaim starts after the last committed document, so committed
  results are never duplicated.
- **Terminal state and webhook intent.** The terminal state, the reference to
  the sealed terminal-event payload, and a pending webhook intent are committed
  in one write. A pending intent left by a crash is delivered again once its
  delivery claim expires, from the same immutable payload. Delivery is
  at-least-once: a crash after the HTTP request but before its result is
  recorded can repeat a webhook with an identical body.
- **Retries and backoff.** Transient, resource, contention, and dependency
  failures release the job for a retry with capped exponential backoff
  (`OPENMED_SERVICE_JOBS_RETRY_BASE_SECONDS`, default 5, up to
  `OPENMED_SERVICE_JOBS_RETRY_MAX_SECONDS`, default 300). Other document errors
  are recorded as failed documents, as before.
- **Poison quarantine.** A job that fails or loses its lease on
  `OPENMED_SERVICE_JOBS_MAX_ATTEMPTS` attempts (default 3), or whose input
  payload is damaged, fails authentication, or does not match the submission,
  moves to `quarantined` with a value-free reason code and keeps its input
  payload. It never runs again on its own.
- **Replay.** `DeidentifyJobQueue.replay(job_id, JobReplayAuthorization(...))`
  re-queues a quarantined job as a new, linked execution with a fresh attempt
  budget. The authorization carries only a reviewer digest, an evidence digest,
  and a reason code; an optional `replay_authorizer` hook can enforce your own
  policy. The quarantine result and every earlier transition stay in the
  record. The service ships no replay route: expose replay to remote callers
  only behind your own authorization.
- **Cancellation.** `DeidentifyJobQueue.cancel(job_id)` is committed before
  anything else and fences a running worker. A cancelled job never returns to
  `queued`.
- **Startup reconcile.** When the queue starts, and every
  `OPENMED_SERVICE_JOBS_RECONCILE_SECONDS` (default 5; `0` disables the
  background pass), it reclaims expired leases, resumes queued work whose
  backoff has elapsed, quarantines jobs that exhausted their attempts, fails
  queued jobs whose input expired or (in memory-only mode) did not survive the
  restart, and re-drives expired webhook intents. A record whose recovery state
  is corrupt or ambiguous fails closed with `recovery_state_invalid` and never
  becomes a success.
- **Shutdown.** With encrypted payloads, jobs that never started stay `queued`
  and resume at the next start, and an interrupted running job does not use up
  an attempt. With memory-only payloads, they are `cancelled` as described
  above, because their input cannot survive the process.

Restart-safe resume needs encrypted payload retention; memory-only payloads
are lost with the process. Recovery metadata never contains payload content,
worker identity, lease identifiers, storage paths, or secrets. Append-only
decision receipts for quarantine, lease reclaim, cancellation, recovery
failure, and replay are available from
`DeidentifyJobQueue.decision_receipts()`, and
`DeidentifyJobQueue.recovery_evidence()` returns value-free counts and receipts
for audit evidence.

## Local Store Configuration

The default metadata store is
`~/.cache/openmed/service-jobs.json`. Override it for containers or tests:

```bash
OPENMED_SERVICE_JOBS_STORE_PATH=/var/lib/openmed/jobs.json \
OPENMED_SERVICE_JOBS_TTL_SECONDS=86400 \
OPENMED_SERVICE_JOBS_WORKERS=2 \
python -m openmed.service.logging --host 0.0.0.0 --port 8080
```

`OPENMED_SERVICE_JOBS_TTL_SECONDS` controls terminal metadata cleanup.
`OPENMED_SERVICE_JOBS_WORKERS` bounds concurrent job execution. The local
backend is intended for one service process and does not provide multi-node
distribution.

Payload retention is configured separately:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENMED_SERVICE_JOBS_PAYLOAD_MODE` | `memory`, or `encrypted` when a key provider is passed | `memory` or `encrypted`. `encrypted` without a key provider refuses to start. |
| `OPENMED_SERVICE_JOBS_PAYLOAD_DIR` | `~/.cache/openmed/service-job-payloads` | Directory for encrypted payload files, created with mode `0700`. |
| `OPENMED_SERVICE_JOBS_PAYLOAD_RETENTION_SECONDS` | `86400` | Maximum payload retention; also the default when a job requests none. |

Recovery is configured with these variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENMED_SERVICE_JOBS_LEASE_SECONDS` | `60` | Lease duration; a crashed worker's job is reclaimed after it. |
| `OPENMED_SERVICE_JOBS_MAX_ATTEMPTS` | `3` | Attempts per execution before poison quarantine. |
| `OPENMED_SERVICE_JOBS_RETRY_BASE_SECONDS` | `5` | Backoff before the second attempt; it doubles per attempt. |
| `OPENMED_SERVICE_JOBS_RETRY_MAX_SECONDS` | `300` | Backoff ceiling. |
| `OPENMED_SERVICE_JOBS_RECONCILE_SECONDS` | `5` | Interval of the background reconcile pass; `0` disables it. |

Jobs resume across restarts only with encrypted payload retention. Metadata
writes replace the store file atomically, so they survive a process crash. The
store serves one service process; it does not coordinate writers across
processes.
