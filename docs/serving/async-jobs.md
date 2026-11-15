# Async De-identification Jobs

`POST /jobs` accepts a multi-document de-identification batch and returns a job
id immediately. The service processes jobs on a bounded local worker pool and
persists only job metadata: status, progress, keyed document digests, label
counts, offsets, keyed span digests, and webhook delivery status. See
[Keyed Digests](#keyed-digests) for how digests are made and when they are
omitted.

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

## Capacity Limits

Each pending job keeps its raw documents in memory until a worker finishes it,
so submissions are admitted against fixed ceilings before any job metadata is
written. A pending job is one that is queued or running.

| Variable | Default | Bounds |
|---|---|---|
| `OPENMED_SERVICE_JOBS_MAX_PENDING` | `64` | Jobs queued or running at once |
| `OPENMED_SERVICE_JOBS_MAX_QUEUED_BYTES` | `67108864` (64 MiB) | UTF-8 bytes of document text held by all pending jobs |
| `OPENMED_SERVICE_JOBS_MAX_DOCUMENTS` | `1000` | Documents in one job |

Each value must be a positive integer; an invalid value stops the service at
startup. A rejected submission creates no job record and queues no work.

When the pending-job or queued-byte ceiling is reached, `POST /jobs` returns
`503` with a `Retry-After` header and a value-free error. Retry the same
request after the hinted delay:

```json
{
  "error": {
    "code": "job_queue_full",
    "message": "Job queue is full. Retry later.",
    "details": {"reason": "pending_jobs", "limit": 64, "retry_after_seconds": 5}
  }
}
```

`details.reason` is `pending_jobs` or `queued_text_bytes`.

A job that can never fit returns `422` instead, and retrying it unchanged will
not succeed:

- `job_too_many_documents` (`reason: documents_per_job`): the job has more
  documents than `OPENMED_SERVICE_JOBS_MAX_DOCUMENTS`. Split it into smaller
  jobs.
- `job_too_large` (`reason: job_text_bytes`): the job's text alone is larger
  than `OPENMED_SERVICE_JOBS_MAX_QUEUED_BYTES`.

Rejections carry only the code, reason, and configured limit. They never echo
document text or document ids.

When `OPENMED_SERVICE_METRICS_ENABLED` is on, `GET /metrics` exposes two
unlabelled aggregate gauges:

- `openmed_service_job_queue_depth`: jobs queued or running.
- `openmed_service_job_queue_text_bytes`: UTF-8 bytes of document text held by
  those jobs.

Capacity is returned when a job finishes, fails, or is cancelled at shutdown,
so both gauges return to `0` once the queue drains.

## Poll Status

```bash
curl -s http://127.0.0.1:8080/jobs/<job-id>
```

Status moves from `queued` to `running` and then to `done` or `failed`. A job
whose work never started when the service shut down is recorded as `cancelled`
with error code `cancelled_on_shutdown`; running and finished jobs keep their
own states. Status changes and terminal results are saved as soon as they
happen. Progress (`progress_percent`, counts, `label_histogram`, and `spans`)
is saved in coalesced checkpoints rather than after every document; see
[Progress checkpoints](#progress-checkpoints). Every record carries
`schema_version` (currently `3`) and a `digests` marker. Terminal records
include:

- `processed_count` and `failed_count`
- `label_histogram`
- `spans` with `document_id`, offsets, labels, confidence, and a keyed
  `text_digest`
- `documents` with ids, lengths, and a keyed `text_digest`
- `webhook` with a keyed `url_digest` instead of the callback URL
- `error` with a value-free `code`, the exception `type`, and a fixed message
- `payloads` with the opaque payload references and their lifecycle state
- `webhook_delivery`, which is `pending` from the terminal commit until the
  callback settles (see [Webhook Delivery Status](#webhook-delivery-status))

The job API does not return raw redacted text. Use synchronous
`POST /pii/deidentify` when a caller needs the redacted payload in the response.

When service authentication is enabled, a job is visible only to the principal
that submitted it and to principals holding `jobs:admin`; every other caller
receives `404`. See [Job Ownership](authentication.md#job-ownership) for the
required `OPENMED_SERVICE_DIGEST_KEY` setting and how older records are
handled.

## Webhook Payloads

Terminal jobs trigger a signed webhook when `webhook` is provided. The payload
contains no raw PHI and no unkeyed hashes:

```json
{
  "schema_version": 2,
  "event": "job.done",
  "job_id": "<job-id>",
  "status": "done",
  "progress_percent": 100.0,
  "document_count": 2,
  "processed_count": 2,
  "failed_count": 0,
  "label_histogram": {"NAME": 2, "PHONE": 1},
  "digests": {
    "mode": "keyed",
    "alg": "hmac-sha256",
    "key_id": "jobs-2026-11",
    "reason": null
  },
  "spans": [
    {
      "document_id": "note-1",
      "start": 8,
      "end": 20,
      "label": "NAME",
      "text_digest": {
        "alg": "hmac-sha256",
        "key_id": "jobs-2026-11",
        "value": "<64 hex characters>"
      },
      "confidence": 0.99
    }
  ],
  "documents": [
    {
      "id": "note-1",
      "length": 42,
      "text_digest": {
        "alg": "hmac-sha256",
        "key_id": "jobs-2026-11",
        "value": "<64 hex characters>"
      }
    }
  ],
  "error": null,
  "completed_at": "2026-07-01T10:00:00Z"
}
```

Webhook requests include:

- `X-OpenMed-Event`: `job.done` or `job.failed`
- `X-OpenMed-Delivery-Id`: opaque id shared by every retry of this delivery
- `X-OpenMed-Attempt`: attempt number, starting at `1`
- `X-OpenMed-Timestamp`: Unix timestamp
- `X-OpenMed-Nonce`: unique opaque request nonce
- `X-OpenMed-Webhook-Signature`: `v2,kid=<key id>,sha256=<hmac>`
- `X-OpenMed-Signature`: `sha256=<hmac>` (version 1, for older receivers)

The signatures use the shared
[HMAC request-signing scheme](request-signing.md). Version 2 binds the key id,
event, delivery id, and attempt number around the version 1 string, which
covers `POST`, the exact callback path and query, timestamp, nonce, and SHA-256
digest of the canonical JSON body. Each delivery attempt is signed
independently with a fresh nonce, so retries remain compatible with
receiver-side replay protection, and receivers can use
`openmed.service.webhook_replay.WebhookReceiver` to apply a delivery once.
Job webhooks are signed with the per-job `secret`; their key id is derived
from it. Non-2xx responses and transport errors are retried with exponential
backoff.

The body is built once, when the job reaches its terminal state, and sealed
into the job's terminal-event payload together with the delivery id. Delivery
and every retry, including one after a restart, read that immutable payload;
they never rebuild the body from mutable job state or mint a new delivery id.
`webhook_delivery` carries the fields described in
[Webhook Delivery Status](#webhook-delivery-status) plus `delivery_id`,
`signature_version`, `attempt_receipts` (attempt number, key id, signed time,
status code, and error code per attempt), and `payload_ref`, the opaque
reference of the terminal-event payload that was delivered. When the event
cannot be sealed or read back, the delivery never reaches the sender and
`error` keeps the controlled payload code (for example `payload_storage_full`)
instead of `sender_error`.

## Published Contracts

The service payloads that clients and webhook receivers depend on are published
as versioned JSON Schemas in the `openmed.core.schemas.json` package and
registered in the bundled fingerprint snapshot:

| Contract | Schema |
| --- | --- |
| `POST /jobs` and `GET /jobs/{job_id}` response | `service_job_record.schema.json` |
| Terminal webhook body | `service_webhook_payload.schema.json` |
| REST error envelope | `service_error_envelope.schema.json` |

Load and validate a response locally:

```python
from jsonschema import Draft202012Validator

from openmed.core.schemas import load_schema

record = client.get(f"/jobs/{job_id}").json()
Draft202012Validator(load_schema("service_job_record")).validate(record)
```

Each schema sets `additionalProperties: false` and lists every field the service
currently emits, so removing a required field or adding an undeclared field
fails validation instead of drifting silently. The payloads themselves carry no
`schema_version` field; the contract version is the registered schema version in
`openmed/core/schemas/json/schema-fingerprints.json`, which
`tests/unit/core/test_schemas_versioning.py` and
`tests/unit/service/test_contract_schemas.py` pin against live service output.

## Webhook Delivery Status

Webhooks are delivered on a separate bounded background executor, not on the
job worker. A job worker commits the terminal record with
`webhook_delivery.status` set to `pending` and immediately picks up the next
queued job, so a slow or failing receiver cannot stall other jobs.

`webhook_delivery` contains only controlled values:

```json
{
  "status": "failed",
  "success": false,
  "attempts": 3,
  "status_code": 500,
  "error": "http_status"
}
```

| `status` | Meaning |
| --- | --- |
| `pending` | Delivery is queued or in progress; `success` is `null`. |
| `delivered` | The receiver returned a 2xx response. |
| `failed` | All attempts failed. |
| `abandoned` | Delivery stopped at the wall-time ceiling or at shutdown. |

| `error` | Meaning |
| --- | --- |
| `http_status` | The last attempt received a non-2xx response (`status_code` is set). |
| `timeout` | The last attempt timed out. |
| `connect` | The last attempt could not connect. |
| `transport` | The last attempt failed with another transport error. |
| `deadline_exceeded` | The server-side delivery wall-time ceiling was reached. |
| `cancelled` | The service shut down before the delivery settled. |
| `signing_key_unavailable` | No usable signing key was available for an attempt. |
| `sender_error` | The delivery callable raised or returned an unexpected value. |

Callback URLs, exception text, and payload content are never stored in
`webhook_delivery`. `attempts` counts the HTTP attempts reported by the
delivery; it is `0` when a delivery is abandoned before it reports, for
example when shutdown reaches its drain bound while a request is in flight.

Records written before this change may contain a free-form `error` string and
no `status` field.

### Time Budgets and Shutdown

The per-request `max_attempts` and `backoff_seconds` settings are bounded by a
server-side ceiling on the total wall time of one delivery, measured from the
terminal commit and including retries, backoff waits, and time spent queued
behind other deliveries. Each attempt timeout is capped at the remaining
budget, and a backoff that would cross the ceiling ends the delivery as
`abandoned` with `deadline_exceeded`. Without the ceiling, `max_attempts: 10`
with `backoff_seconds: 60` would schedule 30,660 seconds of backoff.

`DeidentifyJobQueue.shutdown()` cancels backoff waits immediately, waits up to
the drain bound for in-flight deliveries to settle, and then records any
remaining delivery as `abandoned` with `cancelled`. Delivery threads are daemon
threads, so they do not keep the process alive after shutdown returns.

These limits are constructor parameters of `DeidentifyJobQueue`:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `webhook_workers` | `2` | Concurrent webhook deliveries. |
| `webhook_max_delivery_seconds` | `300.0` | Total wall-time ceiling per delivery. |
| `webhook_drain_seconds` | `2.0` | Shutdown wait for in-flight deliveries. |

A custom `webhook_sender` receives `cancel_event` and `max_total_seconds`
keyword arguments only when its signature accepts them (or `**kwargs`).
Senders with the original keyword set keep working, but cannot observe
cancellation or the ceiling until they settle; shutdown still returns within
the drain bound and records them as `abandoned`.

A custom `webhook_sender` passed to `DeidentifyJobQueue` is called with the
same keywords as `deliver_webhook`, including the sealed `delivery_id`.

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
  job reaches a terminal state, or (`cancelled`) when shutdown cancels the job.
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

## Keyed Digests

Span text, document text and the webhook URL are never stored or returned.
Each is referenced by an HMAC-SHA256 digest under the deployment digest key,
so a low-entropy identifier such as a date, phone number, postal code or
record number cannot be recovered by hashing candidate values. Every digest is
an object with three fields:

| Field | Meaning |
|---|---|
| `alg` | Always `hmac-sha256` |
| `key_id` | Public identifier of the key that made the digest |
| `value` | 64 lowercase hexadecimal characters |

Digests are domain-separated: span text, document text and the webhook URL
use the fixed domains `job-span-text`, `job-document-text` and
`job-webhook-url`, so a span digest never equals a document or URL digest of
the same text, nor the job owner digest. For one key a value always produces
the same digest, so a receiver can match repeated values across spans, jobs
and payloads by comparing `value` within one `key_id`. Digests from different
`key_id` values are not comparable. The key itself never appears in records,
payloads, logs or errors.

Configure the key with the same variables used for
[Job Ownership](authentication.md#job-ownership); one key serves both:

```bash
OPENMED_SERVICE_DIGEST_KEY="$(openssl rand -hex 32)" \
OPENMED_SERVICE_DIGEST_KEY_ID=jobs-2026-11 \
python -m openmed.service.logging --host 127.0.0.1 --port 8080
```

The record-level `digests` marker states how the record's digests were made:

| `mode` | `reason` | Meaning |
|---|---|---|
| `keyed` | `null` | Digests are present; `alg` and `key_id` name the key |
| `omitted` | `digest_key_unconfigured` | No key is configured outside the `dev` and `test` profiles, so every `text_digest` and `url_digest` is `null` |
| `omitted` | `legacy_digests_removed` | The record was written before keyed digests; its unkeyed hashes were removed |

Without `OPENMED_SERVICE_DIGEST_KEY`, the `dev` and `test` profiles digest
under a random per-process key whose `key_id` starts with `ephemeral-`, so
those digests change on every restart. Every other profile omits digests
instead of falling back to an unkeyed hash.

Rotating the key, by changing `OPENMED_SERVICE_DIGEST_KEY` and
`OPENMED_SERVICE_DIGEST_KEY_ID`, changes every digest for new jobs. Records
written before the rotation keep their old digests and old `key_id` until
their TTL expires; they are not re-keyed.

### Versions and Compatibility

Consumers detect the contract from `schema_version`:

| Surface | Before | Now |
|---|---|---|
| Job records (`POST /jobs`, `GET /jobs/{job_id}`, the local store) | `1` or `2`: unkeyed `text_hash` and `url_hash` strings (`sha256:<hex>`) | `3`: keyed `text_digest` and `url_digest` objects plus the `digests` marker |
| Terminal webhook body | no `schema_version` field (version 1): unkeyed `text_hash` strings | `2`: keyed `text_digest` objects plus the `digests` marker |

The service never emits `text_hash` or `url_hash` for jobs, because doing so
would keep the enumerable value available. Receivers that read `text_hash`
must switch to `text_digest` and treat a `null` digest as "not available".

The local store reads files written with schema versions 1, 2 and 3 for the
whole 4.x series and always writes version 3. When it loads an older file it
removes every unkeyed `text_hash` and `url_hash`, sets the matching
`text_digest` or `url_digest` to `null`, marks the record
`legacy_digests_removed` and rewrites the file at once, so the unkeyed hashes
do not stay on disk. Unkeyed hashes cannot be re-keyed because the source text
is never stored. Offsets, labels, counts and ownership are kept. Migrated
records expire on their normal TTL.

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
`OPENMED_SERVICE_JOBS_WORKERS` bounds concurrent job execution; see
[Capacity Limits](#capacity-limits) for the submission ceilings. The local
backend is intended for one service process and does not provide multi-node
distribution.

## Progress Checkpoints

Each progress checkpoint rewrites the whole local store, so the service limits
how often a running job writes one. A checkpoint is written only when both
conditions hold:

- at least 1 second has passed since the job's previous progress write, and
- at least 1% of the job's documents (rounded up to a whole document) have
  finished, successfully or not, since the previous checkpoint.

A job therefore writes at most 100 progress checkpoints regardless of its
document count, plus one write each for submission, the `running` transition,
the terminal result, and the consumed input payload, and one more for webhook
delivery status when a webhook is configured. The bytes a job writes grow linearly with its document
count. The last document's progress is saved with the terminal result rather
than as a separate checkpoint.

`GET /jobs/{job_id}` returns the most recently saved record, so polled
progress can trail the worker by up to one checkpoint. Counts, label
histograms, and spans in a response always describe the same set of
documents, and `done` or `failed` records always carry the complete result.

Every write goes to a temporary file in the store directory that then
atomically replaces the store. If a write fails, the temporary file is removed,
the store keeps its previous contents, and the affected in-memory records
are restored to match.

Payload retention is configured separately:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENMED_SERVICE_JOBS_PAYLOAD_MODE` | `memory`, or `encrypted` when a key provider is passed | `memory` or `encrypted`. `encrypted` without a key provider refuses to start. |
| `OPENMED_SERVICE_JOBS_PAYLOAD_DIR` | `~/.cache/openmed/service-job-payloads` | Directory for encrypted payload files, created with mode `0700`. |
| `OPENMED_SERVICE_JOBS_PAYLOAD_RETENTION_SECONDS` | `86400` | Maximum payload retention; also the default when a job requests none. |

Payload retention does not make jobs restart-safe by itself: queued work is
still process-local, and a restart does not resume interrupted jobs.
