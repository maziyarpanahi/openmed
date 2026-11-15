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
[Progress checkpoints](#progress-checkpoints). Terminal records include:

- `processed_count` and `failed_count`
- `label_histogram`
- `spans` with `document_id`, offsets, labels, confidence, and text hashes
- `documents` with ids, lengths, and document hashes
- `error` with a value-free `code`, the exception `type`, and a fixed message
- `payloads` with the opaque payload references and their lifecycle state
- `webhook_delivery`, which is `pending` from the terminal commit until the
  callback settles (see [Webhook Delivery Status](#webhook-delivery-status))

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
`webhook_delivery` carries the fields described in
[Webhook Delivery Status](#webhook-delivery-status) plus `payload_ref`, the
opaque reference of the terminal-event payload that was delivered. When the
event cannot be sealed or read back, the delivery never reaches the sender and
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
