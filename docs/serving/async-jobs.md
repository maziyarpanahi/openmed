# Async De-identification Jobs

`POST /jobs` accepts a multi-document de-identification batch and returns a job
id immediately. The service processes jobs on a bounded local worker pool and
persists only job metadata: status, progress, document hashes, label counts,
offsets, span hashes, and webhook delivery status.

Raw submitted text is not written to the local job store. It remains in memory
only until a worker finishes the job.

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

Status moves from `queued` to `running` and then to `done` or `failed`.
Progress is updated after each document. Terminal records include:

- `processed_count` and `failed_count`
- `label_histogram`
- `spans` with `document_id`, offsets, labels, confidence, and text hashes
- `documents` with ids, lengths, and document hashes
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
