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

Status moves from `queued` to `running` and then to `done` or `failed`.
Progress is updated after each document. Terminal records include:

- `processed_count` and `failed_count`
- `label_histogram`
- `spans` with `document_id`, offsets, labels, confidence, and text hashes
- `documents` with ids, lengths, and document hashes
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
