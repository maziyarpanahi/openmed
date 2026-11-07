# Async De-identification Jobs

`POST /jobs` accepts a multi-document de-identification batch and returns a job
id immediately. The service processes jobs on a bounded local worker pool and
persists only job metadata: status, progress, keyed document digests, label
counts, offsets, keyed span digests, and webhook delivery status. See
[Keyed Digests](#keyed-digests) for how digests are made and when they are
omitted.

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
Progress is updated after each document. Every record carries
`schema_version` (currently `3`) and a `digests` marker. Terminal records
include:

- `processed_count` and `failed_count`
- `label_histogram`
- `spans` with `document_id`, offsets, labels, confidence, and a keyed
  `text_digest`
- `documents` with ids, lengths, and a keyed `text_digest`
- `webhook` with a keyed `url_digest` instead of the callback URL
- `webhook_delivery` after callback attempts complete

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
- `X-OpenMed-Timestamp`: Unix timestamp
- `X-OpenMed-Nonce`: unique opaque request nonce
- `X-OpenMed-Signature`: `sha256=<hmac>`

The signature uses the shared
[HMAC request-signing scheme](request-signing.md), covering `POST`, the exact
callback path and query, timestamp, nonce, and SHA-256 digest of the canonical
JSON body. Each delivery attempt is signed independently with a fresh nonce, so
retries remain compatible with receiver-side replay protection. Non-2xx
responses and transport errors are retried with exponential backoff.

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
`OPENMED_SERVICE_JOBS_WORKERS` bounds concurrent job execution. The local
backend is intended for one service process and does not provide multi-node
distribution.
