# REST API Recipes

Copy-paste recipes for the OpenMed REST service. Each primary request below is
a ready-to-run `curl` command paired with an equivalent Python
[`requests`](https://requests.readthedocs.io/) snippet. Additional variants may
be shown in one form; the real response shapes let callers wire up parsing and
error handling in one pass.

This page is the task-oriented companion to the endpoint reference in
[REST Service](rest-service.md). Use the reference for the full field-by-field
contract and configuration knobs; use this page to make a successful call fast.

All examples use **synthetic data only**. Never paste real patient text into a
shared terminal, shell history, or issue tracker.

Model-backed responses can echo the request text and detected entity text.
Treat request and response payloads as PHI in real deployments: do not put them
in application logs, telemetry, caches, or unencrypted artifacts. The examples
print fields only because every value shown here is synthetic.

## Start the service

Install the service extra and start Uvicorn:

```bash
uv pip install -e ".[hf,service]"
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

The standalone Python snippets below use `requests`, which is intentionally not
a server dependency. Install it explicitly in the client environment:

```bash
uv pip install requests
```

The first model-backed request downloads model weights unless they are already
cached. After download, inference stays local. For an air-gapped deployment,
pre-seed the cache and follow the
[local-only offline configuration](configuration.md#local-only-offline-mode).

Or run the packaged container with Docker Compose (maps port `8080`, persists the
Hugging Face cache in a named volume). See
[REST Service — Docker Compose](rest-service.md#docker-compose) for the full
setup:

```bash
docker compose up -d
```

The checked-in Compose mapping publishes port `8080` on every host interface.
For local-only use, prefer the loopback-bound Uvicorn command above or restrict
the published port to loopback with the host firewall or a Compose override.
Before allowing remote access, terminate TLS at a reverse proxy, enable
[REST authentication](serving/authentication.md), and restrict the trusted-host
allowlist. CORS is a browser policy, not authentication or transport security.

The examples below assume the service is reachable at
`http://127.0.0.1:8080`. Export it once so the `curl` snippets stay short:

```bash
export OPENMED_URL="http://127.0.0.1:8080"
```

The Python snippets share one base URL:

```python
import requests

BASE_URL = "http://127.0.0.1:8080"
```

By default the service only trusts loopback `Host` headers
(`localhost,127.0.0.1,[::1]`) and CORS is off. If you call it from another host
configure `OPENMED_SERVICE_TRUSTED_HOSTS`; for a browser front-end, also
configure `OPENMED_SERVICE_CORS_ORIGINS` — see
[Browser and Host Allowlists](rest-service.md#browser-and-host-allowlists).
These allowlists do not encrypt or authenticate traffic; remote deployments
still need HTTPS and authentication.

## Health check — `GET /health`

The cheapest call to confirm the service is up. It never loads a model.

```bash
curl --max-time 10 "$OPENMED_URL/health"
```

```python
response = requests.get(f"{BASE_URL}/health", timeout=10)
response.raise_for_status()
print(response.json())
```

Response:

```json
{
  "status": "ok",
  "service": "openmed-rest",
  "version": "1.9.1",
  "profile": "prod"
}
```

`GET /health` is the backward-compatible health alias. For orchestrators, use
`GET /livez` (process liveness) and `GET /readyz` (startup readiness). Before
startup completes, `/readyz` returns `503` with `error.code` set to `not_ready`.

## Analyze clinical text — `POST /analyze`

Run a medical NER model over free text. `model_name` defaults to
`disease_detection_superclinical`; `confidence_threshold` defaults to `0.0`.

```bash
curl -sS --max-time 310 -X POST "$OPENMED_URL/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient started imatinib for CML.",
    "model_name": "disease_detection_superclinical",
    "confidence_threshold": 0.5
  }'
```

```python
payload = {
    "text": "Patient started imatinib for CML.",
    "model_name": "disease_detection_superclinical",
    "confidence_threshold": 0.5,
}
response = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=310)
response.raise_for_status()
result = response.json()
for entity in result["entities"]:
    print(entity["label"], entity["text"], round(entity["confidence"], 3))
```

Representative response (same shape as
`analyze_text(..., output_format="dict")`; scores and timings vary by
hardware):

```json
{
  "text": "Patient started imatinib for CML.",
  "entities": [
    {
      "text": "CML",
      "label": "DISEASE",
      "confidence": 0.957,
      "start": 29,
      "end": 32,
      "metadata": {
        "sentence_index": 0,
        "sentence_text": "Patient started imatinib for CML.",
        "sentence_start": 0,
        "sentence_end": 33,
        "span_valid": true
      }
    }
  ],
  "model_name": "disease_detection_superclinical",
  "timestamp": "2026-07-11T16:58:55.987165",
  "processing_time": 1.527,
  "metadata": {
    "sentence_detection": true,
    "sentence_count": 1,
    "sentence_language": "en",
    "medical_tokenizer": true,
    "max_length": 512
  }
}
```

## Extract PII — `POST /pii/extract`

Detect personally identifiable information. Unless `model_name` is set, OpenMed
selects the recommended PII model for `lang`. The 22 supported PII language
codes: `am`, `ar`, `de`, `en`, `es`, `fr`, `he`, `hi`, `id`, `it`, `ja`, `ko`, `nl`,
`pt`, `ro`, `sw`, `te`, `th`, `tr`, `xh`, `zh`, and `zu`. Chinese currently uses the documented
multilingual default-model placeholder. `confidence_threshold` defaults to `0.5`.

```bash
curl -sS --max-time 310 -X POST "$OPENMED_URL/pii/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient Jordan Ramirez, MRN 4482910, called from 555-0147.",
    "lang": "en",
    "use_smart_merging": true
  }'
```

```python
payload = {
    "text": "Patient Jordan Ramirez, MRN 4482910, called from 555-0147.",
    "lang": "en",
    "use_smart_merging": True,
}
response = requests.post(f"{BASE_URL}/pii/extract", json=payload, timeout=310)
response.raise_for_status()
for entity in response.json()["entities"]:
    print(entity["label"], entity["start"], entity["end"], entity["text"])
```

Representative response (same shape as `extract_pii(...).to_dict()`; scores and
timings vary by hardware):

```json
{
  "text": "Patient Jordan Ramirez, MRN 4482910, called from 555-0147.",
  "entities": [
    {
      "text": "Jordan",
      "label": "first_name",
      "confidence": 0.999,
      "start": 8,
      "end": 14,
      "metadata": {"span_valid": true}
    },
    {
      "text": "Ramirez",
      "label": "last_name",
      "confidence": 0.999,
      "start": 15,
      "end": 22,
      "metadata": {"span_valid": true}
    },
    {
      "text": "MRN 4482910",
      "label": "medical_record_number",
      "confidence": 0.708,
      "start": 24,
      "end": 35,
      "metadata": {"span_valid": true}
    },
    {
      "text": "555-0147",
      "label": "phone_number",
      "confidence": 0.992,
      "start": 49,
      "end": 57,
      "metadata": {"span_valid": true}
    }
  ],
  "model_name": "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
  "timestamp": "2026-07-11T16:58:02.718948",
  "processing_time": 1.772,
  "metadata": {
    "sentence_detection": true,
    "sentence_count": 1,
    "sentence_language": "en",
    "medical_tokenizer": true,
    "max_length": 512,
    "clinical_protection": {
      "source": "openmed/core/data/clinical_protect_terms.txt",
      "version": "clinical-protect-terms-v1",
      "protected_term_count": 71,
      "checked_spans": 2,
      "suppressed_spans": 0,
      "enabled": true
    }
  }
}
```

## De-identify text — `POST /pii/deidentify`

Redact detected PII. `method` is one of `mask`, `remove`, `replace`, `hash`, or
`shift_dates` (default `mask`). `keep_mapping` defaults to `false` and is
intentionally omitted here. Enabling it returns original identifiers in a
placeholder-to-original mapping and should be reserved for controlled,
reversible workflows. `confidence_threshold` defaults to `0.7`.

```bash
curl -sS --max-time 310 -X POST "$OPENMED_URL/pii/deidentify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Call 555-0147 to confirm the appointment.",
    "method": "mask",
    "lang": "en"
  }'
```

```python
payload = {
    "text": "Call 555-0147 to confirm the appointment.",
    "method": "mask",
    "lang": "en",
}
response = requests.post(f"{BASE_URL}/pii/deidentify", json=payload, timeout=310)
response.raise_for_status()
result = response.json()
print(result["deidentified_text"])
print("redacted:", result["num_entities_redacted"])
```

Representative abridged response (`deidentify(...).to_dict()`; scores, timings,
and nested provenance metadata vary by runtime):

```json
{
  "original_text": "Call 555-0147 to confirm the appointment.",
  "deidentified_text": "Call [phone_number] to confirm the appointment.",
  "pii_entities": [
    {
      "text": "555-0147",
      "label": "phone_number",
      "entity_type": "phone_number",
      "start": 5,
      "end": 13,
      "confidence": 0.986,
      "redacted_text": "[phone_number]",
      "canonical_label": "PHONE",
      "sources": ["ml"],
      "evidence": {
        "raw_label": "phone_number",
        "language": "en",
        "model_id": "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
      },
      "threshold": 0.7,
      "action": "mask",
      "surrogate": "[phone_number]",
      "metadata": {"span_valid": true}
    }
  ],
  "method": "mask",
  "timestamp": "2026-07-11T16:58:43.445519",
  "num_entities_redacted": 1,
  "metadata": {
    "sentence_detection": true,
    "sentence_count": 1,
    "sentence_language": "en",
    "medical_tokenizer": true,
    "max_length": 512,
    "safety_sweep": {
      "source": "safety_sweep",
      "patterns_version": "safety-sweep-v1",
      "spans_added": 0
    }
  },
  "audit_report": null
}
```

To shift dates instead of masking, use `method: "shift_dates"` with an optional
`date_shift_days`:

```bash
curl -sS --max-time 310 -X POST "$OPENMED_URL/pii/deidentify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient Jordan Ramirez was admitted on 2026-01-02.",
    "method": "shift_dates",
    "date_shift_days": 30,
    "lang": "en"
  }'
```

The deprecated `shift_dates: true` boolean is still accepted as an alias for
`method: "shift_dates"`.

## Inspect loaded models — `GET /models/loaded`

Report the warm-pool cache, resident models, and idle-unload countdown. No model
is loaded by this call.

```bash
curl --max-time 10 "$OPENMED_URL/models/loaded"
```

```python
response = requests.get(f"{BASE_URL}/models/loaded", timeout=10)
response.raise_for_status()
state = response.json()
print("warm:", state["warm_models"])
print("resident cap:", state["max_resident_models"])
```

Response immediately after an unconfigured local service starts:

```json
{
  "default_keep_alive_seconds": null,
  "max_resident_models": null,
  "memory_budget_bytes": null,
  "resident_memory_bytes": 0,
  "pending_memory_bytes": 0,
  "memory_admission_wait_seconds": 0.05,
  "warm_models": [],
  "models": {}
}
```

After a model-backed request, `models` contains one entry per resolved model and
reports its cached resources, active requests, residency, idle-unload countdown,
and memory footprint. `warm_models` lists only the configured preload set.
`default_keep_alive_seconds`, `max_resident_models`, and `memory_budget_bytes`
are `null` when their optional env vars are unset.

## Unload a model — `POST /models/unload`

Free one inactive model, or all of them. If a model still has active requests,
the service leaves it loaded and reports the count.

Unload one model:

```bash
curl -sS --max-time 30 -X POST "$OPENMED_URL/models/unload" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "disease_detection_superclinical"}'
```

```python
payload = {"model_name": "disease_detection_superclinical"}
response = requests.post(f"{BASE_URL}/models/unload", json=payload, timeout=30)
response.raise_for_status()
print(response.json())
```

Representative response after the analyze recipe has loaded the default disease
model (released resource counts vary by backend):

```json
{
  "unloaded": true,
  "model_name": "OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M",
  "active_requests": 0,
  "loading": false,
  "released": {"models": 0, "tokenizers": 0, "pipelines": 1}
}
```

Unload every inactive model:

```bash
curl -sS --max-time 30 -X POST "$OPENMED_URL/models/unload" \
  -H "Content-Type: application/json" \
  -d '{"all": true}'
```

```python
response = requests.post(
    f"{BASE_URL}/models/unload", json={"all": True}, timeout=30
)
response.raise_for_status()
print(response.json())
```

Representative response after both an analysis pipeline and a PII pipeline were
cached (released resource counts vary by backend):

```json
{
  "unloaded": true,
  "released": {"models": 0, "tokenizers": 0, "pipelines": 2},
  "active_models": {}
}
```

Send `model_name` to unload one model or `all: true` to unload all inactive
models. Sending neither returns the validation error envelope described below.
Do not send both: the current schema treats `all: true` as the all-model
operation when both fields are present.

## Handling errors

After a request passes the configured host and CORS middleware, application
endpoint errors use one JSON envelope, so a single handler covers validation,
bad-request, timeout, and internal errors:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed",
    "details": [
      {
        "field": "body.text",
        "message": "Value error, Text must not be blank",
        "type": "value_error"
      }
    ],
    "request_id": "recipe-validation-error"
  }
}
```

Common `error.code` values include `validation_error`, `bad_request`, `timeout`,
`not_ready`, `rate_limited`, `backpressure`, `service_busy`,
`circuit_breaker_open`, and `internal_error`. Authentication and privacy-gateway
features add their own documented codes. `details` may be a list (validation
errors), an object (for example `{"timeout_seconds": 300}`), or `null`. Every
HTTP response carries an `X-Request-ID` header. Application-generated error
envelopes also echo it as `error.request_id` for correlating logs; middleware
rejections may provide only the response header.

Reproduce the validation envelope with a blank `text`:

```bash
curl -sS --max-time 60 -X POST "$OPENMED_URL/analyze" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: recipe-validation-error" \
  -d '{"text": "   "}'
```

```python
def analyze(text: str) -> dict:
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={"text": text},
        headers={"X-Request-ID": "recipe-validation-error"},
        timeout=60,
    )
    if response.status_code >= 400:
        error = response.json()["error"]
        request_id = response.headers.get("X-Request-ID")
        raise RuntimeError(
            f"{response.status_code} {error['code']}: {error['message']} "
            f"(request_id={request_id})"
        )
    return response.json()


analyze("   ")  # raises RuntimeError with code "validation_error"
```

## Typed Python client

The `service` extra ships a typed synchronous client,
`openmed.service.client.OpenMedClient`, built on `httpx`. It maps the analysis,
PII, privacy-gateway, and model-cache endpoints to methods and raises
`OpenMedAPIError` on any non-2xx response, so you skip the manual status checks
above:

```python
from openmed.service.client import OpenMedAPIError, OpenMedClient

with OpenMedClient("http://127.0.0.1:8080", timeout=310.0) as client:
    analysis = client.analyze(
        "Patient started imatinib for CML.",
        model_name="disease_detection_superclinical",
        confidence_threshold=0.5,
    )
    pii = client.extract_pii(
        "Patient Jordan Ramirez, MRN 4482910, called from 555-0147.",
        lang="en",
    )
    redacted = client.deidentify(
        "Patient Jordan Ramirez was admitted on 2026-01-02.",
        method="mask",
    )
    loaded = client.loaded_models()

    try:
        client.unload_model("disease_detection_superclinical")
    except OpenMedAPIError as exc:
        print(exc.status_code, exc.code, exc.message, exc.request_id)
```

`OpenMedAPIError` exposes the service error `code`, `message`, optional
`details`, HTTP `status_code`, and the `request_id` from the response header (or
the outgoing client request ID when the response has none). Use
`client.unload_all_models()` to drop every inactive model in one call.

The `310`-second model-call timeout is intentionally just above the production
profile's default `300`-second service deadline, so the client can receive the
service's structured timeout envelope. Raise both values together if your
deployment uses a longer server deadline.

## Related pages

- [REST Service](rest-service.md) — full endpoint reference, configuration env
  vars, allowlists, and the Docker/Compose run path.
- [REST Authentication](serving/authentication.md) — optional API-key and
  bearer-token enforcement in front of these endpoints.
- [Async REST Jobs & Webhooks](serving/async-jobs.md) — for large
  de-identification batches that should not hold a client connection open.
