# REST API Recipes

Copy-paste recipes for the OpenMed REST service. Every request below is a
ready-to-run `curl` one-liner paired with an equivalent Python
[`requests`](https://requests.readthedocs.io/) snippet, plus the real response
shape so callers can wire up parsing and error handling in one pass.

This page is the task-oriented companion to the endpoint reference in
[REST Service](rest-service.md). Use the reference for the full field-by-field
contract and configuration knobs; use this page to make a successful call fast.

All examples use **synthetic data only**. Never paste real patient text into a
shared terminal, shell history, or issue tracker.

## Start the service

Install the service extra and start Uvicorn:

```bash
uv pip install -e ".[hf,service]"
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

Or run the packaged container with Docker Compose (maps port `8080`, persists the
Hugging Face cache in a named volume). See
[REST Service — Docker Compose](rest-service.md#docker-compose) for the full
setup:

```bash
docker compose up -d
```

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
or a browser front-end, configure `OPENMED_SERVICE_TRUSTED_HOSTS` and
`OPENMED_SERVICE_CORS_ORIGINS` first — see
[Browser and Host Allowlists](rest-service.md#browser-and-host-allowlists).

## Health check — `GET /health`

The cheapest call to confirm the service is up. It never loads a model.

```bash
curl "$OPENMED_URL/health"
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
  "version": "1.7.0",
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
curl -sS -X POST "$OPENMED_URL/analyze" \
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
response = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=60)
response.raise_for_status()
result = response.json()
for entity in result["entities"]:
    print(entity["label"], entity["text"], round(entity["confidence"], 3))
```

Response (same shape as `analyze_text(..., output_format="dict")`):

```json
{
  "text": "Patient started imatinib for CML.",
  "entities": [
    {
      "text": "imatinib",
      "label": "CHEM",
      "confidence": 0.994,
      "start": 16,
      "end": 24,
      "metadata": {}
    },
    {
      "text": "CML",
      "label": "DISEASE",
      "confidence": 0.981,
      "start": 29,
      "end": 32,
      "metadata": {}
    }
  ],
  "model_name": "disease_detection_superclinical",
  "timestamp": "2026-01-02T09:30:00",
  "processing_time": 0.042,
  "metadata": {"sentence_detection": true}
}
```

## Extract PII — `POST /pii/extract`

Detect personally identifiable information. `model_name` defaults to the small
multilingual PII model; set `lang` to one of the supported ISO codes (`en`,
`fr`, `de`, `it`, `es`, `nl`, `hi`, `te`, `pt`, `ar`, `he`, `ja`, `tr`, `id`,
`th`). `confidence_threshold` defaults to `0.5`.

```bash
curl -sS -X POST "$OPENMED_URL/pii/extract" \
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
response = requests.post(f"{BASE_URL}/pii/extract", json=payload, timeout=60)
response.raise_for_status()
for entity in response.json()["entities"]:
    print(entity["label"], entity["start"], entity["end"], entity["text"])
```

Response (same shape as `extract_pii(...).to_dict()`):

```json
{
  "text": "Patient Jordan Ramirez, MRN 4482910, called from 555-0147.",
  "entities": [
    {
      "text": "Jordan Ramirez",
      "label": "NAME",
      "confidence": 0.987,
      "start": 8,
      "end": 22,
      "metadata": {}
    },
    {
      "text": "4482910",
      "label": "ID",
      "confidence": 0.973,
      "start": 28,
      "end": 35,
      "metadata": {}
    },
    {
      "text": "555-0147",
      "label": "PHONE",
      "confidence": 0.965,
      "start": 49,
      "end": 57,
      "metadata": {}
    }
  ],
  "model_name": "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
  "timestamp": "2026-01-02T09:30:00",
  "processing_time": 0.031,
  "metadata": {"lang": "en"}
}
```

## De-identify text — `POST /pii/deidentify`

Redact detected PII. `method` is one of `mask`, `remove`, `replace`, `hash`, or
`shift_dates` (default `mask`). Set `keep_mapping: true` to receive a
placeholder-to-original `mapping` for reversible workflows. `confidence_threshold`
defaults to `0.7`.

```bash
curl -sS -X POST "$OPENMED_URL/pii/deidentify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient Jordan Ramirez was admitted on 2026-01-02.",
    "method": "mask",
    "lang": "en",
    "keep_mapping": true
  }'
```

```python
payload = {
    "text": "Patient Jordan Ramirez was admitted on 2026-01-02.",
    "method": "mask",
    "lang": "en",
    "keep_mapping": True,
}
response = requests.post(f"{BASE_URL}/pii/deidentify", json=payload, timeout=60)
response.raise_for_status()
result = response.json()
print(result["deidentified_text"])
print("redacted:", result["num_entities_redacted"])
```

Response (`deidentify(...).to_dict()`; the `mapping` field appears only when
`keep_mapping=true` and mapping data exists):

```json
{
  "original_text": "Patient Jordan Ramirez was admitted on 2026-01-02.",
  "deidentified_text": "Patient [NAME] was admitted on 2026-01-02.",
  "pii_entities": [
    {
      "text": "Jordan Ramirez",
      "label": "NAME",
      "entity_type": "NAME",
      "start": 8,
      "end": 22,
      "confidence": 0.987,
      "redacted_text": "[NAME]",
      "canonical_label": null,
      "sources": [],
      "evidence": {},
      "threshold": null,
      "action": null,
      "surrogate": null,
      "metadata": {}
    }
  ],
  "method": "mask",
  "timestamp": "2026-01-02T09:30:00",
  "num_entities_redacted": 1,
  "metadata": {},
  "audit_report": null,
  "mapping": {"[NAME]": "Jordan Ramirez"}
}
```

To shift dates instead of masking, use `method: "shift_dates"` with an optional
`date_shift_days`:

```bash
curl -sS -X POST "$OPENMED_URL/pii/deidentify" \
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
curl "$OPENMED_URL/models/loaded"
```

```python
response = requests.get(f"{BASE_URL}/models/loaded", timeout=10)
response.raise_for_status()
state = response.json()
print("warm:", state["warm_models"])
print("resident cap:", state["max_resident_models"])
```

Response with one resident model:

```json
{
  "default_keep_alive_seconds": 600.0,
  "max_resident_models": 2,
  "memory_budget_bytes": null,
  "resident_memory_bytes": 0,
  "pending_memory_bytes": 0,
  "memory_admission_wait_seconds": 5.0,
  "warm_models": ["disease_detection_superclinical"],
  "models": {
    "OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M": {
      "models": 0,
      "tokenizers": 0,
      "pipelines": 1,
      "active_requests": 0,
      "keep_alive_seconds_remaining": 287.4,
      "resident": true,
      "loading": false,
      "footprint_bytes": 0,
      "pending_footprint_bytes": 0
    }
  }
}
```

When no model is cached, `models` is `{}` and `warm_models` lists only the
configured preload set. `default_keep_alive_seconds`, `max_resident_models`, and
`memory_budget_bytes` are `null` when their optional env vars are unset.

## Unload a model — `POST /models/unload`

Free one inactive model, or all of them. If a model still has active requests,
the service leaves it loaded and reports the count.

Unload one model:

```bash
curl -sS -X POST "$OPENMED_URL/models/unload" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "disease_detection_superclinical"}'
```

```python
payload = {"model_name": "disease_detection_superclinical"}
response = requests.post(f"{BASE_URL}/models/unload", json=payload, timeout=30)
response.raise_for_status()
print(response.json())
```

Response:

```json
{
  "unloaded": true,
  "model_name": "disease_detection_superclinical",
  "active_requests": 0,
  "loading": false,
  "released": {"models": 1, "tokenizers": 1, "pipelines": 1}
}
```

Unload every inactive model:

```bash
curl -sS -X POST "$OPENMED_URL/models/unload" \
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

Response:

```json
{
  "unloaded": true,
  "released": {"models": 1, "tokenizers": 1, "pipelines": 2},
  "active_models": {}
}
```

Exactly one of `model_name` or `all: true` is required; sending neither returns
the validation error envelope described below.

## Handling errors

Every non-2xx response uses one JSON envelope, so a single handler covers
validation, bad-request, timeout, and internal errors:

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
    ]
  }
}
```

The `error.code` is one of `validation_error`, `bad_request`, `timeout`,
`not_ready`, `service_busy`, `circuit_breaker_open`, `internal_error`, or one of
the `privacy_gateway_*` codes. `details` may be a list (validation errors), an
object (for example `{"timeout_seconds": 300}`), or `null`. Responses may also
carry an `X-Request-ID` header (echoed as `request_id` in the envelope) for
correlating logs.

Reproduce the validation envelope with a blank `text`:

```bash
curl -sS -X POST "$OPENMED_URL/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "   "}'
```

```python
def analyze(text: str) -> dict:
    response = requests.post(
        f"{BASE_URL}/analyze", json={"text": text}, timeout=60
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
`openmed.service.client.OpenMedClient`, built on `httpx`. It maps each endpoint
to a method and raises `OpenMedAPIError` on any non-2xx response, so you skip the
manual status checks above:

```python
from openmed.service.client import OpenMedAPIError, OpenMedClient

with OpenMedClient("http://127.0.0.1:8080", timeout=30.0) as client:
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
        keep_mapping=True,
    )
    loaded = client.loaded_models()

    try:
        client.unload_model("disease_detection_superclinical")
    except OpenMedAPIError as exc:
        print(exc.status_code, exc.code, exc.message, exc.request_id)
```

`OpenMedAPIError` exposes the service error `code`, `message`, optional
`details`, HTTP `status_code`, and any `request_id` returned by the service or a
proxy. Use `client.unload_all_models()` to drop every inactive model in one call.

## Related pages

- [REST Service](rest-service.md) — full endpoint reference, configuration env
  vars, allowlists, and the Docker/Compose run path.
- [REST Authentication](serving/authentication.md) — optional API-key and
  bearer-token enforcement in front of these endpoints.
- [Async REST Jobs & Webhooks](serving/async-jobs.md) — for large
  de-identification batches that should not hold a client connection open.
