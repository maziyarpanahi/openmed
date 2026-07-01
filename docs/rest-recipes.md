# REST API Recipes

Copy-paste recipes for the OpenMed REST service. Each call below ships as a
ready-to-run `curl` one-liner and an equivalent Python [`requests`](https://requests.readthedocs.io/)
snippet so you can reach a first successful response without assembling requests
by hand.

For the full endpoint reference — request fields, response schemas, environment
variables, and the error envelope — see the
[REST Service guide](./rest-service.md). This page is the task-oriented
companion to that reference.

!!! note "Synthetic data only"
    Every example uses synthetic, non-real patient text. Never send real PHI to
    a service you do not control.

## Start the service

The recipes assume the API is listening on `http://127.0.0.1:8080`. The quickest
path is Docker Compose, which maps port **8080** and sets `OPENMED_PROFILE=prod`:

```bash
docker compose up -d
```

Or run it directly with uvicorn:

```bash
uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

A shared base URL is reused by every snippet:

```bash
export OPENMED_URL="http://127.0.0.1:8080"
```

```python
import requests

BASE_URL = "http://127.0.0.1:8080"
```

## `GET /health`

Confirm the service is up before sending inference traffic.

```bash
curl "$OPENMED_URL/health"
```

```python
resp = requests.get(f"{BASE_URL}/health", timeout=10)
resp.raise_for_status()
print(resp.json())
```

Success response:

```json
{
  "status": "ok",
  "service": "openmed-rest",
  "version": "0.6.2",
  "profile": "prod"
}
```

## `POST /pii/extract`

Detect PII spans in free text.

```bash
curl -X POST "$OPENMED_URL/pii/extract" \
  -H "Content-Type: application/json" \
  -d '{"text": "Paciente: Maria Garcia, DNI: 12345678Z", "lang": "es", "use_smart_merging": true}'
```

```python
payload = {
    "text": "Paciente: Maria Garcia, DNI: 12345678Z",
    "lang": "es",
    "use_smart_merging": True,
}
resp = requests.post(f"{BASE_URL}/pii/extract", json=payload, timeout=300)
resp.raise_for_status()
print(resp.json())
```

The response mirrors `extract_pii(...).to_dict()`. The exact entities depend on
the model; the shape looks like:

```json
{
  "text": "Paciente: Maria Garcia, DNI: 12345678Z",
  "entities": [
    {
      "text": "Maria Garcia",
      "label": "PER",
      "confidence": 0.99,
      "start": 10,
      "end": 22,
      "metadata": {}
    }
  ],
  "model_name": "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
  "timestamp": "2026-01-01T00:00:00Z",
  "processing_time": 0.12,
  "metadata": {}
}
```

## `POST /pii/deidentify`

Redact detected PII. The default `method` is `mask`; pass `keep_mapping` to get
the redacted-to-original mapping back for re-identification.

```bash
curl -X POST "$OPENMED_URL/pii/deidentify" \
  -H "Content-Type: application/json" \
  -d '{"text": "Paciente: Maria Garcia, DNI: 12345678Z", "method": "mask", "lang": "es", "keep_mapping": true}'
```

```python
payload = {
    "text": "Paciente: Maria Garcia, DNI: 12345678Z",
    "method": "mask",
    "lang": "es",
    "keep_mapping": True,
}
resp = requests.post(f"{BASE_URL}/pii/deidentify", json=payload, timeout=300)
resp.raise_for_status()
print(resp.json()["deidentified_text"])
```

The response mirrors `deidentify(...).to_dict()`:

```json
{
  "original_text": "Paciente: Maria Garcia, DNI: 12345678Z",
  "deidentified_text": "Paciente: [PER], DNI: [ID]",
  "pii_entities": [
    {
      "text": "Maria Garcia",
      "label": "PER",
      "entity_type": "PER",
      "start": 10,
      "end": 22,
      "confidence": 0.99,
      "redacted_text": "[PER]"
    }
  ],
  "method": "mask",
  "timestamp": "2026-01-01T00:00:00Z",
  "num_entities_redacted": 2,
  "metadata": {},
  "audit_report": null
}
```

When `keep_mapping=true` and mapping data exists, a `mapping` field is included.

## `POST /analyze`

Run general medical NER over text.

```bash
curl -X POST "$OPENMED_URL/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient started imatinib for CML.", "model_name": "disease_detection_superclinical", "confidence_threshold": 0.0}'
```

```python
payload = {
    "text": "Patient started imatinib for CML.",
    "model_name": "disease_detection_superclinical",
    "confidence_threshold": 0.0,
}
resp = requests.post(f"{BASE_URL}/analyze", json=payload, timeout=300)
resp.raise_for_status()
print(resp.json())
```

The response mirrors `analyze_text(..., output_format="dict")`:

```json
{
  "text": "Patient started imatinib for CML.",
  "entities": [
    {
      "text": "CML",
      "label": "DISEASE",
      "confidence": 0.98,
      "start": 29,
      "end": 32,
      "metadata": {}
    }
  ],
  "model_name": "OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M",
  "timestamp": "2026-01-01T00:00:00Z",
  "processing_time": 0.34,
  "metadata": {}
}
```

## Handling errors

All non-2xx responses share one JSON envelope, so a single handler covers
validation, bad-request, timeout, and internal errors:

```json
{
  "error": {
    "code": "validation_error|bad_request|timeout|internal_error",
    "message": "human-readable summary",
    "details": null
  }
}
```

For example, an empty `text` field returns `422` with:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed",
    "details": [
      {
        "field": "body.text",
        "message": "Text must not be blank",
        "type": "value_error"
      }
    ]
  }
}
```

Branch on the envelope from Python instead of only checking the status code:

```python
resp = requests.post(f"{BASE_URL}/pii/extract", json={"text": ""}, timeout=300)
if resp.status_code >= 400:
    error = resp.json()["error"]
    raise RuntimeError(f"{error['code']}: {error['message']}")
data = resp.json()
```

## See also

- [REST Service guide](./rest-service.md) — full endpoint reference, environment
  variables, the error envelope, and the Docker Compose runbook.
- [Examples & Copy/Paste Recipes](./examples.md) — notebooks and scripts for the
  Python library.
