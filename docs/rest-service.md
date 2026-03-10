# REST Service (v0.6.2)

OpenMed `v0.6.2` hardens the FastAPI service introduced in `v0.6.1` while keeping the same four endpoints:

- `GET /health`
- `POST /analyze`
- `POST /pii/extract`
- `POST /pii/deidentify`

This release adds stricter request validation, shared model/pipeline reuse, optional startup preload, and a unified non-2xx error envelope.

## Run Locally

Install the service dependencies:

```bash
uv pip install -e ".[hf,service]"
```

Start the API server:

```bash
uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

Optional profile selection (defaults to `prod`):

```bash
OPENMED_PROFILE=dev uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

Optional shared model preload at startup:

```bash
OPENMED_SERVICE_PRELOAD_MODELS=disease_detection_superclinical,OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \
uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

`OPENMED_SERVICE_PRELOAD_MODELS` is a comma-separated list of registry aliases or full Hugging Face ids. Empty entries are ignored and duplicates are removed.

## Reliability Changes

- Requests now run against one shared service runtime per process, including a shared `OpenMedConfig` and shared `ModelLoader`.
- Blocking inference is executed off the event loop and guarded by the active profile timeout (`prod=300s`, `test=60s`, etc.).
- Non-2xx responses use one JSON envelope across validation, bad-request, timeout, and internal errors.
- `/pii/deidentify` still accepts the legacy `shift_dates` boolean, but it is now a deprecated alias for `method="shift_dates"`.

## Endpoints

### `GET /health`

Health response:

```json
{
  "status": "ok",
  "service": "openmed-rest",
  "version": "0.6.2",
  "profile": "prod"
}
```

### `POST /analyze`

Request body:

```json
{
  "text": "Patient started imatinib for CML.",
  "model_name": "disease_detection_superclinical",
  "confidence_threshold": 0.0,
  "group_entities": false,
  "aggregation_strategy": "simple"
}
```

Returns the same shape as OpenMed `analyze_text(..., output_format="dict")`.

### `POST /pii/extract`

Request body:

```json
{
  "text": "Paciente: Maria Garcia, DNI: 12345678Z",
  "lang": "es",
  "use_smart_merging": true
}
```

Returns the same shape as `extract_pii(...).to_dict()`.

### `POST /pii/deidentify`

Request body:

```json
{
  "text": "Paciente: Maria Garcia, DNI: 12345678Z",
  "method": "mask",
  "lang": "es",
  "keep_mapping": true
}
```

Date shifting:

```json
{
  "text": "Paciente: Maria Garcia, fecha: 15/01/2020",
  "method": "shift_dates",
  "date_shift_days": 30,
  "lang": "es"
}
```

The deprecated `shift_dates: true` boolean is still accepted as an alias for `method: "shift_dates"`.

Returns `deidentify(...).to_dict()`. When `keep_mapping=true` and mapping data exists, a `mapping` field is included.

## Error Envelope

All non-2xx responses use this shape:

```json
{
  "error": {
    "code": "validation_error|bad_request|timeout|internal_error",
    "message": "human-readable summary",
    "details": null
  }
}
```

Validation example:

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

Timeout example:

```json
{
  "error": {
    "code": "timeout",
    "message": "Request exceeded configured timeout of 300 seconds",
    "details": {
      "timeout_seconds": 300
    }
  }
}
```

## Docker

Build:

```bash
docker build -t openmed:0.6.2 .
```

Run:

```bash
docker run --rm -p 8080:8080 \
  -e OPENMED_PROFILE=prod \
  -e OPENMED_SERVICE_PRELOAD_MODELS=disease_detection_superclinical \
  openmed:0.6.2
```

Smoke check:

```bash
curl http://127.0.0.1:8080/health
```
