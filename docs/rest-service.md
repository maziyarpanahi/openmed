# REST Service (v0.6.2)

OpenMed `v0.6.2` hardens the FastAPI service introduced in `v0.6.1` with shared
model reuse, explicit model unloading, and idle model cleanup:

- `GET /health`
- `GET /models/loaded`
- `POST /models/unload`
- `POST /analyze`
- `POST /pii/extract`
- `POST /pii/deidentify`

This release adds stricter request validation, shared model/pipeline reuse, optional startup preload, model keep-alive controls, and a unified non-2xx error envelope.

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

Optional default model keep-alive:

```bash
OPENMED_SERVICE_KEEP_ALIVE=10m uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

`OPENMED_SERVICE_KEEP_ALIVE` accepts seconds as a number or duration strings such as `30s`, `5m`, `1h30m`, or `1d`. Omit it for indefinite caching, use `0` for unload-after-request behavior, or use request-level `keep_alive` to override the default for one call.

Optional request text cap:

```bash
OPENMED_SERVICE_MAX_TEXT_LENGTH=250000 uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

`OPENMED_SERVICE_MAX_TEXT_LENGTH` caps the `text` field accepted by `/analyze`, `/pii/extract`, and `/pii/deidentify`. The default is `1,000,000` characters. Oversized requests return the standard `422` validation envelope; split larger documents client-side or route them through batch processing.

## Reliability Changes

- Requests now run against one shared service runtime per process, including a shared `OpenMedConfig` and shared `ModelLoader`.
- Blocking inference is executed off the event loop and guarded by the active profile timeout (`prod=300s`, `test=60s`, etc.).
- Text-bearing inference requests are capped before model execution to bound memory use.
- Loaded model pipelines can be released manually with `POST /models/unload`.
- Inference requests accept `keep_alive` to schedule model unloading after the model becomes idle.
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

### `GET /models/loaded`

Returns currently cached model resources and idle-unload status:

```json
{
  "default_keep_alive_seconds": 600.0,
  "models": {
    "OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M": {
      "models": 0,
      "tokenizers": 0,
      "pipelines": 1,
      "active_requests": 0,
      "keep_alive_seconds_remaining": 287.4
    }
  }
}
```

### `POST /models/unload`

Unload one inactive model:

```json
{
  "model_name": "disease_detection_superclinical"
}
```

Unload all inactive models:

```json
{
  "all": true
}
```

If a model has active requests, the service leaves it loaded and reports the active request count.

### `POST /analyze`

Request body:

```json
{
  "text": "Patient started imatinib for CML.",
  "model_name": "disease_detection_superclinical",
  "confidence_threshold": 0.0,
  "group_entities": false,
  "aggregation_strategy": "simple",
  "keep_alive": "5m"
}
```

Returns the same shape as OpenMed `analyze_text(..., output_format="dict")`.

### `POST /pii/extract`

Request body:

```json
{
  "text": "Paciente: Maria Garcia, DNI: 12345678Z",
  "lang": "es",
  "use_smart_merging": true,
  "keep_alive": "10m"
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
  "keep_mapping": true,
  "keep_alive": "10m"
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
  -e OPENMED_SERVICE_KEEP_ALIVE=10m \
  -e OPENMED_SERVICE_PRELOAD_MODELS=disease_detection_superclinical \
  openmed:0.6.2
```

### Docker Compose

Use the provided `docker-compose.yml` to build and start the service with a single command.
The Compose setup maps port **8080**, sets `OPENMED_PROFILE=prod`, and persists the HF model cache in a named volume so models are not re-downloaded on restart.

```bash
docker compose up -d
```

Verify the service started correctly:

```bash
docker compose ps
# The STATUS column should show "(healthy)"
```

Stop and clean up:

```bash
docker compose down
```

Smoke check:

```bash
curl http://127.0.0.1:8080/health
```

> **Tip:** For sensitive environment variables (e.g. `HF_TOKEN`), store them in a `.env` file and add `.env` to `.gitignore` — never commit secrets to version control.
