# REST Service (v0.6.2)

OpenMed `v0.6.2` hardens the FastAPI service introduced in `v0.6.1` with shared
model reuse, explicit model unloading, and idle model cleanup:

- `GET /health`
- `GET /livez`
- `GET /readyz`
- `GET /models/loaded`
- `POST /models/unload`
- `POST /analyze`
- `POST /pii/extract`
- `POST /pii/deidentify`

This release adds stricter request validation, shared model/pipeline reuse, optional startup preload, bounded warm-pool residency, model keep-alive controls, and a unified non-2xx error envelope.

## Run Locally

Install the service dependencies:

```bash
uv pip install -e ".[hf,service]"
```

Start the API server:

```bash
uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

## Python Client

The service extra includes the typed sync client and its `httpx` dependency:

```bash
uv pip install -e ".[hf,service]"
```

Use `OpenMedClient` against a running service:

```python
from openmed.service.client import OpenMedAPIError, OpenMedClient

with OpenMedClient("http://127.0.0.1:8080", timeout=30.0) as client:
    result = client.analyze(
        "Patient started imatinib for CML.",
        model_name="disease_detection_superclinical",
    )
    pii = client.extract_pii("Paciente: Maria Garcia", lang="es")
    redacted = client.deidentify(
        "Paciente: Maria Garcia",
        method="mask",
        keep_mapping=True,
    )
    loaded = client.loaded_models()
```

Non-2xx responses raise `OpenMedAPIError` with the service error `code`,
`message`, optional `details`, HTTP status, and any `X-Request-ID` returned by
the service or proxy:

```python
try:
    client.unload_model("disease_detection_superclinical")
except OpenMedAPIError as exc:
    print(exc.status_code, exc.code, exc.message, exc.request_id)
```

## Static OpenAPI Spec

The committed OpenAPI document lives at `docs/api/openapi.json`. Regenerate it
after changing REST routes, request schemas, response schemas, or service
metadata:

```bash
.venv/bin/python scripts/export_openapi.py
```

The export command imports `openmed.service.app.create_app()`, calls
`app.openapi()`, stamps `info.version` from `openmed.__version__`, and writes
deterministic JSON with sorted keys. The unit test suite includes a drift guard
that compares the committed artifact byte-for-byte against a fresh in-memory
export.

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

Optional warm-pool resident model limit:

```bash
OPENMED_SERVICE_PRELOAD_MODELS=disease_detection_superclinical \
OPENMED_SERVICE_MAX_RESIDENT_MODELS=2 \
uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

`OPENMED_SERVICE_MAX_RESIDENT_MODELS` bounds how many models remain resident in the shared warm-pool. When the limit is exceeded, the least-recently-used idle model is unloaded. Omit it for unbounded resident model caching.

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

Optional dynamic request batching:

```bash
OPENMED_SERVICE_BATCHING_ENABLED=true \
OPENMED_SERVICE_BATCH_MAX_SIZE=8 \
OPENMED_SERVICE_BATCH_MAX_WAIT_MS=25 \
uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

Dynamic batching is off by default. When enabled, `/pii/extract` groups
compatible requests and dispatches them through the PII batch helper; models
with true batch backends get one backend batch, while model families whose
batch helper falls back to per-text analysis still preserve per-request results.
`/analyze` uses one backend pipeline call for compatible requests with
`sentence_detection=false`; requests that need sentence segmentation or other
non-batch-compatible settings are still coalesced but executed independently.
`OPENMED_SERVICE_BATCH_MAX_SIZE` must be a positive integer.
`OPENMED_SERVICE_BATCH_MAX_WAIT_MS` is a non-negative wait window in
milliseconds.

Optional graceful-shutdown drain timeout:

```bash
OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS=30 uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

`OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS` is a non-negative number of seconds.
During shutdown, readiness is flipped off, new model-backed work is rejected,
and the service waits up to this timeout for in-flight `/analyze`,
`/pii/extract`, and `/pii/deidentify` requests to finish. The default is `30`.

## Reliability Changes

- Requests now run against one shared service runtime per process, including a shared `OpenMedConfig` and bounded warm-pool loader.
- Blocking inference is executed off the event loop and guarded by the active profile timeout (`prod=300s`, `test=60s`, etc.).
- Text-bearing inference requests are capped before model execution to bound memory use.
- Loaded model pipelines can be released manually with `POST /models/unload`.
- `OPENMED_SERVICE_MAX_RESIDENT_MODELS` evicts the least-recently-used idle model when mixed-model traffic exceeds the configured resident limit.
- Inference requests accept `keep_alive` to schedule model unloading after the model becomes idle.
- Dynamic request batching can be enabled for compatible `/analyze` and `/pii/extract` traffic with `OPENMED_SERVICE_BATCHING_ENABLED=true`.
- `/livez` reports process liveness, `/readyz` reports startup readiness, and `/health` remains the backward-compatible health alias.
- Graceful shutdown rejects new model-backed requests and drains in-flight model-backed requests for up to `OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS`.
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

`GET /health` remains the backward-compatible health alias.

### `GET /livez`

Liveness response:

```json
{
  "status": "ok",
  "service": "openmed-rest"
}
```

### `GET /readyz`

Readiness response after startup preload completes:

```json
{
  "status": "ready",
  "service": "openmed-rest"
}
```

Before startup readiness, `/readyz` returns `503` with `error.code` set to
`not_ready`.

### `GET /models/loaded`

Returns currently cached model resources and idle-unload status:

```json
{
  "default_keep_alive_seconds": 600.0,
  "max_resident_models": 2,
  "warm_models": ["disease_detection_superclinical"],
  "models": {
    "OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M": {
      "models": 0,
      "tokenizers": 0,
      "pipelines": 1,
      "active_requests": 0,
      "keep_alive_seconds_remaining": 287.4,
      "resident": true
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
    "code": "validation_error|bad_request|timeout|not_ready|internal_error",
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
  -e OPENMED_SERVICE_MAX_RESIDENT_MODELS=2 \
  openmed:0.6.2
```

### Docker Compose

Use the provided `docker-compose.yml` to build and start the service with a
single command. The Compose setup maps port **8080**, sets
`OPENMED_PROFILE=prod`, and persists the Hugging Face cache in a named volume.
`OPENMED_CACHE_DIR` points inside that mounted cache so service model downloads
are reused across restarts.

```bash
docker compose up -d
```

Verify the service started correctly:

```bash
docker compose ps
# The STATUS column should show "(healthy)"
```

Stop the container:

```bash
docker compose down
```

To remove the persisted model cache too, delete the named volume:

```bash
docker compose down --volumes
```

Smoke check:

```bash
curl http://127.0.0.1:8080/health
```

Optional values such as `HF_TOKEN`, `OPENMED_PROFILE`,
`OPENMED_CACHE_DIR`, `OPENMED_SERVICE_PRELOAD_MODELS`, and
`OPENMED_SERVICE_MAX_RESIDENT_MODELS` can be supplied from a local `.env` file.
Keep `.env` ignored and never commit secrets to version control.
