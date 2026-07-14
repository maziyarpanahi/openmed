# REST Service

OpenMed's FastAPI service provides shared model reuse, explicit model unloading,
streaming PII extraction, asynchronous jobs, privacy-gateway egress, and SMART
Backend Services ingestion. The generated OpenAPI document is the source of
truth for exact request and response schemas. Its current public operations are:

- `GET /health`
- `GET /livez`
- `GET /readyz`
- `GET /models/loaded`
- `POST /models/unload`
- `POST /analyze`
- `POST /pii/extract`
- `POST /pii/extract/stream`
- `POST /pii/deidentify`
- `POST /fhir/smart-backend/ingestions`
- `GET /fhir/smart-backend/ingestions/{job_id}`
- `GET /fhir/smart-backend/ingestions/{job_id}/summary`
- `POST /jobs`
- `GET /jobs/{job_id}`
- `POST /privacy-gateway/complete`

The opt-in `GET /metrics` route is intentionally excluded from the generated
schema and returns `404` unless metrics are enabled.

The service includes stricter request validation, shared model/pipeline reuse,
optional startup preload, bounded warm-pool residency, model keep-alive
controls, optional no-PHI OpenTelemetry tracing, and a consistent application
error envelope.

For ready-to-run `curl` and Python `requests` snippets covering the common
calls, see the task-oriented [REST API Recipes](rest-recipes.md) page.

For large de-identification batches that should not hold a client connection
open, use [Async REST Jobs & Webhooks](serving/async-jobs.md).

## Run Locally

Install the service dependencies:

```bash
uv pip install -e ".[hf,service]"
```

Start the API server:

```bash
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

Keep this loopback bind for local use. Before exposing the service on a network,
configure [authentication](serving/authentication.md), TLS at the ingress or
reverse proxy, and an exact trusted-host allowlist.

## Postman Collection

A ready-to-import [Postman collection](api/openmed.postman_collection.json)
(`docs/api/openmed.postman_collection.json`) ships with one example request per
endpoint group so you can exercise the API without writing any client code.
Import it into Postman (or any tool that reads the Postman v2.1 schema), then set
the `base_url` collection variable to your server (it defaults to
`http://127.0.0.1:8080`, matching the Python client's safe loopback default).
Use an `https://` base URL for every non-loopback deployment; never send
credentials or clinical data over plaintext HTTP. For an authenticated
deployment, configure
collection-level authorization in Postman or add either an `X-API-Key` header or
an `Authorization: Bearer <token>` header. The collection intentionally stores
no credentials. Every example body uses synthetic clinical text only — no real
PHI. Keep those bodies synthetic because real patient text can persist in
Postman history, console output, shared workspaces, exports, or screenshots.

Before polling, copy the identifier returned by the async-job or SMART-ingestion
POST request into `job_id`. The SMART request leaves
`{{smart_private_key_pem}}` unresolved on purpose. Define it only in a secure
local-client environment; in Postman, the secure variable can be backed by
Local Vault. Escape PEM newlines as `\n` for the JSON body, and never replace
the reference with a real key in a saved or exported collection.

## Python Client

The service extra includes the typed sync client and its `httpx` dependency:

```bash
uv pip install -e ".[hf,service]"
```

Use `OpenMedClient` against a running service:

All patient-like content in the examples below is synthetic. Do not paste real
patient text, reversible mappings, or credentials into documentation, issues,
logs, or screenshots.

```python
from openmed.service.client import OpenMedAPIError, OpenMedClient

with OpenMedClient("http://127.0.0.1:8080", timeout=300.0) as client:
    result = client.analyze(
        "Patient started imatinib for CML.",
        model_name="disease_detection_superclinical",
    )
    pii = client.extract_pii("Paciente: Maria Garcia", lang="es")
    redacted = client.deidentify(
        "Paciente: Maria Garcia",
        method="mask",
    )
    llm = client.privacy_gateway(
        "Patient Maria Garcia called 555-0100.",
        confidence_threshold=0.9,
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
OPENMED_PROFILE=dev uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

## Browser and Host Allowlists

The service is strict by default:

- CORS is off unless `OPENMED_SERVICE_CORS_ORIGINS` is set. Cross-origin
  browser requests are not granted `Access-Control-Allow-Origin` by default.
- Trusted host checking is always on. `OPENMED_SERVICE_TRUSTED_HOSTS` defaults
  to `localhost,127.0.0.1,[::1]`, so loopback clients pass and unexpected Host
  headers are rejected with the standard error envelope.

Both variables accept comma-separated allowlists. CORS origins must be exact
scheme/host/port origins and cannot use wildcards. Setting
`OPENMED_SERVICE_TRUSTED_HOSTS` replaces the loopback default, so include every
host the service should accept.

Example browser front-end configuration:

```bash
OPENMED_SERVICE_CORS_ORIGINS=http://localhost:5173 \
OPENMED_SERVICE_TRUSTED_HOSTS=127.0.0.1,localhost \
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

For a reverse proxy in front of the service, configure the public browser
origin and the host header forwarded to Uvicorn:

```bash
OPENMED_SERVICE_CORS_ORIGINS=https://clinic-ui.example.com \
OPENMED_SERVICE_TRUSTED_HOSTS=api.example.com \
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

Optional shared model preload at startup:

```bash
OPENMED_SERVICE_PRELOAD_MODELS=disease_detection_superclinical,OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

`OPENMED_SERVICE_PRELOAD_MODELS` is a comma-separated list of registry aliases or full Hugging Face ids. Empty entries are ignored and duplicates are removed.

Optional warm-pool resident model limit:

```bash
OPENMED_SERVICE_PRELOAD_MODELS=disease_detection_superclinical \
OPENMED_SERVICE_MAX_RESIDENT_MODELS=2 \
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

`OPENMED_SERVICE_MAX_RESIDENT_MODELS` bounds how many models remain resident in the shared warm-pool. When the limit is exceeded, the least-recently-used idle model is unloaded. Omit it for unbounded resident model caching.

Optional default model keep-alive:

```bash
OPENMED_SERVICE_KEEP_ALIVE=10m uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

`OPENMED_SERVICE_KEEP_ALIVE` accepts seconds as a number or duration strings such as `30s`, `5m`, `1h30m`, or `1d`. Omit it for indefinite caching, use `0` for unload-after-request behavior, or use request-level `keep_alive` to override the default for one call.

Optional request text cap:

```bash
OPENMED_SERVICE_MAX_TEXT_LENGTH=250000 uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

`OPENMED_SERVICE_MAX_TEXT_LENGTH` caps the `text` field accepted by `/analyze`,
`/pii/extract`, `/pii/extract/stream`, `/pii/deidentify`, `/jobs`, and
`/privacy-gateway/complete`. The default is `1,000,000` characters. Oversized
requests return the standard `422` validation envelope; split larger documents
client-side or route them through batch processing.

Optional privacy-gateway egress endpoint:

```bash
OPENMED_SERVICE_PRIVACY_GATEWAY_ENDPOINT=https://llm-proxy.example.com/complete \
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

`POST /privacy-gateway/complete` refuses to call an external LLM unless
`OPENMED_SERVICE_PRIVACY_GATEWAY_ENDPOINT` is configured by the operator or the
FastAPI app is given an explicit `app.state.privacy_gateway_transport` callable.
The request body never accepts an arbitrary URL. The gateway redacts locally,
forwards only the redacted prompt, keeps the placeholder map in process memory
for that request, runs an independent outbound tripwire scan, and records only
PHI-free audit metadata.

Optional dynamic request batching:

```bash
OPENMED_SERVICE_BATCHING_ENABLED=true \
OPENMED_SERVICE_BATCH_MAX_SIZE=8 \
OPENMED_SERVICE_BATCH_MAX_WAIT_MS=25 \
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
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

Optional request coalescing:

```bash
OPENMED_SERVICE_COALESCING_ENABLED=true \
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

Request coalescing is off by default. When enabled, identical concurrent
`/analyze`, `/pii/extract`, and `/pii/deidentify` requests share one in-flight
model computation keyed by endpoint, normalized text, and request options. The
single result, or the leader error, is fanned out to all joined waiters. This is
not a persistent response cache; entries are evicted shortly after completion.

Optional model resilience controls:

```bash
OPENMED_SERVICE_RETRY_MAX_ATTEMPTS=3 \
OPENMED_SERVICE_RETRY_BACKOFF_INITIAL_SECONDS=0.05 \
OPENMED_SERVICE_CIRCUIT_BREAKER_FAILURE_THRESHOLD=3 \
OPENMED_SERVICE_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS=30 \
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

Model load and inference work is retried with bounded exponential backoff and
jitter, then counted against an in-process circuit breaker keyed by resolved
model/backend. While a breaker is open, model-backed endpoints return `503`
with error code `circuit_breaker_open` and a `Retry-After` header. See
[`Serving Resilience`](serving/resilience.md) for the full set of knobs.

Optional graceful-shutdown drain timeout:

```bash
OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS=30 uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

`OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS` is a non-negative number of seconds.
During shutdown, readiness is flipped off, new model-backed work is rejected,
and the service waits up to this timeout for in-flight `/analyze`,
`/pii/extract`, `/pii/deidentify`, and `/privacy-gateway/complete` requests to
finish. The default is `30`.

Optional pull-only Prometheus metrics endpoint:

```bash
OPENMED_SERVICE_METRICS_ENABLED=true uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

`GET /metrics` is disabled by default and returns `404` unless
`OPENMED_SERVICE_METRICS_ENABLED` is set to a truthy value such as `true` or
`1` before the app starts. When enabled, the endpoint renders Prometheus 0.0.4
text exposition for aggregate request counts, request duration histograms,
in-flight request count, warm-pool model load/eviction counters, and aggregate
circuit-breaker state gauges. Metrics
are pull-only: OpenMed does not push them to any remote service. Scrape it from
a locally scoped Prometheus or sidecar, and avoid exposing it directly to
untrusted networks. Metric labels are limited to static route templates and
HTTP status codes; text, model outputs, entities, client identity, document
content, and PHI are never used as label values.

Optional MLX-LM paged KV-cache budget for long-note generation:

```bash
OPENMED_SERVICE_MLX_PAGED_KV_CACHE_BUDGET=512MiB \
OPENMED_SERVICE_MLX_PAGED_KV_CACHE_PAGE_TOKENS=128 \
OPENMED_SERVICE_MLX_PAGED_KV_CACHE_CHUNK_TOKENS=512 \
OPENMED_SERVICE_MLX_PAGED_KV_CACHE_BYTES_PER_TOKEN=65536 \
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

`OPENMED_SERVICE_MLX_PAGED_KV_CACHE_BUDGET` is disabled when unset. It accepts
raw bytes or size suffixes such as `MiB` and `GiB`. The exact dense-equivalent
context is
`floor(budget / (page_tokens * bytes_per_token)) * page_tokens`. If
`OPENMED_SERVICE_MLX_PAGED_KV_CACHE_WINDOW_TOKENS` is lower than that capacity,
the lower window is used. Longer prompts remain bounded to the configured
resident window and surface aggregate occupancy/eviction counters through
`/metrics`:

- `openmed_service_mlx_paged_kv_cache_occupancy_pages`
- `openmed_service_mlx_paged_kv_cache_capacity_pages`
- `openmed_service_mlx_paged_kv_cache_peak_pages`
- `openmed_service_mlx_paged_kv_cache_eviction_total`
- `openmed_service_mlx_paged_kv_cache_budget_bytes`

Optional OpenTelemetry tracing:

```bash
OPENMED_SERVICE_OTLP_ENDPOINT=http://otel-collector:4318/v1/traces \
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

Tracing is disabled by default and the OTLP exporter is created only when
`OPENMED_SERVICE_TRACING_ENABLED=true` or `OPENMED_SERVICE_OTLP_ENDPOINT` is
set. Request spans honor incoming W3C `traceparent` headers and include only
no-PHI attributes such as route templates, status codes, request IDs, stage
names, counts, labels, lengths, and durations. See
[REST Tracing](./serving/tracing.md) for the full configuration contract.

## Reliability Changes

- Requests now run against one shared service runtime per process, including a shared `OpenMedConfig` and bounded warm-pool loader.
- Blocking inference is executed off the event loop and guarded by the active profile timeout (`prod=300s`, `test=60s`, etc.).
- Text-bearing inference requests are capped before model execution to bound memory use.
- Loaded model pipelines can be released manually with `POST /models/unload`.
- `/privacy-gateway/complete` redacts PHI before the configured external LLM
  egress and re-identifies only after validating returned placeholders.
- `OPENMED_SERVICE_MAX_RESIDENT_MODELS` evicts the least-recently-used idle model when mixed-model traffic exceeds the configured resident limit.
- Inference requests accept `keep_alive` to schedule model unloading after the model becomes idle.
- Dynamic request batching can be enabled for compatible `/analyze` and `/pii/extract` traffic with `OPENMED_SERVICE_BATCHING_ENABLED=true`.
- CORS remains disabled unless exact origins are listed, and Host headers are
  checked against the configured trusted-host allowlist.
- Identical in-flight inference requests can be coalesced with `OPENMED_SERVICE_COALESCING_ENABLED=true`.
- Model load and inference work uses bounded retry and an in-process circuit
  breaker. Open breakers fail fast with `503` and `Retry-After`.
- `/livez` reports process liveness, `/readyz` reports startup readiness, and `/health` remains the backward-compatible health alias.
- Graceful shutdown rejects new model-backed requests and drains in-flight model-backed requests for up to `OPENMED_SERVICE_SHUTDOWN_DRAIN_SECONDS`.
- `/metrics` is opt-in, pull-only, and exposes aggregate counts, gauges, and latency histograms without PHI-derived labels.
- OpenTelemetry tracing is opt-in, honors incoming trace context, and exports only no-PHI span attributes when configured.
- Application endpoint errors and invalid Host headers use one JSON envelope
  across validation, bad-request, timeout, and internal errors. A CORS
  preflight rejected by Starlette's middleware can instead use its native
  plain-text response.
- `/pii/deidentify` still accepts the legacy `shift_dates` boolean, but it is now a deprecated alias for `method="shift_dates"`.

## Endpoints

### `GET /health`

Health response:

```json
{
  "status": "ok",
  "service": "openmed-rest",
  "version": "<installed OpenMed version>",
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

### `POST /privacy-gateway/complete`

Request body:

```json
{
  "text": "Patient Maria Garcia called 555-0100.",
  "confidence_threshold": 0.9,
  "detector_confidence_floor": 0.0,
  "policy": "strict",
  "disallowed_entity_categories": [],
  "lang": "en",
  "keep_alive": "10m"
}
```

The service detects PHI locally, replaces spans with `OPENMED_PHI` placeholder
tokens, runs an independent outbound tripwire scan, sends only the redacted
prompt to the operator-configured transport, and substitutes known placeholders
back into the external response. Unknown or mangled placeholders fail closed.

Successful response shape:

```json
{
  "request_id": "4e22b0c3-4b56-4d2f-9c0d-9f2c9d331c21",
  "redacted_prompt": "Patient <<OPENMED_PHI_NAME_...>> called <<OPENMED_PHI_PHONE_...>>.",
  "external_response": "Echo Patient <<OPENMED_PHI_NAME_...>> called <<OPENMED_PHI_PHONE_...>>.",
  "reidentified_text": "Echo Patient Maria Garcia called 555-0100.",
  "entity_counts": {
    "NAME": 1,
    "PHONE": 1
  },
  "placeholder_hashes": ["..."],
  "audit": {
    "record_hash": "...",
    "verified": true
  }
}
```

## Error Envelope

Application endpoint errors and invalid Host headers use this shape:

A CORS preflight is handled before the endpoint. If Starlette rejects its
origin, method, or requested headers, the response can be plain text rather
than this JSON envelope. Clients should check the status and content type before
decoding an error body.

```json
{
  "error": {
    "code": "<stable machine-readable error code>",
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
docker build -t openmed:local .
```

Run:

```bash
docker run --rm -p 127.0.0.1:8080:8080 \
  -e OPENMED_PROFILE=prod \
  -e OPENMED_SERVICE_KEEP_ALIVE=10m \
  -e OPENMED_SERVICE_PRELOAD_MODELS=disease_detection_superclinical \
  -e OPENMED_SERVICE_MAX_RESIDENT_MODELS=2 \
  -e OPENMED_SERVICE_CORS_ORIGINS=http://localhost:5173 \
  -e OPENMED_SERVICE_TRUSTED_HOSTS=127.0.0.1,localhost \
  openmed:local
```

### Docker Compose

Use the provided `docker-compose.yml` to build and start the service with a
single command. The Compose setup publishes host port **8080**, sets
`OPENMED_PROFILE=prod`, and persists the standard Hugging Face cache directory
in a named volume so downloads placed in that cache are reused across restarts.
`OPENMED_CACHE_DIR` is also passed through for data and deployment workflows
that explicitly read it; it is not a generic `OpenMedConfig.cache_dir`
override.

The checked-in Compose port mapping listens on the host's interfaces. Treat it
as network exposure: for local-only troubleshooting, change the mapping to
`127.0.0.1:8080:8080`. Before intentional network exposure, configure
authentication, TLS, and the trusted-host allowlist.

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
`OPENMED_SERVICE_MAX_RESIDENT_MODELS`, `OPENMED_SERVICE_CORS_ORIGINS`, and
`OPENMED_SERVICE_TRUSTED_HOSTS` can be supplied from a local `.env` file.
Keep `.env` ignored and never commit secrets to version control.
