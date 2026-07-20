---
name: serving-openmed-rest-api
description: "Stand up OpenMed's FastAPI REST service for clinical NER, PII extraction, and de-identification, with health checks, model keep-alive/unload, optional dynamic batching, and no-PHI logging. Use when the user wants to serve OpenMed over HTTP, deploy a de-id/NER REST API, run an inference endpoint for clinical text, add a /analyze or /pii/deidentify route, or containerize OpenMed as a service. Covers the service extra, launching create_app with uvicorn, the real endpoints (/health, /analyze, /pii/extract, /pii/deidentify, /models/loaded, /models/unload), request/response shapes, ServiceRuntime env-var configuration, and self-hosted auth/CORS/TLS notes."
license: Apache-2.0
metadata:
  project: OpenMed
  category: deployment-ops
  pairs: adjacent
  version: "1.0"
---

# Serving OpenMed over REST

`openmed.service` is a hardened **FastAPI** app exposing OpenMed's NER, PII
extraction, and de-identification over HTTP. It is built to be **self-hosted**:
models run on-device, there's no telemetry, and the request schemas reject raw
PHI from spilling into errors. Use it when callers need request/response
inference; use `batch-processing-clinical-text` for corpora.

## When to use this skill

To put OpenMed behind an HTTP endpoint your own apps call — an internal de-id
microservice, an NER backend, a containerized inference tier. For agent/tool
integration prefer the MCP server (`deploying-openmed-mcp`); for offline bulk
work use batch processing.

## Quick start

```bash
pip install "openmed[service]"          # FastAPI + uvicorn + pydantic

# Launch the ASGI app (factory create_app, or the module-level `app`)
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8000
```

```python
# Or build it in-process (e.g. to mount under a parent app / add middleware):
from openmed.service import create_app
app = create_app()
```

```bash
curl -s localhost:8000/health
# {"status":"ok","service":"openmed-rest","version":"...","profile":"prod"}

curl -s localhost:8000/analyze -H 'content-type: application/json' -d '{
  "text": "Patient received 75mg clopidogrel for NSTEMI.",
  "model_name": "disease_detection_superclinical"
}'

curl -s localhost:8000/pii/deidentify -H 'content-type: application/json' -d '{
  "text": "John Doe called 555-123-4567 on 01/15/2020.",
  "method": "mask"
}'
```

## Endpoints (confirmed in `openmed/service/app.py`)

| Method & path | Purpose | Request schema |
| --- | --- | --- |
| `GET /health` | liveness + version + active profile | — |
| `GET /models/loaded` | cache/keep-alive status of resident models | — |
| `POST /models/unload` | unload one model or all inactive models | `ModelUnloadRequest` (`model_name` or `all=true`) |
| `POST /analyze` | clinical NER | `AnalyzeRequest` |
| `POST /pii/extract` | detect PII/PHI spans | `PIIExtractRequest` |
| `POST /pii/deidentify` | mask/remove/replace/hash/shift-dates PHI | `PIIDeidentifyRequest` |

Request fields (from `openmed/service/schemas.py`, strict — unknown fields are
rejected):

- **`AnalyzeRequest`**: `text` (required), `model_name`
  (`"disease_detection_superclinical"`), `confidence_threshold` (0.0),
  `group_entities`, `aggregation_strategy` (`simple|first|average|max`),
  `sentence_detection`, `sentence_language`, `sentence_clean`,
  `use_fast_tokenizer`, `keep_alive`.
- **`PIIExtractRequest`**: `text`, `model_name`
  (default `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1`),
  `confidence_threshold` (0.5), `use_smart_merging`, `lang`
  (`en/fr/de/it/es/nl/hi/te/pt/ar/ja/tr`), `normalize_accents`, `keep_alive`.
- **`PIIDeidentifyRequest`**: same base plus `method`
  (`mask|remove|replace|hash|shift_dates`, default `mask`),
  `confidence_threshold` (0.7), `keep_year`, `shift_dates`, `date_shift_days`,
  `keep_mapping`, `policy`, `use_smart_merging`, `use_safety_sweep`.

Responses are the OpenMed result `to_dict()` (e.g. `{text, entities[...], ...}`).
Errors use a stable envelope: `{"error": {"code", "message", "details"}}` with
`422` validation_error, `400` bad_request, `504` timeout, `500` internal_error.

## Configuring the runtime (env vars)

`ServiceRuntime.from_env()` reads the process environment at startup
(`openmed/service/runtime.py`):

| Env var | Effect |
| --- | --- |
| `OPENMED_PROFILE` | config profile (`prod` default) |
| `OPENMED_SERVICE_PRELOAD_MODELS` | comma list of models to warm at startup |
| `OPENMED_SERVICE_KEEP_ALIVE` | default idle keep-alive before unload |
| `OPENMED_SERVICE_MAX_RESIDENT_MODELS` | cap resident models (warm pool) |
| `OPENMED_SERVICE_BATCHING_ENABLED` | enable dynamic request batching |
| `OPENMED_SERVICE_BATCH_MAX_SIZE` | max dynamic batch size (default 8) |
| `OPENMED_SERVICE_BATCH_MAX_WAIT_MS` | batch-collection window (default 5ms) |

```bash
OPENMED_SERVICE_PRELOAD_MODELS="disease_detection_superclinical" \
OPENMED_SERVICE_BATCHING_ENABLED=true \
uvicorn openmed.service.app:app --host 0.0.0.0 --port 8000
```

Preloading avoids first-request latency; the warm pool keeps hot models resident
and idle-unloads the rest. `/analyze` and `/pii/extract` coalesce concurrent
requests when batching is enabled.

## Workflow

1. **Install + launch.** `pip install "openmed[service]"`, then run
   `uvicorn openmed.service.app:app` (or build with `create_app()`).
2. **Configure the runtime** via env vars before start: set
   `OPENMED_PROFILE`, preload your hot models, and decide keep-alive / max
   resident / batching to fit the box.
3. **Front it with auth/TLS.** Place a reverse proxy or gateway (API keys/mTLS,
   CORS allow-list) ahead of the app — it has none built in.
4. **Health-check + warm.** Poll `GET /health`; preloaded models warm during
   the lifespan startup so the first real request isn't cold.
5. **Call the endpoints** (`/analyze`, `/pii/extract`, `/pii/deidentify`) with
   the strict JSON schemas; handle the `{"error": {...}}` envelope.
6. **Manage memory** with `GET /models/loaded` and `POST /models/unload` as
   traffic shifts between models.

## Containerizing

```dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir "openmed[service]"
ENV OPENMED_SERVICE_PRELOAD_MODELS="disease_detection_superclinical"
EXPOSE 8000
CMD ["uvicorn", "openmed.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Bake/mount the model cache so containers don't re-download on every start; the
service runs offline after that.

## Hand-off to / from OpenMed

- **Same engine, different surface:** `/analyze` → `openmed.analyze_text`,
  `/pii/extract` → `openmed.extract_pii`, `/pii/deidentify` →
  `openmed.deidentify`. Results match the library exactly.
- **Agents/tools:** for Claude Code / Codex / chat clients, expose the same
  capabilities as MCP tools instead (`deploying-openmed-mcp`).
- **Bulk:** for corpora, call `batch-processing-clinical-text` in a worker, not
  per-request HTTP.

## Edge cases & gotchas

- **No built-in auth/CORS/TLS.** The app ships hardened input validation but no
  authentication. Put it behind your own reverse proxy / API gateway (mTLS,
  API keys, CORS allow-list) before any real traffic. Bind `127.0.0.1` for
  local use; only expose `0.0.0.0` behind that proxy.
- **No-PHI logging.** Don't add request/response body logging — that's PHI.
  The error envelope is designed to avoid echoing input; keep it that way. Log
  status codes, timings, and model names only.
- **Strict schemas.** Unknown JSON fields are rejected (`extra="forbid"`); a bad
  `lang`/`method`/`model_name` returns `422`/`400` with a field-level reason.
- **Cold start vs memory.** Preloading + a high `MAX_RESIDENT_MODELS` trades RAM
  for latency; tune to the box.
- **Timeouts return `504`** per the profile's configured `timeout`; long inputs
  may need a larger profile or pre-chunking.
- **`keep_mapping`/`policy` outputs are sensitive.** A de-id response with a
  mapping re-identifies patients — only enable it for trusted callers and store
  the mapping securely, never in service logs.

## Standards & references

- FastAPI: https://fastapi.tiangolo.com/
- Uvicorn (ASGI server): https://www.uvicorn.org/
- OpenAPI (the service auto-serves `/docs` and `/openapi.json`):
  https://www.openapis.org/
- HIPAA de-identification, 45 CFR 164.514(b):
  https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- OpenMed source: `openmed/service/app.py` (routes), `openmed/service/runtime.py`
  (`ServiceRuntime`), `openmed/service/schemas.py` (request models).
