# REST Service (v0.6.1 MVP)

OpenMed `v0.6.1` adds a Dockerized FastAPI service with four endpoints:

- `GET /health`
- `POST /analyze`
- `POST /pii/extract`
- `POST /pii/deidentify`

This release is intentionally minimal and focuses on exposing existing OpenMed Python APIs over HTTP.

## Run Locally

Install the service dependencies:

```bash
pip install -e ".[hf,service]"
```

Start the API server:

```bash
uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

Optional profile selection (defaults to `prod`):

```bash
OPENMED_PROFILE=dev uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080
```

## Endpoints

### `GET /health`

Health response:

```json
{
  "status": "ok",
  "service": "openmed-rest",
  "version": "0.6.1",
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
  "group_entities": false
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

Returns `deidentify(...).to_dict()`. When `keep_mapping=true` and mapping data exists, a `mapping` field is included.

## Docker

Build:

```bash
docker build -t openmed:0.6.1 .
```

Run:

```bash
docker run --rm -p 8080:8080 -e OPENMED_PROFILE=prod openmed:0.6.1
```

Smoke check:

```bash
curl http://127.0.0.1:8080/health
```
