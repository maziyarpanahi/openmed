# REST Service Authentication

OpenMed REST authentication is off by default for local development. Enable it
before exposing the service outside a trusted loopback or private subnet:

```bash
OPENMED_SERVICE_AUTH_ENABLED=true \
OPENMED_SERVICE_AUTH_API_KEYS='[
  {
    "id": "clinic-api",
    "principal": "clinic-api",
    "key_hash": "sha256:<sha256-hex-digest>",
    "scopes": ["analyze:write", "pii:read", "pii:write", "models:read"]
  }
]' \
python -m openmed.service.logging --host 127.0.0.1 --port 8080
```

Static API keys are configured as SHA-256 hashes only. Send the raw key on the
request as `X-API-Key: <key>` or `Authorization: ApiKey <key>`. The service
hashes the presented key in memory and compares it with the configured digest;
raw API keys are not stored in service config.

## JWT Bearer Tokens

JWT bearer authentication validates HS256 and RS256 signatures against a
configured JWKS. Tokens must include `exp`; expired tokens are rejected.
`exp`, `iat`, and `nbf` must be finite JSON numbers, not strings, booleans,
nulls, `NaN`, or infinities. `OPENMED_SERVICE_AUTH_JWT_LEEWAY_SECONDS` allows
a finite non-negative clock skew. The production profile defaults
`OPENMED_SERVICE_AUTH_JWT_MAX_LIFETIME_SECONDS` to 3600; when a lifetime bound
is active, `iat` is mandatory and `exp - iat` cannot exceed that bound.
Invalid/non-finite validation clocks fail closed; decoder failures discard
private credential bytes and their exception contexts. Other profiles can opt
into the same setting. There is no unlimited sentinel
value; configure a reviewed finite bound for longer-lived tokens.

```bash
OPENMED_SERVICE_AUTH_ENABLED=true \
OPENMED_SERVICE_AUTH_JWKS_FILE=/etc/openmed/jwks.json \
OPENMED_SERVICE_AUTH_JWT_ISSUER=https://issuer.example.com/ \
OPENMED_SERVICE_AUTH_JWT_AUDIENCE=openmed-rest \
python -m openmed.service.logging --host 127.0.0.1 --port 8080
```

You can also set `OPENMED_SERVICE_AUTH_JWKS` to an inline JWKS JSON object with
a top-level `keys` array. JWT scopes are read from `scope`, `scp`, or `scopes`
claims. `scope` may be a space-delimited string; `scp` and `scopes` may be
lists.

## Route Scopes

When authentication is enabled, the current built-in route scopes are:

| Route | Required scope |
|---|---|
| `GET /models/loaded` | `models:read` |
| `POST /models/unload` | `models:write` |
| `POST /analyze` | `analyze:write` |
| `POST /pii/extract` | `pii:read` |
| `POST /pii/extract/stream` | `pii:read` |
| `POST /pii/deidentify` | `pii:write` |
| `POST /pii/deidentify/stream` | `pii:write` |
| `POST /privacy-gateway/complete`, `POST /openhim/deidentify` | `pii:write` |
| `POST /jobs` / `GET /jobs/{job_id}` | `jobs:write` / `jobs:read` |
| Bulk export/import and SMART start/cancel | `bulk:write` |
| Bulk/SMART status, report, manifest and summary | `bulk:read` |
| `POST /omop/load` | `omop:write` |
| `POST /profile` | `profile:write` |
| `POST /cohort/resolve` | `cohort:read` |
| `GET /v1/journey/resources` | `journey:read` |
| `POST /v1/decisions` | `decisions:write` |
| `POST /brief` / `POST /ground` | `brief:write` / `ground:write` |
| `/graphql` | Selected field scopes; see [GraphQL](graphql.md) |
| `GET /service/metadata` | Any authenticated principal |

`GET /health`, `GET /livez`, `GET /readyz`, and `GET /metrics` are exempt so
health checks and metrics scraping can remain separate from model-work
authorization. In the `dev` profile, `/docs`, `/redoc`, and `/openapi.json`
are exempt too.

## API Docs and Service Metadata

The live API docs are a development aid. Only `OPENMED_PROFILE=dev` serves
Swagger UI at `/docs`, ReDoc at `/redoc`, and the live `/openapi.json`, without
authentication. Every other profile, including an unset profile (which means
`prod`), does not register these routes. With authentication off they return
`404`. With deny-by-default authentication, an anonymous caller gets the same
`401` as any other unregistered path, and an authenticated caller gets `404`.
The committed [`docs/api/openapi.json`](../api/openapi.json) remains the
published contract.

To serve the live schema outside `dev`, opt in:

```bash
OPENMED_PROFILE=prod \
OPENMED_SERVICE_AUTH_ENABLED=true \
OPENMED_SERVICE_AUTH_API_KEYS='[...]' \
OPENMED_SERVICE_OPENAPI_ENABLED=true \
python -m openmed.service.logging --host 127.0.0.1 --port 8080
```

In this mode only `GET /openapi.json` is served, and it requires any
authenticated principal; a request without credentials returns `401`. Swagger
UI and ReDoc stay off because their pages load scripts from a public CDN. If
authentication is disabled, the opted-in schema is as open as every other
route. Set `OPENMED_SERVICE_OPENAPI_ENABLED=false` to turn the docs off in
`dev` too. Values other than a boolean fail startup.

Outside `dev`, the unauthenticated `GET /health` response is only:

```json
{"status": "ok", "service": "openmed-rest"}
```

Read the package version and active profile from `GET /service/metadata`,
which requires authentication whenever authentication is enabled:

```bash
curl -H "X-API-Key: $OPENMED_API_KEY" http://127.0.0.1:8080/service/metadata
```

```json
{"service": "openmed-rest", "version": "<installed version>", "profile": "prod"}
```

The `dev` profile keeps `version` and `profile` in `/health`. The metadata
route is an operator route like `/metrics`, so it is not part of the published
OpenAPI contract.

The reference launcher, `python -m openmed.service.logging`, starts uvicorn
with its `server` response header turned off. The Dockerfiles, Helm chart, and
Compose files use this launcher. If you run uvicorn directly, pass
`--no-server-header`.

Route scope requirements can be overridden with
`OPENMED_SERVICE_AUTH_ROUTE_SCOPES`:

```bash
OPENMED_SERVICE_AUTH_ROUTE_SCOPES='{
  "POST /analyze": ["analyze:write"],
  "POST /pii/deidentify": ["pii:write"],
  "GET /jobs/{job_id}": ["jobs:read"]
}'
```

Overrides match registered route templates, never concrete job identifiers.
Unknown keys fail startup. Every built-in route has a `ROUTE_POLICIES` entry
with its scopes and admission class; adding an undeclared route fails the
registry validation. Model and heavy routes share rate/concurrency and drain
controls, including OMOP, profiling, cohorts and Journey reads.

Lazy included routers use FastAPI's effective route contexts, including HTTP
and WebSocket prefixes. Older flat route lists retain their original matcher.
An unsupported route representation fails registry validation instead of
skipping scope checks; nested routes and method mismatches have regression
controls, including the current reference-container FastAPI runtime.

Migration from earlier releases: credentials that previously accessed these
routes with an unrelated scope must receive the matching grant. `*` and
`namespace:*` still work. The deny-by-default setting remains available for
unmatched requests; it does not disable declared route scopes.

## Errors and Failed Attempts

Missing or invalid credentials return the standard service error envelope with
HTTP `401` and a `WWW-Authenticate` challenge. Valid credentials without the
required route scope return HTTP `403`. Error bodies do not echo request text,
tokens, API keys, or other caller-supplied PHI.

Failed authentication attempts are rate limited in process:

```bash
OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_RPS=5 \
OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_BURST=10 \
OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_KEY=peer
```

The failure limiter only counts failed authentication attempts. Successful
authenticated requests are not charged against it.

## Mutual TLS

For certificate-bearing workload identity, including deployments that require
mTLS before JWT or API-key authorization, see
[Mutual TLS Client Authentication](mtls.md).
