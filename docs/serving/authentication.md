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

`GET /health`, `GET /livez`, `GET /readyz`, `GET /metrics`, `/docs`,
`/redoc`, and `/openapi.json` are exempt so health checks, local API docs, and
metrics scraping can remain separate from model-work authorization.

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

## Middleware Order

Authentication reads request headers only. The service runs its HTTP layers in
this order, outermost first:

1. Trusted-host validation. An unexpected `Host` returns the `400` envelope
   before authentication, the failure limiter, or the request throttle runs.
2. CORS, when `OPENMED_SERVICE_CORS_ORIGINS` is set. Preflight requests are
   answered here and do not need credentials.
3. mTLS, API-key and JWT authentication, scope checks and the failed-attempt
   limiter. Missing or invalid credentials are rejected before the request
   body is read.
4. Request-body limits, correlation IDs, tracing, the request throttle,
   metrics and readiness.

Because authentication runs before the correlation layer, `401`, `403` and
`auth_rate_limited` `429` responses do not carry an `X-Request-ID` header or a
`request_id` field.

## Browser Clients

When CORS and authentication are both enabled, preflight responses allow the
`Authorization`, `X-API-Key` and `X-OpenMed-Priority` request headers in
addition to the default content and `X-Request-ID` headers. Credential headers
are not offered while authentication is off. Responses to allowed origins
expose `X-Request-ID` and `Retry-After`, and error responses such as `401`,
`403`, `413` and `429` keep their CORS headers so browser code can read the
JSON error envelope. Origins must still be listed exactly; wildcard origins are
rejected at startup, and requests from other origins receive no
`Access-Control-Allow-Origin` header.

## Mutual TLS

For certificate-bearing workload identity, including deployments that require
mTLS before JWT or API-key authorization, see
[Mutual TLS Client Authentication](mtls.md).
