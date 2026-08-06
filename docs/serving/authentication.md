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
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

Static API keys are configured as SHA-256 hashes only. Send the raw key on the
request as `X-API-Key: <key>` or `Authorization: ApiKey <key>`. The service
hashes the presented key in memory and compares it with the configured digest;
raw API keys are not stored in service config.

## JWT Bearer Tokens

JWT bearer authentication validates HS256 and RS256 signatures against a
configured JWKS. Tokens must include `exp`; expired tokens are rejected.

```bash
OPENMED_SERVICE_AUTH_ENABLED=true \
OPENMED_SERVICE_AUTH_JWKS_FILE=/etc/openmed/jwks.json \
OPENMED_SERVICE_AUTH_JWT_ISSUER=https://issuer.example.com/ \
OPENMED_SERVICE_AUTH_JWT_AUDIENCE=openmed-rest \
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
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
| `POST /pii/deidentify` | `pii:write` |

`GET /health`, `GET /livez`, `GET /readyz`, `GET /metrics`, `/docs`,
`/redoc`, and `/openapi.json` are exempt so health checks, local API docs, and
metrics scraping can remain separate from model-work authorization.

Route scope requirements can be extended or overridden with
`OPENMED_SERVICE_AUTH_ROUTE_SCOPES`:

```bash
OPENMED_SERVICE_AUTH_ROUTE_SCOPES='{
  "POST /analyze": ["analyze:write"],
  "POST /pii/deidentify": ["pii:write"]
}'
```

`OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT` defaults to `true`. In that mode,
non-exempt routes not present in the route-scope map still require a valid
credential, even if they do not require a specific scope. Set it to `false`
only when another layer authorizes those routes.

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
