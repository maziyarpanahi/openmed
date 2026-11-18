# REST Response Headers

REST responses can carry PHI: extracted entities, de-identified text,
reversible mappings, clinical briefs, job results and journey resources.
Browsers and intermediaries may retain any HTTP response, including `GET`
results, so the OpenMed REST service marks every non-probe response as
non-storable and hardens how browsers handle it.

## Header set

Every response from the REST app, except the probe routes below, carries:

| Header | Value | Purpose |
| --- | --- | --- |
| `Cache-Control` | `no-store` | Browsers, proxies and CDNs must not store the response. |
| `Pragma` | `no-cache` | The same signal for HTTP/1.0 caches. |
| `X-Content-Type-Options` | `nosniff` | Browsers must not reinterpret JSON or NDJSON as HTML or script. |
| `Referrer-Policy` | `no-referrer` | Request URLs, including job and resource identifiers, stay out of outbound `Referer` headers. |
| `X-Frame-Options` | `DENY` | Legacy clients refuse to render the response in a frame. |
| `Content-Security-Policy` | `frame-ancestors 'none'` | Current browsers refuse to render the response in a frame. |

Both framing headers are sent. `frame-ancestors` is the current standard and
takes precedence in browsers that support it; `X-Frame-Options` covers older
clients. The CSP sets no other directive, so it does not restrict scripts or
styles on the generated API documents.

Coverage includes:

- successful JSON responses, such as `POST /pii/extract`,
  `POST /pii/deidentify`, `POST /brief`, `GET /jobs/{job_id}` and
  `GET /v1/journey/resources`;
- NDJSON streams, such as `POST /pii/deidentify/stream`; only the response
  start is rewritten, so streamed bodies are never buffered;
- error envelopes for unknown routes (`404`), unsupported methods (`405`),
  validation failures (`422`), rejected hosts (`400`), authentication
  failures (`401`), throttling and backpressure, and unhandled exceptions
  (`500`);
- the `/graphql` transport, as described below.

The policy wraps the application's complete middleware stack, outside
Starlette's `ServerErrorMiddleware`, so the `500` envelope rendered for an
unhandled exception is covered as well. Responses the ASGI server writes on
its own, for example when it rejects a malformed HTTP request before the
application runs, are outside the application and do not carry the headers.

## Handler-set headers

The policy never adds a second copy of a header. When a handler or an inner
middleware already set one:

- a `Cache-Control` value that contains `no-store` is kept, so a stricter
  value such as `no-store, max-age=0` survives; any other value, such as
  `public, max-age=600`, is replaced with `no-store`;
- a `Pragma` value other than `no-cache` is replaced;
- `X-Content-Type-Options` is always exactly `nosniff`, its only defined
  value;
- `Referrer-Policy` and `X-Frame-Options` are enforced as `no-referrer` and
  `DENY`, including when a handler supplies weaker values;
- existing `Content-Security-Policy` restrictions remain in force alongside
  a separate `frame-ancestors 'none'` policy. Policies are combined in one
  header as a comma-separated list, preserving each policy boundary.

## Exemptions

Probe routes pass through unchanged so orchestrators and load balancers see
the same responses as before:

- `GET /health`
- `GET /livez`
- `GET /readyz`

The generated API documents contain no PHI and declare a revalidation policy
instead of `no-store`:

| Route | `Cache-Control` |
| --- | --- |
| `GET /openapi.json` | `no-cache` |
| `GET /docs` | `no-cache` |
| `GET /docs/oauth2-redirect` | `no-cache` |
| `GET /redoc` | `no-cache` |

`no-cache` allows a browser to keep the document but requires revalidation
before reuse. These responses omit `Pragma` and still carry the nosniff,
referrer and framing headers. Paths match exactly; other paths under `/docs`
keep the strict default. When a deployment disables a document route, it
drops out of the exemption list.

## GraphQL

`/graphql` receives the same policy. Headers already present on a GraphQL
response, for example from a GraphQL transport middleware that sets
`Cache-Control`, `Pragma` and `Referrer-Policy`, are detected and not
repeated, so each header appears exactly once.

## Tests

`tests/unit/service/test_response_headers.py` enumerates every registered
route, calls it once with synthetic input and fails when a response lacks the
headers. Probe routes, the declared API documents and the GraphQL WebSocket
route are the only listed exemptions, so a newly added route is checked
automatically.

```bash
python -m pytest tests/unit/service/test_response_headers.py -q
```
