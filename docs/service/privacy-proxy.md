# Local privacy proxy

OpenMed includes a small OpenAI-compatible chat-completions boundary for
applications that already speak the common message-completion protocol. It
listens locally, redacts text before calling an injected inference transport,
and restores the placeholders only in the local response.

The proxy exposes both `POST /v1/chat/completions` and
`POST /chat/completions`. A request uses the familiar shape:

```json
{
  "model": "local-fixture",
  "messages": [
    {"role": "user", "content": "Contact Avery Example at 555-0100."}
  ]
}
```

Create the app with a local transport. The transport receives a copied request
mapping whose text leaves contain deterministic `OPENMED_PHI` placeholders;
the in-memory replacement map is not passed in metadata and is discarded when
the request finishes.

```python
from typing import Any

from openmed.service.privacy_proxy import create_app


def local_transport(payload: dict[str, Any], **_: Any) -> str:
    # Call an on-device model here. `payload` is already redacted.
    return payload["messages"][-1]["content"]


app = create_app(transport=local_transport)
```

Run that application with:

```bash
uvicorn my_proxy:app --host 127.0.0.1 --port 8081
```

The module-level `openmed.service.privacy_proxy.app` is import-safe, but has no
transport configured and therefore rejects completion requests with a
PHI-free `503` response. It never creates a network client. Inject an
explicitly local transport before serving requests.

Set `"stream": true` to receive standard server-sent events. The transport
may return text chunks, OpenAI-style chunk mappings, or a synchronous or
asynchronous iterable of either. Placeholder tokens split across chunks are
buffered until they can be restored; the replacement map itself is never sent
as an event.

## Inbound protections

`create_app` uses the same inbound controls as the REST service. They run in
this order, and the first two read headers only:

1. **Trusted host.** Only `localhost`, `127.0.0.1` and `[::1]` are accepted
   by default, with any port. Any other `Host` header, such as a DNS
   rebinding name, gets `400 bad_request` before the body is read.
2. **Authentication (opt-in).** When enabled, missing, invalid or
   under-scoped credentials get `401` or `403` before the body is read.
   `GET /health` stays open.
3. **Request size.** Bodies over the byte limit get
   `413 request_size_exceeded` before JSON parsing. Chunked bodies are
   counted as they arrive.

After parsing, the proxy checks the message count and the total characters
across every text value it would redact. Both checks run before the extractor
or the transport is called. Streaming responses also have a wall-clock bound.

| Setting | Default | Purpose |
| --- | --- | --- |
| `OPENMED_PRIVACY_PROXY_TRUSTED_HOSTS` | `localhost,127.0.0.1,[::1]` | Comma-separated `Host` allowlist. `*` is refused. |
| `OPENMED_PRIVACY_PROXY_MAX_REQUEST_BYTES` | `16777216` (16 MiB) | Largest request body, from 1 byte to 256 MiB. |
| `OPENMED_PRIVACY_PROXY_MAX_MESSAGES` | `256` | Most messages in one request. |
| `OPENMED_PRIVACY_PROXY_MAX_TEXT_CHARS` | `1000000` | Most characters across all redacted text values. |
| `OPENMED_PRIVACY_PROXY_MAX_STREAM_SECONDS` | `300` | Longest wall-clock time for one streaming response. |

`create_app` also takes `trusted_hosts`, `operational_limits` (an
`OperationalLimits`), `limits` (a `PrivacyProxyLimits`) and `auth_config` (a
`ServiceAuthConfig`). An argument overrides its setting. An invalid setting
stops app creation with an error that names the setting but never echoes its
value.

Authentication reuses the REST service settings and is off unless
`OPENMED_SERVICE_AUTH_ENABLED=true`. API keys
(`OPENMED_SERVICE_AUTH_API_KEYS`) and JWT bearer tokens
(`OPENMED_SERVICE_AUTH_JWKS` or `OPENMED_SERVICE_AUTH_JWKS_FILE`) work as
described in [REST authentication](../serving/authentication.md). Both
completion routes require the `pii:write` scope, the same scope as
`POST /privacy-gateway/complete`. They require credentials even when
`OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT=false`. To use other scopes, pass an
`auth_config` whose `route_scopes` already lists the two routes.

```bash
OPENMED_SERVICE_AUTH_ENABLED=true \
OPENMED_SERVICE_AUTH_API_KEYS='[{"id": "proxy", "key_hash": "sha256:<sha256-hex-digest>", "scopes": ["pii:write"]}]' \
uvicorn my_proxy:app --host 127.0.0.1 --port 8081
```

When the stream bound passes, the proxy stops reading from the transport and
cancels an asynchronous transport. A synchronous iterator cannot be
interrupted while it blocks, so the bound is checked after each chunk and a
late chunk is dropped. The stream then ends with a `privacy_proxy_timeout`
event and no `[DONE]` marker. The response status is already `200` by then,
so clients must read the error event.

The resolved settings are kept on `app.state.trusted_hosts`,
`app.state.operational_limits` and `app.state.auth`. A deployment check can
read them without sending a request. They hold no raw credentials.

## Error codes

Proxy errors share one envelope:
`{"error": {"code": ..., "message": ..., "details": {"reason": ...}}}`. The
message is fixed, and no error carries message text, placeholders or the
replacement map.

| Status | Code | Reasons |
| --- | --- | --- |
| 400 | `bad_request` | Untrusted `Host` header (shared service envelope) |
| 401 / 403 / 429 | `authentication_required`, `invalid_credentials`, `forbidden`, `auth_rate_limited` | Shared service auth envelope |
| 413 | `request_size_exceeded` | Body over the byte limit (shared operational-limit envelope) |
| 400 | `privacy_proxy_invalid_request` | `invalid_json`, `messages_required`, `invalid_message`, `message_role_required`, `message_content_required`, `model_invalid`, `stream_invalid`, `request_id_invalid` |
| 413 | `privacy_proxy_limit_exceeded` | `message_count_exceeded`, `text_length_exceeded` |
| 400 | `privacy_proxy_redaction_failed` | `redaction_failed`, `outbound_tripwire_detected`, `placeholder_collision` and policy reasons |
| 502 | `privacy_proxy_transport_failed` | `transport_failed`, `transport_not_callable`, `invalid_transport_response` |
| 502 | `privacy_proxy_response_rejected` | `response_rejected`, `duplicate_placeholder`, `mangled_placeholder`, `invalid_stream_chunk` and restoration reasons |
| 502 | `privacy_proxy_error` | `privacy_proxy_error` (unexpected failure) |
| 503 | `privacy_proxy_not_configured` | `missing_transport` |
| Stream event | `privacy_proxy_timeout` | `stream_duration_exceeded` |

These checks cover inbound requests only. The transport you inject still
decides where redacted payloads go.

This is a privacy boundary, not a compliance certification or a clinical
decision system. Use synthetic fixtures in development and validate residual
risk for the deployment context.
