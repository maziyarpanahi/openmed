# Local privacy proxy

OpenMed includes a small OpenAI-compatible chat-completions boundary for
applications that already speak the common message-completion protocol. It
listens locally, redacts text before calling an injected inference transport,
and restores the placeholders only in the local response.

The proxy exposes both `POST /v1/chat/completions` and
`POST /chat/completions`. Both routes require a single
`Content-Type: application/json` header and accept parameters such as
`charset=utf-8`. A missing, repeated, or unsupported content type returns
HTTP 415 with the reason `unsupported_content_type` before the proxy reads
the request body.

Send requests using this JSON shape.

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

This is a privacy boundary, not a compliance certification or a clinical
decision system. Use synthetic fixtures in development and validate residual
risk for the deployment context.

## Reason codes

Every proxy error uses the same PHI-free body. The message is constant, and
`details.reason` says which check failed:

```json
{
  "error": {
    "code": "privacy_proxy_invalid_request",
    "message": "Privacy proxy request failed",
    "details": {"reason": "messages_required"}
  }
}
```

Status `400` and `415` mean the proxy refused the request: either the request
is malformed or local redaction blocked it before anything left the machine.
Status `502` and `503` mean the proxy or its transport could not complete a
valid request. With `"stream": true`, errors that occur after the stream has
started arrive as a final `data:` event with the same body, because the HTTP
status has already been sent as `200`.

<!-- privacy-proxy-reason-codes:start -->
| Reason | Status | `error.code` | Meaning | What to do |
| --- | --- | --- | --- | --- |
| `invalid_request` | 400 | `privacy_proxy_invalid_request` | The body is valid JSON but not an object. | Send a JSON object. |
| `invalid_json` | 400 | `privacy_proxy_invalid_request` | The body is not valid JSON. | Send a valid JSON body. |
| `messages_required` | 400 | `privacy_proxy_invalid_request` | `messages` is missing, not a list, or empty. | Send at least one message. |
| `invalid_message` | 400 | `privacy_proxy_invalid_request` | An item in `messages` is not an object. | Send each message as an object. |
| `message_role_required` | 400 | `privacy_proxy_invalid_request` | A message has a missing or blank `role`. | Set `role` on every message. |
| `message_content_required` | 400 | `privacy_proxy_invalid_request` | A message has no `content`, `tool_calls` or `function_call`. | Add content to the message. |
| `model_invalid` | 400 | `privacy_proxy_invalid_request` | `model` is not a non-empty string. | Send a model name or omit the field. |
| `stream_invalid` | 400 | `privacy_proxy_invalid_request` | `stream` is not a boolean. | Send `true` or `false`. |
| `request_id_invalid` | 400 | `privacy_proxy_invalid_request` | The `X-Request-ID` header is not 1-128 characters from `A-Z`, `a-z`, `0-9`, `_`, `.`, `:` and `-`. | Fix or omit the header. |
| `unsupported_content_type` | 415 | `privacy_proxy_invalid_request` | The `Content-Type` header is missing, repeated, or not `application/json`. | Send one `Content-Type: application/json` header. |
| `redaction_failed` | 400 | `privacy_proxy_redaction_failed` | The local detector or redaction step failed. | Check the detector or model installation; the request was not forwarded. |
| `outbound_tripwire_detected` | 400 | `privacy_proxy_redaction_failed` | The independent tripwire scan found possible PHI left after redaction. | Treat it as a blocked request; review the detector and policy. |
| `redaction_confidence_below_threshold` | 400 | `privacy_proxy_redaction_failed` | A detected entity scored below the policy's minimum confidence. | Review the policy threshold or the detector model. |
| `disallowed_entity_category` | 400 | `privacy_proxy_redaction_failed` | The policy forbids forwarding text containing a detected category. | Remove that content or change the policy. |
| `missing_entity_offsets` | 400 | `privacy_proxy_redaction_failed` | The detector returned an entity without character offsets. | Fix the injected extractor. |
| `invalid_entity_offsets` | 400 | `privacy_proxy_redaction_failed` | The detector returned offsets outside the text or with an end that is not after the start. | Fix the injected extractor. |
| `overlapping_entity_spans` | 400 | `privacy_proxy_redaction_failed` | The detector returned overlapping entity spans. | Fix the injected extractor. |
| `placeholder_collision` | 400 | `privacy_proxy_redaction_failed` | Redaction produced a placeholder that was already used in the request. | Retry; report it if it repeats. |
| `missing_transport` | 503 | `privacy_proxy_not_configured` | No transport is configured, for example on the module-level `app`. | Create the app with `create_app(transport=...)`. |
| `transport_failed` | 502 | `privacy_proxy_transport_failed` | The transport raised an exception. | Check the local model server. |
| `transport_not_callable` | 502 | `privacy_proxy_transport_failed` | The transport is not callable and has no usable `complete` or `stream` method. | Pass a callable transport. |
| `invalid_transport_response` | 502 | `privacy_proxy_transport_failed` | The transport response has no text content the proxy can read. | Return text, an OpenAI-style response, or a mapping with `content`. |
| `response_rejected` | 502 | `privacy_proxy_response_rejected` | The response could not be restored safely. | Check the transport output format. |
| `invalid_stream_chunk` | 502 | `privacy_proxy_response_rejected` | A stream chunk is not text, a mapping, or a valid chunk shape. | Yield text or OpenAI-style chunk mappings. |
| `mangled_placeholder` | 502 | `privacy_proxy_response_rejected` | A placeholder in a streamed chunk was altered and cannot be restored. | Make the model keep placeholder tokens unchanged. |
| `malformed_placeholder` | 502 | `privacy_proxy_response_rejected` | The response contains a placeholder-like token with an invalid shape. | Make the model keep placeholder tokens unchanged. |
| `unknown_placeholder` | 502 | `privacy_proxy_response_rejected` | The response contains a placeholder that this request never issued. | Make the model keep placeholder tokens unchanged. |
| `duplicate_placeholder` | 502 | `privacy_proxy_response_rejected` | A placeholder appears more often than the restoration policy allows. | Check whether the model repeats placeholder tokens. |
| `duplicate_mapping_placeholder` | 502 | `privacy_proxy_response_rejected` | The request's replacement map defines a placeholder twice. | Report it; this indicates a redaction bug. |
| `restoration_limit_exceeded` | 502 | `privacy_proxy_response_rejected` | The response or mapping exceeded a restoration size or count limit. | Shorten the response or raise the restoration limits. |
| `invalid_request_scope` | 502 | `privacy_proxy_response_rejected` | The restoration state is not valid for this request. | Report it; this indicates a state bug. |
| `unsupported_response_content` | 502 | `privacy_proxy_response_rejected` | The response is not text or JSON-like content. | Return text or JSON-compatible values. |
| `duplicate_response_key` | 502 | `privacy_proxy_response_rejected` | Restoring placeholders in object keys would merge two keys. | Avoid placeholders in response object keys. |
| `inbound_restoration_error` | 502 | `privacy_proxy_response_rejected` | The replacement map or response content could not be read. | Check the transport output encoding. |
| `privacy_proxy_error` | 502 | `privacy_proxy_error` | An unexpected internal error occurred. | Retry; report it if it repeats. |

The lower-level Python APIs `OutboundRequestPrivacyFilter`,
`RequestStateStore` and `InboundRestorationStore` raise exceptions with a
`reason_code` attribute instead of returning HTTP responses. They reuse the
codes above where the meaning is the same and add these:

| Reason | Raised as | Meaning | What to do |
| --- | --- | --- | --- |
| `outbound_privacy_error` | `OutboundPrivacyError` | An outbound privacy check failed. | Inspect the exception type. |
| `invalid_request_id` | `OutboundPrivacyError` | `request_id` contains unsupported characters. | Use 1-128 characters from `A-Z`, `a-z`, `0-9`, `_`, `.`, `:` and `-`. |
| `redactor_required` | `RedactorRequiredError` | No callable local redactor is configured. | Pass `redactor` or `redact`. |
| `unsupported_request_body` | `UnsupportedRequestBodyError` | The body is not a JSON object with supported message content. | Send JSON chat messages. |
| `redaction_incomplete` | `RedactionOutputError` | The redacted output still contains a replaced original value. | Fix the redactor; nothing was sent. |
| `replacement_collision` | `RedactionError` | One replacement token maps to two different values. | Make the redactor's tokens unique per value. |
| `replacement_state_error` | `ReplacementStateError` | Request replacement state could not be kept. | Inspect the exception type. |
| `invalid_replacement_state` | `ReplacementStateError` | Replacement state cannot become inbound restoration state. | Check the replacement tokens and request id. |
| `duplicate_request_state` | `ReplacementStateError` | State already exists for this request id. | Use a new request id per request. |
| `replacement_state_limit` | `ReplacementStateLimitError` | The state store is full. | Remove finished requests with `pop` or `discard`, or raise `max_entries`. |
| `replacement_state_not_found` | `RequestStateNotFoundError` | No state exists for this request id. | Call `save` before `get` or `pop`. |
| `missing_request_state` | `UnknownRequestStateError` | No inbound restoration state exists for this request. | Call `put` before restoring the response. |
<!-- privacy-proxy-reason-codes:end -->
