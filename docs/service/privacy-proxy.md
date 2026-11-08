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

## Messages-shaped requests

Clients built on the Anthropic Messages API shape can point their base URL at
the proxy and use `POST /v1/messages` (or `POST /messages` when the base URL
already ends in `/v1`). The same content-type rules apply.

```json
{
  "model": "local-fixture",
  "max_tokens": 256,
  "system": "The patient is Avery Example.",
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "Call 555-0100."}]}
  ]
}
```

The proxy requires `model`, a positive integer `max_tokens`, and a non-empty
`messages` list with `user` and `assistant` roles. `content` and `system` may
be a string or a list of `text` blocks. The `system` text and every text block
go through the same detector, outbound tripwire and placeholder map as the
chat route. `model`, `role`, `type`, `stop_sequences` and numeric fields are
forwarded unchanged. Other string fields, such as `metadata`, are redacted the
same way as on the chat route. Transport metadata carries `"api": "messages"`.

Invalid requests return HTTP 400 with a stable `reason` such as
`max_tokens_invalid`, `messages_required` or `message_role_invalid`, and no
request text. Until their redaction rules exist, the route fails closed on
non-text content blocks (`content_block_unsupported`) and on
`"stream": true` (`stream_unsupported`). Inbound `x-api-key` and
`Authorization` headers are never passed to the transport.

A transport may return a Messages-shaped mapping, which is restored and
returned as-is, or plain text, which is wrapped in a minimal `message`
response with one text block.

This is a privacy boundary, not a compliance certification or a clinical
decision system. Use synthetic fixtures in development and validate residual
risk for the deployment context.
