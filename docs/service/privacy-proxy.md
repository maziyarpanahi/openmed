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

This is a privacy boundary, not a compliance certification or a clinical
decision system. Use synthetic fixtures in development and validate residual
risk for the deployment context.
