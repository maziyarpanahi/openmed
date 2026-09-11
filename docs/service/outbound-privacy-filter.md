# Outbound request privacy filter

`openmed.service.privacy_proxy.outbound` provides the local boundary between
message-style request construction and a configured inference transport. It
accepts JSON request bodies with a top-level `messages` list, runs an injected
local redactor over textual message content, and returns a transformed body
plus request-scoped replacement state.

The module has no HTTP client and does not load a model. Configure a local
redactor explicitly; the transport remains outside this boundary and is only
given the returned body.

## Example

```python
from openmed.service.privacy_proxy.outbound import (
    OutboundRequestPrivacyFilter,
    RedactionResult,
)


def redact_locally(text: str) -> RedactionResult:
    original = "Synthetic Patient"
    token = "<NAME>"
    if original not in text:
        return RedactionResult(text)
    return RedactionResult(text.replace(original, token), {token: original})


privacy_filter = OutboundRequestPrivacyFilter(redact_locally)
prepared = privacy_filter.transform(
    {
        "model": "local-model",
        "messages": [
            {"role": "user", "content": "Summarize Synthetic Patient."}
        ],
    },
    request_id="request-1",
)

# Dispatch only this transformed body through the configured transport.
remote_body = prepared.body

# Keep the state local for the response-restoration stage.
replacement_state = privacy_filter.get_state(prepared.request_id)
replacement_map = replacement_state.replacements
```

`replacement_map` is read-only and has a PHI-safe string representation, but
its values remain available to a local restoration stage. Call
`discard_state()` after restoration, or `consume_state()` when the handoff is
one-time. The in-memory state store is bounded and rejects new requests when
its configured capacity is reached.

## Supported request shapes

- `application/json` and `application/json; charset=utf-8` media types.
- A JSON object containing a `messages` list.
- Message `content` as a string, or a list of typed text parts with `type` set
  to `text`, `input_text`, or `output_text`.

Multimodal parts, binary bodies, invalid JSON, and other message content
types are rejected before a transformed request is returned. The caller must
decide how to handle the safe failure; the filter never silently forwards an
unsupported body.

Do not log request bodies, redactor results, or replacement mappings. Use
`PreparedOutboundRequest.to_metadata()` and
`RequestReplacementState.to_metadata()` for counts and request identifiers
only. All examples in this guide use synthetic values.
