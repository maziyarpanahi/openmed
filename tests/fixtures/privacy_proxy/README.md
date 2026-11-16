# Synthetic Privacy-Proxy Fixtures

All values are invented. Each JSON envelope carries `synthetic: true` and
contains either a `request` body or a sequence of response `chunks`.
Pass only the inner body to the proxy; the marker is fixture metadata.

| File | Coverage |
| --- | --- |
| `single_turn.json` | One text message with an invented name and phone. |
| `multipart_text.json` | Multiple text parts in one message. |
| `tool_exchange.json` | Assistant function-call arguments and the matching tool result. |
| `image_part.json` | Text plus an `image_url` part at a reserved, non-resolving domain. |
| `split_placeholder.json` | One placeholder split across three OpenAI-style stream chunks. |

Use `tests.fixtures.privacy_proxy.load_fixture(name)` with the filename stem.
Loading a fixture performs no network calls and does not fetch the image.
The split placeholder is synthetic protocol data, not a real redaction mapping.
