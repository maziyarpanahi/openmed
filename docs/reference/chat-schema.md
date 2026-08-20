# Role-message training schema

`openmed.traces.schemas.chat` provides a local-only adapter for training records
with a `messages` array. It rewrites message content while preserving message
order, roles, tool identifiers, tool inputs, and auxiliary metadata.

## Redact a record

Inject a deterministic callable that accepts one text value and returns its
replacement. The adapter does not load a model or make a network request when
the callable is supplied.

```python
from openmed.traces.schemas.chat import redact_chat_record


def local_redactor(text: str) -> str:
    return text.replace("synthetic person", "[PERSON]")


record = {
    "messages": [
        {"role": "user", "content": "synthetic person"},
        {"role": "assistant", "content": "synthetic answer"},
    ],
    "metadata": {"split": "train", "source": "synthetic"},
}

redacted = redact_chat_record(record, text_redactor=local_redactor)
```

The input is never mutated. Message arrays retain their original order, and
only the `content` field is visited. A string content value is passed directly
to the redactor. Structured content visits text parts such as:

```python
{"type": "text", "text": "synthetic text"}
{"type": "input_text", "text": "synthetic text"}
{"type": "tool_result", "content": [{"type": "text", "text": "synthetic result"}]}
```

Image URLs, tool names, tool IDs, tool inputs, and unknown nested fields remain
untouched. This conservative boundary prevents a schema adapter from
interpreting identifiers or auxiliary metadata as message text.

## Redact a message array

Use `redact_chat_messages` when the surrounding record is already separated:

```python
from openmed.traces.schemas.chat import redact_chat_messages

messages = [
    {"role": "user", "content": "synthetic question"},
    {"role": "assistant", "content": [{"type": "text", "text": "synthetic answer"}]},
]
redacted_messages = redact_chat_messages(messages, text_redactor=local_redactor)
```

For counters and safe content paths, use the corresponding
`*_with_report` function. Reports contain counts and structural paths only;
they never include source text or replacement text. Nonstandard
caller-controlled path keys are represented by deterministic
`key_sha256_...` segments, including when a report is constructed directly.

The `RoleMessageSchemaAdapter` also exposes `detect`, `walk`, `reconstruct`,
and `transform` for callers that use a training-schema registry. All
reconstruction paths return deep-copied values and preserve untouched mapping
and sequence structure.
