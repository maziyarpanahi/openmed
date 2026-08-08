# Training conversation schemas

Training exports often describe the same conversation with different nested
structures. OpenMed provides a small, local-only registry so a redaction
workflow can select text fields without flattening the surrounding record.

The built-in registry recognizes these layouts:

- `messages`: role/content message lists, including `conversation.messages`
  and `data.messages` wrappers.
- `sharegpt`: ShareGPT-style `conversations[].value` lists.
- `preference`: prompt/chosen/rejected records, including nested message lists
  under a response field.

## Detect and walk content

Detection is structural and deterministic. It does not load a model, read a
dataset, or make a network request.

```python
from openmed.traces.schemas.registry import TrainingSchemaRegistry

registry = TrainingSchemaRegistry()
record = {
    "messages": [
        {"role": "user", "content": "Synthetic user value"},
        {"role": "assistant", "content": "Synthetic answer"},
    ]
}

schema = registry.resolve(record)
print(schema.name)  # messages
for path, text in registry.walk(record, schema=schema):
    print(path, text)
```

Each path is a tuple of mapping keys and list indexes. Role labels, metadata,
and other structural fields are not returned as content.

## Reconstruct a redacted copy

Pass replacements keyed by paths returned from `walk`, or use `transform` with
a local text function. Reconstruction always returns a copy and preserves the
original nesting.

```python
redacted = registry.transform(
    record,
    lambda text: text.replace("Synthetic", "[REDACTED]"),
)
```

## Explicit selection and ambiguity

Auto-detection fails closed when no schema matches or when more than one schema
matches. An ambiguous record must be selected explicitly before any copy is
reconstructed:

```python
redacted = registry.transform(
    record,
    lambda text: "[REDACTED]",
    schema="messages",
)
```

Use `AmbiguousSchemaError` and `UnknownSchemaError` to report a safe decision
state to a caller. Registry errors contain schema names and paths only; they do
not include record values.

Custom schemas can implement the `TrainingConversationSchema` protocol and be
registered with `registry.register(...)`. Registration is in-process and has
no discovery or network side effects.
