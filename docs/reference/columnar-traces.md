# Columnar trace-dataset adapter

`openmed.traces.schemas.columnar` adapts PyArrow trace datasets one bounded
`RecordBatch` at a time. It rewrites only configured text paths, keeps the
input Arrow schema unchanged, and leaves labels and every unselected column
untouched.

PyArrow is optional:

```bash
uv pip install -e ".[columnar]"
```

## Redact one batch

Pass a deterministic local text redactor when the replacement policy is more
specific than the safe default. Dotted paths traverse nested structs; paths
through lists of structs are supported as well.

```python
import pyarrow as pa

from openmed.traces.schemas.columnar import redact_record_batch

batch = pa.record_batch(
    {
        "trace_id": ["trace-a"],
        "payload": [{"message": "SYNTHETIC_NOTE_A"}],
        "label": [1],
    }
)

redacted = redact_record_batch(
    batch,
    text_columns=["payload.message"],
    text_redactor=lambda value: value.replace("SYNTHETIC_", "MASKED_"),
)
```

If `text_redactor` is omitted, every non-empty selected string is replaced
with the deterministic `"[REDACTED]"` marker. The adapter never loads a model
or makes a network request.

## Stream bounded batches

Use `iter_redacted_record_batches` for a record-batch reader, Arrow dataset, or
other iterable. Oversized incoming batches are sliced before redaction, so
the redactor receives only bounded work and the source is not materialized.

```python
from openmed.traces.schemas.columnar import iter_redacted_record_batches

for batch in iter_redacted_record_batches(
    source_batches,
    text_columns=["payload.message", ("payload", "events", "text")],
    text_redactor=local_redactor,
    batch_size=512,
):
    write_batch(batch)
```

`ColumnarTraceSchemaAdapter` packages the same configuration for repeated
calls through `adapt_batch` and `adapt_batches`.

## Schema and privacy contract

- The output batch has the input schema verbatim, including logical types,
  field metadata, nullability, and row count.
- Only selected string, large-string, dictionary-string, or list-like text
  values are rewritten. Labels and other columns are copied through unchanged.
- Missing paths, non-text selections, and redactor failures raise a
  `ColumnarTraceAdapterError` whose message contains schema context only; raw
  source values are not included.
- The module emits no logs or reports and contains no dataset fixtures. Use
  synthetic offline fixtures in downstream tests.
