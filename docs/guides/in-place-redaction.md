# Transactional in-place redaction

`openmed.traces.transactional_redact` safely replaces one local text trace
after a deterministic redactor has produced and validated the complete
replacement. The operation does not select a model, download an artifact, or
make a mandatory network call. Supply a cached or otherwise local redactor
from the application that owns the trace.

```python
from pathlib import Path

from openmed.traces import transactional_redact


def redact_local(text: str) -> str:
    # Connect this to a deterministic, local-only redaction pipeline.
    return text.replace("Synthetic-Trace-Value-001", "[REDACTED]")


result = transactional_redact(
    Path("trace.jsonl"),
    redact_local,
    validator=lambda candidate: "Synthetic-Trace-Value-001" not in candidate,
)
print(result.to_dict())
```

The source is read before the redactor runs. The replacement is encoded,
validated, written to a sibling temporary file, flushed, and synced before
`os.replace` atomically swaps the directory entry. If redaction, validation,
source-change detection, temporary-file writing, backup creation, or the
exchange fails, the source remains unchanged and temporary artifacts are
removed. A caller interruption follows the same cleanup path.

Backups are enabled by default. The first backup is named
`<trace>.bak`; if that path already exists, exclusive creation selects
`<trace>.bak.1`, `<trace>.bak.2`, and so on without overwriting prior backups.
Pass `backup=False` only when a separate recovery mechanism is already in
place. An explicit `backup_path` is treated as a preferred name and receives
the same collision-safe suffix behavior.

Permission bits and access/modification timestamps are preserved by default.
Use `preserve_permissions=False` or `preserve_timestamps=False` when the
replacement should receive fresh metadata. `preserve_metadata` sets both
options together.

Reports contain only byte counts and transaction status. Transaction errors
are deliberately value-free, so raw trace text is not echoed by this module.
The feature is a file-safety primitive, not a compliance certification or a
guarantee of zero residual privacy risk.
