# Local trace-store discovery

OpenMed can inventory supported local agent-trace stores before any trace
payload is opened or changed. The discovery pass returns only a store type,
regular-file count, and aggregate byte size:

```python
from openmed.traces import discover_trace_stores

for store in discover_trace_stores():
    print(store.store_type, store.file_count, store.byte_size)
```

The default rules are explicit and platform-aware. They cover the hidden
home-directory stores used by local agent clients and their conventional
macOS, Linux, and Windows application-data locations. A caller can provide
its own mapping when it needs a controlled inventory:

```python
from pathlib import Path

stores = discover_trace_stores(
    {"synthetic": Path("/var/tmp/synthetic-traces")},
)
```

Store types are report metadata, so use short, lower-case, PHI-free identifiers
containing only letters, numbers, dots, underscores, or hyphens. Discovery
normalizes an invalid caller-supplied label to `custom` instead of returning it.

Discovery is local-only and metadata-only. It uses directory enumeration and
file metadata, does not open or parse payloads, does not follow symlinks, and
silently skips missing or unreadable roots and descendants. Results are
deterministically sorted and intentionally do not include raw paths or error
text, so they can be placed in a privacy-safe aggregate report.

## Opting out

Disable the default pass for a process with:

```bash
export OPENMED_TRACE_DISCOVERY=off
```

Set `enabled=False` for a single call. The function makes no network call, and
it never requires a model, credential, or remote service.

Additional roots can be supplied through `OPENMED_TRACE_ROOTS`. Use
`store_type=path` entries separated by the platform path separator (`:` on
Unix-like systems and `;` on Windows):

```bash
export OPENMED_TRACE_ROOTS='synthetic=/var/tmp/synthetic-traces'
```

These entries are additive when using the default platform rules. Use the
explicit `roots` argument when a caller needs the inventory to contain only a
known set of local directories.

This feature is an inventory aid, not a compliance certification or guarantee.
Keep real trace content out of logs, fixtures, reports, and support bundles.
