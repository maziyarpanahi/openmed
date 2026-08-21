# Offline artifact inventory

`openmed.interop.artifact_inventory` creates a deterministic local inventory
for synthetic or already-sanitized artifact files. It records only normalized
relative paths, byte counts, stable media types, and SHA-256 content
fingerprints; it never stores or renders file contents and makes no network
call.

## Build an inventory

Pass relative paths with an explicit local root. Inputs are sorted by their
normalized POSIX path before hashing and reporting:

```python
from pathlib import Path

from openmed.interop.artifact_inventory import build_artifact_inventory

inventory = build_artifact_inventory(
    ["reports/alpha.json", "reports/summary.md"],
    root=Path("synthetic-artifacts"),
)
```

Traversal (`..`), duplicate normalized paths, symlink escapes, directories,
missing files, and read failures are rejected. Error messages contain only an
input entry number and a safe reason, so a failed run does not echo a caller's
path or file contents. An inventory accepts at most 10,000 entries; larger or
non-terminating caller iterables are stopped at that boundary before file
access.

## Counts-only reports

JSON and Markdown renderers are counts-only by default. They contain the
artifact count, total bytes, media-type counts, and unique fingerprint count:

```python
from openmed.interop.artifact_inventory import (
    render_artifact_inventory_json,
    render_artifact_inventory_markdown,
)

json_text = render_artifact_inventory_json(inventory)
markdown_text = render_artifact_inventory_markdown(inventory)
```

When a local review needs the metadata index as well, pass
`counts_only=False`. The expanded form still includes metadata only, never
artifact bytes or decoded content. Use `inventory.write_json(...)` or
`inventory.write_markdown(...)` to persist either form locally. JSON
indentation accepts `None` or an integer from 0 through 8, preventing report
formatting from becoming a channel for caller-injected values.
