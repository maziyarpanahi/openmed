# Parallel trace files

`openmed.traces.parallel` processes independent local trace files with bounded
worker shards. It passes paths to workers instead of loading records into the
driver, creates a fresh store for each file, and merges outcomes by their
original input index. A slow file therefore cannot reorder the returned
results or share pseudonym state with a neighboring file.

```python
from pathlib import Path

from openmed.traces import run_trace_files


def redact_trace(path: Path, store: dict[str, int]) -> dict[str, int | str]:
    """Process one local file using only synthetic, caller-owned state."""
    store["records"] = sum(1 for line in path.read_text().splitlines() if line)
    return {"file_index": path.stem, "record_count": store["records"]}


if __name__ == "__main__":
    result = run_trace_files(
        [Path("synthetic-000.jsonl"), Path("synthetic-001.jsonl")],
        redact_trace,
        store_factory=dict,
        shard_size=1,
        max_workers=2,
    )

    if result.is_complete:
        ordered_values = result.ordered_values
```

The handler and store factory should be module-level, importable callables when
process execution is desired. `spawn` is the default start method. If spawning
is unavailable, process references are not picklable, or pool startup fails,
the same bounded shards run sequentially and `result.fallback_reason` records a
safe reason code. Set `use_processes=False` when a handler intentionally needs
local-only resources.

`TraceRunResult.to_dict()` is an operator-safe report: it contains counts,
input indexes, timings, and sanitized exception type names, but never paths,
callback values, exception messages, or trace content. Use
`result.raise_for_errors()` or `raise_on_error=True` when a failed input should
abort the caller after the safe metadata has been collected.

The process path requires callback return values to be pickleable. Handlers
that write already-redacted outputs to caller-owned local paths can return a
small count or digest instead. No network service or model download is needed
by the executor.
