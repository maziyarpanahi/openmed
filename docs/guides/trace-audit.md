# Counts-only local trace privacy inventory

`openmed.traces.audit` provides a read-only aggregation boundary for local
trace scanners. It reports the scale of discovered exposure without storing or
printing matched values, snippets, exception details, or trace payloads.

The input contract is deliberately small: each finding supplies a store label,
category, file label, and a half-open byte range. Unknown mapping fields are
ignored. A caller can also submit file status records for `scanned`, `skipped`,
`unreadable`, and `unsupported` inputs.

```python
from openmed.traces.audit import TraceScan, build_trace_audit

report = build_trace_audit(
    findings=[
        {
            "store": "codex",
            "category": "message",
            "file": "trace.jsonl",
            "start": 12,
            "end": 28,
        }
    ],
    scans=[
        TraceScan("codex", "trace.jsonl", "scanned"),
        TraceScan("codex", "unreadable.jsonl", "unreadable"),
    ],
)

print(report.to_terminal())
json_payload = report.to_json()
```

The JSON payload contains deterministic totals and aggregate rows under
`stores`, `categories`, `files`, and `findings`. Rows contain counts, derived
byte totals, and byte ranges only. Derived byte totals merge overlapping ranges
within each file while keeping matching offsets from different files distinct.
File labels are replaced by deterministic SHA-256-derived identifiers before
they enter a report, so even a basename containing patient data is not echoed.
Known store and category identifiers remain readable; unknown caller-provided
labels are hashed for the same reason.

The report builder does not open, rewrite, or delete files and performs no
network operation. It should be placed after a local scanner that has already
decided which metadata-only findings and statuses to submit. This inventory is
an operational summary, not a compliance certification or a guarantee that a
trace contains no sensitive data.
