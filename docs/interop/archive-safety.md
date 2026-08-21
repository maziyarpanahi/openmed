# Offline archive safety

`openmed.interop.archive_safety` classifies archive-member metadata before a
file-redaction workflow extracts anything. It is deliberately local-only: the
module does not open archives, read member contents, follow links, or make
network calls.

## Decisions

`inspect_archive_members` returns an `ArchiveSafetyReport` with one of three
decisions:

- `allow`: the supplied metadata is within policy and has no structural
  findings.
- `quarantine`: resource limits were exceeded. A separate review must approve
  the archive before extraction.
- `reject`: traversal, links, duplicate normalized paths, malformed metadata,
  or another structural safety finding was detected. The archive must not be
  extracted.

Diagnostics contain reason codes and counts only. Member paths and link targets
are not copied into reports, serialized output, or exception messages.
Report reason iteration and member kind normalization are bounded before
caller-controlled values are processed.

```python
from openmed.interop.archive_safety import (
    ArchiveDecision,
    ArchiveSafetyPolicy,
    inspect_archive_members,
)

policy = ArchiveSafetyPolicy(
    max_entries=1_000,
    max_total_uncompressed_bytes=64 * 1024 * 1024,
    max_expansion_ratio=50,
)
report = inspect_archive_members(
    [
        {
            "path": "records/summary.txt",
            "compressed_size": 120,
            "uncompressed_size": 480,
            "kind": "file",
        }
    ],
    policy,
)

assert report.decision is ArchiveDecision.ALLOW
```

The evaluator treats both `/` and `\\` as archive separators, rejects
absolute and parent-traversal paths, detects normalized duplicate paths,
rejects symbolic and hard links, and quarantines entry-count, size, and
expansion-ratio limit breaches. The default limits are intentionally bounded;
callers may tighten them for a specific workflow.
