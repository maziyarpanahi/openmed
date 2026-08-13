# Audit-artifact retention

Audit evidence should remain useful without becoming a second store of source
documents or identifiers. `openmed.risk.audit_retention` provides a local,
deterministic retention planner for counts-only audit artifacts.

The planner accepts an opaque artifact identifier, a creation timestamp, an
explicit disposition, and numeric counters. It applies a caller-supplied
`RetentionRule` for each disposition. A `delete` rule removes artifacts at or
after its age boundary from the planned retained set; a `retain` rule provides
an indefinite hold. Missing dispositions fail closed rather than silently
using an implicit default.

```python
from datetime import datetime, timedelta, timezone

from openmed.risk.audit_retention import (
    AuditRetentionPolicy,
    RetentionRule,
    scrub_audit_artifacts,
)

as_of = datetime(2026, 8, 9, tzinfo=timezone.utc)
policy = AuditRetentionPolicy(
    rules={
        "operational": RetentionRule(max_age=timedelta(days=30)),
        "legal_hold": RetentionRule(max_age=None, action="retain"),
    }
)
report = scrub_audit_artifacts(
    [
        {
            "artifact_id": "local-artifact-a",
            "created_at": "2026-08-01T00:00:00Z",
            "disposition": "operational",
            "counts": {"masked": 12, "reviewed": 4},
        }
    ],
    policy,
    as_of=as_of,
)

if report.deleted_artifact_count:
    # Delete matching records in the caller's local store, then verify the
    # resulting counts-only iterable before committing that store update.
    remaining_artifacts = load_remaining_counts_only_artifacts()
    assert report.verify_remaining_artifacts(remaining_artifacts)
```

## Safety and verification

The function is a pure planning operation. It does not perform deletion, read
the network, use credentials, or write a file. The storage adapter that owns
the records is responsible for applying `deleted_artifacts` and must retain
the report alongside its local audit controls.

Each artifact is committed to a deterministic SHA-256 fingerprint. The report
contains the input and remaining-set fingerprints, a fingerprint for the
deletion evidence, aggregate counter totals, dispositions, ages, and an
integrity digest. It never serializes artifact identifiers, creation
timestamps, counter names, source paths, text, or exception payloads. After a
deletion pass, `verify_remaining_artifacts()` recomputes the fingerprint from
the caller's remaining counts-only records and detects omission, insertion,
reordering-independent mutation, or retention of an expired artifact.

The evaluation timestamp is explicit so the same inputs and policy produce the
same report. Use synthetic offline fixtures in tests and do not place source
text, patient identifiers, encounter identifiers, or paths in audit-artifact
records.

This evidence supports local retention verification. It is not a compliance
certification, a legal determination, or a clinical decision.
