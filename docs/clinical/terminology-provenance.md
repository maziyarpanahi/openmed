# Terminology snapshot provenance

OpenMed terminology resolution is local-first. A terminology snapshot can be
audited without retaining its terms by recording a small value-free manifest:
the source name, source release, SHA-256 checksum, import time, and an explicit
expiry policy. Snapshot bytes are used only to calculate the checksum and are
never copied into a manifest or report.

This metadata supports reproducibility and freshness review. It is provenance
for an assistive terminology workflow, not a compliance certification or a
clinical decision guarantee.

## Build a manifest

```python
from openmed.clinical.terminology.provenance import (
    ExpiryPolicy,
    build_snapshot_manifest,
)

manifest = build_snapshot_manifest(
    "synthetic local terminology",
    "2026.01",
    b"snapshot bytes supplied by a local loader",
    imported_at="2026-01-01T00:00:00Z",
    expiry_policy=ExpiryPolicy(max_age_days=365, reject_expired=True),
)

manifest.write_json("terminology-manifest.json")
```

The serialized manifest contains only metadata:

```json
{
  "checksum": "sha256:<64 hexadecimal characters>",
  "expiry_policy": {
    "max_age_days": 365,
    "reject_expired": true
  },
  "imported_at": "2026-01-01T00:00:00Z",
  "schema_version": 1,
  "source_name": "synthetic local terminology",
  "source_version": "2026.01"
}
```

For a snapshot already checked by a local loader, pass its checksum instead of
the bytes:

```python
manifest = build_snapshot_manifest(
    "synthetic local terminology",
    "2026.01",
    checksum="sha256:<64 hexadecimal characters>",
    imported_at="2026-01-01T00:00:00Z",
)
```

## Enforce expiry locally

Freshness evaluation takes an explicit reference time. This makes reports
deterministic and avoids embedding the machine clock in an audit artifact.

```python
from openmed.clinical.terminology.provenance import (
    build_freshness_report,
    require_fresh_snapshot,
)

report = build_freshness_report(
    [manifest],
    as_of="2026-02-01T00:00:00Z",
)
if not report.ok:
    require_fresh_snapshot(manifest, as_of=report.as_of)
```

`require_fresh_snapshot()` raises `SnapshotExpiredError` only when a manifest
is expired and its policy has `reject_expired=True`. A report marks snapshots
as `fresh`, `expiring`, `expired`, `not_yet_imported`, or
`no_expiry_policy`.

## Render reports

`build_provenance_report()` and `build_freshness_report()` return structured
reports with deterministic JSON and Markdown renderers. Their input manifests
are sorted by source name, source version, and checksum, so caller iteration
order does not change the output:

```python
from openmed.clinical.terminology.provenance import (
    build_provenance_report,
    render_freshness_report,
    render_provenance_report,
)

provenance = build_provenance_report([manifest])
freshness_markdown = render_freshness_report(
    [manifest],
    as_of="2026-02-01T00:00:00Z",
)
provenance_json = render_provenance_report([manifest], format="json")
```

Only provenance metadata, checksums, dates, policy decisions, and aggregate
freshness counts appear in these reports. Terminology values, caller note text,
paths containing source content, credentials, and network responses are out of
scope.
