# Offline Dependency Risk Report

`openmed.risk.dependency_risk_report` creates a reproducible summary of a
local lockfile and a caller-supplied advisory snapshot. It is intended for
privacy-sensitive builds where package-manager or advisory-service calls are
not permitted.

The function performs no network access and does not invoke a package manager:

```python
from pathlib import Path

from openmed.risk import dependency_risk_report

report = dependency_risk_report(
    {
        "dependencies": [
            {
                "name": "demo-package",
                "version": "1.2.3",
                "vulns": [{"id": "CVE-2026-0001", "severity": "high"}],
            }
        ]
    },
    Path("uv.lock"),
)
```

The advisory snapshot may be a parsed mapping, JSON text, or a local JSON
path. The lockfile may be a parsed mapping, TOML text, or a local TOML path.
The parser understands pip-audit's `dependencies` shape, OSV's `results`
shape, and a compact `packages`/`advisories` shape.

## Correlation and categories

Every unique `name`/`version` pair in the lockfile is included in sorted order.
The unversioned editable-local project entry that `uv.lock` uses for OpenMed's
own source tree is skipped because it has no locked package version. An
advisory without a version applies to every locked version with the same
normalized package name. A versioned advisory applies only to an exact locked
version. A package with an advisory that cannot be matched to its locked
version is classified as `unknown` so a stale snapshot cannot silently look
safe.

The category is the highest normalized severity among matching advisories:

| Category | Meaning |
|---|---|
| `critical` | Critical or CVSS score at least 9.0 |
| `high` | High or CVSS score at least 7.0 |
| `medium` | Medium/moderate or CVSS score at least 4.0 |
| `low` | Low or CVSS score above 0 |
| `unknown` | An advisory exists but its severity is missing or cannot be matched |
| `none` | No matching advisory is present |

An advisory that has no recognized severity is conservatively classified as
`unknown`.

## Output safety

The serialized report contains package names, locked versions, normalized risk
categories, and aggregate counts. It intentionally omits advisory IDs,
descriptions, URLs, fixed-version lists, paths, and all other source fields.
Malformed-input errors use generic messages and do not echo payload values.
Use `dependency_risk_report_json` or `write_dependency_risk_report` for
deterministic JSON serialization.

The result has this shape:

```json
{
  "artifact": "offline_dependency_risk",
  "offline": true,
  "packages": [
    {"name": "demo-package", "risk_category": "high", "version": "1.2.3"}
  ],
  "schema_version": 1,
  "summary": {
    "affected_packages": 1,
    "advisory_matches": 1,
    "risk_categories": {
      "critical": 0,
      "high": 1,
      "medium": 0,
      "low": 0,
      "unknown": 0,
      "none": 0
    },
    "total_packages": 1,
    "unmatched_advisories": 0
  }
}
```

The report is a review aid, not a compliance certification or clinical
decision guarantee. Use synthetic advisory snapshots in tests and keep any
source snapshot governed separately from the report artifact.
