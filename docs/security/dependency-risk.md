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

The advisory snapshot may be a plain parsed mapping, JSON text, or a local JSON
path. The lockfile may be a plain parsed mapping, TOML text, or a local TOML
path. The parser understands pip-audit's `dependencies` shape, a compact
`packages`/`advisories` shape, and enriched `results` entries that carry both
package identity and advisory data. A standard OSV batch response does not
repeat query package identities, so callers must join those identities into
the snapshot before generating this report.

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
Advisory IDs are reduced to internal SHA-256 correlation fingerprints and are
never serialized. Files, record counts, advisory fan-out, nested severity data,
indentation, and output size are bounded. Malformed-input errors use generic
messages and do not echo payload values. Use `dependency_risk_report_json` or
`write_dependency_risk_report` for deterministic JSON serialization; file
output is replaced atomically only after the complete report is rendered.

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

## Vulnerability gate exposure evidence

The CI vulnerability gate is separate from the offline risk-report API above.
`scripts/security/vulnerability_scan_gate.py` annotates each explicitly selected
lockfile finding with its exact locked package/version exposure:

```sh
uv run --no-sync python scripts/security/vulnerability_scan_gate.py \
  --report vulnerability-reports/trivy-image.json \
  --report vulnerability-reports/trivy-lockfile.json \
  --lock-report vulnerability-reports/trivy-lockfile.json \
  --lockfile uv.lock \
  --review-extra-only nltk=nltk \
  --output vulnerability-reports/vulnerability-scan-summary.json
```

The tool copies the lock into an isolated temporary project and runs
[`uv export`](https://docs.astral.sh/uv/reference/cli/#uv-export) offline with
`--frozen`, `--no-build`, `--no-python-downloads`, `--no-default-groups`,
`--no-dev` and `--no-emit-project`. It exports core, then each declared extra
separately. uv interprets transitive extras, platform/Python markers and its
resolution forks; the gate does not approximate their meaning with a plain
dependency-graph walk. No packages are installed and the workspace lock remains
unchanged. The gate requires uv with CycloneDX 1.5 export support (validated with
uv 0.9.17). Missing tooling, invalid locks or failed exports fail closed without
forwarding package-manager stderr or private paths.

Exposure includes every platform/Python branch retained by each universal
export. It describes potential installation contexts, rather than the packages
installed on the current machine. `core: true` means the exact package/version
appears in the core closure; `extras` names the single-extra installs containing
it, including any core dependencies shared by those installs. Combinations of
multiple extras are outside the report's explicit scope
`core_and_each_extra_separately_all_lock_platforms`. A locked version absent
from these contexts has status `not_in_examined_contexts`; it may still be
reachable through a combination of extras. A scanner version absent from the
lock has status `unmatched_version` and `core: null`, never a claimed absence
from core. An image finding receives no lockfile attribution even if its
package/version happens to match the lock.

The version-2 gate summary and job output contain only bounded package names,
versions, advisory identifiers, controlled severity/status/context codes and
counts, and declared extras. Scanner targets, titles, URLs, report paths and
source code are omitted. The `findings` array includes findings below the
blocking threshold so their exposure remains reviewable. Original Trivy JSON
and SARIF artifacts remain separate scanner evidence governed by the existing
workflow. Exposure cannot suppress a finding: the existing threshold,
package/target-specific waiver matching, expiry and refusal of waivers for
findings with a fixed version all still apply. Invalid or expired waiver
policy still exits with failure; its summary retains exposure with
`policy_status: invalid`, `threshold: null` and no claimed waiver/blocking
decisions.

`--review-extra-only PACKAGE=MODULE` additionally verifies that the reviewed
distribution is present only outside core and that the Python source tree
contains no direct or literal dynamic import of its module. Distribution and
import names are supplied explicitly because they can differ. This static
guard does not execute source code; computed module names and runtime imports
inside third-party integrations still require review. The unit suite checks
the current NLTK boundary and proves that a synthetic direct or literal
dynamic import fails the guard. The CI gate applies this check to `openmed/`;
if NLTK is removed from the lock or intentionally integrated, review the guard
configuration separately rather than silently weakening it.

The focused offline check is:

```sh
.venv/bin/python -m pytest tests/unit/security/test_vulnerability_scan_gate.py -q
```

This evidence supports a maintainer's allowlist review. Adding, renewing,
removing or replacing a waiver remains a separate decision; this reporting
change makes none of those changes.

## Temporary NLTK exception for the universal lockfile

On 2026-10-03, the maintainer approved a time-limited exception for
`CVE-2026-81726` / [GHSA-8mgp-746c-j5xp](https://github.com/nltk/nltk/security/advisories/GHSA-8mgp-746c-j5xp),
scoped to the `nltk` package in `uv.lock` through **2026-10-17**. At review,
upstream listed no patched release and the latest published NLTK was 3.10.3.
The finding remains present in scanner evidence; it is not considered fixed.

The affected model-artifact import/export APIs can read or write outside
configured NLTK path-security roots when given untrusted paths. OpenMed has no
direct calls to these APIs. NLTK is not in the core dependency closure or the
reference service image. The universal lock includes it through the optional
`agents`, `llamaindex`, `medspacy`, `quickumls`, and `scrubadub` extras. This
review does not prove that every downstream configuration is unexploitable:
do not let untrusted workflows select model import/export paths in those
integrations, and use OS-level containment when running untrusted code.

The policy in `deploy/security/cve-allowlist.yaml` remains at the HIGH
threshold. The exception cannot suppress a different package, artifact, or
advisory. It stops applying when the scanner reports a fixed version and
fails closed after its expiry. Upgrade to a verified patched release and
remove the exception when available; renewal requires a fresh maintainer
decision. Model training is unrelated to this dependency review.
