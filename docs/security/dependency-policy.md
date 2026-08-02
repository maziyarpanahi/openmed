# Dependency Policy

OpenMed treats vulnerable Python dependencies and GitHub Actions as release
blockers when a fixed version is available.

## CI Scanner

The CI security job installs OpenMed with development extras, then runs
`scripts/security/pip_audit_gate.py`. The gate wraps `pip-audit`, writes
`pip-audit-report.json`, and fails when an advisory has a known fixed version.
Fixable advisories must be resolved by upgrading the affected dependency.

The gate allows time-boxed ignores from `docs/security/pip-audit-ignore.toml`
for advisories that do not have a usable fix yet. Unfixable advisories without
an active ignore fail CI so exceptions stay visible. Each ignore must include:

- `id`: the vulnerability ID reported by `pip-audit`
- `reason`: why the advisory cannot be fixed immediately
- `review_by`: an ISO date for re-checking the exception

Expired ignores fail CI. Remove an ignore as soon as the dependency can be
upgraded.

## Dependabot

Dependabot checks Python packages and GitHub Actions weekly. Python dependency
updates are grouped into one pull request, and GitHub Actions updates are
grouped into one pull request. Dependabot PRs are reviewed and tested manually;
auto-merge is intentionally out of scope.

## Static Analysis

Bandit still runs in CI. The full report is uploaded as `bandit-report.json`,
and the job blocks on high-severity findings so new critical issues are not
lost in the existing lower-severity backlog.
