# Secret Scanning

OpenMed uses Gitleaks to catch committed credentials, private tokens, local
credential files, and scanner canaries before they reach shared history. The
same scanner configuration is used locally by pre-commit and in GitHub Actions.

## Local Setup

Install the repository hooks after setting up the development environment:

```bash
pre-commit install
```

The Gitleaks hook is pinned in `.pre-commit-config.yaml`. It scans staged
changes with `.gitleaks.toml` and ignores only findings recorded in
`.secrets.baseline`.

Run the hook manually when changing authentication, release, deployment, or CI
files:

```bash
pre-commit run gitleaks
```

Do not bypass the hook for real credentials. If a local example needs a token,
use a placeholder such as `<TOKEN>` or keep the value in an untracked local
file.

## CI Gate

The `Secret Scan` workflow installs the pinned Gitleaks release after verifying
its checksum. It then runs two checks on every pull request and push:

1. Copy `tests/fixtures/secret_scan_canary.txt` to a temporary,
   non-allowlisted path and confirm Gitleaks fails on that synthetic secret.
2. Scan the pull request or push commit range with `.gitleaks.toml` and
   `.secrets.baseline`.

If the canary is missed, CI fails because the scanner is not working. If a new
non-baselined secret appears in the diff, CI fails and the value is redacted in
scanner output. If branch protection uses selected required checks, require the
`Secret Scan / secret-scan` check to make this gate merge-blocking.

## Baseline Maintenance

`.secrets.baseline` is a redacted Gitleaks JSON report for known synthetic
findings. Keep baseline entries narrow and only for committed synthetic
fixtures that are intentional.

When maintaining the baseline:

- Prefer changing examples to placeholders instead of adding baseline entries.
- Keep `.gitleaks.toml` allowlists scoped by path and pattern.
- Keep `.secrets.baseline` redacted; do not commit real secret values.
- Explain every new baseline or allowlist entry in the pull request.

If a scan reports a real credential, revoke or rotate it immediately, remove it
from the branch before merging, and avoid posting the value in issues, pull
requests, logs, or screenshots.
