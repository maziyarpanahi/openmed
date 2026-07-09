# Secret Handling

OpenMed must not commit credentials, private tokens, patient data, or other
secrets. Keep local credentials in untracked files such as `creds.txt`,
`.pypirc`, `secrets.json`, or local environment files.

Use GitHub Actions secrets or environment secrets for CI credentials. For
model publishing credentials, follow `docs/security/hf-token-policy.md`.

## Local Checks

Install the repository hooks before committing:

```bash
pre-commit install
```

The secret-scanning hook is pinned in `.pre-commit-config.yaml` and uses the
rules in `.gitleaks.toml` plus the committed `.secrets.baseline`. See
[Secret Scanning](secret-scanning.md) for local setup, CI behavior, and
baseline maintenance.

Run the staged-change secret scanner manually when changing auth, release, or
CI files:

```bash
pre-commit run gitleaks
```

## CI Gate

CI installs the pinned scanner release after verifying its checksum, then scans
the committed changes in every pull request and push. The workflow also copies
`tests/fixtures/secret_scan_canary.txt` to a temporary, non-allowlisted path
and expects the scanner to fail that canary scan. If the canary is missed, CI
fails because the gate is not working.

Repository admins should also enable GitHub secret scanning and push protection
where the repository plan supports it.

## False Positives

Prefer replacing realistic-looking examples with placeholders such as
`<TOKEN>` or `<PYPI_TOKEN>`. If a false positive cannot be avoided, add the
narrowest possible allowlist entry in `.gitleaks.toml` or redacted baseline
entry in `.secrets.baseline`, scoped by path and pattern, and explain the
reason in the pull request.

## If A Secret Is Committed

Revoke or rotate the credential immediately. Remove it from the branch before
merging, and avoid posting real secret values or private data in issues, pull
requests, logs, or screenshots.
