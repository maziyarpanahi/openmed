# Secret Handling

OpenMed must not commit credentials, private tokens, patient data, or other
secrets. Keep local credentials in untracked files such as `creds.txt`,
`.pypirc`, `secrets.json`, or local environment files.

Use GitHub Actions secrets or environment secrets for CI credentials. For
Hugging Face tokens, follow `docs/security/hf-token-policy.md`.

## Local Checks

Install the repository hooks before committing:

```bash
pre-commit install
```

The Gitleaks hook uses the pinned Docker image from `.pre-commit-config.yaml`,
so Docker must be available locally.

Run the staged-change secret scanner manually when changing auth, release, or
CI files:

```bash
pre-commit run gitleaks
```

## CI Gate

CI runs Gitleaks with `.gitleaks.toml` on every pull request and push. The
workflow also creates a fake token in a temporary `creds.txt` file and expects
Gitleaks to fail that canary scan. If the canary passes, CI fails because the
secret-scanning gate is not working.

Repository admins should also enable GitHub secret scanning and push protection
where the repository plan supports it.

## False Positives

Prefer replacing realistic-looking examples with placeholders such as
`<HF_TOKEN>` or `<PYPI_TOKEN>`. If a false positive cannot be avoided, add the
narrowest possible allowlist entry in `.gitleaks.toml`, scoped by path and
pattern, and explain the reason in the pull request.

## If A Secret Is Committed

Revoke or rotate the credential immediately. Remove it from the branch before
merging, and avoid posting real secret values or private data in issues, pull
requests, logs, or screenshots.
