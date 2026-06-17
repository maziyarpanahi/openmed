# HF Write Token Policy

OpenMed model publication uses a dedicated `HF_WRITE_TOKEN` secret for CI runs that write converted artifacts to the
OpenMed organization. This token is separate from runtime read tokens and package publishing credentials.

## Scope

- Create a fine-grained token with org-write access limited to the OpenMed organization.
- Do not grant admin, billing, or account-management permissions.
- Do not reuse a personal development token, read token, or package publishing token.
- Treat exposure as org-wide write access to OpenMed model repositories.

## Storage

- Store the value as the `HF_WRITE_TOKEN` secret in the `hf-publish` GitHub Actions protected environment.
- Keep the secret out of repository-level Actions secrets unless a workflow cannot be environment-bound.
- Require environment protection before jobs can read the secret.
- Only repository administrators who can manage environment secrets may replace or delete it. The saved value cannot be
  read back after it is stored.

## CI Use

The `convert-models.yml` workflow exposes the secret only to the protected `publish-hf` job. The publish guard uses a
step-level environment binding:

```yaml
environment:
  name: hf-publish
steps:
  - name: Require HF write token before publish
    env:
      HF_WRITE_TOKEN: ${{ secrets.HF_WRITE_TOKEN }}
```

The publish job must check that `HF_WRITE_TOKEN` is set before running any upload command. Logs may mention the secret
name but must never print the value.

## Rotation

- Rotate the token every 90 days, or immediately after maintainer turnover, suspicious workflow activity, or accidental
  disclosure.
- Create the replacement token first, update the `hf-publish` environment secret, then run a manual publish credential
  check before deleting the old token.
- Record the rotation date and operator in the release notes or private operations log without copying the token value.

## Revocation And Blast Radius

If `HF_WRITE_TOKEN` is exposed:

1. Revoke the token from the token provider immediately.
2. Delete or replace the `hf-publish` environment secret.
3. Disable queued or running publish workflows until the replacement secret is in place.
4. Audit model repositories in the OpenMed organization for unexpected commits, files, tags, or metadata changes.
5. Re-run the last known-good publish workflow after the audit if any artifact needs restoration.

The blast radius is org-wide write access to OpenMed model repositories. The token must not have package publishing,
repository administration, billing, or account ownership permissions.
