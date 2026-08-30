# Manual Hugging Face Model Publication Policy

OpenMed does not convert or publish model artifacts from GitHub Actions. The
repository must not store a Hugging Face write token in GitHub Actions secrets,
and no workflow may receive `HF_WRITE_TOKEN`. Model releases are explicit local
operations performed on hardware selected and controlled by a maintainer.

This policy is separate from package publishing credentials and runtime read
tokens.

## Execution Boundary

- Build, convert, evaluate, and gate the candidate on a maintainer-controlled
  machine with sufficient storage, memory, and accelerator support.
- Review the exact source model, target model repository, artifact directory,
  format, gate evidence, and intended repository visibility before publishing.
- Publish only through an explicit local command initiated for that candidate.
- Do not add a cron trigger, hosted conversion job, hosted publish job, or CI
  environment carrying a Hugging Face write token.
- Keep package releases and model releases independent.
- Never treat model publication as authorization to create, modify, or change
  the visibility of a Hugging Face Space.

## Token Scope And Storage

- Use a fine-grained token with org-write access limited to the OpenMed model
  repositories required for the release.
- Do not grant admin, billing, account-management, or Space-management
  permissions.
- Load the token from a local secret manager only for the explicit local
  command, then remove it from the process environment.
- Never store the token in repository files, shell history, logs, workflow
  secrets, workflow artifacts, or release evidence.
- Do not reuse a personal development token, read token, or package publishing
  token.

Treat exposure as org-wide write access to OpenMed model repositories.

## Manual Release Sequence

1. Validate the reviewed queue or candidate configuration without publishing.
2. Build the artifact and run the applicable release gates locally.
3. Inspect the artifact inventory, hashes, generated model card, target model
   repository, and visibility decision.
4. Set `HF_WRITE_TOKEN` from the local secret manager and run the explicit
   local publish command.
5. Verify the uploaded revision and perform the synthetic smoke test.
6. Unset the token and retain only PHI-free hashes, offsets, and gate evidence.

Local tools must fail closed when the token or required evidence is missing.
They must not log token values or change repository visibility as a side effect
of uploading files.

## Rotation And Revocation

- Rotate a publication token every 90 days, or immediately after maintainer
  turnover, suspicious activity, or accidental disclosure.
- Create and verify the replacement before revoking the old token.
- Record the rotation date and operator in a private operations log without
  copying the token value.

If a token is exposed:

1. Revoke the token immediately.
2. Stop any local publication process using it.
3. Audit OpenMed model repositories for unexpected commits, files, tags, or
   metadata changes.
4. Restore affected artifacts from reviewed, hash-verified local evidence.
