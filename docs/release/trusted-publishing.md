# PyPI Trusted Publishing

OpenMed publishes the `openmed` wheel and source distribution through PyPI
Trusted Publishing. The release workflow does not use a stored PyPI API token:
GitHub issues an OIDC identity token to the protected publishing job, and PyPI
exchanges that identity for a short-lived upload credential.

## Workflow contract

The only PyPI publishing workflow is `.github/workflows/publish.yml`.

- It runs from `push` events for `v*` tags only.
- It does not run from `pull_request` or forked pull request events.
- The reusable provenance job in `.github/workflows/provenance.yml` builds and
  checks the distributions, generates SLSA provenance, and verifies the
  attestations before upload.
- The publish job downloads those verified distributions, uses
  `pypa/gh-action-pypi-publish`, and grants only `contents: read` plus
  `id-token: write`.
- The publish action is configured without `user`, `password`, or
  `PYPI_API_TOKEN`.
- PEP 740 Sigstore attestations are enabled for the PyPI upload, so each
  distribution is published with signed provenance tied to the same OIDC
  identity.

Do not add a second PyPI publishing workflow. Do not add `hatch publish`,
Twine upload credentials, or a `PYPI_API_TOKEN` secret back to release CI.

## PyPI project setup

Configure the trusted publisher on the PyPI `openmed` project before cutting a
tagged release:

1. Open the PyPI project settings for `openmed`.
2. Add a GitHub trusted publisher with these values:
   - Owner: `maziyarpanahi`
   - Repository name: `openmed`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. Ensure the GitHub `pypi` environment exists and is protected according to
   the release policy.

If PyPI reports an invalid publisher during release, check those four fields
first. The workflow filename and environment name must match exactly.

## Release checklist

Before pushing a version tag:

```bash
grep -R "pypa/gh-action-pypi-publish" .github/workflows/*.yml
if grep -R "PYPI_API_TOKEN\|hatch publish" .github/workflows/*.yml; then
  echo "Legacy PyPI token publishing is still present"
  exit 1
fi
.venv/bin/python -m pytest tests/ -q
```

The first command should identify exactly one workflow. The second command
should find nothing. The test command must pass before the release tag is
pushed.

After the tagged publish succeeds, verify the PyPI release page lists provenance
or attestations for the uploaded wheel and source distribution. For the
repository-level SLSA provenance check, see
[SLSA Build Provenance](../supply-chain/provenance.md).

## Token retirement

Once the trusted publisher is configured and one tagged publish succeeds, retire
the old PyPI token path:

- Delete `PYPI_API_TOKEN` from repository secrets and from the `pypi`
  environment secrets, if present.
- Remove local `.pypirc` or shell-profile entries that were only used for
  OpenMed package uploads.
- Do not recreate a broad PyPI token for CI. If an emergency manual upload is
  ever required, create a short-lived project-scoped token outside the normal CI
  path and revoke it immediately after use.
