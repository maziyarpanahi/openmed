# PyPI Publishing

OpenMed publishes the `openmed` wheel and source distribution from the
tag-driven `.github/workflows/publish.yml` workflow. The current production
path uses the project-scoped `PYPI_API_TOKEN` GitHub secret for upload, after a
separate provenance job builds, checks, attests, and verifies the distributions.

PyPI Trusted Publishing is the preferred future path, but it must not be used
until the PyPI `openmed` project has a trusted publisher that exactly matches
this repository, workflow file, and GitHub environment.

## Workflow contract

The only PyPI publishing workflow is `.github/workflows/publish.yml`.

- It runs from `push` events for `v*` tags only.
- It does not run from `pull_request` or forked pull request events.
- The reusable provenance job in `.github/workflows/provenance.yml` builds and
  checks the distributions, generates SLSA provenance, and verifies the
  attestations before upload.
- The publish job downloads those verified distributions, uses
  `pypa/gh-action-pypi-publish`, and grants only `contents: read`.
- The publish job attaches the `pypi` GitHub environment so it can read the
  environment-scoped `PYPI_API_TOKEN` secret. That environment must not have
  reviewer, wait-timer, or branch-policy gates that block tagged releases.
- The publish action is configured with `password: ${{ secrets.PYPI_API_TOKEN }}`
  and `attestations: false`.
- PyPI-native PEP 740 attestations are disabled while token upload is active,
  because the PyPA action supports those attestations only with Trusted
  Publishing. The repository-level SLSA provenance artifact is still generated
  and verified before upload.

Do not add a second PyPI publishing workflow. Do not add `hatch publish` or
Twine upload commands back to release CI.

## v1.8.0 Incident Lessons

On 2026-07-09, the first `v1.8.0` PyPI publish failed because the release
workflow used `pypa/gh-action-pypi-publish` without a password. In that mode,
the action falls back to Trusted Publishing. PyPI rejected the GitHub OIDC
exchange with `invalid-publisher` because the `openmed` project did not have a
trusted publisher matching this repository, `publish.yml`, and the `pypi`
environment.

Two follow-up checks matter:

- `PYPI_API_TOKEN` is currently an environment secret on the GitHub `pypi`
  environment, not a repository secret. The publish job must keep
  `environment: pypi` while token upload is active.
- GitHub OIDC attestation retrieval can fail independently of PyPI upload. SLSA
  provenance should be attempted and verified when GitHub's identity-token
  service is healthy, but transient attestation failures must not hide the
  separate PyPI credential contract.

The regression tests in `tests/unit/test_publish_workflow_version.py` and
`tests/unit/release/test_provenance_workflow.py` are the local guardrails for
this contract. Update them in the same change as any PyPI release workflow
change.

## PyPI project setup

To migrate back to Trusted Publishing, configure the trusted publisher on the
PyPI `openmed` project first:

1. Open the PyPI project settings for `openmed`.
2. Add a GitHub trusted publisher with these values:
   - Owner: `maziyarpanahi`
   - Repository name: `openmed`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. Ensure the GitHub `pypi` environment exists and is not blocking tagged
   releases with approval, wait-timer, or branch-policy gates.
4. Remove the `password` input from the publish action, grant the publish job
   `id-token: write`, and set `attestations: true`.

If PyPI reports an invalid publisher during release, check those fields first.
The workflow filename and optional environment name must match exactly.

## Release checklist

Before pushing a version tag:

```bash
grep -R "pypa/gh-action-pypi-publish" .github/workflows/*.yml
if grep -R "hatch publish" .github/workflows/*.yml; then
  echo "Legacy Hatch publishing is still present"
  exit 1
fi
.venv/bin/python -m pytest tests/ -q
```

The first command should identify exactly one workflow. The second command
should find nothing. The test command must pass before the release tag is
pushed.

After the tagged publish succeeds, verify the PyPI release page lists the
uploaded wheel and source distribution. For the repository-level SLSA
provenance check, see [SLSA Build Provenance](../supply-chain/provenance.md).

## Token Handling

Keep `PYPI_API_TOKEN` project-scoped and rotate it if there is any evidence of
exposure. Once the trusted publisher is configured and one tagged publish
succeeds through the tokenless path, retire the token path:

- Delete `PYPI_API_TOKEN` from repository secrets and from the `pypi`
  environment secrets, if present.
- Remove local `.pypirc` or shell-profile entries that were only used for
  OpenMed package uploads.
- Do not recreate a broad PyPI token for CI. If an emergency manual upload is
  ever required, create a short-lived project-scoped token outside the normal CI
  path and revoke it immediately after use.
