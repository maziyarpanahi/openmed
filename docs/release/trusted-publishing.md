# Registry Publishing

OpenMed publishes the `openmed` wheel and source distribution from the
tag-driven `.github/workflows/publish.yml` workflow. The same workflow validates
and publishes the `openmed` npm package for browsers and Node.js. Both packages
must match the release tag before either registry upload begins.

The current PyPI path uses the project-scoped `PYPI_API_TOKEN` GitHub secret.
The npm path uses the short-lived `NPM_ACCESS_TOKEN` secret from the GitHub
`npm` environment and publishes with Sigstore provenance.

PyPI Trusted Publishing is the preferred future path, but it must not be used
until the PyPI `openmed` project has a trusted publisher that exactly matches
this repository, workflow file, and GitHub environment.

## PyPI workflow contract

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
- The evidence job runs after the publish job, so it cannot gate the PyPI
  upload on GitHub OIDC or Sigstore availability. Signing is best effort, but
  evidence that is produced must verify against the `publish.yml` workflow
  identity and the release commit before it is attached to the release, or the
  evidence job fails.

Do not add a second PyPI publishing workflow. Do not add `hatch publish` or
Twine upload commands back to release CI.

## npm workflow contract

The JavaScript package source lives in `js/openmedkit-web`, but it is published
under the existing unscoped npm name `openmed`.

- `js/openmedkit-web/package.json`, `openmed/__about__.py`, and the `v*` tag must
  contain the same semantic version.
- The `npm-verify` job uses Node.js 24, installs only from the committed lockfile,
  rejects any npm audit finding, builds both ESM and CommonJS distributions,
  typechecks the public API, runs the Web runtime tests, and inspects the package
  tarball.
- PyPI and npm publication both depend on the Python provenance job and the npm
  verification job.
- The `npm-publish` job attaches the `npm` GitHub environment, grants
  `contents: read` and `id-token: write`, and reads only the environment-scoped
  `NPM_ACCESS_TOKEN` secret.
- The publish job builds before exposing the token, then
  `npm publish --ignore-scripts --access public --provenance` uploads the package
  without running lifecycle hooks in the credential-bearing step. Provenance
  links the package to the tag workflow and source commit.
- The release SBOM job starts only after both PyPI and npm publication succeed.

Do not publish `@openmed/openmedkit-web`; the public package name is `openmed`.
Do not add another npm publishing workflow or place a plaintext npm token in an
`.npmrc` file.

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

Keep `NPM_ACCESS_TOKEN` scoped to the npm `openmed` package with publish access,
store it only in the GitHub `npm` environment, and rotate it before its 90-day
expiry. The package already exists, so npm Trusted Publishing can replace the
token without a bootstrap release. When migrating, configure npm for owner
`maziyarpanahi`, repository `openmed`, workflow `publish.yml`, and environment
`npm`; then remove `NODE_AUTH_TOKEN` from the publish step and delete the secret.
