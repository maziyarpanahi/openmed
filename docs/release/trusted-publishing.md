# Registry Publishing

OpenMed publishes the `openmed` wheel and source distribution from the
tag-driven `.github/workflows/publish.yml` workflow. The same workflow validates
and publishes the `openmed` npm package for browsers and Node.js. Both packages
must match the release tag before either registry upload begins. A guarded
manual dispatch exists only to recover an existing immutable tag after a
workflow-only failure.

The current PyPI path uses the project-scoped `PYPI_API_TOKEN` GitHub secret.
The npm path uses GitHub Actions OIDC Trusted Publishing in the `npm`
environment, with Sigstore provenance and no stored publishing token.

PyPI Trusted Publishing is the preferred future path, but it must not be used
until the PyPI `openmed` project has a trusted publisher that exactly matches
this repository, workflow file, and GitHub environment.

## PyPI workflow contract

The only PyPI publishing workflow is `.github/workflows/publish.yml`.

- It runs automatically from `push` events for `v*` tags.
- A maintainer may manually dispatch it with an existing `vX.Y.Z` tag to
  recover a failed tag run. The workflow checks out that exact tag, verifies
  the checked-out commit against the tag, and repeats every build and
  verification gate before publishing.
- It does not run from `pull_request` or forked pull request events.
- The reusable provenance job in `.github/workflows/provenance.yml` builds and
  checks the distributions, generates SLSA provenance, and verifies the
  attestations before upload.
- The source configuration requires Core Metadata 2.4 for both wheel and sdist
  until the pinned PyPI publishing action supports Core Metadata 2.5. The
  reusable workflow additionally installs Hatchling 1.31.0 and builds without
  isolation so recovery of an older immutable tag that predates that source
  setting cannot silently select a newer incompatible backend.
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
  identity and triggering workflow revision before it is attached to the
  release, or the evidence job fails. The attached `release-source.json`
  separately records the exact immutable tag commit that was checked out,
  including during a recovery dispatch from `master`.
- Conventional commits determine the minimum safe version bump. The tag may
  intentionally select a larger minor or major version, but it must be newer
  than the previous tag, meet or exceed that minimum, and exactly match both
  package manifests.

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
  `contents: read` and `id-token: write`, and runs on a GitHub-hosted runner.
  npm exchanges that job's identity for a short-lived publishing credential;
  no `NPM_ACCESS_TOKEN` or `NODE_AUTH_TOKEN` is required.
- The publish job builds first, then
  `npm publish --ignore-scripts --access public --provenance` uploads the package
  without running lifecycle hooks in the credential-bearing step. Provenance
  links the package to the tag workflow and source commit.
- Recovery dispatches query an existing npm version before publishing.
  An existing package is skipped only when its registry `gitHead` matches the
  immutable tag and its downloaded tarball contents match a fresh tag build;
  any mismatch fails closed.
- The release SBOM job starts only after both PyPI and npm publication succeed.

Do not publish `@openmed/openmedkit-web`; the public package name is `openmed`.
Do not add another npm publishing workflow or place a plaintext npm token in an
`.npmrc` file.

### npm trusted publisher setup and recovery

In the npm `openmed` package settings, configure a GitHub Actions trusted
publisher for user `maziyarpanahi`, repository `openmed`, workflow filename
`publish.yml` (without a directory), and environment `npm`. Enable direct
`npm publish`; separate dist-tag management permission is not needed. Keep
these values aligned with the workflow. npm requires Node.js 22.14.0+ and
npm 11.5.1+; the workflow uses Node.js 24 with a compatible bundled npm CLI.

If PyPI has already published and only npm failed, retry the failed npm job
after correcting the trusted publisher. Use `gh run rerun RUN_ID --repo
maziyarpanahi/openmed --job NPM_JOB_ID` so the successful PyPI job is not
repeated. For an older tagged workflow that still checks a nonempty npm secret,
retain that secret until recovery completes: compatible npm versions try OIDC
before falling back to token authentication. Remove the obsolete secret only
after a successful OIDC publication. Do not move the existing tag.

See [npm Trusted Publishing](https://docs.npmjs.com/trusted-publishers/) for
the provider setup and supported runner requirements.

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

## v2.1.0 Incident Lessons

On 2026-08-12, Hatchling 1.32.0 began emitting Core Metadata 2.5 by default.
The `v2.1.0` provenance job selected that newly released backend because the
build requirement was unconstrained. Local `twine check` and provenance passed,
but `pypa/gh-action-pypi-publish@v1.14.1` rejected the wheel before upload. npm
had already published `openmed@2.1.0`, so a blind recovery dispatch would have
attempted to publish the immutable npm version twice.

Wheel and sdist targets now explicitly emit Core Metadata 2.4, the recovery
workflow pins Hatchling 1.31.0 for older immutable tags, and npm recovery is
content-aware and idempotent. Keep the tag immutable: recover through
`workflow_dispatch` from current `master`, which uses the repaired workflow
while checking out and verifying the original tag.

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

If an immutable tag run fails before either registry upload, fix and merge the
workflow guard first, then recover the existing tag through the same production
workflow:

```bash
gh workflow run publish.yml --ref master -f tag=vX.Y.Z
```

Do not delete, move, or recreate the tag. If npm already contains the version,
the recovery guard must prove that its `gitHead` and tarball contents match the
immutable tag before skipping npm and resuming the remaining publication jobs.
If PyPI already contains either distribution, inspect both PyPI artifacts
before any recovery dispatch because production PyPI uploads remain
non-idempotent.

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

npm uses OIDC Trusted Publishing, so publishing-token rotation is unnecessary.
After the first successful OIDC publication and any older tagged-run recovery,
delete the obsolete `NPM_ACCESS_TOKEN` GitHub environment secret. Future tags
use the workflow identity automatically; do not restore a stored npm token.
