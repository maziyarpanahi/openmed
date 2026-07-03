# Reproducible Dependencies

OpenMed keeps dependency resolution auditable through `pyproject.toml`,
`uv.lock`, the existing `lockfile-drift` CI job, and the reproducible-lock gate.
This page explains what the gate checks, how to refresh the lock, and how to
read the reproducibility artifact.

## CI Gate

The `Reproducible Lock Gate` workflow runs when `pyproject.toml`, `uv.lock`, or
the workflow itself changes. It can also be run manually from GitHub Actions.

The workflow has three jobs:

- `lock-integrity` checks that `uv.lock` matches `pyproject.toml`, every
  non-root package has an exact version, every checked package is registry
  backed, and every checked package has at least one `sha256:` hash in its
  `sdist` or `wheels` entries.
- `reproducibility` creates two independent, cache-free frozen installs for the
  default Python 3.11 dependency environment and compares the sorted
  `uv pip freeze` output byte for byte.
- `lock-gate` is the summary status check that only passes when integrity and
  reproducibility both pass.

The root `openmed` project entry is skipped because `uv.lock` records it as the
local editable project rather than a third-party distribution.

## Digest Artifact

Successful runs upload `reproducibility-digest.txt` with three fields:

```text
digest: sha256:...
packages: 142
uv-lock-sha256: sha256:...
```

- `digest` fingerprints the sorted package/version/hash tuples checked from the
  lockfile.
- `packages` is the number of non-root packages included in that digest.
- `uv-lock-sha256` is the SHA-256 of the raw `uv.lock` file.

Compare digest artifacts between releases when you need to prove that the
dependency set did not change unexpectedly:

```bash
gh run download <run-id-1> -n reproducibility-digest -D run1/
gh run download <run-id-2> -n reproducibility-digest -D run2/
diff run1/reproducibility-digest.txt run2/reproducibility-digest.txt
```

## Refresh Workflow

Refresh `uv.lock` whenever you add, remove, or change dependencies in
`pyproject.toml`, or when the `lockfile-drift` job reports that the lock is out
of date.

1. Update the dependency declaration.

```toml
[project]
dependencies = [
    "transformers>=4.41,<5",
]
```

2. Re-resolve the lockfile.

```bash
uv lock
```

3. Verify that the lock is consistent and inspect the diff.

```bash
uv lock --check
git diff -- pyproject.toml uv.lock
```

4. Test a frozen install from the new lock.

```bash
uv sync --frozen --no-install-project
```

5. Commit the dependency declaration and lockfile together.

```bash
git add pyproject.toml uv.lock
git commit -m "deps: update <package-name> to x.y.z"
git push
```

For a new dependency, `uv add` updates `pyproject.toml` and `uv.lock` together:

```bash
uv add "httpx>=0.27,<1"
uv lock --check
uv sync --frozen --no-install-project
```

## Local Reproducibility Check

The CI workflow is the source of truth, but the install comparison can be
replicated locally before pushing:

```bash
uv sync --frozen --no-cache --no-install-project --python 3.11
uv pip freeze | sort > /tmp/openmed-resolve-1.txt

rm -rf .venv
uv sync --frozen --no-cache --no-install-project --python 3.11
uv pip freeze | sort > /tmp/openmed-resolve-2.txt

diff /tmp/openmed-resolve-1.txt /tmp/openmed-resolve-2.txt
```

If the diff is empty, the two frozen installs produced the same package set.

## Failure Guide

| Error | Meaning | Fix |
|---|---|---|
| `uv.lock is out of date` | `pyproject.toml` and `uv.lock` disagree. | Run `uv lock` and commit the updated lockfile. |
| `MISSING VERSION` | A checked lock entry has no exact version. | Re-run `uv lock`; do not edit the lockfile manually. |
| `UNSUPPORTED SOURCE` | A checked dependency is not registry backed. | Replace local, git, URL, or editable dependencies with published package releases. |
| `MISSING HASHES` | A checked dependency has no `sha256:` hash. | Re-run `uv lock`; if it persists, inspect the package publication. |
| `INVALID HASH TYPE` | A checked dependency uses a non-SHA-256 hash. | Re-resolve with `uv lock` or replace the dependency source. |
| `Non-reproducible install` | Two frozen clean installs produced different package lists. | Re-run the workflow once for transient index issues; investigate the package diff if it repeats. |
