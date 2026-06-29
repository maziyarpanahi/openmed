# Reproducible Dependencies
 
OpenMed pins every transitive dependency to an exact version and a sha256
integrity hash.  This document explains how the lock file works, what the
reproducibility gate checks, and how to refresh everything when dependencies
change.

## The CI gate (`reproducible-lock.yml`)
 
The gate runs automatically on every push or pull request that touches
`uv.lock` or `pyproject.toml`.  It can also be triggered manually from the
Actions tab.  It contains three jobs that must all pass before the gate is
considered green.
 
### Job 1 — `lock-integrity`
 
Parses `uv.lock` directly and enforces:
 
- Every package has an exact pinned version (no floating specifiers).
- Every package was sourced from a registry (PyPI).  Dependencies declared via
  `git+`, a local path, or an editable install are **rejected** because they
  cannot carry content-addressable hashes and make the environment
  non-reproducible across machines.
- Every package has at least one `sha256` integrity hash recorded in either its
  `sdist` or `wheels` entries.
On success it produces a **reproducibility digest** — a single `sha256:` hash
that fingerprints the complete dependency set — and uploads it as a CI artifact
called `reproducibility-digest`.
 
### Job 2 — `reproducibility`
 
Installs the full dependency set twice in completely independent, cache-free
virtual environments and asserts the two resulting package lists are
byte-for-byte identical.  Both resolve snapshots are uploaded as the
`resolve-snapshots` artifact (even on failure) so any divergence can be
audited.
 
### Job 3 — `lock-gate`
 
A summary job that only passes if both jobs above passed.  Branch protection
rules and downstream workflows reference this job by name (`lock-gate`) as
their required status check.
 
---
 
## When to refresh the lock
 
Refresh `uv.lock` whenever you:
 
- Add, remove, or change a dependency in `pyproject.toml`.
- Want to pull in newer patch releases of existing dependencies.
- See the `lockfile-drift` job in `ci.yml` fail because `pyproject.toml` was
  edited without updating the lock.
---
 
## How to refresh
 
### 1. Update `pyproject.toml`
 
Add, remove, or change the constraint you need.  For example:
 
```toml
# pyproject.toml
[project]
dependencies = [
    "transformers>=4.41,<5",   # changed upper bound
]
```
 
### 2. Re-resolve and update the lock file
 
```bash
uv lock
```
 
This regenerates `uv.lock` in place.  The new file will contain updated version
pins and fresh sha256 hashes for every affected package and all of its
transitive dependencies.
 
### 3. Verify the new lock locally
 
```bash
# Confirm the lock is consistent with pyproject.toml
uv lock --check
 
# Install from the new lock into a clean venv
uv sync --frozen --no-install-project
 
# Inspect what changed
git diff uv.lock
```
 
### 4. Run the reproducibility gate locally (optional but recommended)
 
The integrity script is plain Python with no dependencies beyond the stdlib.
You can run it directly before pushing:
 
```bash
python3 scripts/release/check_lock_integrity.py
```
 
Or if you want to replicate the two-resolve check:
 
```bash
# Resolve 1
uv sync --frozen --no-cache --no-install-project
uv pip freeze | sort > /tmp/resolve-1.txt
 
# Resolve 2
rm -rf .venv
uv sync --frozen --no-cache --no-install-project
uv pip freeze | sort > /tmp/resolve-2.txt
 
# Compare
diff /tmp/resolve-1.txt /tmp/resolve-2.txt && echo "Reproducible."
```
 
### 5. Commit both files
 
```bash
git add pyproject.toml uv.lock
git commit -m "deps: update <package-name> to x.y.z"
git push
```
 
The `reproducible-lock.yml` gate will fire automatically on the push and
validate the new lock.
 
---
 
## Reading the reproducibility digest artifact
 
After every successful gate run, a `reproducibility-digest` artifact is
available in the Actions run summary.  Download it and inspect:
 
```
digest: sha256:8f2a9c1e...
packages: 142
uv-lock-sha256: sha256:3d7b0fa2...
```
 
- **`digest`** — fingerprint of the full dependency tree (name, version, and
  all hashes for every package, sorted and hashed together).  Compare this
  across releases to confirm the dependency set did not change unexpectedly.
- **`packages`** — total number of resolved packages checked.
- **`uv-lock-sha256`** — hash of the raw `uv.lock` file itself.
To compare two releases:
 
```bash
gh run download <run-id-1> -n reproducibility-digest -D run1/
gh run download <run-id-2> -n reproducibility-digest -D run2/
diff run1/reproducibility-digest.txt run2/reproducibility-digest.txt
```
 
If the digests match, the dependency set is identical between the two runs.
 
---
 
## Resolving gate failures
 
| Error message | Cause | Fix |
|---|---|---|
| `uv.lock is out of date` | `pyproject.toml` was edited without running `uv lock` | Run `uv lock` and commit `uv.lock` |
| `MISSING VERSION` | A package entry in `uv.lock` has no pinned version | Should not happen with `uv lock`; re-run `uv lock` |
| `UNPINNABLE SOURCE` | A dependency is declared as `git+`, a local path, or editable | Replace with a published PyPI release |
| `MISSING HASHES` | A package has no sha256 hashes in its lock entry | Re-run `uv lock`; if persists, the package may lack published wheels/sdist |
| `Non-reproducible install` | Two clean installs produced different package lists | Usually a transient network issue; re-run the workflow. If it persists, open an issue. |
 
---
 
## Adding a new dependency: full example
 
```bash
# 1. Declare it
uv add "httpx>=0.27,<1"          # updates pyproject.toml and uv.lock together
 
# 2. Verify
uv lock --check
uv sync --frozen --no-install-project
 
# 3. Commit
git add pyproject.toml uv.lock
git commit -m "deps: add httpx for async HTTP client support"
git push
```
 
`uv add` handles steps 1 and 2 of the manual refresh flow in one command.  The
CI gate then validates the result automatically.
 