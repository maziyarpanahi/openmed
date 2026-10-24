# Dependency Policy

OpenMed treats vulnerable Python dependencies and GitHub Actions as release
blockers when a fixed version is available.

## CI Scanner

The CI security job installs OpenMed with development extras, then runs
`scripts/security/pip_audit_gate.py`. The gate wraps `pip-audit`, writes
`pip-audit-report.json`, and fails when an advisory has a known fixed version.
Fixable advisories must be resolved by upgrading the affected dependency.

The gate allows time-boxed ignores from `docs/security/pip-audit-ignore.toml`
for advisories that do not have a usable fix yet. Unfixable advisories without
an active ignore fail CI so exceptions stay visible. Each ignore must include:

- `id`: the vulnerability ID reported by `pip-audit`
- `reason`: why the advisory cannot be fixed immediately
- `review_by`: an ISO date for re-checking the exception

Expired ignores fail CI. Remove an ignore as soon as the dependency can be
upgraded.

## Image dependencies and Python compatibility

The `agents`, `beam`, `llamaindex`, `mlx`, `multimodal`, and `ocr-paddle` extras
require Pillow 13 in both wheel metadata and the frozen lock.
[Pillow 13 drops Python 3.10](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html#python-3-10),
which OpenMed no longer supports either. The base SDK remains Pillow-free.
Upgrade legacy image environments before processing untrusted input; vulnerable
Pillow is unsupported. The lock also requires Certifi 2026.10.19. No audit
exception is added.

Pillow tests run on every supported Python version. Beam needs the explicit,
DirectRunner-tested
[installation overrides](../integrations/beam.md); pip does not inherit uv
project overrides. Other runners and image transforms need separate validation.
Brand checks retain generator-version provenance, master bytes, derivative
pixels and committed hashes; upgrades do not authorize image drift.

## Optional storage dependencies

The v3.1 release review raises the existing optional dependency closures to
`aiohttp>=3.14.4`, `banks>=2.5.1`, `mako>=1.4.2`, `multidict>=6.9.1` and
`tinytag>=2.3.0` for the fixes reported by the October 20 lockfile scan.
Each affected extra publishes its minimum version, and the universal lock
uses the same floors. No package is added to the core install; AWQ retains
its Linux-only markers. The unpatched NLTK finding is evaluated separately
under the [time-limited dependency review](dependency-risk.md).

Every optional profile that resolves `fsspec` requires `fsspec>=2026.6.0` in
published package metadata as well as the frozen dependency graph. The same
profiles require `jinja2>=3.1.6`, since an older sandbox can still admit
[private traversal through the attr/format filter](https://github.com/pallets/jinja/security/advisories/GHSA-cpwx-vrp4-4pq7).
This includes
model and orchestration profiles that reach filesystem support transitively;
the base SDK still does not install it. The AWQ floor retains its Linux-only
marker. The frozen cloud profile pairs fsspec 2026.6.0 with s3fs 2026.6.0 to
respect the datasets and S3 adapter version bounds.

The floor addresses
[reference-template code execution](https://github.com/fsspec/filesystem_spec/security/advisories/GHSA-27vj-qcqg-25rc).
Synthetic regressions cover all three reference-template paths, rejecting
private attribute traversal and indirect format access while retaining ordinary
variable interpolation.
Filesystem initialization happens before downstream protocol checks, so those
checks are not a substitute for installing the patched dependency.

## Optional graph and Beam dependencies

The `langgraph` and `agents` profiles require `langgraph-sdk>=0.4.4` to address
[resource-scoped authorization action handling](https://github.com/langchain-ai/langgraph/security/advisories/GHSA-fvww-7h3r-vfhp).
The `beam` profile requires `pymongo>=4.18.2` for the upstream URI host-parsing
and BSON buffer-size fixes. Both floors are published and mirrored in the frozen
resolver constraints without overriding parent dependency bounds. Neither
package is added to the base or development-only environment.

PyMongo 4.18 requires MongoDB Server 4.4 or newer. OpenMed has no native MongoDB
storage API, but callers using upstream Beam MongoDB IO must account for that
[driver compatibility change](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/reference/upgrade/).
No scanner threshold or vulnerability exception is changed by these upgrades.

## Optional model runtimes

Torch-backed extras require `torch>=2.13.0`, including extras that bring Torch
transitively through model, quantization, or OCR packages. The same floor applies
to current container, standalone, model-smoke, and conversion recipes. Torch
remains optional: the base SDK, CLI, MCP, and minimal ONNX Runtime profiles do
not install it. The SDK supports Python 3.11 and newer.

This floor includes the fixes for the
[weights-only checkpoint advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p),
[TorchScript advisory](https://github.com/advisories/GHSA-rrmf-rvhw-rf47), and
[LSTM-cell advisory](https://github.com/advisories/GHSA-qfhq-4f3w-5fph).
It is a dependency boundary, not a claim that every model artifact or converter
has been qualified. Do not enable remote model code or relax artifact-integrity
controls to work around an incompatible model.

Torch 2.13 publishes wheels for Linux aarch64/x86_64, Windows AMD64, and macOS
14+ Apple Silicon. It does not provide an Intel macOS wheel. A failed
optional runtime installation on an unsupported platform must not downgrade to
a vulnerable release or raise the base SDK's Python minimum. Use a supported
runtime/platform or an independently validated non-Torch path.
The PyPI Linux Torch 2.13 distribution depends on CUDA 13 packages; recheck GPU
architecture and driver compatibility before deployment. The reference Docker
recipe selects the upstream CPU wheel index instead. macOS CPU/CoreML smoke
tests do not qualify CUDA, AWQ, or GPTQ execution on a production GPU.

Audit the exact optional profile used by a deployment, not only the CI `dev`
environment. For example, after a frozen `dev,hf,service` sync, run the same
dependency gate without changing that environment. Advisory IDs without usable
fixes still require the explicit, time-boxed process above; a Torch upgrade is
not an exception for other packages. Prefer operator-selected, pinned, trusted
artifacts and safetensors where available, but do not treat those controls as a
substitute for dependency updates or a general guarantee about native parsers.

The scanned root and published multi-architecture service containers use the
same digest-pinned runtime bases and library/tooling repairs. Their build context
copies only the Python package, required package metadata, retained registry
files and bundled synthetic red-team fixture, not unrelated desktop source
trees. Dependencies are checked before the installer is removed from the final
runtime. Add optional packages by rebuilding the image; the production image is
not an interactive package-installation environment. Scan the actual final
image, including vendored dependencies, rather than assuming the installed
top-level versions or the repository lock describe all of its contents.

The PyTorch CPU wheel index also serves its own, older copies of Torch's
dependencies. The service images therefore install patched PyPI releases of
`filelock` and `setuptools` before Torch, so Torch's unpinned `filelock`
requirement is already satisfied and the CPU index cannot replace it. The
universal lock separately requires `brotlicffi>=1.2.1.0` and
`setuptools>=84.1.0`.

## Python 3.11 floor and the Snowflake filelock override

OpenMed requires Python 3.11 or newer. Python 3.10 reached end of life in
October 2026, and `filelock` 4.1.1 and later, including the 4.2.1 release that
fixes CVE-2026-112940, CVE-2026-112913 and CVE-2026-112907, require Python 3.11.

Every `snowflake-snowpark-python` release requires
`snowflake-connector-python<5`, and the connector's 4.x releases cap
`filelock<4`. The universal lock therefore overrides that cap with
`filelock>=4.2.1` rather than dropping the `snowflake` extra. The connector uses
only `FileLock`, `BaseFileLock` and `Timeout` for its OCSP and CRL caches.
filelock 4's breaking change is limited to the `SoftReadWriteLock` on-disk
protocol, which the connector does not use. Its cache save, lock-timeout,
fork, thread-local CRL and cleanup paths were exercised offline with filelock
4.2.4 before the override was adopted. Recheck this when the connector or
Snowpark changes its `filelock` or connector bounds, and remove the override
once Snowpark accepts a connector release without the cap.

The override has two visible effects. `uv pip check` reports the connector's
declared `filelock<4` requirement in a frozen `snowflake` environment, and pip
does not inherit uv project overrides, so `pip install "openmed[snowflake]"`
still resolves a vulnerable filelock 3.x release. pip users must install
`filelock>=4.2.1` afterwards, as the [Snowflake guide](../snowflake-udf.md)
shows.

## Dependabot

Dependabot checks Python packages and GitHub Actions weekly. Python dependency
updates are grouped into one pull request, and GitHub Actions updates are
grouped into one pull request. Dependabot PRs are reviewed and tested manually;
auto-merge is intentionally out of scope.

## GitHub Actions references and permissions

Remote actions in workflows and local composite actions must use full commit
SHAs, with the upstream version in a comment. Resolve each SHA from the action's
own repository; a pin freezes that revision but does not establish that its code
is safe. Keep Dependabot's GitHub Actions updates enabled and review the upstream
changes before accepting a new pin.

The CI repository-policy job enforces immutable references. Run the same checks
locally without network access:

```bash
python scripts/release/check_github_actions_refs.py --require-sha
python scripts/release/check_github_actions_refs.py --require-sha --workflows-dir .github/actions
```

Grant publishing, attestation, and OIDC permissions only to jobs that use them.
The container workflow's test jobs use `contents: read`; its publish job receives
the write permissions after both test jobs pass and is excluded from pull requests.

## Static Analysis

Bandit still runs in CI. The full report is uploaded as `bandit-report.json`,
and the job blocks on high-severity findings so new critical issues are not
lost in the existing lower-severity backlog.
