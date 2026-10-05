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

## Optional model runtimes

Torch-backed extras require `torch>=2.13.0`, including extras that bring Torch
transitively through model, quantization, or OCR packages. The same floor applies
to current container, standalone, model-smoke, and conversion recipes. Torch
remains optional: the base SDK, CLI, MCP, and minimal ONNX Runtime profiles do
not install it. The SDK still supports Python 3.10 and newer.

This floor includes the fixes for the
[weights-only checkpoint advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p),
[TorchScript advisory](https://github.com/advisories/GHSA-rrmf-rvhw-rf47), and
[LSTM-cell advisory](https://github.com/advisories/GHSA-qfhq-4f3w-5fph).
It is a dependency boundary, not a claim that every model artifact or converter
has been qualified. Do not enable remote model code or relax artifact-integrity
controls to work around an incompatible model.

Torch 2.13 publishes Python 3.10 wheels for Linux aarch64/x86_64, Windows AMD64,
and macOS 14+ Apple Silicon. It does not provide an Intel macOS wheel. A failed
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
