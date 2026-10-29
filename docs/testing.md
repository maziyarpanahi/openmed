# Testing & QA

Keep regressions away from your clinical workflows by leaning on the existing test suite and smoke runners. This page
summarizes what already exists so you can extend it confidently.

## Actively tested Python and desktop matrix

The core SDK supports **CPython 3.11, 3.12 and 3.13** on Linux, Windows
and macOS. These versions and OS families are declared in `pyproject.toml`'s
classifiers. `requires-python >=3.11` is an installation floor, not a promise
that an unlisted future interpreter or alternative Python implementation is
qualified. Adding one requires a reviewed metadata, CI and documentation change.
Python 3.10 reached end of life in October 2026 and is no longer supported. Every
supported interpreter provides the standard-library `tomllib`, so the SDK has no
TOML backport dependency.

| Lane | Required combinations | What it proves |
| --- | --- | --- |
| `sdk-compatibility` core contracts | Pushes and `full-ci` pull requests: all three versions × `ubuntu-latest`, `windows-latest`, `macos-latest` (9 jobs). Other pull requests: Linux 3.11 and 3.13, Windows 3.12 and macOS 3.12 (4 jobs) | Privacy sentinels, full traceback rendering, platform filesystem contracts and smoke-runner negative tests |
| `sdk-compatibility` installed artifacts | The same jobs; both wheel and sdist; core and `[cli,service,mcp]` installs separately | Installed CLI and brief API/adapters, packaged resources, missing-resource rejection and source-tree isolation |
| Optional-runtime absence | Both install profiles in every compatibility job | Typed missing-MLX error and fail-closed brief; no cloud fallback, weights or model downloads |
| `test` full regression | Pushes and `full-ci` pull requests: Linux, Windows and macOS 3.11–3.12 | The complete Python regression suite, including the `slow` and `fuzz` tiers |
| `test` pull-request tier | Linux 3.11 and 3.12 (fast tier); Windows and macOS 3.12 (portability checks) | Every test except `slow` and `fuzz` on Linux, plus the platform-sensitive portability list on Windows and macOS |

Adding the `full-ci` label starts a new full CI run. Other label additions do
not start CI jobs.

The stable `sdk-compatibility-gate` requires every compatibility job in the run to succeed
and is a dependency of the existing `build` gate. Failed, skipped or cancelled
matrix execution cannot count as passing. Jobs run independently (`fail-fast:
false`) with a 25-minute limit. The matrix uses standard hosted-runner hardware;
it does not qualify every processor architecture, Windows ACL policy, GPU or
optional native backend.

Successful core installation does **not** imply optional model-runtime support.
MLX/Core ML execution needs supported Apple hardware and its declared extras;
other native adapters have their own runtime requirements. Runtime-specific
integration/model evaluation remains separate from this no-weights SDK gate.
An unavailable optional runtime must return its documented typed error/refusal;
a core import, CLI, privacy or resource failure is always a broken lane, never
a blanket platform skip. No trained checkpoint is a prerequisite for SDK release.

See [offline install smoke checks](install/smoke-check.md) for the exact local
artifact commands, synthetic-provider boundaries and value-free JSON reports.

## Test taxonomy

| Marker | Location | Purpose |
| --- | --- | --- |
| `unit` (default) | `tests/unit/**` | Fast validation of model registry helpers, config utilities, and core APIs. |
| `integration` | opt-in | Exercises multi-component flows (e.g., pipeline creation + formatter). |
| `slow` | any | Resource-scaling, cross-process, subprocess end-to-end and package-wide scan tests. Skipped on pull requests; run on pushes to `master` and on pull requests labelled `full-ci`. |
| `fuzz` | any | Hypothesis property tests. Same schedule as `slow`. |

Configure these markers via `pytest.ini` entries in `pyproject.toml`.

## Running the suite

```bash
uv pip install ".[dev,hf]"
make lint
make format-check
make lint-swift             # for Swift/OpenMedKit changes
pytest                      # fast unit/integration mix
pytest -n auto -m "not slow and not fuzz"   # what a pull request runs
pytest -n auto                              # the full suite
pytest -m slow              # only long-running cases
```

### Keeping the suite fast and stable

- Mark a test `@pytest.mark.slow` when it measures memory or scaling, starts a
  fresh interpreter or CLI subprocess, builds git repositories, or scans the
  whole package. Pull requests skip it; `master` runs it on every push.
- Do not assert wall-clock durations. Shared runners vary too much for timing
  thresholds to say anything about a regression.
- Do not pin exact byte sizes, digests of incidental output, or the versions
  currently in `uv.lock`. The dependency and size gates enforce those; a unit
  test that repeats them only fails on unrelated changes.
- Build expensive fixtures once (module or session scope, or a cached
  template that each test copies) instead of rebuilding them per case.
- Sweep a data table (every locale, command or vector) inside one test with
  the `subtests` fixture rather than one parametrized case per row. Each row
  is still reported on failure, and collection stays small.

For zero-shot smoke checks:

```bash
uv pip install ".[gliner]"
python scripts/smoke_gliner.py --limit 2 --threshold 0.4 --adapter
```

`tests/run-tests.sh` stitches together lint, format checks, unit tests, and slow smoke checks. Use it as the baseline
for CI.

## Docs & API checks

- Add `uv run mkdocs build --strict` to your CI to fail on missing nav entries, duplicate anchors, or broken markdown.
- Add a lightweight API smoke test to ensure packaging and extras stay in sync:

  ```python
  from openmed import analyze_text

  result = analyze_text("QA ping", model_name="disease_detection_superclinical")
  assert result.entities is not None
  ```

## Coverage ideas

When adding new features, consider tests for:

- Model registry entries (ensuring new keys appear in `list_model_categories`).
- Public API options and default behaviors.
- Formatter behaviours (HTML/CSS attributes, metadata propagation).
- Zero-shot adapters (BIO/BILOU conversions).

Following this checklist keeps the docs accurate and the automation pipelines green.
