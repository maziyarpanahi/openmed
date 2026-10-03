# Testing & QA

Keep regressions away from your clinical workflows by leaning on the existing test suite and smoke runners. This page
summarizes what already exists so you can extend it confidently.

## Actively tested Python and desktop matrix

The core SDK supports **CPython 3.10, 3.11, 3.12 and 3.13** on Linux, Windows
and macOS. These versions and OS families are declared in `pyproject.toml`'s
classifiers. `requires-python >=3.10` is an installation floor, not a promise
that an unlisted future interpreter or alternative Python implementation is
qualified. Adding one requires a reviewed metadata, CI and documentation change.
Python 3.10 installs the MIT-licensed `tomli` backport required by the SDK's TOML
readers; newer interpreters use the standard-library `tomllib` instead. The
backport is a conditional core dependency, not an accidental development extra.

| Lane | Required combinations | What it proves |
| --- | --- | --- |
| `sdk-compatibility` core contracts | All four versions × `ubuntu-latest`, `windows-latest`, `macos-latest` (12 jobs) | Privacy sentinels, full traceback rendering, platform filesystem contracts and smoke-runner negative tests |
| `sdk-compatibility` installed artifacts | The same 12 jobs; both wheel and sdist; core and `[cli,service,mcp]` installs separately | Installed CLI and brief API/adapters, packaged resources, missing-resource rejection and source-tree isolation |
| Optional-runtime absence | Both install profiles in every compatibility job | Typed missing-MLX error and fail-closed brief; no cloud fallback, weights or model downloads |
| `test` full regression | Linux 3.10–3.12; Windows/macOS 3.11–3.12 | The complete existing Python regression suite, in addition to bounded compatibility coverage |

The stable `sdk-compatibility-gate` requires every compatibility job to succeed
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
| `slow` | opt-in | Runs heavier GLiNER or Hugging Face calls; disabled unless you pass `-m slow`. |

Configure these markers via `pytest.ini` entries in `pyproject.toml`.

## Running the suite

```bash
uv pip install ".[dev,hf]"
make lint
make format-check
make lint-swift             # for Swift/OpenMedKit changes
pytest                      # fast unit/integration mix
pytest -m "not slow"        # default behaviour
pytest -m slow              # only long-running cases
```

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
