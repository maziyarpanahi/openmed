# Testing & QA

Keep regressions away from your clinical workflows by leaning on the existing test suite and smoke runners. This page
summarizes what already exists so you can extend it confidently.

## Test taxonomy

| Marker | Location | Purpose |
| --- | --- | --- |
| `unit` (default) | `tests/unit/**` | Fast validation of model registry helpers, config utilities, CLI parser, etc. |
| `integration` | opt-in | Exercises multi-component flows (e.g., pipeline creation + formatter). |
| `slow` | opt-in | Runs heavier GLiNER or Hugging Face calls; disabled unless you pass `-m slow`. |

Configure these markers via `pytest.ini` entries in `pyproject.toml`.

## Running the suite

```bash
uv pip install ".[dev,hf]"
pytest                      # fast unit/integration mix
pytest -m "not slow"        # default behaviour
pytest -m slow              # only long-running cases
```

For zero-shot smoke checks:

```bash
uv pip install ".[gliner]"
python scripts/smoke_gliner.py --limit 2 --threshold 0.4 --adapter
```

`tests/run-tests.sh` stitches together typical workflows (lint, unit tests, smoke scripts). Use it as the baseline for CI.

## Docs & CLI checks

- Add `uv run mkdocs build --strict` to your CI to fail on missing nav entries, duplicate anchors, or broken markdown.
- Combine CLI runs with fixtures to ensure packaging and extras stay in sync:

  ```bash
  openmed analyze --model disease_detection_superclinical --text "QA ping" --output-format json >/tmp/result.json
  jq .entities[0] /tmp/result.json
  ```

## Coverage ideas

When adding new features, consider tests for:

- Model registry entries (ensuring new keys appear in `list_model_categories`).
- CLI options (e.g., new `openmed analyze` flag).
- Formatter behaviours (HTML/CSS attributes, metadata propagation).
- Zero-shot adapters (BIO/BILOU conversions).

Following this checklist keeps the docs accurate and the automation pipelines green.
