# Privacy Filter Multilingual Studio

Interactive side-by-side demo: **OpenMed/privacy-filter-multilingual-mlx** (54
PII categories, 16 languages) vs the upstream **openai/privacy-filter** baseline
(8 coarse categories). Click any of the 16 language tabs to load a worked
example, hit **Run Comparison**, and watch both models scan the same input.

![hero](https://placehold.co/1200x720/0d6e6e/f7f4ec?text=Two-pane+side-by-side)

## What you get

- **Two models loaded once and cached.** Same call, same input, two different
  label spaces side by side.
- **16 language tabs** with synthetic clinical / business examples in each:
  English, Spanish, French, German, Italian, Dutch, Portuguese, Turkish,
  Arabic, Hindi, Bengali, Telugu, Vietnamese, Chinese, Japanese, Korean.
- **Mask / Randomize toggle** — placeholder pills (`[FIRSTNAME]`) or
  deterministic Faker surrogates.
- **Color-coded pills** by entity role (identity / contact / address / dates /
  govID / financial / crypto / vehicle / digital / auth) so the difference in
  vocabulary granularity is visible at a glance.

## Requirements

- macOS on Apple Silicon (MLX runtime).
- Both models cached locally (the studio loads them on the first request).
  - `OpenMed/privacy-filter-multilingual-mlx` — pulled from HF on first run
    when `OPENMED_PRIVACY_FILTER_DOWNLOAD=1` or via the **Allow Downloads**
    toggle in the UI.
  - **Baseline:** the studio expects a local copy of
    `openai/privacy-filter` converted into the OpenMed-MLX layout (see below).

## Build the baseline once

The mlx-community port of `openai/privacy-filter` ships with HF-naming
(`model.safetensors`) which doesn't match OpenMed's loader expectations. The
private export pipeline can produce the right layout in seconds:

    cd ../../../openmed-mlx-export
    ./scripts/export/convert-mlx \
        --model openai/privacy-filter \
        --output out/mlx/openai-privacy-filter-bf16

Default location is the one the studio reads. Override with
`OPENMED_BASELINE_PATH=/path/to/baseline-dir` if you keep yours elsewhere.

## Run

From the openmed repository root:

    OPENMED_PRIVACY_FILTER_DOWNLOAD=1 \
        uvicorn examples.privacy_filter_multilingual_studio.app:app \
        --port 8780

Then open <http://127.0.0.1:8780>.

The first request loads both pipelines (~5 s each on M-series); all subsequent
requests reuse the cached pipelines and complete in 100–800 ms depending on
input length.

## Files

| Path | What it is |
|---|---|
| `app.py` | FastAPI server: loads both pipelines, exposes `/api/examples` and `/api/run`. |
| `examples.py` | 16 language tabs × 2 examples each. All synthetic PII. |
| `static/index.html` | Topbar + language strip + editor + side-by-side result panes. |
| `static/app.js` | Tab switching, example loading, `/api/run` driver, pill rendering with role-based colours. |
| `static/styles.css` | Studio styling (light + dark themes). |
| `static/assets/logo.svg` | OpenMed mark. |

## Adding a new language or example

Edit `examples.py`:

1. Add the language to `LANGUAGE_META` if missing.
2. Add a tuple of `StudioExample(...)` entries to `LANGUAGE_EXAMPLES[code]`.

The frontend picks them up automatically — no JS changes needed.
