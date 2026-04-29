# OpenMed Privacy Filter Studio

Interactive two-pane web app for PII de-identification, built on the
OpenMed unified `extract_pii` / `deidentify` API. Paste text on the left,
press **Run**, see the output on the right with each detected entity
highlighted by a colorful category label.

Two output modes:

- **Mask** — replace each entity with a `[CATEGORY]` placeholder.
- **Randomize** — replace each entity with a deterministic, locale-aware
  Faker surrogate (same seed → same output, repeated mentions of the
  same person resolve to one fake identity).

Defaults to the **Nemotron-PII MLX BF16** checkpoint
(`OpenMed/privacy-filter-nemotron-mlx`) on Apple Silicon. On any other
host the unified API automatically substitutes the matching PyTorch
checkpoint (`OpenMed/privacy-filter-nemotron`) with a one-time
`UserWarning`, so the same UI runs everywhere.

## Run

From the repository root:

```bash
pip install -e ".[mlx,service]"      # or ".[hf,service]" off Apple Silicon
python -m uvicorn examples.privacy_filter_studio.app:app --reload --port 8770
```

Open http://127.0.0.1:8770

The first run downloads the MLX (or PyTorch) checkpoint from the Hub.
Set `OPENMED_PRIVACY_FILTER_DOWNLOAD=1` (or tick the **Allow Downloads**
toggle in the UI) to allow the fetch; subsequent runs are cache-only.

## Override the model

```bash
OPENMED_STUDIO_MODEL=OpenMed/privacy-filter-nemotron-mlx-8bit \
  python -m uvicorn examples.privacy_filter_studio.app:app --port 8770
```

Any name accepted by `extract_pii(model_name=...)` works — including the
OpenAI baseline (`OpenMed/privacy-filter-mlx`, `openai/privacy-filter`)
or a local path to your own fine-tune.
