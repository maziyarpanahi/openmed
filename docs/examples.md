# Examples & Copy/Paste Recipes

This page curates the most useful samples already in the repository so you can jump straight to runnable notebooks or
scripts.

## Notebooks (`examples/notebooks/`)

| Notebook | Highlights |
| --- | --- |
| `getting_started.ipynb` | Mirrors the Quick Start guide with step-by-step installation, registry exploration, and a first call to `analyze_text`. |
| `OpenMed_CLI_Demo.ipynb` | Shows how to shell out to the CLI from notebooks, capture JSON output, and compare it with direct Python calls. |
| `Sentence_Detection_Batching.ipynb` | Demonstrates pySBD-based segmentation, batching, and how to align predictions back to the original paragraphs. |
| `ZeroShot_NER_Tour.ipynb` | Walks through GLiNER indexing, domain defaults, inference API usage, and the adapter that converts spans into BIO/BILOU schemes. |

Run them with VS Code, Jupyter, or Google Colab—each relies on the same `uv pip install ".[hf]"` baseline.

## Scripts & tools

| Path | What it does |
| --- | --- |
| `examples/analyze_cli.py` | Thin wrapper around `analyze_text` ideal for batch jobs or serverless functions. Accepts text, model name, and formatter options via argparse. |
| `python -m openmed.zero_shot.cli.*` | Collection of zero-shot utilities (`index`, `labels`, `infer`) now namespaced inside the package to keep the repo root tidy. |
| `scripts/smoke_gliner.py` | Runs a bounded set of GLiNER models/texts to confirm zero-shot dependencies are installed before releasing. |
| `tests/run-tests.sh` | Convenience runner that stitches together unit, integration, and smoke tests; extend it to include docs builds or CLI flows. |

## Copy-ready snippets

You can find these directly in the docs:

- [Analyze Text Helper](./analyze-text.md) — dict/JSON/HTML/CSV outputs with metadata.
- [ModelLoader & Pipelines](./model-loader.md) — caching, token helpers, multi-model setups.
- [Advanced NER & Output Formatting](./output-formatting.md) — span filtering and conversions.
- [Zero-shot Toolkit](./zero-shot-ner.md) — indexing, label defaults, CLI parity.

## Sample automation pipeline

```bash
#!/usr/bin/env bash
set -euo pipefail

uv pip install ".[hf,docs]"
openmed models list
openmed analyze --model disease_detection_superclinical --text "$SAMPLE_NOTE" --output-format json > artifacts/result.json
uv run mkdocs build --strict
python scripts/smoke_gliner.py --limit 1 --threshold 0.5
```

Use this pattern in CI to guarantee models, docs, and zero-shot flows stay healthy before publishing.
