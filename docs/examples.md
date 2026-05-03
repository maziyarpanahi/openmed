# Examples & Copy/Paste Recipes

This page curates the most useful samples already in the repository so you can jump straight to runnable notebooks or
scripts.

## Notebooks (`examples/notebooks/`)

| Notebook | Highlights |
| --- | --- |
| `getting_started.ipynb` | Mirrors the Quick Start guide with step-by-step installation, registry exploration, and a first call to `analyze_text`. |
| `Sentence_Detection_Batching.ipynb` | Demonstrates pySBD-based segmentation, batching, and how to align predictions back to the original paragraphs. |
| `ZeroShot_NER_Tour.ipynb` | Walks through GLiNER indexing, domain defaults, inference API usage, and the adapter that converts spans into BIO/BILOU schemes. |

Run them with VS Code, Jupyter, or Google Colab—each relies on the same `uv pip install ".[hf]"` baseline.

## Scripts & tools

| Path | What it does |
| --- | --- |
| `examples/pii_model_comparison.py` | Compares multiple PII models across shared sample text and summarizes extraction quality. |
| `examples/pii_multilingual_new_languages.py` | Exercises Dutch, Hindi, Telugu, and Portuguese registry entries, locale-specific regex matches, and optional live extraction with the new public checkpoints. |
| `examples/privacy_filter_stream/` | Web demo that streams raw agent traces beside a delayed, redacted or Faker-obfuscated shareable stream. |
| `scripts/smoke_gliner.py` | Runs a bounded set of GLiNER models/texts to confirm zero-shot dependencies are installed before releasing. |
| `tests/run-tests.sh` | Convenience runner that stitches together unit, integration, and smoke tests; extend it to include docs builds and API smoke checks. |

## Apple Silicon & Swift recipes

OpenMed `1.2.0` adds release-critical Apple entry points:

- [MLX Backend](./mlx-backend.md) for Python on Apple Silicon Macs, including Privacy Filter and experimental GLiNER-family artifacts
- [OpenMedKit (Swift Package)](./swift-openmedkit.md) for macOS, iOS, and iPadOS apps

Python MLX quick check:

```bash
uv pip install ".[mlx]"
uv run python -c "from openmed.core.backends import get_backend; print(type(get_backend()).__name__)"
```

Swift MLX quick check:

```swift
import OpenMedKit

let modelDirectory = try await OpenMedModelStore.downloadMLXModel(
    repoID: "OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx"
)

let openmed = try OpenMed(backend: .mlx(modelDirectoryURL: modelDirectory))
let entities = try openmed.extractPII("Patient John Doe, DOB 1990-05-15")
```

## Copy-ready snippets

You can find these directly in the docs:

- [Analyze Text Helper](./analyze-text.md) — dict/JSON/HTML/CSV outputs with metadata.
- [REST Service (MVP)](./rest-service.md) — FastAPI endpoints and Docker runbook.
- [MLX Backend](./mlx-backend.md) — Apple Silicon Python runtime and artifact packaging.
- [OpenMedKit (Swift)](./swift-openmedkit.md) — native app runtime for macOS, iOS, and iPadOS.
- [ModelLoader & Pipelines](./model-loader.md) — caching, token helpers, multi-model setups.
- [Advanced NER & Output Formatting](./output-formatting.md) — span filtering and conversions.
- [Zero-shot Toolkit](./zero-shot-ner.md) — indexing, label defaults, and inference APIs.

## Sample automation pipeline

```bash
#!/usr/bin/env bash
set -euo pipefail

uv pip install ".[hf,docs]"
python examples/pii_model_comparison.py > artifacts/result.txt
uv run mkdocs build --strict
python scripts/smoke_gliner.py --limit 1 --threshold 0.5
```

Use this pattern in CI to guarantee models, docs, and zero-shot flows stay healthy before publishing.
