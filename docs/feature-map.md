# Feature Map & Capabilities

This page inventories every surfaced capability in OpenMed so you can see how the docs map back to the codebase. Use it
as the starting point when you are unsure which page (or module) to visit.

## Model lifecycle

| Area | What it covers | Where to look |
| --- | --- | --- |
| Model registry | Curated metadata (`ModelInfo`, categories, suggestions) for every OpenMed Hugging Face release. | `openmed/core/model_registry.py`, [Model Registry](./model-registry.md) |
| Discovery & loading | Hugging Face discovery, optional auth, caching, tokenizers, pipeline helpers. | `openmed/core/models.py`, [ModelLoader & Pipelines](./model-loader.md) |
| One-call inference | `analyze_text`, validation, pySBD segmentation, output formatting for dict/JSON/HTML/CSV. | `openmed/__init__.py:analyze_text`, [Analyze Text Helper](./analyze-text.md) |
| Zero-shot toolkit | GLiNER-powered indexing, label maps, adapters, smoke scripts. | `ner_tools/`, `label_maps/`, [Zero-shot Toolkit](./zero-shot-ner.md) |

## Processing & outputs

| Area | What it covers | Where to look |
| --- | --- | --- |
| Advanced NER filtering | Entity spanning, score filtering, punct stripping, BIO-aware grouping. | `openmed/processing/advanced_ner.py`, [Advanced NER & Output Formatting](./output-formatting.md) |
| Formatting utilities | `PredictionResult`, copy-ready dict/JSON/HTML/CSV outputs, metadata injection. | `openmed/processing/outputs.py`, [Advanced NER & Output Formatting](./output-formatting.md) |
| Text utilities | Sentence detection, tokenization helpers, text cleaning. | `openmed/processing/`, [ModelLoader & Pipelines](./model-loader.md) |

## Tooling & ops

| Area | What it covers | Where to look |
| --- | --- | --- |
| Configuration | YAML/ENV + per-run overrides via `OpenMedConfig`; CLI config store. | `openmed/core/config.py`, [Configuration & Validation](./configuration.md) |
| CLI | `openmed` console (`analyze`, `models list/info`, `config show/set`). | `openmed/cli/main.py`, [CLI & Automation](./cli.md) |
| Testing | `tests/run-tests.sh`, unit/integration markers, smoke runners. | `tests/`, [Testing & QA](./testing.md) |
| Examples | Notebooks, CLI/analysis snippets, `examples/`. | `examples/`, [Examples & Copy/Paste Recipes](./examples.md) |
| Automation | Make targets, GitHub Actions (CI, publish, docs). | `Makefile`, `.github/workflows/*.yml`, [Contributing & Releases](./contributing.md) |

## Suggested reading order

1. [Quick Start](./getting-started.md) — install + first inference.
2. [Analyze Text Helper](./analyze-text.md) — single-call orchestration.
3. [ModelLoader & Pipelines](./model-loader.md) — lower-level control.
4. [Model Registry](./model-registry.md) — pick the right checkpoint.
5. [Advanced NER & Output Formatting](./output-formatting.md) — polish predictions.
6. [Zero-shot Toolkit](./zero-shot-ner.md) — GLiNER workflows.
7. [CLI & Automation](./cli.md) and [Examples](./examples.md) — day-to-day tools.
8. [Testing & QA](./testing.md) + [Contributing](./contributing.md) — team processes.
