# Feature Map & Capabilities

This page inventories every surfaced capability in OpenMed so you can see how the docs map back to the codebase. Use it
as the starting point when you are unsure which page (or module) to visit.

## Model lifecycle

| Area | What it covers | Where to look |
| --- | --- | --- |
| Model registry | Curated metadata (`ModelInfo`, categories, suggestions) for every OpenMed Hugging Face release. | `openmed/core/model_registry.py`, [Model Registry](./model-registry.md) |
| Discovery & loading | Hugging Face discovery, optional auth, caching, tokenizers, pipeline helpers. | `openmed/core/models.py`, [ModelLoader & Pipelines](./model-loader.md) |
| One-call inference | `analyze_text`, validation, pySBD segmentation, output formatting for dict/JSON/HTML/CSV. | `openmed/__init__.py:analyze_text`, [Analyze Text Helper](./analyze-text.md) |
| PII detection & de-identification | `extract_pii`, `deidentify`, smart entity merging, HIPAA Safe Harbor compliance. | `openmed/core/pii.py`, `openmed/core/pii_entity_merger.py`, [PII Detection & Smart Merging](./pii-smart-merging.md) |
| Zero-shot toolkit | GLiNER-powered indexing, label maps, adapters, smoke scripts. | `openmed/zero_shot/**`, [Zero-shot Toolkit](./zero-shot-ner.md) |

## Processing & outputs

| Area | What it covers | Where to look |
| --- | --- | --- |
| Advanced NER filtering | Entity spanning, score filtering, punct stripping, BIO-aware grouping. | `openmed/processing/advanced_ner.py`, [Advanced NER & Output Formatting](./output-formatting.md) |
| Formatting utilities | `PredictionResult`, copy-ready dict/JSON/HTML/CSV outputs, metadata injection. | `openmed/processing/outputs.py`, [Advanced NER & Output Formatting](./output-formatting.md) |
| Text utilities | Sentence detection, tokenization helpers, text cleaning. | `openmed/processing/`, [ModelLoader & Pipelines](./model-loader.md) |

## Tooling & ops

| Area | What it covers | Where to look |
| --- | --- | --- |
| Configuration | YAML/ENV + per-run overrides via `OpenMedConfig`; profile-aware runtime settings. | `openmed/core/config.py`, [Configuration & Validation](./configuration.md) |
| Configuration Profiles | Built-in profiles (dev, prod, test, fast) + custom profiles. | `openmed/core/config.py`, [Configuration Profiles](./profiles.md) |
| Batch Processing | Multi-text/file processing with progress and aggregation. | `openmed/processing/batch.py`, [Batch Processing](./batch-processing.md) |
| Performance Profiling | Timing utilities, metrics, and profiling decorators. | `openmed/utils/profiling.py`, [Performance Profiling](./profiling.md) |
| Testing | `tests/run-tests.sh`, unit/integration markers, smoke runners. | `tests/`, [Testing & QA](./testing.md) |
| Examples | Notebooks, scripts, and analysis snippets in `examples/`. | `examples/`, [Examples & Copy/Paste Recipes](./examples.md) |
| Automation | Make targets, GitHub Actions (CI, publish, docs). | `Makefile`, `.github/workflows/*.yml`, [Contributing & Releases](./contributing.md) |

## Suggested reading order

1. [Quick Start](./getting-started.md) — install + first inference.
2. [Analyze Text Helper](./analyze-text.md) — single-call orchestration.
3. [PII Detection & Smart Merging](./pii-smart-merging.md) — de-identification (v0.5.0).
4. [Batch Processing](./batch-processing.md) — multi-text processing.
5. [ModelLoader & Pipelines](./model-loader.md) — lower-level control.
6. [Model Registry](./model-registry.md) — pick the right checkpoint.
7. [Configuration Profiles](./profiles.md) — dev/prod/test profiles.
8. [Advanced NER & Output Formatting](./output-formatting.md) — polish predictions.
9. [Zero-shot Toolkit](./zero-shot-ner.md) — GLiNER workflows.
10. [Performance Profiling](./profiling.md) — timing and metrics.
11. [Examples](./examples.md) — runnable notebooks and scripts.
12. [Testing & QA](./testing.md) + [Contributing](./contributing.md) — team processes.
