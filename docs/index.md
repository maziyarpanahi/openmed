# OpenMed Documentation

OpenMed bundles curated biomedical models, advanced extraction utilities, and one-call orchestration so you can ship
clinical NLP workflows without wrangling infrastructure. This documentation keeps the most copied snippets and workflows
close at hand—each section is Markdown-first, searchable, and optimized for quick scanning or copy/paste into notebooks.

OpenMed `1.4.0` expands the Apple story:

- **Python MLX** on Apple Silicon Macs through `openmed[mlx]`
- **OpenMedKit** for native macOS, iOS, and iPadOS apps
- **Shared MLX artifacts** so Python and Swift can consume the same packaged model layout
- **Native Privacy Filter, OpenAI Nemotron Privacy Filter, OpenMed Multilingual Privacy Filter, and experimental GLiNER-family runtimes** for local-first PII, classification, and relation extraction workflows

## What you get

- **Curated registries** – discoverable Hugging Face models with metadata (domain, size, device guidance).
- **One-line orchestration** – `analyze_text` wraps validation, inference, and formatting for scripts, notebooks, or services.
- **PII detection & de-identification** – HIPAA-compliant smart entity merging for production-ready de-identification.
- **Apple Silicon acceleration** – MLX-backed Python inference plus Swift-native app integration through `OpenMedKit`.
- **REST service MVP** – FastAPI endpoints for `/health`, `/analyze`, `/pii/extract`, and `/pii/deidentify`.
- **Advanced NER post-processing** – score-aware grouping, PHI-friendly filtering, and CSV/JSON/HTML export helpers.
- **Composable config** – `OpenMedConfig` reads YAML/ENV so deployments stay reproducible across laptops and clusters.

!!! tip "Copy-friendly defaults"
    Every page in this site exposes code fences with copy buttons and callouts so teammates (or AI copilots) can lift the
    exact snippet they need. Use the search shortcut (`/` or `cmd/ctrl + K`) to jump straight to an entity, API call,
    or API surface.

## First look

```python
from openmed import analyze_text

result = analyze_text(
    "Patient started on imatinib for chronic myeloid leukemia.",
    model_name="disease_detection_superclinical",
    confidence_threshold=0.55,
)

for entity in result.entities:
    print(entity.label, entity.text, entity.confidence)
```

```bash
uv pip install "openmed[hf]"
uv run python examples/pii_model_comparison.py
```

The rest of the docs expand on this snippet—head to **Quick Start** for the end-to-end setup, then explore the guides for
configuration, zero-shot GLiNER workflows, and advanced processing helpers.

## 1.4.0 release highlights

- [MLX Backend](./mlx-backend.md) – Python MLX on Apple Silicon, Privacy Filter family support, shared artifact packaging, and backend auto-detection.
- [OpenMedKit (Swift Package)](./swift-openmedkit.md) – native macOS/iOS/iPadOS integration with MLX, CoreML, Privacy Filter, OpenMed Multilingual Privacy Filter, and experimental GLiNER-family APIs.
- [CoreML Packaging](./coreml-export.md) – current status of the bundled Apple model route alongside the new MLX flow.
- [Examples & Copy/Paste Recipes](./examples.md) – release-friendly snippets for Python, PII, batch jobs, and Apple runtimes.

## How these docs are structured

1. [Quick Start](./getting-started.md) – fastest path to a working environment plus a copy/paste script.
2. [Feature Map](./feature-map.md) – see how every capability maps back to the code.
3. Core guides:
   - [Analyze Text Helper](./analyze-text.md) for single-call inference.
   - [REST Service (MVP)](./rest-service.md) for Dockerized HTTP endpoints.
   - [PII Detection & Smart Merging](./pii-smart-merging.md) for HIPAA-compliant de-identification (v0.5.0).
   - [Batch Processing](./batch-processing.md) for multi-text/file processing.
   - [ModelLoader & Pipelines](./model-loader.md) for long-running jobs.
   - [Model Registry](./model-registry.md) to pick the right checkpoint.
   - [Configuration Profiles](./profiles.md) for dev/prod/test switching.
   - [Advanced NER & Output Formatting](./output-formatting.md) to polish spans.
   - [Medical-Aware Tokenizer](./medical-tokenizer.md) for better clinical token boundaries.
   - [Configuration & Validation](./configuration.md) to keep deployments reproducible.
   - [Zero-shot Toolkit](./zero-shot-ner.md) when you need GLiNER workflows.
   - [Performance Profiling](./profiling.md) for timing and optimization.
   - [Examples](./examples.md) and [Testing & QA](./testing.md) for day-to-day operations.
4. [Contributing & Releases](./contributing.md) – how we cut releases, publish docs, and keep CI green.

Need something that is not here yet? Drop an issue on GitHub and mention the missing recipe. Every addition is just a
Markdown file away.
