# OpenMed Documentation

OpenMed bundles curated biomedical models, advanced extraction utilities, and one-call orchestration so you can ship
clinical NLP workflows without wrangling infrastructure. This documentation keeps the most copied snippets and workflows
close at hand—each section is Markdown-first, searchable, and optimized for quick scanning or copy/paste into notebooks.

## What you get

- **Curated registries** – discoverable Hugging Face models with metadata (domain, size, device guidance).
- **One-line orchestration** – `analyze_text` wraps validation, inference, and formatting for CLI, scripts, or services.
- **Advanced NER post-processing** – score-aware grouping, PHI-friendly filtering, and CSV/JSON/HTML export helpers.
- **Composable config** – `OpenMedConfig` reads YAML/ENV so deployments stay reproducible across laptops and clusters.

!!! tip "Copy-friendly defaults"
    Every page in this site exposes code fences with copy buttons and callouts so teammates (or AI copilots) can lift the
    exact snippet they need. Use the search shortcut (`/` or `cmd/ctrl + K`) to jump straight to an entity, CLI command,
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
uv run python examples/analyze_cli.py
```

The rest of the docs expand on this snippet—head to **Quick Start** for the end-to-end setup, then explore the guides for
configuration, zero-shot GLiNER workflows, and advanced processing helpers.

## How these docs are structured

1. **Quick Start** – fastest path to a working environment plus a copy/paste script.
2. **Guides** – configuration, validation, and zero-shot NER toolkit tasks.
3. **Project** – contribution and release workflows so the automation stays in sync with packaging.

Need something that is not here yet? Drop an issue on GitHub and mention the missing recipe. Every addition is just a
Markdown file away.
