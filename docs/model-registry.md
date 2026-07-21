# Model Registry

OpenMed ships a manifest-backed registry (`openmed.core.model_registry.OPENMED_MODELS`) that annotates every official
checkpoint with metadata such as category, specialization, recommended confidence, Hugging Face IDs, device fit, and
benchmark summaries. Use it to pick the right model, surface dropdowns in UIs, or validate incoming requests.

## Exploring the registry

```python
from openmed.core.model_registry import (
    get_all_models,
    list_model_categories,
    get_models_by_category,
    get_model_info,
    get_model_suggestions,
)

print(list_model_categories())

oncology_models = get_models_by_category("Oncology")
for info in oncology_models:
    print(info.display_name, info.model_id, info.recommended_confidence)

info = get_model_info("disease_detection_superclinical")
print(info.description, info.entity_types)

suggestions = get_model_suggestions("Metastatic breast cancer on paclitaxel.")
for model_key, info, reason in suggestions:
    print(model_key, info.display_name, reason)
```

- `ModelInfo` objects include `display_name`, `category`, `entity_types`, size hints, benchmark data, optional latency/RAM
  maps, optional recommended tier, audited `script_coverage`, and a default confidence threshold.
- `get_model_suggestions` leans on lightweight heuristics to recommend models based on text snippets or hints (disease,
  pharma, oncology, etc.).

## Metadata for UIs & validation

- `ModelInfo.size_category`, `.size_mb`, `.latency_ms`, `.peak_ram_mb`, and `.recommended_tier` help you decide whether a
  model can fit on CPU-only infrastructure or a target device tier.
- `entity_types` feed dropdowns or filter chips in your frontend.
- `recommended_confidence` can drive slider defaults or guardrails on API calls (pass it to `analyze_text`).
- `get_pii_models_by_language` excludes any model whose audited tokenizer is explicitly `unsupported` for a script
  claimed by that language. The underlying UNK, byte-fallback, and tokens-per-grapheme measurements remain available on
  `ModelInfo.script_coverage` for diagnostics and UI warnings.

## Keeping the registry fresh

If you add a new Hugging Face checkpoint, refresh `models.jsonl` and, when measurements are available, enrich it with
benchmark and device-fit results. See [Model Manifest](./model-manifest.md) for the schema and merge command.

Manifest rows drive `openmed/core/model_registry.py`; avoid hand-editing the registry for new models. A new row should
include:

1. The full HF model id (`OpenMed/...`) and core release metadata.
2. Representative canonical entity labels.
3. Required 11-script tokenizer coverage for PII-family entries, plus optional benchmark, latency, RAM, and
   recommended-tier enrichment.

CI will enforce type safety through the unit tests, and the docs automatically pick up the new entry via the examples
above.
