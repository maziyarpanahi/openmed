# Model Registry

OpenMed ships a curated registry (`openmed.core.model_registry.OPENMED_MODELS`) that annotates every official checkpoint
with metadata such as category, specialization, recommended confidence, and Hugging Face IDs. Use it to pick the right
model, surface dropdowns in UIs, or validate incoming requests.

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

- `ModelInfo` objects include `display_name`, `category`, `entity_types`, size hints, and a default confidence threshold.
- `get_model_suggestions` leans on lightweight heuristics to recommend models based on text snippets or hints (disease,
  pharma, oncology, etc.).

## CLI parity

```bash
openmed models list              # registry + remote (HF) by default
openmed models list --include-remote
openmed models info disease_detection_superclinical
```

The CLI uses the same helpers but returns terminal-friendly tables/JSON, making it easy to integrate into scripts or CI.

## Metadata for UIs & validation

- `ModelInfo.size_category` and `.size_mb` help you decide whether a model can fit on CPU-only infrastructure.
- `entity_types` feed dropdowns or filter chips in your frontend.
- `recommended_confidence` can drive slider defaults or guardrails on API calls (pass it to `analyze_text`).

## Keeping the registry fresh

If you add a new Hugging Face checkpoint, update `openmed/core/model_registry.py` with:

1. A unique key (e.g., `radiology_detection_superclinical`).
2. The full HF model id (`OpenMed/...`), user-friendly name, category, specialization.
3. Representative `entity_types` and a recommended confidence based on evaluation runs.

CI will enforce type safety through the unit tests, and the docs automatically pick up the new entry via the examples
above.
