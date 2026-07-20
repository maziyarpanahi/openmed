---
name: choosing-openmed-models
description: "Discover and pick the right OpenMed model for a clinical or biomedical task, domain, or language. Use when the user asks which OpenMed model to use, wants to list model categories, find a Disease vs Oncology vs Privacy/PII model, get a PII model for a specific language, search models by size or task, or inspect a model's labels and metadata before loading. Covers list_model_categories, get_models_by_category, get_pii_models_by_language, get_default_pii_model, search_models(ModelQuery(...)), get_model_info, and the openmed models CLI. Pairs with loading-openmed-models."
license: Apache-2.0
metadata:
  project: OpenMed
  category: openmed-core
  pairs: adjacent
  version: "1.0"
---

# Choosing OpenMed Models

OpenMed ships a registry of clinical and biomedical NER models grouped into 12
categories. **Never hardcode a model list** — query the registry at runtime so
your code stays correct as models are added. This skill helps you go from "I need
to find diseases in Spanish discharge notes" to a concrete model key.

## When to use

- The user knows the task (find diseases / tumors / PHI) but not the model.
- You need the right **PII model for a language** (es, fr, de, …).
- You want to filter models by size, task, or tier before loading.
- You want to inspect a model's labels, params, and license first.

Once you have a key, hand off to `loading-openmed-models` to load it.

## Install

```bash
pip install openmed         # registry queries work without the [hf] extra
```

## Quick start: browse categories, then pick

```python
import openmed

# 1) The 12 categories
openmed.list_model_categories()
# ['Medical', 'Privacy', 'Anatomy', 'Hematology', 'Chemical', 'Disease',
#  'Genomics', 'Oncology', 'Species', 'Pathology', 'Pharmaceutical', 'Protein']

# 2) Models in a category -> list[ModelInfo]
for m in openmed.get_models_by_category("Disease"):
    print(m.model_id, "|", m.size_category, "|", m.entity_types)

# 3) Inspect one model before loading
info = openmed.get_model_info("OpenMed/OpenMed-NER-DiseaseDetect-BigMed-278M")
print(info.display_name, info.task, info.param_count, info.license)
```

`get_models_by_category` and `get_all_models` return `ModelInfo` objects.
`get_all_models()` returns a `dict[str, ModelInfo]` keyed by registry key.

## What `ModelInfo` tells you

Every model exposes (real attributes):

```text
model_id          # HF repo id, e.g. "OpenMed/OpenMed-NER-DiseaseDetect-BigMed-278M"
display_name      # human-friendly name
category          # one of the 12 categories
specialization    # e.g. "disease entity detection"
entity_types      # list[str] of labels the model emits, e.g. ["DISEASE", ...]
size_category     # "Tiny" | "Small" | "Medium" | "Large" | "XLarge"
recommended_confidence   # suggested confidence_threshold for this model
family            # "NER" | "PII" | ...
task              # "token-classification"
languages         # e.g. ["en"], ["es"]
param_count       # e.g. 278000000
license           # e.g. "apache-2.0"
```

Use `entity_types` to confirm the model emits the labels you need, and
`recommended_confidence` as a sensible default `confidence_threshold`.

## Disease vs Oncology vs Privacy: worked choices

```python
import openmed

# Disease conditions in a general clinical note:
disease = openmed.get_models_by_category("Disease")
# e.g. "OpenMed/OpenMed-NER-DiseaseDetect-BigMed-278M"
#      "OpenMed/OpenMed-NER-DiseaseDetect-BioClinical-108M" (smaller/faster)

# Tumors, staging, oncologic findings -> Oncology, not Disease:
onco = openmed.get_models_by_category("Oncology")
# e.g. "OpenMed/OpenMed-NER-OncologyDetect-BigMed-278M"

# PHI / PII detection -> Privacy category:
privacy = openmed.get_models_by_category("Privacy")
```

Rule of thumb: **bigger (278M/560M) = more accurate, slower**; **smaller
(108M, "Small"/"Tiny") = faster, edge-friendly**. Start with a mid-size model and
size up only if recall is short.

## Pick a PII model by language

```python
import openmed

# All PII models for Spanish -> dict[str, ModelInfo]
es_models = openmed.get_pii_models_by_language("es")

# The recommended default PII model id for a language:
default_es = openmed.get_default_pii_model("es")
print(default_es)   # HF repo id, or None if unsupported
```

`deidentify(..., lang="es")` and `extract_pii(..., lang="es")` already select an
appropriate default — use these helpers when you need to override or to confirm
coverage. Supported de-id languages live in
`openmed.SUPPORTED_LANGUAGES` (en es pt fr de it nl hi te ar tr ja).

## Structured search with `ModelQuery`

For filtering by task, language, size, or tier, use the typed search:

```python
from openmed import search_models, ModelQuery

results = search_models(ModelQuery(
    task="token-classification",
    language="en",
    max_params=200_000_000,   # keep it small for on-device
    license="apache-2.0",
))
for r in results:
    print(r.repo_id, r.param_count, r.languages, r.formats)
```

Each result is a `ModelSearchResult` with fields like `repo_id`, `family`, `task`,
`languages`, `tier`, `param_count`, `architecture`, `base_model`, `formats`,
`canonical_labels`, `license`, and `released`. `ModelQuery` filters include
`task`, `language`, `tier`, `max_params`, `min_params`, `format`, `license`, and a
free-text `query`.

## Let OpenMed suggest a model from text

```python
import openmed

for key, info, reason in openmed.get_model_suggestions(
    "Stage III adenocarcinoma with metastasis to regional lymph nodes."
):
    print(key, "->", reason)
```

`get_model_suggestions(text)` returns `(registry_key, ModelInfo, reason)` tuples —
handy when the domain is unclear from the request.

## CLI

```bash
openmed models list                 # registry keys (add --include-remote to query the Hub)
openmed models info <registry-key>  # max sequence length for a key
openmed analyze --text "Stage III adenocarcinoma." --model oncology_detection_bigmed_278m
```

## Hand-off to / from OpenMed

- **To `loading-openmed-models`:** pass the chosen `model_id`/registry key as
  `model_name=` to `ModelLoader.load_model(...)` or `openmed.analyze_text(...)`.
- **To `extracting-clinical-entities`:** use the model's `recommended_confidence`
  as your `confidence_threshold` and verify `entity_types` matches your schema.
- **To de-identification:** feed `get_default_pii_model(lang)` into
  `openmed.deidentify(model_name=..., lang=...)`.

```python
import openmed
key = "oncology_detection_bigmed_278m"
info = openmed.get_model_info(key)
result = openmed.analyze_text(
    "Stage III adenocarcinoma with nodal metastasis.",
    model_name=key,
    confidence_threshold=info.recommended_confidence,
)
```

## Edge cases & gotchas

- **Category, not keyword.** "cancer" is the **Oncology** category; "diabetes" is
  **Disease**. Check `entity_types` if unsure which fits.
- **`get_default_pii_model(lang)` can return `None`** for an unsupported language —
  fall back to a supported one and warn, do not silently use English on non-English
  text.
- **`search_models` reads a committed manifest**, so it only returns models that
  have been catalogued — combine with `get_all_models()` for the full registry.
- **Match labels before committing.** A model in the right category may still not
  emit the exact label you need; confirm via `entity_types` / `canonical_labels`.
- **Licensing.** All OpenMed registry models are permissively licensed; do not
  swap in models that bundle restricted terminologies (UMLS/SNOMED/CPT).

## Standards & references

- OpenMed model org & cards: https://huggingface.co/OpenMed
- Canonical PII label taxonomy: `openmed.CANONICAL_LABELS` (see
  `extracting-pii-entities`).
