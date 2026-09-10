---
search:
  exclude: true
---

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

## Registry slots and pointers

Committed release-channel state lives in `gates/registry_state.json` (schema
v2), keyed by `family::tier::format` slots — the same normalized key used by
`gates/baseline.json`, `gates/rollout_state.json`, and the release ledger.
Slots are sparse release channels: one is created only when a checkpoint is
promoted with a RELEASABLE gate report whose `(family, tier, format)`
coordinates match its manifest row, and `latest` names the selected shipping
target for that channel, not the only manifest candidate. Untiered checkpoints
use the `family::none::format` slot. Each slot records assigned SemVer per
checkpoint — `1.0.0` for the first target, a minor bump for each new target,
preserved on reload and re-promotion, and never parsed from the repo id (a
`-v1` in a checkpoint name is part of the upstream model name, not a version
sequence).

The committed registry state also publishes generated model cards for the
[latest PII checkpoint](./model-cards/registry/pii-small-mlx-fp-latest.md) and
its
[last-green rollback target](./model-cards/registry/pii-small-mlx-fp-last-green.md).
A v1 (family-keyed) state file migrates deterministically with
`python scripts/release/registry_ctl.py migrate`; the migration refuses
ambiguous coordinates and leaves the file unchanged on any error.

## Metadata for UIs & validation

- `ModelInfo.size_category`, `.size_mb`, `.latency_ms`, `.peak_ram_mb`, and `.recommended_tier` help you decide whether a
  model can fit on CPU-only infrastructure or a target device tier.
- `entity_types` feed dropdowns or filter chips in your frontend.
- `recommended_confidence` can drive slider defaults or guardrails on API calls (pass it to `analyze_text`).
- `get_pii_models_by_language` excludes any model whose audited tokenizer is explicitly `unsupported` for a script
  claimed by that language. The underlying UNK, byte-fallback, and tokens-per-grapheme measurements remain available on
  `ModelInfo.script_coverage` for diagnostics and UI warnings.

## Plan downloads before using data

Use `models size` to inspect download, disk, and estimated peak RAM requirements
from the committed manifest. The default path is offline-safe and does not
contact Hugging Face:

```bash
OPENMED_OFFLINE=1 openmed models size disease_detection_tiny
openmed models size --budget-mb 100
openmed models size --budget-mb 100 --format json
```

With a budget, the command lists only models whose remaining download fits and
recommends the smallest qualifying snapshot for each task. Models already in
the local Hugging Face cache are marked `cached — 0 MB to download` and count
as zero against the budget.

Use `--remote` only when you explicitly want to refresh an estimate from the
current Hub file metadata. Supplying an alias keeps that opt-in lookup focused:

```bash
openmed models size disease_detection_tiny --remote
```

Sizes use decimal megabytes (1 MB = 1,000,000 bytes). Remote inspection requires
the optional `openmed[hf]` dependencies; ordinary offline estimates do not.

## Keeping the registry fresh

If you add a new Hugging Face checkpoint, refresh `models.jsonl` and, when measurements are available, enrich it with
benchmark and device-fit results. See [Model Manifest](./model-manifest.md) for the schema and merge command.

Manifest rows drive `openmed/core/model_registry.py`; avoid hand-editing the registry for new models. A new row should
include:

1. The full HF model id (`OpenMed/...`) and core release metadata.
2. Representative canonical entity labels.
3. Required 11-script tokenizer coverage for PII-family entries, plus optional benchmark, latency, RAM, and
   recommended-tier enrichment.
4. Per-format download sizes and per-script recall/leakage-floor evidence when
   the release model card advertises those decision fields.

Regenerate all committed catalog surfaces from the local manifest, then review
the resulting README, registry-card, and table changes:

```bash
python scripts/manifest/regenerate_surfaces.py
```

The command does not contact Hugging Face. CI runs the same command and rejects
any generated diff, so manifest refreshes must commit every derived surface.

<!-- BEGIN MANIFEST CATALOG SURFACES -->
## Manifest-backed catalog

The committed manifest contains 2,266 entries across 33 model-backed PII languages. Family counts: General=9, NER=1,093, PII=1,018, Vision=3, ZeroShot=143.

<!-- BEGIN MANIFEST MODEL TABLE -->
| Model | Family | Task | Languages | Tier | Formats |
|---|---|---|---|---|---|
| `OpenMed/Ministral-3B-PII-Preview` | General | text-generation | ar, bn, de, en, es, fr, hi, it, ja, ko, nl, pt, tr, zh | - | pytorch |
| `OpenMed/laneformer-2b-it-q4-mlx` | General | text-generation | en | - | mlx-4bit |
| `OpenMed/maple-preview-4bit-mlx` | General | text-generation | en | - | mlx-4bit, pytorch |
| `OpenMed/maple-preview-4bit-onnx-android` | General | text-generation | en | - | onnx |
| `OpenMed/maple-preview-4bit-onnx-webgpu` | General | text-generation | en | - | onnx |
| `OpenMed/maple-preview-8bit-mlx` | General | text-generation | en | - | mlx-8bit, pytorch |
| `OpenMed/Ministral-3B-Medical-v1` | General | unknown | - | - | pytorch |
| `OpenMed/Qwen2.5-VL-3B-Medical-v1` | General | unknown | - | - | pytorch |
| `OpenMed/Qwen3.5-2B-Medical-v1` | General | unknown | - | - | pytorch |
| `OpenMed/OpenMed-ClinicalNER-SuperClinical-Large-434M-v1-onnx-android` | NER | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-BigMed-278M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-BigMed-278M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-BigMed-278M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-BigMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-BigMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-BigMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-BioClinical-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-BioClinical-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-BioClinical-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-BioMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-BioMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-BioMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-BioMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-BioMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-BioMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-BioPatient-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-BioPatient-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-BioPatient-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-33M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-33M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-33M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-EuroMed-212M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-EuroMed-212M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-ModernClinical-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-ModernClinical-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-ModernClinical-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-ModernClinical-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-ModernMed-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-ModernMed-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-ModernMed-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-ModernMed-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-MultiMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-MultiMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-MultiMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-MultiMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-MultiMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-MultiMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-PubMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-PubMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-PubMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-PubMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-PubMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-PubMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-PubMed-v2-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-PubMed-v2-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-PubMed-v2-109M-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-SnowMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-SnowMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-SnowMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperClinical-141M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperClinical-141M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperClinical-141M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperClinical-184M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperClinical-184M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperClinical-184M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperClinical-434M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperClinical-434M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperClinical-434M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperMedical-125M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperMedical-125M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperMedical-125M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperMedical-355M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperMedical-355M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-SuperMedical-355M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-135M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-135M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-135M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-65M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-65M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-65M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-66M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-66M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-66M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-82M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-82M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-82M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BigMed-278M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BigMed-278M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BigMed-278M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BigMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BigMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BigMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BioClinical-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BioClinical-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BioClinical-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BioMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BioMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BioMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BioMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BioMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BioMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BioPatient-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BioPatient-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-BioPatient-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ElectraMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ElectraMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ElectraMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ElectraMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ElectraMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ElectraMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ElectraMed-33M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ElectraMed-33M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ElectraMed-33M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ElectraMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ElectraMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ElectraMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ModernClinical-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ModernClinical-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ModernClinical-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ModernClinical-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ModernMed-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ModernMed-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ModernMed-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-ModernMed-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-MultiMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-MultiMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-MultiMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-MultiMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-MultiMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-MultiMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-PubMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-PubMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-PubMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-PubMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-PubMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-PubMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-PubMed-v2-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-PubMed-v2-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-PubMed-v2-109M-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SnowMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SnowMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SnowMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperClinical-141M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperClinical-141M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperClinical-141M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperClinical-184M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperClinical-184M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperClinical-184M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperClinical-434M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperClinical-434M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperClinical-434M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperMedical-125M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperMedical-125M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperMedical-125M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperMedical-355M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperMedical-355M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-SuperMedical-355M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-135M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-135M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-135M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-65M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-65M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-65M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-66M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-66M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-66M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-82M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-82M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-BloodCancerDetect-TinyMed-82M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-BigMed-278M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-BigMed-278M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-BigMed-278M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-BigMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-BigMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-BigMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-BioClinical-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-BioClinical-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-BioClinical-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-BioMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-BioMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-BioMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-BioMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-BioMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-BioMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-BioPatient-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-BioPatient-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-BioPatient-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-ElectraMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-ElectraMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-ElectraMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-ElectraMed-33M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-ElectraMed-33M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-ElectraMed-33M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-ElectraMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-ElectraMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-ElectraMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-EuroMed-212M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-EuroMed-212M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-ModernClinical-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-ModernClinical-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-ModernClinical-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-ModernClinical-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-ModernMed-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-ModernMed-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-ModernMed-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-ModernMed-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-MultiMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-MultiMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-MultiMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-MultiMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-MultiMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-MultiMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-PubMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-PubMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-PubMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-PubMed-v2-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-PubMed-v2-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-PubMed-v2-109M-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-SnowMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-SnowMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-SnowMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperClinical-141M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperClinical-141M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperClinical-141M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperClinical-184M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperClinical-184M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperClinical-184M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperClinical-434M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperClinical-434M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperClinical-434M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperMedical-125M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperMedical-125M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperMedical-125M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperMedical-355M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperMedical-355M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-SuperMedical-355M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-TinyMed-135M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-TinyMed-135M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-TinyMed-135M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-TinyMed-65M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-TinyMed-65M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-TinyMed-65M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-TinyMed-66M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-TinyMed-66M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-TinyMed-66M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-ChemicalDetect-TinyMed-82M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-TinyMed-82M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ChemicalDetect-TinyMed-82M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-DNADetect-BigMed-278M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-BigMed-278M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-BigMed-278M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-BigMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-BigMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-BigMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-BioClinical-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-BioClinical-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-BioClinical-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-BioMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-BioMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-BioMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-BioMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-BioMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-BioMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-BioPatient-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-BioPatient-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-BioPatient-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-ElectraMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-ElectraMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-ElectraMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-ElectraMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-ElectraMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-ElectraMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-ElectraMed-33M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-ElectraMed-33M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-ElectraMed-33M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-ElectraMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-ElectraMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-ElectraMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-EuroMed-212M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-EuroMed-212M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-ModernClinical-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-ModernClinical-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-ModernClinical-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-ModernClinical-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-ModernMed-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-ModernMed-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-ModernMed-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-ModernMed-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-MultiMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-MultiMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-MultiMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-MultiMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-MultiMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-MultiMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-PubMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-PubMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-PubMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-PubMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-PubMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-PubMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-PubMed-v2-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-PubMed-v2-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-PubMed-v2-109M-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-SnowMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-SnowMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-SnowMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-SuperClinical-141M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-SuperClinical-141M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-SuperClinical-141M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-SuperClinical-184M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-SuperClinical-184M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-SuperClinical-184M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-SuperClinical-434M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-SuperClinical-434M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-SuperClinical-434M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-SuperMedical-125M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-SuperMedical-125M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-SuperMedical-125M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-SuperMedical-355M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-SuperMedical-355M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-SuperMedical-355M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DNADetect-TinyMed-135M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-TinyMed-135M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-TinyMed-135M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-DNADetect-TinyMed-65M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-TinyMed-65M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-TinyMed-65M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-DNADetect-TinyMed-66M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-TinyMed-66M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-TinyMed-66M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-DNADetect-TinyMed-82M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-DNADetect-TinyMed-82M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DNADetect-TinyMed-82M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-BigMed-278M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-BigMed-278M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-BigMed-278M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-BigMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-BigMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-BigMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-BioClinical-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-BioClinical-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-BioClinical-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-BioMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-BioMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-BioMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-BioMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-BioMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-BioMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-BioPatient-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-BioPatient-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-BioPatient-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-33M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-33M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-33M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-ElectraMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-ModernClinical-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-ModernClinical-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-ModernClinical-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-ModernClinical-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-ModernMed-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-ModernMed-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-ModernMed-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-ModernMed-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-MultiMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-MultiMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-MultiMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-MultiMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-MultiMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-MultiMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-PubMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-PubMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-PubMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-PubMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-PubMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-PubMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-PubMed-v2-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-PubMed-v2-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-PubMed-v2-109M-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-SnowMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-SnowMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-SnowMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-141M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-141M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-141M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-184M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-184M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-184M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperMedical-125M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperMedical-125M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperMedical-125M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperMedical-355M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperMedical-355M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-SuperMedical-355M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-135M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-135M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-135M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-65M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-65M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-65M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-66M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-66M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-66M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-82M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-82M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-82M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-BigMed-278M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-BigMed-278M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-BigMed-278M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-BigMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-BigMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-BigMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-BioClinical-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-BioClinical-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-BioClinical-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-BioMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-BioMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-BioMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-BioMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-BioMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-BioMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-BioPatient-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-BioPatient-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-BioPatient-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-ElectraMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-ElectraMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-ElectraMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-ElectraMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-ElectraMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-ElectraMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-ElectraMed-33M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-ElectraMed-33M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-ElectraMed-33M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-ElectraMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-ElectraMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-ElectraMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-EuroMed-212M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-EuroMed-212M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-ModernClinical-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-ModernClinical-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-ModernClinical-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-ModernClinical-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-ModernMed-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-ModernMed-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-ModernMed-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-ModernMed-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-MultiMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-MultiMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-MultiMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-MultiMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-MultiMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-MultiMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-PubMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-PubMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-PubMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-PubMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-PubMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-PubMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-PubMed-v2-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-PubMed-v2-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-PubMed-v2-109M-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-SnowMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-SnowMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-SnowMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperClinical-141M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperClinical-141M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperClinical-141M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperClinical-184M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperClinical-184M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperClinical-184M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperClinical-434M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperClinical-434M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperClinical-434M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperMedical-125M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperMedical-125M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperMedical-125M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperMedical-355M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperMedical-355M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-SuperMedical-355M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-TinyMed-135M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-TinyMed-135M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-TinyMed-135M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-TinyMed-65M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-TinyMed-65M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-TinyMed-65M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-TinyMed-66M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-TinyMed-66M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-TinyMed-66M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-GenomeDetect-TinyMed-82M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-TinyMed-82M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomeDetect-TinyMed-82M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-BigMed-278M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-BigMed-278M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-BigMed-278M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-BigMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-BigMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-BigMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-BioClinical-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-BioClinical-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-BioClinical-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-BioMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-BioMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-BioMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-BioMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-BioMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-BioMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-BioPatient-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-BioPatient-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-BioPatient-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-ElectraMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-ElectraMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-ElectraMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-ElectraMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-ElectraMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-ElectraMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-ElectraMed-33M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-ElectraMed-33M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-ElectraMed-33M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-ElectraMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-ElectraMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-ElectraMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-EuroMed-212M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-EuroMed-212M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-ModernClinical-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-ModernClinical-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-ModernClinical-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-ModernClinical-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-ModernMed-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-ModernMed-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-ModernMed-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-ModernMed-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-MultiMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-MultiMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-MultiMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-MultiMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-MultiMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-MultiMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-PubMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-PubMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-PubMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-PubMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-PubMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-PubMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-PubMed-v2-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-PubMed-v2-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-PubMed-v2-109M-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-SnowMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-SnowMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-SnowMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperClinical-141M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperClinical-141M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperClinical-141M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperClinical-184M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperClinical-184M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperClinical-184M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperClinical-434M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperClinical-434M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperClinical-434M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperMedical-125M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperMedical-125M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperMedical-125M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperMedical-355M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperMedical-355M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-SuperMedical-355M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-TinyMed-135M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-TinyMed-135M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-TinyMed-135M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-TinyMed-65M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-TinyMed-65M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-TinyMed-65M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-TinyMed-66M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-TinyMed-66M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-TinyMed-66M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-GenomicDetect-TinyMed-82M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-TinyMed-82M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-GenomicDetect-TinyMed-82M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-BigMed-278M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-BigMed-278M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-BigMed-278M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-BigMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-BigMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-BigMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-BioClinical-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-BioClinical-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-BioClinical-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-BioMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-BioMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-BioMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-BioMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-BioMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-BioMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-BioPatient-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-BioPatient-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-BioPatient-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-33M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-33M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-33M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-EuroMed-212M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-EuroMed-212M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-ModernClinical-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-ModernClinical-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-ModernClinical-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-ModernClinical-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-ModernMed-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-ModernMed-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-MultiMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-MultiMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-MultiMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-MultiMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-MultiMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-MultiMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-PubMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-PubMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-PubMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-PubMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-PubMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-PubMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-PubMed-v2-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-PubMed-v2-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-PubMed-v2-109M-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-SnowMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-SnowMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-SnowMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-141M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-141M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-141M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-184M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-184M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-184M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-434M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-434M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-434M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperMedical-125M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperMedical-125M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperMedical-125M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperMedical-355M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperMedical-355M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-SuperMedical-355M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-TinyMed-135M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-TinyMed-135M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-TinyMed-135M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-TinyMed-65M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-TinyMed-65M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-TinyMed-65M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-TinyMed-66M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-TinyMed-66M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-TinyMed-66M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-OncologyDetect-TinyMed-82M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-TinyMed-82M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OncologyDetect-TinyMed-82M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-BigMed-278M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-BigMed-278M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-BigMed-278M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-BigMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-BigMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-BigMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-BioClinical-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-BioClinical-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-BioClinical-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-BioMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-BioMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-BioMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-BioMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-BioMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-BioMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-BioPatient-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-BioPatient-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-BioPatient-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-ElectraMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-ElectraMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-ElectraMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-ElectraMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-ElectraMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-ElectraMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-ElectraMed-33M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-ElectraMed-33M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-ElectraMed-33M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-ElectraMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-ElectraMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-ElectraMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-ModernClinical-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-ModernClinical-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-ModernClinical-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-ModernClinical-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-ModernMed-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-ModernMed-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-ModernMed-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-ModernMed-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-MultiMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-MultiMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-MultiMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-MultiMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-MultiMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-MultiMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-PubMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-PubMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-PubMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-PubMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-PubMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-PubMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-PubMed-v2-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-PubMed-v2-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-PubMed-v2-109M-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-SnowMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-SnowMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-SnowMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperClinical-141M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperClinical-141M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperClinical-141M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperClinical-184M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperClinical-184M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperClinical-184M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperClinical-434M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperClinical-434M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperClinical-434M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperMedical-125M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperMedical-125M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperMedical-125M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperMedical-355M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperMedical-355M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-SuperMedical-355M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-TinyMed-135M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-TinyMed-135M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-TinyMed-135M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-TinyMed-65M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-TinyMed-65M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-TinyMed-65M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-TinyMed-66M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-TinyMed-66M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-TinyMed-66M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-OrganismDetect-TinyMed-82M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-TinyMed-82M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-OrganismDetect-TinyMed-82M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-BigMed-278M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-BigMed-278M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-BigMed-278M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-BigMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-BigMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-BigMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-BioClinical-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-BioClinical-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-BioClinical-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-BioMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-BioMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-BioMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-BioMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-BioMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-BioMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-BioPatient-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-BioPatient-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-BioPatient-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-ElectraMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-ElectraMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-ElectraMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-ElectraMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-ElectraMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-ElectraMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-ElectraMed-33M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-ElectraMed-33M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-ElectraMed-33M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-ElectraMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-ElectraMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-ElectraMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-EuroMed-212M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-EuroMed-212M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-ModernClinical-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-ModernClinical-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-ModernClinical-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-ModernClinical-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-ModernMed-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-ModernMed-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-ModernMed-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-ModernMed-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-MultiMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-MultiMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-MultiMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-MultiMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-MultiMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-MultiMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-PubMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-PubMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-PubMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-PubMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-PubMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-PubMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-PubMed-v2-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-PubMed-v2-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-PubMed-v2-109M-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-SnowMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-SnowMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-SnowMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperClinical-141M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperClinical-141M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperClinical-141M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperClinical-184M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperClinical-184M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperClinical-184M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperClinical-434M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperClinical-434M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperClinical-434M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperMedical-125M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperMedical-125M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperMedical-125M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperMedical-355M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperMedical-355M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-SuperMedical-355M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-TinyMed-135M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-TinyMed-135M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-TinyMed-135M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-TinyMed-65M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-TinyMed-65M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-TinyMed-65M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-TinyMed-66M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-TinyMed-66M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-TinyMed-66M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-PathologyDetect-TinyMed-82M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-TinyMed-82M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PathologyDetect-TinyMed-82M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-BigMed-278M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-BigMed-278M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-BigMed-278M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-BigMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-BigMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-BigMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-BioClinical-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-BioClinical-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-BioClinical-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-BioMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-BioMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-BioMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-BioMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-BioMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-BioMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-BioPatient-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-BioPatient-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-BioPatient-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-ElectraMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-ElectraMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-ElectraMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-ElectraMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-ElectraMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-ElectraMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-ElectraMed-33M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-ElectraMed-33M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-ElectraMed-33M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-ElectraMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-ElectraMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-ElectraMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-EuroMed-212M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-EuroMed-212M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-ModernClinical-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-ModernClinical-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-ModernClinical-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-ModernClinical-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-ModernMed-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-ModernMed-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-ModernMed-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-ModernMed-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-MultiMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-MultiMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-MultiMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-MultiMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-MultiMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-MultiMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-PubMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-PubMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-PubMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-PubMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-PubMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-PubMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-PubMed-v2-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-PubMed-v2-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-PubMed-v2-109M-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-SnowMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-SnowMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-SnowMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-141M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-141M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-141M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-184M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-184M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-184M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperMedical-125M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperMedical-125M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperMedical-125M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperMedical-355M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperMedical-355M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-SuperMedical-355M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-TinyMed-135M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-TinyMed-135M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-TinyMed-135M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-TinyMed-65M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-TinyMed-65M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-TinyMed-65M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-TinyMed-66M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-TinyMed-66M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-TinyMed-66M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-PharmaDetect-TinyMed-82M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-TinyMed-82M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-PharmaDetect-TinyMed-82M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-BigMed-278M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-BigMed-278M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-BigMed-278M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-BigMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-BigMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-BigMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-BioClinical-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-BioClinical-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-BioClinical-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-BioMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-BioMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-BioMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-BioMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-BioMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-BioMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-BioPatient-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-BioPatient-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-BioPatient-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-ElectraMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-ElectraMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-ElectraMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-ElectraMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-ElectraMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-ElectraMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-ElectraMed-33M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-ElectraMed-33M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-ElectraMed-33M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-ElectraMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-ElectraMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-ElectraMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-EuroMed-212M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-EuroMed-212M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-ModernClinical-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-ModernClinical-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-ModernClinical-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-ModernClinical-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-ModernMed-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-ModernMed-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-ModernMed-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-ModernMed-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-MultiMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-MultiMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-MultiMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-MultiMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-MultiMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-MultiMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-PubMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-PubMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-PubMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-PubMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-PubMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-PubMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-PubMed-v2-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-PubMed-v2-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-PubMed-v2-109M-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-SnowMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-SnowMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-SnowMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperClinical-141M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperClinical-141M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperClinical-141M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperClinical-184M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperClinical-184M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperClinical-184M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperClinical-434M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperClinical-434M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperClinical-434M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperMedical-125M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperMedical-125M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperMedical-125M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperMedical-355M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperMedical-355M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-SuperMedical-355M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-TinyMed-135M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-TinyMed-135M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-TinyMed-135M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-TinyMed-65M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-TinyMed-65M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-TinyMed-65M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-TinyMed-66M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-TinyMed-66M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-TinyMed-66M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-ProteinDetect-TinyMed-82M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-TinyMed-82M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-ProteinDetect-TinyMed-82M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-BigMed-278M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-BigMed-278M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-BigMed-278M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-BigMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-BigMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-BigMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-BioClinical-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-BioClinical-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-BioClinical-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-BioMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-BioMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-BioMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-BioMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-BioMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-BioMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-BioPatient-108M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-BioPatient-108M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-BioPatient-108M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-ElectraMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-ElectraMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-ElectraMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-ElectraMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-ElectraMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-ElectraMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-ElectraMed-33M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-ElectraMed-33M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-ElectraMed-33M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-ElectraMed-560M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-ElectraMed-560M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-ElectraMed-560M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-ModernClinical-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-ModernClinical-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-ModernClinical-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-ModernClinical-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-ModernMed-149M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-ModernMed-149M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-ModernMed-395M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-ModernMed-395M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-MultiMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-MultiMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-MultiMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-MultiMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-MultiMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-MultiMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-PubMed-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-PubMed-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-PubMed-109M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-PubMed-335M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-PubMed-335M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-PubMed-335M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-PubMed-v2-109M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-PubMed-v2-109M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-PubMed-v2-109M-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-SnowMed-568M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-SnowMed-568M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-SnowMed-568M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperClinical-141M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperClinical-141M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperClinical-141M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperClinical-184M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperClinical-184M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperClinical-184M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperClinical-434M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperClinical-434M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperClinical-434M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperMedical-125M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperMedical-125M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperMedical-125M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperMedical-355M` | NER | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperMedical-355M-mlx` | NER | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-SuperMedical-355M-v1-onnx-android` | NER | token-classification | en | - | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-TinyMed-135M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-TinyMed-135M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-TinyMed-135M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-TinyMed-65M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-TinyMed-65M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-TinyMed-65M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-TinyMed-66M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-TinyMed-66M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-TinyMed-66M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-NER-SpeciesDetect-TinyMed-82M` | NER | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-TinyMed-82M-mlx` | NER | token-classification | en | Tiny | mlx-fp, pytorch |
| `OpenMed/OpenMed-NER-SpeciesDetect-TinyMed-82M-v1-onnx-android` | NER | token-classification | en | Tiny | onnx |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1` | PII | token-classification | ar | Large | pytorch |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1-mlx` | PII | token-classification | ar | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1-onnx-android` | PII | token-classification | ar | Large | onnx |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1` | PII | token-classification | ar | Large | pytorch |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1-mlx` | PII | token-classification | ar | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1-onnx-android` | PII | token-classification | ar | Large | onnx |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` | PII | token-classification | bn | Large | pytorch |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1-onnx-android` | PII | token-classification | bn | Large | onnx |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1` | PII | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1-mlx` | PII | token-classification | en | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1-onnx-android` | PII | token-classification | en | Base | onnx |
| `OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1` | PII | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1-onnx-android` | PII | token-classification | en | Base | onnx |
| `OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1` | PII | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1-mlx` | PII | token-classification | en | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1-onnx-android` | PII | token-classification | en | Base | onnx |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1` | PII | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-mlx` | PII | token-classification | en | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-onnx-android` | PII | token-classification | en | Base | onnx |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` | PII | token-classification | zh | Large | pytorch |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1-onnx-android` | PII | token-classification | zh | Large | onnx |
| `OpenMed/OpenMed-PII-Chinese-BioClinicalModern-Large-395M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-Chinese-BioClinicalModern-Large-395M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-Chinese-ModernMed-Large-395M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-Chinese-ModernMed-Large-395M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-Chinese-NomicMed-Large-395M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-Chinese-NomicMed-Large-395M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1` | PII | token-classification | en | XLarge | pytorch |
| `OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1-onnx-android` | PII | token-classification | en | XLarge | onnx |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1` | PII | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1-mlx` | PII | token-classification | en | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1-onnx-android` | PII | token-classification | en | Base | onnx |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1` | PII | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1-mlx` | PII | token-classification | en | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1-onnx-android` | PII | token-classification | en | Base | onnx |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1` | PII | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx` | PII | token-classification | en | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx-q8` | PII | token-classification | en | Small | mlx-8bit, pytorch |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android` | PII | token-classification | en | Small | onnx |
| `OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1` | PII | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1-onnx-android` | PII | token-classification | en | Base | onnx |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1-mlx` | PII | token-classification | nl | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1-mlx` | PII | token-classification | nl | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1` | PII | token-classification | nl | Base | pytorch |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1-mlx` | PII | token-classification | nl | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1-onnx-android` | PII | token-classification | nl | Base | onnx |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Base-149M-v1` | PII | token-classification | nl | Base | pytorch |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Base-149M-v1-onnx-android` | PII | token-classification | nl | Base | onnx |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Large-395M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Large-395M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1` | PII | token-classification | nl | Base | pytorch |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1-mlx` | PII | token-classification | nl | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1-onnx-android` | PII | token-classification | nl | Base | onnx |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1-mlx` | PII | token-classification | nl | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1` | PII | token-classification | nl | Base | pytorch |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1-mlx` | PII | token-classification | nl | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1-onnx-android` | PII | token-classification | nl | Base | onnx |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1` | PII | token-classification | nl | Base | pytorch |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1-mlx` | PII | token-classification | nl | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1-onnx-android` | PII | token-classification | nl | Base | onnx |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1-mlx` | PII | token-classification | nl | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1` | PII | token-classification | nl | Base | pytorch |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1-mlx` | PII | token-classification | nl | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1-onnx-android` | PII | token-classification | nl | Base | onnx |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1-mlx` | PII | token-classification | nl | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1-mlx` | PII | token-classification | nl | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1` | PII | token-classification | nl | Base | pytorch |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1-mlx` | PII | token-classification | nl | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1-onnx-android` | PII | token-classification | nl | Base | onnx |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1-mlx` | PII | token-classification | nl | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1` | PII | token-classification | nl | Small | pytorch |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1-mlx` | PII | token-classification | nl | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1-onnx-android` | PII | token-classification | nl | Small | onnx |
| `OpenMed/OpenMed-PII-Dutch-ClinicalLongformer-Base-149M-v1` | PII | token-classification | nl | Base | pytorch |
| `OpenMed/OpenMed-PII-Dutch-ClinicalLongformer-Base-149M-v1-onnx-android` | PII | token-classification | nl | Base | onnx |
| `OpenMed/OpenMed-PII-Dutch-EuroMed-Large-210M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-EuroMed-Large-210M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1` | PII | token-classification | nl | Small | pytorch |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1-mlx` | PII | token-classification | nl | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1-onnx-android` | PII | token-classification | nl | Small | onnx |
| `OpenMed/OpenMed-PII-Dutch-GTEMed-Base-149M-v1` | PII | token-classification | nl | Base | pytorch |
| `OpenMed/OpenMed-PII-Dutch-GTEMed-Base-149M-v1-onnx-android` | PII | token-classification | nl | Base | onnx |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1` | PII | token-classification | nl | Small | pytorch |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1-mlx` | PII | token-classification | nl | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1-onnx-android` | PII | token-classification | nl | Small | onnx |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1` | PII | token-classification | nl | Small | pytorch |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1-mlx` | PII | token-classification | nl | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1-onnx-android` | PII | token-classification | nl | Small | onnx |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Base-149M-v1` | PII | token-classification | nl | Base | pytorch |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Base-149M-v1-onnx-android` | PII | token-classification | nl | Base | onnx |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Large-395M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Large-395M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-NomicMed-Large-395M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-NomicMed-Large-395M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-QwenMed-XLarge-600M-v1` | PII | token-classification | nl | XLarge | pytorch |
| `OpenMed/OpenMed-PII-Dutch-QwenMed-XLarge-600M-v1-onnx-android` | PII | token-classification | nl | XLarge | onnx |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1-mlx` | PII | token-classification | nl | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1` | PII | token-classification | nl | Base | pytorch |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1-mlx` | PII | token-classification | nl | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1-onnx-android` | PII | token-classification | nl | Base | onnx |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1-mlx` | PII | token-classification | nl | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1` | PII | token-classification | nl | Small | pytorch |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1-mlx` | PII | token-classification | nl | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1-onnx-android` | PII | token-classification | nl | Small | onnx |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1` | PII | token-classification | nl | Base | pytorch |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1-mlx` | PII | token-classification | nl | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1-onnx-android` | PII | token-classification | nl | Base | onnx |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1-mlx` | PII | token-classification | nl | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1-mlx` | PII | token-classification | nl | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1` | PII | token-classification | nl | Base | pytorch |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1-mlx` | PII | token-classification | nl | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1-onnx-android` | PII | token-classification | nl | Base | onnx |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1` | PII | token-classification | nl | Large | pytorch |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1-mlx` | PII | token-classification | nl | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1-onnx-android` | PII | token-classification | nl | Large | onnx |
| `OpenMed/OpenMed-PII-EuroMed-Large-210M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-EuroMed-Large-210M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1` | PII | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx` | PII | token-classification | en | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-onnx-android` | PII | token-classification | en | Small | onnx |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1-mlx` | PII | token-classification | fr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1-mlx` | PII | token-classification | fr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1` | PII | token-classification | fr | Base | pytorch |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1-mlx` | PII | token-classification | fr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1-onnx-android` | PII | token-classification | fr | Base | onnx |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1` | PII | token-classification | fr | Base | pytorch |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1-onnx-android` | PII | token-classification | fr | Base | onnx |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Large-395M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Large-395M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1` | PII | token-classification | fr | Base | pytorch |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1-mlx` | PII | token-classification | fr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1-onnx-android` | PII | token-classification | fr | Base | onnx |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1-mlx` | PII | token-classification | fr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1` | PII | token-classification | fr | Base | pytorch |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1-mlx` | PII | token-classification | fr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1-onnx-android` | PII | token-classification | fr | Base | onnx |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1` | PII | token-classification | fr | Base | pytorch |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1-mlx` | PII | token-classification | fr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1-onnx-android` | PII | token-classification | fr | Base | onnx |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1-mlx` | PII | token-classification | fr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1` | PII | token-classification | fr | Base | pytorch |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1-mlx` | PII | token-classification | fr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1-onnx-android` | PII | token-classification | fr | Base | onnx |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1-mlx` | PII | token-classification | fr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1-mlx` | PII | token-classification | fr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1` | PII | token-classification | fr | Base | pytorch |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1-mlx` | PII | token-classification | fr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1-onnx-android` | PII | token-classification | fr | Base | onnx |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1-mlx` | PII | token-classification | fr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1` | PII | token-classification | fr | Small | pytorch |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1-mlx` | PII | token-classification | fr | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1-onnx-android` | PII | token-classification | fr | Small | onnx |
| `OpenMed/OpenMed-PII-French-ClinicalLongformer-Base-149M-v1` | PII | token-classification | fr | Base | pytorch |
| `OpenMed/OpenMed-PII-French-ClinicalLongformer-Base-149M-v1-onnx-android` | PII | token-classification | fr | Base | onnx |
| `OpenMed/OpenMed-PII-French-EuroMed-Large-210M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-EuroMed-Large-210M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1` | PII | token-classification | fr | Small | pytorch |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1-mlx` | PII | token-classification | fr | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1-onnx-android` | PII | token-classification | fr | Small | onnx |
| `OpenMed/OpenMed-PII-French-GTEMed-Base-149M-v1` | PII | token-classification | fr | Base | pytorch |
| `OpenMed/OpenMed-PII-French-GTEMed-Base-149M-v1-onnx-android` | PII | token-classification | fr | Base | onnx |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1` | PII | token-classification | fr | Small | pytorch |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1-mlx` | PII | token-classification | fr | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1-onnx-android` | PII | token-classification | fr | Small | onnx |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1` | PII | token-classification | fr | Small | pytorch |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1-mlx` | PII | token-classification | fr | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1-onnx-android` | PII | token-classification | fr | Small | onnx |
| `OpenMed/OpenMed-PII-French-ModernMed-Base-149M-v1` | PII | token-classification | fr | Base | pytorch |
| `OpenMed/OpenMed-PII-French-ModernMed-Base-149M-v1-onnx-android` | PII | token-classification | fr | Base | onnx |
| `OpenMed/OpenMed-PII-French-ModernMed-Large-395M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-ModernMed-Large-395M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-NomicMed-Large-395M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-NomicMed-Large-395M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-QwenMed-XLarge-600M-v1` | PII | token-classification | fr | XLarge | pytorch |
| `OpenMed/OpenMed-PII-French-QwenMed-XLarge-600M-v1-onnx-android` | PII | token-classification | fr | XLarge | onnx |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1-mlx` | PII | token-classification | fr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1` | PII | token-classification | fr | Base | pytorch |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1-mlx` | PII | token-classification | fr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1-onnx-android` | PII | token-classification | fr | Base | onnx |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1-mlx` | PII | token-classification | fr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1` | PII | token-classification | fr | Small | pytorch |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1-mlx` | PII | token-classification | fr | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1-onnx-android` | PII | token-classification | fr | Small | onnx |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1` | PII | token-classification | fr | Base | pytorch |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1-mlx` | PII | token-classification | fr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1-onnx-android` | PII | token-classification | fr | Base | onnx |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1-mlx` | PII | token-classification | fr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1-mlx` | PII | token-classification | fr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1` | PII | token-classification | fr | Base | pytorch |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1-mlx` | PII | token-classification | fr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1-onnx-android` | PII | token-classification | fr | Base | onnx |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1` | PII | token-classification | fr | Large | pytorch |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1-mlx` | PII | token-classification | fr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1-onnx-android` | PII | token-classification | fr | Large | onnx |
| `OpenMed/OpenMed-PII-GTEMed-Base-149M-v1` | PII | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-PII-GTEMed-Base-149M-v1-onnx-android` | PII | token-classification | en | Base | onnx |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1-mlx` | PII | token-classification | de | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1-mlx` | PII | token-classification | de | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1` | PII | token-classification | de | Base | pytorch |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1-mlx` | PII | token-classification | de | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1-onnx-android` | PII | token-classification | de | Base | onnx |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Base-149M-v1` | PII | token-classification | de | Base | pytorch |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Base-149M-v1-onnx-android` | PII | token-classification | de | Base | onnx |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Large-395M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Large-395M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1` | PII | token-classification | de | Base | pytorch |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1-mlx` | PII | token-classification | de | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1-onnx-android` | PII | token-classification | de | Base | onnx |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1-mlx` | PII | token-classification | de | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1` | PII | token-classification | de | Base | pytorch |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1-mlx` | PII | token-classification | de | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1-onnx-android` | PII | token-classification | de | Base | onnx |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1` | PII | token-classification | de | Base | pytorch |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1-mlx` | PII | token-classification | de | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1-onnx-android` | PII | token-classification | de | Base | onnx |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1-mlx` | PII | token-classification | de | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1` | PII | token-classification | de | Base | pytorch |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1-mlx` | PII | token-classification | de | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1-onnx-android` | PII | token-classification | de | Base | onnx |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1-mlx` | PII | token-classification | de | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1-mlx` | PII | token-classification | de | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1` | PII | token-classification | de | Base | pytorch |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1-mlx` | PII | token-classification | de | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1-onnx-android` | PII | token-classification | de | Base | onnx |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1-mlx` | PII | token-classification | de | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1` | PII | token-classification | de | Small | pytorch |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1-mlx` | PII | token-classification | de | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1-onnx-android` | PII | token-classification | de | Small | onnx |
| `OpenMed/OpenMed-PII-German-ClinicalLongformer-Base-149M-v1` | PII | token-classification | de | Base | pytorch |
| `OpenMed/OpenMed-PII-German-ClinicalLongformer-Base-149M-v1-onnx-android` | PII | token-classification | de | Base | onnx |
| `OpenMed/OpenMed-PII-German-EuroMed-Large-210M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-EuroMed-Large-210M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1` | PII | token-classification | de | Small | pytorch |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1-mlx` | PII | token-classification | de | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1-onnx-android` | PII | token-classification | de | Small | onnx |
| `OpenMed/OpenMed-PII-German-GTEMed-Base-149M-v1` | PII | token-classification | de | Base | pytorch |
| `OpenMed/OpenMed-PII-German-GTEMed-Base-149M-v1-onnx-android` | PII | token-classification | de | Base | onnx |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1` | PII | token-classification | de | Small | pytorch |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1-mlx` | PII | token-classification | de | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1-onnx-android` | PII | token-classification | de | Small | onnx |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1` | PII | token-classification | de | Small | pytorch |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1-mlx` | PII | token-classification | de | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1-onnx-android` | PII | token-classification | de | Small | onnx |
| `OpenMed/OpenMed-PII-German-ModernMed-Base-149M-v1` | PII | token-classification | de | Base | pytorch |
| `OpenMed/OpenMed-PII-German-ModernMed-Base-149M-v1-onnx-android` | PII | token-classification | de | Base | onnx |
| `OpenMed/OpenMed-PII-German-ModernMed-Large-395M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-ModernMed-Large-395M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-NomicMed-Large-395M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-NomicMed-Large-395M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-QwenMed-XLarge-600M-v1` | PII | token-classification | de | XLarge | pytorch |
| `OpenMed/OpenMed-PII-German-QwenMed-XLarge-600M-v1-onnx-android` | PII | token-classification | de | XLarge | onnx |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1-mlx` | PII | token-classification | de | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1` | PII | token-classification | de | Base | pytorch |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1-mlx` | PII | token-classification | de | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1-onnx-android` | PII | token-classification | de | Base | onnx |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1-mlx` | PII | token-classification | de | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1` | PII | token-classification | de | Small | pytorch |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1-mlx` | PII | token-classification | de | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1-onnx-android` | PII | token-classification | de | Small | onnx |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1` | PII | token-classification | de | Base | pytorch |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1-mlx` | PII | token-classification | de | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1-onnx-android` | PII | token-classification | de | Base | onnx |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1-mlx` | PII | token-classification | de | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1-mlx` | PII | token-classification | de | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1` | PII | token-classification | de | Base | pytorch |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1-mlx` | PII | token-classification | de | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1-onnx-android` | PII | token-classification | de | Base | onnx |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1` | PII | token-classification | de | Large | pytorch |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1-mlx` | PII | token-classification | de | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1-onnx-android` | PII | token-classification | de | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1-mlx` | PII | token-classification | hi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1-mlx` | PII | token-classification | hi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1` | PII | token-classification | hi | Base | pytorch |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1-mlx` | PII | token-classification | hi | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1-onnx-android` | PII | token-classification | hi | Base | onnx |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Base-149M-v1` | PII | token-classification | hi | Base | pytorch |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Base-149M-v1-onnx-android` | PII | token-classification | hi | Base | onnx |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Large-395M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Large-395M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1` | PII | token-classification | hi | Base | pytorch |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1-mlx` | PII | token-classification | hi | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1-onnx-android` | PII | token-classification | hi | Base | onnx |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1-mlx` | PII | token-classification | hi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1` | PII | token-classification | hi | Base | pytorch |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1-mlx` | PII | token-classification | hi | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1-onnx-android` | PII | token-classification | hi | Base | onnx |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1` | PII | token-classification | hi | Base | pytorch |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1-mlx` | PII | token-classification | hi | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1-onnx-android` | PII | token-classification | hi | Base | onnx |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1-mlx` | PII | token-classification | hi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1` | PII | token-classification | hi | Base | pytorch |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1-mlx` | PII | token-classification | hi | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1-onnx-android` | PII | token-classification | hi | Base | onnx |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1-mlx` | PII | token-classification | hi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1-mlx` | PII | token-classification | hi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1` | PII | token-classification | hi | Base | pytorch |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1-mlx` | PII | token-classification | hi | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1-onnx-android` | PII | token-classification | hi | Base | onnx |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1-mlx` | PII | token-classification | hi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1` | PII | token-classification | hi | Small | pytorch |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1-mlx` | PII | token-classification | hi | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1-onnx-android` | PII | token-classification | hi | Small | onnx |
| `OpenMed/OpenMed-PII-Hindi-ClinicalLongformer-Base-149M-v1` | PII | token-classification | hi | Base | pytorch |
| `OpenMed/OpenMed-PII-Hindi-ClinicalLongformer-Base-149M-v1-onnx-android` | PII | token-classification | hi | Base | onnx |
| `OpenMed/OpenMed-PII-Hindi-EuroMed-Large-210M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-EuroMed-Large-210M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1` | PII | token-classification | hi | Small | pytorch |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1-mlx` | PII | token-classification | hi | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1-onnx-android` | PII | token-classification | hi | Small | onnx |
| `OpenMed/OpenMed-PII-Hindi-GTEMed-Base-149M-v1` | PII | token-classification | hi | Base | pytorch |
| `OpenMed/OpenMed-PII-Hindi-GTEMed-Base-149M-v1-onnx-android` | PII | token-classification | hi | Base | onnx |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1` | PII | token-classification | hi | Small | pytorch |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1-mlx` | PII | token-classification | hi | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1-onnx-android` | PII | token-classification | hi | Small | onnx |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1` | PII | token-classification | hi | Small | pytorch |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1-mlx` | PII | token-classification | hi | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1-onnx-android` | PII | token-classification | hi | Small | onnx |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Base-149M-v1` | PII | token-classification | hi | Base | pytorch |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Base-149M-v1-onnx-android` | PII | token-classification | hi | Base | onnx |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Large-395M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Large-395M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-NomicMed-Large-395M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-NomicMed-Large-395M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-QwenMed-XLarge-600M-v1` | PII | token-classification | hi | XLarge | pytorch |
| `OpenMed/OpenMed-PII-Hindi-QwenMed-XLarge-600M-v1-onnx-android` | PII | token-classification | hi | XLarge | onnx |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1-mlx` | PII | token-classification | hi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1` | PII | token-classification | hi | Base | pytorch |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1-mlx` | PII | token-classification | hi | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1-onnx-android` | PII | token-classification | hi | Base | onnx |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1-mlx` | PII | token-classification | hi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1` | PII | token-classification | hi | Small | pytorch |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1-mlx` | PII | token-classification | hi | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1-onnx-android` | PII | token-classification | hi | Small | onnx |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1` | PII | token-classification | hi | Base | pytorch |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1-mlx` | PII | token-classification | hi | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1-onnx-android` | PII | token-classification | hi | Base | onnx |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1-mlx` | PII | token-classification | hi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1-mlx` | PII | token-classification | hi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1` | PII | token-classification | hi | Base | pytorch |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1-mlx` | PII | token-classification | hi | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1-onnx-android` | PII | token-classification | hi | Base | onnx |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1` | PII | token-classification | hi | Large | pytorch |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1-mlx` | PII | token-classification | hi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1-onnx-android` | PII | token-classification | hi | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1-mlx` | PII | token-classification | it | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1-mlx` | PII | token-classification | it | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1` | PII | token-classification | it | Base | pytorch |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1-mlx` | PII | token-classification | it | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1-onnx-android` | PII | token-classification | it | Base | onnx |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1` | PII | token-classification | it | Base | pytorch |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1-onnx-android` | PII | token-classification | it | Base | onnx |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1` | PII | token-classification | it | Base | pytorch |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1-mlx` | PII | token-classification | it | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1-onnx-android` | PII | token-classification | it | Base | onnx |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1-mlx` | PII | token-classification | it | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1` | PII | token-classification | it | Base | pytorch |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1-mlx` | PII | token-classification | it | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1-onnx-android` | PII | token-classification | it | Base | onnx |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1` | PII | token-classification | it | Base | pytorch |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1-mlx` | PII | token-classification | it | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1-onnx-android` | PII | token-classification | it | Base | onnx |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1-mlx` | PII | token-classification | it | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1` | PII | token-classification | it | Base | pytorch |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1-mlx` | PII | token-classification | it | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1-onnx-android` | PII | token-classification | it | Base | onnx |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1-mlx` | PII | token-classification | it | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1-mlx` | PII | token-classification | it | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1` | PII | token-classification | it | Base | pytorch |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1-mlx` | PII | token-classification | it | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1-onnx-android` | PII | token-classification | it | Base | onnx |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1-mlx` | PII | token-classification | it | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1` | PII | token-classification | it | Small | pytorch |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1-mlx` | PII | token-classification | it | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1-onnx-android` | PII | token-classification | it | Small | onnx |
| `OpenMed/OpenMed-PII-Italian-ClinicalLongformer-Base-149M-v1` | PII | token-classification | it | Base | pytorch |
| `OpenMed/OpenMed-PII-Italian-ClinicalLongformer-Base-149M-v1-onnx-android` | PII | token-classification | it | Base | onnx |
| `OpenMed/OpenMed-PII-Italian-EuroMed-Large-210M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-EuroMed-Large-210M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1` | PII | token-classification | it | Small | pytorch |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1-mlx` | PII | token-classification | it | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1-onnx-android` | PII | token-classification | it | Small | onnx |
| `OpenMed/OpenMed-PII-Italian-GTEMed-Base-149M-v1` | PII | token-classification | it | Base | pytorch |
| `OpenMed/OpenMed-PII-Italian-GTEMed-Base-149M-v1-onnx-android` | PII | token-classification | it | Base | onnx |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1` | PII | token-classification | it | Small | pytorch |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1-mlx` | PII | token-classification | it | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1-onnx-android` | PII | token-classification | it | Small | onnx |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1` | PII | token-classification | it | Small | pytorch |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1-mlx` | PII | token-classification | it | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1-onnx-android` | PII | token-classification | it | Small | onnx |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Base-149M-v1` | PII | token-classification | it | Base | pytorch |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Base-149M-v1-onnx-android` | PII | token-classification | it | Base | onnx |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Large-395M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Large-395M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-NomicMed-Large-395M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-NomicMed-Large-395M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-QwenMed-XLarge-600M-v1` | PII | token-classification | it | XLarge | pytorch |
| `OpenMed/OpenMed-PII-Italian-QwenMed-XLarge-600M-v1-onnx-android` | PII | token-classification | it | XLarge | onnx |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1-mlx` | PII | token-classification | it | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1` | PII | token-classification | it | Base | pytorch |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1-mlx` | PII | token-classification | it | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1-onnx-android` | PII | token-classification | it | Base | onnx |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1-mlx` | PII | token-classification | it | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1` | PII | token-classification | it | Small | pytorch |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1-mlx` | PII | token-classification | it | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1-onnx-android` | PII | token-classification | it | Small | onnx |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1` | PII | token-classification | it | Base | pytorch |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1-mlx` | PII | token-classification | it | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1-onnx-android` | PII | token-classification | it | Base | onnx |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1-mlx` | PII | token-classification | it | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1-mlx` | PII | token-classification | it | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1` | PII | token-classification | it | Base | pytorch |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1-mlx` | PII | token-classification | it | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1-onnx-android` | PII | token-classification | it | Base | onnx |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1` | PII | token-classification | it | Large | pytorch |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1-mlx` | PII | token-classification | it | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1-onnx-android` | PII | token-classification | it | Large | onnx |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1` | PII | token-classification | ja | Large | pytorch |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1-mlx` | PII | token-classification | ja | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1-onnx-android` | PII | token-classification | ja | Large | onnx |
| `OpenMed/OpenMed-PII-Japanese-NomicMed-Large-395M-v1` | PII | token-classification | ja | Large | pytorch |
| `OpenMed/OpenMed-PII-Japanese-NomicMed-Large-395M-v1-onnx-android` | PII | token-classification | ja | Large | onnx |
| `OpenMed/OpenMed-PII-Japanese-QwenMed-XLarge-600M-v1` | PII | token-classification | ja | XLarge | pytorch |
| `OpenMed/OpenMed-PII-Japanese-QwenMed-XLarge-600M-v1-onnx-android` | PII | token-classification | ja | XLarge | onnx |
| `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1` | PII | token-classification | ko | Large | pytorch |
| `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1-onnx-android` | PII | token-classification | ko | Large | onnx |
| `OpenMed/OpenMed-PII-Korean-QwenMed-XLarge-600M-v1` | PII | token-classification | ko | XLarge | pytorch |
| `OpenMed/OpenMed-PII-Korean-QwenMed-XLarge-600M-v1-onnx-android` | PII | token-classification | ko | XLarge | onnx |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1` | PII | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx` | PII | token-classification | en | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-onnx-android` | PII | token-classification | en | Small | onnx |
| `OpenMed/OpenMed-PII-ModernMed-Base-149M-v1` | PII | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-PII-ModernMed-Base-149M-v1-onnx-android` | PII | token-classification | en | Base | onnx |
| `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-NomicMed-Large-395M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-NomicMed-Large-395M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1` | PII | token-classification | pt | Large | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1-mlx` | PII | token-classification | pt | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1-onnx-android` | PII | token-classification | pt | Large | onnx |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1` | PII | token-classification | pt | Large | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1-mlx` | PII | token-classification | pt | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1-onnx-android` | PII | token-classification | pt | Large | onnx |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1` | PII | token-classification | pt | Base | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1-mlx` | PII | token-classification | pt | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1-onnx-android` | PII | token-classification | pt | Base | onnx |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Base-149M-v1` | PII | token-classification | pt | Base | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Base-149M-v1-onnx-android` | PII | token-classification | pt | Base | onnx |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Large-395M-v1` | PII | token-classification | pt | Large | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Large-395M-v1-onnx-android` | PII | token-classification | pt | Large | onnx |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1` | PII | token-classification | pt | Base | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1-mlx` | PII | token-classification | pt | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1-onnx-android` | PII | token-classification | pt | Base | onnx |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1` | PII | token-classification | pt | Large | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1-mlx` | PII | token-classification | pt | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1-onnx-android` | PII | token-classification | pt | Large | onnx |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1` | PII | token-classification | pt | Base | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1-mlx` | PII | token-classification | pt | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1-onnx-android` | PII | token-classification | pt | Base | onnx |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1` | PII | token-classification | pt | Base | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1-mlx` | PII | token-classification | pt | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1-onnx-android` | PII | token-classification | pt | Base | onnx |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1` | PII | token-classification | pt | Large | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1-mlx` | PII | token-classification | pt | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1-onnx-android` | PII | token-classification | pt | Large | onnx |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1` | PII | token-classification | pt | Base | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1-mlx` | PII | token-classification | pt | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1-onnx-android` | PII | token-classification | pt | Base | onnx |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1` | PII | token-classification | pt | Large | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1-mlx` | PII | token-classification | pt | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1-onnx-android` | PII | token-classification | pt | Large | onnx |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1` | PII | token-classification | pt | Large | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1-mlx` | PII | token-classification | pt | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1-onnx-android` | PII | token-classification | pt | Large | onnx |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1` | PII | token-classification | pt | Base | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1-mlx` | PII | token-classification | pt | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1-onnx-android` | PII | token-classification | pt | Base | onnx |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1` | PII | token-classification | pt | Large | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1-mlx` | PII | token-classification | pt | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1-onnx-android` | PII | token-classification | pt | Large | onnx |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1` | PII | token-classification | pt | Small | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1-mlx` | PII | token-classification | pt | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1-onnx-android` | PII | token-classification | pt | Small | onnx |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalLongformer-Base-149M-v1` | PII | token-classification | pt | Base | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalLongformer-Base-149M-v1-onnx-android` | PII | token-classification | pt | Base | onnx |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1` | PII | token-classification | pt | Small | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1-mlx` | PII | token-classification | pt | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1-onnx-android` | PII | token-classification | pt | Small | onnx |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1` | PII | token-classification | pt | Small | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1-mlx` | PII | token-classification | pt | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1-onnx-android` | PII | token-classification | pt | Small | onnx |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1` | PII | token-classification | pt | Small | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1-mlx` | PII | token-classification | pt | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1-onnx-android` | PII | token-classification | pt | Small | onnx |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Base-149M-v1` | PII | token-classification | pt | Base | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Base-149M-v1-onnx-android` | PII | token-classification | pt | Base | onnx |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Large-395M-v1` | PII | token-classification | pt | Large | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Large-395M-v1-onnx-android` | PII | token-classification | pt | Large | onnx |
| `OpenMed/OpenMed-PII-Portuguese-NomicMed-Large-395M-v1` | PII | token-classification | pt | Large | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-NomicMed-Large-395M-v1-onnx-android` | PII | token-classification | pt | Large | onnx |
| `OpenMed/OpenMed-PII-Portuguese-QwenMed-XLarge-600M-v1` | PII | token-classification | pt | XLarge | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-QwenMed-XLarge-600M-v1-onnx-android` | PII | token-classification | pt | XLarge | onnx |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | PII | token-classification | pt | Large | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1-mlx` | PII | token-classification | pt | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1-onnx-android` | PII | token-classification | pt | Large | onnx |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1` | PII | token-classification | pt | Base | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1-mlx` | PII | token-classification | pt | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1-onnx-android` | PII | token-classification | pt | Base | onnx |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1` | PII | token-classification | pt | Large | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1-mlx` | PII | token-classification | pt | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1-onnx-android` | PII | token-classification | pt | Large | onnx |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1` | PII | token-classification | pt | Small | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1-mlx` | PII | token-classification | pt | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1-onnx-android` | PII | token-classification | pt | Small | onnx |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1` | PII | token-classification | pt | Base | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1-mlx` | PII | token-classification | pt | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1-onnx-android` | PII | token-classification | pt | Base | onnx |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1` | PII | token-classification | pt | Large | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1-mlx` | PII | token-classification | pt | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1-onnx-android` | PII | token-classification | pt | Large | onnx |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1` | PII | token-classification | pt | Base | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1-mlx` | PII | token-classification | pt | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1-onnx-android` | PII | token-classification | pt | Base | onnx |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1` | PII | token-classification | pt | Large | pytorch |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1-mlx` | PII | token-classification | pt | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1-onnx-android` | PII | token-classification | pt | Large | onnx |
| `OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1` | PII | token-classification | en | XLarge | pytorch |
| `OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1-onnx-android` | PII | token-classification | en | XLarge | onnx |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1-mlx` | PII | token-classification | es | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1-mlx` | PII | token-classification | es | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1` | PII | token-classification | es | Base | pytorch |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1-mlx` | PII | token-classification | es | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1-onnx-android` | PII | token-classification | es | Base | onnx |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Base-149M-v1` | PII | token-classification | es | Base | pytorch |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Base-149M-v1-onnx-android` | PII | token-classification | es | Base | onnx |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Large-395M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Large-395M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1` | PII | token-classification | es | Base | pytorch |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1-mlx` | PII | token-classification | es | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1-onnx-android` | PII | token-classification | es | Base | onnx |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1-mlx` | PII | token-classification | es | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1` | PII | token-classification | es | Base | pytorch |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1-mlx` | PII | token-classification | es | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1-onnx-android` | PII | token-classification | es | Base | onnx |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1` | PII | token-classification | es | Base | pytorch |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1-mlx` | PII | token-classification | es | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1-onnx-android` | PII | token-classification | es | Base | onnx |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1-mlx` | PII | token-classification | es | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1` | PII | token-classification | es | Base | pytorch |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1-mlx` | PII | token-classification | es | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1-onnx-android` | PII | token-classification | es | Base | onnx |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1-mlx` | PII | token-classification | es | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1-mlx` | PII | token-classification | es | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1` | PII | token-classification | es | Base | pytorch |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1-mlx` | PII | token-classification | es | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1-onnx-android` | PII | token-classification | es | Base | onnx |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1-mlx` | PII | token-classification | es | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1` | PII | token-classification | es | Small | pytorch |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1-mlx` | PII | token-classification | es | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1-onnx-android` | PII | token-classification | es | Small | onnx |
| `OpenMed/OpenMed-PII-Spanish-ClinicalLongformer-Base-149M-v1` | PII | token-classification | es | Base | pytorch |
| `OpenMed/OpenMed-PII-Spanish-ClinicalLongformer-Base-149M-v1-onnx-android` | PII | token-classification | es | Base | onnx |
| `OpenMed/OpenMed-PII-Spanish-EuroMed-Large-210M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-EuroMed-Large-210M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1` | PII | token-classification | es | Small | pytorch |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1-mlx` | PII | token-classification | es | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1-onnx-android` | PII | token-classification | es | Small | onnx |
| `OpenMed/OpenMed-PII-Spanish-GTEMed-Base-149M-v1` | PII | token-classification | es | Base | pytorch |
| `OpenMed/OpenMed-PII-Spanish-GTEMed-Base-149M-v1-onnx-android` | PII | token-classification | es | Base | onnx |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1` | PII | token-classification | es | Small | pytorch |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1-mlx` | PII | token-classification | es | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1-onnx-android` | PII | token-classification | es | Small | onnx |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1` | PII | token-classification | es | Small | pytorch |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1-mlx` | PII | token-classification | es | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1-onnx-android` | PII | token-classification | es | Small | onnx |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Base-149M-v1` | PII | token-classification | es | Base | pytorch |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Base-149M-v1-onnx-android` | PII | token-classification | es | Base | onnx |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Large-395M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Large-395M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-NomicMed-Large-395M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-NomicMed-Large-395M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-QwenMed-XLarge-600M-v1` | PII | token-classification | es | XLarge | pytorch |
| `OpenMed/OpenMed-PII-Spanish-QwenMed-XLarge-600M-v1-onnx-android` | PII | token-classification | es | XLarge | onnx |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1-mlx` | PII | token-classification | es | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1` | PII | token-classification | es | Base | pytorch |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1-mlx` | PII | token-classification | es | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1-onnx-android` | PII | token-classification | es | Base | onnx |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1-mlx` | PII | token-classification | es | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1` | PII | token-classification | es | Small | pytorch |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1-mlx` | PII | token-classification | es | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1-onnx-android` | PII | token-classification | es | Small | onnx |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1` | PII | token-classification | es | Base | pytorch |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1-mlx` | PII | token-classification | es | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1-onnx-android` | PII | token-classification | es | Base | onnx |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1-mlx` | PII | token-classification | es | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1-mlx` | PII | token-classification | es | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1` | PII | token-classification | es | Base | pytorch |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1-mlx` | PII | token-classification | es | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1-onnx-android` | PII | token-classification | es | Base | onnx |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1` | PII | token-classification | es | Large | pytorch |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1-mlx` | PII | token-classification | es | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1-onnx-android` | PII | token-classification | es | Large | onnx |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1` | PII | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1-mlx` | PII | token-classification | en | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1-onnx-android` | PII | token-classification | en | Base | onnx |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | PII | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx` | PII | token-classification | en | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-onnx-android` | PII | token-classification | en | Small | onnx |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1` | PII | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1-mlx` | PII | token-classification | en | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1-onnx-android` | PII | token-classification | en | Base | onnx |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1-mlx` | PII | token-classification | te | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1-mlx` | PII | token-classification | te | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1` | PII | token-classification | te | Base | pytorch |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1-mlx` | PII | token-classification | te | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1-onnx-android` | PII | token-classification | te | Base | onnx |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Base-149M-v1` | PII | token-classification | te | Base | pytorch |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Base-149M-v1-onnx-android` | PII | token-classification | te | Base | onnx |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Large-395M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Large-395M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1` | PII | token-classification | te | Base | pytorch |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1-mlx` | PII | token-classification | te | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1-onnx-android` | PII | token-classification | te | Base | onnx |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1-mlx` | PII | token-classification | te | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1` | PII | token-classification | te | Base | pytorch |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1-mlx` | PII | token-classification | te | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1-onnx-android` | PII | token-classification | te | Base | onnx |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1` | PII | token-classification | te | Base | pytorch |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1-mlx` | PII | token-classification | te | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1-onnx-android` | PII | token-classification | te | Base | onnx |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1-mlx` | PII | token-classification | te | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1` | PII | token-classification | te | Base | pytorch |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1-mlx` | PII | token-classification | te | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1-onnx-android` | PII | token-classification | te | Base | onnx |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1-mlx` | PII | token-classification | te | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1-mlx` | PII | token-classification | te | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1` | PII | token-classification | te | Base | pytorch |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1-mlx` | PII | token-classification | te | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1-onnx-android` | PII | token-classification | te | Base | onnx |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1-mlx` | PII | token-classification | te | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1` | PII | token-classification | te | Small | pytorch |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1-mlx` | PII | token-classification | te | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1-onnx-android` | PII | token-classification | te | Small | onnx |
| `OpenMed/OpenMed-PII-Telugu-ClinicalLongformer-Base-149M-v1` | PII | token-classification | te | Base | pytorch |
| `OpenMed/OpenMed-PII-Telugu-ClinicalLongformer-Base-149M-v1-onnx-android` | PII | token-classification | te | Base | onnx |
| `OpenMed/OpenMed-PII-Telugu-EuroMed-Large-210M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-EuroMed-Large-210M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1` | PII | token-classification | te | Small | pytorch |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1-mlx` | PII | token-classification | te | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1-onnx-android` | PII | token-classification | te | Small | onnx |
| `OpenMed/OpenMed-PII-Telugu-GTEMed-Base-149M-v1` | PII | token-classification | te | Base | pytorch |
| `OpenMed/OpenMed-PII-Telugu-GTEMed-Base-149M-v1-onnx-android` | PII | token-classification | te | Base | onnx |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1` | PII | token-classification | te | Small | pytorch |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1-mlx` | PII | token-classification | te | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1-onnx-android` | PII | token-classification | te | Small | onnx |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1` | PII | token-classification | te | Small | pytorch |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1-mlx` | PII | token-classification | te | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1-onnx-android` | PII | token-classification | te | Small | onnx |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Base-149M-v1` | PII | token-classification | te | Base | pytorch |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Base-149M-v1-onnx-android` | PII | token-classification | te | Base | onnx |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Large-395M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Large-395M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-NomicMed-Large-395M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-NomicMed-Large-395M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-QwenMed-XLarge-600M-v1` | PII | token-classification | te | XLarge | pytorch |
| `OpenMed/OpenMed-PII-Telugu-QwenMed-XLarge-600M-v1-onnx-android` | PII | token-classification | te | XLarge | onnx |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1-mlx` | PII | token-classification | te | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1` | PII | token-classification | te | Base | pytorch |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1-mlx` | PII | token-classification | te | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1-onnx-android` | PII | token-classification | te | Base | onnx |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1-mlx` | PII | token-classification | te | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1` | PII | token-classification | te | Small | pytorch |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1-mlx` | PII | token-classification | te | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1-onnx-android` | PII | token-classification | te | Small | onnx |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1` | PII | token-classification | te | Base | pytorch |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1-mlx` | PII | token-classification | te | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1-onnx-android` | PII | token-classification | te | Base | onnx |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1-mlx` | PII | token-classification | te | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1-mlx` | PII | token-classification | te | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1` | PII | token-classification | te | Base | pytorch |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1-mlx` | PII | token-classification | te | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1-onnx-android` | PII | token-classification | te | Base | onnx |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1` | PII | token-classification | te | Large | pytorch |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1-mlx` | PII | token-classification | te | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1-onnx-android` | PII | token-classification | te | Large | onnx |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1` | PII | token-classification | tr | Large | pytorch |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1-mlx` | PII | token-classification | tr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1-onnx-android` | PII | token-classification | tr | Large | onnx |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1` | PII | token-classification | tr | Large | pytorch |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1-mlx` | PII | token-classification | tr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1-onnx-android` | PII | token-classification | tr | Large | onnx |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1` | PII | token-classification | tr | Base | pytorch |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1-mlx` | PII | token-classification | tr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1-onnx-android` | PII | token-classification | tr | Base | onnx |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Base-149M-v1` | PII | token-classification | tr | Base | pytorch |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Base-149M-v1-onnx-android` | PII | token-classification | tr | Base | onnx |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Large-395M-v1` | PII | token-classification | tr | Large | pytorch |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Large-395M-v1-onnx-android` | PII | token-classification | tr | Large | onnx |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1` | PII | token-classification | tr | Base | pytorch |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1-mlx` | PII | token-classification | tr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1-onnx-android` | PII | token-classification | tr | Base | onnx |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1` | PII | token-classification | tr | Large | pytorch |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1-mlx` | PII | token-classification | tr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1-onnx-android` | PII | token-classification | tr | Large | onnx |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1` | PII | token-classification | tr | Base | pytorch |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1-mlx` | PII | token-classification | tr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1-onnx-android` | PII | token-classification | tr | Base | onnx |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1` | PII | token-classification | tr | Base | pytorch |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1-mlx` | PII | token-classification | tr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1-onnx-android` | PII | token-classification | tr | Base | onnx |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1` | PII | token-classification | tr | Large | pytorch |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1-mlx` | PII | token-classification | tr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1-onnx-android` | PII | token-classification | tr | Large | onnx |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1` | PII | token-classification | tr | Base | pytorch |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1-mlx` | PII | token-classification | tr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1-onnx-android` | PII | token-classification | tr | Base | onnx |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1` | PII | token-classification | tr | Large | pytorch |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1-mlx` | PII | token-classification | tr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1-onnx-android` | PII | token-classification | tr | Large | onnx |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1` | PII | token-classification | tr | Large | pytorch |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1-mlx` | PII | token-classification | tr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1-onnx-android` | PII | token-classification | tr | Large | onnx |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1` | PII | token-classification | tr | Base | pytorch |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1-mlx` | PII | token-classification | tr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1-onnx-android` | PII | token-classification | tr | Base | onnx |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1` | PII | token-classification | tr | Large | pytorch |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1-mlx` | PII | token-classification | tr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1-onnx-android` | PII | token-classification | tr | Large | onnx |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1` | PII | token-classification | tr | Small | pytorch |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1-mlx` | PII | token-classification | tr | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1-onnx-android` | PII | token-classification | tr | Small | onnx |
| `OpenMed/OpenMed-PII-Turkish-ClinicalLongformer-Base-149M-v1` | PII | token-classification | tr | Base | pytorch |
| `OpenMed/OpenMed-PII-Turkish-ClinicalLongformer-Base-149M-v1-onnx-android` | PII | token-classification | tr | Base | onnx |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1` | PII | token-classification | tr | Small | pytorch |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1-mlx` | PII | token-classification | tr | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1-onnx-android` | PII | token-classification | tr | Small | onnx |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1` | PII | token-classification | tr | Small | pytorch |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1-mlx` | PII | token-classification | tr | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1-onnx-android` | PII | token-classification | tr | Small | onnx |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1` | PII | token-classification | tr | Small | pytorch |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1-mlx` | PII | token-classification | tr | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1-onnx-android` | PII | token-classification | tr | Small | onnx |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Base-149M-v1` | PII | token-classification | tr | Base | pytorch |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Base-149M-v1-onnx-android` | PII | token-classification | tr | Base | onnx |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Large-395M-v1` | PII | token-classification | tr | Large | pytorch |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Large-395M-v1-onnx-android` | PII | token-classification | tr | Large | onnx |
| `OpenMed/OpenMed-PII-Turkish-NomicMed-Large-395M-v1` | PII | token-classification | tr | Large | pytorch |
| `OpenMed/OpenMed-PII-Turkish-NomicMed-Large-395M-v1-onnx-android` | PII | token-classification | tr | Large | onnx |
| `OpenMed/OpenMed-PII-Turkish-QwenMed-XLarge-600M-v1` | PII | token-classification | tr | XLarge | pytorch |
| `OpenMed/OpenMed-PII-Turkish-QwenMed-XLarge-600M-v1-onnx-android` | PII | token-classification | tr | XLarge | onnx |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1` | PII | token-classification | tr | Large | pytorch |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1-mlx` | PII | token-classification | tr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1-onnx-android` | PII | token-classification | tr | Large | onnx |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1` | PII | token-classification | tr | Base | pytorch |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1-mlx` | PII | token-classification | tr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1-onnx-android` | PII | token-classification | tr | Base | onnx |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1` | PII | token-classification | tr | Large | pytorch |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1-mlx` | PII | token-classification | tr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1-onnx-android` | PII | token-classification | tr | Large | onnx |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1` | PII | token-classification | tr | Small | pytorch |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx` | PII | token-classification | tr | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-onnx-android` | PII | token-classification | tr | Small | onnx |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1` | PII | token-classification | tr | Base | pytorch |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1-mlx` | PII | token-classification | tr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1-onnx-android` | PII | token-classification | tr | Base | onnx |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1` | PII | token-classification | tr | Large | pytorch |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1-mlx` | PII | token-classification | tr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1-onnx-android` | PII | token-classification | tr | Large | onnx |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1` | PII | token-classification | tr | Base | pytorch |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1-mlx` | PII | token-classification | tr | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1-onnx-android` | PII | token-classification | tr | Base | onnx |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1` | PII | token-classification | tr | Large | pytorch |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1-mlx` | PII | token-classification | tr | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1-onnx-android` | PII | token-classification | tr | Large | onnx |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1` | PII | token-classification | vi | Large | pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1-mlx` | PII | token-classification | vi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1-onnx-android` | PII | token-classification | vi | Large | onnx |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1` | PII | token-classification | vi | Large | pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1-mlx` | PII | token-classification | vi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1-onnx-android` | PII | token-classification | vi | Large | onnx |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1` | PII | token-classification | vi | Large | pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1-mlx` | PII | token-classification | vi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1-onnx-android` | PII | token-classification | vi | Large | onnx |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1` | PII | token-classification | vi | Large | pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1-mlx` | PII | token-classification | vi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1-onnx-android` | PII | token-classification | vi | Large | onnx |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1` | PII | token-classification | vi | Base | pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1-mlx` | PII | token-classification | vi | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1-onnx-android` | PII | token-classification | vi | Base | onnx |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1` | PII | token-classification | vi | Large | pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1-mlx` | PII | token-classification | vi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1-onnx-android` | PII | token-classification | vi | Large | onnx |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1` | PII | token-classification | vi | Small | pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1-mlx` | PII | token-classification | vi | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1-onnx-android` | PII | token-classification | vi | Small | onnx |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1` | PII | token-classification | vi | Large | pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1-mlx` | PII | token-classification | vi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1-onnx-android` | PII | token-classification | vi | Large | onnx |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1` | PII | token-classification | vi | Large | pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1-mlx` | PII | token-classification | vi | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1-onnx-android` | PII | token-classification | vi | Large | onnx |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1` | PII | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1-mlx` | PII | token-classification | en | Base | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1-onnx-android` | PII | token-classification | en | Base | onnx |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1` | PII | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1-mlx` | PII | token-classification | en | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1-onnx-android` | PII | token-classification | en | Large | onnx |
| `OpenMed/privacy-filter-mlx-8bit` | PII | token-classification | en | - | mlx-8bit, pytorch |
| `OpenMed/privacy-filter-multilingual` | PII | token-classification | am, ar, as, cs, da, de, el, en, es, fr, he, hi, id, it, ja, mr, nl, no, or, pt, ro, sv, sw, te, th, tr, uk, xh, zu | - | pytorch |
| `OpenMed/privacy-filter-multilingual-mlx` | PII | token-classification | am, ar, as, cs, da, de, el, en, es, fr, he, hi, id, it, ja, mr, nl, no, or, pt, ro, sv, sw, te, th, tr, uk, xh, zu | - | mlx-fp, pytorch |
| `OpenMed/privacy-filter-multilingual-mlx-8bit` | PII | token-classification | am, ar, as, cs, da, de, el, en, es, fr, he, hi, id, it, ja, mr, nl, no, or, pt, ro, sv, sw, te, th, tr, uk, xh, zu | - | mlx-8bit, pytorch |
| `OpenMed/privacy-filter-multilingual-v2` | PII | token-classification | am, ar, as, cs, da, de, el, en, es, fr, he, hi, id, it, ja, mr, nl, no, or, pt, ro, sv, sw, te, th, tr, uk, xh, zu | - | pytorch |
| `OpenMed/privacy-filter-multilingual-v2-mlx` | PII | token-classification | am, ar, as, cs, da, de, el, en, es, fr, he, hi, id, it, ja, mr, nl, no, or, pt, ro, sv, sw, te, th, tr, uk, xh, zu | - | mlx-fp, pytorch |
| `OpenMed/privacy-filter-multilingual-v2-mlx-8bit` | PII | token-classification | am, ar, as, cs, da, de, el, en, es, fr, he, hi, id, it, ja, mr, nl, no, or, pt, ro, sv, sw, te, th, tr, uk, xh, zu | - | mlx-8bit, pytorch |
| `OpenMed/privacy-filter-nemotron` | PII | token-classification | en | - | pytorch |
| `OpenMed/privacy-filter-nemotron-mlx` | PII | token-classification | en | - | mlx-fp, pytorch |
| `OpenMed/privacy-filter-nemotron-mlx-8bit` | PII | token-classification | en | - | mlx-8bit, pytorch |
| `OpenMed/privacy-filter-nemotron-v2` | PII | token-classification | de, en, es, fr, it, nl, pt | - | pytorch |
| `OpenMed/privacy-filter-nemotron-v2-mlx` | PII | token-classification | de, en, es, fr, it, nl, pt | - | mlx-fp, pytorch |
| `OpenMed/privacy-filter-nemotron-v2-mlx-8bit` | PII | token-classification | de, en, es, fr, it, nl, pt | - | mlx-8bit, pytorch |
| `OpenMed/Ministral-3B-MedVL` | Vision | visual-question-answering | - | - | pytorch |
| `OpenMed/Qwen2.5-3B-MedVL` | Vision | visual-question-answering | - | - | pytorch |
| `OpenMed/Qwen3.5-2B-MedVL` | Vision | visual-question-answering | - | - | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Anatomy-Base-220M` | ZeroShot | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Anatomy-Large-459M` | ZeroShot | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Anatomy-Large-459M-mlx` | ZeroShot | token-classification | - | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Anatomy-Medium-209M` | ZeroShot | token-classification | en | Medium | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Anatomy-Medium-209M-mlx` | ZeroShot | token-classification | - | Medium | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Anatomy-Multi-209M` | ZeroShot | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Anatomy-Multi-209M-mlx` | ZeroShot | token-classification | - | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Anatomy-Small-166M` | ZeroShot | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Anatomy-Small-166M-mlx` | ZeroShot | token-classification | - | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Anatomy-Tiny-60M` | ZeroShot | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Anatomy-XLarge-770M` | ZeroShot | token-classification | en | XLarge | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-BloodCancer-Base-220M` | ZeroShot | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-BloodCancer-Large-459M` | ZeroShot | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-BloodCancer-Large-459M-mlx` | ZeroShot | token-classification | - | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-BloodCancer-Medium-209M` | ZeroShot | token-classification | en | Medium | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-BloodCancer-Medium-209M-mlx` | ZeroShot | token-classification | - | Medium | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-BloodCancer-Multi-209M` | ZeroShot | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-BloodCancer-Multi-209M-mlx` | ZeroShot | token-classification | - | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-BloodCancer-Small-166M` | ZeroShot | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-BloodCancer-Small-166M-mlx` | ZeroShot | token-classification | - | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-BloodCancer-Tiny-60M` | ZeroShot | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-BloodCancer-XLarge-770M` | ZeroShot | token-classification | en | XLarge | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Chemical-Base-220M` | ZeroShot | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Chemical-Large-459M` | ZeroShot | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Chemical-Large-459M-mlx` | ZeroShot | token-classification | - | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Chemical-Medium-209M` | ZeroShot | token-classification | en | Medium | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Chemical-Medium-209M-mlx` | ZeroShot | token-classification | - | Medium | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Chemical-Multi-209M` | ZeroShot | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Chemical-Multi-209M-mlx` | ZeroShot | token-classification | - | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Chemical-Small-166M` | ZeroShot | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Chemical-Small-166M-mlx` | ZeroShot | token-classification | - | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Chemical-Tiny-60M` | ZeroShot | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Chemical-XLarge-770M` | ZeroShot | token-classification | en | XLarge | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-DNA-Base-220M` | ZeroShot | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-DNA-Large-459M` | ZeroShot | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-DNA-Large-459M-mlx` | ZeroShot | token-classification | - | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-DNA-Medium-209M` | ZeroShot | token-classification | en | Medium | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-DNA-Medium-209M-mlx` | ZeroShot | token-classification | - | Medium | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-DNA-Multi-209M` | ZeroShot | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-DNA-Multi-209M-mlx` | ZeroShot | token-classification | - | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-DNA-Small-166M` | ZeroShot | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-DNA-Small-166M-mlx` | ZeroShot | token-classification | - | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-DNA-Tiny-60M` | ZeroShot | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-DNA-XLarge-770M` | ZeroShot | token-classification | en | XLarge | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Disease-Base-220M` | ZeroShot | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Disease-Large-459M` | ZeroShot | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Disease-Large-459M-mlx` | ZeroShot | token-classification | - | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Disease-Medium-209M` | ZeroShot | token-classification | en | Medium | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Disease-Medium-209M-mlx` | ZeroShot | token-classification | - | Medium | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Disease-Multi-209M` | ZeroShot | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Disease-Multi-209M-mlx` | ZeroShot | token-classification | - | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Disease-Small-166M` | ZeroShot | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Disease-Small-166M-mlx` | ZeroShot | token-classification | - | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Disease-Tiny-60M` | ZeroShot | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Disease-XLarge-770M` | ZeroShot | token-classification | en | XLarge | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genome-Base-220M` | ZeroShot | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genome-Large-459M` | ZeroShot | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genome-Large-459M-mlx` | ZeroShot | token-classification | - | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genome-Medium-209M` | ZeroShot | token-classification | en | Medium | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genome-Medium-209M-mlx` | ZeroShot | token-classification | - | Medium | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genome-Multi-209M` | ZeroShot | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genome-Multi-209M-mlx` | ZeroShot | token-classification | - | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genome-Small-166M` | ZeroShot | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genome-Small-166M-mlx` | ZeroShot | token-classification | - | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genome-Tiny-60M` | ZeroShot | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genome-XLarge-770M` | ZeroShot | token-classification | en | XLarge | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genomic-Base-220M` | ZeroShot | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genomic-Large-459M` | ZeroShot | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genomic-Large-459M-mlx` | ZeroShot | token-classification | - | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genomic-Medium-209M` | ZeroShot | token-classification | en | Medium | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genomic-Medium-209M-mlx` | ZeroShot | token-classification | - | Medium | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genomic-Multi-209M` | ZeroShot | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genomic-Multi-209M-mlx` | ZeroShot | token-classification | - | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genomic-Small-166M` | ZeroShot | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genomic-Small-166M-mlx` | ZeroShot | token-classification | - | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genomic-Tiny-60M` | ZeroShot | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Genomic-XLarge-770M` | ZeroShot | token-classification | en | XLarge | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Oncology-Base-220M` | ZeroShot | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Oncology-Large-459M` | ZeroShot | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Oncology-Large-459M-mlx` | ZeroShot | token-classification | - | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Oncology-Medium-209M` | ZeroShot | token-classification | en | Medium | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Oncology-Medium-209M-mlx` | ZeroShot | token-classification | - | Medium | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Oncology-Multi-209M` | ZeroShot | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Oncology-Multi-209M-mlx` | ZeroShot | token-classification | - | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Oncology-Small-166M` | ZeroShot | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Oncology-Small-166M-mlx` | ZeroShot | token-classification | - | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Oncology-Tiny-60M` | ZeroShot | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Oncology-XLarge-770M` | ZeroShot | token-classification | en | XLarge | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Organism-Base-220M` | ZeroShot | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Organism-Large-459M` | ZeroShot | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Organism-Large-459M-mlx` | ZeroShot | token-classification | - | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Organism-Medium-209M` | ZeroShot | token-classification | en | Medium | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Organism-Medium-209M-mlx` | ZeroShot | token-classification | - | Medium | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Organism-Multi-209M` | ZeroShot | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Organism-Multi-209M-mlx` | ZeroShot | token-classification | - | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Organism-Small-166M` | ZeroShot | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Organism-Small-166M-mlx` | ZeroShot | token-classification | - | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Organism-Tiny-60M` | ZeroShot | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Organism-XLarge-770M` | ZeroShot | token-classification | en | XLarge | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pathology-Base-220M` | ZeroShot | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pathology-Large-459M` | ZeroShot | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pathology-Large-459M-mlx` | ZeroShot | token-classification | - | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pathology-Medium-209M` | ZeroShot | token-classification | en | Medium | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pathology-Medium-209M-mlx` | ZeroShot | token-classification | - | Medium | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pathology-Multi-209M` | ZeroShot | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pathology-Multi-209M-mlx` | ZeroShot | token-classification | - | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pathology-Small-166M` | ZeroShot | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pathology-Small-166M-mlx` | ZeroShot | token-classification | - | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pathology-Tiny-60M` | ZeroShot | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pathology-XLarge-770M` | ZeroShot | token-classification | en | XLarge | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pharma-Base-220M` | ZeroShot | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pharma-Large-459M` | ZeroShot | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pharma-Large-459M-mlx` | ZeroShot | token-classification | - | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pharma-Medium-209M` | ZeroShot | token-classification | en | Medium | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pharma-Medium-209M-mlx` | ZeroShot | token-classification | - | Medium | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pharma-Multi-209M` | ZeroShot | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pharma-Multi-209M-mlx` | ZeroShot | token-classification | - | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pharma-Small-166M` | ZeroShot | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pharma-Small-166M-mlx` | ZeroShot | token-classification | - | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pharma-Tiny-60M` | ZeroShot | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Pharma-XLarge-770M` | ZeroShot | token-classification | en | XLarge | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Protein-Base-220M` | ZeroShot | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Protein-Large-459M` | ZeroShot | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Protein-Large-459M-mlx` | ZeroShot | token-classification | - | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Protein-Medium-209M` | ZeroShot | token-classification | en | Medium | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Protein-Medium-209M-mlx` | ZeroShot | token-classification | - | Medium | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Protein-Multi-209M` | ZeroShot | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Protein-Multi-209M-mlx` | ZeroShot | token-classification | - | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Protein-Small-166M` | ZeroShot | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Protein-Small-166M-mlx` | ZeroShot | token-classification | - | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Protein-Tiny-60M` | ZeroShot | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Protein-XLarge-770M` | ZeroShot | token-classification | en | XLarge | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Species-Base-220M` | ZeroShot | token-classification | en | Base | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Species-Large-459M` | ZeroShot | token-classification | en | Large | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Species-Large-459M-mlx` | ZeroShot | token-classification | - | Large | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Species-Medium-209M` | ZeroShot | token-classification | en | Medium | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Species-Medium-209M-mlx` | ZeroShot | token-classification | - | Medium | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Species-Multi-209M` | ZeroShot | token-classification | en | - | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Species-Multi-209M-mlx` | ZeroShot | token-classification | - | - | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Species-Small-166M` | ZeroShot | token-classification | en | Small | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Species-Small-166M-mlx` | ZeroShot | token-classification | - | Small | mlx-fp, pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Species-Tiny-60M` | ZeroShot | token-classification | en | Tiny | pytorch |
| `OpenMed/OpenMed-ZeroShot-NER-Species-XLarge-770M` | ZeroShot | token-classification | en | XLarge | pytorch |
<!-- END MANIFEST MODEL TABLE -->

## Manifest benchmark evidence

The committed manifest contains 47 model rows with named benchmark evidence. Missing metrics remain explicit rather than being inferred.

<!-- BEGIN MANIFEST BENCHMARK TABLE -->
| Model | Suite | Dataset | Micro F1 | Recall | Leakage |
|---|---|---|---:|---:|---:|
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` | - | AI4Privacy + Synthetic Bengali PII | 0.857 | 0.8619 | - |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` | - | AI4Privacy + Synthetic Chinese PII | 0.7589 | 0.7517 | - |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-EuroMed-Large-210M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-GTEMed-Base-149M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-ModernMed-Base-149M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-NomicMed-Large-395M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/privacy-filter-multilingual` | - | ai4privacy/pii-masking-200k | - | - | - |
| `OpenMed/privacy-filter-multilingual-mlx` | - | ai4privacy/pii-masking-200k | - | - | - |
| `OpenMed/privacy-filter-multilingual-mlx-8bit` | - | ai4privacy/pii-masking-200k | - | - | - |
| `OpenMed/privacy-filter-multilingual-v2` | - | ai4privacy/pii-masking-200k | - | - | - |
| `OpenMed/privacy-filter-multilingual-v2-mlx` | - | ai4privacy/pii-masking-200k | - | - | - |
| `OpenMed/privacy-filter-multilingual-v2-mlx-8bit` | - | ai4privacy/pii-masking-200k | - | - | - |
| `OpenMed/privacy-filter-nemotron` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/privacy-filter-nemotron-mlx` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/privacy-filter-nemotron-mlx-8bit` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/privacy-filter-nemotron-v2` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/privacy-filter-nemotron-v2-mlx` | - | nvidia/Nemotron-PII | - | - | - |
| `OpenMed/privacy-filter-nemotron-v2-mlx-8bit` | - | nvidia/Nemotron-PII | - | - | - |
<!-- END MANIFEST BENCHMARK TABLE -->
<!-- END MANIFEST CATALOG SURFACES -->
