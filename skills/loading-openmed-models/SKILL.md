---
name: loading-openmed-models
description: "Load OpenMed clinical/biomedical NER models from the Hugging Face Hub or a local path and reuse them efficiently across calls. Use when the user wants to load an OpenMed model, control the model cache, run fully offline after a one-time download, reuse a ModelLoader to avoid reloading, set a cache_dir or device, or pick between a registry key, a full Hugging Face id, and a local directory. Pairs with choosing-openmed-models (pick the model) and extracting-clinical-entities (run it)."
license: Apache-2.0
metadata:
  project: OpenMed
  category: openmed-core
  pairs: adjacent
  version: "1.0"
---

# Loading OpenMed Models

OpenMed models download **once** from the Hugging Face Hub into a local cache,
then run **fully on-device** — no network, no telemetry. This skill covers how to
load a model, reuse it across many calls without reloading weights, point at a
local copy, and run offline.

## When to use

- You are about to run NER repeatedly and want to load the model **once**.
- You need to control where weights are cached (`cache_dir`) or force CPU/GPU.
- You must run **offline** in a locked-down or air-gapped environment.
- You are choosing between a registry key, a full HF id, or a local directory.

For *which* model to load, see `choosing-openmed-models`. To actually run it, see
`extracting-clinical-entities`.

## Install

```bash
pip install "openmed[hf]"   # adds Hugging Face transformers + hub download
```

## The three ways to name a model

`analyze_text`, `extract_pii`, `load_model`, and `ModelLoader.load_model` all
accept the same `model_name` in three forms:

| Form | Example | Notes |
| --- | --- | --- |
| Registry key | `"disease_detection_superclinical"` | Short, resolved via the bundled registry. |
| Full HF id | `"OpenMed/OpenMed-NER-DiseaseDetect-BigMed-278M"` | Anything `org/name`; downloaded from the Hub. |
| Local path | `"/models/my-openmed-ner"` | An existing directory; loaded with `local_files_only=True`. |

A bare name without `/` is prefixed with the default org (`OpenMed`). An existing
local path is detected automatically and never hits the network.

## Quick start: load and reuse a loader

The single most important pattern — build one `ModelLoader`, pass it everywhere.
The loader caches models, tokenizers, and pipelines in memory, so the second call
is instant.

```python
import openmed
from openmed import ModelLoader, OpenMedConfig

# One loader, reused across calls. Weights load on the first call only.
loader = ModelLoader()

notes = [
    "Patient prescribed 500 mg metformin for type 2 diabetes.",
    "History of myocardial infarction; started on atorvastatin.",
]

for note in notes:
    result = openmed.analyze_text(
        note,
        model_name="disease_detection_superclinical",
        loader=loader,          # <-- reuse; no reload on subsequent calls
        output_format="dict",
    )
    print(result.entities)
```

Without `loader=`, each `analyze_text` call constructs a fresh `ModelLoader`. The
underlying Hugging Face cache still prevents re-downloads, but you pay to
re-instantiate the pipeline — avoid that in loops and services.

## Load weights directly

When you want the raw model/tokenizer (e.g. to inspect config or build a custom
pipeline):

```python
from openmed import load_model

bundle = load_model("disease_detection_superclinical")
model     = bundle["model"]
tokenizer = bundle["tokenizer"]
config    = bundle["config"]
```

`load_model(model_name, config=None, **kwargs)` is a thin convenience wrapper that
builds a `ModelLoader` and calls `loader.load_model(...)`. For reuse, prefer
constructing the loader yourself:

```python
loader = ModelLoader()
bundle = loader.load_model("disease_detection_superclinical")
# Second call returns the cached bundle (no reload):
bundle2 = loader.load_model("disease_detection_superclinical")
# Force a fresh load if you replaced files on disk:
fresh = loader.load_model("disease_detection_superclinical", force_reload=True)
```

## Configure the cache, device, and org

`OpenMedConfig` is a dataclass. Pass it to `ModelLoader(config=...)`.

```python
from openmed import ModelLoader, OpenMedConfig

config = OpenMedConfig(
    cache_dir="/data/openmed-cache",   # default: ~/.cache/openmed
    device="cpu",                       # None = auto-detect
    default_org="OpenMed",              # prepended to bare model names
    hf_token=None,                      # or set env HF_TOKEN for private repos
)
loader = ModelLoader(config)
```

Relevant `OpenMedConfig` fields: `cache_dir`, `device`, `default_org`, `hf_token`,
`timeout` (default 300s), `backend` (`None` auto / `"hf"` / `"mlx"`), `log_level`.
`hf_token` falls back to the `HF_TOKEN` environment variable.

## First-run download, then fully offline

1. **First run (online):** the model is fetched from the Hub into `cache_dir`.
2. **Every run after:** transformers serves from cache with no network call.

To *guarantee* no network access (air-gapped, CI, PHI environments), set the
standard Hugging Face offline switch before importing:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Or vendor the model and pass a **local path** — that path is loaded with
`local_files_only=True` and never contacts the Hub:

```python
result = openmed.analyze_text(note, model_name="/models/openmed-disease-ner")
```

To pre-warm a cache for offline use, run one inference (or `load_model`) once with
network access, then disable it.

## Check a model's maximum sequence length

Useful before chunking long documents:

```python
from openmed import get_model_max_length, ModelLoader

loader = ModelLoader()
max_len = get_model_max_length("disease_detection_superclinical", loader=loader)
print(max_len)   # e.g. 512 — None if it can't be inferred
```

`get_model_max_length(model_name, *, config=None, loader=None)` delegates to
`loader.get_max_sequence_length(model_name)`. Pass the same `loader` you use for
inference so the tokenizer is loaded only once.

## Free memory when done

The loader holds models in RAM until released:

```python
loader.unload_model("disease_detection_superclinical")  # drop one model
loader.unload_all_models()                               # drop everything
loader.loaded_models()                                   # inspect what's cached
```

## Hand-off to / from OpenMed

- **From `choosing-openmed-models`:** that skill yields a model key or HF id; feed
  it straight into `ModelLoader.load_model(...)` or as `model_name=`.
- **To `extracting-clinical-entities`:** pass your reused `loader=` into
  `openmed.analyze_text(...)` so a long batch loads weights exactly once.
- **To de-identification:** `openmed.extract_pii(..., loader=loader)` and
  `openmed.deidentify(..., loader=loader)` accept the same loader — share one
  loader across NER and PHI steps in a pipeline.

```python
loader = ModelLoader(OpenMedConfig(cache_dir="/data/openmed-cache"))
phi   = openmed.deidentify(note, method="mask", loader=loader)
ner   = openmed.analyze_text(phi.deidentified_text, loader=loader)
```

## Edge cases & gotchas

- **`pip install openmed` alone is not enough to download models** — add the
  `[hf]` extra (or have `transformers` + `huggingface_hub` installed). `ModelLoader`
  raises `ImportError` with an install hint if transformers is missing.
- **Local path vs registry key collision:** if a bare name happens to exist as a
  directory, the local path wins. Use an absolute path to be explicit.
- **`force_reload=True`** is required after you overwrite files in a local model
  directory; otherwise the in-memory cache is served.
- **Private repos** need `hf_token` (or `HF_TOKEN`) and `HF_HUB_OFFLINE` unset for
  the first download.
- **No PHI in the cache path or logs.** Cache *model weights*, never patient text.
  `cache_dir` should not live inside a PHI data directory.
- **Permissive licensing only.** OpenMed models are Apache-2.0. Do not stage
  UMLS/SNOMED/CPT/MIMIC/i2b2/n2c2 assets in the cache — those stay out-of-process
  under the user's own license.

## Standards & references

- Hugging Face Hub caching & offline mode:
  https://huggingface.co/docs/huggingface_hub/guides/manage-cache and
  https://huggingface.co/docs/transformers/installation#offline-mode
- OpenMed model org on the Hub: https://huggingface.co/OpenMed
