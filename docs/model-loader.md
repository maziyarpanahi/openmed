# ModelLoader & Pipelines

`openmed.core.models.ModelLoader` is the backbone for all runtime integration. It centralizes Hugging Face discovery,
credential management, caching, and pipeline instantiation so you can move between quick experiments and production
runners without rewriting glue code.

## When to use it

- You want to reuse a single tokenizer/pipeline across many documents.
- You need to load multiple models (e.g., disease + pharma) side-by-side.
- You are deploying a service/CLI and prefer to hydrate everything at startup.
- You require maximum control over device placement, dtype, batch size, or tokenizer configuration.

## Essentials

```python
from openmed.core import ModelLoader, OpenMedConfig

config = OpenMedConfig(
    device="cuda",
    cache_dir="~/.cache/openmed",
    hf_token="hf_api_token_if_needed",
)
loader = ModelLoader(config=config)

pipeline = loader.create_pipeline(
    "disease_detection_superclinical",
    task="token-classification",
    aggregation_strategy="simple",
    use_fast_tokenizer=True,
)

raw = pipeline("Administered paclitaxel alongside trastuzumab.")
```

- `create_pipeline` accepts any kwargs supported by `transformers.pipeline`.
- Tokens are cached per model/config combination; repeated calls reuse the same HF objects.
- `ModelLoader.get_max_sequence_length(model_name)` infers tokenizer limits when you need manual truncation logic.

## Discovery helpers

```python
from openmed.core import ModelLoader

loader = ModelLoader()
print(loader.list_available_models(include_registry=True, include_remote=False))
print(loader.list_available_models(include_registry=True, include_remote=True)[:5])
```

These functions power `openmed.list_models()` and the CLI equivalents. Use them to present dropdowns in UIs or to
pre-flight deployments before running inference.

## Device & caching strategy

- **CPU-only** deployments: leave `device="cpu"` and skip installing GPU runtimes. Transformers will use `torch` CPU wheels.
- **GPU** deployments: set `device="cuda"` or `cuda:1`, and optionally configure `torch_dtype="auto"` via
  `OpenMedConfig.pipeline`.
- **Air-gapped** or repeated builds: point `cache_dir` at a persistent volume; ModelLoader instructs HF to reuse it.
- Provide `hf_token` when you consume gated models, or rely on `HfFolder` credentials.

## Token helpers

If you need raw token alignment, use the tokenization utilities that ship alongside the loader:

```python
from openmed.processing import TokenizationHelper

model_data = loader.load_model("anatomy_detection_electramed")
token_helper = TokenizationHelper(model_data["tokenizer"])
encoding = token_helper.tokenize_with_alignment("BP 120/80. Start metformin 500mg bid.")
print(encoding["tokens"][:10])
```

`load_model` lets you access the underlying HF `AutoModel` + `AutoTokenizer` for workflows that outgrow pipelines.

## Sentence detection reuse

`ModelLoader` does not run pySBD itself, but it exposes hooks so `analyze_text` can pass in the tokenizer/pipeline it
creates. If you batch many texts with custom segmentation, build the segmenter once and reuse it:

```python
from openmed.processing import sentences

segmenter = sentences.create_segmenter(language="en", clean=True)
for doc in docs:
    segments = sentences.segment_text(doc, segmenter=segmenter)
```

## Troubleshooting checklist

- **“Tokenizer length mismatch”**: call `loader.get_max_sequence_length(model_name)` and set `max_length` explicitly.
- **“Model not found”**: confirm it exists in `openmed.core.model_registry.OPENMED_MODELS` or pass a full HF path and set
  `include_remote=True` if you rely on discovery.
- **Slow cold-starts**: prefetch pipelines at startup and mount the cache dir on SSD/NVMe storage.
