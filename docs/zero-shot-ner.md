# Zero-shot NER Toolkit

The zero-shot NER toolkit extends OpenMed with GLiNER, GLiFormer, and GLiNER2 models, domain-aware
label defaults, and inference helpers that complement the existing
token-classification stack.

For a short task-oriented walkthrough, see
[Zero-shot NER How-to](./zero-shot-howto.md).

## Installation

Install the GLiNER optional dependencies:

```bash
uv pip install ".[gliner]"
uv pip install ".[gliformer]"  # GLiFormer 0.1.3+
uv pip install ".[gliner2]"    # Local GLiNER2 / GLiNER2.5, gliner2[local] 2.1+
```

Install only the extra for the model family you use. Dependencies are imported
when loading a model. GLiNER2 uses its local `AutoExtractor` API to select span
or boundary checkpoints; it never uses the hosted extraction API.

Recognized Hugging Face IDs and local checkpoint configurations work without
`models/index.json`. An optional index supplies aliases, domains, and per-model
label defaults. Ambiguous unindexed IDs require an explicit `family`.

## Model indexing (Python API)

Create or refresh the index by pointing the API at your model directory:

```python
from pathlib import Path

from openmed.ner import build_index, write_index

models_dir = Path("/path/to/models")
index = build_index(models_dir)
write_index(index, Path("models/index.json"), pretty=True)
```

- Generates `index.json` with metadata (`id`, `family`, `domains`, `languages`).
- Deduplicates model identifiers and records the generation timestamp.

## Domain defaults

Inspect the curated label maps and discover available domains:

```python
from openmed.ner import available_domains, get_default_labels

print(available_domains())
print(get_default_labels("biomedical"))
print(get_default_labels("endocrinology"))
```

The defaults are packaged in
`openmed/zero_shot/data/label_maps/defaults.json` and can be overridden in tests
or deployments by supplying a custom path to the high-level APIs.

## Registered GLiNER-BioMed backbone

The default index includes the public `Ihor/gliner-biomed-large-v1.0`
checkpoint as the stable `GLINER_BIOMED_MODEL_ID` entry. Its metadata declares
the `gliner` family, English support, `biomedical` and `clinical` domains, and
the Apache-2.0 license. The checkpoint weights are not bundled: install the
optional GLiNER dependency and let the configured local cache provide the
weights when inference runs.

When the request does not provide labels, `domain="biomedical"` selects the
packaged biomedical defaults (`Disease`, `Drug`, `Gene`, and `Organism`). The
clinical domain remains available through `domain="clinical"` with its own
default preset.

For the common biomedical path, use the convenience wrapper:

```python
from openmed.ner import infer_biomedical

response = infer_biomedical(
    "Aspirin was used for synthetic fever documentation.",
    threshold=0.55,
)
```

## Inference API

Programmatic usage:

```python
from openmed.ner import NerRequest, infer

req = NerRequest(
    model_id="Ihor/gliner-biomed-large-v1.0",
    text="Imatinib inhibits BCR-ABL in chronic myeloid leukaemia.",
    threshold=0.55,
    domain="biomedical",
)
resp = infer(req)
for entity in resp.entities:
    print(entity.label, entity.text, entity.score)
```

Label resolution precedence is:

1. Explicit `request.labels`
2. `request.domain` defaults
3. Model-specific `default_labels` (the disease checkpoint uses `disease`)
4. Domain inferred from the index entry
5. Generic fallback label set

## GLiFormer disease extraction (v3.2)

```python
from openmed import OpenMedConfig
from openmed.ner import GLIFORMER_DISEASE_MODEL_ID, NerRequest, infer

text = "A synthetic note mentions asthma and diabetes."
response = infer(
    NerRequest(model_id=GLIFORMER_DISEASE_MODEL_ID, text=text, threshold=0.7),
    config=OpenMedConfig(device="cpu"),
)
for entity in response.entities:
    assert text[entity.start:entity.end] == entity.text
    print(entity.label, entity.text, entity.score)
```

`GLIFORMER_DISEASE_MODEL_ID` is
`OpenMed/GLiFormer-BioNER-Disease-Base-v1`. Its built-in record declares the
`gliformer` family, English, Apache-2.0, and `default_labels=("disease",)`.
Weights remain separate from the SDK. Authentication and cache configuration
are honored for restricted checkpoints.

General checkpoints such as `fastino/gliner2.5-base-v1` accept explicit labels:

```python
response = infer(NerRequest(
    model_id="fastino/gliner2.5-base-v1",
    text="A synthetic note mentions asthma.",
    labels=["disease"],
    threshold=0.7,
))
```

For an ambiguous identifier, set `NerRequest(..., family="gliformer")` or
`family="gliner2"`. A local checkpoint directory is accepted as `model_id`,
with its family read from saved configuration. A record's `path` takes
precedence over its index alias when loading. `flat_ner=False` retains
overlapping spans. All occurrences retain the runtime's half-open character
offsets and confidence; missing offsets or scores are rejected instead of
reconstructed by substring search.

For the same models with sentence chunking, JSON/HTML/CSV rendering, and
assertion metadata, use [`analyze_text`](analyze-text.md). Local-only settings
restrict loading to cached assets and block networking during inference.

Run the existing Python example with a chosen native model:

```bash
OPENMED_EXAMPLE_ALLOW_DOWNLOAD=1 python examples/clinical_ner_families.py \
  --model-id OpenMed/GLiFormer-BioNER-Disease-Base-v1 --labels disease --threshold 0.7
```

## Token classification adapter

Transform span entities into token-level BIO/BILOU labels:

```python
from openmed.ner import to_token_classification

tokens = to_token_classification(resp.entities, req.text, scheme="BILOU")
print(tokens.labels())
```

The adapter honors entity groups, resolves overlapping spans with score-based
priority, and falls back to regex tokenization when no tokenizer is supplied.

## Smoke tests

Use the bundled script for a lightweight end-to-end validation:

```bash
python scripts/smoke_gliner.py --limit 2 --threshold 0.4 --adapter
```

The script loads GLiNER models from the index, applies sample domain-specific
texts, and optionally prints token-level labels. It aborts early when GLiNER
dependencies are missing.

## Testing

Run the unit tests (excluding slow smoke checks) with:

```bash
python3 -m pytest tests/unit/ner -m "not slow"
```

Include slow tests for full coverage:

```bash
python3 -m pytest -m slow
```

The suite uses mocks for GLiNER APIs, so no model downloads occur during unit
testing.
