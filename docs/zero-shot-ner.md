# Zero-shot NER Toolkit

The zero-shot NER toolkit extends OpenMed with GLiNER-powered models, domain-aware
label defaults, and inference helpers that complement the existing
token-classification stack.

## Installation

Install the GLiNER optional dependencies:

```bash
uv pip install ".[gliner]"
```

If you plan to run smoke tests or inference, ensure the GLiNER checkpoints are
available locally (by default the toolkit looks for `models/index.json`).

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
```

The defaults are packaged in `openmed/zero_shot/data/label_maps/defaults.json` and can be overridden in
tests or deployments by supplying a custom path to the high-level APIs.

## Inference API

Programmatic usage:

```python
from openmed.ner import NerRequest, infer

req = NerRequest(
    model_id="gliner-biomed-tiny",
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
3. Domain inferred from the index entry
4. Generic fallback label set

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
