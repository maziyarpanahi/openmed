# Zero-shot NER Toolkit

The zero-shot NER toolkit extends OpenMed with GLiNER-powered models, domain-aware
label defaults, and inference helpers that complement the existing
token-classification stack.

For a short task-oriented walkthrough, see
[Zero-shot NER How-to](./zero-shot-howto.md).

## Installation

Install the GLiNER optional dependencies:

```bash
uv pip install ".[gliner]"
```

GLiNER2.5 checkpoints use a separate extra, because they ship in the
Apache-2.0 `gliner2` package rather than in `gliner`:

```bash
uv pip install ".[gliner2]"
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

## GLiNER2.5 checkpoints

The `gliner2` family loads GLiNER2.5 checkpoints through
`gliner2.AutoExtractor`, which reads the architecture saved with each
checkpoint (`span` or `boundary`). The legacy `GLiNER2.from_pretrained` loader
is span-only, and the v1 `gliner` package cannot load these checkpoints at all.
The default index registers all six public checkpoints, so `infer()` routes
them to this family:

| Checkpoint | Parameters | Languages | Architecture | Decision-tuned |
| --- | --- | --- | --- | --- |
| `fastino/gliner2.5-small-v1` | 74M | English | boundary | no |
| `fastino/gliner2.5-base-v1` | 194M | English | boundary | no |
| `fastino/gliner2.5-multi-v1` | 287M | multilingual | boundary | no |
| `fastino/GLiNER2.5-multi-Decide` | 287M | multilingual | boundary | yes |
| `fastino/GLiNER2.5-Decide` | 486M | English | span | yes |
| `fastino/GLiNER2.5-Decide-1B` | 1.19B | English | span | yes |

All six are Apache-2.0, need no remote code and run on CPU. The same metadata
is available from `openmed.ner.families.gliner2.list_gliner2_checkpoints()`.
Weights are not bundled.

```python
from openmed.ner.families.gliner2 import load_gliner2_handle

handle = load_gliner2_handle("fastino/gliner2.5-small-v1")
text = "Synthetic patient was started on metformin 500 mg for type 2 diabetes."

entities = handle.extract_entities(text, ["medication", "condition"])
note_type = handle.classify_text(
    text, {"note_type": ["prescription", "billing", "scheduling"]}
)["note_type"]
relations = handle.extract_relations(text, ["treated_with"])
records = handle.extract_json(text, {"prescription": ["drug::str", "dose::str"]})
```

- `extract_entities` returns OpenMed `Entity` objects whose `text` always equals
  `source[start:end]`. Texts with more word tokens than `chunk_size` (384 by
  default) go through `gliner2`'s long-document helper, and offsets are mapped
  back to the whole document.
- `classify_text` returns a score for every label: a softmax distribution for
  single-label tasks and independent sigmoid probabilities for
  `multi_label` tasks. Selection is derived from those scores.
- `extract_relations` returns head and tail spans with offsets, and
  `extract_json` returns schema-driven records whose extractive fields are
  spans and whose fixed-choice fields are label scores.

Loading is local-first. The handle resolves the checkpoint to a local snapshot
before loading it: a cached snapshot is reused without a network call, a
download happens only when the snapshot is missing and offline mode is off, and
`OPENMED_OFFLINE=1` or `local_files_only=True` turns a cache miss into an
`OfflineModeError`. Inference makes no network calls; the `gliner2` cloud API
client is refused. Checkpoints that request remote code (`auto_map`) or ship no
`model.safetensors` are refused. The handle records the model id, the resolved
revision (the snapshot commit hash) and the license, and nothing in the family
logs or caches request text.

### GLiNER2.5 decision backend

`GLiNER2DecisionBackend` plugs a GLiNER2.5 checkpoint into the
[fixed-option decision contract](./api/fixed-option-decisions.md). Each request
becomes one `classify_text` task, and the backend returns the probability of
every option so calibrated abstention can judge confidence and margin:

| Decision mode | GLiNER2 task |
| --- | --- |
| `fixed_choice`, `ordered_preference` | single-label task over the options (softmax scores) |
| `boolean_choice` | yes/no labels when the options read as yes/no, true/false or y/n; otherwise a two-label task |
| `multi_label` | `multi_label` task with a sigmoid score for every option; the calibration profile's threshold selects labels |
| `scalar_score` | probability of the affirmative yes/no label |

```python
from openmed.ner.families.gliner2_decision import GLiNER2DecisionBackend
from openmed.structured import DecisionRequest, decide

backend = GLiNER2DecisionBackend.from_pretrained(
    "fastino/GLiNER2.5-multi-Decide", task="request"
)
result = decide(
    DecisionRequest(
        mode="fixed_choice",
        input_text="¿Puedo pedir una cita con dermatología esta semana?",
        options=("book_appointment", "refill_prescription", "test_results"),
    ),
    backend=backend,
)
```

The backend identity records the model id, resolved revision, license and the
`gliner2` runtime, and construction fails when the license is unknown or not
permissive. The task name is part of the GLiNER2 prompt, so a short
question-like name can sharpen decisions. Results stay assistive review
artifacts: they always require human review, and clinical uses go through
`GuardedSuggestion`.

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
