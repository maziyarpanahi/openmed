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

Loading follows the same path as the other native families:
`load_gliner2_handle` resolves a Hub id to one local snapshot and loads it with
`AutoExtractor`, and `local_files_only=True` uses the cache only. `infer()` and
`analyze_text` apply `OPENMED_OFFLINE` before loading, and
`GLiNER2DecisionBackend.from_pretrained` follows it when `local_files_only` is
not given. The typed helpers run with the network blocked in offline mode, and
the `gliner2` cloud API client is refused. The handle records the model id, the
resolved revision (the snapshot commit hash) and the license, and nothing in
the family logs or caches request text.

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

The native scorer maps the five 1.0 scoring modes described above. System One
1.1 per-request instructions, option descriptions and ordered scalar levels
that this scorer cannot represent return `unsupported` before inference. The
backend constructor's `task` remains the model-facing question prompt.

### Multilingual language-decision evaluation

`openmed.eval.language_decisions` measures whether
`fastino/GLiNER2.5-multi-Decide` decisions could supply a fallback or
abstention signal for language routing. It is an evaluation only: routing
defaults and language-pack activation do not change.

The evaluation set is built from the synthetic language-pack fixtures
(`openmed/eval/golden/fixtures/i18n/`, the regional checksum fixtures and the
code-mixed Hinglish fixture). Gold answers come from fixture declarations only:
the record language and locale, annotated script runs, declared code-mixing and
the script a pack declares. Each sample is asked one question:

| Kind | Question | Options |
| --- | --- | --- |
| `language` | Which language is this? | every pack and fixture language (55) |
| `language_in_script` | Same question | languages natively written in the scripts the text contains |
| `script` | Which script is this? | scripts that packs declare (19) |
| `dialect` | Which regional variety? | the pack's default locale plus fixture locales |
| `code_switch` | Does this mix languages? | yes / no |

The baseline is the stdlib `LanguageRouter` (optional CLD2 disabled),
`detect_script` for scripts, and the pack's default locale for dialects. A
router answer counts as evidence-backed when script, marker, context or cue
evidence decided it rather than the pack-priority fallback; script answers
always do, and default-locale dialect answers never do. The model side runs
through `decide()` with the default calibration (confidence 0.70, margin 0.15).
Five policies are compared: always trust the router, trust only
evidence-backed router answers, trust the model, accept the router answer only
when the model confidently agrees, and fall back to the model's confident
answer when the router has no evidence. A silent error is an accepted wrong
answer.

Results for the pinned revision `e173bca1f0c4217d7a55b3b0c705d5ece8a0218d`
(462 samples, CPU):

| Kind | n | Router accuracy | Evidence-backed router: coverage / accuracy | Model top-1 | Model coverage / accuracy | Model fallback: coverage / silent errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `language` | 133 | 42.9% | 25.6% / 100% | 23.3% | 9.8% / 76.9% | 34.6% / 1.5% |
| `language_in_script` | 114 | 34.2% | 14.9% / 100% | 45.6% | 34.2% / 84.6% | 43.9% / 5.3% |
| `script` | 81 | 100% | 100% / 100% | 9.9% | 1.2% / 0% | 100% / 0% |
| `dialect` | 17 | 58.8% | 0% / n/a | 70.6% | 70.6% / 91.7% | 70.6% / 5.9% |
| `code_switch` | 117 | 91.5% | 15.4% / 100% | 68.4% | 64.1% / 76.0% | 67.5% / 15.4% |

Evidence-backed router answers were never wrong, and Unicode script detection
was exact, so the deterministic path already has a zero-silent-error abstention
signal; its limit is coverage. The model helps only when its options are
limited to the text's script candidates, where it recovers Latin- and
Cyrillic-script languages the router sends to the pack-priority fallback, at
the cost of confident errors between close languages. It is unusable for
script detection, collapses toward abstention over 55 open-set options, and
answers "no" for most declared code-mixed notes (3 of 25 correct). On dialects
it beats the default locale (12 of 17 against 10 of 17), but the set has 17
notes whose locale tags follow the identifier locale and includes one confident
error, which is too little evidence for a routing change. Per-language and
low-resource counts are in
[`benchmarks/gliner2-language-decisions.json`](benchmarks/gliner2-language-decisions.json);
the report holds counts, rates, codes and a dataset digest, never fixture text.

Reproduce the summary with `openmed[gliner2]` installed:

```bash
python scripts/benchmarks/gliner2_language_decisions.py \
  --output docs/benchmarks/gliner2-language-decisions.json
```

`--check docs/benchmarks/gliner2-language-decisions.json` compares a fresh run
with the committed summary instead; router counts must match exactly and model
counts may drift by two samples. The opt-in integration test
`tests/integration/test_gliner2_language_decisions.py` runs the same check
with every socket blocked after the first download
(`OPENMED_RUN_GLINER2_CHECKPOINT_TESTS=1`).

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
