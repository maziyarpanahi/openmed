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
The default index registers the six Fastino checkpoints below, so `infer()` routes
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

### OpenMed disease checkpoint

`OpenMed/GLiNER2.5-BioNER-Disease-Base-v1` is also registered for English
biomedical and clinical NER, with Apache-2.0 metadata and the single default
label `disease`. It uses the GLiNER2.5 boundary architecture. Install
`openmed[gliner2]`; no Transformers token-classification pipeline or remote
Python code is needed for this checkpoint.

```python
from openmed import analyze_text

result = analyze_text(
    "Synthetic note: the patient has asthma and chronic kidney disease.",
    model_id="OpenMed/GLiNER2.5-BioNER-Disease-Base-v1",
    revision="fccb29669b847cd6ca6384660d06c750c55366e2",
    threshold=0.7,
)
```

The label defaults to `["disease"]`; `0.7` is an explicit threshold in this
example, not a change to the SDK-wide default. The published checkpoint is
fine-tuned for disease spans. Its base model's other tasks are not qualified
by this registration. See the
[model card](https://huggingface.co/OpenMed/GLiNER2.5-BioNER-Disease-Base-v1)
for its evaluation scope and limitations.

After downloading, `local_files_only=True` and the pinned revision load from
the configured cache. A saved `extractor` / `BoundaryExtractor` configuration
also selects GLiNER2.5 when a local directory has an arbitrary name. Hub cache
snapshots and named exports retain the disease label default; supply
`labels=["disease"]` or an explicit index record when renaming an export
without its model identity. Weights remain outside the SDK package and the
model registry's promotion pointers are unchanged.

### Generic task helpers

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
| `language` | Which language is this? | every pack and fixture language (58) |
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
(514 samples, CPU), remeasured on 2026-10-30 with Torch 2.13.0, GLiNER2 2.1.0
and Transformers 5.12.1, seed 0 and four CPU threads. The checkpoint was cached
and every socket was blocked. The dataset digest is
`sha256:af23c29f96904705fc1e22f506703e96efff33668d5705c882ea27e15f4c95b5`.

| Kind | n | Router accuracy | Evidence-backed router: coverage / accuracy | Model top-1 | Model coverage / accuracy | Model fallback: coverage / silent errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `language` | 151 | 43.0% | 27.2% / 100.0% | 22.5% | 8.0% / 66.7% | 34.4% / 2.0% |
| `language_in_script` | 129 | 34.1% | 16.3% / 100.0% | 41.9% | 36.4% / 72.3% | 48.1% / 8.5% |
| `script` | 85 | 100.0% | 100.0% / 100.0% | 11.8% | 1.2% / 0.0% | 100.0% / 0.0% |
| `dialect` | 17 | 58.8% | 0.0% / n/a | 70.6% | 70.6% / 91.7% | 70.6% / 5.9% |
| `code_switch` | 132 | 90.9% | 16.7% / 100.0% | 69.7% | 65.1% / 75.6% | 69.7% / 15.9% |

On these synthetic fixtures, every evidence-backed router answer was correct,
and deterministic script detection matched all 85 script questions. The
coverage figures above bound that result; it is not a guarantee for new text.
On the 129 script-conditioned language questions, model top-1 was 54/129; its
confidence-gated fallback accepted 62 answers, including 11 wrong answers.

The model abstained on 139 of 151 open-set language questions with 58 options.
It identified declared code-mixing on 3 of 28 mixed notes and gave a confident
wrong answer on its only accepted script question. On dialects it matched
12 of 17 labels against the default locale's 10 of 17, with one confident error.
Those 17 synthetic locale tags follow identifier locales and provide too little
evidence for a routing change. These results measure routing decisions only;
they do not measure extraction or de-identification accuracy or establish
clinical qualification. Routing and activation defaults remain unchanged.

Per-language and low-resource counts are in
[`benchmarks/gliner2-language-decisions.json`](benchmarks/gliner2-language-decisions.json).
The report holds counts, rates, codes and a dataset digest, never fixture text.
Sinhala fixture coverage is included; it does not add a bundled Sinhala model.

After caching the pinned checkpoint, reproduce the summary with
`openmed[gliner2]` installed:

```bash
python scripts/benchmarks/gliner2_language_decisions.py \
  --revision e173bca1f0c4217d7a55b3b0c705d5ece8a0218d \
  --local-files-only \
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
