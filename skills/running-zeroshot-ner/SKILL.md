---
name: running-zeroshot-ner
description: "Extract arbitrary, custom entity types from clinical or biomedical text with no fine-tuning using OpenMed's GLiNER / GLiNER2 zero-shot support. Use when the user wants to define their own labels on the fly (e.g. Drug, Symptom, Device, Procedure), has no labelled data or a label set not covered by a fine-tuned model, or asks about openmed zero deps/index/infer, the gliner extra, or GLiNER. Pairs adjacent to extracting-clinical-entities (use that for high-accuracy fixed-schema NER) and loading-openmed-models."
license: Apache-2.0
metadata:
  project: OpenMed
  category: openmed-core
  pairs: adjacent
  version: "1.0"
---

# Running Zero-Shot NER

Zero-shot NER lets you extract entity types you **name at inference time** — no
training, no labelled data. OpenMed wraps GLiNER (v1) and GLiNER2 behind a small
index + inference layer, exposed via the `openmed zero` CLI and the `openmed.ner`
Python API. It runs on-device.

## When to use

- Your label set is **custom or evolving** ("Device", "Implant", "Allergen") and no
  fine-tuned OpenMed model emits exactly those labels.
- You have **no labelled data** to fine-tune with.
- You need a quick prototype or a one-off extraction over an unusual schema.

**When to prefer a fine-tuned model instead** (`extracting-clinical-entities`):
for a fixed, well-supported schema (diseases, drugs, anatomy), a fine-tuned
OpenMed model is more accurate and faster than zero-shot. Zero-shot trades some
accuracy for total label flexibility — use it for coverage of new types, then
graduate to a fine-tuned model once the schema stabilises.

## Install

```bash
pip install "openmed[gliner]"   # pulls GLiNER (and GLiNER2 if a recent gliner is installed)
openmed zero deps               # diagnostic: prints "GLiNER v1: ok" / "GLiNER v2: ok"
```

`openmed zero deps` only **checks** availability — it does not install anything.

## The two-step workflow: index, then infer

GLiNER checkpoints live as local model directories. OpenMed resolves them by a
short `model_id` via an `index.json`, so you build the index once and run inference
many times.

1. **`openmed zero index <models_dir>`** — scan a directory of downloaded GLiNER /
   GLiNER2 checkpoints and write `index.json` (model ids, family, domains, paths).
2. **`openmed zero infer "<text>" --model-id <id>`** — run extraction against a
   model from the index, with labels you supply.

```bash
# 1) Build the index over your local models (writes <models_dir>/index.json)
openmed zero index /models/gliner --output /models/gliner/index.json

# 2) Run zero-shot NER with your OWN labels (comma-separated)
openmed zero infer "Patient on insulin glargine via an insulin pump for type 1 diabetes." \
  --model-id gliner-biomedical \
  --labels "Drug,Device,Disease" \
  --threshold 0.5 \
  --index-path /models/gliner/index.json
```

Output is JSON: each entity has `text`, `start`, `end`, `label`, and `score`.

CLI flags:

- `zero infer`: positional `text`; `--model-id/-m` (required, an id from the
  index), `--labels/-l` (comma-separated custom labels), `--domain/-d` (label
  preset hint), `--threshold/-c` (default `0.5`), `--index-path/-i`.
- `zero index`: positional `models_dir`; `--output/-o`, `--pretty/--compact`.

If you omit `--labels`, OpenMed falls back to the `--domain` defaults (or generic
defaults). Passing explicit `--labels` is what makes it truly zero-shot.

## Python API

The same flow in code via `openmed.ner`:

```python
from openmed.ner import infer, NerRequest

request = NerRequest(
    model_id="gliner-biomedical",          # id from your index.json
    text="Started on insulin glargine via an insulin pump for type 1 diabetes.",
    labels=["Drug", "Device", "Disease"],  # your custom labels — no fine-tuning
    threshold=0.5,
)
response = infer(request, index_path="/models/gliner/index.json")

for ent in response.entities:
    print(f"{ent.label:8} {ent.text!r:30} {ent.score:.2f} [{ent.start}:{ent.end}]")
```

`NerRequest` fields: `model_id`, `text`, `labels` (None ⇒ domain/default labels),
`domain`, `threshold`. `infer(...)` returns a `NerResponse` whose `.entities` are
`Entity` objects with `.text`, `.start`, `.end`, `.label`, `.score`.

Build / load the index from Python too:

```python
from openmed.ner import build_index, write_index, load_index, is_gliner_available

if is_gliner_available():
    index = build_index("/models/gliner")
    write_index(index, "/models/gliner/index.json")
    index = load_index("/models/gliner/index.json")
```

Helpful label utilities:

```python
from openmed.ner import get_default_labels, available_domains
available_domains()              # domains with built-in label presets
get_default_labels("clinical")   # default labels for a domain hint
```

## Writing good labels

Zero-shot quality hinges on label phrasing. Prefer natural, specific noun phrases:

- Good: `["Drug", "Medical Device", "Disease", "Symptom", "Procedure"]`
- Weak: `["X", "thing", "misc"]`

Tune `threshold` to trade recall for precision. Start at `0.5` and raise it if you
see spurious spans.

## Hand-off to / from OpenMed

- **From `loading-openmed-models`:** zero-shot uses local GLiNER checkpoints rather
  than the OpenMed registry; download them once, then point `zero index` at the
  directory.
- **To `extracting-clinical-entities`:** once your label schema stabilises and a
  fine-tuned OpenMed model covers it, switch to `openmed.analyze_text` for higher
  accuracy and speed. The output shape (label + offsets + score) is parallel, so
  downstream code changes little.
- **To de-identification:** run `openmed.deidentify` **before** zero-shot NER in a
  PHI workflow, then extract entities from the redacted text.

## Edge cases & gotchas

- **`zero infer` needs an index.** Run `zero index <models_dir>` first, or pass a
  valid `--index-path`; the `--model-id` must exist in that index.
- **`zero deps` doesn't install.** It reports status only — install with
  `pip install "openmed[gliner]"`.
- **GLiNER2 needs a recent `gliner`** (≈0.3.0+) and a GLiNER2/Fastino checkpoint;
  `openmed zero deps` shows whether v2 is available.
- **Accuracy vs. flexibility.** Zero-shot is for coverage of new/custom types, not
  for squeezing out maximum F1 on a standard schema.
- **Permissive licensing & local-first.** Use permissively licensed GLiNER
  checkpoints; keep everything on-device and out of PHI logs.

## Standards & references

- GLiNER (zero-shot NER): https://github.com/urchade/GLiNER
- GLiNER paper: https://arxiv.org/abs/2311.08526
- OpenMed model org: https://huggingface.co/OpenMed
