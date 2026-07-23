---
name: extracting-clinical-entities
description: "Run clinical and biomedical named-entity recognition on medical text with OpenMed's analyze_text. Use when the user wants to extract diseases, drugs, anatomy, genes, or other biomedical entities from notes; needs NER output as dict/json/html/csv; wants to filter by confidence, group entities, toggle sentence detection, or save spans to JSONL; or wants the openmed analyze CLI. Pairs with loading-openmed-models and choosing-openmed-models, and runs after deidentifying-clinical-text in a privacy-first pipeline."
license: Apache-2.0
metadata:
  project: OpenMed
  category: openmed-core
  pairs: adjacent
  version: "1.0"
---

# Extracting Clinical Entities

`openmed.analyze_text` runs a token-classification model over medical text and
returns structured entities with character offsets and confidence scores. It runs
**on-device** after a one-time model download.

## When to use

- Pull diseases, medications, anatomy, genes, proteins, etc. out of clinical text.
- You need exact **character spans** (start/end) plus **confidence** per entity.
- You want output as objects, JSON, an HTML highlight view, or CSV.
- You are building the "extract entities" stage of a clinical NLP pipeline.

To choose a model, see `choosing-openmed-models`. To load it once and reuse it,
see `loading-openmed-models`. **In a PHI workflow, de-identify first** (see
`deidentifying-clinical-text`), then run NER on the redacted text.

## Install

```bash
pip install "openmed[hf]"
```

## Quick start

```python
import openmed

note = (
    "Patient prescribed 500 mg metformin for type 2 diabetes mellitus. "
    "Reports intermittent chest pain; ruled out myocardial infarction."
)

result = openmed.analyze_text(
    note,
    model_name="disease_detection_superclinical",  # registry key, HF id, or local path
    output_format="dict",                           # dict | json | html | csv
    confidence_threshold=0.5,
)

for ent in result.entities:
    print(f"{ent.label:12} {ent.text!r:40} {ent.confidence:.2f} [{ent.start}:{ent.end}]")
```

With `output_format="dict"` you get a `PredictionResult`. The fields you use most:

```text
result.text          # the original input text
result.entities      # list of entity objects
result.model_name    # which model produced these
ent.text             # the surface string
ent.label            # entity type, e.g. "DISEASE"
ent.confidence       # model score in [0, 1]   (NOTE: .confidence, not .score)
ent.start / ent.end  # character offsets into result.text
```

## Output formats

`analyze_text(...)` returns different types depending on `output_format`:

| `output_format` | Return type | Use for |
| --- | --- | --- |
| `"dict"` (default) | `PredictionResult` object | Programmatic access via `.entities`. |
| `"json"` | `str` (JSON) | Logging, APIs, writing to disk. |
| `"html"` | `str` (HTML) | A highlighted preview of the note. |
| `"csv"` | `str` (CSV) | Spreadsheet / quick review. |

```python
import openmed

note = "Started atorvastatin 40 mg; history of myocardial infarction."

json_str = openmed.analyze_text(note, output_format="json")
html_str = openmed.analyze_text(note, output_format="html")   # render in a browser
csv_str  = openmed.analyze_text(note, output_format="csv")
```

## Key parameters

```python
openmed.analyze_text(
    text,
    model_name="disease_detection_superclinical",
    output_format="dict",
    confidence_threshold=0.5,    # drop entities below this score; None keeps all
    aggregation_strategy="simple",  # HF subword aggregation; None for raw tokens
    group_entities=False,        # merge adjacent same-label spans into one
    include_confidence=True,     # include scores in formatted output
    sentence_detection=True,     # pySBD sentence splitting (better long-doc spans)
    sentence_language="en",
    loader=None,                 # pass a reused ModelLoader (see loading skill)
)
```

- **`confidence_threshold`** — the most useful knob. Use the model's
  `recommended_confidence` (from `get_model_info`) as a starting point.
- **`group_entities=True`** — merges `"type"`, `"2"`, `"diabetes"` fragments into a
  single `"type 2 diabetes"` span. Turn on for cleaner output.
- **`sentence_detection=True`** (default) — splits long notes into sentences before
  inference for more accurate offsets and to respect model max length. Requires
  pySBD; if unavailable it silently falls back to whole-text inference.

## Save results to JSONL

One line per note keeps offsets and labels for downstream grounding or eval:

```python
import json
import openmed

notes = [
    "Type 2 diabetes managed with metformin.",
    "Acute myocardial infarction; started aspirin and atorvastatin.",
]

with open("entities.jsonl", "w", encoding="utf-8") as fh:
    for i, note in enumerate(notes):
        result = openmed.analyze_text(note, output_format="dict")
        fh.write(json.dumps({
            "doc_id": i,
            "text": result.text,
            "model": result.model_name,
            "entities": [
                {"label": e.label, "text": e.text,
                 "start": e.start, "end": e.end,
                 "confidence": round(e.confidence, 4)}
                for e in result.entities
            ],
        }) + "\n")
```

Store **offsets and labels**, not extra copies of free text, in PHI contexts.

## CLI

```bash
openmed analyze --text "Type 2 diabetes managed with metformin." \
  --model disease_detection_superclinical \
  --format json \
  --threshold 0.5 \
  --group

# Or analyze a file:
openmed analyze --input-file note.txt --model disease_detection_superclinical -o csv
```

Flags: `--text/-t`, `--input-file/-f`, `--model/-m`, `--format/-o`
(`dict|json|html|csv`), `--threshold/-c`, `--group`, `--no-confidence`,
`--sentence-detection/--no-sentence-detection`.

## Hand-off to / from OpenMed

- **From `loading-openmed-models`:** pass your reused `loader=` so a batch loads
  weights once.
- **From `deidentifying-clinical-text`:** run NER on `result.deidentified_text`,
  not raw PHI:

  ```python
  deid = openmed.deidentify(raw_note, method="mask", policy="hipaa_safe_harbor")
  ner  = openmed.analyze_text(deid.deidentified_text, output_format="dict")
  ```

- **To terminology grounding (out-of-process):** map `ent.text`/`ent.label` to
  RxNorm / LOINC / SNOMED using the user's own licensed service — OpenMed does not
  bundle restricted terminologies.
- **To batch processing:** for large corpora use `openmed.process_batch(...)` /
  `BatchProcessor` (see processing utilities) with a shared loader.

## Edge cases & gotchas

- **Attribute is `.confidence`, not `.score`.** Entity objects extend
  `EntityPrediction` (`text`, `label`, `confidence`, `start`, `end`).
- **Right model for the labels.** A Disease model won't emit oncology staging or
  gene labels — pick the category in `choosing-openmed-models` and check
  `entity_types`.
- **Offsets index `result.text`.** Slice the original string with `start:end`; the
  surface form in `ent.text` is whitespace-trimmed.
- **Long documents:** keep `sentence_detection=True` so chunks respect the model's
  max length (`get_model_max_length`); disabling it can truncate long notes.
- **NER assists, it does not diagnose.** Treat output as decision support; surface
  a disclaimer for any clinical-facing use.
- **No raw PHI in logs.** Log labels, offsets, and hashes — never patient text.

## Standards & references

- Token classification / NER (Hugging Face):
  https://huggingface.co/docs/transformers/tasks/token_classification
- pySBD sentence boundary detection: https://github.com/nipunsadvilkar/pySBD
- OpenMed model cards: https://huggingface.co/OpenMed
