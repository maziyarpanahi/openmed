# Analyze Text Helper

`openmed.analyze_text` is the top-level orchestrator that most users start with. It validates input, spins up a
token-classification pipeline, segments sentences with pySBD, and normalizes the output so you can copy dict/JSON/HTML/CSV
payloads straight into downstream systems.

## Quick reference

```python
from openmed import analyze_text

result = analyze_text(
    text="Patient started on imatinib for chronic myeloid leukemia.",
    model_name="disease_detection_superclinical",
    aggregation_strategy="simple",
    output_format="dict",
    include_confidence=True,
    confidence_threshold=0.55,
    group_entities=False,
    metadata={"source": "clinic-note-42"},
)
print(result.model)
print(result.entities[:3])
payload = result.to_dict()
```

### Key arguments

- `model_name`: registry alias, full Hugging Face id, or local model directory. Use `openmed.list_models()` if you need
  auto-discovery.
- `model_id`: alias for `model_name`, supported for API-style callers that use model-id terminology.
- `aggregation_strategy`: forwarded to the HF pipeline. `simple` (default) yields grouped tokens; `None` keeps raw tokens.
- `output_format`: `"dict"` (default, returns `AnalyzeResult`), `"json"`, `"html"`, or `"csv"`.
- `include_confidence` & `confidence_threshold`: control the final payload; defaults keep all scores.
- `group_entities`: merge nearby spans of the same label after formatting.
  Leave this `False` when `aggregation_strategy="simple"` should preserve the
  model's separate BIO entities, especially for diagnosis or medication lists.
- `formatter_kwargs`: forwarded to `openmed.processing.format_predictions`.
- Sentence options (`sentence_detection`, `sentence_language`, `sentence_clean`, `sentence_segmenter`) wrap pySBD so each
  prediction carries the sentence span; disable them if latency matters more than helper metadata.

### Preserve diagnosis-list boundaries

`aggregation_strategy="simple"` already combines the subword tokens belonging
to each BIO entity. Enabling `group_entities` adds a second, label-only merge;
it can join distinct same-label items separated by a small gap, including
comma-separated diagnoses.

Medical-token remapping is useful when a model splits clinical terms poorly,
but it can also coalesce adjacent same-label spans. Disable both optional merge
layers when the model's original BIO boundaries are the desired output:

```python
from openmed import OpenMedConfig, analyze_text

text = (
    "Diagnoses: hypertension, diabetic peripheral neuropathy, "
    "urinary incontinence"
)
result = analyze_text(
    text,
    model_name="OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M-mlx",
    config=OpenMedConfig(
        backend="mlx",
        use_medical_tokenizer=False,
    ),
    aggregation_strategy="simple",
    confidence_threshold=0.7,
    group_entities=False,
)

print([entity.text for entity in result.entities])
# ['hypertension', 'diabetic peripheral neuropathy', 'urinary incontinence']
```

When the source is a structured problem list, retain its item boundaries and
analyze each item separately or preserve one item per line. Do not split an
unpunctuated clinical phrase on whitespace: valid diagnoses such as
`diabetic peripheral neuropathy` and `urinary incontinence` are multiword
entities. If the original list structure is unavailable, use a caller-supplied
terminology linker or grounder to validate any attempted split.

## Chunking & truncation

```python
result = analyze_text(
    text=long_report,
    model_name="pharma_detection_superclinical",
    aggregation_strategy=None,  # work with raw tokens
    max_length=512,             # forwarded to HF pipeline
    truncation=True,            # enforce length (default)
    sentence_detection=False,   # skip pySBD to save ~2ms per note
)
```

When you need full-control over tokenizer behaviour:

- Pass `max_length`/`truncation` via `pipeline_kwargs`. If you skip truncation, the helper sets the tokenizer max length to
  unlimited (0) so HF pipelines accept longer inputs.
- Provide `batch_size` or `num_workers` in `pipeline_kwargs` and they will be forwarded to the pipeline call but not to the
  constructor.
- Enable medical token remapping with `OpenMedConfig(use_medical_tokenizer=True)` to group outputs onto clinical-friendly
  tokens without changing the model tokenizer.

## Loading from a local path

Pass an existing model directory to `model_name` or `model_id` when the model files are already present on disk:

```python
import os
from openmed import OpenMedConfig, analyze_text

local_path = os.path.abspath("./models/OpenMed-NER-DiseaseDetect-SuperClinical-434M")
config = OpenMedConfig(device="cpu")

result = analyze_text(
    "Patient presents with chronic myeloid leukemia and Type 2 diabetes.",
    model_id=local_path,
    config=config,
)

for entity in result.entities:
    print(entity.text, entity.label)

legacy_payload = result.to_dict()
print(legacy_payload["model_name"])
```

When the identifier points to an existing local path, OpenMed asks Transformers to load with `local_files_only=True` by
default. That keeps air-gapped deployments from validating or downloading the model from the Hugging Face Hub. If any
required tokenizer, config, or weight file is missing, loading fails locally with the underlying Transformers error.

## Streaming multiple texts

`analyze_text` is optimized for single inputs. For batch jobs, keep a `ModelLoader` instance around and reuse its
pipelines:

```python
from openmed import ModelLoader, format_predictions

loader = ModelLoader()
pipeline = loader.create_pipeline("disease_detection_superclinical")

for note in notes:
    raw = pipeline(note, batch_size=16)
    formatted = format_predictions(raw, note, model_name="Disease Detection")
    print(formatted.entities[:3])
```

See [ModelLoader & Pipelines](./model-loader.md) for details on caching, GPU selection, and tokenizer reuse.

## HTML/CSV rendering

```python
html = analyze_text(
    text,
    model_name="oncology_detection_superclinical",
    output_format="html",
    formatter_kwargs={
        "html_class": "openmed-highlights",
        "tag_colors": {"CANCER": "#d97706"},
    },
)

csv_rows = analyze_text(
    text,
    model_name="pharma_detection_superclinical",
    output_format="csv",
)
```

The HTML formatter emits a ready-to-embed snippet for dashboards; CSV mode writes row strings (header + body). Both respect
`confidence_threshold` and `group_entities`.

## Validation behaviours

Behind the scenes `analyze_text` calls:

- `validate_input` — trims whitespace and enforces max lengths.
- `validate_model_name` — normalizes registry aliases.
- Sentence detection (`openmed.processing.sentences`) — optional pySBD segmentation with language hints.
- `OutputFormatter` — see [Advanced NER & Output Formatting](./output-formatting.md) for available kwargs.

If you need custom validation or logging, inject your own `OpenMedConfig` or reuse a configured `ModelLoader`.
