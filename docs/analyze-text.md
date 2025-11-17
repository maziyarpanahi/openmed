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
    output_format="json",
    include_confidence=True,
    confidence_threshold=0.55,
    group_entities=True,
    metadata={"source": "clinic-note-42"},
)
print(result[:200])  # JSON string in this configuration
```

### Key arguments

- `model_name`: registry alias or full Hugging Face id. Use `openmed.list_models()` if you need auto-discovery.
- `aggregation_strategy`: forwarded to the HF pipeline. `simple` (default) yields grouped tokens; `None` keeps raw tokens.
- `output_format`: `"dict"` (default, returns `PredictionResult`), `"json"`, `"html"`, or `"csv"`.
- `include_confidence` & `confidence_threshold`: control the final payload; defaults keep all scores.
- `group_entities`: merge adjacent spans of the same label after formatting.
- `formatter_kwargs`: forwarded to `openmed.processing.format_predictions`.
- Sentence options (`sentence_detection`, `sentence_language`, `sentence_clean`, `sentence_segmenter`) wrap pySBD so each
  prediction carries the sentence span; disable them if latency matters more than helper metadata.

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
