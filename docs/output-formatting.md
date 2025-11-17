# Advanced NER & Output Formatting

Post-processing matters as much as the base model. OpenMed codifies the heuristics from the public demos so you can go
from noisy token output to high-quality, copy-pasteable spans in a few lines.

## Advanced NER processor

`openmed.processing.advanced_ner.AdvancedNERProcessor` applies the same filtering stack used in the OpenMed Gradio app:

- Confidence filtering with a configurable threshold (`min_confidence`).
- Punctuation-only and short-span removal.
- Regex-based exclusions for common false positives.
- Optional edge stripping and gap-aware merging of adjacent entities.
- Smart BIO grouping fixes overlapping spans when `aggregation_strategy=None`.

```python
from openmed.processing.advanced_ner import create_advanced_processor

processor = create_advanced_processor(
    min_confidence=0.65,
    merge_adjacent=True,
    max_merge_gap=8,
)

raw = pipeline(text)  # HF token-classification output
entities = processor.process_pipeline_output(text, raw)

for span in entities:
    print(span.label, span.text, span.score)
```

Use it when you need deterministic filtering outside of `analyze_text` or when you operate on raw tokens.

## OutputFormatter & PredictionResult

`openmed.processing.OutputFormatter` normalizes predictions into dictionaries, JSON strings, HTML snippets, or CSV rows.
The dataclasses in `openmed/processing/outputs.py` ensure the payload stays type-safe and ready for logging.

```python
from openmed.processing import format_predictions

formatted = format_predictions(
    raw_predictions,
    original_text,
    model_name="Disease Detection",
    include_confidence=True,
    confidence_threshold=0.6,
    group_entities=True,
)

print(formatted.entities[0].to_dict())
print(formatted.to_dict())
```

### HTML output

```python
from openmed.processing.outputs import OutputFormatter

formatter = OutputFormatter(group_entities=True)
result = formatter.format_predictions(raw_predictions, text, model_name="Oncology")
html = formatter.to_html(result, tag_colors={"Cancer": "#f97316"})
```

The HTML helper wraps highlighted spans with semantic tags (`data-entity="Cancer"`) so your dashboards can apply custom
styles or tooltips.

### CSV output

```python
csv_lines = formatter.to_csv(result)
print("\n".join(csv_lines[:5]))
```

CSV export is handy when you need to feed BI tools or spreadsheets without additional ETL code.

## Sentence spans & metadata

- `analyze_text` attaches sentence spans (when pySBD is enabled) and forwards `metadata` objects so each entity can carry
  extra context (e.g., the originating service, clinical section, or ontological hints).
- The formatter ensures `confidence`, `start`, and `end` offsets are normalized to built-in `float`/`int` so serializing to
  JSON never fails due to NumPy/PyTorch dtypes.

## Guardrails

Pair the formatter with validation helpers from `openmed.utils.validation`:

```python
from openmed.utils.validation import (
    validate_confidence_threshold,
    validate_output_format,
    validate_batch_size,
)
```

These guardrails keep API endpoints resilient against out-of-range parameters and malformed payloads.
