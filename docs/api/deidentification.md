# De-identification API

The `deidentify()` helper returns a typed `DeidentificationResult` with the
redacted text, detected `PIIEntity` spans, result metadata, and a `to_dict()`
representation for JSON-style workflows.

## Entity Table Export

Use `DeidentificationResult.to_dataframe()` when you want to inspect the
entities detected in a single de-identification result as a table:

```python
from openmed import deidentify

result = deidentify("Patient John Doe called 555-1234", method="mask")
entities = result.to_dataframe()
```

The method imports pandas lazily, so importing `openmed` or
`openmed.core.pii` does not import pandas. If pandas is not installed, calling
`to_dataframe()` raises an actionable `ImportError` with the install command.

The returned DataFrame has one row per detected entity and always uses this
column order:

| Column | Description |
| ------ | ----------- |
| `text` | Entity text span captured by the detector. |
| `label` | Detector label for the span. |
| `entity_type` | Normalized PII entity type stored on the result entity. |
| `start` | Character start offset in the original text. |
| `end` | Character end offset in the original text. |
| `confidence` | Detector confidence score. |
| `action` | De-identification policy action applied to the entity, when available. |
| `result_id` | Stable hash-derived identifier shared by every row from the result. |

When no PII entities are present, `to_dataframe()` returns an empty DataFrame
with the same columns.
