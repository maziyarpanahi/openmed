# Procedure-to-indication relation candidates

`generate_procedure_indication_candidates()` creates a candidate only when a
procedure and condition-like span occur in the same section and sentence,
within the configured distance, with explicit linking language such as
“performed for,” “due to,” or “to evaluate.” Proximity alone is insufficient.

```python
from openmed.clinical import generate_procedure_indication_candidates

text = "Plan: Biopsy performed for possible lung mass."
candidates = generate_procedure_indication_candidates(
    text,
    [
        {"label": "PROCEDURE", "start": 6, "end": 12},
        {"label": "CONDITION", "start": 36, "end": 45},
    ],
)
```

The result retains offset-and-hash evidence for both endpoints and the linking
cue, plus the indication's controlled assertion axes. It never serializes the
source surfaces. Every candidate has `confirmation_required=True` and
`appropriateness_assessed=False`; no appropriateness or clinical-value judgment
is made. Candidate generation is deterministic and performs no network call.

