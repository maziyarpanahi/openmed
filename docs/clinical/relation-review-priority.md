# Relation review priority policy

`assign_review_priority()` maps three controlled metadata fields to a configured
human-review queue band:

- relation type;
- conflict state (`none`, `competing`, `unresolved`, or `contradictory`);
- evidence completeness (`complete`, `partial`, or `missing`).

```python
from openmed.clinical import assign_review_priority

priority = assign_review_priority(
    "diagnosis_to_treatment",
    "unresolved",
    "partial",
)
assert priority.band == "band_a"
```

`band_a`, `band_b`, and `band_c` are value-free queue-order labels, with A
reviewed before B and B before C under the default policy. They are not clinical
severity or urgency labels. The result explicitly records
`clinical_urgency_inferred=False` and never inspects note text, diagnoses,
treatments, or measurement values.

Applications can provide an immutable `ReviewPriorityPolicy` with local integer
weights and thresholds. Identical metadata and policy produce identical output;
the implementation is local and performs no network call.
