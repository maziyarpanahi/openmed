# Laboratory-result relation candidates

`generate_lab_result_candidates()` links upstream analyte and value spans with
nearby unit, reference-interval, specimen, and observation-time evidence.
Endpoints must share a section and deterministic sentence boundary and remain
within the configured character distance.

```python
from openmed.clinical import generate_lab_result_candidates

text = "Labs: Sodium 140 mmol/L."
candidates = generate_lab_result_candidates(
    text,
    [
        {
            "label": "ANALYTE",
            "start": 6,
            "end": 12,
            "expected_unit": "mmol/L",
        },
        {"label": "LAB_VALUE", "start": 13, "end": 16},
        {"label": "UNIT", "start": 17, "end": 23},
    ],
)
```

Units are parsed with OpenMed's local UCUM subset. A parsed unit whose dimension
conflicts with `expected_unit`, `expected_units`, `allowed_units`, or reference
interval unit metadata is rejected. Unknown or missing units remain explicit
incomplete candidates; the function does not guess.

When one value can link to multiple analytes, or one analyte can link to
multiple values, all bounded candidates are preserved with
`conflict_state="competing"` and reciprocal opaque candidate identifiers. The
output contains offsets, hashes, normalized unit dimensions, and controlled
metadata—not raw analyte names or values. It is deterministic, offline, and
always requires human review.

