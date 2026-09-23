# Evidence-bound laboratory sequence summaries

`openmed.clinical.summary_lab_trends` renders chronological laboratory value
sequences only when every observation has compatible normalized analyte, unit,
and observation-time fields plus an opaque evidence identifier.

```python
from openmed.clinical.summary_lab_trends import (
    LabTrendObservation,
    render_lab_trend_summary,
)

view = render_lab_trend_summary(
    [
        LabTrendObservation(
            analyte="normalized analyte",
            value=8.1,
            unit="mg/dL",
            observed_at="2026-01-01",
            evidence_id="sha256:" + "a" * 64,
        ),
        LabTrendObservation(
            analyte="normalized analyte",
            value=8.7,
            unit="mg/dL",
            observed_at="2026-02-01",
            evidence_id="sha256:" + "b" * 64,
        ),
    ]
)
```

The renderer normalizes supported units with OpenMed's local unit parser and
orders ISO dates or timestamps deterministically. It does not drop incompatible
points to make a partial sequence appear valid. Missing fields, invalid values,
different analytes or dimensions, incompatible time precision, and conflicting
values at one time suppress the entire sequence.

An insufficient result contains only controlled codes and an observation count;
it omits analytes, units, times, evidence identifiers, and values. A successful
result contains the sequence but no direction, delta, abnormality, or clinical
interpretation. This offline view is for clinician review and is not a diagnosis,
treatment recommendation, originating laboratory report, or compliance
certification.
