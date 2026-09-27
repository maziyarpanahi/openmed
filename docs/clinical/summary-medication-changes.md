# Evidence-bound medication-change summaries

`openmed.clinical.summary_medication_changes` renders medication starts, stops,
and dose changes only from typed relations that explicitly carry an assertion
status and opaque evidence identifiers.

```python
from openmed.clinical.summary_medication_changes import (
    MedicationChangeRelation,
    render_medication_change_summary,
)

view = render_medication_change_summary(
    [
        MedicationChangeRelation(
            medication="normalized medication",
            change_type="dose_changed",
            assertion_status="confirmed",
            evidence_ids=("sha256:" + "a" * 64,),
            previous_dose="5 mg",
            new_dose="10 mg",
            effective_time="2026-02-01",
        )
    ]
)
```

Only `confirmed` relations render. Possible, conditional, refuted, incomplete,
or malformed records are withheld with controlled review codes. Conflicting
changes for the same normalized medication and effective time are also withheld
as one conflict group. Review artifacts identify input indexes but do not echo
medication, dose, time, assertion, or malformed evidence values.

Duplicate statements merge their evidence identifiers. Distinct events remain
event statements; the renderer never derives or claims a final regimen. The
implementation is deterministic, offline, and intended for clinician review,
not prescription decisions or compliance certification.
