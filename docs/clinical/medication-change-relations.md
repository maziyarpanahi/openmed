# Medication-change relation candidates

`generate_medication_change_candidates()` turns existing medication spans and
explicit start, stop, hold, resume, increase, or decrease cues into reviewable
relation candidates. It reuses OpenMed's deterministic local event-frame
extractor and can attach a nearby event-time span.

```python
from openmed.clinical import generate_medication_change_candidates

text = "Plan: hold aspirin today."
start = text.index("aspirin")
candidates = generate_medication_change_candidates(
    text,
    [{"label": "MEDICATION", "start": start, "end": start + 7}],
)
```

Each candidate carries offset-and-SHA-256 evidence for the medication, trigger,
and optional time; separate medication and event assertion axes; a conflict
state; and a bounded confidence. Serialized candidates do not copy source
surfaces. Invalid offsets and unsupported inputs fail with value-free messages.

Every candidate has `review_required=True` and `prescribing_action=False`.
Candidates never create an order, treatment plan, or recommendation. The API is
deterministic, performs no network call, and requires upstream medication spans.
