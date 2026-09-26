# Temporality-bucketed timelines

`build_timeline()` groups already-tagged clinical spans into historical,
recent, and hypothetical lanes. It is a deterministic view for local review,
not a clinical chronology of record. The caller must supply each span's
temporality axis and half-open source offsets. Missing certainty is represented
as uncertain; missing dates stay unanchored.

```python
from openmed.clinical import build_timeline, extract_timex

note = "History of fever. Fever returned 3 days ago."
time_expr = extract_timex(note, document_time="2026-06-15")[0]
timeline = build_timeline(
    [
        {"text": "fever", "label": "CONDITION", "start": 11, "end": 16,
         "temporality": "historical"},
        {"text": "Fever", "label": "CONDITION", "start": 18, "end": 23,
         "temporality": "recent", "time_expr": time_expr},
    ]
)
timeline.lanes["historical"]
timeline.lanes["recent"]
timeline.to_jsonl()
```

When supplied, an anchored TIMEX3 `TimeExpr` provides a normalized date for
ordering. An unanchored relative expression keeps its source offsets,
fingerprint, controlled direction, and normalized duration but does not receive
a date. A conservative `TemporalInterval`
with unknown or conflicting components also remains unanchored. Other spans
fall back to document-offset order within each lane. Hypothetical spans never
appear in historical or recent lanes.

The serialized schema includes a version, an advisory disclaimer, source
offsets, controlled event kinds, normalized time when explicitly known, and
source fingerprints when the expression text is available. It does not copy
note surfaces or arbitrary source identifiers. Consumers can use the offsets
to display the original evidence in their local, access-controlled source.
The JSONL form is byte-stable for the same spans regardless of input order.
