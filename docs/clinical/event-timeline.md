# Clinical Event Timeline Assembler

`assemble_timeline()` joins already extracted entities with normalized times,
assertion axes, and optional coreference chains. It is an offline,
deterministic organization primitive for downstream summaries and trajectory
views; it does not extract text, infer a diagnosis, or make a clinical
decision.

## Assemble events

```python
from openmed.clinical import assemble_timeline

timeline = assemble_timeline(
    entities=[
        {
            "id": "mention-1",
            "entity": "condition-1",
            "label": "CONDITION",
            "event_kind": "onset",
            "start": 0,
            "end": 8,
            "section": "assessment",
        }
    ],
    normalized_times={"mention-1": {"value": "2026-01-03"}},
    assertions={
        "mention-1": {
            "temporality": "recent",
            "certainty": "certain",
            "negation": "affirmed",
            "experiencer": "patient",
        }
    },
)

timeline.events[0].normalized_time
# "2026-01-03"
timeline.events[0].source_span
# (0, 8)
```

Entity records are joined by explicit ids, source offsets, or mapping keys.
Normalized times may be `NormalizedTimex` objects, value mappings, or an
id-keyed mapping. Assertions may be `ClinicalAssertion` objects or mappings
with `temporality`, `certainty`, `negation`, and `experiencer` axes.

## Ordering and deduplication

Absolute ISO dates, datetimes, and date intervals are ordered chronologically.
Equal timestamps use source offsets and then semantic labels as deterministic
tie-breakers. Values that are relative, unresolved, or marked
`granularity_flags=["unanchored"]` remain in `timeline.unanchored_events`, the
stable trailing partial-order bucket; they are never silently discarded.

When `chains` is supplied, mentions in the same coreference chain share a
deduplication identity. Events merge only when that identity, normalized time,
and all assertion axes match. Merged events retain the earliest source span
and record every contributing offset in `event.provenance["source_spans"]`.

Negated and family-experiencer assertions stay in the result:

```python
event.assertion.negation       # "negated"
event.assertion.experiencer    # "family"
```

## Privacy boundary

`ClinicalEvent.to_dict()` contains entity and event labels, normalized time,
section, assertion axes, source offsets, and safe provenance offsets. It does
not persist a `text` or `surface` field. Keep the source note beside the
timeline only when the caller's privacy policy permits it; offsets provide the
join back to the original document when review requires it.

The timeline is assistive metadata for review and downstream organization. It
must not automatically trigger diagnosis, triage, treatment, escalation, or
other clinical decisions.
