# Evidence-linked clinical timelines

`build_timeline_graph()` combines typed event spans, assertion context, and
temporal evidence into a deterministic graph for review and downstream
organization. It is an assistive record, not a diagnosis, treatment
recommendation, clinical decision, or medical-device output.

## Build a graph

Events use inclusive/exclusive source offsets. A timestamp is optional; when it
is present, it is normalized to an ISO string for ordering. Assertion axes can
be supplied as a `ClinicalAssertion` or as a mapping.

```python
from openmed.clinical import build_timeline_graph

note = "Synthetic procedure occurred on 2026-06-01. Synthetic finding followed."
procedure_start = note.index("Synthetic procedure")
finding_start = note.index("Synthetic finding")

graph = build_timeline_graph(
    [
        {
            "id": "event-procedure",
            "type": "procedure",
            "start": procedure_start,
            "end": procedure_start + len("Synthetic procedure"),
            "text": "Synthetic procedure",
            "timestamp": "2026-06-01",
            "assertion": {
                "temporality": "recent",
                "certainty": "certain",
                "negation": "affirmed",
            },
            "temporal_evidence": [
                {
                    "start": note.index("2026-06-01"),
                    "end": note.index("2026-06-01") + 10,
                    "value": "2026-06-01",
                    "type": "DATE",
                }
            ],
        },
        {
            "id": "event-finding",
            "type": "finding",
            "start": finding_start,
            "end": finding_start + len("Synthetic finding"),
            "timestamp": "2026-06-01",
        },
    ],
    temporal_links=[
        {
            "source": "event-procedure",
            "target": "event-finding",
            "relation": "before",
            "evidence_start": note.index("2026-06-01"),
            "evidence_end": note.index("2026-06-01") + 10,
            "evidence_value": "2026-06-01",
        }
    ],
    document_text=note,
)
```

`graph.ordered_events` follows precedence links. Events with equal timestamps
are tie-broken by timestamp availability, source start/end offsets, normalized
event type, and event id. Reordering the input iterable therefore produces the
same graph. `after` links are preserved as `after` in the serialized record but
are reversed internally for topological ordering.

## Privacy-safe output

Source text is accepted only for in-memory hashing. `to_dict()` and `to_json()`
contain event types, offsets, normalized temporal values, assertion axes,
content hashes, link types, and the assistive disclaimer; they do not contain
event surfaces, temporal surfaces, or arbitrary caller metadata. Callers can
also provide their own SHA-256 or HMAC-SHA-256 hash when the source text is not
available.

```python
payload = graph.to_dict()
payload["events"][0]["source_offsets"]
# [0, 19]
payload["events"][0]["text_hash"]
# "sha256:<content fingerprint>"
```

The graph is local and deterministic: it does not read the wall clock, make a
network request, load a model, or log source text. A directed cycle in
`before`/`after` links raises `TimelineGraphCycleError`; `overlap` links do not
impose a precedence edge. This failure is intentional so an inconsistent
temporal graph cannot be presented as an ordered timeline.

All outputs are assistive evidence for clinician review. They must not be used
to auto-trigger a diagnosis, treatment, or other clinical decision.
