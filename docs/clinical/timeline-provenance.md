# Clinical Timeline Provenance

`export_timeline_provenance()` produces a deterministic, value-free record for
reviewing which normalized event spans contributed to a timeline. The payload
contains only document-local event identifiers, half-open source offsets,
controlled assertion status, temporal confidence, and SHA-256 policy
fingerprints. It never copies source text, normalized values, or arbitrary event
metadata.

```python
from openmed.clinical import export_timeline_provenance

events = [
    {
        "event_id": "event-1",
        "start": 12,
        "end": 21,
        "text": "SYNTHETIC_EVENT_VALUE",
        "assertion": {"negation": "affirmed", "certainty": "certain"},
        "temporal_confidence": 0.94,
    }
]

payload = export_timeline_provenance(
    events,
    policy={"profile": "synthetic-local-policy", "revision": 1},
)
```

The result is JSON-ready and has this shape:

```json
{
  "schema_version": 1,
  "policy_fingerprint": "sha256:...",
  "events": [
    {
      "event_id": "event-1",
      "source_offsets": {"start": 12, "end": 21},
      "assertion_status": "affirmed",
      "temporal_confidence": 0.94,
      "policy_fingerprint": "sha256:..."
    }
  ],
  "disclaimer": "..."
}
```

Events are ordered by an explicit timeline position when supplied; otherwise
they are ordered by source start, source end, and event identifier. Reordering
the input collection therefore does not change the export. Source values are
omitted by default. Pass `include_value_hashes=True` when a deterministic,
non-reversible content link is needed; only a SHA-256 digest is emitted.

The helper is local-first and rules-based. It does not make a network call or
read the wall clock. The output is assistive audit metadata, not a clinical
decision, compliance certification, diagnosis, or treatment recommendation.
