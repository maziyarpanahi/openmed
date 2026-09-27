# Clinical Relation Duplicate Collapse

`collapse_duplicate_relations()` removes duplicate normalized relation edges
without losing the evidence locations that a reviewer needs. It is useful
after a local extractor, normalization stage, or a batch of document-level
relation candidates has produced repeated edges.

The output is an assistive review record. It is not a diagnosis, treatment
recommendation, compliance certification, or autonomous clinical decision.

## Normalized candidates

Pass a relation type, normalized endpoint identifiers, a score in `[0, 1]`, and
one or more source offsets. Normalized endpoint identifiers should be stable,
non-sensitive concept identifiers such as vocabulary system/code pairs. Source
identifiers are fingerprinted before they are retained in the result.

```python
from openmed.clinical import collapse_duplicate_relations

candidates = [
    {
        "relation_type": "treats",
        "normalized_head": {"system": "SYNTHETIC", "code": "H-001"},
        "normalized_tail": {"system": "SYNTHETIC", "code": "T-001"},
        "source_id": "synthetic-note-a",
        "score": 0.88,
        "evidence": [
            {"start": 12, "end": 28},
            {"start": 44, "end": 60},
        ],
    },
    {
        "relation_type": "treats",
        "normalized_head": "synthetic:h-001",
        "normalized_tail": "synthetic:t-001",
        "source_id": "synthetic-note-b",
        "score": 0.76,
        "evidence": [{"start": 8, "end": 24}],
    },
]

collapsed = collapse_duplicate_relations(
    candidates,
    hash_secret="synthetic-local-key",
)
relation = collapsed[0]
```

Equivalent keys are directed and include the normalized relation type, head,
tail, and controlled assertion/context axes. Reversed endpoints and conflicting
assertion states therefore remain separate relations. Unicode compatibility,
case, and repeated whitespace are normalized deterministically.

## Counts and confidence

Each collapsed relation exposes both counts:

- `mention_count` is the number of unique source/offset evidence locations
  retained in `evidence_locations`.
- `independent_source_count` is the number of unique source fingerprints
  contributing those locations. `source_count` is a short alias.
- `candidate_count` records how many input candidates collapsed into the edge.

Confidence is aggregated once per independent source. For a source with several
mentions or candidate scores, only that source's maximum score is used. Those
per-source maxima are combined with bounded noisy-OR. Copied-forward mentions
can still be reviewed, but they cannot masquerade as independent support.

`evidence_locations` is sorted by source fingerprint and half-open offsets.
Exact duplicate locations are retained once; distinct locations remain
available for review. The JSON-ready record includes no source text, endpoint
surface, arbitrary candidate metadata, or secret:

```python
payload = relation.to_dict()
assert payload["mention_count"] == len(payload["evidence_locations"])
assert payload["independent_source_count"] == 2
assert payload["aggregation"] == "noisy_or_by_independent_source"
```

When an existing structural relation object has no normalized endpoint fields,
the adapter uses a deterministic hash of its endpoint surface for the emitted
identifier and keeps only its source offsets. Callers that already have coded
normalization should provide `normalized_head` and `normalized_tail` so the
result remains interoperable with their terminology layer.

## Privacy and operating boundary

The implementation is local-only and uses the Python standard library. It does
not fetch terminology, call a model service, read the wall clock, or make a
mandatory network request. Source IDs are emitted only as SHA-256 or
caller-keyed HMAC-SHA-256 fingerprints. Evidence contains offsets, not text.
Validation errors use fixed field-level messages and do not echo submitted
values.

Reviewers must verify normalized identifiers, assertion state, source
independence, and evidence offsets before using a collapsed relation in a
downstream clinical workflow. Duplicate collapse reduces counting bias; it
does not establish that a relation is true.
