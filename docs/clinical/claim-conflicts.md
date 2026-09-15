# Evidence-claim contradiction review

`openmed.clinical.claim_conflicts` provides a deterministic, local-only review
boundary for summary claims. A claim cites synthetic or caller-owned evidence
IDs; assertion, temporal, and source-integrity records are joined by those IDs
and incompatible evidence is routed to human review. The module never chooses
which clinical statement is true.

```python
from openmed.clinical import review_claim_conflicts

report = review_claim_conflicts(
    claims=[
        {
            "claim_id": "claim-1",
            "expected_assertion": "affirmed",
            "evidence_ids": ["evidence-a", "evidence-b"],
        }
    ],
    assertion_records=[
        {"record_id": "evidence-a", "assertion": "affirmed"},
        {"record_id": "evidence-b", "assertion": "negated"},
    ],
    temporal_records=[
        {
            "record_id": "evidence-a",
            "interval": {"start": "2026-01-01", "end": "2026-01-02"},
        },
        {
            "record_id": "evidence-b",
            "interval": {"start": "2026-03-01", "end": "2026-03-02"},
        },
    ],
    source_integrity_records=[
        {"record_id": "evidence-a", "status": "verified"},
        {"record_id": "evidence-b", "status": "mismatch"},
    ],
)

assert report.review_state == "review_required"
assert "assertion_conflict" in report.claims[0].review_routes
```

The report exposes claim and record identifiers, counts, fixed review routes,
and SHA-256/HMAC-SHA-256 fingerprints. `to_dict()` and `to_json()` omit source
text, excerpts, and temporal values, so they are suitable for an audit queue or
counts-only operational summary. Missing records, unverified sources, hash
mismatches, incompatible assertion states, and disjoint intervals all require
review. A `clear` result means only that these local consistency checks found no
contradiction; it is not a clinical conclusion.

The function performs no mandatory network call, reads no environment state,
and emits no logs. Callers remain responsible for supplying synthetic or
otherwise authorized records and for qualified clinical review before any use.
