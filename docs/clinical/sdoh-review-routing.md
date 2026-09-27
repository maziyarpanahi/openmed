# SDOH Review Routing

`route_sdoh_review()` sends uncertain SDOH evidence to a typed human-review
queue. Routing covers low confidence, conflict, and explicit `unknown`,
`declined`, or `refused` outcomes.

```python
from openmed.clinical.sdoh_deduplicate import SDOHSourceReference
from openmed.clinical.sdoh_review import (
    SDOHOutcome,
    SDOHReviewEvidence,
    route_sdoh_review,
)

evidence = SDOHReviewEvidence(
    evidence_id="evidence-1",
    category="transportation",
    outcome=SDOHOutcome.UNKNOWN,
    confidence=0.60,
    source_references=(
        SDOHSourceReference("document-local-1", "v1", 4, 15),
    ),
)

queue = route_sdoh_review([evidence], confidence_threshold=0.75)
```

Each queue item preserves the controlled outcome and all source references. It
may carry more than one review state, such as `low_confidence` and
`conflicting`. Every serialized item sets `automated_decision_allowed` to
`false`.

`queue.summary()` is safe for queue monitoring because it contains only total,
state, and category counts. It omits evidence IDs and source references. A
high-confidence affirmed or negated observation is not queued unless it is
conflicting; this only means the router found no configured review reason. It
does not authorize an automatic eligibility, diagnosis, care-denial, or
underwriting decision.

The router uses controlled, value-free records, is deterministic, and performs
no network calls.
