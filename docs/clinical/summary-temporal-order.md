# Summary temporal-order validation

`openmed.clinical.summary_temporal_order` checks whether the event references in
a generated claim follow the order established by an existing privacy-safe
clinical timeline. It is deterministic, offline, and generative-last: the
validator does not accept, regenerate, or rewrite clinical claim text.

```python
from openmed.clinical.summary_temporal_order import (
    validate_summary_temporal_order,
)

result = validate_summary_temporal_order(
    ("event-admission", "event-procedure", "event-discharge"),
    evidence_timeline,
)
if result.review_required:
    route_to_review(result.to_dict())
```

Every ordered pair is evaluated against the transitive closure of retained
`BEFORE` and `AFTER` timeline relations. A relation proven in the opposite
direction produces `order_inversion`. Unknown events, duplicate references,
insufficient references, and pairs with no proven direction produce controlled
review codes rather than being guessed from source offsets or input order.

The result contains only reference indexes, counts, and reason codes. It does
not contain event identifiers, source text, or clinical values, so it can be
used as a value-free review artifact. The original claim remains unchanged.
This is an assistive review check, not a clinical decision or compliance
certification.
