# Structured privacy access-review expiry gate

`openmed.compliance.access_review_expiry` provides a local, deterministic gate
for structured privacy access reviews. It checks the review's issue time,
exclusive expiry boundary, policy fingerprint, and required decision categories
against a clock supplied by the caller.

The model intentionally contains only allow-listed metadata:

- timezone-aware `issued_at` and `expires_at` timestamps;
- an opaque policy fingerprint; and
- structural decision-category identifiers.

It has no identity, request, record, or free-text fields. Mapping values passed
as category-to-decision metadata are ignored and are never copied into a report.
Validation errors describe the failing field without echoing its value.

```python
from datetime import datetime, timezone

from openmed.compliance.access_review_expiry import (
    AccessReview,
    evaluate_access_review,
)

utc = timezone.utc
review = AccessReview(
    issued_at=datetime(2026, 8, 10, 8, 0, tzinfo=utc),
    expires_at=datetime(2026, 8, 10, 16, 0, tzinfo=utc),
    policy_fingerprint="sha256:" + ("a" * 64),
    decision_categories=("purpose", "scope", "retention"),
)
result = evaluate_access_review(
    review,
    expected_policy_fingerprint="sha256:" + ("a" * 64),
    required_decision_categories=("purpose", "scope", "retention"),
    as_of=datetime(2026, 8, 10, 12, 0, tzinfo=utc),
)

assert result.passed
print(result.to_json())
```

The result is always an explicit `pass` or `block`. Block reasons are stable
codes: `not_yet_valid`, `expired`,
`policy_fingerprint_mismatch`, and `missing_decision_categories`. A timestamp
equal to `issued_at` is valid; a timestamp equal to `expires_at` is expired.
Omitting the clock is an input error rather than an implicit read of wall-clock
time, which keeps release checks reproducible.

The JSON report includes timestamps, category identifiers, the category gap,
and a boolean fingerprint comparison; it does not include fingerprint values,
identities, request content, decision notes, or records. This is an offline
technical gate, not a compliance certification, access grant, or clinical
decision guarantee. Review the result against deployment policy before using
it in release automation.
