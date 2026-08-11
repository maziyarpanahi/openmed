# Value-free evidence coverage

`openmed.clinical` provides a deterministic coverage matrix for guarded claims.
It is a review aid only: it does not interpret claims, assess clinical
validity, or make a clinical decision.

The input contract uses an opaque claim identifier, required evidence-class
labels, a review state, and source fingerprints. Claim text, evidence text,
offsets, dates, and model output are not read into the report. Source values
can be fingerprinted locally with `fingerprint_source`; only the resulting
`sha256:` digest is serialized.

```python
from openmed.clinical import (
    build_evidence_coverage_matrix,
    fingerprint_source,
)

matrix = build_evidence_coverage_matrix(
    [
        {
            "claim_id": "claim-a1",
            "required_evidence": [
                {
                    "evidence_class": "local_record",
                    "review_state": "reviewed",
                    "source_fingerprint": fingerprint_source("synthetic-source-a"),
                },
                {"evidence_class": "second_source", "review_state": "missing"},
            ],
        },
        {
            "claim_id": "claim-b2",
            "required_evidence": [
                {
                    "evidence_class": "local_record",
                    "review_state": "unreviewed",
                    "source_fingerprint": fingerprint_source("synthetic-source-b"),
                }
            ],
        },
    ]
)

safe_report = matrix.to_dict()
safe_json = matrix.to_json()
```

The report contains sorted claim/evidence rows, fixed counts for `present`,
`missing`, `conflicting`, and `unreviewed`, plus source-fingerprint,
per-claim, and matrix hashes. Unknown input fields are ignored, so upstream
records may retain their own private data without copying it into this report.
The API performs no mandatory network call.
