# SDOH Category Completeness

`audit_sdoh_completeness()` records what happened to every configured SDOH
category. The controlled states are `processed`, `skipped`, `unsupported`, and
`failed`.

```python
from openmed.clinical.sdoh_completeness import (
    SDOHCategoryResult,
    SDOHCategoryState,
    audit_sdoh_completeness,
)

audit = audit_sdoh_completeness(
    configured_categories=("food", "housing"),
    results=(
        SDOHCategoryResult(
            category="housing",
            state=SDOHCategoryState.PROCESSED,
            finding_count=0,
        ),
    ),
)
```

The example reports `housing` as processed with
`unmentioned_not_negative`. It reports the missing `food` processing record as
failed with `missing_processing_result`. This makes pipeline absence visible
without manufacturing a negative social-history finding.

Skipped, unsupported, and failed categories require a controlled reason code
and cannot claim findings. The audit exposes state and reason counts plus one
record per configured category. It rejects duplicate results and results for
unconfigured categories.

Category and reason codes must use a controlled lowercase format. Do not place
note text, patient identifiers, or free-text exception details in these fields.
The audit is deterministic, value-free, and performs no network calls.
