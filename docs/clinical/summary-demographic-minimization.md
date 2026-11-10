# Demographic minimization for summaries

`openmed.clinical.summary_demographic_minimizer` filters demographic evidence
before local summary generation. A purpose policy is an explicit allowlist: an
attribute class reaches the generator only when that class is listed for the
declared purpose. The control is deterministic and makes no network calls.

```python
import hashlib

from openmed.clinical.summary_demographic_minimizer import (
    DemographicEvidence,
    DemographicPurposePolicy,
    minimize_demographic_evidence,
)


def opaque_id(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


age_class = opaque_id("age-band")
policy = DemographicPurposePolicy(
    purpose_id=opaque_id("synthetic-handoff-summary"),
    allowed_attribute_class_ids=frozenset({age_class}),
)
result = minimize_demographic_evidence(
    [
        DemographicEvidence(age_class, "synthetic-adult"),
        DemographicEvidence(opaque_id("ethnicity"), "synthetic-value"),
    ],
    policy,
)
assert len(result.allowed_evidence) == 1
assert result.report.removed_count == 1
```

Pass only `result.allowed_evidence` to generation. Removed values are not
retained by the result. `result.report` contains the purpose digest, aggregate
counts, removed attribute-class digests, and the fixed reason code
`not_allowed_for_purpose`; it contains no demographic values. Safe logging
should use `result.to_dict()` or `result.report.to_json()`, never generation
inputs.

Purpose and attribute classes use opaque `sha256:<hex>` identifiers. Define
purpose policies outside sensitive records and review them as policy. This
filter does not determine which demographic attributes are clinically
appropriate, certify compliance, or make clinical decisions.
