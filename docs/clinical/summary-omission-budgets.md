# Summary omission budgets

`openmed.clinical.summary_omission_budget` provides a deterministic gate for
checking whether a generated summary omitted policy-important source facts.
It is an offline review control, not a clinical decision or compliance
certification.

Each caller-defined importance class has a positive severity weight and its
own omission limit. Limits are evaluated independently, so strong coverage in
one class cannot compensate for an over-budget class. A mandatory class always
has a zero limit and any omission from it returns
`mandatory_class_omitted`, regardless of aggregate coverage.

```python
import hashlib

from openmed.clinical.summary_omission_budget import (
    ImportanceClassPolicy,
    SummaryEvidenceCoverage,
    evaluate_summary_omission_budget,
)


def opaque_id(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


mandatory = ImportanceClassPolicy(
    class_id=opaque_id("mandatory"),
    severity_weight=10,
    mandatory=True,
)
report = evaluate_summary_omission_budget(
    [
        SummaryEvidenceCoverage(
            evidence_id=opaque_id("synthetic-fact-1"),
            importance_class_id=mandatory.class_id,
            represented=False,
        )
    ],
    [mandatory],
)
assert not report.passed
assert report.refusal_code == "mandatory_class_omitted"
```

Evidence and policy classes use opaque `sha256:<hex>` identifiers. The report
contains only identifiers, counts, weights, limits, and categorical refusal
codes; it never carries source facts, summary text, or evidence identifiers.
Invalid-input exceptions are categorical and do not echo caller values.

The gate does not generate text, call a model, make network requests, or claim
that a summary is clinically complete. Callers remain responsible for defining
and reviewing appropriate importance classes and limits.
