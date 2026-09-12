# Audit-artifact size budgets

Counts-only audit artifacts can still grow beyond the capacity of a queue,
report store, or downstream review surface. OpenMed provides a local,
deterministic gate for checking a synthetic descriptor before a report is
accepted.

The gate accepts either `ArtifactDescriptor` instances or mappings containing
only count metadata. It does not accept a path, open a file, make a network
request, or copy section names and free-form values into its result.

```python
from openmed.compliance import (
    ArtifactDescriptor,
    ArtifactSectionDescriptor,
    ArtifactSizeBudget,
    evaluate_artifact_size_budget,
)

descriptor = ArtifactDescriptor(
    total_bytes=8_192,
    sections=(
        ArtifactSectionDescriptor(
            size_bytes=4_096,
            record_count=120,
            nesting_depth=3,
        ),
        ArtifactSectionDescriptor(
            size_bytes=4_096,
            record_count=80,
            nesting_depth=2,
        ),
    ),
)
budget = ArtifactSizeBudget(
    max_total_bytes=16_384,
    max_section_bytes=8_192,
    max_record_count=250,
    max_nesting_depth=8,
)

result = evaluate_artifact_size_budget(descriptor, budget)
assert result.within_budget
assert result.exceeded_categories == ()
```

The four dimensions are evaluated in this stable order:

| Category | Observed value | Limit |
| --- | --- | --- |
| `total_bytes` | Total serialized size, conservatively bounded by section sizes | `max_total_bytes` |
| `section_bytes` | Largest section size | `max_section_bytes` |
| `record_count` | Explicit total, or the sum of section counts | `max_record_count` |
| `nesting_depth` | Explicit maximum, or the largest section depth | `max_nesting_depth` |

Limits are inclusive. A `None` limit leaves that dimension unchecked. When a
mapping contains extra fields, the evaluator reads only the documented count
fields. The returned `ArtifactSizeBudgetResult` contains observed counts,
configured limits, and deterministic exceeded categories; it never contains
artifact content, identifiers, section labels, or report samples.

This is an operational capacity check, not a compliance certification or a
clinical decision. Select limits for the deployment's storage and processing
capacity, and validate the complete workflow locally.
