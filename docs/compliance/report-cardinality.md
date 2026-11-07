# Privacy report cardinality budgets

Allowlisted fields do not by themselves bound a report. An array can contain
too many items, a map can expose too many distinct keys, and nested containers
can make an otherwise valid report unexpectedly large. OpenMed provides a
local, deterministic budget check for JSON-like typed report shapes.

## Check a report

```python
from openmed.compliance import (
    ReportCardinalityBudget,
    check_report_cardinality,
)

budget = ReportCardinalityBudget(
    max_items_per_field=100,
    max_unique_keys=100,
    max_nesting_depth=4,
    max_aggregate_items=500,
)
result = check_report_cardinality(
    {
        "sections": [{"measurements": [1, 2, 3]}],
        "summary": {"count": 3},
    },
    budget,
)

if not result.allowed:
    print(result.to_json())
```

The default budget is conservative: 100 items per field, 100 unique mapping
keys, eight container levels, and 1,000 aggregate items. Pass an explicit
`ReportCardinalityBudget` when a report contract has narrower limits.

The accepted shape is composed of `None`, booleans, finite numbers, strings,
lists, tuples, and mappings with string keys. Unknown objects, non-finite
numbers, non-string keys, and cycles fail closed. The checker performs no
network or filesystem access.

## Safe violation reports

`ReportCardinalityReport.to_dict()` and `.to_json()` contain the allow/deny
decision, aggregate counts, maximum depth, and violations. Each violation has
only a field path, rule name, observed count, and configured limit. Rejected
items and mapping keys are never copied into the result.

Paths use JSONPath-like notation: `$.field` for fields and `[*]` for array
items. A mapping key is included only when it matches a bounded schema-field
identifier; arbitrary keys are represented by `[key]`. This keeps dynamic map
keys out of logs, exceptions, and reports while retaining useful field paths.

The aggregate count is the sum of the item counts of every container reached,
including the root container. Once a field, key, nesting, or aggregate limit is
exceeded, that branch is not traversed further. This gives callers a
deterministic fail-closed result without inspecting or exposing rejected
values.

This is a technical guardrail, not a compliance certification or clinical
decision guarantee. Validate the budget against the deployment's report
contract and review the resulting evidence locally.
