# Deterministic Summary Length Budgets

`build_summary_length_budget()` creates a bounded planning artifact for a
local clinical-summary generator. It assigns an explicit output-token cap to
each policy-approved evidence class before generation starts. When the
requested evidence does not fit, the artifact identifies every class with
deferred tokens so the caller can surface the omission for review.

The planner is an assistive control, not a summary generator, clinical
decision, compliance certification, or completeness guarantee. A qualified
clinician must review the generated summary and any deferred evidence.

## Plan from token demands

The caller supplies token counts estimated by its local, already-reviewed
evidence pipeline. The budgeting layer does not accept source text or extracted
values and does not invoke a tokenizer, model, or network service.

```python
from openmed.clinical import build_summary_length_budget

budget = build_summary_length_budget(
    max_tokens=96,
    evidence={
        "safety": 18,
        "active_problems": 64,
        "medications": 28,
        "key_findings": 46,
        "follow_up": 24,
    },
)

for allocation in budget.allocations:
    print(allocation.evidence_class, allocation.allocated_tokens)

print(budget.deferred_evidence_classes)
print(budget.to_json())
```

`allocated_tokens` is the class-specific cap the local generator should use.
`requested_tokens` is the caller's non-sensitive estimate, and
`deferred_tokens` is the amount that could not be included under the policy and
global cap. A class is listed in `deferred_evidence_classes` when any amount
was deferred, including a partial truncation. Classes with no demand are
reported as `empty` and do not consume the global budget.

## Default policy

The closed `clinical_summary_v1` policy approves these classes:

| Class | Priority | Weight | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| `safety` | 0 | 4 | 16 | 128 |
| `active_problems` | 1 | 3 | 16 | 192 |
| `medications` | 2 | 2 | 8 | 128 |
| `key_findings` | 3 | 2 | 8 | 160 |
| `procedures` | 4 | 1 | 0 | 96 |
| `pending_items` | 5 | 2 | 8 | 96 |
| `follow_up` | 6 | 2 | 8 | 96 |

Minimums are honored when the global cap can satisfy them. If the cap is too
small, classes receive tokens in priority order, keeping higher-priority
approved evidence visible first. Remaining capacity is distributed by weight
with a largest-remainder calculation; ties are resolved by priority and then
class identifier. This makes the result stable for equivalent inputs regardless
of mapping or iterable order.

The default policy can be inspected or replaced with an explicitly declared
policy:

```python
from openmed.clinical import (
    SummaryEvidenceClassPolicy,
    SummaryLengthBudgetPolicy,
    build_summary_length_budget,
)

policy = SummaryLengthBudgetPolicy(
    policy_id="handoff_summary_v1",
    classes=(
        SummaryEvidenceClassPolicy("safety", priority=0, weight=3, minimum_tokens=8),
        SummaryEvidenceClassPolicy(
            "active_problems", priority=1, weight=2, minimum_tokens=8
        ),
        SummaryEvidenceClassPolicy("follow_up", priority=2, weight=1),
    ),
)

budget = build_summary_length_budget(
    total_tokens=64,
    evidence={"safety": 20, "active_problems": 40, "follow_up": 24},
    policy=policy,
)
```

Custom class identifiers must be stable, lowercase policy metadata such as
`pending_items`; do not use patient names, note text, extracted values, or
other sensitive content as identifiers. Unknown classes are rejected with a
fixed `SummaryLengthBudgetError` reason rather than silently entering a
generation plan.

## Safe reports and local-first operation

`SummaryLengthBudget.to_dict()` and `to_json()` contain only the policy id,
class identifiers, token counts, fixed statuses, and the mandatory
clinician-review guardrails. They do not contain source text, prompts,
identifiers, model output, or filesystem paths. Validation exceptions expose
only fixed reason codes, so callers can record them without copying a rejected
value into logs or audit artifacts.

The module uses only the Python standard library. It performs no mandatory
network call and no model loading. Feed a local generator the individual
`allocation.allocated_tokens` values, keep source resolution inside the
operator-controlled process, and preserve the returned truncation metadata
with the human-review record.
