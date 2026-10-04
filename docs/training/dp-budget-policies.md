# Privacy budget policies

A coordinator can commit to a differential-privacy budget policy before a
federated round starts without implementing an accountant. `DPBudgetPolicy` is
an immutable, versioned contract: it names the accountant family, the
composition rule, the numeric budget bounds, the unit scope of each bound, and
what happens when the budget is exhausted.

```python
from openmed.training import (
    DPAccountant,
    DPBudgetScope,
    DPComposition,
    DPExhaustion,
    build_dp_budget_policy,
    validate_dp_budget_policy,
)

policy = build_dp_budget_policy(
    policy_id="round-policy",
    max_epsilon=8.0,
    max_delta=1e-5,
    epsilon_scope=DPBudgetScope.TOTAL,
    delta_scope=DPBudgetScope.TOTAL,
    accountant=DPAccountant.RENYI,
    composition=DPComposition.BASIC,
    exhaustion=DPExhaustion.REQUIRE_REVIEW,
)

report = validate_dp_budget_policy(policy.to_dict())
assert report.valid
print(report.policy_digest)
```

## Policy fields

| Field | Type | Accepted values |
| --- | --- | --- |
| `policy_id` | string | `[a-z][a-z0-9_.-]{0,63}` |
| `max_epsilon` | number | `0 < max_epsilon <= 1e3` |
| `max_delta` | number | `0 < max_delta < 1` |
| `epsilon_scope` | enum | `total`, `per_round` |
| `delta_scope` | enum | `total`, `per_round` |
| `accountant` | enum | `basic`, `renyi`, `zcdp`, `gaussian` |
| `composition` | enum | `basic`, `advanced` |
| `exhaustion` | enum | `fail_closed`, `refuse_round`, `require_review` |
| `delta_prime` | number | `0 <= delta_prime < max_delta`, non-zero for `advanced` |
| `min_rounds` | integer | `>= 1` |
| `max_rounds` | integer or null | unset, or `>= min_rounds` |
| `schema_version` | string | `openmed.training.dp_budget_policy.v1` |

`epsilon_scope`, `delta_scope`, `accountant`, `composition`, and `exhaustion`
have no default. A policy that omits them is rejected instead of inheriting a
permissive value, and the exhaustion vocabulary is closed: there is no
`continue`, `warn`, or `ignore` member.

## Validation rules

`validate_dp_budget_policy()` accepts a mapping and never raises for a malformed
*content* value. It reports sorted, duplicate-free findings from the closed
vocabulary `DP_BUDGET_POLICY_REASON_CODES`:

- `missing_field`, `unsupported_field`, `invalid_field_type`
- `unsupported_schema_version`
- `non_finite_value`, `epsilon_out_of_range`, `delta_out_of_range`,
  `delta_prime_out_of_range`
- `ambiguous_scope`, `unsupported_accountant`, `unsupported_composition`,
  `unsafe_exhaustion`
- `invalid_policy_id`, `invalid_round_bounds`

Floating-point bounds must be finite, so `NaN`, positive infinity, and negative
infinity are rejected, and a boolean is never accepted as a number. An
`advanced` composition requires a positive `delta_prime`, while a `basic`
composition requires it to be exactly zero.

`build_dp_budget_policy()` and the `DPBudgetPolicy` constructor are strict: a
rejected policy raises `DPBudgetPolicyRejected`, which is a
`DPBudgetPolicyError` carrying the same value-free report. Only a non-mapping
payload raises `DPBudgetPolicyError` directly.

## Serialization

`to_dict()` returns the documented field order with enums as their string
values. `to_json()` emits sorted-key JSON with a trailing newline, so two
independently built policies serialize to identical bytes. `policy_digest` is a
domain-separated `sha256:` digest over the compact, sorted policy mapping, and
the focused tests pin the exact digest of three golden policies.

A rejected report never carries a digest, and a valid report always carries one.

## Privacy boundary

The contract only describes a declared policy. It does not implement an
accountant, select a budget, compose privacy loss, train a model, or claim a
formal guarantee. Rejected reports are value free: they contain reason codes and
field names only, never a caller-supplied policy identifier, and never a site,
cohort, or participant identifier. The JSON serialization of a valid policy
contains only the declared fields.

Run the offline checks with:

```sh
uv run --frozen --extra dev python -m pytest tests/unit/training/test_dp_budget_policy.py -q
```
