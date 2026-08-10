# Privacy exception budgets

OpenMed release workflows can use a privacy exception budget to put a
deterministic ceiling on waived findings. The gate is local and offline: it
does not download policy data, contact a service, or consult the system clock.

## Bounded metadata only

Each exception record contains only:

- a severity label;
- a release scope, which is converted immediately to a SHA-256 fingerprint;
- an inclusive expiry date;
- a policy fingerprint; and
- an optional count for equivalent synthetic records.

Finding text, identifiers, source paths, and other raw values are not accepted
as gate state. Mapping inputs may contain unrelated fields for compatibility,
but the evaluator ignores them. Reports and strict exceptions contain counts,
stable violation codes, and fingerprints only.

An exception with a missing or malformed severity, scope, expiry, or policy
fingerprint is unbounded and fails closed. Expired exceptions also fail closed.
When a maximum expiry duration is configured, callers must pass an explicit
`as_of` date; this avoids a time-dependent result in local or CI execution.

## Example

```python
from openmed.risk import ExceptionBudget, evaluate_exception_budget

budget = ExceptionBudget(
    max_total=2,
    max_by_severity={"high": 1},
    max_expiry_days=30,
)
report = evaluate_exception_budget(
    [
        {
            "severity": "high",
            "scope": "release",
            "expires_on": "2026-09-01",
            "policy_fingerprint": "sha256:"
            "0000000000000000000000000000000000000000000000000000000000000000",
        }
    ],
    budget,
    as_of="2026-08-11",
)
assert report.allowed
```

Use `check_exception_budget` when a denied result must raise
`ExceptionBudgetExceeded`. Its message contains only violation codes; the
verdict is safe to serialize with `to_dict()` for a release report.

This is a release-capacity control, not a compliance certification or a
clinical decision guarantee. It does not decide whether a finding should have
been waived.
