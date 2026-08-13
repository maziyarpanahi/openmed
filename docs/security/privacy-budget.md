# Privacy budget ledger for aggregate releases

`openmed.risk.PrivacyBudgetLedger` provides a local, deterministic gate for
repeated aggregate releases. A caller registers an epsilon and delta ceiling
for each named release context, then records a release immediately before it
is emitted. Epsilon and delta are composed conservatively by sequential
addition within each context.

The ledger does not inspect or store source rows. Its accepted spend records
contain only a safe context identifier, epsilon, delta, and sequence number.
Over-budget requests raise `PrivacyBudgetExceeded` before a spend is appended.
The decision carries numeric projected and remaining budgets, so an integration
can fail closed without putting a request payload in an exception.

## Local usage

```python
from openmed.risk import PrivacyBudgetLedger

ledger = PrivacyBudgetLedger(
    {
        "daily-release": {"epsilon": 1.0, "delta": 1e-5},
    }
)

ledger.record_release("daily-release", epsilon=0.25, delta=2e-6)
evidence = ledger.render_counts_only()
```

`check` is the non-mutating preflight operation. `record_release` performs the
same check and appends the spend only when both ceilings are respected:

```python
decision = ledger.check("daily-release", epsilon=0.5, delta=2e-6)
if decision.allowed:
    ledger.record_release("daily-release", epsilon=0.5, delta=2e-6)
```

Callers should keep the check-and-release boundary close together. The ledger
does not itself emit an aggregate, persist files, or contact a service.

## Evidence contract

`render_counts_only()`, `to_dict()`, and `to_json()` return deterministic
aggregate evidence. The report includes per-context release and rejection
counts, configured ceilings, consumed epsilon/delta totals, and remaining
headroom. It intentionally omits individual spend entries and all source,
row, cell, document, recipient, or free-form request values. Context names are
restricted to safe identifiers and PHI-shaped identifiers are rejected.

This is an accounting gate, not a compliance certification or clinical
decision guarantee. It does not select a privacy mechanism, prove a claimed
epsilon/delta value, or replace review of the release population and threat
model.
