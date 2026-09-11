# Privacy budget ledger for aggregate releases

`openmed.risk.PrivacyBudgetLedger` provides a local, deterministic gate for
repeated aggregate releases. A caller registers an epsilon and delta ceiling
for each named release context, then records a release immediately before it
is emitted. Epsilon and delta are composed conservatively by sequential
addition within each context.

The ledger does not inspect or store source rows. Its accepted spend records
contain only a safe context identifier, epsilon, delta, and sequence number.
Over-budget requests raise `PrivacyBudgetLedgerExceeded` before a spend is
appended. The decision carries numeric projected and remaining budgets, so an
integration can fail closed without putting a request payload in an exception.
The ledger-specific ceiling type is `ReleaseContextPrivacyBudget`; the existing
`openmed.risk.PrivacyBudget` differential-privacy accountant remains unchanged.

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

Checking and charging are protected by one local lock, so concurrent callers
cannot individually pass a stale check and collectively exceed the context
ceiling. `check` remains advisory; `record_release` is the atomic gate.
Even the smallest positive floating-point spend is charged; adding a tiny spend
to an exhausted ceiling cannot round it away. Rejected projections that would
round back to the ceiling are reported conservatively above that ceiling.

## Boundaries

A ledger supports at most 512 contexts and 10,000 accepted spends. Contexts
use a closed, 64-character ASCII identifier grammar. Epsilon is finite,
non-negative, and at most 1,000,000; delta is finite and in `[0, 1)`. Numeric
strings, booleans, duplicate aliases, unknown budget fields, unbounded custom
mappings, and non-finite values are rejected without including caller values
in errors. Configured budgets are exposed only through immutable snapshots.

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
