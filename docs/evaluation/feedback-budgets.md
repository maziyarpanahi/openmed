# Feedback budgets for adaptive benchmarks

Adaptive benchmark participants can reconstruct hidden slices when every
submission receives unlimited exact scores or retries. OpenMed's feedback
budget ledger limits those signals per immutable submission and committed
benchmark epoch. It runs locally, performs no network access, and stores only
digests and counters.

## Bind budgets to immutable inputs

Use the sealed workflow manifest digest as the submission identifier and the
holdout commitment digest as the epoch identifier. This prevents a mutable
label or display name from resetting a budget.

```python
from openmed.eval.governance.feedback_budget import (
    FeedbackBudgetLedger,
    FeedbackBudgetPolicy,
)

policy = FeedbackBudgetPolicy(
    epoch_digest=holdout_commitment.commitment_digest,
    detailed_feedback_limit=1,
    rerun_limit=1,
    coarse_score_boundaries=(0.50, 0.80),
)
ledger = FeedbackBudgetLedger((policy,))

decision = ledger.record_official_attempt(
    policy.epoch_digest,
    sealed_manifest.manifest_digest,
    score=normalized_score,
)
```

The first `detailed_feedback_limit` scored attempts return `detailed` feedback
with the exact normalized score. Later accepted attempts return only a
zero-based `coarse_band`. Boundaries use right insertion, so with `(0.50,
0.80)`, scores below `0.50` are band `0`, scores from `0.50` through values
below `0.80` are band `1`, and scores at or above `0.80` are band `2`.

`rerun_limit` counts standard reruns after the first official attempt. Once the
initial attempt, standard reruns, and any approved failure replacements are
consumed, the ledger returns `rerun_budget_exhausted` without recording or
scoring another attempt.

## Infrastructure failures

An infrastructure failure is still an official attempt. Record it with a
SHA-256 digest of the failure evidence; do not pass logs, stack traces, request
content, or clinical values.

```python
failed = ledger.record_official_attempt(
    policy.epoch_digest,
    sealed_manifest.manifest_digest,
    infrastructure_failure_digest=failure_evidence_digest,
)
```

Recording a failure does not grant another execution. An operator must review
the underlying evidence through an access-controlled workflow and approve that
specific recorded failure:

```python
approval = ledger.approve_infrastructure_failure(
    policy.epoch_digest,
    sealed_manifest.manifest_digest,
    failure_evidence_digest,
    operator_approval_record_digest,
)
```

Each recorded failure and approval digest is one-time use. Approval grants
exactly one replacement attempt; it does not replenish the exact-score budget.
The ledger verifies structural binding and replay protection, but caller-side
authentication and authorization determine who may invoke the approval path.

## Operations and privacy

The implementation is an in-memory enforcement primitive. A production
benchmark must serialize access through one authoritative ledger or persist
equivalent digest-only events transactionally. Otherwise, restarts or multiple
uncoordinated workers could reset or race a budget.

Decisions and snapshots omit epoch, submission, failure, and approval values.
Coarse decisions omit the exact score. Safe audit material consists of schema
versions, closed reason codes, counters, and SHA-256 identifiers; raw holdout
content and clinical identifiers must never be written to logs or artifacts.

Feedback budgets limit benchmark adaptation. They are not a compliance
certification and do not authorize autonomous clinical decisions.
