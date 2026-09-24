# Privacy-safe reviewer disagreement

`openmed.eval.reviewer_disagreement` measures whether pseudonymous clinical
reviewers agree and whether disagreements receive adjudication. It runs locally,
uses no network service, and emits only aggregate counts and rates.

## Input contract

Each case needs at least two `ReviewerDecision` records with distinct
pseudonymous reviewer identifiers. Decision codes are intentionally opaque.
Disagreement cases must use one consistent `DisagreementReason` and one
consistent adjudication status across their records. Free-text reasons are
rejected so clinical text cannot accidentally enter an aggregate report.

```python
from openmed.eval.reviewer_disagreement import (
    DisagreementReason,
    ReviewerDecision,
    reviewer_disagreement_report,
)

decisions = [
    ReviewerDecision("case-01", "reviewer-a", "accept"),
    ReviewerDecision("case-01", "reviewer-b", "accept"),
    ReviewerDecision(
        "case-02",
        "reviewer-a",
        "accept",
        reason=DisagreementReason.EVIDENCE_QUALITY,
        adjudicated=True,
    ),
    ReviewerDecision(
        "case-02",
        "reviewer-b",
        "revise",
        reason=DisagreementReason.EVIDENCE_QUALITY,
        adjudicated=True,
    ),
]

report = reviewer_disagreement_report(decisions, minimum_cell_size=2)
payload = report.to_dict()
```

The agreement denominator is all reviewed cases; its numerator is the number
with unanimous decisions. The adjudication denominator is disagreement cases;
its numerator is the number marked adjudicated. Per-reason cells report the
same adjudication rate within a typed reason.

## Minimum-cell protection

The default minimum cell size is five. A rate is suppressed when its denominator
is below the threshold or when either nonzero side of its binary split is below
the threshold. Suppressing the complementary cell matters: publishing four
agreements out of five cases would reveal the single disagreement even if its
count were hidden separately.

Suppressed rates contain `null` for numerator, denominator, and rate. Reason
categories whose cells are unsafe are omitted, and the report records only how
many reason cells were suppressed. A zero-sized side is safe to publish because
it describes no reviewer or case. An empty adjudication denominator has a
`null` rate and unsuppressed zero counts.

Reports never include case identifiers, reviewer identifiers, decision codes,
or free-text payloads. Exceptions and `ReviewerDecision` representations are
also value-free. Identifiers are still held in process while grouping, so the
caller remains responsible for pseudonymizing them before constructing records
and for protecting process memory.

This metric supports review-quality monitoring. It is not a compliance
certification and does not make or trigger clinical decisions.
