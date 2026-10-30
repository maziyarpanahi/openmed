# Aggregate synthetic fidelity evidence

`openmed.eval.synthetic_fidelity.synthetic_fidelity_report` evaluates declared
local constraint-validity, fact-consistency and task-utility outcomes separately.
Supply `FidelityEvidence` with validated successes/trials for all three axes,
controlled distribution-category counts, and a synthetic unique-record count.
The helper does not calculate medical correctness from generated prose or replace
an independent evaluator. The caller evaluates its records locally and provides
aggregates only; no real reference records, snippets, paths or credentials enter
this API.

Pin a minimum rate per axis and a minimum trial count. No weighted average lets
high utility hide a constraint failure. Exact threshold equality passes.
Insufficient samples suppress rates and intervals and produce `inconclusive`,
never perfect scores. Adequate samples receive descriptive 95% Wilson binomial
intervals. Independence assumptions matter; correlated records do not become
independent because a binomial interval was calculated.

Optionally provide already-approved local reference aggregates. Outputs compare
per-axis rates and report total-variation distance between category proportions.
Reference category names and records are not retained in the report. Reference
rates are suppressed below the pinned minimum sample count. Unique-record share
is a separate declared diversity measure; it is not proof of novel clinical
facts. Distribution similarity is not evidence of task utility or privacy.

Count mappings are immutable snapshots. Controlled codes must come from a
caller-approved vocabulary; syntax checks do not establish PHI safety. Counts
are bounded, distribution categories are limited to 256, and no network call or
artifact publication occurs. `meets_declared_limits` reports the supplied evidence
against chosen thresholds, not clinical validity, compliance certification, or
a release decision. Preserve independent evaluator and dataset provenance outside
this aggregate report.

```sh
.venv/bin/python -m pytest tests/unit/eval/test_synthetic_fidelity.py -q
```
