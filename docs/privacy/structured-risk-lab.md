# Structured privacy and re-identification risk lab

The structured privacy lab is an offline evidence workflow for tabular
clinical data. It profiles explicitly declared direct identifiers,
quasi-identifiers, sensitive attributes, missingness, uniqueness, rare
combinations, and coded population assumptions. It then measures k-anonymity,
l-diversity, and t-closeness before and after deterministic generalization and
whole-privacy-unit suppression.

The output is evidence for qualified review. It is not a legal safe-harbor
certification, an Expert Determination, a universal anonymity guarantee, or a
single risk score that authorizes release.

## Python workflow

All policy choices are explicit. The lab does not infer a quasi-identifier set
or choose an acceptable threshold.

```python
from openmed.structured import (
    StructuredPrivacyPolicy,
    run_structured_privacy_lab,
)

policy = StructuredPrivacyPolicy(
    quasi_identifiers=("age", "postal_prefix"),
    sensitive_attributes=("diagnosis",),
    direct_identifiers=("synthetic_record_id",),
    target_k=3,
    target_l=2,
    target_t=0.5,
    suppression_limit=2,
    membership_max_inference_rate=0.0,
)

result = run_structured_privacy_lab(
    rows,
    policy,
    population_assumptions={
        "scope": "reviewed_synthetic_cohort",
        "population_kind": "synthetic_fixture",
    },
    membership_candidates=local_candidate_rows,
)

if result.meets_policy:
    release_rows = result.records  # Keep this separate from evidence.
evidence_json = result.evidence.to_json()
```

`result.evidence` contains schema and dataset hashes, coded parameters,
population assumptions, before/after k/l/t measurements, transformation and
utility deltas, bounded membership-test results, and limitations. It contains
no raw cell values, record identifiers, equivalence-class keys, or source
paths. Transformed rows are retained only in the local result object and are
not serialized by the evidence methods.

## CLI workflow

The equivalent local workflow is:

```bash
openmed risk lab input.jsonl \
  --evidence structured-risk.json \
  --output release.jsonl \
  --qi age,postal_prefix \
  --sensitive diagnosis \
  --direct-id synthetic_record_id \
  --k 3 --l 2 --t 0.5 \
  --suppression-limit 2 \
  --population-scope reviewed_synthetic_cohort \
  --overwrite
```

The release output is written only when the configured release and optional
membership gates pass. A failed policy still produces aggregate evidence and
returns a non-zero status. Use `--membership-candidates` to run the bounded
local self-test; set `--membership-max-inference-rate` explicitly when that
test is part of the release policy.

## Aggregate differential privacy

Differential privacy is a separate aggregate-release mechanism. The ledger
composes named epsilon/delta spends deterministically and rejects a query that
would exceed the declared budget. The Laplace API accepts only a scalar or a
mapping of numeric aggregates and rejects row-shaped input:

```python
from openmed.risk import AggregateDPBudgetLedger, release_aggregate

ledger = AggregateDPBudgetLedger(max_epsilon=1.0, max_delta=0.0)
aggregate = release_aggregate(
    {"count": 120},
    ledger=ledger,
    epsilon=0.25,
    seed="synthetic-test-seed",
)
```

The resulting ledger and mechanism output say `aggregate_only` and
`row_level_anonymization: false`. A differential-privacy aggregate budget does
not transform, anonymize, or authorize row-level release. The CLI equivalent
is `openmed risk dp-aggregate` with a local JSON object of named numeric
aggregates and explicit `--epsilon`, `--budget-epsilon`, and
`--budget-delta` choices.

Release schema version 2 publishes only the noisy value, accounting metadata,
and a `test_only` flag. It does not retain or serialize the noise draw or a
seed digest, including through dataclass serialization and representations.
Earlier release files containing both `value` and `noise` disclose the exact
input aggregate and must be treated as sensitive source data.

Omit `seed` for production system randomness. Supplying it is for synthetic
experiments only, labels the result `test_only: true`, and provides no production
privacy guarantee. The CLI requires `--test-mode` with `--seed`; neither flag
is appropriate for releasing real aggregates. Accounting records and their
digests are unchanged.

The separate `openmed.risk.differential_privacy` Gaussian helpers now use the
analytic privacy profile from [Balle and Wang, Theorem 8](https://proceedings.mlr.press/v80/balle18a.html),
including epsilon values above one. `gaussian_privacy_delta` checks achieved
privacy loss; near floating-point cancellation it returns a conservative tail
bound. Unrepresentable scales fail closed. Spend records identify
`analytic_gaussian` calibration without recording the noise draw or result.
Seeded Gaussian outputs change deliberately because the old classical scale
was not valid for general epsilon. Laplace calibration is unchanged.

## Limitations

The lab is intentionally bounded and local. k/l/t metrics depend on the
declared population unit and published representations; the membership probe
tests only the supplied candidate population and exact declared QIs; and
aggregate differential privacy does not cover row-level release. Reviewers
must document auxiliary data, recipients, release context, utility needs, and
any additional attack models before deciding whether to release data.
