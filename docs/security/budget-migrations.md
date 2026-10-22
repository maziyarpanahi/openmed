# Differential-privacy budget migrations

`openmed.risk.budget_migration` verifies that an aggregate DP budget ledger can
be migrated without silently resetting spend or weakening a release limit. A
snapshot contains one entry per stable `release_id`:

```python
from openmed.risk import compare_budget_migration, enforce_budget_migration

before = {
    "entries": [
        {
            "release_id": "release-2026-08-01",
            "spent_epsilon": 0.5,
            "spent_delta": 1e-6,
            "max_epsilon": 2.0,
            "max_delta": 1e-5,
            "composition": "basic",
            "policy_fingerprint": "policy-v1",
            "sequence": 1,
        }
    ]
}
after = {
    "entries": [
        {
            "release_id": "release-2026-08-01",
            "spent_epsilon": 0.75,
            "spent_delta": 2e-6,
            "max_epsilon": 2.0,
            "max_delta": 1e-5,
            "composition": "basic",
            "policy_fingerprint": "policy-v1",
            "sequence": 1,
        },
        {
            "release_id": "release-2026-08-02",
            "spent_epsilon": 0.0,
            "spent_delta": 0.0,
            "max_epsilon": 2.0,
            "max_delta": 1e-5,
            "composition": "basic",
            "policy_fingerprint": "policy-v1",
            "sequence": 2,
        },
    ]
}

review = compare_budget_migration(before, after)
assert review.passed
enforce_budget_migration(before, after)  # raises if the migration is blocked
```

For every existing release, cumulative epsilon and delta may only increase,
while the configured epsilon and delta limits may only stay the same or become
smaller. The release identifier, composition method, and policy fingerprint
must remain stable. New releases are allowed when their entries are complete;
an optional sequence number must not be reused. A policy change should use a
new release identifier after review rather than silently changing the policy
under an existing budget.

The verifier is deterministic, in-memory, and makes no network call. It also
understands the `compositions` and `policies` sections emitted by
`DPGenerationBudgetAccountant.to_dict`; when a policy has no explicit
fingerprint, a stable digest of its safe configuration fields is used.

`BudgetMigrationReport.to_dict()`, `to_json()`, and `to_markdown()` contain
only aggregate counts, numeric budget values, SHA-256 digests, composition
names, policy fingerprints, and validated release identifiers. They never
include source rows, request payloads, patient text, or cleartext identifiers.
Malformed entries and sensitive-looking identifiers are rejected without
echoing their values in exceptions. This is a migration safety check, not a
compliance certification or clinical decision guarantee.

## Charging private hyperparameter and checkpoint selection

Migrated ledgers must also carry the privacy loss of tuning. Training several
differentially private candidates and keeping the best one by a private
validation result leaks more than the selected run alone, so charging only the
selected run under-reports the spend. `openmed.risk.private_selection`
provides a `TuningLedger` that records every candidate, including abandoned
ones, and every selection event on private metrics: best-of-k, early stopping,
checkpoint selection and round selection.

Each trial records numeric mechanism parameters (noise multiplier, clipping
norm, sampling rate, round count and optional learning rate), an opaque
`sha256:` lineage digest, a status and a banded outcome. The per-trial
accountant is injected; the ledger never trains a model. Records contain no
metric values beyond the bands `not_evaluated`, `low`, `moderate`, `high` and
`very_high`, no seeds or generator state and no site identities. Unknown
mechanism fields such as `seed` or `site_id` are refused. `band_outcome` maps
a metric value to a band without retaining the value.

### Default charge: composition of every trial plus selection

Without a declared scheme, the ledger charges the basic sequential composition
of every trial's accounted guarantee plus one `scoring_cost` per private
metric evaluation: one for each candidate with an outcome band and one for
each early-stopping, checkpoint or round-selection evaluation. Selection on
public data still charges every trial, because the released model is a
function of all candidates.

```python
from openmed.risk.privacy_budget import PrivacyBudgetLedger
from openmed.risk.private_selection import (
    TrialMechanism,
    TrialPrivacyCost,
    TuningLedger,
    verify_promotion_evidence,
)


def accountant(mechanism: TrialMechanism) -> TrialPrivacyCost:
    # Synthetic stand-in for an injected round accountant.
    return TrialPrivacyCost(epsilon=0.1 * mechanism.round_count, delta=1e-7)


ledger = TuningLedger(
    accountant,
    metric_source="private",
    scoring_cost=TrialPrivacyCost(epsilon=0.05),
)
for index, band in enumerate(("moderate", "high", "low"), start=1):
    ledger.record_trial(
        TrialMechanism(
            noise_multiplier=1.1,
            clipping_norm=1.0,
            sampling_rate=0.01,
            round_count=10,
        ),
        "sha256:" + str(index) * 64,
        outcome_band=band,
    )
ledger.record_selection(2, "checkpoint_selection", evaluation_count=3)
ledger.select(2)

budget = PrivacyBudgetLedger({"tuning": {"epsilon": 8.0, "delta": 1e-5}})
ledger.charge(budget, "tuning")
evidence = ledger.promotion_evidence()
assert evidence.coverage == "all_trials" and evidence.trial_count == 3
verify_promotion_evidence(evidence, ledger)
```

The charge above is three trials of `1.0` epsilon plus six private
evaluations of `0.05`, so `3.3` epsilon and `3e-7` delta. `charge` records the
account once in a shared `PrivacyBudgetLedger`, refuses with `budget_exceeded`
without mutating either ledger, and closes the tuning ledger.

### Declared random-stopping schemes

A `RandomStoppingScheme` applies a published private-selection bound instead
of composition. The number of trials is drawn once by
`TuningLedger.draw_trial_count` from the declared distribution with an
injected `random.SystemRandom` instance; seedable generators such as
`random.Random` are refused with `rng_not_csprng`, and the generator is not
retained. A scheme mapping that declares a fixed trial count, a point-mass
distribution, recording trials before the draw, redrawing, and running more or
fewer trials than drawn are all refused.

| Distribution | Declared per-trial guarantee | Charge | Source |
| --- | --- | --- | --- |
| `geometric` | `(epsilon, 0)`-DP | `3 * epsilon`, delta 0 | Liu and Talwar (STOC 2019), Theorem 3.2 |
| `truncated_negative_binomial`, `logarithmic` | `(epsilon, 0)`-DP | `(2 + eta) * epsilon`, delta 0 | Papernot and Steinke (ICLR 2022), Corollary 3 |
| `truncated_negative_binomial`, `geometric`, `logarithmic` | RDP curve | Theorem 2, equation 5, minimized over declared orders | Papernot and Steinke (ICLR 2022), Theorem 2 |
| `poisson` | RDP curve and `(epsilon, delta)`-DP | `epsilon + mu * delta + log(mu) / (order - 1)` where `exp(epsilon_hat) <= 1 + 1 / (order - 1)` | Papernot and Steinke (ICLR 2022), Theorem 6 |

RDP bounds are converted to approximate DP at the scheme's `target_delta` with
Mironov (CSF 2017), Proposition 3. `truncated_negative_binomial_zcdp_rdp`
exposes Corollary 4 for zCDP candidates. The expected trial counts follow
Definition 1 of Papernot and Steinke; geometric stopping has mean `1 / gamma`.
Schemes whose probability of more than 1,000 trials exceeds `1e-6` are refused
before any trial runs.

The declared per-trial guarantee must cover training and every private metric
evaluation inside that trial; a trial or selection event that would exceed it
is refused with `trial_exceeds_declared_cost`. The charge depends only on the
declared scheme, never on the realized trial count. These bounds hold only
when trials are independent runs of the same candidate-generating procedure
and only the selected output is released, so the account and promotion
evidence omit the realized trial count. Truncated variants (Liu and Talwar
Theorems 3.4 and 3.5), threshold-based selection and cross-program
composition are out of scope.

### Promotion evidence

`promotion_evidence()` requires a charged ledger and a selected trial. It
cites the ledger digest, the accounting method, the charged epsilon and delta,
coverage `all_trials`, the selected lineage digest and, under composition, the
trial count. `verify_promotion_evidence` refuses evidence that cites only the
selected trial (`evidence_selected_trial_only`), covers fewer trials
(`evidence_coverage_incomplete`), does not match the ledger
(`evidence_ledger_mismatch`) or understates the account
(`evidence_account_understated`). All refusals raise `PrivateSelectionError`
with a fixed reason code and at most a field name.
