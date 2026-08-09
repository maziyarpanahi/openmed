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
