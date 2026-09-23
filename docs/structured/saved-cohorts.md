# Saved cohorts and membership evidence

OpenMed can persist a cohort definition and every deterministic execution as
immutable, local artifacts. The contract is intended for reproducible
analytics and dataset preparation. It never enrolls a person, sends outreach,
or initiates a clinical action.

## What is saved

A definition version contains the canonical `PhenotypeDefinition`, its digest,
an explicit schema version, and a deterministic opaque version identifier. An
execution manifest binds that version to all inputs that can change a result:

- a named source `CohortSourceSnapshot` and digest;
- the vocabulary digest;
- the evaluation-policy digest;
- the evaluator version; and
- the exact cohort expression and criterion identifiers.

Changing any of these inputs creates a different execution identifier. Output
membership is separately hashed. A rerun of the same execution succeeds only
when the membership digest is identical.

## Build and persist an execution

The evaluator is responsible for deriving criterion states from the named
point-in-time snapshot. The saved contract accepts only opaque patient, fact,
evidence, and time-window identifiers. It has no field for source text or a
direct patient identifier.

```python
from pathlib import Path

from openmed.structured.cohort import (
    CohortMembership,
    CohortSourceSnapshot,
    CriterionMembership,
    LocalSavedCohortStore,
    MembershipEvidence,
    MembershipState,
    PhenotypeDefinition,
    build_cohort_execution,
    save_cohort_definition,
)

definition = PhenotypeDefinition.load("cohort-definition.json")
version = save_cohort_definition(definition)
snapshot = CohortSourceSnapshot(
    snapshot_id="snapshot_AAAAAAAAAAAAAAAA",
    digest="sha256:" + "a" * 64,
    schema_version="journey-snapshot-v1",
    license_tags=("caller_supplied",),
)
membership = CohortMembership(
    patient_key="patient_AAAAAAAAAAAAAAAA",
    state=MembershipState.MET,
    criteria=(
        CriterionMembership(
            criterion_id="has-condition",
            state=MembershipState.MET,
            evidence=(
                MembershipEvidence(
                    evidence_id="evidence_AAAAAAAAAAAAAAAA",
                    fact_id="fact_AAAAAAAAAAAAAAAA",
                    time_window_id="window_AAAAAAAAAAAAAAAA",
                ),
            ),
        ),
    ),
)
result = build_cohort_execution(
    version,
    source_snapshot=snapshot,
    vocabulary_digest="sha256:" + "b" * 64,
    policy_digest="sha256:" + "c" * 64,
    evaluator_version="local-evaluator-1.0",
    memberships=(membership,),
)
assert result.ok and result.value is not None

store = LocalSavedCohortStore(Path(".openmed/saved-cohorts"))
assert store.put_definition(version).ok
assert store.put_execution(result.value).ok
```

The local store uses append-only, mode-restricted files under a bounded root.
Writing the same bytes is idempotent. A different payload at an existing
content identity is a typed `conflict`, never an overwrite.

## Four-state membership

Every criterion and patient membership uses one of four states:

| State | Meaning | Eligible | Review required |
| --- | --- | --- | --- |
| `met` | Fully resolved expression match | yes | no |
| `not_met` | Fully resolved expression non-match | no | no |
| `unknown` | Evidence is insufficient | no | yes |
| `conflict` | Evidence is contradictory | no | yes |

Unknown and conflict states require a controlled reason code. Four-state
expression evaluation is conservative: unresolved branches dominate resolved
truth values, including in `or` expressions, so uncertainty cannot disappear
behind a matching branch. A membership cannot be marked eligible when any
criterion requires review.

## Criterion-level explanations

`explain_cohort_membership` reuses the saved criterion evidence rather than
building a second explanation format:

```python
from openmed.agent.workflows import explain_cohort_membership

explained = explain_cohort_membership(
    result.value,
    "patient_AAAAAAAAAAAAAAAA",
)
assert explained.ok and explained.value is not None
print(explained.value.to_json())
```

The explanation contains the opaque patient key, criterion states, controlled
reason codes, evidence identifiers, fact identifiers, time-window identifiers,
digests, and the safety advisory. Missing memberships return typed `unknown`;
invalid direct identifiers return `failure` without echoing the input.

## Deterministic reruns

`LocalSavedCohortStore.rerun` loads the original definition and manifest and
passes a `CohortEvaluationContext` to a caller-supplied local evaluator. The
rerun is successful only when the new membership digest matches the persisted
digest. Drift returns `conflict` with code
`cohort_rerun_digest_mismatch` and preserves the candidate run for inspection.

## Compatibility and licensing boundaries

Both persisted artifacts declare schema version `1.0.0` and compatibility
policy `same_major`. The machine-readable schema is bundled as
`saved_cohort.schema.json`. Unsupported versions fail explicitly.

OpenMed records vocabulary digests and caller-supplied license tags, but never
bundles vocabulary content. A source snapshot declaring
`bundled_vocabulary=True` is rejected. Committed examples and tests use only
synthetic identifiers and public numeric concept identifiers.
