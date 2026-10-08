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

## Empty populations and coverage

An empty population requires interpretation. `CohortExecution.review_required`
is true when no evaluated patient meets the expression, including a run with
no membership rows or a run in which every membership is `not_met`. The local
resolver likewise sets `CohortResult.review_required` when `patient_ids` is
empty. Both expose a typed `EmptyPopulationWarning` with code
`empty_population`. A non-empty result gets no empty-population warning;
existing unknown/conflicting saved membership evidence still requires review.

The warning's controlled `sub_reasons` describe measured context:

- `no_source_patients`: the source patient count is known to be zero.
- `concept_set_unmatched`: at least one expanded concept set has no member
  concept present in the source population.
- `unmapped_sources_present`: the measured unmapped event count is positive.
- `coverage_unknown`: at least one coverage measure is unavailable.

These reasons describe coverage limitations, not proof that the clinical
condition is absent or a causal explanation for every failed eligibility rule.
For example, all concepts can be present while temporal or occurrence criteria
produce an empty result. That result still gets `empty_population`, with no
coverage sub-reason when every coverage measure is known.

`CohortCoverage` has `source_patient_count`, `unmapped_source_count`, and
`concept_sets`. Each `ConceptSetCoverage` has a SHA-256 `concept_set_ref`,
`expanded_count`, and `matched_count`. The reference hashes the logical set
identifier; coverage and warnings carry no concept text or patient keys.
Missing measurements serialize as JSON `null`, never zero. An unknown set
inventory is `concept_sets: null`; an explicitly measured inventory with no
sets is an empty array. Counts must be non-negative integers; booleans are
rejected.

The DuckDB/Parquet resolver measures coverage before phenotype eligibility
filters. It counts distinct `person.person_id` values. For each expanded set,
it counts distinct matching concept identifiers in the five supported domain
tables, restricted to persons in that source universe and the requested
vocabulary. These source counts differ from `matched_members` provenance,
which continues to describe only returned patients. Unmapped coverage counts
domain event rows belonging to source persons whose concept is null,
nonpositive, missing from `concept`, or has no vocabulary identifier. It does
not infer absence from free text. If an aggregate query is unavailable, only
that measurement becomes unknown; resolution semantics and evidence remain
unchanged. The source connection should represent a stable caller-owned
snapshot throughout resolution and coverage measurement.

Saved executions accept caller-supplied measured coverage. Without it, all
coverage remains unknown; neither the number of membership rows nor the
source snapshot digest proves the source population size. For a run evaluated
by the local resolver, pass its measured context to the existing builder:

```python
# resolved is a CohortResult evaluated against this exact source_snapshot.
# Membership evaluation and opaque patient-key binding remain caller-owned.
result = build_cohort_execution(
    version,
    source_snapshot=snapshot,
    vocabulary_digest="sha256:" + "b" * 64,
    policy_digest="sha256:" + "c" * 64,
    evaluator_version="local-evaluator-1.0",
    memberships=(membership,),
    coverage=resolved.coverage,
)
```

### Schema and immutable-store compatibility

New saved artifacts use schema `1.1.0` with the existing `same_major` policy.
The strict schema accepts legacy `1.0.x` artifacts as well as the current
same-major shape. Current execution digests bind coverage and the execution
schema version in addition to the manifest and membership digest. Warning
contents and review state are derived and verified when parsing.

Loading a legacy execution first verifies its original manifest, identifiers,
membership digest, execution digest, counts, and review state. Its public view
then upgrades to `1.1.0`, with unknown coverage and the current empty-population
warning. The original manifest and execution identifier remain intact; the
public execution digest changes because the new view binds additional
metadata. `LocalSavedCohortStore` retains the original stored bytes when a
loaded legacy record is put back, so reads and idempotent writes never rewrite
existing immutable records. Serializing the public view emits the current
shape. Changing the loaded record creates a normal immutable-record conflict
at the same execution identity.

Resolver result and provenance identifiers are
`openmed.cohort.result.v1.1` and `openmed.cohort.provenance.v1.1`, respectively,
with `compatibility_policy: same_major`.

### Existing Journey reads

`JourneyResourceRecord.from_cohort_result()` turns either typed result into a
count-only `cohort_run` resource for the existing Python, REST, GraphQL, and SQL
read surfaces. Supply opaque `resource_id` and `cohort_id` values; the adapter
omits patient identifiers and membership evidence. Existing cohort records
with `member_count: 0` also acquire the derived warning. Missing coverage stays
unknown, including legacy resource records.

Selecting cohort fields automatically includes `warnings` and
`review_required` in the policy decision. A custom field policy must permit
those fields or the read is denied with `field_denied`. Coverage counts are
returned when permitted and selected. This preserves safety context without
changing endpoints or bypassing field policy.

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
