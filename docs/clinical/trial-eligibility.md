# Reviewable trial eligibility matching

OpenMed can parse public eligibility text, retrieve candidate studies, and
evaluate supported criteria against an explicitly named Journey snapshot. The
result is a conservative review aid. It is not a clinical recommendation,
eligibility determination, enrollment action, or authorization to contact a
person or study site.

## Parse public criteria

`parse_trial_criteria()` separates inclusion and exclusion fragments and
retains the source study version. Supported deterministic fragments can carry:

- a condition, medication, procedure, or observation concept;
- `exists`, equality, inequality, and ordered numeric operators;
- typed numeric thresholds and normalized units; and
- a bounded `within N days` recency window.

Every fragment that cannot be represented remains in the parsed artifact with
`supported=false` and a controlled reason. Missing eligibility text becomes an
explicit unsupported criterion rather than an empty successful parse.

## Match a named Journey snapshot

```python
from openmed.clinical.journey_contracts import canonical_digest
from openmed.clinical.trials import JourneySignal, evaluate_trial_eligibility

signal = JourneySignal(
    concept_kind="condition",
    concept="Hypertension",
    snapshot_id=journey_snapshot.snapshot_id,
    snapshot_digest=canonical_digest(journey_snapshot.to_dict()),
    fact_ids=("fact_opaqueidentifier01",),
    evidence_ids=("evidence_opaqueidentifier01",),
)

result = evaluate_trial_eligibility(
    study,
    journey_snapshot,
    (signal,),
    evaluated_at="2026-09-21T10:00:00Z",
)
```

Patient values in `JourneySignal` are transient comparison inputs. They are
excluded from `repr`, the serialized result, and `to_review_packet()`. Each
criterion result instead retains only the exact study version, Journey snapshot
identifier and digest, opaque fact/evidence/conflict identifiers, state, and a
controlled reason code.

## Criterion and aggregate states

Each criterion returns one of `met`, `not_met`, `unknown`, `conflict`, or
`unsupported`. Explicitly negated concept evidence can produce `not_met`; an
absence of evidence produces `unknown`.

The aggregate result is `eligible`, `not_eligible`, or `review_required`.
`eligible` requires every inclusion criterion to be met, every exclusion
criterion to be not met, and no unresolved criterion. Missing, conflicting, or
unsupported criteria always block an eligible result under the fail-closed
policy.

`retrieve_trial_candidates()` uses deterministic lexical overlap for local
candidate ranking. `match_trial_candidates()` ranks first and then evaluates a
bounded result set. Candidate and eligibility outputs contain no query concepts
or patient values.

## Evaluation and compatibility

`run_trial_eligibility_benchmark()` reports retrieval recall at K and
criterion-state accuracy over synthetic or otherwise permitted frozen cases.
Reports retain fixture, policy, parser, and matcher provenance.

Serialized eligibility results use schema version `1.0.0` with `same_major`
compatibility and validate against the bundled `trial_eligibility.schema.json`.
The parser and matcher versions are pinned separately so benchmark and review
artifacts remain reproducible.
