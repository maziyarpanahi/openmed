# Evidence-bound care gaps

OpenMed derives conservative care-gap states from an exact, versioned clinical
measure result. Each gap version retains the measure definition, source Journey
snapshot, measurement period, policy, population-result digests, and value-free
fact/evidence custody that produced it.

Care-gap output is decision support for human review only. It does not authorize
diagnosis, treatment, outreach, enrollment, ordering, or any other patient
action.

## Four explicit states

| State | Meaning |
| --- | --- |
| `met` | The required numerator is explicitly met |
| `open` | The initial population and denominator are met, no configured exclusion or exception applies, and the numerator is explicitly not met |
| `not_applicable` | The subject is outside the initial population or denominator, or a configured exclusion or exception applies |
| `insufficient_data` | A required population is missing, unknown, errored, or backed by conflicting inputs |

The default policy is deliberately asymmetric: missing, unknown, errored, or
conflicting data can never produce `open`. An open state requires complete,
explicit population evidence.

## Evaluate a gap

```python
from openmed.clinical import CareGapPolicy, evaluate_care_gap

policy = CareGapPolicy(
    policy_id="care_gap_default",
    version="1.0.0",
    initial_population_id="initial",
    denominator_population_id="denominator",
    numerator_population_id="numerator",
    exclusion_population_id="exclusion",
    exception_population_id="exception",
)

result = evaluate_care_gap(measure_subject_result, policy)
```

The resulting `CareGapEvaluation` contains no clinical values or source text.
Its evidence includes only the exact measure-result identifier and digest,
population-result digests, opaque fact/evidence identifiers, and optional
conflict identifiers.

## Mandatory review for unresolved inputs

An `insufficient_data` state or any attached conflict starts in
`review_status="required"`. Review is a separate workflow dimension and never
silently changes the evaluated gap state.

```text
required -> in_review -> approved
                      -> rejected
```

`begin_care_gap_review` and `complete_care_gap_review` require authorization
and decision digests. Events never retain reviewer identity. Completing review
of insufficient evidence records the human decision but leaves the gap state as
`insufficient_data`; a new open or met state requires a new corrected measure
result.

## Corrections and history

Pass the prior evaluation as `previous=` when evaluating a corrected measure
result. The new version must refer to the same stable gap identity and a
different measure-result digest. It records the prior `version_id` as its
parent. `CareGapHistory` validates exact ancestry, uniqueness, and chronology.

Review transitions are immutable versions too. Their parent identifiers let a
caller preserve the complete required, in-review, and completed chain without
mutating a previous artifact.

## Compatibility

Care-gap evaluations declare schema version `1.0.0` and `same_major`
compatibility. The bundled `care_gap.schema.json` validates the public artifact.
Deserialization additionally verifies the advisory, exact fields, content
digest, version identifier, evidence identifiers, state/review invariants,
event transition graph, event chronology, and parent-version custody.
