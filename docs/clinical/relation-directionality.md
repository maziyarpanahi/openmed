# Guarded Clinical Relation Directionality

Guarded relation producers must establish the semantic source and target before
they score a candidate or render a clinician-review item. The directionality
gate is deterministic, local-only, and fail-closed: it accepts only registered
relation predicates with compatible endpoint types and the canonical
source-to-target direction.

This is a review-safety boundary, not a causal inference engine or an
autonomous clinical decision.

## Validate before scoring

Use `validate_relation_direction()` as the first step in a guarded relation
producer. It accepts endpoint labels, mappings, or span-like objects. Fields
such as source text and confidence are ignored by the gate and never appear in
the returned value-free result.

```python
from openmed.clinical import validate_relation_direction

direction = validate_relation_direction(
    "diagnosis_to_treatment",
    {"label": "CONDITION", "start": 12, "end": 24},
    {"label": "MEDICATION", "start": 35, "end": 43},
    direction="source_to_target",
)

# Only after this succeeds should a producer calculate confidence or construct
# its clinician-review envelope.
assert direction.relation_type == "diagnosis_treatment"
assert direction.direction == "forward"
```

For an existing relation-shaped object or mapping, use
`validate_guarded_relation()`. It reads only the relation predicate, its source
and target endpoints, and an optional direction. It does not inspect a
`score`, `confidence`, review payload, arbitrary metadata, or endpoint text.

## Built-in direction registry

The registry is exposed through `GUARDED_RELATION_TYPES`,
`GUARDED_RELATION_CLASSES`, and `relation_direction_rules()`. Semantic source
and target roles, rather than left-to-right character order, define the
direction. A relation can therefore be valid when the source occurs after the
target in a note; the producer must still identify the semantic roles
correctly.

| Guarded family | Predicates | Source endpoint | Target endpoint |
|---|---|---|---|
| `causal` | `causal`, `causes` | `CAUSE` or `CONDITION` | `EFFECT`, `CONDITION`, `FINDING`, `SYMPTOM`, or `COMPLICATION` |
| `causal` | `caused_by` | `EFFECT` or `CONDITION` | `CAUSE` or `CONDITION` |
| `causal` | `etiology`, `pathogenesis`, `genetic_factor`, `high_risk_factor`, `risk_assessment_factor`, `transmission_route` | `CONDITION` | a registered causal factor |
| `causal` | `complication`, `transforms_to` | `CONDITION` | `CONDITION` |
| `treatment` | `treatment`, `diagnosis_treatment`, `prevention`, `adjuvant_treatment`, `chemotherapy`, `radiation_treatment` | `CONDITION` | `TREATMENT`, `MEDICATION`, `PROCEDURE`, or `CARE_INTERVENTION` |
| `treatment` | `drug_treatment` | `CONDITION` | `MEDICATION` |
| `treatment` | `surgical_treatment` | `CONDITION` | `PROCEDURE` |
| `procedure_indication` | `procedure_indication` | `PROCEDURE` | `CONDITION` or `INDICATION` |
| `medication_indication` | `drug_to_indication` | `MEDICATION` | `CONDITION` or `INDICATION` |

The validator accepts common model-label aliases such as `diagnosis` for
`CONDITION`, `drug` for `MEDICATION`, and `therapy` for `TREATMENT`. The
canonical type is the only type returned in the validation result.

## Typed failures

Invalid inputs raise a `DirectionalityError` subclass before downstream scoring:

- `UnknownRelationTypeError` (`unknown_relation_type`) rejects predicates that
  are not registered.
- `InvalidEndpointTypeError` (`invalid_endpoint_type`) identifies whether the
  source or target type is not allowed for the predicate.
- `InvalidRelationDirectionError` (`invalid_direction`) rejects `reverse` or
  another direction that does not match the guarded predicate.
- `RelationShapeError` (`invalid_relation_shape`) rejects missing relation or
  endpoint fields.

Errors report only canonical relation metadata, expected types, and stable
error codes. Callers may retain source offsets separately for provenance, but
submitted relation values, endpoint surfaces, identifiers, and arbitrary
metadata are not echoed. Applications should branch on `error.code` rather
than matching prose.

`validate_guarded_relations()` applies the same gate to a batch and returns
value-free results in canonical order. A failure rejects the batch rather than
silently dropping a malformed edge.

## Local-first and review boundary

The module uses only the Python standard library and performs no mandatory
network call, model load, or terminology lookup. It stores no source text,
does not emit logs, and does not calculate clinical confidence. A successful
direction result means only that the relation's declared roles satisfy the
registered structural contract. It does not establish causality, treatment
appropriateness, diagnosis, urgency, or clinical truth; downstream outputs
remain assistive and require qualified human review.
