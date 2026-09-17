# Quantitative Relation Unit Compatibility

Quantitative relation candidates are safe to use only when their unit
dimensions are known and appropriate for the relation kind. OpenMed provides a
local, deterministic check for `dose`, `rate`, `concentration`, and
`laboratory` candidates.

The check uses the built-in UCUM-subset parser. It compares dimensions only;
it never converts a numeric value or obtains a unit definition from a network
service. Incompatible, ambiguous, missing, or unknown units are returned as
review findings.

## Check two units

```python
from openmed.clinical import check_unit_compatibility

result = check_unit_compatibility(
    "mg/dL",
    "g/L",
    relation_kind="concentration",
)

assert result.status == "compatible"
assert result.review_required is False
```

Units with different dimensions remain review findings, even when a
domain-specific conversion might be possible:

```python
result = check_unit_compatibility(
    "mg/dL",
    "mmol/L",
    relation_kind="laboratory",
)

assert result.status == "incompatible"
assert result.review_required is True
```

The checker reports only normalized unit labels, dimensions, fixed reason
codes, and optional source offsets. It does not return the candidate's
numeric value or source text.

## Validate relation candidates

`validate_quantitative_relation` accepts a mapping or an existing relation
object. The adapter recognizes explicit pairs such as `value_unit` and
`reference_unit`, or a single `unit` for category validation. Existing
`RelationCandidate` objects can provide their unit through a normalized
attribute or a synthetic numeric-and-unit span.

```python
from openmed.clinical import validate_quantitative_relation

result = validate_quantitative_relation(
    {
        "relation_type": "drug_to_rate",
        "unit": "mg/h",
    }
)

assert result.status == "compatible"
```

The four relation kinds apply these dimension rules:

| Kind | Accepted shape |
| --- | --- |
| `dose` | A dose-like amount, mass, volume, count, activity, or safe ratio without time or concentration denominator |
| `rate` | A quantity per unit time |
| `concentration` | A quantity per volume, including dimensionless percentage-style units |
| `laboratory` | Any known unit dimension; paired values still need matching dimensions |

An unknown or ambiguous unit produces `status == "unknown"`. A known unit in
the wrong shape, or two known units with different dimensions, produces
`status == "incompatible"`. Both statuses set `review_required` and must be
held for human review. Neither outcome silently converts a value.

For a batch, `validate_quantitative_relations` returns a
`UnitCompatibilityReport` in input order. Its JSON representation contains
aggregate counts and value-free result entries, so it can be retained as a
review routing artifact without copying clinical text or quantitative values.

These checks are assistive validation only. They do not establish analyte
equivalence, certify a laboratory interpretation, or replace qualified
clinical judgment.
