# Minimum-necessary structured field selection

Structured exports should contain only the fields required for their declared
use. OpenMed provides a small, deterministic selector for that boundary. The
caller owns two declarative registries:

- a purpose mapping that lists eligible fields and, optionally, fields required
  for a complete selection; and
- a policy profile that can further allow or deny fields.

The selector intersects the purpose fields with the policy allowlist and then
applies the denylist. Unknown purpose mappings and policy profiles fail closed
with an empty selection. If a required field is unavailable or not permitted,
the complete selection is denied rather than silently producing a partial
export.

## Example

```python
from openmed.risk.minimum_necessary import MinimumNecessarySelector

selector = MinimumNecessarySelector(
    purpose_mappings={
        "cohort_review": {
            "fields": ("age_band", "condition_code", "visit_month"),
            "required_fields": ("condition_code",),
        }
    },
    policy_profiles={
        "research_limited": {
            "allowed_fields": ("age_band", "condition_code", "visit_month"),
            "denied_fields": (),
        }
    },
)

record = {
    "age_band": "synthetic-age-band",
    "condition_code": "SYNTHETIC-CODE",
    "visit_month": "synthetic-month",
    "raw_sensitive_value": "SYNTHETIC-SECRET",
}
selection = selector.select(
    record,
    purpose="cohort_review",
    policy_profile="research_limited",
)
export_row = selection.project(record)
```

`export_row` contains only `age_band`, `condition_code`, and `visit_month`.
`selection.to_dict()` and `selection.to_json()` contain field names, counts,
and a stable reason code, but never record or cell values. Keep the projected
row under the caller's normal data-handling controls; the selection explanation
is suitable for an audit trail because it is value-free.

## Safety boundaries

This helper is configuration-driven and makes no mandatory network call. It
does not infer a purpose from record contents, authorize the caller's stated
purpose, classify clinical fields, or provide a HIPAA, GDPR, or other legal
certification. Operators must define and review purpose mappings and policy
profiles for their own data contract. Examples above use synthetic values only.
