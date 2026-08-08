# Privacy policy as data

`openmed.risk.policy_schema` provides a versioned, local-only declaration for
privacy policy intent. It makes the values that affect review and release
decisions visible in a deterministic diff instead of leaving them in prose or
scattered keyword arguments.

This is a configuration contract, not a compliance certification or a clinical
decision guarantee. Operators still need legal, security, and data-governance
review for their deployment.

## Version-one shape

The version-one object contains five policy decisions:

```json
{
  "schema_version": 1,
  "name": "synthetic-clinical-review",
  "jurisdiction": {
    "code": "EU",
    "name": "European Union",
    "region": "test"
  },
  "recall_floors": {
    "default": 0.97,
    "direct_identifier": 0.99,
    "critical": 1.0,
    "by_label": {
      "PERSON": 0.995
    }
  },
  "default_action": "mask",
  "actions": {
    "PERSON": "replace",
    "EMAIL": "mask"
  },
  "surrogate_strategy": {
    "kind": "deterministic",
    "consistent": true,
    "reversible": false,
    "key_ref": "env:OPENMED_POLICY_KEY"
  },
  "audit_retention": {
    "enabled": true,
    "retention_days": 30,
    "include_text": false,
    "include_mappings": false
  }
}
```

`by_label` overrides the applicable recall category. `actions` uses canonical
OpenMed labels and falls back to `default_action`; policy-category action keys
such as `DIRECT_IDENTIFIER` are also accepted for compatibility with existing
policy profiles. Supported actions are `keep`, `redact`, `replace`, `mask`,
`hash`, and `format_preserve`. Unknown actions fail validation before a policy
can be used.

The default policy is explicit and preserves the existing local behavior:

- `mask` is the default action;
- recall floors are `0.97` by default, `0.99` for direct identifiers, and `1.0`
  for critical paths;
- no surrogate state is retained; and
- audit retention is disabled (`retention_days: 0`).

Legacy flat `default_action`, `actions`, `recall_floor`, and `method` fields can
be loaded and are normalized into the version-one shape. New documents should
write `schema_version: 1` and the explicit nested fields.

## Safe loading and deterministic review

```python
from openmed.risk import load_policy_schema

policy = load_policy_schema("policy.json")
assert policy.action_for("PERSON") == "replace"
print(policy.digest)  # stable sha256 digest of canonical policy data
```

Loading accepts mappings, JSON text, and local paths only. It never fetches a
URL or makes a mandatory network call. Canonical JSON sorts keys and uses
finite numeric values, so equivalent policy objects have the same digest.

Surrogate configuration may contain an operator-managed `key_ref`, such as an
environment-variable or vault identifier, but never key material, seeds, raw
source values, or replacement mappings. Audit retention is limited to
privacy-safe hashes, offsets, and aggregates: attempts to retain source text
or mappings are rejected. Validation errors likewise omit configured values so
an invalid policy cannot turn an exception into a data-leak path.

Keep policy files under change control and treat a digest change as a review
event. The schema records intent; it does not itself run a model, store an
audit artifact, or alter the runtime de-identification method.
