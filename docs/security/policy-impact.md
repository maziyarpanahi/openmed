# Policy-impact digest

`openmed.risk.policy_impact` provides an offline dry run for reviewing a
candidate policy version before promotion. It evaluates the effective action,
gate, and waiver state for each safe resource type and aggregates the result by
type.

```python
from openmed.risk import evaluate_policy_impact

baseline = {
    "name": "baseline-v1",
    "actions": {"clinical_note": "keep"},
    "gates": {"clinical_note": "leakage"},
    "waivers": {"clinical_note": False},
}
candidate = {
    "name": "candidate-v2",
    "actions": {"clinical_note": "redact"},
    "gates": {"clinical_note": ["budget", "leakage"]},
    "waivers": {"clinical_note": True},
}

impact = evaluate_policy_impact(
    baseline,
    candidate,
    [{"resource_type": "clinical_note", "count": 4}],
)
print(impact.to_json())
```

The digest contains policy version names, safe resource-type counts, and
counts for each action, gate, and waiver transition. It excludes resource
identifiers, payload values, waiver reasons, and unknown policy fields. Its
`sha256:` value is computed from the canonical counts-only payload, so changing
the order of synthetic resources does not change the digest.

The simulator is read-only: it copies normalized scalar settings and never
mutates a live policy or budget. It performs no network calls. The result is
review evidence, not a compliance certification or a clinical decision.
