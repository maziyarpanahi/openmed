# Privacy-policy simulation matrix

`openmed.risk.policy_simulation` provides a local dry run for comparing two
privacy-policy versions before release. It evaluates synthetic counts by
resource class, configured actions, and explicit boolean gate outcomes. It is
review evidence, not a compliance certification or a clinical decision
guarantee.

## Simulate two versions

Use policy mappings with a stable version, action rules, and the gates that
must pass. Scenarios contain only a resource class, a non-negative count, and
boolean gate outcomes:

```python
from openmed.risk import simulate_policy_matrix

matrix = simulate_policy_matrix(
    {
        "version": "policy-v1",
        "actions": {"PERSON": "keep", "EMAIL": "mask"},
        "default_action": "keep",
        "blocking_gates": ["release"],
    },
    {
        "version": "policy-v2",
        "actions": {"PERSON": "mask", "EMAIL": "redact"},
        "default_action": "keep",
        "blocking_gates": ["release", "no_leak"],
    },
    [
        {
            "scenario_id": "synthetic-case-001",
            "resource_class": "PERSON",
            "count": 2,
            "gate_outcomes": {"release": True, "no_leak": False},
        }
    ],
)

print(matrix.to_markdown())
```

The bundled policy names and `PolicyProfile` objects are also accepted. A
profile's `strict_no_leak` and `safety_sweep_mandatory` settings become
required simulation gates named `no_leak` and `safety_sweep` respectively.
Missing required gate outcomes block a row, so an incomplete scenario fails
closed.

## Read the matrix

Every row contains the base and candidate action, gate outcome, blocked state,
affected count, and processed count. Action changes are classified as
`stronger`, `weaker`, or `unchanged`. A count is affected when its selected
action is not `keep`; processed count is zero for a blocked row. Aggregate
action buckets are weighted by the scenario count, so a change from `keep` to
`mask` moves the synthetic count between action buckets and reports an
increased affected count.

Use `to_dict()`, `to_json()`, or `render_policy_simulation_matrix(...,
fmt="dict")` for machine-readable output. The serialized matrix includes
validated resource-class and action categories, counts, gate statuses, change
classifications, and SHA-256 fingerprints. Scenario identifiers are never
serialized; policy and scenario mappings are copied and normalized, so the
simulation does not mutate caller inputs.

## Security boundary

The simulation is in-memory and offline. It does not load a model, open a
network connection, write a report, mutate an artifact, or consume a privacy
budget. Resource classes and gate names are categories, not source values;
keep them free of patient text and identifiers. Do not place raw PHI,
credentials, restricted datasets, or payload-bearing fields in scenarios or
committed fixtures.
