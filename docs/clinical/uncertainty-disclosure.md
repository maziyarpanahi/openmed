# Uncertainty-disclosure completeness audit

`openmed.clinical.uncertainty_disclosure` provides a local, deterministic
structural audit for guarded claim metadata. It checks whether each claim has
uncertainty categories, reason codes, evidence or provenance references, a
recognized review state, and bounded display limits. It does not interpret the
claim, assess clinical correctness, certify compliance, or make a clinical
decision.

```python
from openmed.clinical.uncertainty_disclosure import audit_uncertainty_disclosures

claims = [
    {
        "claim_id": "synthetic-claim-001",
        "uncertainty_disclosure": {
            "uncertainty_categories": ["epistemic"],
            "reason_codes": ["reason.synthetic"],
            "evidence_references": ["evidence.synthetic.001"],
            "review_state": "pending",
            "display_hints": {"max_chars": 240, "max_items": 4},
        },
    }
]

report = audit_uncertainty_disclosures(claims)
assert report.is_complete
```

The default display bounds are `max_chars` from 1 through 4096 and `max_items`
from 1 through 100. `max_lines` is available from 1 through 100, and callers
can choose which bounded hints are required with `required_display_hints`.
Callers may also provide `required_categories` and a minimum evidence-reference
count. The fields can be top-level or nested under `uncertainty_disclosure`,
`uncertainty`, or `metadata` containers.

Reports are safe for audit logs and review tooling: claim identifiers become
opaque SHA-256 keys, findings use fixed issue codes, and aggregate issue counts
are emitted instead of category names, reason codes, references, display
values, or other claim metadata. The implementation performs no network calls
and does not write files.
