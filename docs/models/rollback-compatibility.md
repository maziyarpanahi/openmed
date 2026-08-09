# Model-registry rollback compatibility

`openmed.core.registry_compatibility` provides a local-only, fail-closed
compatibility report for moving a registry `latest` pointer back to a
last-green checkpoint. It compares metadata; it does not load model weights,
download tokenizers, inspect checkpoint paths, or contact a registry service.

## What is checked

The report requires all of the following to pass:

1. the two checkpoints are identified and belong to the same model family;
2. the rollback checkpoint is the current checkpoint itself or a recorded
   lineage ancestor;
3. the rollback SemVer satisfies the declared compatibility constraint;
4. policy fingerprints match exactly;
5. tokenizer contracts match exactly; and
6. evidence-schema version sets match exactly.

Missing metadata, malformed SemVer constraints, mismatches, or unproven lineage
produce a `blocked` report. A report is `compatible` only when every check
passes. This is a technical reproducibility gate, not a clinical decision or
compliance certification.

## Example

```python
from openmed.core.registry_compatibility import (
    build_rollback_compatibility_report,
)

current = {
    "model_id": "synthetic-model-v2",
    "family": "PII",
    "version": "2.0.0",
    "semver_constraint": ">=1.0.0,<3.0.0",
    "lineage": [
        {
            "relation": "supersedes",
            "from": "synthetic-model-v1",
            "to": "synthetic-model-v2",
        }
    ],
    "policy_fingerprint": "synthetic-policy-v1",
    "tokenizer_ids": [101, 202, 303],
    "evidence_schema_versions": ["openmed.evidence.v1"],
}
rollback = {
    **current,
    "model_id": "synthetic-model-v1",
    "version": "1.0.0",
    "lineage": [],
}

report = build_rollback_compatibility_report(current, rollback)
assert report.compatible
print(report.to_json())
```

`to_json()` and `to_markdown()` contain only decision data, SemVer values,
reason codes, and stable SHA-256 references. Model IDs, policy contents,
tokenizer IDs, lineage values, and paths are never copied into the report.

If registry state already contains `latest` and `last_green` pointers, the
same evaluator can derive those pointers without I/O:

```python
report = build_rollback_compatibility_report(
    registry_state=local_registry_state,
    family="PII",
)
```

The state must include the contract metadata required by the checks (usually
under a local `checkpoints` or `artifacts` mapping). Otherwise the result is
correctly blocked rather than assuming that a last-green pointer is enough.
