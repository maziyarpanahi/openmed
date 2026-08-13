# Nested redaction idempotence

`openmed.risk.idempotence` checks whether a structured redaction result is
stable when the result is processed a second time. It is a deterministic,
local review aid, not a FHIR or OMOP conformance check, a compliance
certification, or a clinical decision guarantee.

## Compare two passes

Passes may be nested JSON mappings, local JSON files, or result objects with a
`resource`/`data`/`output` attribute and a report. Report metadata can contain
aggregate `counts`, a policy or `policy_fingerprint`, and redaction events:

```python
from openmed.risk import check_idempotence

first = {
    "resource": {
        "resourceType": "Bundle",
        "entry": [{"resource": {"resourceType": "Patient", "id": "synthetic-a"}}],
    },
    "report": {
        "policy_fingerprint": "sha256:" + "a" * 64,
        "counts": {"redacted": 1},
        "redactions": [
            {
                "path": "entry[0].resource.id",
                "action": "replace",
                "surrogate": "[SYNTHETIC-ID]",
            }
        ],
    },
}

second = first  # The second pass has the same synthetic evidence.
result = check_idempotence(first, second)
assert result.is_idempotent
```

The result compares nested shape, aggregate counts, action counts and paths,
surrogate fingerprints, and global or event-level policy fingerprints. A
non-idempotent result exposes `non_idempotent_paths` and structured
differences grouped by dimension.

## Privacy properties

Reports include schema paths, scalar kinds, array lengths, counts, safe action
names, and SHA-256 fingerprints only. Source values, replacement values, and
unknown action or policy names are not copied into JSON, Markdown, `repr`, or
exceptions. Surrogate fingerprints are equality evidence, not a claim of
cryptographic anonymization. The checker accepts only in-memory or local JSON
inputs and makes no mandatory network call.

Fixtures for FHIR-shaped resources and OMOP-shaped tables should remain
synthetic and offline. The checker does not validate clinical semantics or
replace a formal privacy review.
