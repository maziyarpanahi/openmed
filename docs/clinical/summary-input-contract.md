# Guarded summary-input contract

`openmed.clinical.summary_input` is the local, fail-closed boundary before a
post-de-identification summary stage. It admits only structured evidence that
has all of the following:

- an allowlisted `evidence_type`;
- an opaque `source_ref` token, with optional character offsets;
- a canonical `sha256:<64 lowercase hex characters>` `policy_fingerprint`;
- an approved review status (`approved`, `reviewed`, or `verified`); and
- optional scalar summary fields from the small allowlist in the module.

The contract does not inspect, copy, or resolve source text. Raw fields such as
`text`, `raw_text`, `surface`, `display`, `identifier`, `payload`, and `value`
are rejected. Use a source token such as
`source:synthetic-note-001` or a SHA-256 reference; never use note text or a
patient identifier as a source reference.

## Validate before generation

```python
from openmed.clinical.summary_input import validate_summary_input

policy_fingerprint = "sha256:" + ("a" * 64)
result = validate_summary_input(
    [
        {
            "evidence_type": "structured_fact",
            "source_ref": "source:synthetic-note-001",
            "policy_fingerprint": policy_fingerprint,
            "review_status": "approved",
            "fields": {"category": "problem", "count": 1},
        }
    ],
    policy_fingerprint=policy_fingerprint,
)

result.require_valid()
guarded_evidence = result.evidence
```

`result.to_dict()` and `result.to_json()` are counts-only validation reports.
They include a fixed set of rejection categories, accepted/rejected counts, and
the validity flag; they do not include evidence values or source references.
Use `guard_summary_input` or `SummaryInputContract.require_valid` when the
summary stage should receive no partial input if any item is rejected.

Rejection counts are deterministic and use these stable categories:

| Category | Meaning |
| --- | --- |
| `invalid_container` | The batch or item is not a supported typed/mapping envelope. |
| `missing_evidence_type` / `unknown_evidence_type` | The evidence type is absent or not allowlisted. |
| `missing_source_reference` / `invalid_source_reference` | Provenance is absent or not an opaque token/offset envelope. |
| `missing_policy_fingerprint` / `invalid_policy_fingerprint` | The policy binding is absent or not canonical SHA-256. |
| `policy_fingerprint_mismatch` | The item is not bound to the configured policy. |
| `missing_review_status` / `unverified_review_status` | Review approval is absent or not approved. |
| `raw_field` | A free-text, raw, identifier, or otherwise unapproved field was supplied. |
| `invalid_safe_field` | A permitted field has a non-scalar or invalid value. |

The implementation is deterministic and performs no mandatory network call. It
is an assistive data-integrity boundary, not a clinical decision, compliance
certification, or substitute for qualified clinical judgment.
