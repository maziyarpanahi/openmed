# Relation-candidate audit report

`openmed.eval.relation_audit` produces a deterministic summary of relation
candidate generation and filtering. It reports counts by relation family,
clinical-note section, and filtering reason so extraction failures can be
investigated without retaining candidate details.

The helper is local-only and uses no model, network, or external service. It
reads only category fields from each candidate. Source text, endpoint text,
offsets, scores, candidate identifiers, and unrecognized metadata are not
copied into the report.

## Usage

```python
from openmed.eval.relation_audit import audit_relation_candidates

report = audit_relation_candidates(
    [
        {
            "relation_type": "drug_to_dose",
            "section": "Medications",
            "filtering_reason": "accepted",
        },
        {
            "relation_type": "problem_to_status",
            "section": "Assessment",
            "filtering_reason": "assertion_refuted",
        },
    ]
)

print(report.to_json())
```

The JSON payload contains only aggregate values:

```json
{
  "artifact": "relation_candidate_audit",
  "by_filtering_reason": {
    "accepted": 1,
    "assertion_refuted": 1
  },
  "by_relation_family": {
    "drug": 1,
    "problem": 1
  },
  "by_section": {
    "assessment": 1,
    "medications": 1
  },
  "candidate_count": 2,
  "schema_version": 1
}
```

Typed relation records that expose `relation_type` or `label` are supported;
for labels such as `drug_to_dose`, the family is derived from the prefix
before `_to_`. If a record does not carry a filtering reason, it is counted as
`accepted` unless its explicit status indicates that it was filtered,
refuted, conditional, or uncertain. Missing sections are counted under
`unsectioned`.

Category labels are normalized to bounded lowercase tokens. Callers should
pass controlled labels for the three dimensions, never source text or
identifiers.

## Serialization

`RelationCandidateAuditReport.to_json()` and `to_markdown()` are byte-stable
for the same aggregate input. Use `write_json()` or `write_markdown()` to
persist an artifact; both create the destination's parent directory locally.

The report is an investigation aid only. It does not certify relation quality,
clinical correctness, or a compliance posture.
