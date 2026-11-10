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

Category labels are normalized to bounded lowercase tokens and checked against
controlled vocabularies. Unrecognized family, section, and filtering-reason
values are counted as `unknown`, `unsectioned`, and `other`, respectively. This
also applies when reading an aggregate report. Callers should still pass
controlled labels, never source text or identifiers.

## Serialization

`RelationCandidateAuditReport.to_json()` and `to_markdown()` are byte-stable
for the same aggregate input. Use `write_json()` or `write_markdown()` to
persist an artifact; both create the destination's parent directory locally.

The report is an investigation aid only. It does not certify relation quality,
clinical correctness, or a compliance posture.


## Validation limits

Each aggregate dimension must sum to the candidate total; conflicting count
aliases and boolean schema versions are rejected. Typed category records are
normalized again on entry and before serialization. Batch input is bounded to
100000 candidates, imported count maps to 4096 entries, and JSON reads to 1 MiB.
Scalar text is not a candidate batch. Serialization accepts only integer
indentation from zero through eight. Read/write failures expose fixed error
categories without retaining raw decoder or filesystem exception details.
These aggregate counts do not provide a privacy guarantee for small cohorts.
