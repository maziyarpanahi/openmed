# Clinical Evidence-Span Overlap Audit

`audit_evidence_spans()` produces a deterministic review signal for duplicate,
nested, partially overlapping, and cross-source evidence ranges. It is fully
offline and does not rewrite provenance or select a preferred record.

!!! warning "Assistive review signal only"
    An overlap finding is not a clinical decision, compliance certification, or
    automatic provenance resolution. A human reviewer or an explicit downstream
    policy must decide how to handle each finding.

## Input contract

Pass `EvidenceSpan` values or mappings containing opaque identifiers and
half-open offsets. `source_id` identifies the coordinate space; `evidence_id`
identifies the record in that source.

```python
from openmed.clinical import EvidenceSpan, audit_evidence_spans

spans = [
    EvidenceSpan("synthetic-note", "opaque-a", 10, 20),
    EvidenceSpan("synthetic-note", "opaque-b", 12, 18),
    EvidenceSpan("synthetic-peer", "opaque-c", 15, 22),
]

report = audit_evidence_spans(spans)
report.counts
# {"exact": 0, "nested": 1, "partial": 0, "cross_source": 2}
```

Mappings may use `source_id`/`source`/`document_id`/`doc_id`,
`evidence_id`/`span_id`/`id`, and `start`/`start_offset` plus
`end`/`end_offset` (or the `start_char`/`end_char` aliases). Identifier values
are preserved exactly; all-whitespace identifiers and zero-length ranges are
rejected. Extra fields are ignored, so source text is never copied into the
audit result.

## Classifications

| Classification | Meaning |
| --- | --- |
| `exact` | Same source and identical half-open offsets. |
| `nested` | Same source and one non-identical range contains the other. |
| `partial` | Same source and the ranges overlap without containment. |
| `cross_source` | Numeric ranges overlap but their source identifiers differ. |

Ranges that only touch at an endpoint do not overlap. Cross-source findings
preserve both source/range references; the audit does not assume that equal
numeric offsets refer to the same source text.

## Privacy and reproducibility

`EvidenceSpanAudit.to_dict()` contains only opaque identifiers, numeric offsets,
classifications, counts, and SHA-256 fingerprints. It does not include source
surfaces, raw evidence values, or model output. Mapping fields such as `text`
are ignored, and validation errors do not echo their values.

Input order is normalized before pair generation. `report.fingerprint` is a
stable `sha256:` fingerprint of the normalized spans, findings, and counts;
each finding also has its own fingerprint. The report can be serialized with
`report.to_json()` without a network call.
