# Medication reconciliation

`reconcile_medications()` collapses medication mentions within one document
into one normalized, reviewable state per medication. It is an assistive
organization layer, not a prescription or clinical decision.

The input can be a mapping, `MedicationMention`, or a grounded span-like
record. Supply a local ingredient or code when available; document-local
coreference chains can provide an additional identity key:

```python
from openmed.clinical import reconcile_medications

records = reconcile_medications(
    [
        {
            "ingredient": "metformin",
            "system": "RXNORM",
            "code": "860975",
            "dose": "500 MG",
            "route": "PO",
            "status": "started",
            "effective_time": "2026-01-01",
            "offset": (10, 19),
        },
        {
            "ingredient": "metformin",
            "system": "RXNORM",
            "code": "860975",
            "dose": "1000 mg",
            "route": "oral",
            "status": "changed",
            "effective_time": "2026-01-15",
            "offset": (82, 91),
        },
    ],
    document_id="synthetic-note-1",
)

record = records[0]
assert record.current_status == "changed"
assert record.current_dose == "1000 mg"
assert record.current_route == "oral"
```

## Identity and ordering

Mentions are grouped using, in order, a supplied coreference entity, an
explicit normalized ingredient, a coded grounding identity, or a normalized
surface fallback. `RXNORM` candidates may be supplied through the existing
grounding record contract. Reconciliation is document-local and never calls a
terminology service by default.

History is ordered by normalized absolute effective timestamps when present.
When timestamps are absent, source offsets provide deterministic document
order. A missing status is conservatively normalized to `continued`. Supported
normalized transitions are `started`, `continued`, `held`, `changed`, and
`stopped`; common start/hold/change/discontinue variants are accepted.

## Dose and route conflicts

The latest normalized timestamp wins when it provides a unique value. At the
same timestamp, section precedence is used (`assessment`/`plan`, then current
medication lists, then narrative history). If conflicting values remain tied,
the current field is `None` and `record.conflicts` contains the normalized
values, field name, and source offsets. Untimestamped disagreements without a
unique section authority are also left unresolved rather than silently merged.

## Privacy and scope

`ReconciledMedication.to_dict()` emits normalized ingredient, dose, route,
status, timestamps, hashes/codes supplied by upstream grounding, and source
offsets. It does not emit source mention text or the source document. The
module does not parse sigs, extract medication relations, reconcile across
documents, or make treatment recommendations.

## Cross-document match confidence

Cross-document candidate scoring is a separate, more conservative API in
`openmed.clinical.medication_reconciliation`. Use
`reconcile_medication_candidates` for that task; the top-level
`reconcile_medications` above retains its document-local contract.

`score_medication_match(left, right)` compares caller-supplied normalized name
or coded identity, dose, route, and temporal evidence. Its default weights are
0.45, 0.25, 0.15, and 0.15 respectively, with a default merge threshold of
0.80. Unknown fields contribute no score. Known identity, dose, route, or
overlapping temporal-status conflicts cause abstention. A name-only match
cannot silently merge. The scorer does not decide whether a dose is clinically
appropriate.

```python
from openmed.clinical import reconcile_medication_candidates, score_medication_match

left = {
    "candidate_id": "synthetic-a",
    "normalized_name": "Synthetic Medication Alpha",
    "dose": "500 mg",
    "route": "PO",
    "event_date": "2026-01-15",
}
right = {
    "candidate_id": "synthetic-b",
    "normalized_name": "synthetic medication alpha",
    "dose": "0.5 g",
    "route": "oral",
    "event_date": "2026-01-15",
}

decision = score_medication_match(left, right)
assert decision.matched
result = reconcile_medication_candidates([left, right])
assert len(result.merged_groups) == 1
```

Candidate grouping checks every cross-pair before merging a group, so a
transitive chain cannot hide a regimen conflict. Rejected pairs remain
reviewable through stable abstention reasons. Serialized audit decisions hash
candidate and source identifiers instead of emitting raw medication or
document values. All processing is local and requires no terminology service.
