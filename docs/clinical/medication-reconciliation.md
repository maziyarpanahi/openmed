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
