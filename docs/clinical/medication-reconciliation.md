# Medication reconciliation confidence

`openmed.clinical.medication_reconciliation` provides deterministic support
for reviewing whether normalized medication candidates from separate documents
can represent one longitudinal medication entry. It does not ground names,
query a vocabulary, select a dose, or make a clinical decision. The caller
must provide the normalized candidates and remains responsible for review.

## Scoring

`score_medication_match(left, right)` compares four independent signals:

| Signal | Default weight | Match behavior |
| --- | ---: | --- |
| Name or shared coded identity | 0.45 | Normalized case and punctuation variants match, or the supplied coding system and code match. |
| Dose | 0.25 | Local dose normalization compares compatible canonical units and amounts. |
| Route | 0.15 | Common route aliases such as `PO` and `oral` are controlled to one value. |
| Temporal evidence | 0.15 | Overlapping windows or the same temporal label match; separated windows receive a deterministic partial score. |

Unknown fields contribute no score. The default threshold is `0.80`, so a
name-only match cannot be silently merged. Known name, code, dose, route, or
overlapping temporal-status conflicts always produce an abstention with stable
reason codes such as `dose_conflict`, `route_conflict`,
`temporal_conflict`, and `insufficient_evidence`.

Dose comparison is limited to equality of caller-supplied normalized values;
this module does not determine whether any dose is appropriate.

```python
from openmed.clinical.medication_reconciliation import (
    reconcile_medications,
    score_medication_match,
)

left = {
    "candidate_id": "synthetic-1",
    "normalized_name": "Synthetic Medication Alpha",
    "dose": "500 mg",
    "route": "PO",
    "event_date": "2026-01-15",
}
right = {
    "candidate_id": "synthetic-2",
    "normalized_name": "synthetic medication alpha",
    "dose": {"value": 0.5, "unit": "g"},
    "route": "oral",
    "event_date": "2026-01-15",
}

decision = score_medication_match(left, right)
assert decision.matched
assert decision.confidence == 1.0

result = reconcile_medications([left, right])
assert len(result.merged_groups) == 1
```

## Conservative grouping and privacy

`reconcile_medications` computes every pair first and merges groups only when
the complete cross-product is compatible. This prevents a transitive chain
from joining two candidates whose regimens conflict. Every rejected pair is
available through `result.abstentions` and its `abstention_reasons`.

Candidates retain normalized values in memory for a review client, but
`MedicationMatchDecision.to_dict()`,
`ReconciledMedicationGroup.to_dict()`, and
`MedicationReconciliationResult.to_dict()` hash candidate identities, names,
doses, and source identifiers. They do not emit raw medication or document
values. The implementation uses only the Python standard library plus
OpenMed's local dose normalizer and makes no mandatory network call.
