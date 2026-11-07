# Guarded clinical relation review yield

`openmed.eval.relation_review_yield` measures whether surfaced guarded
clinical-relation candidates produce useful human-review outcomes. It reports
accepted, corrected, rejected, duplicate, and deferred counts for each
relation class, plus aggregate rates that make review load visible.

The evaluator is deterministic and local-only. It does not load a model, call a
terminology service, rank reviewers, or transmit data.

## Example

```python
from openmed.eval.relation_review_yield import compute_relation_review_yield

report = compute_relation_review_yield(
    [
        {"relation_class": "TREATS", "disposition": "accepted"},
        {"relation_class": "TREATS", "disposition": "corrected"},
        {"relation_class": "TREATS", "disposition": "duplicate"},
        {"relation_class": "CAUSES", "disposition": "deferred"},
    ]
)

print(report.to_dict()["by_relation_class"]["TREATS"])
```

The report contains only counts, controlled relation-class labels, and
derived rates. Rich candidate-like mappings are projected onto
`relation_class` and a controlled disposition; source text, endpoint values,
case identifiers, reviewer identities, confidence scores, and arbitrary
metadata are discarded before aggregation.

## Denominators

Each relation class has five mutually exclusive dispositions:

| Disposition | Meaning |
|---|---|
| `accepted` | The candidate was accepted without correction. |
| `corrected` | The reviewer retained the candidate after correcting it. |
| `rejected` | The reviewer rejected the candidate. |
| `duplicate` | The candidate was redundant with another candidate. |
| `deferred` | The candidate was surfaced but not completed in this review window. |

`review_yield` is:

```text
(accepted + corrected) / (accepted + corrected + rejected + duplicate)
```

Deferred candidates are excluded from that completed-review denominator. Use
`surface_yield` to see the useful-outcome fraction across every surfaced
candidate, including deferred work:

```text
(accepted + corrected) / (accepted + corrected + rejected + duplicate + deferred)
```

The report also includes `duplicate_rate`, `deferred_rate`, and the individual
disposition counts by relation class. These measures describe review workload;
they do not certify relation quality, clinical safety, or compliance.

## Privacy and review safeguards

Keep relation class labels controlled and non-identifying. Do not put case
content, patient identifiers, endpoint values, or reviewer identities in
relation class or disposition fields. The report has no reviewer dimension and
does not rank people. It is an aggregate aid for qualified human review, not an
autonomous clinical decision or a compliance certification.
