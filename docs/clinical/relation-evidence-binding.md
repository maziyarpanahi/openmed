# Relation evidence binding

Higher-risk relation aids must be independently reviewable. Before a relation
enters a summary or review workflow, bind it to:

- a directed head and tail source span;
- at least one source span for the linking evidence;
- an explicit assertion state; and
- an identifier for the source document.

`openmed.clinical.relations.evidence_binding` provides this boundary as a
deterministic, local-only adapter. It does not infer an assertion state or
invent evidence when a producer omits either field.

## Bind a candidate

```python
from openmed.clinical.relations.evidence_binding import (
    bind_relation_evidence,
)

candidate = {
    "relation_type": "medication_change",
    "document_id": "synthetic-relation-document",
    "head": {"start": 0, "end": 8, "label": "MEDICATION"},
    "tail": {"start": 9, "end": 17, "label": "PROBLEM"},
    "evidence_spans": [{"start": 18, "end": 25, "label": "CONTEXT_CUE"}],
    "assertion_state": "affirmed",
    "score": 0.91,
}

bound = bind_relation_evidence(candidate)
payload = bound.to_dict()
```

The returned `GuardedRelation` contains only relation codes, offsets, controlled
assertion metadata, confidence, and opaque document/span identifiers. Raw
source text and arbitrary candidate metadata are not copied. A raw document
identifier is domain-separated and hashed before it is stored; an existing
`sha256:` or `hmac-sha256:` identifier is preserved.

Existing relation objects are accepted too. For example, a
`DocumentLevelRelation` can supply its `head`, `tail`, `score`, and
`evidence_sentence_offsets`; its assertion state must still be supplied
explicitly when it is not already present.

## Gate workflow input

Use `require_guarded_relations()` at the boundary of a summary or review
workflow. It rejects raw or incomplete records and returns a deterministic
ordering:

```python
from openmed.clinical.relations.evidence_binding import (
    require_guarded_relations,
)

review_records = require_guarded_relations((bound,), workflow="review")
summary_records = require_guarded_relations((bound,), workflow="summary")
```

Both workflow paths retain `requires_clinician_review=True` and
`autonomous_decision=False`. `EvidenceBindingError` messages are fixed field
or contract categories and never echo submitted identifiers, labels, or source
values.

The module uses only the Python standard library plus OpenMed's existing label
normalizer, performs no mandatory network call, and is an assistive provenance
guard rather than a diagnosis, treatment decision, compliance certification, or
clinical-device guarantee.
