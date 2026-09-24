# SDOH temporal qualifiers

`openmed.clinical.sdoh_temporal` adds a small, deterministic temporal layer for
social-determinants evidence. It is an evidence annotation and review aid, not
a diagnosis, clinical decision, compliance certification, or medical-device
control.

## Four controlled classes

The qualifier is one of:

| Class | Meaning |
| --- | --- |
| `current` | An explicit cue places the evidence in the current/present period. |
| `historical` | An explicit cue places the evidence in a previous period. |
| `future` | An explicit cue places the evidence in a planned or future period. |
| `unknown` | No explicit cue is available, or multiple classes conflict. |

Present-tense wording alone does not imply `current`. Missing timing is
returned as `unknown` and marked for human review so a past housing or
employment concern is not silently promoted to a current need.

## Qualify existing evidence

Pass source offsets from `SDOHFinding` or another mapping/object with a
`span`, `source_offsets`, or `start`/`end` pair:

```python
from openmed.clinical.sdoh import SDOHFinding
from openmed.clinical.sdoh_temporal import qualify_sdoh_evidence

note = "Formerly had synthetic housing instability."
start = note.index("synthetic housing")
finding = SDOHFinding(
    category="housing",
    value="instability",
    status=None,
    extent=None,
    temporality=None,
    span=(start, start + len("synthetic housing")),
    score=1.0,
)

[evidence] = qualify_sdoh_evidence(note, [finding])
assert evidence.temporal_class == "historical"
assert evidence.source_offsets == finding.span
```

The returned `SDOHTemporalEvidence` record contains the finding offsets,
temporal-cue offsets, controlled class, conflict classes, and
`review_required`. It never contains the source text, SDOH value, or cue
text. `to_dict()` and `to_json()` are therefore safe metadata views for an
audit artifact; retain the protected source separately under the caller's
privacy and access controls.

If the same local sentence contains incompatible cues, the result is
`unknown`, lists the conflicting classes, and requires review. Cue matching is
bounded to the finding's sentence and does not cross `but`, `however`,
`although`, or `yet` clause boundaries. Results are sorted by source offset and
do not use the wall clock, a model download, or a network service.

The output is advisory. A qualified reviewer must decide whether an SDOH
finding should be acted on or represented in a downstream clinical workflow.
