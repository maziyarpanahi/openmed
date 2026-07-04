# Experiencer Refinement

The ConText experiencer axis records who a clinical finding is about. The shipped
layer resolves it only at the section level: a finding under a *Family History*
heading is attributed to the family. Free text is finer-grained than that -- a
single sentence can switch subject ("the patient's *mother* has diabetes", "the
organ *donor* was CMV-positive").

`openmed.clinical.experiencer` refines the experiencer of a governing span using
local subject cues, distinguishing three subjects:

| Value | Subject | Example cues |
|---|---|---|
| `patient` | the patient (default) | none cued |
| `family` | a relative of the patient | mother, father, sibling, maternal, family history |
| `other` | a non-patient, non-relative subject | donor, roommate, partner, coworker |

This axis is a hard safety boundary for coreference-style aggregation: a
family-member or other-subject finding must not be merged into the patient
record.

## Resolving a span

```python
from openmed.clinical import refine_experiencer

text = "The patient's mother has type 2 diabetes"
span = {"start": text.index("diabetes"), "end": len(text), "label": "CONDITION"}

result = refine_experiencer(text, span)
# ExperiencerAssignment(experiencer='family', cue='mother',
#                       cue_offset=(14, 20), source='cue')
```

`refine_experiencer(text, span, *, section_experiencer=None)` returns an
`ExperiencerAssignment`:

| Field | Meaning |
|---|---|
| `experiencer` | `patient`, `family`, or `other`. |
| `cue` | The matched subject cue (lowercased), or `None` when defaulted. |
| `cue_offset` | Half-open offsets of the cue in the source text, or `None`. |
| `source` | `cue`, `section`, or `default` -- how the value was decided. |

## Resolution order

1. **Cue** -- the subject cue nearest the span, within the span's clause, wins.
   Resolution is scoped to the clause (bounded by `.`, `!`, `?`, `;`), so a
   subject named in a previous sentence does not leak across the boundary. On a
   tie the more specific `other` subject wins over `family`.
2. **Section prior** -- when no cue is found and a `section_experiencer` is
   supplied (for example `family` under a *Family History* heading), it is used.
3. **Default** -- otherwise the span is attributed to the `patient`.

An explicit cue always overrides the section prior, so "the patient's father
also has hypertension" resolves to `family` even inside a patient-default
section.

## Notes

Resolution is deterministic and offline. Experiencer refinement is a cue-based
heuristic and is not a substitute for clinician review; validate before any
clinical use.
