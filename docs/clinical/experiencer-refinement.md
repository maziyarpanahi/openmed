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
| `family` | a relative of the patient | mother, father, sibling, maternal, family history, FHx |
| `other` | a non-patient, non-relative subject | donor, roommate, partner, coworker |

This axis is a hard safety boundary for coreference-style aggregation: a
family-member or other-subject finding must not be merged into the patient
record.

## Refining assertions

`refine_experiencer(spans, context_result, text=...)` accepts clinical spans and
an existing `ClinicalContextResult` or `ClinicalAssertion`, then returns
`RefinedExperiencerAssertion` records. Each record preserves the incoming
temporality, certainty, and negation axes while attaching the refined
experiencer to the assertion.

```python
from openmed.clinical import ClinicalAssertion, refine_experiencer

text = "The patient's mother has type 2 diabetes"
span = {"start": text.index("diabetes"), "end": len(text), "label": "CONDITION"}
context = ClinicalAssertion(temporality="recent", certainty="certain")

[result] = refine_experiencer([span], context, text=text)
result.assertion.experiencer
# 'family'
result.assignment.cue
# 'mother'
```

The lower-level `resolve_experiencer(text, span, *, section_experiencer=None)`
API returns only the assignment provenance for one span.

## Resolving a span

```python
from openmed.clinical import resolve_experiencer

text = "The patient's mother has type 2 diabetes"
span = {"start": text.index("diabetes"), "end": len(text), "label": "CONDITION"}

result = resolve_experiencer(text, span)
# ExperiencerAssignment(experiencer='family', cue='mother',
#                       cue_offset=(14, 20), source='cue')
```

`resolve_experiencer(text, span, *, section_experiencer=None)` returns an
`ExperiencerAssignment`:

| Field | Meaning |
|---|---|
| `experiencer` | `patient`, `family`, or `other`. |
| `cue` | The matched subject cue (lowercased), or `None` when defaulted. |
| `cue_offset` | Half-open offsets of the cue in the source text, or `None`. |
| `source` | `cue`, `section`, or `default` -- how the value was decided. |

## Resolution order

1. **Cue** -- the subject cue nearest the span, within the span's clause, wins.
   Resolution is scoped by sentence punctuation (`.`, `!`, `?`, `;`) and the
   contrastive subject-switch markers `but`, `however`, and `whereas`, so a
   subject named in a previous clause does not leak across the boundary.
   Coordinating `and` and `or` deliberately remain inside the same scope because
   they commonly join findings about one subject. On a tie the more specific
   `other` subject wins over `family`.
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
