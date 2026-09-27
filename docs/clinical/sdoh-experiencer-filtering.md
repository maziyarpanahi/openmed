# SDOH Experiencer Filtering

Social-history notes can describe the patient, a household member, or a family
member in the same sentence. Treating every nearby SDOH finding as a patient
finding can therefore put another person's information into the patient's
record.

`openmed.clinical.sdoh_experiencer` applies deterministic local cue rules to
candidate SDOH spans. It recognizes four controlled classes:

| Class | Meaning | Patient-level output |
|---|---|---|
| `patient` | The finding is about the patient. | Included. |
| `household` | A roommate, caregiver, partner, or other household subject. | Excluded. |
| `family` | A relative or family-history subject. | Excluded. |
| `unknown` | No safe subject attribution, an explicit unknown, or conflicting cues. | Excluded and usually queued for review. |

The default without a local cue or section prior is `unknown`. A caller can
provide section spans so the existing section conventions remain available:
Social History supplies a patient prior and Family History supplies a family
prior. A local household or family cue always overrides a patient section
prior.

## Classify and filter

The classifier accepts existing `SDOHFinding` objects or any offset-bearing
candidate mapping. Only offsets and controlled labels are returned; candidate
values and source text are not retained.

```python
from openmed.clinical import filter_sdoh_findings

text = (
    "Social History: the patient reports factor alpha. "
    "The roommate reports factor beta."
)
candidates = [
    {"start": text.index("factor alpha"), "end": text.index("factor alpha") + 12},
    {"start": text.index("factor beta"), "end": text.index("factor beta") + 11},
]

result = filter_sdoh_findings(text, candidates)

[(item.experiencer, item.patient_record_eligible) for item in result.patient]
# [('patient', True)]
[(item.experiencer, item.exclusion_reason) for item in result.excluded]
# [('household', 'non-patient experiencer')]
```

`result` is also iterable as `(patient_evidence, excluded_evidence)`. Each
metadata record has an `input_index`, source offsets, cue offsets, the
classification source, and a review flag. This lets an authorized caller join
the patient-level offsets back to in-memory findings without putting source
values in a report. `result.to_json()` is deterministic and value-free.

## Scope and review

Cues are restricted to the candidate's sentence/clause. Sentence punctuation
and contrastive markers such as `but`, `however`, and `whereas` stop a subject
cue from reaching a separate finding. If distinct subject classes govern one
candidate, the result is `unknown`, records the controlled conflicting classes,
and requires review. The classifier does not call a model, consult the network,
or infer a clinical decision.

This is an assistive filtering aid, not a diagnosis, compliance certification,
or substitute for qualified clinical review. Review the original document in a
separately authorized workflow before using excluded or uncertain evidence.
