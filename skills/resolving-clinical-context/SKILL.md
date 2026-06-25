---
name: resolving-clinical-context
description: "Assign negation, temporality, and uncertainty (the ConText axes) to clinical entities extracted by OpenMed, so \"denies chest pain\" is not counted as chest pain and \"history of MI\" is not counted as an active MI. Use after NER when the user needs assertion status, negation detection, family-history / hypothetical / historical flags, or ConText/NegEx-style classification before grounding entities to FHIR or a problem list. Covers openmed.clinical.resolve_negation / resolve_temporality / resolve_uncertainty / resolve_span_context / assert_context_axes, ClinicalAssertion, and the AFFIRMED/NEGATED, RECENT/HISTORICAL/HYPOTHETICAL, CERTAIN/UNCERTAIN constants. Pairs after extracting-clinical-entities."
license: Apache-2.0
metadata:
  project: OpenMed
  category: clinical-nlp
  pairs: after
  version: "1.0"
---

# Resolving clinical context

NER finds *that* a condition was mentioned; it does not tell you whether the
patient **has** it. "Patient denies chest pain," "history of MI," and "rule out
PE" all surface entities that must **not** be recorded as active, present
findings. OpenMed's `openmed.clinical` ConText layer assigns three deterministic
axes to each span — **negation**, **temporality**, **uncertainty** — turning raw
mentions into clinically faithful assertions before they reach a problem list or
FHIR Condition.

## When to use

- Immediately after `extracting-clinical-entities`, before grounding,
  problem-list building, or analytics.
- The user asks for assertion status, negation handling, "is this affirmed?",
  family-history vs. patient, historical vs. active, or hedged/uncertain findings.
- You are about to map entities to FHIR `verificationStatus` /`clinicalStatus`
  and need the upstream signal.

## Quick start

```python
import openmed
from openmed.clinical import (
    resolve_span_context, assert_context_axes,
    NEGATED, HISTORICAL, HYPOTHETICAL, UNCERTAIN,
)

note = "Patient denies chest pain. History of MI. Concern for PE; rule out DVT."

# 1) Extract entities (registry key, HF id, or local path).
ents = openmed.analyze_text(note, model_name="disease_detection_superclinical",
                            output_format="dict")

# 2) Assign ConText axes per entity. Pass the span text plus a window of cues.
for e in ents:
    span = e["word"]                      # entity surface text
    window = note                         # full sentence/note as modifier context
    ctx = resolve_span_context(span, window)
    print(span, "->", ctx.negation, ctx.temporality, ctx.certainty)

# "chest pain" -> negated   recent      certain     (do NOT record as present)
# "MI"         -> affirmed  historical  certain     (past, not active)
# "PE"         -> affirmed  recent      uncertain   (hedged; flag, don't drop)
```

`resolve_span_context` returns a `ClinicalContextResult(negation, temporality,
certainty)`. For a downstream-grounding-shaped record use `assert_context_axes`,
which returns a `ClinicalAssertion` with a `.to_dict()` that omits unset axes.

## Workflow

1. **Get entities and their context window.** From `analyze_text`, take each
   entity's surface text and the surrounding sentence (or the whole short note)
   as the modifier window. The ConText helpers accept a string, a span mapping
   with a `text`-like key, or any object exposing `.text`, plus optional
   `modifier_hits`.
2. **Resolve negation** with `resolve_negation(span, window)` → `AFFIRMED` or
   `NEGATED`. It uses a NegEx/ConText cue lexicon ("denies," "no evidence of,"
   "without," "negative for"), masks **pseudo-negation** ("not ruled out,"
   "cannot be excluded") so those don't refute the concept, and counts true cues
   with even/odd parity so double-negation is deterministic.
3. **Resolve temporality** with `resolve_temporality(span, window)` → `RECENT`
   (default), `HISTORICAL` ("history of," "h/o," "s/p," "resolved," "PMH"), or
   `HYPOTHETICAL` ("if," "should," "in case of"). A conditional span is treated
   as hypothetical even if a historical cue is also present.
4. **Resolve uncertainty** with `resolve_uncertainty(span, window)` → `CERTAIN`
   or `UNCERTAIN` ("concern for," "suspicious for," "rule out," "probable,"
   "vs," "r/o"). Uncertain spans are **flagged, not dropped**.
5. **Apply the axes downstream.** Drop or refute `NEGATED` spans; route
   `HISTORICAL` to inactive/resolved status; do not record `HYPOTHETICAL` spans
   as present; mark `UNCERTAIN` spans provisional. Use the constants, not string
   literals, so a vocabulary change doesn't silently break comparisons.

## Hand-off to / from OpenMed

- **From** `extracting-clinical-entities`: this skill consumes `analyze_text`
  Disease/Finding entities. Without context resolution, every mention — including
  negated and historical ones — would be (wrongly) treated as present.
- **OpenMed calls:** `from openmed.clinical import resolve_negation,
  resolve_temporality, resolve_uncertainty, resolve_span_context,
  assert_context_axes, ClinicalAssertion` and the `NEGATED/AFFIRMED`,
  `HISTORICAL/RECENT/HYPOTHETICAL`, `CERTAIN/UNCERTAIN` constants.
- **To** `reconciling-problem-lists`: feed each entity plus its
  `ClinicalContextResult` so active vs. resolved vs. historical is decided
  correctly and negated mentions are excluded.
- **To FHIR grounding:** `negation=negated` → `verificationStatus=refuted`;
  `temporality=historical` → inactive/resolved `clinicalStatus`;
  `certainty=uncertain` → `verificationStatus=provisional`. The layer emits the
  axis; it does not build the FHIR record.

## Edge cases & gotchas

- **Window scoping matters.** Pass a sentence-sized window, not the whole
  document — a negation cue three sentences away should not flip an affirmed
  finding. Segment first (`segmenting-clinical-sections`) for long notes.
- **Pseudo-negation is handled, double-check anyway.** "Cannot exclude PE" is
  affirmed-but-uncertain, not negated. The negation layer masks these cues; the
  uncertainty layer is what flags the hedge.
- **Experiencer (family history) is a separate axis.** These helpers cover
  negation/temporality/uncertainty; "mother with breast cancer" being about a
  relative is the experiencer axis and is out of scope here — handle it before
  attributing the finding to the patient.
- **Deterministic, not ML.** ConText is a rule layer: fast, transparent,
  auditable — but cue-list bound. Novel phrasings may need lexicon tuning; it
  will not infer assertion from semantics the way a model might.
- **Advisory only.** Outputs are annotations for review and downstream grounding,
  not autonomous clinical decisions.

## Standards & references

- Chapman et al., *NegEx* — A simple algorithm for negation in discharge
  summaries (2001): https://doi.org/10.1006/jbin.2001.1029
- Harkema et al., *ConText* — negation, experiencer, temporality, certainty
  (2009): https://doi.org/10.1016/j.jbi.2009.05.002
- HL7 FHIR R4 Condition — `clinicalStatus` / `verificationStatus`:
  https://hl7.org/fhir/R4/condition.html
