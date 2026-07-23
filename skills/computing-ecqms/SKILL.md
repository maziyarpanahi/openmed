---
name: computing-ecqms
description: "Compute electronic clinical quality measures (eCQMs) over structured data using CQL/QDM logic, lifting note-derived numerator and exclusion facts from OpenMed to improve measure capture. Use when the user wants to compute an eCQM, evaluate a CMS/ECQI quality measure, improve numerator capture from clinical notes, build CQL/QDM measure logic, or close documentation gaps that structured codes miss. Covers eCQM structure (IPP/denominator/numerator/exclusions), CQL v1.5 and QDM v5.6, MADiE authoring, and mapping OpenMed entities to QDM data elements. Consumes OpenMed analyze_text facts (coded via the linking skills) to supplement structured EHR data; does not replace certified measure engines."
license: Apache-2.0
metadata:
  project: OpenMed
  category: analytics-reporting
  pairs: after
  version: "1.0"
---

# Computing eCQMs

Electronic Clinical Quality Measures (eCQMs) are computed over structured data
using **CQL** (Clinical Quality Language) logic against the **QDM** (Quality
Data Model). Much of what a measure needs — a counseling note, a reason a
service wasn't done, a symptom — lives only in **free text**. This skill uses
OpenMed to lift those facts out of notes (on-device) and feed them into measure
computation so numerators and valid exclusions aren't undercounted.

## When to use this skill

When structured codes under-capture a measure population and the evidence is in
notes: documented exclusions ("patient declined screening"), numerator-relevant
findings, or symptoms gating a measure. Use it *alongside* a certified measure
engine — OpenMed supplements capture; it does not compute or certify the
measure.

## eCQM anatomy (what you're populating)

| Population | Meaning | Where OpenMed helps |
| --- | --- | --- |
| IPP (Initial Population) | everyone the measure could apply to | usually structured (encounters, age) |
| Denominator | IPP meeting base criteria | mostly structured |
| Denominator Exclusion / Exception | valid reasons to remove from denom | **notes**: "declined", "medical reason", "not indicated" |
| Numerator | met the quality action | **notes**: counseling delivered, advice given, status documented |

## Quick start

```python
import openmed

note = (
    "Tobacco use screened today; patient is a current every-day smoker. "
    "Cessation counseling provided and cessation medication offered."
)

result = openmed.analyze_text(note, output_format="dict")
# entities -> {text, label, confidence, start, end}

# Lift two measure-relevant facts (illustrative, for a tobacco-screening eCQM):
facts = {
    "tobacco_status_documented": any(e["label"] in {"smoking_status", "tobacco_use"}
                                     for e in result["entities"]),
    "cessation_intervention_documented": "counseling" in note.lower(),
}
# These become QDM data elements your CQL references (see workflow).
```

Pick the model whose labels match the measure concept (`choosing-openmed-models`)
and code spans to value-set vocabularies via the linking skills before they
enter QDM.

## Workflow

1. **Read the measure.** Get the human-readable spec + CQL + value sets from
   ECQI / MADiE. Identify which populations depend on documentation that
   structured data misses.
2. **De-identify.** Run `openmed.deidentify` on notes before any logging or
   storage; keep the measure keyed by internal patient ids.
3. **Extract facts.** `openmed.analyze_text` for the concepts the measure needs
   (status, intervention, reason-not-done). Use `resolving-clinical-context` to
   drop negated/hypothetical/family-history mentions — a *negated* exclusion is
   not an exclusion.
4. **Code to value sets.** Map entities to the codes the measure's value sets
   expect (SNOMED/LOINC/RxNorm via the linking skills). QDM data elements are
   defined by code membership, not raw strings.
5. **Materialize QDM data elements.** Turn coded, dated facts into QDM elements
   (e.g. `Assessment, Performed`, `Intervention, Performed`,
   `Diagnosis`) with the right author/relevant dates
   (`building-patient-timelines`).
6. **Compute with CQL.** Feed the structured + note-derived QDM into a
   **certified** CQL engine (e.g. the open-source `cqframework` engine). OpenMed
   does not execute CQL.
7. **Reconcile & audit.** Track which population members were added by
   note-derived facts and at what confidence, so QA can review.

## Hand-off to / from OpenMed

- **From OpenMed:** `analyze_text` entities + `clinical` temporality + the
  linking skills (to land facts in the measure's value sets) + `deidentify`
  upstream.
- **To measure tooling:** materialized QDM data elements feed a CQL engine and
  MADiE test decks. Note-derived QDM can also originate from `etl-to-omop-cdm`
  rows if you compute measures on an OMOP store instead.

## Edge cases & gotchas

- **OpenMed supplements, it does not certify.** Measure scoring must run in a
  validated CQL engine. Treat note-derived facts as additional evidence subject
  to review, not as authoritative measure results.
- **Negation flips meaning.** "Screening declined" is an *exclusion*; "screening
  not declined" / "no contraindication" is the opposite. Always run the
  temporality/negation pass before counting.
- **Dates drive measurement periods.** A fact only counts if its relevant date
  falls in the measurement period. Resolve dates first; undated facts can't be
  placed.
- **Value-set membership, not keywords.** A QDM data element is defined by codes
  in the measure's value set. Map entities to those codes — don't match on the
  surface word.
- **No restricted terminology bundling.** SNOMED/LOINC/RxNorm content stays
  out-of-process under your own license; OpenMed provides spans/labels only.
- **No raw PHI in logs or audit.** Record measure provenance by offset, label,
  confidence, and internal id.

## Standards & references

- ECQI Resource Center (eCQM specs, CMS measures): https://ecqi.healthit.gov/
- CQL (Clinical Quality Language) v1.5 spec: https://cql.hl7.org/
- QDM (Quality Data Model) v5.6: https://ecqi.healthit.gov/qdm
- MADiE (Measure Authoring Development Integrated Environment):
  https://madie.cms.gov/
- Open-source CQL engine (HL7 cqframework): https://github.com/cqframework/clinical_quality_language
- OpenMed source: `openmed/processing/` (`analyze_text`), `openmed.clinical`
  (temporality).
