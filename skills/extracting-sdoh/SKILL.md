---
name: extracting-sdoh
description: "Extracts social determinants of health (SDOH) — housing instability, food insecurity, unemployment, transportation barriers, social isolation, financial strain — from clinical narrative and maps the spans to ICD-10-CM Z-codes (Z55–Z65). Use after running OpenMed NER when the user wants SDOH surfacing, Z-code suggestion, health-equity analytics, or to recover SDOH that is documented in free text but not coded. Pairs with OpenMed analyze_text output. Standards: ICD-10-CM Z55–Z65, Gravity Project value sets, n2c2 2022 SDOH track. Trigger keywords: SDOH, social determinants, Z-codes, housing, food insecurity, health equity, Gravity Project."
license: Apache-2.0
metadata:
  project: OpenMed
  category: clinical-nlp
  pairs: after
  version: "1.0"
---

# Extracting SDOH and Mapping to ICD-10-CM Z-Codes

Social determinants of health (SDOH) — the conditions in which people live,
work, and age — drive an estimated 80% of health outcomes, yet they live almost
entirely in free-text narrative. Multiple chart-review studies find SDOH
**documented in notes but coded with a Z-code under ~2% of the time**. The
information is there; the structured signal is not. This skill recovers it: run
OpenMed NER over de-identified notes, then map the resulting spans to the
ICD-10-CM **Z55–Z65** family.

## When to use

- A note clearly describes a social risk ("lives in her car", "skips meals to
  afford insulin", "no ride to dialysis") and you want a coded, queryable signal.
- You are building health-equity dashboards, risk stratification, or
  closed-loop referral feeds and need SDOH as discrete data.
- You want to reconcile what the chart *says* against what was *coded*, and flag
  Z-code gaps for a coder or care team to confirm.

This is a **decision-support** step. It proposes Z-codes; a human assigns them.
SDOH coding is sensitive — never expose individual SDOH inferences outside the
care/coding workflow, and never feed them to coverage or pricing decisions.

## Quick start

De-identify first, run NER, then map spans to Z-codes:

```python
import openmed
from sdoh_zcode_map import SDOH_ZCODES  # see references/sdoh_zcode_map.md

note = (
    "62F with CHF. Reports she lost her apartment last month and is "
    "staying in a shelter. Often runs out of food before month-end. "
    "No car; misses appointments because the bus does not run to clinic."
)

# 1) Strip PHI before any downstream processing or storage.
deid = openmed.deidentify(note, method="replace", policy="hipaa_safe_harbor")

# 2) Run clinical NER. Use an SDOH/clinical model from the registry; discover
#    available keys with openmed.get_models_by_category(...).
result = openmed.analyze_text(deid.text, output_format="dict")

# 3) Map each entity span to a candidate Z-code.
for ent in result["entities"]:
    code = SDOH_ZCODES.get(ent["label"].lower())
    if code:
        print(f"{ent['text']!r:40}  {ent['label']:18}  -> {code}")
```

`analyze_text` returns entities shaped as
`{"text", "label", "confidence", "start", "end", "metadata"}`. The `start`/`end`
offsets index into the text you passed in, so you can anchor every suggested
Z-code back to its exact source span for human review.

## Workflow

1. **De-identify** the note with `openmed.deidentify` (HIPAA Safe Harbor or a
   stricter policy). SDOH text is dense with PHI (addresses, employer names).
2. **Extract entities** with `openmed.analyze_text`. Pick a model whose label
   set covers social concepts; if your model only emits clinical findings, run a
   second pass with a zero-shot model (`openmed zero`) using SDOH labels such as
   `housing_instability`, `food_insecurity`, `unemployment`,
   `transportation_barrier`, `social_isolation`, `financial_strain`.
3. **Map spans to Z-codes** using a curated lookup keyed by label
   (`references/sdoh_zcode_map.md`). Keep the **span offsets** and the model
   `confidence` on every suggestion.
4. **Stage for confirmation.** Emit `(span, label, suggested_code, confidence)`
   tuples for a coder or the Gravity Project pipeline to accept or reject. Do not
   auto-bill a Z-code from an inference alone.
5. **Normalize to value sets.** Align labels to the **Gravity Project** SDOH
   domains so codes are interoperable with FHIR (`Condition`, `Observation`,
   `Goal`) and USCDI v3 SDOH elements.

### Z-code families you will hit most (ICD-10-CM Z55–Z65)

| Domain | Range | Example |
| --- | --- | --- |
| Education / literacy | Z55 | Z55.0 illiteracy |
| Employment | Z56 | Z56.0 unemployment |
| Occupational exposure | Z57 | — |
| Housing / economic | Z59 | Z59.0 homelessness, Z59.41 food insecurity, Z59.82 transportation insecurity |
| Social environment | Z60 | Z60.2 living alone, Z60.4 social exclusion |
| Upbringing | Z62 | — |
| Family / support circumstances | Z63 | Z63.4 disappearance/death of family member |
| Psychosocial circumstances | Z64–Z65 | Z65.1 imprisonment |

The full curated label→code table lives in
[references/sdoh_zcode_map.md](references/sdoh_zcode_map.md).

## Hand-off to / from OpenMed

- **From OpenMed:** this skill consumes `openmed.analyze_text(...)` output
  (`PredictionResult` dict). Each `entity["start"]/["end"]` anchors a Z-code
  suggestion to source text.
- **To OpenMed:** always run `openmed.deidentify` upstream so no raw PHI reaches
  the SDOH store, logs, or coder queue.
- **Onward:** emit suggestions into a FHIR `Condition`/`Observation` with the
  Z-code as `code.coding` (system `http://hl7.org/fhir/sid/icd-10-cm`). OpenMed's
  `openmed.clinical.exporters.fhir` helpers (`to_bundle`, `to_operation_outcome`)
  assemble the envelope; ICD-10-CM itself is public-domain in the US release.

## Edge cases & gotchas

- **Negation and history.** "Denies food insecurity" or "previously homeless,
  now housed" must not produce an active Z-code. Run negation/temporality
  resolution (`openmed.clinical`, `resolving-clinical-context`) before mapping.
- **Hypotheticals and screening prompts.** Template text ("Do you have stable
  housing?") and family-member SDOH ("his mother is unhoused") are common false
  positives — check the subject and modality.
- **One span, one domain.** Do not stack multiple Z-codes onto one phrase; map
  to the most specific single code and let the coder add others.
- **Granularity drift.** ICD-10-CM adds SDOH codes most fiscal years (e.g.
  Z59.4x food, Z59.82 transportation). Pin your code set to a release year and
  re-validate annually.
- **Do not infer protected attributes.** Surface only what the note states;
  never derive race, immigration status, or income bracket as an SDOH "finding".
- **Restricted terminology.** SNOMED CT SDOH refsets and LOINC SDOH panels are
  licensed separately — OpenMed does not bundle them; load the user's own copy
  out-of-process if you cross-map beyond ICD-10-CM.

## Standards & references

- ICD-10-CM official guidelines, Z55–Z65 SDOH codes (CDC/CMS, public domain):
  https://www.cdc.gov/nchs/icd/icd-10-cm.htm
- Gravity Project (HL7 SDOH Clinical Care value sets & FHIR IG):
  https://www.hl7.org/gravity/ and https://confluence.hl7.org/display/GRAV
- n2c2 2022 Track 2 — SDOH extraction shared task (Social History Annotation
  Corpus): https://n2c2.dbmi.hms.harvard.edu/
- CMS ICD-10-CM Z-code SDOH resources:
  https://www.cms.gov/files/document/zcodes-infographic.pdf
- USCDI SDOH data classes: https://www.healthit.gov/isa/uscdi-data-class/sdoh
