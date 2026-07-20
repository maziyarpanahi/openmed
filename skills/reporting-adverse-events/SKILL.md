---
name: reporting-adverse-events
description: "Structures adverse-event mentions that OpenMed extracts into FAERS / ICH E2B(R3) reportable fields — suspect drug, reaction (MedDRA PT), seriousness criteria, and outcome. Use when the user needs to build an individual case safety report (ICSR), populate a FAERS submission, map a narrative to E2B(R3) data elements, classify seriousness (death, life-threatening, hospitalization, disability, congenital anomaly), or assign reaction outcomes. Trigger keywords: adverse event, ADR, ICSR, FAERS, E2B, E2B(R3), suspect drug, seriousness, MedDRA, reaction outcome, pharmacovigilance case. Pairs after OpenMed NER: consume Pharmaceutical/Chemical and Disease entities from openmed.analyze_text. MedDRA is licensed and user-supplied — never bundled. De-identify the narrative with openmed.deidentify before any external submission."
license: Apache-2.0
metadata:
  project: OpenMed
  category: safety-pharmacovigilance
  pairs: after
  version: "1.0"
---

# Reporting adverse events into FAERS / ICH E2B(R3)

A pharmacovigilance case starts as free-text narrative ("68 yo on warfarin
developed GI bleed, hospitalized"). To make it reportable you must structure it
into the **ICH E2B(R3)** data elements that the FDA's **FAERS** (and EMA's
EudraVigilance) expect: a **suspect drug**, one or more **reactions** coded to
**MedDRA** Preferred Terms, **seriousness** criteria, and a **reaction outcome**.

OpenMed extracts the drug and condition spans on-device; this skill turns those
spans plus the narrative into the E2B(R3) skeleton. The reaction coding step
needs **MedDRA**, which is **licensed by the MSSO and user-supplied** — it is
never bundled with OpenMed and must be loaded from the user's own subscription.

## When to use

- A narrative names a drug and an adverse reaction and you need an ICSR
  (Individual Case Safety Report) shell with the right E2B(R3) fields.
- You must classify **seriousness** (E2B sections C.1.7 / E.i.3) — death,
  life-threatening, hospitalization/prolongation, disability, congenital
  anomaly, or "other medically important condition".
- You need to characterize each drug as **suspect / concomitant / interacting**
  (the `drugcharacterization` axis FAERS uses).
- You are pre-filling a 3500A / FAERS electronic submission or staging cases for
  a safety database.

This skill produces a **structured draft for human safety review** — it does not
file reports or perform causality assessment autonomously.

## Quick start

```python
import openmed

narrative = (
    "68-year-old patient on warfarin 5 mg daily developed a gastrointestinal "
    "hemorrhage and was hospitalized. Warfarin was discontinued; the patient "
    "recovered."
)

# 1) Extract drug spans (Pharmaceutical category) on-device.
drugs = openmed.analyze_text(
    narrative,
    model_name="pharma_detection_superclinical",
    output_format="dict",
)["entities"]

# 2) Extract condition / reaction spans (Disease category).
conditions = openmed.analyze_text(
    narrative,
    model_name="disease_detection_superclinical",
    output_format="dict",
)["entities"]

# 3) Assemble an E2B(R3)-shaped ICSR skeleton (reaction PTs filled later via MedDRA).
icsr = {
    "patient": {"age": None, "sex": None},          # from de-identified demographics
    "drugs": [
        {
            "name": e["text"],
            "drugcharacterization": 1,              # 1=suspect 2=concomitant 3=interacting
            "action": None,                          # e.g. drug withdrawn / dose reduced
        }
        for e in drugs
    ],
    "reactions": [
        {
            "verbatim": e["text"],                   # narrative term, pre-MedDRA
            "meddra_pt": None,                       # coded with user's MedDRA dict
            "outcome": None,                         # E2B reaction outcome code
        }
        for e in conditions
    ],
    "seriousness": {
        "serious": None, "death": False, "lifeThreatening": False,
        "hospitalization": True, "disability": False, "congenitalAnomaly": False,
        "otherMedicallyImportant": False,
    },
}
```

## E2B(R3) seriousness and outcome value sets

Seriousness is a set of boolean criteria (E2B E.i.3.2). A case is **serious** if
*any* criterion is true:

| Criterion | E2B element | FAERS field |
| --- | --- | --- |
| Death | E.i.3.2a | `seriousnessdeath` |
| Life-threatening | E.i.3.2b | `seriousnesslifethreatening` |
| Hospitalization / prolonged | E.i.3.2c | `seriousnesshospitalization` |
| Disability / incapacity | E.i.3.2d | `seriousnessdisabling` |
| Congenital anomaly | E.i.3.2e | `seriousnesscongenitalanomali` |
| Other medically important | E.i.3.2f | `seriousnessother` |

Reaction outcome (E2B E.i.7) is a coded value: `1` recovered/resolved,
`2` recovering/resolving, `3` not recovered/not resolved, `4` recovered with
sequelae, `5` fatal, `6` unknown.

Drug characterization (E2B G.k.1): `1` suspect, `2` concomitant, `3` interacting.

## Workflow

1. **De-identify first.** Run `openmed.deidentify(narrative, policy=...)` and
   work from `result.deidentified_text`. Patient name, MRN, and dates must be
   removed/shifted before the case leaves your environment.
2. **Extract drugs and reactions** with the two `analyze_text` calls above.
   Keep each entity's `start`/`end` offsets for traceability.
3. **Characterize each drug** as suspect (`1`), concomitant (`2`), or
   interacting (`3`). The drug that temporally precedes the reaction and was
   acted upon (withdrawn/reduced) is usually the suspect.
4. **Code reactions to MedDRA.** Map each verbatim reaction term to a MedDRA
   **Preferred Term (PT)** and its System Organ Class using the user's licensed
   MedDRA dictionary (see "Edge cases"). Never invent PTs.
5. **Determine seriousness.** Scan the narrative for the six criteria; set
   `serious=True` if any is met. "Hospitalized", "admitted", "ICU" → C.1.7c.
6. **Assign reaction outcome** from the value set above.
7. **Hand the structured draft to a qualified safety reviewer** for causality
   (e.g. WHO-UMC or Naranjo), expectedness, and final submission.

## Hand-off to / from OpenMed

OpenMed's `analyze_text` returns a `dict`; `result["entities"]` is a list whose
items carry `text`, `label`, `confidence`, `start`, `end`. Consume them:

- **From** `extracting-clinical-entities`: Pharmaceutical entities →
  `icsr["drugs"]`; Disease entities → `icsr["reactions"]`. Keep offsets so each
  E2B field is traceable to the source span.
- **From** `normalizing-rxnorm`: optionally attach an RxCUI to each suspect drug
  for product identification (E2B G.k.2.2) before coding.
- **De-identify** with `deidentifying-clinical-text` (`openmed.deidentify`)
  **before** the case is exported or transmitted to any safety database.
- **To** `detecting-pv-signals`: aggregated, coded cases feed disproportionality
  analysis. **To** `querying-openfda-labels`: confirm the reaction is/ isn't a
  labeled event (expectedness).

## Edge cases & gotchas

- **MedDRA is licensed — never bundle it.** MedDRA is distributed by the MSSO
  under subscription; OpenMed ships none of it. Load PTs/LLTs from the user's own
  MedDRA release (the version is itself a reportable field, E2B C.1.x). Verbatim
  reaction text stays in the case until a coder maps it.
- **One reaction term ≠ one PT.** "GI bleed" maps to the PT *Gastrointestinal
  haemorrhage*; keep the verbatim term alongside the coded PT for the audit
  trail. Multi-word reactions span several OpenMed tokens — reassemble by offset.
- **Suspect vs concomitant matters.** Disproportionality and labeling decisions
  hinge on `drugcharacterization`. Do not default every drug to suspect.
- **Seriousness is OR, not a severity scale.** A mild rash that caused
  hospitalization is *serious*; a severe headache that resolved at home may not
  be. Classify by the six regulatory criteria, not by clinical severity words.
- **Causality is out of scope here.** This skill structures the case; it does not
  assert the drug caused the event. Leave causality to the reviewer.
- **Local-first.** NER and de-identification run on-device. Only de-identified,
  structured case data should reach an external safety database, and only under
  the appropriate regulatory agreement.

## Standards & references

- FDA FAERS overview: https://www.fda.gov/drugs/surveillance/fda-adverse-event-reporting-system-faers
- ICH E2B(R3) ICSR implementation guide: https://www.ich.org/page/efficacy-guidelines (E2B(R3))
- FDA E2B(R3) regional implementation: https://www.fda.gov/industry/fda-data-standards-advisory-board/ich-e2br3-individual-case-safety-report-icsr
- MedDRA (licensed, user-supplied): https://www.meddra.org/
- FDA MedWatch 3500A reporting: https://www.fda.gov/safety/medical-product-safety-information/medwatch-fda-safety-information-and-adverse-event-reporting-program
