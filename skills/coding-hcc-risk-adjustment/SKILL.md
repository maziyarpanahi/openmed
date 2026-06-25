---
name: coding-hcc-risk-adjustment
description: "Maps chronic conditions extracted by OpenMed to CMS-HCC V28 risk-adjustment categories and estimates a RAF (Risk Adjustment Factor) score as decision support. Use when the user wants to surface risk-adjustable diagnoses from notes, map ICD-10-CM codes to HCC categories, estimate or reconcile a patient/panel RAF, find suspected-but-undocumented HCCs, or check MEAT documentation support. Trigger keywords: HCC, CMS-HCC, V28, RAF score, risk adjustment, Medicare Advantage, hierarchical condition category, MEAT, recapture, suspect HCC, RADV. Pairs after OpenMed NER + ICD-10 coding: consume Disease/Pathology entities from openmed.analyze_text, code them (see coding-icd10), then roll up to HCCs. CMS-HCC mappings and weights are public from CMS. This is a coding-support aid for human review, never autonomous risk-adjustment coding."
license: Apache-2.0
metadata:
  project: OpenMed
  category: terminology-coding
  pairs: after
  version: "1.0"
---

# Mapping conditions to CMS-HCC V28 and estimating RAF

Surface and risk-adjust the chronic conditions OpenMed extracts by mapping them
to **CMS-HCC** categories (the **V28** model, phasing in for payment years
2024–2026) and estimating a **RAF** (Risk Adjustment Factor) score. CMS pays
Medicare Advantage plans based on RAF, so accurate, *documented* capture of
chronic disease matters — and much of that signal lives in the **narrative note**,
exactly what OpenMed reads.

This is **decision support for coders/clinicians**, not autonomous coding. The
output is "candidate HCCs + estimated RAF + the documentation that supports (or
fails to support) each one," for human validation.

CMS-HCC crosswalks (ICD-10-CM → HCC) and the category coefficients are **public**
— CMS publishes them annually. Nothing restricted is bundled.

## When to use

- You want to **find risk-adjustable diagnoses** mentioned in a note that may not
  be on the coded problem list ("suspect HCCs" / recapture).
- You need to **map ICD-10-CM codes to V28 HCCs** and apply the hierarchy.
- You want an **estimated RAF** for a patient or panel for review.
- You need to check whether a diagnosis has **MEAT** support
  (Monitored, Evaluated, Assessed, Treated) in the documentation.

Pairs with `coding-icd10` (you need ICD-10-CM codes first) and may consume
`mapping-to-snomed` output upstream.

## Quick start (public CMS crosswalk + coefficients)

CMS publishes the V28 ICD-10-CM→HCC mapping and the model coefficients. Load them
locally (public files) and apply the model:

```python
import csv

# 1) ICD-10-CM -> HCC (V28) crosswalk from the CMS Risk Adjustment files.
icd_to_hcc = {}                       # "E1122" -> "HCC38" (Diabetes w/ complication)
with open("cms_hcc_v28_icd_map.csv") as fh:
    for row in csv.DictReader(fh):
        icd_to_hcc[row["icd10cm"].replace(".", "")] = row["hcc_v28"]

# 2) HCC -> RAF coefficient for the relevant model segment (e.g. CNA community).
hcc_weight = {}                       # "HCC38" -> 0.166 (illustrative)
with open("cms_hcc_v28_coefficients.csv") as fh:
    for row in csv.DictReader(fh):
        hcc_weight[row["hcc"]] = float(row["coefficient"])

# 3) Apply the HCC hierarchy: a more severe HCC in a family suppresses milder
#    ones (e.g. acute MI suppresses angina). Load the hierarchy from CMS.
hierarchy = {                         # parent HCC -> HCCs it zeroes out
    # "HCC37": {"HCC38"},  # illustrative; use the official V28 hierarchy file
}

def apply_hierarchy(hccs: set[str]) -> set[str]:
    kept = set(hccs)
    for parent in hccs:
        kept -= hierarchy.get(parent, set())
    return kept

def estimate_raf(icd_codes: list[str], demo_factor: float = 0.0) -> dict:
    hccs = {icd_to_hcc[c] for c in icd_codes if c in icd_to_hcc}
    hccs = apply_hierarchy(hccs)
    disease_raf = sum(hcc_weight.get(h, 0.0) for h in hccs)
    return {"hccs": sorted(hccs),
            "disease_raf": round(disease_raf, 3),
            "estimated_raf": round(disease_raf + demo_factor, 3)}
```

The `demo_factor` (age/sex, dual/disability, institutional status) comes from the
CMS demographic tables — add it for a full RAF; omit for the disease component.

## Workflow

1. **Extract** condition spans with OpenMed (Disease/Pathology/Oncology models).
2. **Code** each to ICD-10-CM (see `coding-icd10`) — HCCs key off ICD-10-CM.
3. **Map** ICD-10-CM → V28 HCC via the CMS crosswalk.
4. **Apply the hierarchy** so only the most severe HCC in each family counts.
5. **Sum coefficients** for the correct model segment + add the demographic factor
   to estimate RAF.
6. **Attach MEAT evidence**: for each candidate HCC, cite the note text that
   Monitors/Evaluates/Assesses/Treats the condition. No MEAT → flag as
   "unsupported / needs clinician confirmation," not a captured HCC.
7. **Emit** candidate HCCs + estimated RAF + supporting offsets for human review.

## Hand-off from OpenMed

`openmed.analyze_text(..., output_format="dict")` returns `entities`, each a dict
with `text`, `label`, `confidence`, `start`, `end`. Use the offsets to pull MEAT
evidence sentences:

```python
import openmed

note = ("Problem list: type 2 diabetes with diabetic nephropathy; COPD. "
        "Plan: continue metformin, ordered HbA1c, refer nephrology.")
result = openmed.analyze_text(
    note,
    model_name="disease_detection_superclinical",   # Disease category
    output_format="dict",
)

DX_LABELS = {"DISEASE", "CONDITION", "PATHOLOGY"}
suspects = []
for ent in result["entities"]:
    if ent["label"] in DX_LABELS:
        # 1) code to ICD-10-CM (coding-icd10) -> e.g. "E1122"
        icd = map_to_icd10cm(ent["text"])           # your coding step
        hcc = icd_to_hcc.get(icd)
        if hcc:
            # MEAT: capture the sentence around the span for the reviewer
            sent = note[max(0, ent["start"] - 60): ent["end"] + 80]
            suspects.append({"condition": ent["text"], "icd10cm": icd,
                             "hcc": hcc, "span": (ent["start"], ent["end"]),
                             "meat_context": sent})

raf = estimate_raf([s["icd10cm"] for s in suspects])
print(raf, suspects)        # candidates + estimate, for coder validation
```

Carry OpenMed's `start`/`end` offsets so every suspect HCC links to the exact
documentation; this is what makes the suggestion auditable for RADV. Store codes,
HCCs, and offsets — not the raw note.

## Edge cases & gotchas

- **Decision support, never autonomous coding.** Risk-adjustment coding is
  audited (CMS RADV) and has direct payment and compliance consequences. Output
  *suspects with evidence* for a certified coder/clinician; never submit
  HCCs automatically.
- **MEAT is required.** A diagnosis merely *mentioned* (e.g. in history) without
  being Monitored/Evaluated/Assessed/Treated in the encounter generally cannot be
  captured. Always attach MEAT evidence and flag bare mentions as unsupported.
- **V28 dropped ~2,000 codes.** The V28 transition removed many ICD-10-CM codes
  from HCC mapping (notably diabetes-without-complication, some vascular and
  inflammatory codes). A code that mapped under V24 may map to *nothing* under
  V28 — use the V28 crosswalk, not V24, and don't assume continuity.
- **Hierarchy suppression.** Within a disease family only the most severe HCC
  counts; summing all of them inflates RAF. Apply the official V28 hierarchy.
- **Model segment matters.** Coefficients differ by segment (community vs
  institutional, aged vs disabled, new enrollee). Use the right segment's table or
  the RAF is wrong.
- **Negation/uncertainty.** "No evidence of CHF" or "rule out malignancy" must not
  become captured HCCs. Resolve assertion/negation in OpenMed before mapping.
- **Annual model updates.** CMS revises the model and weights yearly and is
  blending V24/V28 across payment years 2024–2026; pin and record which model
  version and payment year your estimate used.
- **Licensing.** CMS-HCC crosswalks/coefficients and ICD-10-CM are public. Do not
  bundle restricted vocabularies (CPT, SNOMED, UMLS) to support this — keep those
  user-supplied and out-of-process.
- **Local-first.** OpenMed NER runs on-device; HCC mapping uses local CMS tables.
  No PHI needs to leave the process at all.

## Standards & references

- CMS Risk Adjustment (HCC models, files, coefficients):
  https://www.cms.gov/medicare/payment/medicare-advantage-rates-statistics/risk-adjustment
- CMS-HCC model software & ICD-10 mappings (V28):
  https://www.cms.gov/medicare/health-plans/medicareadvtgspecratestats/risk-adjustors
- ICD-10-CM (public domain), for the upstream codes:
  https://www.cms.gov/medicare/coding-billing/icd-10-codes
- RADV (Risk Adjustment Data Validation) overview:
  https://www.cms.gov/research-statistics-data-and-systems/monitoring-programs/recovery-audit-program-parts-c-and-d
- Companion skills: `coding-icd10` (codes feed HCCs), `mapping-to-snomed`.
