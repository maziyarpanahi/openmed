---
name: structuring-radiology-reports
description: "Converts free-text radiology narratives into structured findings and impression — with measurements, laterality, anatomy, and follow-up recommendations — after OpenMed NER. Use when the user has a CT/MRI/X-ray/ultrasound/mammography report and needs the sections split (technique, comparison, findings, impression), lesion measurements and laterality captured, BI-RADS / Lung-RADS assessment categories pulled, or incidental findings and recommended follow-up tracked. Trigger keywords: radiology report, findings, impression, RadLex, DICOM-SR, BI-RADS, Lung-RADS, ACR, laterality, measurement, nodule, incidental finding, follow-up, structured reporting. Pairs after OpenMed: run openmed.analyze_text on the report (Anatomy/Disease/measurement entities), then assemble structured findings. De-identify the report first. Decision-support only — not a diagnostic medical device."
license: Apache-2.0
metadata:
  project: OpenMed
  category: imaging-ocr
  pairs: after
  version: "1.0"
---

# Structuring radiology reports

A radiology report is prose, but its *meaning* is structured: a **technique**, a
**comparison**, a list of **findings** (each with anatomy, laterality, and a
measurement), and an **impression** that may carry an **assessment category**
(BI-RADS, Lung-RADS) and a **follow-up recommendation**. This skill turns the
narrative into that structure so findings are trackable — especially
**incidental findings** that need downstream follow-up.

OpenMed extracts the anatomy, disease/finding, and measurement spans on-device;
this skill organizes them into sectioned, coded findings. It is
**decision-support, not a diagnostic device** — every structured finding must be
attributable back to its source sentence for radiologist review.

## When to use

- You have a CT/MRI/X-ray/US/mammography report and need
  `{technique, comparison, findings[], impression}` with measurements and
  laterality.
- You must capture **BI-RADS** (breast) or **Lung-RADS** (lung screening)
  assessment categories and the recommended action.
- You need to **track incidental findings** and the follow-up interval/modality
  the report recommends.
- You are mapping findings toward **RadLex** terms or a **DICOM-SR** structured
  report.

## Quick start

```python
import openmed

report = (
    "TECHNIQUE: CT chest without contrast.\n"
    "COMPARISON: CT 2023-11-02.\n"
    "FINDINGS: A 8 mm solid nodule is noted in the right upper lobe, "
    "unchanged. No pleural effusion.\n"
    "IMPRESSION: 8 mm right upper lobe nodule, stable. Lung-RADS 2. "
    "Recommend annual low-dose CT screening."
)

# 1) De-identify the report on-device first (synthetic example shown).
deid = openmed.deidentify(report, policy="hipaa_safe_harbor")
text = deid.deidentified_text

# 2) Run NER for anatomy / finding / measurement spans.
ents = openmed.analyze_text(
    text,
    model_name="anatomy_detection_superclinical",   # Anatomy category
    output_format="dict",
)["entities"]

# 3) Split sections by header, then attach entities + measurements per finding.
import re
SECTION = re.compile(r"(?im)^(TECHNIQUE|COMPARISON|FINDINGS|IMPRESSION)\s*:")
sections, last, name = {}, 0, None
for m in SECTION.finditer(text):
    if name: sections[name] = text[last:m.start()].strip()
    name, last = m.group(1).upper(), m.end()
if name: sections[name] = text[last:].strip()

structured = {
    "technique": sections.get("TECHNIQUE"),
    "comparison": sections.get("COMPARISON"),
    "findings": _split_findings(sections.get("FINDINGS", "")),   # one per sentence
    "impression": sections.get("IMPRESSION"),
    "measurements": re.findall(r"\b\d+(?:\.\d+)?\s?(?:mm|cm)\b", text),
    "laterality": sorted({w for w in ("right", "left", "bilateral")
                          if re.search(rf"\b{w}\b", text, re.I)}),
    "assessment": (re.search(r"\b(?:BI-RADS|Lung-RADS)\s*\d[A-C]?\b", text, re.I)
                   or [None])[0] if re.search(r"RADS", text, re.I) else None,
    "follow_up": _extract_followup(sections.get("IMPRESSION", "")),
}
```

`_split_findings` / `_extract_followup` are your sentence splitter and a
recommendation matcher ("recommend …", "follow-up in N months"); keep each
finding tied to its source sentence offsets.

## Workflow

1. **De-identify first.** `openmed.deidentify(report, policy=...)`; structure
   from `deidentified_text`. Patient name, MRN, accession, and dates go before
   anything is stored or shared.
2. **Split sections** by the standard headers (TECHNIQUE, COMPARISON, FINDINGS,
   IMPRESSION; also HISTORY/INDICATION). Reports vary — fall back to position if
   headers are missing.
3. **Run `analyze_text`** for anatomy and finding entities; capture measurements
   ("8 mm", "1.2 cm") and laterality ("right", "left", "bilateral") near each
   finding.
4. **Build one structured finding per observation**: `{anatomy, finding,
   laterality, measurement, change_vs_prior, source_offsets}`. "Unchanged",
   "stable", "increased", "new" capture temporal change against the comparison.
5. **Pull the assessment category** (BI-RADS 0-6, Lung-RADS 1-4X) from the
   impression and the **recommended follow-up** (modality + interval).
6. **Flag incidental findings** — findings unrelated to the exam indication — and
   route them to a follow-up tracker so they aren't lost.
7. **Map toward RadLex / DICOM-SR** if you need coded interoperability, and
   surface the whole structure to a radiologist for verification.

## Hand-off to / from OpenMed

OpenMed's `analyze_text` returns a `dict`; `result["entities"]` items carry
`text`, `label`, `confidence`, `start`, `end`.

- **From** `extracting-clinical-entities`: Anatomy and Disease/finding entities
  populate each structured finding; keep offsets so every field traces to a
  source sentence.
- **From** `extracting-lab-tables` / OCR: if the report is a scan, OCR it first
  (`openmed.multimodal.ocr.ocr`), then run NER on the recognized text.
- **From** `segmenting-clinical-sections`: reuse section detection if your reports
  don't use canonical headers.
- **To** `building-patient-timelines`: dated findings + change-vs-prior feed a
  longitudinal view (e.g. nodule size over time).
- **To** `extracting-dicom-metadata`: pair the structured findings with the
  study's DICOM metadata when assembling a DICOM-SR object.
- **De-identify** with `deidentifying-clinical-text` (`openmed.deidentify`)
  before any export. Everything runs **on-device**.

## Edge cases & gotchas

- **Negation and uncertainty change meaning.** "No pleural effusion" and "cannot
  exclude metastasis" are findings *about* absence/uncertainty — don't record
  them as positive findings. Use `resolving-clinical-context`
  (`openmed.clinical`) for negation/hedging before asserting a finding.
- **Laterality errors are clinically dangerous.** "Right" vs "left" must bind to
  the correct finding; a misattributed side can drive wrong-site decisions. Tie
  laterality to the nearest anatomy span by offset, not document-wide.
- **Measurements need their unit and axis.** "8 mm" vs "0.8 cm" are equal; a
  bare "8" is ambiguous. Capture the unit; for masses, capture all reported
  dimensions ("2.1 x 1.4 cm"), not just the first.
- **Assessment categories have controlled value sets.** BI-RADS 0-6 and
  Lung-RADS 1, 2, 3, 4A, 4B, 4X each map to a defined management action — don't
  invent or round categories; pull the literal value from the impression.
- **Incidental findings get lost.** A renal cyst mentioned in a chest CT is the
  classic missed follow-up. Explicitly separate incidental from
  indication-related findings and push incidentals to a tracker.
- **The impression is the actionable summary**, but findings may contain detail
  the impression omits — structure both, and prefer the impression for
  follow-up/assessment.
- **Decision-support disclaimer.** This is not a diagnostic medical device; it
  organizes text a radiologist authored. Every structured field must be reviewable
  against its source. Do not auto-act on a derived category or follow-up without
  clinician sign-off.

## Standards & references

- RadLex (RSNA radiology lexicon): https://radlex.org/
- DICOM Structured Reporting (PS3.16 templates): https://www.dicomstandard.org/
- ACR BI-RADS Atlas: https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Bi-Rads
- ACR Lung-RADS: https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Lung-Rads
- RSNA Radiology Reporting templates: https://www.rsna.org/practice-tools/data-tools-and-standards/radreport-template-library
- ACR Incidental Findings white papers: https://www.acr.org/Clinical-Resources/Incidental-Findings
- HL7 FHIR R4 DiagnosticReport (imaging): https://hl7.org/fhir/R4/diagnosticreport.html
