---
name: extracting-dicom-metadata
description: "Reads DICOM file headers and DICOM-SR (Structured Report) content to pull study/series metadata and embedded report text, and flags PHI carried in header tags. Use before OpenMed processing when ingesting imaging data (CT/MR/CR/US, radiology SR) and you need the report narrative de-identified and analyzed, plus a list of header tags that must be scrubbed. Hand SR/report text to openmed.deidentify and openmed.analyze_text; use pydicom to read tags. Trigger keywords: DICOM, pydicom, DICOM-SR, structured report, PatientName, study metadata, PACS, radiology report, PS3."
license: Apache-2.0
metadata:
  project: OpenMed
  category: data-ingestion
  pairs: before
  version: "1.0"
---

# Extracting DICOM Metadata & Report Text for OpenMed

DICOM (Digital Imaging and Communications in Medicine) files carry far more than
pixels: a **header** of tagged attributes (patient, study, series, equipment)
and, for **DICOM-SR (Structured Reports)**, a content tree holding the actual
radiology/cardiology *report text*. Two jobs sit here: pull the report narrative
for NLP, and **flag the PHI in the header** so it gets scrubbed. This skill does
both, then hands narrative to OpenMed. Header tags are read with `pydicom`
(external, MIT-licensed); de-identification of the extracted text is OpenMed's.

## When to use

- You ingest DICOM from PACS/VNA or a research archive and want the SR report
  text mined with clinical NLP.
- You must enumerate PHI-bearing header tags before sharing/exporting images.
- You have DICOM-SR objects (e.g. radiology measurements + impression) whose
  content tree contains the dictated report.

## DICOM headers in one minute

Every attribute has a **tag** `(gggg,eeee)` (group, element), a **VR** (value
representation, e.g. `PN` person name, `DA` date, `UI` UID), and a value. PHI
clusters in well-known tags:

| Tag | Name | VR | Notes |
| --- | --- | --- | --- |
| (0010,0010) | PatientName | PN | direct identifier |
| (0010,0020) | PatientID | LO | MRN |
| (0010,0030) | PatientBirthDate | DA | DOB |
| (0010,1040) | PatientAddress | LO | address |
| (0008,0090) | ReferringPhysicianName | PN | provider |
| (0008,0020/0030) | StudyDate / StudyTime | DA/TM | dates |
| (0008,0050) | AccessionNumber | SH | order id |
| (0008,103E) | SeriesDescription | LO | free text — may leak PHI |
| (0020,4000) | ImageComments | LT | free text — may leak PHI |
| (0040,A730) | ContentSequence | SQ | DICOM-SR report tree |

## Quick start

Read the header, pull SR report text, flag PHI tags, hand off to OpenMed:

```python
import pydicom
import openmed

ds = pydicom.dcmread("study.dcm")

# 1) Enumerate PHI-bearing header tags (report, do not log values).
PHI_TAGS = [
    (0x0010, 0x0010), (0x0010, 0x0020), (0x0010, 0x0030), (0x0010, 0x1040),
    (0x0008, 0x0090), (0x0008, 0x0050), (0x0008, 0x0020), (0x0008, 0x0030),
]
present_phi = [hex_pair for hex_pair in PHI_TAGS if hex_pair in ds]

# 2) Extract report text from a DICOM-SR content tree (recursively).
def sr_text(dataset):
    chunks = []
    for item in dataset.get("ContentSequence", []):
        vt = item.get("ValueType")
        if vt == "TEXT" and "TextValue" in item:
            chunks.append(item.TextValue)
        if "ContentSequence" in item:          # nested CONTAINER
            chunks.append(sr_text(item))
    return "\n".join(c for c in chunks if c)

report = sr_text(ds)
# Some modalities stash narrative in free-text header tags too:
for tag in ("ImageComments", "SeriesDescription", "StudyDescription"):
    if tag in ds and isinstance(ds.get(tag), str):
        report += "\n" + ds.get(tag)

# 3) De-identify the narrative, then run NER.
if report.strip():
    deid = openmed.deidentify(report, method="replace", policy="hipaa_safe_harbor")
    result = openmed.analyze_text(deid.text, output_format="dict")
```

`pydicom` reads tags by keyword (`ds.PatientName`) or by `(group, element)`.
DICOM-SR text lives in the recursive `ContentSequence` content tree.

## Workflow

1. **Read the dataset** with `pydicom.dcmread` (use `stop_before_pixels=True`
   for header-only/metadata work — faster, avoids loading pixels).
2. **Walk the SR content tree.** `ContentSequence` nests `CONTAINER`, `TEXT`,
   `CODE`, `NUM`, `PNAME` nodes; concatenate `TEXT.TextValue` (and relevant
   `CODE`/`NUM` measurements) in document order to reconstruct the report.
3. **Inventory PHI tags.** Flag the standard identifier tags *and* free-text
   tags (`ImageComments`, `*Description`) that frequently leak PHI. Report tag
   presence — never echo the values into logs.
4. **De-identify → analyze** the report narrative with OpenMed.
5. **Scrub the header** before any image export using a DICOM de-identification
   profile (PS3.15 Annex E / Basic Application Level Confidentiality). OpenMed
   de-identifies the *narrative*; header scrubbing is a separate DICOM step.

## Hand-off to / from OpenMed

- **To OpenMed:** SR report text (and free-text header tags) →
  `openmed.deidentify` → `openmed.analyze_text`.
- **Header de-id is out of scope for OpenMed** — OpenMed handles the *text*
  narrative; use a DICOM-native de-identifier (pydicom + PS3.15 profile, or a
  PACS de-id node) to scrub `(0010,xxxx)` and burned-in-pixel PHI. This skill's
  job is to flag those tags so they aren't missed.
- **Re-link by UID, not PHI.** Carry `StudyInstanceUID`/`SeriesInstanceUID` as
  rejoin keys; these are not identifiers but should be re-mapped consistently if
  the profile requires UID remapping.

## Edge cases & gotchas

- **Pixel-burned PHI.** Ultrasound and secondary-capture images often burn name/
  MRN/date into the *pixels* — header scrubbing alone is insufficient; flag
  modalities (US, SC, XC) for pixel review/OCR. OpenMed's multimodal/OCR intake
  can read burned-in text for redaction screening.
- **Private tags.** Vendor `(gggg,eeee)` odd-group private tags can hide PHI;
  PS3.15 requires removing or whitelisting them — don't trust unknown tags.
- **Date shifting must be consistent.** If you date-shift `StudyDate`, shift all
  related dates by the same offset to preserve temporal relationships.
- **SR value types.** Not all SR content is narrative — `NUM` (measurements),
  `CODE` (coded findings), `PNAME` (person names, PHI!) need different handling;
  don't dump `PNAME` into NLP text.
- **Character sets.** Honor `SpecificCharacterSet (0008,0005)`; non-Latin
  patient names need correct decoding before de-id.
- **Read-only intake.** Treat source DICOM as immutable; write de-identified
  copies, never overwrite originals.

## Standards & references

- DICOM standard (PS3.x), Part 6 Data Dictionary (tags):
  https://www.dicomstandard.org/current
- PS3.15 Annex E — Attribute Confidentiality Profiles (de-identification):
  https://dicom.nema.org/medical/dicom/current/output/html/part15.html#chapter_E
- DICOM-SR (PS3.3 Structured Reporting; PS3.16 templates):
  https://dicom.nema.org/medical/dicom/current/output/html/part03.html
- pydicom documentation: https://pydicom.github.io/
- DICOM PS3.16 TID 2000 (Basic Diagnostic Imaging Report):
  https://dicom.nema.org/medical/dicom/current/output/html/part16.html
