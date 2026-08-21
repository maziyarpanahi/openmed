---
name: ask-openmed
description: "Choose the first OpenMed workflow skill for an intake, privacy, extraction, exchange, or verification request using deterministic local routing. Use when a goal is broad, spans several clinical-data stages, or leaves the data-sensitivity status unclear; the privacy gate is selected before downstream work when raw clinical or personal content may be present."
license: Apache-2.0
metadata:
  project: OpenMed
  category: openmed-core
  pairs: adjacent
  version: "1.0"
---

# Ask OpenMed

Use this skill as the first pass when a request does not name a focused OpenMed
workflow. It selects an existing skill identifier; it does not process input,
download a model, call a service, or make a clinical decision. No mandatory
network call is part of this route.

## Deterministic routing contract

Apply these rules in order:

1. Normalize only the stated goal: lowercase it, trim surrounding whitespace,
   and collapse repeated whitespace. Do not inspect or copy the data payload.
2. Apply the intake boundary. If the goal matches an Intake row, select the
   first matching intake skill even when the goal also names a later stage. If
   sensitivity is unclear, its next handoff is the **privacy gate**:
   [deidentifying-clinical-text](../deidentifying-clinical-text/SKILL.md), before
   extraction, exchange, or verification sees the parsed content.
3. For a goal with no intake cue, apply the privacy override. If it asks to
   analyze, extract, exchange, share, upload, or verify clinical or personal
   content and is not explicitly marked safe, start at the privacy gate. The
   only safe markers are `synthetic input`, `synthetic note`, `synthetic
   record`, `synthetic dataset`, `input is synthetic`, `data is synthetic`,
   `already de-identified`, and `already deidentified`. A marker does not count
   when `no`, `not`, `never`, `unknown`, or `uncertain` occurs in the four
   normalized words before it. Treat every ambiguous safety statement as
   sensitive.
4. Otherwise inspect the remaining tables in the fixed handoff order shown
   below. Treat
   each comma-separated cue as a case-insensitive substring of the normalized
   goal, and select the first matching row. Every cue cell is a comma-separated
   list of alternatives; conjunctions have no special meaning. Do not add
   synonyms or infer cues from the payload. This stage order and row order
   break every tie.
5. If no row matches, use [building-with-openmed](../building-with-openmed/SKILL.md)
   as the orientation fallback.

Route output should contain only the selected category, skill identifier,
matched rule index, and next handoff. Never echo the request, input values,
exception text, or detected spans into a log or report.

For a multi-stage request, route one stage at a time in this fixed handoff
order: **intake → privacy → extraction → exchange → verification**. A later
stage does not bypass the privacy gate merely because it was named first.

## Intake

Choose the first skill for turning an external document or record format into
local, processable input.

| Goal cues | First skill | Continue with |
| --- | --- | --- |
| laboratory table, lab table | [extracting-lab-tables](../extracting-lab-tables/SKILL.md) | privacy, then verification |
| scan, fax, image, PDF, CSV, table, document OCR | [ingesting-clinical-documents](../ingesting-clinical-documents/SKILL.md) | privacy, then extraction |
| C-CDA, CCD, CDA XML | [parsing-ccda-documents](../parsing-ccda-documents/SKILL.md) | privacy, then extraction |
| HL7 v2, ADT, ORU, MDM, OBX, pipe-delimited feed | [parsing-hl7v2-messages](../parsing-hl7v2-messages/SKILL.md) | privacy, then extraction |
| DICOM header, DICOM-SR metadata | [extracting-dicom-metadata](../extracting-dicom-metadata/SKILL.md) | privacy, then extraction |
| pull FHIR records, page FHIR records, FHIR server | [fetching-fhir-resources](../fetching-fhir-resources/SKILL.md) | privacy, then extraction |

## Privacy

Use the privacy table both for an explicit privacy goal and for the privacy
override. The first row is the default gate for unspecified sensitive input.

| Goal cues | First skill | Continue with |
| --- | --- | --- |
| remove clinical identifiers, mask clinical identifiers, redact clinical text, anonymize clinical text, de-identify clinical text | [deidentifying-clinical-text](../deidentifying-clinical-text/SKILL.md) | extraction or exchange |
| find identifiers, detect identifiers, PII entities | [extracting-pii-entities](../extracting-pii-entities/SKILL.md) | de-identification or review |
| privacy policy, privacy profile | [configuring-privacy-policies](../configuring-privacy-policies/SKILL.md) | de-identification |
| PHI in logs, PHI in errors, PHI in telemetry, no-PHI logging | [enforcing-nophi-logging](../enforcing-nophi-logging/SKILL.md) | verification |
| de-identify non-English text, multilingual de-identification | [deidentifying-multilingual-text](../deidentifying-multilingual-text/SKILL.md) | extraction |
| stable pseudonyms, approved linkage pseudonyms | [pseudonymizing-for-gdpr](../pseudonymizing-for-gdpr/SKILL.md) | verification |

## Extraction

Choose the focused skill after intake and privacy handling when the request is
about finding or structuring clinical meaning.

| Goal cues | First skill | Continue with |
| --- | --- | --- |
| diseases, drugs, anatomy, genes, clinical NER | [extracting-clinical-entities](../extracting-clinical-entities/SKILL.md) | exchange or verification |
| custom entity labels, zero-shot extraction | [running-zeroshot-ner](../running-zeroshot-ner/SKILL.md) | exchange or verification |
| housing, food, work, transport, SDOH | [extracting-sdoh](../extracting-sdoh/SKILL.md) | verification |
| lab values, units, reference ranges, abnormal flags | [parsing-lab-values](../parsing-lab-values/SKILL.md) | terminology or exchange |
| radiology findings, radiology measurements, radiology impression | [structuring-radiology-reports](../structuring-radiology-reports/SKILL.md) | exchange or verification |
| note sections, clinical sections, section segmentation | [segmenting-clinical-sections](../segmenting-clinical-sections/SKILL.md) | extraction |

## Exchange

Choose an exchange skill only after the source is safe to handle. These skills
may use a user-supplied endpoint or terminology service; this router itself
performs no network call.

| Goal cues | First skill | Continue with |
| --- | --- | --- |
| FHIR R4 resources, export to FHIR | [exporting-to-fhir](../exporting-to-fhir/SKILL.md) | bundle or conformance verification |
| FHIR transaction Bundle, FHIR batch Bundle | [assembling-fhir-bundles](../assembling-fhir-bundles/SKILL.md) | conformance verification |
| FHIR Bulk Data export, bulk FHIR | [exporting-bulk-fhir](../exporting-bulk-fhir/SKILL.md) | privacy, then extraction |
| SMART-on-FHIR app, SMART-on-FHIR launch | [scaffolding-smart-on-fhir](../scaffolding-smart-on-fhir/SKILL.md) | conformance verification |
| terminology validation, terminology expansion, terminology translation, terminology server | [querying-terminology-service](../querying-terminology-service/SKILL.md) | exchange or verification |

## Verification

Use verification routes for a stated release, safety, leakage, audit, risk, or
conformance check. If the source is not explicitly safe, the privacy override
still wins first.

| Goal cues | First skill | Decision or handoff |
| --- | --- | --- |
| residual identifiers, de-identification leakage | [auditing-deid-leakage](../auditing-deid-leakage/SKILL.md) | block release on a finding |
| de-identification audit trail, no-PHI audit trail | [auditing-deidentification-runs](../auditing-deidentification-runs/SKILL.md) | retain offsets, hashes, and provenance |
| leakage gate, release leakage | [evaluating-with-leakage-gates](../evaluating-with-leakage-gates/SKILL.md) | fail closed on leakage |
| re-identification risk, k-anonymity, quasi-identifiers | [reviewing-reidentification-risk](../reviewing-reidentification-risk/SKILL.md) | review residual risk |
| Safe Harbor | [auditing-safe-harbor-checklist](../auditing-safe-harbor-checklist/SKILL.md) | review the no-PHI report |
| FHIR US Core, USCDI conformance | [validating-us-core](../validating-us-core/SKILL.md) | correct the resource before exchange |
| HIPAA privacy checklist, HIPAA security checklist | [checking-hipaa-compliance](../checking-hipaa-compliance/SKILL.md) | address gaps before release |

## Ambiguous goals and escalation examples

Use the privacy override and the fixed handoff order for ambiguous requests:

- **“Turn this clinical note into FHIR.”** The sensitivity is unstated, so
  start at [deidentifying-clinical-text](../deidentifying-clinical-text/SKILL.md),
  then hand off to [extracting-clinical-entities](../extracting-clinical-entities/SKILL.md)
  and [exporting-to-fhir](../exporting-to-fhir/SKILL.md).
- **“Extract medications from a synthetic note.”** The input is explicitly
  synthetic, so route directly to
  [extracting-clinical-entities](../extracting-clinical-entities/SKILL.md).
- **“OCR this image and extract findings.”** Route to
  [ingesting-clinical-documents](../ingesting-clinical-documents/SKILL.md),
  then apply the privacy gate before extraction.
- **“Is this dataset safe to share?”** If de-identification is not stated,
  start with [deidentifying-clinical-text](../deidentifying-clinical-text/SKILL.md)
  and then use [auditing-deid-leakage](../auditing-deid-leakage/SKILL.md).
- **“Verify this clinical dataset; it is not de-identified.”** The negated
  safety statement does not count as a safe marker, so start at
  [deidentifying-clinical-text](../deidentifying-clinical-text/SKILL.md).
- **“Which OpenMed workflow fits?”** Use the orientation fallback
  [building-with-openmed](../building-with-openmed/SKILL.md), then re-route once
  the goal and sensitivity status are explicit.

These are routing examples only. Keep all demonstrations synthetic or
placeholder-based, and keep route diagnostics free of source content.
