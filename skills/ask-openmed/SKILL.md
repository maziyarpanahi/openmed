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
2. Apply the privacy override. If the goal asks to analyze, extract, exchange,
   share, upload, or verify clinical or personal content and does not say that
   the input is synthetic or already de-identified, start at the **privacy
   gate**: [deidentifying-clinical-text](../deidentifying-clinical-text/SKILL.md).
   An intake-only goal may start with its format parser, but the gate must run
   before the parsed content reaches a downstream stage.
3. Otherwise match the first row in the relevant table below. When several
   rows match, use the first row in that table and do not guess between
   alternatives.
4. If no row matches, use [building-with-openmed](../building-with-openmed/SKILL.md)
   as the orientation fallback.

Route output should contain only the selected category, skill identifier, and
the next handoff. Never echo the request, input values, exception text, or
detected spans into a log or report.

For a multi-stage request, route one stage at a time in this fixed handoff
order: **intake → privacy → extraction → exchange → verification**. A later
stage does not bypass the privacy gate merely because it was named first.

## Intake

Choose the first skill for turning an external document or record format into
local, processable input.

| Goal cues | First skill | Continue with |
| --- | --- | --- |
| scan, fax, image, PDF, CSV, table, or document OCR | [ingesting-clinical-documents](../ingesting-clinical-documents/SKILL.md) | privacy, then extraction |
| C-CDA, CCD, or CDA XML | [parsing-ccda-documents](../parsing-ccda-documents/SKILL.md) | privacy, then extraction |
| HL7 v2, ADT, ORU, MDM, OBX, or pipe-delimited feed | [parsing-hl7v2-messages](../parsing-hl7v2-messages/SKILL.md) | privacy, then extraction |
| DICOM header or DICOM-SR metadata | [extracting-dicom-metadata](../extracting-dicom-metadata/SKILL.md) | privacy, then extraction |
| pull or page records from a FHIR server | [fetching-fhir-resources](../fetching-fhir-resources/SKILL.md) | privacy, then extraction |
| extract rows from a laboratory table | [extracting-lab-tables](../extracting-lab-tables/SKILL.md) | privacy, then verification |

## Privacy

Use the privacy table both for an explicit privacy goal and for the privacy
override. The first row is the default gate for unspecified sensitive input.

| Goal cues | First skill | Continue with |
| --- | --- | --- |
| remove, mask, redact, anonymize, or de-identify clinical text | [deidentifying-clinical-text](../deidentifying-clinical-text/SKILL.md) | extraction or exchange |
| find identifiers without changing the source | [extracting-pii-entities](../extracting-pii-entities/SKILL.md) | de-identification or review |
| choose a privacy policy or profile | [configuring-privacy-policies](../configuring-privacy-policies/SKILL.md) | de-identification |
| keep PHI out of logs, errors, or telemetry | [enforcing-nophi-logging](../enforcing-nophi-logging/SKILL.md) | verification |
| de-identify non-English text | [deidentifying-multilingual-text](../deidentifying-multilingual-text/SKILL.md) | extraction |
| create stable pseudonyms for approved linkage | [pseudonymizing-for-gdpr](../pseudonymizing-for-gdpr/SKILL.md) | verification |

## Extraction

Choose the focused skill after intake and privacy handling when the request is
about finding or structuring clinical meaning.

| Goal cues | First skill | Continue with |
| --- | --- | --- |
| diseases, drugs, anatomy, genes, or clinical NER | [extracting-clinical-entities](../extracting-clinical-entities/SKILL.md) | exchange or verification |
| custom entity labels or zero-shot extraction | [running-zeroshot-ner](../running-zeroshot-ner/SKILL.md) | exchange or verification |
| housing, food, work, transport, or other SDOH | [extracting-sdoh](../extracting-sdoh/SKILL.md) | verification |
| lab values, units, reference ranges, or abnormal flags | [parsing-lab-values](../parsing-lab-values/SKILL.md) | terminology or exchange |
| radiology findings, measurements, or impression | [structuring-radiology-reports](../structuring-radiology-reports/SKILL.md) | exchange or verification |
| note sections such as history, medications, or assessment | [segmenting-clinical-sections](../segmenting-clinical-sections/SKILL.md) | extraction |

## Exchange

Choose an exchange skill only after the source is safe to handle. These skills
may use a user-supplied endpoint or terminology service; this router itself
performs no network call.

| Goal cues | First skill | Continue with |
| --- | --- | --- |
| turn extracted entities into FHIR R4 resources | [exporting-to-fhir](../exporting-to-fhir/SKILL.md) | bundle or conformance verification |
| build a FHIR transaction or batch Bundle | [assembling-fhir-bundles](../assembling-fhir-bundles/SKILL.md) | conformance verification |
| run a FHIR Bulk Data export | [exporting-bulk-fhir](../exporting-bulk-fhir/SKILL.md) | privacy, then extraction |
| build a SMART-on-FHIR app or launch flow | [scaffolding-smart-on-fhir](../scaffolding-smart-on-fhir/SKILL.md) | conformance verification |
| validate, expand, or translate through a terminology server | [querying-terminology-service](../querying-terminology-service/SKILL.md) | exchange or verification |

## Verification

Use verification routes for a stated release, safety, leakage, audit, risk, or
conformance check. If the source is not explicitly safe, the privacy override
still wins first.

| Goal cues | First skill | Decision or handoff |
| --- | --- | --- |
| check a de-identified output for residual identifiers | [auditing-deid-leakage](../auditing-deid-leakage/SKILL.md) | block release on a finding |
| create or inspect a no-PHI de-identification audit trail | [auditing-deidentification-runs](../auditing-deidentification-runs/SKILL.md) | retain offsets, hashes, and provenance |
| gate an evaluation or release on leakage | [evaluating-with-leakage-gates](../evaluating-with-leakage-gates/SKILL.md) | fail closed on leakage |
| assess re-identification, k-anonymity, or quasi-identifiers | [reviewing-reidentification-risk](../reviewing-reidentification-risk/SKILL.md) | review residual risk |
| check Safe Harbor categories | [auditing-safe-harbor-checklist](../auditing-safe-harbor-checklist/SKILL.md) | review the no-PHI report |
| check FHIR US Core or USCDI conformance | [validating-us-core](../validating-us-core/SKILL.md) | correct the resource before exchange |
| review a HIPAA privacy or security checklist | [checking-hipaa-compliance](../checking-hipaa-compliance/SKILL.md) | address gaps before release |

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
- **“Which OpenMed workflow fits?”** Use the orientation fallback
  [building-with-openmed](../building-with-openmed/SKILL.md), then re-route once
  the goal and sensitivity status are explicit.

These are routing examples only. Keep all demonstrations synthetic or
placeholder-based, and keep route diagnostics free of source content.
