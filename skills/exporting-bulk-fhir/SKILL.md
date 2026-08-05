---
name: exporting-bulk-fhir
description: "Kick off and harvest a FHIR Bulk Data $export (system-, group-, or patient-level) and stream the resulting NDJSON into a batch OpenMed de-identification + NER pipeline at cohort scale. Covers the async kickoff (Prefer respond-async) -> poll Content-Location -> download NDJSON flow, the Bulk Data Access IG, _type/_since filters, and feeding DocumentReference/DiagnosticReport notes into openmed.deidentify in batch. Use when the user needs population-scale note extraction from an EHR or data warehouse to feed OpenMed, mentions bulk export, $export, NDJSON, Flat FHIR, or cohort de-identification. Pairs before the OpenMed de-id/NER pipeline."
license: Apache-2.0
metadata:
  project: OpenMed
  category: fhir-interop
  pairs: before
  version: "1.0"
---

# Exporting Bulk FHIR

When you need *cohort-scale* clinical text — not one patient in a UI — you use
the FHIR **Bulk Data Access** (`$export`) operation: an async job that emits
**NDJSON** files of resources you then stream into OpenMed for batch
de-identification and NER. This skill sits **before** the OpenMed pipeline: it is
how the notes arrive.

## When to use

Reach for it when the source is an EHR or FHIR data warehouse and the volume is
a population/group (thousands of patients), the workload is headless (no
clinician UI), and the goal is to batch-feed `openmed.deidentify` /
`openmed.analyze_text`. Triggers: "bulk export", "$export", "NDJSON", "Flat
FHIR", "cohort de-identification", "export all notes". For a single in-chart
patient with a UI, use `scaffolding-smart-on-fhir` instead.

## Three export levels

- **System** — `GET [base]/$export` — everything the client is authorized for.
- **Group** — `GET [base]/Group/[id]/$export` — a defined cohort (most common).
- **Patient** — `GET [base]/Patient/$export` — all patients in scope.

Bulk export uses **SMART Backend Services** auth (a `system/*.read`-scoped
client-credentials token via a signed JWT assertion), not an interactive launch.

## Quick start: kickoff → poll → download

```bash
# 1) Kickoff (async). Ask for clinical-note-bearing resource types.
curl -s -X GET \
  'https://ehr.example/fhir/Group/cohort-42/$export?_type=DocumentReference,DiagnosticReport&_since=2024-01-01T00:00:00Z' \
  -H 'Authorization: Bearer <backend-services-token>' \
  -H 'Accept: application/fhir+json' \
  -H 'Prefer: respond-async' -D -
# -> 202 Accepted
#    Content-Location: https://ehr.example/fhir/bulkstatus/JOB123

# 2) Poll the status URL until complete
curl -s 'https://ehr.example/fhir/bulkstatus/JOB123' \
  -H 'Authorization: Bearer <token>'
# 202 + X-Progress while running; 200 + a manifest JSON when done:
# { "transactionTime": "...", "request": "...", "requiresAccessToken": true,
#   "output": [
#     { "type": "DocumentReference",
#       "url": "https://ehr.example/fhir/bulkfiles/dr-1.ndjson" },
#     { "type": "DiagnosticReport",
#       "url": "https://ehr.example/fhir/bulkfiles/dx-1.ndjson" } ] }

# 3) Download each NDJSON file (one FHIR resource per line)
curl -s 'https://ehr.example/fhir/bulkfiles/dr-1.ndjson' \
  -H 'Authorization: Bearer <token>' -o dr-1.ndjson
```

Key headers/params: `Prefer: respond-async` (required to start the job),
`Content-Location` (the status/polling URL), `_type` (limit resource types),
`_since` (incremental export), `_typeFilter` (server-side resource filtering).
Delete the job when done: `DELETE <status-url>`.

## Stream NDJSON into OpenMed (batch)

NDJSON is one resource per line — stream it; do not load the whole file. Pull the
note text out of each `DocumentReference`/`DiagnosticReport` and run OpenMed
**on-device**, in batch:

```python
import base64, json, openmed

def note_text(resource: dict) -> str | None:
    # DocumentReference.content[].attachment.data (base64) or .url -> Binary
    for content in resource.get("content", []):
        att = content.get("attachment", {})
        if att.get("data"):
            return base64.b64decode(att["data"]).decode("utf-8", "replace")
    # DiagnosticReport.presentedForm[].data
    for form in resource.get("presentedForm", []):
        if form.get("data"):
            return base64.b64decode(form["data"]).decode("utf-8", "replace")
    return None

with open("dr-1.ndjson", "r", encoding="utf-8") as fh:
    for line in fh:                              # streaming, line by line
        resource = json.loads(line)
        text = note_text(resource)
        if not text:
            continue
        # De-identify every note before anything downstream sees it
        deid = openmed.deidentify(text, method="replace", policy="hipaa_safe_harbor")
        # Then NER on the de-identified text
        entities = openmed.analyze_text(
            deid.text, model_name="disease_detection_superclinical")
        # ... persist de-identified text + spans; never persist raw PHI
```

For large cohorts, parallelise across files (each NDJSON file is independent)
and reuse a single OpenMed model loader across notes to avoid reloading weights.

## Workflow

1. Obtain a SMART Backend Services token (`system/DocumentReference.read`, etc.).
2. Kickoff `$export` at the right level with `_type` (and `_since` for
   incrementals) + `Prefer: respond-async`.
3. Poll `Content-Location` until `200`; read the manifest `output[]`.
4. Download each NDJSON file (send the token if `requiresAccessToken`).
5. Stream each line → extract note text → `openmed.deidentify` →
   `openmed.analyze_text`.
6. Export findings to FHIR if needed (`exporting-to-fhir`,
   `assembling-fhir-bundles`).
7. `DELETE` the bulk job to free server storage.

## Hand-off to / from OpenMed

- **Into OpenMed (the point of this skill):** NDJSON note text → batch
  `openmed.deidentify` is the primary hand-off. De-identify **first**; treat
  every exported note as PHI until it has been through the de-id pass.
- **Back to FHIR:** the spans from `analyze_text` → `exporting-to-fhir` →
  `to_bundle`; write back only if your governance allows.
- **Local-first at scale:** OpenMed runs on-device, so the cohort never leaves
  your infrastructure for NLP. Only the *export* traffic touches the EHR.

## Edge cases & gotchas

- **It's async — never block on the kickoff.** A 202 + `Content-Location` is
  success; poll with backoff and honour `Retry-After`/`X-Progress`.
- **Files can be huge.** Stream NDJSON line-by-line; do not `json.load` a whole
  file. Parallelise per file, not per line.
- **`requiresAccessToken`.** If the manifest says so, send the bearer token when
  downloading the NDJSON files too.
- **De-identify before persistence.** Raw exported notes are PHI; the first
  durable artifact must be de-identified. Verify de-id with `openmed.eval`
  leakage gates (`evaluating-with-leakage-gates`), not F1 alone.
- **Note formats vary.** Text may be inline base64, an external `Binary`
  reference, or RTF/HTML in `presentedForm`. Normalise to plain text before
  OpenMed; for scanned PDFs use OpenMed's document/OCR intake.
- **Clean up the job.** Servers may cap concurrent/stored exports; `DELETE` the
  status URL when finished.
- **Scope minimally.** Request only the resource types you will process; honour
  the cohort's consent/governance.

## Standards & references

- FHIR Bulk Data Access (Flat FHIR) IG: https://hl7.org/fhir/uv/bulkdata/
- `$export` operation: https://hl7.org/fhir/uv/bulkdata/export.html
- Async request pattern: https://hl7.org/fhir/R4/async.html
- SMART Backend Services auth: https://hl7.org/fhir/uv/bulkdata/authorization/
- NDJSON: https://github.com/ndjson/ndjson-spec
