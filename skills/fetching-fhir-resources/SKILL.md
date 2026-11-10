---
name: fetching-fhir-resources
description: "Fetches and pages FHIR R4 resources (Patient, DocumentReference, DiagnosticReport, Observation, Condition) from a FHIR REST server, decodes base64 attachments, and extracts clinical narrative for OpenMed. Use before OpenMed processing when pulling charts from an EHR FHIR API (Epic, Cerner/Oracle, HAPI, or any US Core server) and you need the note text de-identified and analyzed, then results rejoined by patient. Hand narrative to openmed.deidentify and openmed.analyze_text; openmed.interop.fhir_operations implements a $de-identify operation over Bundles. Trigger keywords: FHIR, R4, US Core, DocumentReference, DiagnosticReport, Bundle, _revinclude, presentedForm, base64, EHR API."
license: Apache-2.0
metadata:
  project: OpenMed
  category: data-ingestion
  pairs: before
  version: "1.0"
---

# Fetching FHIR R4 Resources for OpenMed

FHIR R4 is the modern EHR API: a RESTful, JSON-or-XML interface over resources
like `Patient`, `Encounter`, `Condition`, `Observation`, `DiagnosticReport`,
and `DocumentReference`. The unstructured clinical text you want for NLP lives
in **`DocumentReference.content.attachment`** and
**`DiagnosticReport.presentedForm`** — usually **base64-encoded** PDF, RTF, or
plain text. This skill pulls those resources, pages through results, decodes the
attachments, and hands the narrative to OpenMed.

## When to use

- You have FHIR R4 access to an EHR (Epic, Oracle Health/Cerner, HAPI, Medplum,
  Azure/Google/AWS HealthLake) and want note text for de-id and NER.
- You need to page a large search result set safely (`Bundle.link[next]`).
- You want to pull a patient's documents/reports and rejoin NLP output by
  patient and encounter.

## FHIR REST in one minute

Search is `GET [base]/[Type]?param=value`. Results come back as a
**searchset `Bundle`**; the next page is the URL in `Bundle.link` where
`relation == "next"`. Use `_count` to size pages, `_revinclude` to pull related
resources in one round trip, and `_since`/`_lastUpdated` for incremental sync.

```
GET /Patient?identifier=http://hospital.org/mrn|12345
GET /DocumentReference?patient=Patient/abc&category=clinical-note&_count=50
GET /DiagnosticReport?patient=Patient/abc&_revinclude=Observation:related
```

## Quick start

Page a search, decode attachments, hand narrative to OpenMed:

```python
import base64
import requests
import openmed

BASE = "https://fhir.example.org/r4"
HEADERS = {"Accept": "application/fhir+json", "Authorization": "Bearer <token>"}

def iter_bundle(url, params=None):
    """Yield resources across all pages following Bundle.link[next]."""
    while url:
        bundle = requests.get(url, params=params, headers=HEADERS, timeout=30).json()
        for entry in bundle.get("entry", []):
            yield entry.get("resource", {})
        params = None  # next links are fully-qualified
        url = next(
            (l["url"] for l in bundle.get("link", []) if l.get("relation") == "next"),
            None,
        )

def attachment_text(att):
    """Decode a FHIR Attachment to text (handles base64 and inline text/plain)."""
    if att.get("data"):
        raw = base64.b64decode(att["data"])
        if att.get("contentType", "").startswith("text/"):
            return raw.decode("utf-8", "replace")
        return ""  # PDF/RTF: route to OpenMed multimodal/OCR intake instead
    return ""

# Pull a patient's clinical notes and analyze each.
for doc in iter_bundle(f"{BASE}/DocumentReference",
                       {"patient": "Patient/abc",
                        "category": "clinical-note", "_count": 50}):
    for content in doc.get("content", []):
        text = attachment_text(content.get("attachment", {}))
        if not text.strip():
            continue
        deid = openmed.deidentify(text, method="replace", policy="hipaa_safe_harbor")
        result = openmed.analyze_text(deid.text, output_format="dict")
        patient_ref = doc.get("subject", {}).get("reference")  # rejoin key
```

## Workflow

1. **Authenticate.** Most production FHIR endpoints use SMART-on-FHIR OAuth2
   (client-credentials for backend services). Scope to the minimum
   (`system/DocumentReference.read`, `system/DiagnosticReport.read`).
2. **Search narrowly.** Filter by `patient`, `category`, `type` (LOINC),
   `date`, and `_count`. Prefer server-side filtering over client-side.
3. **Page** via `Bundle.link[next]` until exhausted. Never assume one page.
4. **Extract narrative:** `DocumentReference.content.attachment` and
   `DiagnosticReport.presentedForm`. Decode base64; for PDF/RTF/scanned
   content, route bytes to OpenMed's document intake (`multimodal`/`ocr`)
   rather than decoding as UTF-8.
5. **De-identify → analyze** each narrative with OpenMed.
6. **Rejoin** results to `subject.reference` (patient) and `context.encounter`
   so downstream consumers can group by patient/encounter — storing hashed,
   not raw, identifiers.

## Hand-off to / from OpenMed

- **To OpenMed (client-side):** decoded narrative → `openmed.deidentify` →
  `openmed.analyze_text`. Carry `subject.reference` as the rejoin key.
- **Server-side `$de-identify`:** `openmed.interop.fhir_operations` implements
  the FHIR `$de-identify` *operation logic* over the OpenMed privacy pipeline:
  - `de_identify_resource(resource, policy=..., method=...)`
  - `de_identify_bundle(bundle, policy=..., method=...)`
  - `de_identify(parameters)` — accepts/returns a `Parameters` envelope and
    reports modified element paths as an `OperationOutcome`.
  It de-identifies free-text strings, identifier values, and `text.div`
  narrative while never altering codes, references, systems, or temporal
  values. Use this to de-identify a whole fetched Bundle before storage:

  ```python
  from openmed.interop.fhir_operations import de_identify_bundle
  safe_bundle = de_identify_bundle(bundle, policy="hipaa_safe_harbor",
                                   method="replace")
  ```
- **Onward:** re-export structured findings with
  `openmed.clinical.exporters.fhir` (`to_bundle`, `to_operation_outcome`).

## Edge cases & gotchas

- **Attachments are often base64.** `attachment.data` is base64; large files use
  `attachment.url` (a separate Binary fetch) instead. Handle both.
- **Non-text content types.** `application/pdf`, `text/rtf`, scanned TIFF — do
  not `utf-8` decode these; send bytes to OpenMed multimodal/OCR intake.
- **Pagination loops.** Some servers emit cyclic or stale `next` links; cap page
  count and dedupe by resource `id`.
- **`_revinclude` vs `_include`.** `_include` pulls referenced resources;
  `_revinclude` pulls resources that *reference* yours. Mixing them changes
  Bundle entry `search.mode` (`match` vs `include`) — filter on it.
- **Versioning & profiles.** Confirm the server is R4 (`/metadata`
  CapabilityStatement) and US Core-conformant; field cardinality differs across
  FHIR versions.
- **Throttling.** Respect `429`/`Retry-After`; batch with `_count` and back off.
- **PHI everywhere.** A FHIR resource is PHI by definition — never log raw
  resources; de-identify before persistence or analytics.

## Standards & references

- FHIR R4 specification: https://hl7.org/fhir/R4/
- FHIR RESTful API & search: https://hl7.org/fhir/R4/http.html and
  https://hl7.org/fhir/R4/search.html
- US Core Implementation Guide: https://hl7.org/fhir/us/core/
- DocumentReference: https://hl7.org/fhir/R4/documentreference.html
- DiagnosticReport (`presentedForm`): https://hl7.org/fhir/R4/diagnosticreport.html
- SMART on FHIR (backend services auth): https://hl7.org/fhir/smart-app-launch/
