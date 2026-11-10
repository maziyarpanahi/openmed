---
name: parsing-ccda-documents
description: "Parses C-CDA / CCD XML clinical documents to extract human-readable section narrative plus coded entries, keyed by section LOINC codes and templateIds. Use before OpenMed processing when ingesting C-CDA R2.1 documents (CCD, Discharge Summary, H&P, Consultation Note) exported from an EHR and you need the narrative section text de-identified and analyzed. Hand section narrative to openmed.deidentify and openmed.analyze_text; XML-aware de-identification that preserves CDA markup is available via openmed.interop.cda. Trigger keywords: C-CDA, CCD, CDA, clinical document, templateId, LOINC section, narrative block, discharge summary XML, ClinicalDocument."
license: Apache-2.0
metadata:
  project: OpenMed
  category: data-ingestion
  pairs: before
  version: "1.0"
---

# Parsing C-CDA / CCD Documents for OpenMed

C-CDA (Consolidated Clinical Document Architecture) is the XML document standard
behind Meaningful Use / ONC certification — the CCD, Discharge Summary, History
& Physical, and Consultation Note you get when an EHR "exports a chart". Each
document is a `ClinicalDocument` with a header (patient, authors, encounter) and
a `structuredBody` of **sections**. Every section has *two* representations: a
human-readable **narrative `<text>` block** and machine-readable **coded
entries**. The narrative is what you feed to clinical NLP. This skill extracts
it and hands it to OpenMed.

## When to use

- You receive C-CDA R2.1 / CCD documents (Direct messaging, patient portal
  export, HIE) and want the free-text section narrative for de-id and NER.
- You need to pair narrative spans with the section they came from (problems,
  meds, allergies, results, plan, H&P narrative).
- You want XML-safe de-identification that keeps the document parseable.

## C-CDA structure in one minute

```xml
<ClinicalDocument xmlns="urn:hl7-org:v3">
  <recordTarget><patientRole>
    <id extension="12345" root="..."/>
    <patient><name><given>Jane</given><family>Doe</family></name>
      <birthTime value="19700115"/></patient>
  </patientRole></recordTarget>
  <component><structuredBody>
    <component><section>
      <templateId root="2.16.840.1.113883.10.20.22.2.5.1"/>   <!-- Problems -->
      <code code="11450-4" codeSystem="2.16.840.1.113883.6.1"/> <!-- LOINC -->
      <title>Problems</title>
      <text>Active problems: Type 2 diabetes, hypertension.</text>  <!-- narrative -->
      <entry>...coded SNOMED/ICD entries...</entry>
    </section></component>
  </structuredBody></component>
</ClinicalDocument>
```

Sections are identified by **`templateId/@root`** and by **section `code`**
(LOINC). The CDA namespace is `urn:hl7-org:v3`.

## Quick start

Extract section narrative by LOINC code, then hand off to OpenMed:

```python
import openmed
from xml.etree import ElementTree as ET

NS = {"hl7": "urn:hl7-org:v3"}
SECTION_LOINC = {
    "11450-4": "problems", "10160-0": "medications", "48765-2": "allergies",
    "30954-2": "results",  "18776-5": "plan",        "10164-2": "hpi",
    "8648-8": "hospital_course", "11488-4": "consult_note",
}

root = ET.parse("ccd.xml").getroot()
for section in root.findall(".//hl7:section", NS):
    code_el = section.find("hl7:code", NS)
    loinc = code_el.get("code") if code_el is not None else None
    text_el = section.find("hl7:text", NS)
    if text_el is None:
        continue
    narrative = "".join(text_el.itertext()).strip()       # flatten narrative block
    if not narrative:
        continue

    deid = openmed.deidentify(narrative, method="replace", policy="hipaa_safe_harbor")
    result = openmed.analyze_text(deid.text, output_format="dict")
    section_name = SECTION_LOINC.get(loinc, loinc)
    # attach (section_name, result) for downstream consumers
```

`"".join(text_el.itertext())` flattens the narrative block (which may contain
`<paragraph>`, `<list>`, `<table>`, `<content>` markup) into plain text.

## XML-aware whole-document de-identification

When you need to redact PHI from the *document* (header ids, names, addresses,
dates) while keeping the CDA XML valid and parseable, use the bundled adapter
rather than regexing the raw XML:

```python
from openmed.interop.cda import redact_cda, is_cda_document

if is_cda_document("ccd.xml"):
    safe_xml = redact_cda("ccd.xml")     # returns redacted XML string
```

`redact_cda` applies `DEFAULT_PHI_ELEMENT_MAP` (patient id hashed, name/address/
telecom null-flavored, birthTime and effectiveTime date-shifted) to header
elements *and* sweeps section narrative text — operating on text nodes only so
surrounding markup stays intact. Pass `text_redactor=` to plug an extra
free-text callback (e.g. an `openmed.deidentify` wrapper), `date_shift_days=`
for a fixed shift, and `keep_year=True` to preserve years.

## Workflow

1. **Confirm it's CDA.** `is_cda_document(...)` checks for a `ClinicalDocument`
   root. Reject XML with `DOCTYPE`/`ENTITY` declarations (XXE risk) — the
   adapter does this for you.
2. **Read the header** for context: patient, author, `effectiveTime`,
   `documentType` (`ClinicalDocument/code` LOINC). Treat all header values as PHI.
3. **Walk sections** by `templateId` or section `code` (LOINC). Map to your
   section vocabulary.
4. **Flatten narrative** `<text>` with `itertext()`; preserve the section→text
   association for span attribution.
5. **De-identify → analyze** each narrative with OpenMed. Prefer coded
   `<entry>` data when it already exists; use NLP to recover what is *only* in
   narrative.

## Hand-off to / from OpenMed

- **To OpenMed:** flattened section narrative → `openmed.deidentify` →
  `openmed.analyze_text`. Keep `(section LOINC, narrative)` so entities trace
  back to their section.
- **Adapter:** `openmed.interop.cda` provides `redact_cda`, `is_cda_document`,
  `PhiElementRule`, and `DEFAULT_PHI_ELEMENT_MAP` for namespace-aware,
  markup-preserving de-identification. It also registers an `.xml` document
  handler with OpenMed's multimodal intake, so `.xml` files are auto-detected as
  CDA and redacted on ingest.
- **Onward:** re-emit findings via `openmed.clinical.exporters.fhir` or align
  narrative-derived problems to the section's coded entries.

## Edge cases & gotchas

- **Narrative vs entries can disagree.** The human-readable `<text>` is
  authoritative for display, coded `<entry>` for machines — they sometimes drift.
  Reconcile, and prefer narrative for what NLP must recover.
- **`<content ID=...>`/`<reference>` linkage.** Narrative `<content>` elements
  carry IDs referenced by entries (`<reference value="#problem1"/>`); use them to
  link a coded entry to its exact narrative phrase.
- **Tables and lists.** Section narrative often uses `<table>`/`<list>`;
  `itertext()` flattens these — re-impose structure if column meaning matters.
- **Namespaces & prefixes.** Always bind the `urn:hl7-org:v3` namespace; some
  documents add `sdtc:` extensions and `xsi:` typing.
- **XXE / unsafe XML.** Never parse untrusted CDA with entity expansion enabled;
  the adapter rejects `DOCTYPE`/`ENTITY` outright — do the same in custom parsers.
- **Restricted terminology.** Coded entries reference SNOMED CT, RxNorm, LOINC;
  OpenMed does not bundle SNOMED/CPT — resolve codes against the user's own
  licensed terminology out-of-process.

## Standards & references

- C-CDA R2.1 Implementation Guide (HL7):
  https://www.hl7.org/implement/standards/product_brief.cfm?product_id=492
- HL7 CDA R2 base standard:
  https://www.hl7.org/implement/standards/product_brief.cfm?product_id=7
- C-CDA section templateIds & LOINC section codes (HL7 C-CDA Online):
  https://www.hl7.org/ccdasearch/
- LOINC document & section codes: https://loinc.org/
- ONC C-CDA scorecard / validation: https://site.healthit.gov/c-cda-validator
