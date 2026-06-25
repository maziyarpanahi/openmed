---
name: assembling-fhir-bundles
description: "Package multiple FHIR R4 resources produced from OpenMed output into a single valid transaction Bundle ready to POST to an EHR, using OpenMed's verified bundle assembler openmed.clinical.exporters.fhir.to_bundle. Covers deterministic urn:uuid fullUrls, automatic in-Bundle reference rewriting, request blocks (method/url) for transaction vs batch, and conditional create. Use after exporting-to-fhir when the user has several Condition/Observation/MedicationStatement resources and wants one transaction Bundle, mentions Bundle, transaction, references, or posting to a FHIR server. Builds on exporting-to-fhir; pairs after."
license: Apache-2.0
metadata:
  project: OpenMed
  category: fhir-interop
  pairs: after
  version: "1.0"
---

# Assembling FHIR Bundles

A FHIR server ingests one **transaction Bundle**, not loose resources, and the
resources inside it must cross-reference each other (`Condition.subject` →
Patient, `Observation.encounter` → Encounter, `DiagnosticReport.result` →
Observation). OpenMed ships a **deterministic, mechanical** Bundle assembler —
`openmed.clinical.exporters.fhir.to_bundle` — that wraps the resources you built
in `exporting-to-fhir` into a valid R4 Bundle and wires up the references.

## When to use

Use after you have a *list* of standalone resources from `exporting-to-fhir` and
the destination is a FHIR server. Reach for it when the user says "build a
Bundle", "transaction", "POST these resources", or needs internal references
resolved. To check the Bundle against US Core, hand off to `validating-us-core`.

## What OpenMed gives you (verified API)

```python
from openmed.clinical.exporters.fhir import to_bundle, deterministic_fullurl

bundle = to_bundle(
    resources,                       # Sequence[Mapping] each with a resourceType
    doc_id="note-2024-03-02-001",    # seeds stable urn:uuid fullUrls
    bundle_type="transaction",       # "transaction" | "batch" | "collection" | ...
)
```

`to_bundle` does exactly three things, and **never** synthesises or validates:

1. **Deterministic `fullUrl`.** Each resource gets a `urn:uuid` seeded by
   `doc_id` + its index, so the *same input always produces byte-identical
   output* (golden-test friendly). You can pre-compute the same urn with
   `deterministic_fullurl(doc_id, index)`.
2. **Reference rewriting.** Any `{"reference": "ResourceType/id"}` whose target
   is present in the Bundle is repointed at that resource's `fullUrl`. References
   to resources *absent* from the Bundle (e.g. a Patient removed by
   de-identification) are left untouched — no dangling internal refs.
3. **Request blocks.** For `transaction`/`batch` bundles each entry gets a
   `request` block (`{"method": "POST", "url": "<ResourceType>"}`) so the server
   knows to create it.

It **raises** `ValueError` if a resource lacks `resourceType`, or if two
resources share the same `ResourceType/id` (duplicate ids would silently corrupt
the reference map).

## Quick start

```python
from openmed.clinical.exporters.fhir import to_bundle
from openmed.clinical.exporters.codeable_concept_simple import coding, codeable_concept

condition = {
    "resourceType": "Condition", "id": "cond-1",
    "clinicalStatus": {"coding": [{
        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
        "code": "active"}]},
    "code": codeable_concept(
        [coding("snomed", "44054006", "Diabetes mellitus type 2")],
        text="type 2 diabetes"),
    "subject": {"reference": "Patient/patient-1"},   # internal ref, rewritten
}
medication = {
    "resourceType": "MedicationStatement", "id": "med-1", "status": "active",
    "medicationCodeableConcept": codeable_concept(
        [coding("rxnorm", "860975", "metformin 500 MG Oral Tablet")],
        text="metformin 500 mg"),
    "subject": {"reference": "Patient/patient-1"},
}
patient = {
    "resourceType": "Patient", "id": "patient-1",
    "gender": "unknown",                              # de-identified, synthetic
}

bundle = to_bundle([patient, condition, medication],
                   doc_id="demo-note", bundle_type="transaction")
```

## Worked: the transaction Bundle

```json
{
  "resourceType": "Bundle",
  "type": "transaction",
  "entry": [
    {
      "fullUrl": "urn:uuid:6f1c...e2",
      "resource": { "resourceType": "Patient", "id": "patient-1", "gender": "unknown" },
      "request": { "method": "POST", "url": "Patient" }
    },
    {
      "fullUrl": "urn:uuid:9a3b...77",
      "resource": {
        "resourceType": "Condition", "id": "cond-1",
        "subject": { "reference": "urn:uuid:6f1c...e2" }
      },
      "request": { "method": "POST", "url": "Condition" }
    },
    {
      "fullUrl": "urn:uuid:c0d4...19",
      "resource": {
        "resourceType": "MedicationStatement", "id": "med-1",
        "subject": { "reference": "urn:uuid:6f1c...e2" }
      },
      "request": { "method": "POST", "url": "MedicationStatement" }
    }
  ]
}
```

Note `Condition.subject` and `MedicationStatement.subject` were rewritten from
`"Patient/patient-1"` to the Patient entry's `fullUrl` — that is what makes the
transaction resolvable in a single POST.

## Workflow

1. Build resources with `exporting-to-fhir`; give each a unique `id`.
2. Put any resource you reference *inside the same Bundle* — including the
   Patient — so the reference resolves. References to resources you intend to be
   already on the server (e.g. an existing Patient) are left as literal
   `"Patient/<id>"`; resolve those with **conditional create** (below).
3. Call `to_bundle(resources, doc_id=<stable>, bundle_type="transaction")`.
4. POST the whole Bundle to the server base: `POST [base] {Bundle}`.
5. Validate first against US Core (`validating-us-core`).

### Conditional create (don't duplicate an existing Patient)

`to_bundle` writes `POST <ResourceType>` request blocks. To make a transaction
*idempotent*, post-process the entry's `request` to add an `ifNoneExist` query
so the server reuses an existing match instead of creating a duplicate:

```python
for entry in bundle["entry"]:
    if entry["resource"]["resourceType"] == "Patient":
        entry["request"]["ifNoneExist"] = "identifier=http://hospital.example|MRN-REDACTED"
```

The server creates the Patient only if no match exists; otherwise it links the
references to the existing one. `PUT` with a known id is the alternative for
true upserts.

## Hand-off to / from OpenMed

- **From OpenMed:** the resource list comes from `exporting-to-fhir`, which in
  turn comes from `openmed.analyze_text`. Keep `doc_id` stable per source
  document so re-running the pipeline yields the same Bundle.
- **De-identify a built Bundle:** `openmed.interop.fhir_operations.de_identify_bundle(bundle)`
  walks every entry's free text + XHTML narrative and de-identifies it while
  preserving Bundle `type`, entry order, `fullUrl`s, `request` blocks, and
  references — codes, systems, and temporal values are never altered. Use it as
  a final safety pass before transmission if any narrative might carry PHI.
- **OperationOutcome:** a transaction either fully succeeds or fully fails; the
  server returns a Bundle of responses (or an `OperationOutcome` on error).
  Surface those to the user; for *your own* pre-flight findings use
  `to_operation_outcome(...)` from the same package.

## Edge cases & gotchas

- **Resources without an `id`** are valid but **unreferenceable** — nothing can
  point at them and they will not be reference-rewrite targets.
- **Duplicate `ResourceType/id` raises.** This is intentional: a duplicate id
  would silently overwrite an entry in the reference map and corrupt
  cross-references. Make ids unique.
- **External references are left alone.** Only references whose target is *in
  the Bundle* are rewritten; a `"Patient/existing-123"` you mean to resolve on
  the server stays literal — pair it with `ifNoneExist` or a `PUT`.
- **`transaction` vs `batch`.** `transaction` is atomic (all-or-nothing, server
  resolves `urn:uuid` references); `batch` is independent per-entry and does not
  guarantee reference resolution. Use `transaction` when entries reference each
  other.
- **`collection`/`document` bundles** get no `request` blocks (only
  `transaction`/`batch` do) — correct, since they are not meant to be POSTed for
  creation.
- **The assembler does not validate profiles.** Run `validating-us-core` before
  submission.

## Standards & references

- FHIR R4 Bundle: https://hl7.org/fhir/R4/bundle.html
- Transactions & batches: https://hl7.org/fhir/R4/http.html#transaction
- Bundle references / `fullUrl` resolution: https://hl7.org/fhir/R4/bundle.html#references
- Conditional create (`ifNoneExist`): https://hl7.org/fhir/R4/http.html#cond-update
