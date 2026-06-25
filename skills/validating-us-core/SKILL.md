---
name: validating-us-core
description: "Validate FHIR R4 resources and Bundles against US Core / USCDI profiles with the official HL7 FHIR validator before submitting to an EHR. Covers running validator_cli.jar (or the public validator.fhir.org), declaring meta.profile, must-support elements, common conformance gaps (missing code/category/status), and turning validator output into a FHIR OperationOutcome. Use after exporting-to-fhir / assembling-fhir-bundles to check OpenMed-produced FHIR for US Core conformance, when the user mentions US Core, USCDI, must-support, profile validation, or Epic/Cerner ingestion requirements. Pairs after."
license: Apache-2.0
metadata:
  project: OpenMed
  category: fhir-interop
  pairs: after
  version: "1.0"
---

# Validating US Core

Producing *syntactically* valid R4 (which `exporting-to-fhir` and
`assembling-fhir-bundles` do) is not the same as conforming to **US Core** — the
HL7 US realm profiles that EHRs (Epic, Cerner/Oracle Health) require for
ingestion and that USCDI mandates for certified exchange. This skill validates
OpenMed-produced FHIR against US Core *before* you submit it.

## When to use

Use it as the gate right before submission, after you have assembled a Bundle.
Reach for it when the user says "US Core", "USCDI", "must-support", "will Epic
accept this", or "validate my FHIR". It is the conformance counterpart to the
mechanical builders — OpenMed builds the JSON; the HL7 validator judges it.

## Quick start: run the official validator

The reference implementation is the HL7 `validator_cli.jar` (the same engine
behind https://validator.fhir.org). Validate against the US Core package by IG:

```bash
# One-time: get the validator
curl -L -o validator_cli.jar \
  https://github.com/hapifhir/org.hl7.fhir.core/releases/latest/download/validator_cli.jar

# Validate a resource/Bundle against the current US Core IG
java -jar validator_cli.jar condition.json \
  -version 4.0.1 \
  -ig hl7.fhir.us.core \
  -tx https://tx.fhir.org            # terminology server for code validation
```

`-ig hl7.fhir.us.core` pulls the current published US Core package; pin a
version (e.g. `-ig hl7.fhir.us.core#6.1.0`) for reproducible CI. The validator
exits non-zero on errors and prints issues with FHIRPath locations.

For ad-hoc checks without a JVM, paste the JSON into the public validator UI at
https://validator.fhir.org (do **not** paste real PHI — validate synthetic or
de-identified resources only).

## Declare the profile you claim

US Core only validates against a profile if the resource *claims* it via
`meta.profile`. Add the canonical URL for the profile you target:

```json
{
  "resourceType": "Condition",
  "meta": {
    "profile": [
      "http://hl7.org/fhir/us/core/StructureDefinition/us-core-condition-problems-health-concerns"
    ]
  }
}
```

Then `java -jar validator_cli.jar condition.json -ig hl7.fhir.us.core` checks it
against that profile's constraints, including **must-support** elements.

## Common conformance gaps (from OpenMed output)

OpenMed NER gives you the clinical mention; US Core wants structured context.
The recurring gaps when going from raw spans to US Core:

| Gap | US Core expects | Fix in the exporter |
| --- | --- | --- |
| Missing `code.coding` | A coded value (SNOMED/ICD-10 for Condition; LOINC for Observation; RxNorm for medication) | Ground the span; `codeable_concept([...])` with a real `coding`, not just `text` |
| Missing `category` | `encounter-diagnosis`/`problem-list-item` (Condition), `laboratory`/`vital-signs` (Observation) | Set `category` in the resource shell |
| Missing `clinicalStatus` / `status` | Required status fields | Set them per the `exporting-to-fhir` cheat-sheet |
| Missing `subject` | A resolvable Patient reference | Reference an in-Bundle Patient; let `to_bundle` rewrite it |
| Unbound `valueQuantity.code` | UCUM unit code | Use `system: http://unitsofmeasure.org` + UCUM code |
| Vital signs not on the vitals profile | `us-core-vital-signs` shape (LOINC code, `vital-signs` category) | Use the vitals LOINC + category |

"Must-support" means the producer **must** populate the element when the data
exists. The validator flags must-support omissions as warnings; certified
systems may reject them.

## Workflow

1. Export + assemble the Bundle (`exporting-to-fhir`, `assembling-fhir-bundles`).
2. Add `meta.profile` for the US Core profile each resource targets.
3. Run `validator_cli.jar` with `-ig hl7.fhir.us.core` and a `-tx` server.
4. Read the issues: **error** = will be rejected; **warning** = must-support /
   best practice. Fix errors in the exporter, not by hand-editing JSON.
5. Re-validate until clean; wire the validator into CI on synthetic fixtures.
6. Submit (`assembling-fhir-bundles` for the transaction POST).

### Turn validator output into an OperationOutcome

If you run validation programmatically, adapt the result into a FHIR
`OperationOutcome` with OpenMed's helper so the rest of your pipeline speaks one
shape:

```python
from openmed.clinical.exporters.fhir import from_validation_result

# `result` exposes issues, or errors/warnings/information buckets
outcome = from_validation_result(result)   # -> R4 OperationOutcome dict
```

`from_validation_result` understands either an `issues` collection or
`errors`/`warnings`/`information` buckets (strings or issue objects) and emits a
clean R4 `OperationOutcome` (all-ok when empty). It only reads structural
metadata — keep diagnostics PHI-free.

## Hand-off to / from OpenMed

- **Validate OpenMed-produced FHIR:** the input is the Bundle from
  `assembling-fhir-bundles`; the output is conformance issues you fix back in
  `exporting-to-fhir`.
- **OperationOutcome bridge:** `from_validation_result` /
  `to_operation_outcome` / `OperationOutcomeIssue` (all in
  `openmed.clinical.exporters.fhir`) convert validator findings to R4.
- **No PHI in validation:** validate synthetic or de-identified resources. If a
  narrative might carry PHI, run `openmed.interop.fhir_operations.de_identify_bundle`
  first.

## Edge cases & gotchas

- **No `meta.profile`, no profile check.** The validator validates base R4 only
  unless the resource claims the profile (or you force it with `-profile <url>`).
- **Terminology binding needs a `-tx` server.** Without `-tx`, code-system /
  value-set bindings are not fully checked; many US Core `required` bindings
  will be missed. Point `-tx` at `https://tx.fhir.org` or your own Ontoserver.
- **Pin the IG version** in CI (`hl7.fhir.us.core#<version>`). US Core revisions
  change must-support and bindings; an unpinned run drifts.
- **Reference resolution in Bundles.** Validate the *whole Bundle* so
  `urn:uuid` references resolve; validating a lone resource flags references it
  cannot see.
- **USCDI ≠ US Core.** USCDI is the data-element regulation; US Core is the FHIR
  profile set that implements it. Conform to the US Core profile for the
  matching USCDI class.
- **Warnings can still block ingestion.** Some EHRs reject must-support
  omissions even though the validator calls them warnings. Treat must-support as
  required for production.

## Standards & references

- US Core Implementation Guide: https://hl7.org/fhir/us/core/
- US Core profiles list: https://hl7.org/fhir/us/core/profiles-and-extensions.html
- USCDI: https://www.healthit.gov/isp/united-states-core-data-interoperability-uscdi
- HL7 FHIR validator (CLI + docs): https://confluence.hl7.org/display/FHIR/Using+the+FHIR+Validator
- Public validator: https://validator.fhir.org
- Public terminology server: https://tx.fhir.org
