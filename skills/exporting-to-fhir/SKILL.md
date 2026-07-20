---
name: exporting-to-fhir
description: "Convert OpenMed NER output (entities from openmed.analyze_text) into FHIR R4 resources — Condition, MedicationStatement, Observation — using OpenMed's built-in FHIR R4 export helpers in openmed.clinical.exporters. Covers the verified CodeableConcept builder (coding, codeable_concept, system_uri), deterministic fullUrl references, and OperationOutcome reporting. Use after running OpenMed NER when the user wants standards-conformant FHIR JSON, mentions FHIR, Condition/Observation/MedicationStatement, CodeableConcept, RxNorm/LOINC/ICD-10/SNOMED coding, or interoperability with an EHR. Pairs after extracting-clinical-entities; feeds assembling-fhir-bundles and validating-us-core."
license: Apache-2.0
metadata:
  project: OpenMed
  category: fhir-interop
  pairs: after
  version: "1.0"
---

# Exporting to FHIR

OpenMed's NER (`openmed.analyze_text`) returns spans — text, label, offsets,
confidence. To make those spans interoperable you wrap each clinically relevant
span in a **FHIR R4 resource** (`Condition`, `MedicationStatement`,
`Observation`, ...) carrying a coded `CodeableConcept`. OpenMed ships the
**mechanical** R4 export helpers for this in `openmed.clinical.exporters`; you
own the small amount of clinical mapping (which span becomes which resource).

## When to use

Use this after NER, when the consumer is a FHIR system (an EHR, a registry, a
data lake on FHIR). Reach for it when the user says "export to FHIR", "make a
Condition/Observation", "build a CodeableConcept", or needs RxNorm/LOINC/ICD-10/
SNOMED-coded resources. For packaging many resources into one transaction
Bundle, hand off to `assembling-fhir-bundles`. To check the result against US
Core, hand off to `validating-us-core`.

## What OpenMed gives you (verified API)

OpenMed deliberately ships the *purely mechanical* pieces and leaves clinical
judgement to you. The verified entry points:

```python
# CodeableConcept builder — openmed/clinical/exporters/codeable_concept_simple.py
from openmed.clinical.exporters.codeable_concept_simple import (
    system_uri,        # vocab id -> canonical HL7 system URI
    coding,            # (system, code, display) -> Coding dict
    codeable_concept,  # [Coding, ...] -> CodeableConcept dict (deterministic order)
)

# Bundle + reference + OperationOutcome — openmed/clinical/exporters/fhir/
from openmed.clinical.exporters.fhir import (
    to_bundle,                 # [resource, ...] -> R4 transaction Bundle
    deterministic_fullurl,     # (doc_id, index) -> stable urn:uuid
    OperationOutcomeIssue,     # issue dataclass
    to_operation_outcome,      # [issue, ...] -> OperationOutcome
    from_validation_result,    # validator result -> OperationOutcome
)
```

`system_uri` knows these vocabularies out of the box: `rxnorm`, `icd-10-cm`,
`loinc`, `snomed`, `hpo`, `mesh` (and passes through any `http(s)://` URI
unchanged). It is the single source of truth for vocab-id → system-URI mapping.

There is **no** `to_condition()` / `to_observation()` magic function. You build
the resource shell (a small dict) and drop a `codeable_concept(...)` into its
coded slot. That is by design: the resource *type* and *clinical status* are
decisions OpenMed will not make for you.

## Quick start: entity → Condition

```python
import openmed
from openmed.clinical.exporters.codeable_concept_simple import coding, codeable_concept

# 1) NER (synthetic note — no real PHI)
result = openmed.analyze_text(
    "Assessment: type 2 diabetes mellitus, stable.",
    model_name="disease_detection_superclinical",
)
# result.entities -> EntityPrediction(text, label, confidence, start, end)
span = result.entities[0]            # e.g. text="type 2 diabetes mellitus"

# 2) Ground to a code OUT OF PROCESS (your terminology server / mapping table).
#    OpenMed bundles no restricted vocab — see querying-terminology-service.
icd_code, snomed_code = "E11.9", "44054006"

# 3) Build the CodeableConcept with OpenMed's builder
cc = codeable_concept(
    [
        coding("snomed", snomed_code, "Diabetes mellitus type 2"),
        coding("icd-10-cm", icd_code, "Type 2 diabetes mellitus without complications"),
    ],
    text=span.text,
)

# 4) Assemble the resource shell yourself
condition = {
    "resourceType": "Condition",
    "id": "cond-1",
    "clinicalStatus": {
        "coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
            "code": "active",
        }]
    },
    "verificationStatus": {
        "coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
            "code": "confirmed",
        }]
    },
    "category": [{
        "coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/condition-category",
            "code": "encounter-diagnosis",
        }]
    }],
    "code": cc,                                  # OpenMed-built CodeableConcept
    "subject": {"reference": "Patient/patient-1"},
    "recordedDate": "2024-03-02",
}
```

`codeable_concept` sorts codings deterministically (SNOMED, LOINC, RxNorm,
ICD-10-CM, HPO, MeSH first; everything else alphabetical), so the JSON is
byte-stable across runs — important for diffable pipelines and golden tests.

## Worked: the resulting `Condition.code`

```json
{
  "code": {
    "coding": [
      { "system": "http://snomed.info/sct", "code": "44054006",
        "display": "Diabetes mellitus type 2" },
      { "system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "E11.9",
        "display": "Type 2 diabetes mellitus without complications" }
    ],
    "text": "type 2 diabetes mellitus"
  }
}
```

## Workflow

1. **NER** — `openmed.analyze_text(note, model_name=...)` → `result.entities`.
2. **Classify** each span: a disease label → `Condition`; a drug → `MedicationStatement`;
   a lab/vital/measurement → `Observation`.
3. **Ground** the surface text to a code out of process (terminology server or
   your own map). Never invent codes; if you cannot ground a span, emit a
   `CodeableConcept` with only `text` and no `coding`.
4. **Build** the `CodeableConcept` with `coding(...)` + `codeable_concept(...)`.
5. **Wrap** it in the resource shell (set `clinicalStatus`/`status`, `subject`,
   dates). Use the cheat-sheet below.
6. **Reference** the Patient/Encounter via `{"reference": "Patient/<id>"}`.
7. Pass the list to `to_bundle(...)` (`assembling-fhir-bundles`) and validate
   (`validating-us-core`).

### Resource cheat-sheet (where the CodeableConcept goes)

| OpenMed entity kind | FHIR resource | Coded slot | Required status field |
| --- | --- | --- | --- |
| Disease / diagnosis | `Condition` | `code` | `clinicalStatus`, `verificationStatus` |
| Drug / medication | `MedicationStatement` | `medicationCodeableConcept` | `status` (e.g. `active`) |
| Lab / vital / finding | `Observation` | `code` (+ `valueQuantity`/`valueCodeableConcept`) | `status` (e.g. `final`) |
| Procedure | `Procedure` | `code` | `status` |
| Allergy | `AllergyIntolerance` | `code` | `clinicalStatus` |

### MedicationStatement (drug span)

```python
med = {
    "resourceType": "MedicationStatement",
    "id": "med-1",
    "status": "active",
    "medicationCodeableConcept": codeable_concept(
        [coding("rxnorm", "860975", "metformin hydrochloride 500 MG Oral Tablet")],
        text="metformin 500 mg",
    ),
    "subject": {"reference": "Patient/patient-1"},
}
```

### Observation (lab/vital span)

```python
obs = {
    "resourceType": "Observation",
    "id": "obs-1",
    "status": "final",
    "category": [{"coding": [{
        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
        "code": "laboratory",
    }]}],
    "code": codeable_concept(
        [coding("loinc", "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood")],
        text="HbA1c",
    ),
    "valueQuantity": {
        "value": 7.4, "unit": "%",
        "system": "http://unitsofmeasure.org", "code": "%",
    },
    "subject": {"reference": "Patient/patient-1"},
}
```

## Hand-off to / from OpenMed

- **From OpenMed:** `result.entities` (`EntityPrediction.text/.label/.confidence
  /.start/.end`) is the input. Keep `confidence` and the offsets in an
  extension or a side log so the resource is auditable back to the source span.
- **To OpenMed:** before exporting a note that still contains PHI, run
  `openmed.deidentify(...)`; or de-identify a *built* resource/Bundle with
  `openmed.interop.fhir_operations.de_identify_resource` /
  `de_identify_bundle` (see that module — it walks free-text + narrative and
  never touches codes, references, systems, or temporal values).
- **OperationOutcome:** report any spans you could not map as
  `OperationOutcomeIssue(severity="warning", code="incomplete", diagnostics=...,
  expression="Condition.code")` → `to_operation_outcome([...])`. Keep
  diagnostics PHI-free (offsets/labels, never raw identifiers).

## Edge cases & gotchas

- **No code? Still valid.** A `CodeableConcept` with only `text` and no
  `coding` is legal R4. Emit it rather than inventing a code, and flag it via
  OperationOutcome. US Core may still require a code — see `validating-us-core`.
- **`system_uri` raises** on an unknown short id that is not a URL. Pass a known
  short id (`rxnorm`/`loinc`/`snomed`/`icd-10-cm`/`hpo`/`mesh`) or a full
  `http(s)://` system URI.
- **Negation / temporality.** OpenMed NER finds the mention, not its assertion.
  A negated ("no diabetes") or historical span should change
  `verificationStatus`/`clinicalStatus` or be dropped. Resolve assertion first
  (`resolving-clinical-context`, `openmed.clinical`).
- **Stable ids.** Give each resource a unique `id`; `to_bundle` rejects
  duplicate `ResourceType/id` pairs because they corrupt cross-references.
- **Local-first.** Grounding to RxNorm/LOINC/SNOMED is out-of-process with the
  user's own credentials. OpenMed bundles no restricted terminology.

## Standards & references

- FHIR R4 Condition: https://hl7.org/fhir/R4/condition.html
- FHIR R4 MedicationStatement: https://hl7.org/fhir/R4/medicationstatement.html
- FHIR R4 Observation: https://hl7.org/fhir/R4/observation.html
- FHIR R4 CodeableConcept: https://hl7.org/fhir/R4/datatypes.html#CodeableConcept
- HL7 terminology systems (system URIs): https://hl7.org/fhir/R4/terminologies-systems.html
- FHIR R4 OperationOutcome: https://hl7.org/fhir/R4/operationoutcome.html
