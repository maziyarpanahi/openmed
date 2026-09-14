# FHIR-to-OMOP round-trip conformance

OpenMed's bridge is a local-first, synthetic-fixture conformance slice for
FHIR R4 and OMOP CDM v5.4. It accepts caller-supplied vocabulary records only;
it does not bundle OMOP, UMLS, SNOMED CT, CPT, or any other restricted
terminology. A missing mapping always remains `concept_id = 0` with an
explicit reason.

## Field-level mapping matrix

| FHIR resource / element | OMOP target | Preserved sidecar fields | Round-trip status |
| --- | --- | --- | --- |
| `Patient.id` | `person.person_id` | `source_id_hash`, deterministic person key | Identifier is pseudonymized; patient demographics are intentionally lossy. |
| `Encounter.id`, `Encounter.subject.reference`, `Encounter.period.start` | `visit_occurrence` | `source_id_hash`, `visit_source_hash`, period values | Supported for visit identity and period. |
| `Encounter.class` | `visit_occurrence.visit_concept_id` | `fhir_code` coding | Code remains source-only unless a user mapping is supplied. |
| `Condition.code` | `condition_occurrence` and `note_nlp` | `element_path`, source/target concept IDs, vocabulary snapshot, offsets | Supported coded content. |
| `Condition.clinicalStatus`, `Condition.verificationStatus` | `fhir_code` | Coded content and path | Reconstructed as FHIR status `CodeableConcept`s. |
| `Condition.onsetDateTime` | `condition_occurrence.condition_start_date` | `date_path` | Supported date precision as supplied. |
| `Observation.code` | `measurement` and `note_nlp` | `element_path`, source/target concept IDs, vocabulary snapshot, offsets | Supported coded content. |
| `Observation.valueQuantity` | `fhir_value` | Number, unit, UCUM system/code | Reconstructed when present. |
| `Observation.valueInteger`, `valueDecimal`, `valueBoolean`, `valueDateTime` | `fhir_value` | Typed scalar | Reconstructed when present. |
| `Observation.valueCodeableConcept` | `fhir_code` | Coded content and path | Reconstructed as a coded value. |
| `Procedure.code` | `procedure_occurrence` and `note_nlp` | `element_path`, source/target concept IDs, vocabulary snapshot, offsets | Supported coded content. |
| `MedicationStatement.medicationCodeableConcept` | `drug_exposure` and `note_nlp` | `element_path`, source/target concept IDs, vocabulary snapshot, offsets | Supported coded content. |
| `MedicationRequest.medicationCodeableConcept` | `drug_exposure` and `note_nlp` | `element_path`, source/target concept IDs, vocabulary snapshot, offsets | Supported coded content. |
| `MedicationStatement.encounter.reference`, `MedicationRequest.encounter.reference` | `visit_occurrence_id` | Hash-only encounter reference | Supported as a de-identified visit link. |

Every clinical row has a deterministic `idempotent_key`, a `source_note_hash`,
and a NOTE_NLP link. The `fhir_provenance` sidecar links that row to the
hash-only source resource, source element path, source and target concept
distinction, vocabulary snapshot, and exact NOTE_NLP offsets.

## Unsupported elements and defined loss

The bridge records paths, never discarded source values, in
`fhir_information_loss`. The current intentionally lossy list is:

- Patient names, identifiers, telecom, gender, birth date, and address;
- Encounter participants, locations, service provider, and diagnoses;
- Condition abatement, recorder, asserter, evidence, and notes;
- Observation category, performer, interpretation, body site, method,
  specimen, device, reference range, members, provenance links, components,
  and `valueString`;
- Procedure period, performer, location, reason, and outcome;
- Medication statement periods, information source, dosage, reasons, and
  notes;
- Medication request requester, dosage, dispense request, substitution, prior
  prescription, reasons, and notes;
- original resource identifiers and references, which are replaced with
  deterministic hashes in the reconstructed de-identified Bundle;
- FHIR narrative `text` and free-form CodeableConcept text that does not
  duplicate a coding display.

This is a declared supported subset, not a claim of full FHIR or OHDSI
certification. Unsupported resources fail closed rather than being silently
assigned to an OMOP domain.

## Synthetic conformance report

The committed fixture under
`tests/fixtures/interop/fhir_omop_bundle.json` covers Patient, Encounter,
Condition, Observation, MedicationRequest, and Procedure. The corresponding
user-supplied vocabulary snapshot maps the condition, HbA1c, and medication
codes; the procedure code intentionally remains unmapped. The offline report
checks:

1. deterministic reload keys and row counts;
2. OMOP concept foreign-key resolution, including concept zero;
3. source resource, element path, vocabulary snapshot, and NOTE_NLP offset
   traceability;
4. an explicit unmapped reason for every concept-zero coding; and
5. reconstruction of supported coded content.

`tests/fixtures/interop/fhir_omop_fabricated_concept.json` is a negative
fixture. It embeds an OMOP concept identifier in FHIR input; the loader rejects
it and the conformance report fails with `fabricated_concept_identifier`.

Use the focused checks with:

```bash
.venv/bin/python -m pytest tests/unit/interop/test_fhir_omop.py -q
```
