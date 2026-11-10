# OMOP CDM v5.4 — fields for NLP-derived clinical events

Reference field lists for the three event tables most commonly populated from
OpenMed `analyze_text` output. **R** = required (NOT NULL in the v5.4 spec),
**O** = optional. Authoritative spec:
https://ohdsi.github.io/CommonDataModel/cdm54.html — defer to it on conflicts.

OpenMed ships **no** OHDSI vocabulary. Concept ids below are resolved against
the Athena vocabulary bundle you download under your own license.

## condition_occurrence (conditions / diagnoses → SNOMED standard)

| Field | Req | Source from OpenMed pipeline |
| --- | --- | --- |
| `condition_occurrence_id` | R | surrogate key you generate |
| `person_id` | R | internal patient key (never PHI) |
| `condition_concept_id` | R | STANDARD SNOMED concept via `'Maps to'`; `0` if unmapped |
| `condition_start_date` | R | event date (`building-patient-timelines`) |
| `condition_start_datetime` | O | only if a real time exists |
| `condition_end_date` | O | resolution date if stated |
| `condition_end_datetime` | O | optional |
| `condition_type_concept_id` | R | NLP / "from note" type concept (resolve in your bundle) |
| `condition_status_concept_id` | O | e.g. confirmed/provisional if modeled |
| `stop_reason` | O | usually NULL for NLP |
| `provider_id` | O | if known |
| `visit_occurrence_id` | O | link to the encounter/visit |
| `visit_detail_id` | O | optional |
| `condition_source_value` | O | de-identified surface string for QA |
| `condition_source_concept_id` | O | OHDSI CONCEPT for your ICD-10-CM/SNOMED source code |
| `condition_status_source_value` | O | optional |

## drug_exposure (medications → RxNorm standard)

| Field | Req | Source from OpenMed pipeline |
| --- | --- | --- |
| `drug_exposure_id` | R | surrogate key |
| `person_id` | R | internal patient key |
| `drug_concept_id` | R | STANDARD RxNorm concept via `'Maps to'`; `0` if unmapped |
| `drug_exposure_start_date` | R | med-start date |
| `drug_exposure_start_datetime` | O | optional |
| `drug_exposure_end_date` | R | end date; if unknown, derive from days_supply or set to start |
| `drug_exposure_end_datetime` | O | optional |
| `verbatim_end_date` | O | optional |
| `drug_type_concept_id` | R | NLP / "from note" type concept |
| `stop_reason` | O | if stated |
| `refills` | O | parsed sig, if present |
| `quantity` | O | parsed quantity |
| `days_supply` | O | parsed/derived |
| `sig` | O | de-identified dosing instruction text |
| `route_concept_id` | O | route mapped to standard concept (PO, IV, …) |
| `lot_number` | O | rarely from notes |
| `provider_id` | O | prescriber if known |
| `visit_occurrence_id` | O | link to encounter |
| `drug_source_value` | O | de-identified surface string |
| `drug_source_concept_id` | O | OHDSI CONCEPT for your RxNorm source code |
| `route_source_value` | O | raw route string |
| `dose_unit_source_value` | O | raw unit string |

## measurement (labs / vitals → LOINC standard)

| Field | Req | Source from OpenMed pipeline |
| --- | --- | --- |
| `measurement_id` | R | surrogate key |
| `person_id` | R | internal patient key |
| `measurement_concept_id` | R | STANDARD LOINC concept via `'Maps to'`; `0` if unmapped |
| `measurement_date` | R | result/collection date |
| `measurement_datetime` | O | optional |
| `measurement_time` | O | optional |
| `measurement_type_concept_id` | R | NLP / "from note" type concept |
| `operator_concept_id` | O | `<`, `>`, `=` if parsed |
| `value_as_number` | O | numeric result (e.g. 8.2) |
| `value_as_concept_id` | O | standard concept for qualitative results (positive/negative) |
| `unit_concept_id` | O | standard concept for the unit (%, mg/dL) |
| `range_low` | O | reference low if stated |
| `range_high` | O | reference high if stated |
| `provider_id` | O | optional |
| `visit_occurrence_id` | O | link to encounter |
| `visit_detail_id` | O | optional |
| `measurement_source_value` | O | de-identified surface string |
| `measurement_source_concept_id` | O | OHDSI CONCEPT for your LOINC source code |
| `unit_source_value` | O | raw unit string |
| `value_source_value` | O | raw value string |

## Mapping checklist

1. Resolve `*_source_concept_id` from your source code (ICD-10-CM / RxNorm / LOINC).
2. Follow `CONCEPT_RELATIONSHIP` `'Maps to'` to the STANDARD concept for `*_concept_id`.
3. Route to the table matching the **standard concept's `domain_id`** (not the source code's type).
4. Set the **NLP type concept** so analysts can isolate note-derived rows.
5. Attach valid dates; send undated facts to staging, not the CDM.
6. Keep `*_source_value` de-identified — no raw PHI.
