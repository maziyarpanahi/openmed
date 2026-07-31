---
name: etl-to-omop-cdm
description: "Map OpenMed-extracted, terminology-coded conditions, drugs, and measurements into OMOP CDM v5.4 clinical tables (condition_occurrence, drug_exposure, measurement) for OHDSI/ATLAS analytics. Use when the user wants to load NLP-derived facts into an OMOP database, build an OHDSI ETL from clinical notes, populate condition_occurrence or drug_exposure from text, or standardize note-derived findings to OMOP standard concepts. Covers the source-to-standard concept mapping pattern, required vs optional CDM fields, type concepts for NLP-derived rows, and the user-supplied OHDSI vocabulary (CONCEPT/CONCEPT_RELATIONSHIP). Consumes coded OpenMed analyze_text output (after SNOMED/RxNorm/LOINC linking) and produces OMOP-conformant rows."
license: Apache-2.0
metadata:
  project: OpenMed
  category: analytics-reporting
  pairs: after
  version: "1.0"
---

# ETL to OMOP CDM

The OMOP Common Data Model (CDM) is the OHDSI standard for observational health
data. This skill maps **OpenMed-derived clinical facts** — entities from
`analyze_text` that you have already linked to a source terminology — into the
OMOP clinical event tables `condition_occurrence`, `drug_exposure`, and
`measurement`. The NLP runs **on-device**; OMOP loading is a downstream,
deterministic transform.

## When to use this skill

After you have (a) extracted entities with OpenMed and (b) coded them to a
source vocabulary (ICD-10-CM / SNOMED for conditions, RxNorm for drugs, LOINC
for labs — see the linking skills). Use this skill to turn those coded facts
into OMOP rows. It is *not* a clinical NER skill and *not* a code-linking skill;
it assumes both are done.

## Quick start

```python
import openmed

note = "Assessment: type 2 diabetes mellitus. Started metformin 500 mg PO BID. HbA1c 8.2%."
result = openmed.analyze_text(note, output_format="dict")
# result["entities"] -> [{text,label,confidence,start,end}, ...]

# You then code each entity to a SOURCE concept using the OHDSI vocabulary you
# downloaded (see linking-umls-concepts / normalizing-rxnorm / mapping-loinc),
# and map SOURCE -> STANDARD via CONCEPT_RELATIONSHIP ('Maps to').
fact = {
    "person_id": 1001,
    "domain": "Condition",
    "source_code": "E11.9",          # ICD-10-CM, from your coding step
    "source_vocabulary": "ICD10CM",
    "source_concept_id": 45533010,    # OHDSI CONCEPT for E11.9 (lookup)
    "standard_concept_id": 201826,    # 'Maps to' -> SNOMED 'Type 2 diabetes mellitus'
    "start_date": "2024-03-12",       # from building-patient-timelines
    "char_span": (fact_start, fact_end),
}
```

OpenMed never ships UMLS/SNOMED/RxNorm/LOINC content. You supply the OHDSI
**vocabulary bundle** (Athena download) and do the lookups under your own
license. OpenMed provides the spans and labels.

## The source → standard pattern (the heart of OMOP)

Every clinical event row carries **two** concept ids:

1. `*_source_concept_id` — the OHDSI CONCEPT for your original code (e.g. the
   ICD-10-CM or RxNorm code your linking step produced).
2. `*_concept_id` — the **standard** concept, obtained by following
   `CONCEPT_RELATIONSHIP.relationship_id = 'Maps to'` from the source concept.
   Conditions standardize to **SNOMED**, drugs to **RxNorm**, measurements to
   **LOINC**. If no mapping exists, set the standard id to `0`.

## Domain → table → fields

See `references/omop_cdm_v5_4_fields.md` for the full per-table field list. Core
mapping by OpenMed entity domain:

| OpenMed entity domain | OMOP table | Standard vocab | Key date / value fields |
| --- | --- | --- | --- |
| Disease / Condition | `condition_occurrence` | SNOMED | `condition_start_date`, optional `condition_end_date` |
| Drug / Medication | `drug_exposure` | RxNorm | `drug_exposure_start_date`, `drug_exposure_end_date`, `quantity`, `sig` |
| Lab / Measurement | `measurement` | LOINC | `measurement_date`, `value_as_number`, `unit_concept_id`, `value_as_concept_id` |

## Type concepts: mark rows as NLP-derived

Every event row needs a `*_type_concept_id` recording **provenance**. For facts
derived from clinical text, OHDSI uses the type concept **32831 "EHR episode
record" / "Note"** family — specifically prefer a *"...from note"* /
*"NLP"*-flavored standard type concept from the `Type Concept` vocabulary in
your bundle. Do not invent ids; resolve the type concept against the vocabulary
you loaded so cohort builders can filter NLP-derived rows.

## Workflow

1. **Extract & code.** `analyze_text` → entities; link each to a source code
   (linking skills). Resolve `source_concept_id` and the `'Maps to'` standard
   concept from your Athena vocabulary.
2. **Resolve dates.** Attach start (and end, where known) dates per
   `building-patient-timelines`. OMOP date fields are `DATE`; keep the matching
   `*_datetime` only if you truly have a time.
3. **Assign `person_id`.** Join to your `person` table by an internal key — not
   by any PHI string. De-identify upstream.
4. **Build rows** with required keys, the source+standard concept pair, the
   NLP type concept, and a unique surrogate `*_occurrence_id` /
   `*_exposure_id` / `measurement_id`.
5. **Stage `*_source_value`** (the raw surface string, after de-id) for QA
   traceability — but never put raw PHI there.
6. **Conform & validate.** Run OHDSI **Achilles**/**DataQualityDashboard** on
   the loaded CDM before analytics.

## Hand-off to / from OpenMed

- **From OpenMed:** `analyze_text` entities (offsets + labels), `deidentify`
  upstream, and the per-domain linking skills (`linking-umls-concepts`,
  `normalizing-rxnorm`, `mapping-loinc`, `mapping-to-snomed`,
  `coding-icd10`).
- **To OHDSI:** loaded `condition_occurrence` / `drug_exposure` / `measurement`
  rows are consumed by ATLAS, Achilles, and cohort definitions — and by
  `computing-ecqms` for measure denominators/numerators.

## Edge cases & gotchas

- **No license bundling.** OpenMed does not include SNOMED/RxNorm/LOINC/UMLS.
  Download the OHDSI vocabularies from Athena and run lookups under your own
  agreement. This is a hard rule.
- **Unmapped → standard concept 0.** When `'Maps to'` yields nothing, set
  `*_concept_id = 0` and keep the source ids. Never fabricate a standard id.
- **Domain routing follows the standard concept's domain**, not the source
  code's apparent type. An ICD-10 code can map to a SNOMED concept whose
  `domain_id` is `Observation` or `Measurement` — load it into the table the
  standard concept dictates.
- **NLP rows are lower-assurance.** Tag them with the NLP/`from note` type
  concept and carry `confidence` (e.g. in a companion table) so analysts can
  threshold. Don't silently mix them with structured EHR rows.
- **Dates are required and must be valid.** Undated note facts can't populate a
  `*_start_date`; route them to your "needs review" staging, not into the CDM
  with a placeholder date.
- **`measurement` units and values.** Parse `value_as_number` + `unit` (mapped
  to a `unit_concept_id`); for qualitative results use `value_as_concept_id`.

## Standards & references

- OMOP CDM v5.4 specification: https://ohdsi.github.io/CommonDataModel/cdm54.html
- OMOP standardized vocabularies & the 'Maps to' relationship:
  https://ohdsi.github.io/CommonDataModel/vocabulary.html
- OHDSI Athena vocabulary download (user-supplied): https://athena.ohdsi.org/
- The Book of OHDSI (ETL & vocabulary chapters): https://ohdsi.github.io/TheBookOfOhdsi/
- Data Quality Dashboard: https://ohdsi.github.io/DataQualityDashboard/
- OpenMed source: `openmed/processing/` (`analyze_text` output shape).
