---
name: defining-cohort-phenotypes
description: "Authors computable phenotype and cohort definitions in the OHDSI ATLAS / CIRCE style over the OMOP CDM, combining standard concept sets with NLP-derived features that OpenMed extracts. Use when the user wants to define a patient cohort, write a computable phenotype, reuse PheKB or OHDSI Phenotype Library logic, build concept sets, or augment code-based criteria with text features. Trigger keywords: phenotype, cohort definition, OHDSI, ATLAS, CIRCE, OMOP CDM, concept set, PheKB, Phenotype Library, eMERGE, computable phenotype. Pairs adjacent to OpenMed: NLP features from openmed.analyze_text augment code-based phenotypes for entities that are poorly captured by structured codes. OMOP CDM and OHDSI tools are open source; restricted vocabularies (SNOMED, CPT) are user-supplied."
license: Apache-2.0
metadata:
  project: OpenMed
  category: research-genomics
  pairs: adjacent
  version: "1.0"
---

# Defining cohort phenotypes (OHDSI / OMOP CDM)

A **computable phenotype** is a portable, executable definition of "which patients
have condition X" — concept sets plus inclusion logic that runs against any
**OMOP CDM**-compliant database. In the OHDSI stack, ATLAS authors these visually,
**CIRCE** serializes them to a standardized **JSON** representation, and that JSON
compiles to database-specific SQL. This skill helps you author such definitions
and **augment them with NLP features** that OpenMed extracts from clinical text —
exactly the signals that structured codes miss.

OMOP CDM, ATLAS, CIRCE, and the OHDSI Phenotype Library are open source. The
*vocabulary content* you reference (SNOMED CT, CPT4, ICD) is **user-supplied** —
do not bundle restricted terminologies; load them into your own OMOP vocabulary
tables with your own licenses.

## When to use

- You need a reproducible cohort definition for analytics or research.
- You want to reuse an existing **PheKB** or **OHDSI Phenotype Library** definition
  and adapt it.
- A phenotype depends on facts that live only in **free text** (e.g. smoking
  status, symptom severity, social context) and code-based logic alone is weak.

For terminology grounding of individual entities, see `coding-icd10`,
`normalizing-rxnorm`, `mapping-loinc`; this skill is about composing them into a
cohort.

## Anatomy of a CIRCE cohort definition

A CIRCE cohort definition JSON has two parts: **ConceptSets** (the code lists) and
an **expression** (entry event + inclusion rules). Shape (abridged):

```jsonc
{
  "ConceptSets": [{
    "id": 0, "name": "Type 2 diabetes",
    "expression": { "items": [{
      "concept": { "CONCEPT_ID": 201826,           // OMOP standard concept
                   "CONCEPT_CODE": "44054006",      // SNOMED (user vocab)
                   "VOCABULARY_ID": "SNOMED" },
      "includeDescendants": true                    // pull the hierarchy
    }] }
  }],
  "PrimaryCriteria": {                              // entry event
    "CriteriaList": [{ "ConditionOccurrence": { "CodesetId": 0 } }],
    "ObservationWindow": { "PriorDays": 0, "PostDays": 0 },
    "PrimaryCriteriaLimit": { "Type": "First" }
  },
  "InclusionRules": [{
    "name": "Adult at index",
    "expression": { "Type": "ALL", "CriteriaList": [{
      "Criteria": { "ConditionEra": { "AgeAtStart": { "Value": 18, "Op": "gte" } } }
    }] }
  }]
}
```

You author this in ATLAS (recommended) or by hand. The OHDSI Phenotype Library
ships hundreds of vetted definitions as exactly this JSON; reuse before you write.

## Augmenting with OpenMed NLP features

Code-based phenotypes are blind to facts that only appear in notes. The pattern is
**materialize an NLP feature as OMOP rows, then reference it like any concept set.**

```python
import openmed

# 1) Extract the text feature OpenMed is good at (e.g. tobacco use, symptom)
note = "Patient is a current smoker, ~1 pack/day, with worsening dyspnea."
res = openmed.analyze_text(note, model_name="disease_detection_superclinical",
                           output_format="dict")

# 2) Write a derived OBSERVATION (or a custom cohort attribute) per patient,
#    mapping each extracted entity to a standard concept (grounded out-of-process).
#    e.g. Observation: "Current smoker" -> a SNOMED concept in your vocab.

# 3) Reference that concept in a CIRCE ConceptSet, so the phenotype combines
#    structured codes AND the NLP-derived flag in one inclusion rule.
```

This mirrors how **eMERGE** and PheKB phenotypes mix structured codes with NLP:
the NLP step contributes high-recall flags for concepts that ICD/CPT capture
poorly, and CIRCE composes them with the rest of the logic.

## Workflow

1. **Start from a library definition** if one exists (OHDSI Phenotype Library /
   PheKB) and adapt; otherwise design entry event + inclusion rules.
2. **Build concept sets** from standard OMOP concepts; set `includeDescendants`
   to capture hierarchies. Vocabulary content comes from your own licensed tables.
3. **Identify text-only criteria** the codes miss; extract them with
   `openmed.analyze_text` and materialize as OMOP rows / cohort attributes.
4. **Assemble** the CIRCE JSON (concept sets + expression) — in ATLAS or directly.
5. **Validate** against OMOP CDM: generate SQL, run on a (synthetic/de-identified)
   database, review cohort counts; iterate with PheValuator-style checks.
6. **Document** human-readable logic alongside the JSON for portability.

## Hand-off to / from OpenMed

- **OpenMed → phenotype features.** `openmed.analyze_text` over notes yields
  Disease, Pharmaceutical, Genomics, Oncology, and social/behavioral spans. Ground
  each to a standard concept (`coding-icd10`, `normalizing-rxnorm`, `mapping-loinc`,
  or your SNOMED map) and write it into OMOP so CIRCE can reference it.
- **Phenotype → OpenMed scope.** A cohort definition tells you *which notes to
  process*: run OpenMed only on the cohort's documents to extract the features the
  phenotype needs, keeping compute and PHI exposure minimal.
- Run **locally** on de-identified or synthetic OMOP data. De-identify notes with
  `openmed.deidentify` before they enter any shared analytics environment.

## Edge cases & gotchas

- **Standard vs source concepts.** OMOP maps source codes (ICD-10-CM) to standard
  concepts (usually SNOMED). Build concept sets on **standard** concepts and let
  the source-to-standard map do the translation, or you will miss rows.
- **Descendants matter.** Forgetting `includeDescendants` silently drops the
  hierarchy (e.g. all diabetes subtypes). Forgetting nothing can over-capture —
  review the resolved concept list.
- **NLP feature provenance.** Tag NLP-derived OMOP rows distinctly (e.g. a
  type_concept indicating "derived from NLP") so analysts know the signal is
  probabilistic, not adjudicated.
- **Vocabulary licensing.** SNOMED CT, CPT4, and similar require their own
  licenses and are **not** redistributed here — load them into your OMOP vocab.
- **Portability ≠ equivalence.** The same JSON runs everywhere, but data capture
  differs by site; validate cohort counts per source before trusting them.
- **Not clinical advice.** Phenotype membership supports research/analytics; it is
  not a diagnosis.

## Standards & references

- OMOP Common Data Model — https://ohdsi.github.io/CommonDataModel/
- The Book of OHDSI (cohorts & phenotypes) — https://ohdsi.github.io/TheBookOfOhdsi/
- ATLAS — https://github.com/OHDSI/Atlas
- CIRCE (cohort expression → SQL) — https://github.com/OHDSI/circe-be
- OHDSI Phenotype Library — https://github.com/OHDSI/PhenotypeLibrary
- PheKB phenotype knowledge base — https://phekb.org/
- WebAPI (programmatic cohort definitions) — https://github.com/OHDSI/WebAPI
