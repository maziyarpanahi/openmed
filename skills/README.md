# OpenMed Skills

Portable [Agent Skills](https://agentskills.io) for building with **OpenMed** — the on-device, Apache-2.0 clinical & biomedical NLP library. Each skill is a folder with a `SKILL.md` that works unchanged in **Claude Code**, **OpenAI Codex**, **OpenCode**, and any agent on the open standard. Drop them in and your coding agent learns to wire up OpenMed pipelines — de-identification, NER, FHIR export, evaluation — plus the upstream/downstream healthcare tasks around them.

**68 skills** across 13 categories.

## Get running in one command

The same `SKILL.md` folders work unchanged in **Claude Code**, **OpenAI Codex**, **OpenCode**, and any agent that follows the [open standard](https://agentskills.io). Install into every agent you have at once:

```bash
git clone https://github.com/maziyarpanahi/openmed && cd openmed
./install-skills.sh          # -> Claude Code, Codex, OpenCode, and ~/.agents/skills
```

Or target one agent:

| Agent | Command | Skills directory |
| --- | --- | --- |
| **Claude Code** | `./install-skills.sh claude` | `~/.claude/skills/` |
| **OpenAI Codex** | `./install-skills.sh codex` | `~/.codex/skills/` |
| **OpenCode** | `./install-skills.sh opencode` | `~/.config/opencode/skills/` |
| **Any other agent** | `./install-skills.sh agents` | `~/.agents/skills/` |

No clone? Copy the folders directly with `cp -r skills/*/ ~/.claude/skills/` (swap the path per agent). Claude Code users can also install as a plugin, no clone needed:

```text
/plugin marketplace add maziyarpanahi/openmed
/plugin install openmed-skills@openmed-skills
```

## Try it in 30 seconds

After installing, just ask your agent in plain language — it finds the right skill and writes correct, on-device OpenMed code for you:

> **You:** De-identify this discharge note and pull out the medications with OpenMed — *"Pt John Doe (MRN 12345), seen 2024-03-02, started on metformin 500mg BID."*

> **Your agent** loads `deidentifying-clinical-text` + `extracting-clinical-entities` and produces:

```python
import openmed
note = "Pt John Doe (MRN 12345), seen 2024-03-02, started on metformin 500mg BID."
deid = openmed.deidentify(note, policy="hipaa_safe_harbor")   # PHI removed on-device
meds = openmed.analyze_text(deid.deidentified_text,
                            model_name="pharma_detection_superclinical")
```

→ Name, MRN, and date are redacted locally and `metformin 500mg BID` is returned as a medication entity. No cloud call, no PHI leaves the machine.

New here? Start with **[building-with-openmed](building-with-openmed/SKILL.md)** — it maps every task to the right skill and the real OpenMed API.

## Catalog

Legend: `→ before` runs upstream of OpenMed, `after →` consumes its output, `↔ adjacent` is a neighbouring task.

### OpenMed core — build with OpenMed directly

| Skill | What it does | Pairs |
| --- | --- | --- |
| [`building-with-openmed`](building-with-openmed/SKILL.md) | Orient and bootstrap any project that uses OpenMed, the on-device clinical and biomedical NLP library, for named-entity recognition, PHI …. | ↔ adjacent |
| [`choosing-openmed-models`](choosing-openmed-models/SKILL.md) | Discover and pick the right OpenMed model for a clinical or biomedical task, domain, or language. | ↔ adjacent |
| [`deidentifying-clinical-text`](deidentifying-clinical-text/SKILL.md) | Remove, mask, or replace PHI/PII in clinical free text on-device with OpenMed's deidentify(). | ↔ adjacent |
| [`extracting-clinical-entities`](extracting-clinical-entities/SKILL.md) | Run clinical and biomedical named-entity recognition on medical text with OpenMed's analyze_text. | ↔ adjacent |
| [`extracting-pii-entities`](extracting-pii-entities/SKILL.md) | Detect PHI/PII spans in clinical text with OpenMed's extract_pii without altering the text. | ↔ adjacent |
| [`loading-openmed-models`](loading-openmed-models/SKILL.md) | Load OpenMed clinical/biomedical NER models from the Hugging Face Hub or a local path and reuse them efficiently across calls. | ↔ adjacent |
| [`reidentifying-text`](reidentifying-text/SKILL.md) | Reversibly de-identify clinical text with OpenMed and later restore the original PHI from a saved mapping. | ↔ adjacent |
| [`running-zeroshot-ner`](running-zeroshot-ner/SKILL.md) | Extract arbitrary, custom entity types from clinical or biomedical text with no fine-tuning using OpenMed's GLiNER / GLiNER2 zero-shot su…. | ↔ adjacent |

### Data ingestion — feed clinical text into OpenMed

| Skill | What it does | Pairs |
| --- | --- | --- |
| [`extracting-dicom-metadata`](extracting-dicom-metadata/SKILL.md) | Reads DICOM file headers and DICOM-SR (Structured Report) content to pull study/series metadata and embedded report text, and flags PHI c…. | → before |
| [`fetching-fhir-resources`](fetching-fhir-resources/SKILL.md) | Fetches and pages FHIR R4 resources (Patient, DocumentReference, DiagnosticReport, Observation, Condition) from a FHIR REST server, decod…. | → before |
| [`generating-synthea-data`](generating-synthea-data/SKILL.md) | Generates synthetic but realistic patient records (FHIR R4 bundles, C-CDA documents, CSV) with MITRE Synthea for development, CI fixtures…. | ↔ adjacent |
| [`parsing-ccda-documents`](parsing-ccda-documents/SKILL.md) | Parses C-CDA / CCD XML clinical documents to extract human-readable section narrative plus coded entries, keyed by section LOINC codes an…. | → before |
| [`parsing-hl7v2-messages`](parsing-hl7v2-messages/SKILL.md) | Decodes pipe-delimited HL7 v2.x messages (ADT, ORU, MDM, ORM) into structured segments/fields/components and surfaces OBX-5 and NTE-3 fre…. | → before |

### De-identification & privacy

| Skill | What it does | Pairs |
| --- | --- | --- |
| [`auditing-deid-leakage`](auditing-deid-leakage/SKILL.md) | Adversarially scan already-de-identified clinical text for residual identifiers and emit a leakage report that blocks release on any hit. | after → |
| [`auditing-deidentification-runs`](auditing-deidentification-runs/SKILL.md) | Produce a signed, reproducible, no-PHI audit trail for an OpenMed de-identification run via deidentify(audit=True). | after → |
| [`auditing-safe-harbor-checklist`](auditing-safe-harbor-checklist/SKILL.md) | Verify OpenMed de-identified output against all 18 HIPAA Safe Harbor identifier categories and report residual re-identification risk. | after → |
| [`configuring-privacy-policies`](configuring-privacy-policies/SKILL.md) | Select and customize OpenMed's seven bundled privacy policy profiles for de-identification, and build custom surrogate generators. | ↔ adjacent |
| [`deidentifying-multilingual-text`](deidentifying-multilingual-text/SKILL.md) | De-identify non-English clinical text on-device with OpenMed by passing lang= and locale= to deidentify(). | ↔ adjacent |
| [`generating-synthetic-surrogates`](generating-synthetic-surrogates/SKILL.md) | Replace detected PHI with realistic, type-matched fake values in OpenMed so clinical notes stay readable and parseable instead of full of…. | after → |
| [`pseudonymizing-for-gdpr`](pseudonymizing-for-gdpr/SKILL.md) | Apply GDPR-grade pseudonymization to clinical or personal text with OpenMed, keeping a separately-held re-linkage key so the data can be …. | after → |
| [`reviewing-reidentification-risk`](reviewing-reidentification-risk/SKILL.md) | Run expert-determination-style quasi-identifier risk scoring (k-anonymity, l-diversity) plus OpenMed's empirical re-identification attack…. | after → |
| [`shifting-clinical-dates`](shifting-clinical-dates/SKILL.md) | Apply consistent per-patient date shifting in OpenMed that preserves intervals between events while satisfying HIPAA Safe Harbor's date rule. | after → |

### Clinical NLP — refine OpenMed output

| Skill | What it does | Pairs |
| --- | --- | --- |
| [`extracting-sdoh`](extracting-sdoh/SKILL.md) | Extracts social determinants of health (SDOH) — housing instability, food insecurity, unemployment, transportation barriers, social isola…. | after → |
| [`parsing-lab-values`](parsing-lab-values/SKILL.md) | Parse laboratory values and reference ranges from clinical text and flag results as low, normal, high, or critical with OpenMed. | after → |
| [`reconciling-problem-lists`](reconciling-problem-lists/SKILL.md) | Deduplicate and reconcile OpenMed-extracted conditions into one clean active problem list with clinical status (active / resolved / histo…. | after → |
| [`resolving-clinical-context`](resolving-clinical-context/SKILL.md) | Assign negation, temporality, and uncertainty (the ConText axes) to clinical entities extracted by OpenMed, so \"denies chest pain\" is n…. | after → |
| [`segmenting-clinical-sections`](segmenting-clinical-sections/SKILL.md) | Split a clinical note into canonical sections (Chief Complaint, HPI, PMH, Medications, Allergies, Assessment & Plan, etc.) before running…. | → before |
| [`summarizing-clinical-notes`](summarizing-clinical-notes/SKILL.md) | Produces structured, citation-anchored summaries of clinical notes — one-liner, hospital course, and problem-oriented views — where every…. | after → |

### Terminology & coding

| Skill | What it does | Pairs |
| --- | --- | --- |
| [`coding-hcc-risk-adjustment`](coding-hcc-risk-adjustment/SKILL.md) | Maps chronic conditions extracted by OpenMed to CMS-HCC V28 risk-adjustment categories and estimates a RAF (Risk Adjustment Factor) score…. | after → |
| [`coding-icd10`](coding-icd10/SKILL.md) | Suggests candidate ICD-10-CM diagnosis codes (and ICD-10-PCS procedure codes) for diagnoses and procedures extracted by OpenMed, with rat…. | after → |
| [`linking-umls-concepts`](linking-umls-concepts/SKILL.md) | Links entities extracted by OpenMed to UMLS Metathesaurus CUIs using the USER'S OWN UTS API key, with nothing from the Metathesaurus bund…. | after → |
| [`mapping-loinc`](mapping-loinc/SKILL.md) | Maps laboratory and clinical observation names extracted by OpenMed to LOINC codes using the public Regenstrief LOINC and FHIR terminolog…. | after → |
| [`mapping-to-snomed`](mapping-to-snomed/SKILL.md) | Maps clinical concept spans extracted by OpenMed to SNOMED CT concepts through a USER-SUPPLIED terminology server (the user's own Ontoser…. | after → |
| [`normalizing-rxnorm`](normalizing-rxnorm/SKILL.md) | Normalizes drug mentions extracted by OpenMed to RxNorm RxCUIs using the free public RxNav/RxNorm REST API. | after → |

### FHIR & interoperability

| Skill | What it does | Pairs |
| --- | --- | --- |
| [`assembling-fhir-bundles`](assembling-fhir-bundles/SKILL.md) | Package multiple FHIR R4 resources produced from OpenMed output into a single valid transaction Bundle ready to POST to an EHR, using Ope…. | after → |
| [`bridging-presidio-and-spacy`](bridging-presidio-and-spacy/SKILL.md) | Combine OpenMed clinical NLP with Microsoft Presidio, spaCy, or LangChain through OpenMed's built-in interop adapter registry (openmed.in…. | ↔ adjacent |
| [`exporting-bulk-fhir`](exporting-bulk-fhir/SKILL.md) | Kick off and harvest a FHIR Bulk Data $export (system-, group-, or patient-level) and stream the resulting NDJSON into a batch OpenMed de…. | → before |
| [`exporting-to-fhir`](exporting-to-fhir/SKILL.md) | Convert OpenMed NER output (entities from openmed.analyze_text) into FHIR R4 resources — Condition, MedicationStatement, Observation — us…. | after → |
| [`querying-terminology-service`](querying-terminology-service/SKILL.md) | Call a user-supplied FHIR terminology server ($validate-code, $expand, $lookup, $translate) to validate and expand clinical codes without…. | ↔ adjacent |
| [`scaffolding-smart-on-fhir`](scaffolding-smart-on-fhir/SKILL.md) | Scaffold a SMART-on-FHIR app (SMART App Launch v2 — EHR launch and standalone launch, OAuth2 PKCE, scopes, token handling, fhirContext) s…. | ↔ adjacent |
| [`validating-us-core`](validating-us-core/SKILL.md) | Validate FHIR R4 resources and Bundles against US Core / USCDI profiles with the official HL7 FHIR validator before submitting to an EHR. | after → |

### Evaluation & quality

| Skill | What it does | Pairs |
| --- | --- | --- |
| [`auditing-subgroup-fairness`](auditing-subgroup-fairness/SKILL.md) | Audit an OpenMed NER or de-identification model for performance disparities across demographic subgroups (sex, age band, race/ethnicity w…. | ↔ adjacent |
| [`authoring-model-cards`](authoring-model-cards/SKILL.md) | Generate a model card for an OpenMed clinical NER or de-identification model documenting intended use, quantitative metrics, subgroup per…. | after → |
| [`benchmarking-clinical-ner`](benchmarking-clinical-ner/SKILL.md) | Score an OpenMed clinical or biomedical NER model against a user-supplied gold corpus with entity-level precision, recall, and F1, then b…. | ↔ adjacent |
| [`building-gold-corpus`](building-gold-corpus/SKILL.md) | Scaffold a synthetic gold-standard annotation project for evaluating OpenMed NER and de-identification models — label schema, annotation …. | ↔ adjacent |
| [`evaluating-with-leakage-gates`](evaluating-with-leakage-gates/SKILL.md) | Evaluate an OpenMed de-identification or clinical NER model against the leakage-first release gates G1a through G8, which gate releases o…. | ↔ adjacent |
| [`gating-deid-leakage`](gating-deid-leakage/SKILL.md) | Add a CI gate that fails the build when an OpenMed de-identification model's recall on a held-out PHI set drops below threshold or any cr…. | ↔ adjacent |

### Research & genomics

| Skill | What it does | Pairs |
| --- | --- | --- |
| [`annotating-variants`](annotating-variants/SKILL.md) | Annotates VCF variants and normalizes HGVS nomenclature with public, license-free annotators (Ensembl VEP REST, VEP/SnpEff/ANNOVAR offlin…. | ↔ adjacent |
| [`defining-cohort-phenotypes`](defining-cohort-phenotypes/SKILL.md) | Authors computable phenotype and cohort definitions in the OHDSI ATLAS / CIRCE style over the OMOP CDM, combining standard concept sets w…. | ↔ adjacent |
| [`mining-pubmed-literature`](mining-pubmed-literature/SKILL.md) | Searches and fetches PubMed and PMC via NCBI E-utilities (ESearch then EFetch/ESummary) to gather biomedical evidence and build text corpora. | ↔ adjacent |
| [`parsing-trial-eligibility`](parsing-trial-eligibility/SKILL.md) | Parses free-text clinical-trial eligibility criteria into structured inclusion and exclusion logic, then matches them against patient fac…. | after → |
| [`searching-clinicaltrials`](searching-clinicaltrials/SKILL.md) | Searches ClinicalTrials.gov for studies by condition, intervention, and recruitment status using the modern v2 REST API with cursor (page…. | ↔ adjacent |

### Imaging & OCR intake

| Skill | What it does | Pairs |
| --- | --- | --- |
| [`extracting-lab-tables`](extracting-lab-tables/SKILL.md) | Detects and extracts tabular laboratory panels from PDFs, scans, and images into structured rows ready for OpenMed and FHIR. | → before |
| [`ingesting-clinical-documents`](ingesting-clinical-documents/SKILL.md) | Turn scanned faxes, images, and CSV/CDA exports into clean text ready for OpenMed de-identification and NER, fully on-device. | → before |
| [`structuring-radiology-reports`](structuring-radiology-reports/SKILL.md) | Converts free-text radiology narratives into structured findings and impression — with measurements, laterality, anatomy, and follow-up r…. | after → |

### Compliance & regulatory

| Skill | What it does | Pairs |
| --- | --- | --- |
| [`auditing-part11-trails`](auditing-part11-trails/SKILL.md) | Generates and verifies 21 CFR Part 11-style audit trails — who/what/when, electronic signatures, and tamper-evidence — for OpenMed pipeli…. | ↔ adjacent |
| [`checking-hipaa-compliance`](checking-hipaa-compliance/SKILL.md) | Runs a HIPAA Privacy and Security Rule checklist over a data pipeline and produces a gap report before deploying OpenMed on PHI. | ↔ adjacent |

### Safety & pharmacovigilance

| Skill | What it does | Pairs |
| --- | --- | --- |
| [`detecting-pv-signals`](detecting-pv-signals/SKILL.md) | Computes disproportionality signals — PRR, ROR, EBGM, and IC (BCPNN) — over FAERS / OpenFDA drug-event data to flag potential safety signals. | ↔ adjacent |
| [`querying-openfda-labels`](querying-openfda-labels/SKILL.md) | Looks up FDA drug labels, NDC directory entries, indications, boxed warnings, and recalls/enforcement actions via the free public OpenFDA…. | ↔ adjacent |
| [`reporting-adverse-events`](reporting-adverse-events/SKILL.md) | Structures adverse-event mentions that OpenMed extracts into FAERS / ICH E2B(R3) reportable fields — suspect drug, reaction (MedDRA PT), …. | after → |

### Analytics & reporting

| Skill | What it does | Pairs |
| --- | --- | --- |
| [`building-patient-timelines`](building-patient-timelines/SKILL.md) | Assemble a chronological patient timeline from OpenMed-extracted clinical events, normalizing dates and resolving relative time expressio…. | after → |
| [`computing-ecqms`](computing-ecqms/SKILL.md) | Compute electronic clinical quality measures (eCQMs) over structured data using CQL/QDM logic, lifting note-derived numerator and exclusi…. | after → |
| [`etl-to-omop-cdm`](etl-to-omop-cdm/SKILL.md) | Map OpenMed-extracted, terminology-coded conditions, drugs, and measurements into OMOP CDM v5.4 clinical tables (condition_occurrence, dr…. | after → |

### Deployment & ops

| Skill | What it does | Pairs |
| --- | --- | --- |
| [`batch-processing-clinical-text`](batch-processing-clinical-text/SKILL.md) | Run large-scale batch NER, PII extraction, or de-identification over many clinical notes on-device with OpenMed, with sharding, checkpoin…. | ↔ adjacent |
| [`deploying-openmed-mcp`](deploying-openmed-mcp/SKILL.md) | Run OpenMed's Model Context Protocol (MCP) server so coding agents (Claude Code, Codex) and chat clients can call clinical NER, PII extra…. | ↔ adjacent |
| [`enforcing-nophi-logging`](enforcing-nophi-logging/SKILL.md) | Add a logging and telemetry guard that scrubs or blocks PHI from logs, traces, and error reports around an OpenMed deployment. | ↔ adjacent |
| [`running-openmed-ondevice`](running-openmed-ondevice/SKILL.md) | Run OpenMed models fully on-device with the MLX (Apple Silicon), CoreML (iOS/macOS), or ONNX/WebGPU (cross-platform/browser) backends, in…. | ↔ adjacent |
| [`serving-openmed-rest-api`](serving-openmed-rest-api/SKILL.md) | Stand up OpenMed's FastAPI REST service for clinical NER, PII extraction, and de-identification, with health checks, model keep-alive/unl…. | ↔ adjacent |

## Authoring & validation

Every skill follows the open spec: kebab-case folder name matching `name`, a `description` that states *what it does and when to use it*, a body under 500 lines, and detail pushed into `references/`. Validate and regenerate this catalog with:

```bash
python skills/build_catalog.py --check   # CI gate
python skills/build_catalog.py           # rewrite README + marketplace
```
