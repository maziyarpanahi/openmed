# Clinical Domain Label Catalog

**Disclaimer:** This catalog is generated from the packaged zero-shot domain label map and canonical label metadata. Labels, policy classes, risk levels, coding system hints, and fixture pointers are for offline evaluation and review workflow planning only; they are not clinical guidance and must not be used to infer diagnosis, treatment, coding, or coverage decisions.

## Biomedical

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Disease | CONDITION | CLINICAL_CONCEPT | low | ICD-10-CM, ICD-10-CN, SNOMED | Not shipped |
| Drug | MEDICATION | CLINICAL_CONCEPT | low | RxNorm, CN-DRUG, SNOMED | Not shipped |
| Gene | GENE_SYMBOL | CLINICAL_CONCEPT | low | SNOMED | Not shipped |
| Organism | MICROORGANISM | CLINICAL_CONCEPT | low | SNOMED, LOINC | Not shipped |

## Clinical

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Problem | CONDITION | CLINICAL_CONCEPT | low | ICD-10-CM, ICD-10-CN, SNOMED | Not shipped |
| Treatment | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Test | LAB_TEST | CLINICAL_CONCEPT | low | LOINC, SNOMED | Not shipped |
| BodyPart | BODY_SITE | CLINICAL_CONCEPT | low | SNOMED | Not shipped |

## Genomic

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Variant | VARIANT_DESCRIPTOR | CLINICAL_CONCEPT | low | SNOMED | Not shipped |
| Gene | GENE_SYMBOL | CLINICAL_CONCEPT | low | SNOMED | Not shipped |
| Transcript | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Phenotype | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |

## Genomic Variant

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| GeneSymbol | GENE_SYMBOL | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/genomic_variant.jsonl |
| VariantDescriptor | VARIANT_DESCRIPTOR | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/genomic_variant.jsonl |
| ProteinChange | PROTEIN_CHANGE | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/genomic_variant.jsonl |
| Zygosity | ZYGOSITY | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/genomic_variant.jsonl |
| AlleleFrequency | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/genomic_variant.jsonl |
| ClinicalSignificance | CLINICAL_SIGNIFICANCE | CLINICAL_CONCEPT | low | SNOMED, HPO | tests/fixtures/clinical/genomic_variant.jsonl |
| ReferenceTranscript | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/genomic_variant.jsonl |

## Finance

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Company | ORGANIZATION | QUASI_IDENTIFIER | medium | None | Not shipped |
| Ticker | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Instrument | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Event | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |

## Legal

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Case | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Court | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Judge | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Statute | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |

## News

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Person | PERSON | DIRECT_IDENTIFIER | high | None | Not shipped |
| Organization | ORGANIZATION | QUASI_IDENTIFIER | medium | None | Not shipped |
| Location | LOCATION | QUASI_IDENTIFIER | medium | None | Not shipped |
| Date | DATE | QUASI_IDENTIFIER | medium | None | Not shipped |

## Ecommerce

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Product | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Brand | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Price | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Attribute | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |

## Cybersecurity

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Vulnerability | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| CVE | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Software | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Vendor | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |

## Chemistry

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Compound | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Reaction | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Property | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Unit | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |

## Organism

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Species | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Strain | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Taxon | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Habitat | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |

## Education

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Course | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Topic | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Institution | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Degree | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |

## Social

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Person | PERSON | DIRECT_IDENTIFIER | high | None | Not shipped |
| Handle | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Hashtag | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| URL | URL | DIRECT_IDENTIFIER | high | None | Not shipped |

## Public Health

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Condition | CONDITION | CLINICAL_CONCEPT | low | ICD-10-CM, ICD-10-CN, SNOMED | Not shipped |
| Intervention | PROCEDURE | CLINICAL_CONCEPT | low | SNOMED | Not shipped |
| Outcome | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Population | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |

## Cardiology

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| CardiacFinding | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| ECGFinding | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| EjectionFraction | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| CardiacProcedure | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| CardiacDevice | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Anatomy | BODY_SITE | CLINICAL_CONCEPT | low | SNOMED | Not shipped |

## Microbiology

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Microorganism | MICROORGANISM | CLINICAL_CONCEPT | low | SNOMED, LOINC | Not shipped |
| Antibiotic | ANTIBIOTIC | CLINICAL_CONCEPT | low | RxNorm, SNOMED | Not shipped |
| Susceptibility | SUSCEPTIBILITY | CLINICAL_CONCEPT | low | LOINC, SNOMED | Not shipped |
| SpecimenSource | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| CultureResult | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |

## Dermatology

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| SkinLesion | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Morphology | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Distribution | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Anatomy | BODY_SITE | CLINICAL_CONCEPT | low | SNOMED | Not shipped |

## Ophthalmology

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| EyeFinding | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| VisualAcuity | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| IntraocularPressure | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | Not shipped |
| Anatomy | BODY_SITE | CLINICAL_CONCEPT | low | SNOMED | Not shipped |

## Generic

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| Person | PERSON | DIRECT_IDENTIFIER | high | None | Not shipped |
| Organization | ORGANIZATION | QUASI_IDENTIFIER | medium | None | Not shipped |
| Location | LOCATION | QUASI_IDENTIFIER | medium | None | Not shipped |
| Date | DATE | QUASI_IDENTIFIER | medium | None | Not shipped |

## Anesthesia

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| AnesthesiaType | ANESTHESIA_TYPE | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/anesthesia.jsonl |
| AnestheticAgent | ANESTHETIC_AGENT | CLINICAL_CONCEPT | low | RxNorm, SNOMED | tests/fixtures/clinical/anesthesia.jsonl |
| AirwayManagement | AIRWAY_MANAGEMENT | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/anesthesia.jsonl |
| ASAClass | ASA_CLASS | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/anesthesia.jsonl |
| MonitoringModality | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/anesthesia.jsonl |
| IntraoperativeEvent | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/anesthesia.jsonl |

## Nutrition Diet

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| DietType | DIET_TYPE | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/nutrition_diet.jsonl |
| NutritionTarget | NUTRITION_TARGET | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/nutrition_diet.jsonl |
| Supplement | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/nutrition_diet.jsonl |
| FeedingRoute | FEEDING_ROUTE | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/nutrition_diet.jsonl |
| IntakeFinding | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/nutrition_diet.jsonl |
| NutritionalStatus | NUTRITIONAL_STATUS | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/nutrition_diet.jsonl |
| FluidRestriction | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/nutrition_diet.jsonl |

## Endocrinology

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| GlycemicMeasure | GLYCEMIC_MEASURE | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/endocrinology.jsonl |
| ThyroidFunctionMeasure | THYROID_MEASURE | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/endocrinology.jsonl |
| HormoneLevel | HORMONE_LEVEL | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/endocrinology.jsonl |
| InsulinRegimen | INSULIN_REGIMEN | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/endocrinology.jsonl |
| MetabolicFinding | CONDITION | CLINICAL_CONCEPT | low | ICD-10-CM, ICD-10-CN, SNOMED | tests/fixtures/clinical/endocrinology.jsonl |
| EndocrineGland | BODY_SITE | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/endocrinology.jsonl |

## Gastroenterology

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| EndoscopicFinding | ENDOSCOPIC_FINDING | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/gastroenterology.jsonl |
| GISymptom | GI_SYMPTOM | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/gastroenterology.jsonl |
| BowelPrepQuality | GI_SCORE | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/gastroenterology.jsonl |
| BiopsySite | BODY_SITE | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/gastroenterology.jsonl |
| GIScore | GI_SCORE | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/gastroenterology.jsonl |
| LesionMorphology | POLYP_DESCRIPTOR | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/gastroenterology.jsonl |
| PolypDescriptor | POLYP_DESCRIPTOR | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/gastroenterology.jsonl |

## Nephrology Renal

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| RenalFunctionMeasure | RENAL_FUNCTION_MEASURE | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/nephrology_renal.jsonl |
| CKDStage | CKD_STAGE | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/nephrology_renal.jsonl |
| DialysisModality | DIALYSIS_MODALITY | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/nephrology_renal.jsonl |
| UrineFinding | URINE_FINDING | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/nephrology_renal.jsonl |
| FluidStatus | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/nephrology_renal.jsonl |
| RenalReplacementAccess | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/nephrology_renal.jsonl |
| ElectrolyteDisturbance | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/nephrology_renal.jsonl |

## Immunization

**Alignment:** The display labels are shaped for the planned OM-138 FHIR Immunization exporter: VaccineName maps to vaccineCode, DoseNumber to protocolApplied.doseNumber[x], AdministrationRoute to route, AdministrationSite to site, VaccineLot to lotNumber, AdministrationDate to occurrence[x], and VaccineSeries to protocolApplied.series. This is extraction metadata only; it does not create exporter, recommendation, dosing, or scheduling logic.

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| VaccineName | VACCINE_NAME | CLINICAL_CONCEPT | low | RxNorm, SNOMED | tests/fixtures/clinical/immunization.jsonl |
| DoseNumber | DOSE_NUMBER | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/immunization.jsonl |
| AdministrationRoute | ADMINISTRATION_ROUTE | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/immunization.jsonl |
| AdministrationSite | BODY_SITE | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/immunization.jsonl |
| VaccineLot | VACCINE_LOT | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/immunization.jsonl |
| AdministrationDate | DATE | QUASI_IDENTIFIER | medium | None | tests/fixtures/clinical/immunization.jsonl |
| VaccineSeries | VACCINE_SERIES | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/immunization.jsonl |

## Pulmonology

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| SpirometryMeasure | SPIROMETRY_MEASURE | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/pulmonology.jsonl |
| OxygenSupport | OXYGEN_SUPPORT | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/pulmonology.jsonl |
| RespiratoryFinding | RESPIRATORY_FINDING | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/pulmonology.jsonl |
| DyspneaGrade | DYSPNEA_GRADE | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/pulmonology.jsonl |
| LungAuscultation | RESPIRATORY_FINDING | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/pulmonology.jsonl |
| PFTInterpretation | OTHER | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, ICD-10-CN, HPO, RxNorm, CN-DRUG, LOINC | tests/fixtures/clinical/pulmonology.jsonl |
| AirwayDevice | AIRWAY_MANAGEMENT | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/pulmonology.jsonl |

## Pediatrics Growth

| Label | Canonical Label | Category | Risk Level | System Hints | Fixture Path |
| --- | --- | --- | --- | --- | --- |
| GrowthParameter | GROWTH_PARAMETER | CLINICAL_CONCEPT | low | SNOMED, LOINC | tests/fixtures/clinical/pediatrics_growth.jsonl |
| GrowthPercentile | GROWTH_PERCENTILE | CLINICAL_CONCEPT | low | SNOMED, LOINC | tests/fixtures/clinical/pediatrics_growth.jsonl |
| GrowthZScore | GROWTH_PERCENTILE | CLINICAL_CONCEPT | low | SNOMED, LOINC | tests/fixtures/clinical/pediatrics_growth.jsonl |
| DevelopmentalMilestone | DEVELOPMENTAL_MILESTONE | CLINICAL_CONCEPT | low | SNOMED, LOINC | tests/fixtures/clinical/pediatrics_growth.jsonl |
| FeedingHistory | NUTRITIONAL_STATUS | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/pediatrics_growth.jsonl |
| PediatricFinding | CONDITION | CLINICAL_CONCEPT | low | ICD-10-CM, SNOMED | tests/fixtures/clinical/pediatrics_growth.jsonl |
