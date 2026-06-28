**Disclaimer:** Deterministic status vocabulary helpers for SDOH cue normalization. The helpers in this module normalize explicit status cues into compact canonical values for downstream SDOH extractors. Outputs are advisory labels for review and downstream processing, not clinical decisions. They deliberately do not infer a status from absence of mention; text with no configured cue returns 'unknown' unless an explicit context axis is supplied by the caller.

### Biomedical

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Disease | CLINICAL_CONCEPT | low | ICD-10-CM, SNOMED | tests/fixtures/clinical/context_traps.jsonl |
| Drug | CLINICAL_CONCEPT | low | RxNorm, SNOMED | tests/fixtures/clinical/context_traps.jsonl |
| Gene | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Organism | CLINICAL_CONCEPT | low | SNOMED, LOINC | tests/fixtures/clinical/context_traps.jsonl |

### Clinical

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Problem | CLINICAL_CONCEPT | low | ICD-10-CM, SNOMED | tests/fixtures/clinical/context_traps.jsonl |
| Treatment | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Test | CLINICAL_CONCEPT | low | LOINC, SNOMED | tests/fixtures/clinical/context_traps.jsonl |
| BodyPart | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/context_traps.jsonl |

### Genomic

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Variant | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Gene | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Transcript | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Phenotype | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |

### Finance

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Company | QUASI_IDENTIFIER | medium | None | tests/fixtures/clinical/context_traps.jsonl |
| Ticker | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Instrument | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Event | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |

### Legal

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Case | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Court | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Judge | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Statute | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |

### News

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Person | DIRECT_IDENTIFIER | high | None | tests/fixtures/clinical/context_traps.jsonl |
| Organization | QUASI_IDENTIFIER | medium | None | tests/fixtures/clinical/context_traps.jsonl |
| Location | QUASI_IDENTIFIER | medium | None | tests/fixtures/clinical/context_traps.jsonl |
| Date | QUASI_IDENTIFIER | medium | None | tests/fixtures/clinical/context_traps.jsonl |

### Ecommerce

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Product | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Brand | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Price | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Attribute | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |

### Cybersecurity

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Vulnerability | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| CVE | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Software | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Vendor | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |

### Chemistry

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Compound | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Reaction | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Property | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Unit | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |

### Organism

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Species | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Strain | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Taxon | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Habitat | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |

### Education

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Course | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Topic | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Institution | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Degree | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |

### Social

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Person | DIRECT_IDENTIFIER | high | None | tests/fixtures/clinical/context_traps.jsonl |
| Handle | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Hashtag | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| URL | DIRECT_IDENTIFIER | high | None | tests/fixtures/clinical/context_traps.jsonl |

### Public Health

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Condition | CLINICAL_CONCEPT | low | ICD-10-CM, SNOMED | tests/fixtures/clinical/context_traps.jsonl |
| Intervention | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/context_traps.jsonl |
| Outcome | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Population | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |

### Cardiology

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| CardiacFinding | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| ECGFinding | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| EjectionFraction | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| CardiacProcedure | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| CardiacDevice | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Anatomy | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/context_traps.jsonl |

### Microbiology

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Microorganism | CLINICAL_CONCEPT | low | SNOMED, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Antibiotic | CLINICAL_CONCEPT | low | RxNorm, SNOMED | tests/fixtures/clinical/context_traps.jsonl |
| Susceptibility | CLINICAL_CONCEPT | low | LOINC, SNOMED | tests/fixtures/clinical/context_traps.jsonl |
| SpecimenSource | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| CultureResult | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |

### Dermatology

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| SkinLesion | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Morphology | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Distribution | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Anatomy | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/context_traps.jsonl |

### Ophthalmology

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| EyeFinding | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| VisualAcuity | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| IntraocularPressure | CLINICAL_CONCEPT | low | SNOMED, ICD-10-CM, HPO, RxNorm, LOINC | tests/fixtures/clinical/context_traps.jsonl |
| Anatomy | CLINICAL_CONCEPT | low | SNOMED | tests/fixtures/clinical/context_traps.jsonl |

### Generic

| Label | Category | Risk Level | System Hints | Fixture Path |
|-------|----------|------------|--------------|--------------|
| Person | DIRECT_IDENTIFIER | high | None | tests/fixtures/clinical/context_traps.jsonl |
| Organization | QUASI_IDENTIFIER | medium | None | tests/fixtures/clinical/context_traps.jsonl |
| Location | QUASI_IDENTIFIER | medium | None | tests/fixtures/clinical/context_traps.jsonl |
| Date | QUASI_IDENTIFIER | medium | None | tests/fixtures/clinical/context_traps.jsonl |
