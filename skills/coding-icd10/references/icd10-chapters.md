# ICD-10-CM chapter ranges (FY2025)

Use the first-letter / code-range to route an OpenMed diagnosis span to the right
chapter before searching for a specific code. Ranges are the official ICD-10-CM
chapter blocks; a human coder validates the final code.

| Ch | Code range | Title |
| --- | --- | --- |
| 1  | A00–B99 | Certain infectious and parasitic diseases |
| 2  | C00–D49 | Neoplasms |
| 3  | D50–D89 | Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism |
| 4  | E00–E89 | Endocrine, nutritional and metabolic diseases |
| 5  | F01–F99 | Mental, behavioral and neurodevelopmental disorders |
| 6  | G00–G99 | Diseases of the nervous system |
| 7  | H00–H59 | Diseases of the eye and adnexa |
| 8  | H60–H95 | Diseases of the ear and mastoid process |
| 9  | I00–I99 | Diseases of the circulatory system |
| 10 | J00–J99 | Diseases of the respiratory system |
| 11 | K00–K95 | Diseases of the digestive system |
| 12 | L00–L99 | Diseases of the skin and subcutaneous tissue |
| 13 | M00–M99 | Diseases of the musculoskeletal system and connective tissue |
| 14 | N00–N99 | Diseases of the genitourinary system |
| 15 | O00–O9A | Pregnancy, childbirth and the puerperium |
| 16 | P00–P96 | Certain conditions originating in the perinatal period |
| 17 | Q00–Q99 | Congenital malformations, deformations and chromosomal abnormalities |
| 18 | R00–R99 | Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified |
| 19 | S00–T88 | Injury, poisoning and certain other consequences of external causes |
| 20 | V00–Y99 | External causes of morbidity |
| 21 | Z00–Z99 | Factors influencing health status and contact with health services |
| 22 | U00–U85 | Codes for special purposes (e.g. U07.1 COVID-19, emergency use) |

## ICD-10-PCS sections (inpatient procedures, first character)

PCS is a separate, 7-character procedure system (Medical and Surgical sections,
etc.). Sections by first character:

| Char | Section |
| --- | --- |
| 0 | Medical and Surgical |
| 1 | Obstetrics |
| 2 | Placement |
| 3 | Administration |
| 4 | Measurement and Monitoring |
| 5 | Extracorporeal or Systemic Assistance and Performance |
| 6 | Extracorporeal or Systemic Therapies |
| 7 | Osteopathic |
| 8 | Other Procedures |
| 9 | Chiropractic |
| B | Imaging |
| C | Nuclear Medicine |
| D | Radiation Therapy |
| F | Physical Rehabilitation and Diagnostic Audiology |
| G | Mental Health |
| H | Substance Abuse Treatment |
| X | New Technology |

## Notes
- ICD-10-CM = diagnoses (outpatient & inpatient); ICD-10-PCS = inpatient
  procedures only. CPT/HCPCS (restricted, AMA-licensed) cover outpatient
  procedures — not in scope here.
- Official files are public domain from CMS (see SKILL.md references).
- GEMs (General Equivalence Mappings) cross-walk ICD-9 ↔ ICD-10 and are
  approximate, many-to-many; never treat a GEM hit as an exact, billable mapping.
