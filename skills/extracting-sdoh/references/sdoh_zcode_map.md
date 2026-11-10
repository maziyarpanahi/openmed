# SDOH label → ICD-10-CM Z-code lookup

A starter mapping from common SDOH entity labels to ICD-10-CM Z-codes
(Z55–Z65). ICD-10-CM is public domain in the US release; align labels to the
**Gravity Project** domains for interoperability. **Validate against the current
fiscal-year code set** — CMS adds/retires SDOH codes most years. These are
*suggestions for human confirmation*, not billing assignments.

Use as a Python dict (label is lowercased model output):

```python
SDOH_ZCODES = {
    # --- Education & literacy (Z55) ---
    "illiteracy": "Z55.0",
    "low_literacy": "Z55.0",
    "schooling_unavailable": "Z55.1",
    "educational_maladjustment": "Z55.4",
    "low_education": "Z55.9",

    # --- Employment (Z56) ---
    "unemployment": "Z56.0",
    "change_of_job": "Z56.1",
    "threat_of_job_loss": "Z56.2",
    "stressful_work_schedule": "Z56.3",
    "workplace_discord": "Z56.4",
    "uncongenial_work": "Z56.5",
    "work_stress": "Z56.6",
    "employment_problem": "Z56.9",

    # --- Housing & economic circumstances (Z59) ---
    "homelessness": "Z59.0",
    "unsheltered_homelessness": "Z59.01",
    "sheltered_homelessness": "Z59.02",
    "housing_instability": "Z59.1",          # inadequate housing
    "inadequate_housing": "Z59.1",
    "discord_with_neighbors": "Z59.2",
    "institutional_residence": "Z59.3",
    "food_insecurity": "Z59.41",
    "lack_of_food": "Z59.41",
    "lack_of_safe_drinking_water": "Z59.42",
    "material_hardship": "Z59.5",            # extreme poverty
    "low_income": "Z59.6",
    "insufficient_insurance": "Z59.7",
    "foreclosure": "Z59.811",
    "eviction": "Z59.812",
    "housing_other_instability": "Z59.819",
    "transportation_barrier": "Z59.82",
    "transportation_insecurity": "Z59.82",
    "financial_strain": "Z59.86",
    "material_hardship_other": "Z59.87",
    "housing_economic_problem": "Z59.9",

    # --- Social environment (Z60) ---
    "social_isolation": "Z60.2",             # problems living alone
    "lives_alone": "Z60.2",
    "acculturation_difficulty": "Z60.3",
    "social_exclusion": "Z60.4",
    "rejection_discrimination": "Z60.5",
    "social_environment_problem": "Z60.9",

    # --- Upbringing (Z62) ---
    "inadequate_parental_supervision": "Z62.0",
    "parental_overprotection": "Z62.1",
    "upbringing_away_from_parents": "Z62.21",
    "institutional_upbringing": "Z62.22",
    "hostility_toward_child": "Z62.3",
    "emotional_neglect_of_child": "Z62.4",

    # --- Primary support / family circumstances (Z63) ---
    "relationship_problem_with_partner": "Z63.0",
    "family_discord": "Z63.8",
    "absence_of_family_member": "Z63.31",
    "death_of_family_member": "Z63.4",
    "family_disruption_separation": "Z63.5",
    "dependent_relative_needing_care": "Z63.6",
    "caregiver_stress": "Z63.6",
    "support_problem": "Z63.9",

    # --- Psychosocial circumstances (Z64–Z65) ---
    "unwanted_pregnancy_problem": "Z64.0",
    "multiparity_problem": "Z64.1",
    "conviction_legal_problem": "Z65.0",
    "imprisonment": "Z65.1",
    "release_from_prison": "Z65.2",
    "legal_circumstances_other": "Z65.3",
    "victim_of_crime_terrorism": "Z65.4",
    "exposure_to_disaster_war": "Z65.5",
    "psychosocial_circumstance_other": "Z65.8",
    "psychosocial_circumstance_unspecified": "Z65.9",
}
```

## Gravity Project domain alignment

Map your labels onto these Gravity SDOH Clinical Care domains so the codes feed
USCDI/FHIR cleanly: Food Insecurity, Housing Instability, Homelessness,
Inadequate Housing, Transportation Insecurity, Financial Strain, Material
Hardship, Employment Status, Educational Attainment, Social Connection /
Isolation, Stress, Intimate Partner Violence, Elder Abuse, Veteran Status,
Health Insurance Coverage Status, Digital Access. Each Gravity domain ships
LOINC question panels (screening), SNOMED CT problem codes, and ICD-10-CM
mappings — load SNOMED/LOINC from the user's own licensed copy; only ICD-10-CM
is bundled here.

Sources: CMS ICD-10-CM SDOH Z-code set; HL7 Gravity Project value sets
(https://www.hl7.org/gravity/).
