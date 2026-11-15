# Synthetic OMOP cohorts

`openmed.training.synthetic.omop_cohorts` maps explicitly synthetic-marked existing
`ClinicalFact` graphs into coherent OMOP-shaped person, visit, condition, drug,
measurement, procedure and observation tables. It also renders synthetic source
notes with actual content hashes and stable note/event links, using the existing
OMOP cohort checker for relational/concept/provenance validation.

`generate_omop_cohort(facts, seed=..., cohort_condition_codes=...)` derives positive
BIGINT identities from graph identity and seed. Missing visits, cross-person
encounters, duplicate facts, invalid dates and future causal parents reject.
Visits span the declared synthetic effective event dates. Concept records use
explicit `OPENMED_SYNTHETIC` placeholders with no standard-concept claim; no
Athena, UMLS, SNOMED or CPT content is distributed. Person demographic values are
fictional fixture defaults, with birth year 30 years before the earliest visit
(clipped at year 1) so date-shifted early visits cannot precede birth. Numeric measurement truth remains `None` when the
graph contains only a code, rather than being invented.

The artifact contains `tables` and gold `cohort` membership. Membership uses the
intersection of all requested controlled condition codes, calculated from declared
synthetic condition facts. `validate_synthetic_omop` independently checks existing
runtime relational rules, visit/event chronology, synthetic markers, actual note
hashes and cohort truth. Orphan people, mismatched visits, reversed dates and
altered membership are separate negative controls.

Only 1,000 graph facts and 128 criterion codes are accepted; validation is capped
at 10,000 rows. No database write, network, real-record import or restricted
terminology lookup occurs. These are structural/utility test fixtures, not
clinical plausibility, full CDM conformance, disease progression or treatment
benefit assurances. Cohort flags are synthetic gold labels, never clinical
eligibility recommendations.
