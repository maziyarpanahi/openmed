# Synthetic medication courses

`openmed.training.synthetic.medication_courses` materializes explicitly declared
synthetic prescription, titration, adherence, discontinuation and outcome facts.
It bundles no drug vocabulary, dose recommendations or real patient data.

`MedicationCourseSpec` uses controlled medication/indication/unit/route codes,
relative start/stop days, a positive initial dose, and optional `DoseChange`,
`AdherenceEvent` and `MedicationOutcome` collections. Dose changes lie strictly
inside the active course; taken/missed events lie in its half-open active interval.
Duplicate/conflicting same-day dose or adherence declarations reject. Outcomes
may follow discontinuation only inside an explicitly declared followup window.
Their gold relation is `observed_after_start`, never inferred causation.

`generate_medication_course(spec, seed=...)` returns stable opaque IDs, gold
half-open regimen intervals, ordered events and relation labels. Seeds vary
identities; doses and outcomes come from the caller's declared synthetic recipe,
never randomly invented clinical facts. No model, network or restricted
terminology download occurs. Each event family is capped at 1,000 entries.

Use `validate_medication_course` independently to reject regimen gaps, invalid
doses, dangling relations, mismatched stop dates or adherence outside the linked
regimen. Controlled error categories contain no private source values. These
fixtures support extraction/relationship evaluation; they do not establish drug
safety, clinical plausibility, treatment utility or medical-device readiness.
