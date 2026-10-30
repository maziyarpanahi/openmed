# Structured code-switched synthetic dialogues

`generate_code_switch_dialogue` generates authored semantic turn sequences with
patient, caregiver, interpreter and clinician roles. Each `DialogueTurnSpec`
selects explicit English/Spanish/Hindi segments, an authored fictional clinical
concept, assertion status, experiencer role and optional correction reference.
The generator does not concatenate unrelated random strings: every switched
segment preserves the same declared clinical concept/status and role context.

Existing locale-PHI generators provide seeded synthetic PERSON spans, consistent
for the same experiencer role and locale. Every span is checked against shared
extended-grapheme boundaries and exact generated text. Language-tagged ranges
retain within-turn switch points. Existing Journey `ClinicalFact`,
`ClinicalArtifact` and `EvidenceLocator` records bind clinical gold annotations to
actual statement offsets; no alternate clinical truth graph is introduced.

Contradictory status declarations for the same concept/experiencer flag all
involved turns for adjudication. Uncertainty also requires review. Corrections
must point backward to the same clinical identity and retain parent fact/evidence
links; they never erase the earlier assertion or auto-resolve a conflict.
Reports expose counts, controlled roles/languages and artifact hashes without
names or transcript content. Training exports contain only generated synthetic
turns, spans and declared facts.

The three language templates are limited authored demonstrations, not native
review or clinical linguistic qualification. Concepts are fictional software
stimuli, not medication advice or patient records. No translation model, remote
service, real dataset or autonomous clinical decision is involved. PHI generation
is reused rather than replacing deterministic privacy checks. Synthetic origin
and permissive template rights do not establish model quality or privacy.
